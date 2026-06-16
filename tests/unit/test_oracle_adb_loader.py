# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for :class:`locus.rag.loaders.oracle.OracleADBLoader`.

The loader's contract is column-to-field mapping plus CLOB
materialisation — neither needs a live database. We stub
``oracledb.create_pool_async`` to return an async context-managed
connection that hands back a scripted cursor, then introspect the
yielded :class:`Document` objects.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from locus.rag.loaders.oracle import OracleADBLoader
from locus.rag.stores.base import Document


# --------------------------------------------------------------------------- #
# Mock plumbing
# --------------------------------------------------------------------------- #


class _FakeCLOB:
    """Stand-in for oracledb's AsyncLOB — only the ``read`` method matters.

    A real ``AsyncLOB.read()`` is an awaitable that returns the CLOB
    bytes/str. We expose the same surface so the loader's ``hasattr(...,
    'read')`` branch fires and it actually awaits us.
    """

    def __init__(self, payload: str) -> None:
        self._payload = payload
        self.read_called = 0

    async def read(self) -> str:
        self.read_called += 1
        return self._payload


def _make_cursor(
    rows: list[tuple],
    description: list[tuple],
) -> MagicMock:
    """Build a mock async cursor that replays one batch of ``rows``.

    ``description`` follows DB-API: list of ``(name, type, ...)``.
    """
    cursor = MagicMock()
    cursor.arraysize = 0
    cursor.description = description
    cursor.execute = AsyncMock()

    # fetchmany returns rows once, then empty list to terminate the loop.
    call_state = {"served": False}

    async def _fetchmany(_size: int) -> list[tuple]:
        if call_state["served"]:
            return []
        call_state["served"] = True
        return rows

    cursor.fetchmany = _fetchmany

    # Async context-manager support — cursor is used via
    # ``async with conn.cursor() as cursor``.
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=None)
    return cursor


def _make_pool(cursor: MagicMock) -> MagicMock:
    """Build a mock pool whose ``acquire()`` yields a conn returning ``cursor``."""
    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cursor)
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=None)

    pool = MagicMock()
    # acquire() is sync-returning an async-context-manager.
    pool.acquire = MagicMock(return_value=conn)
    pool.close = AsyncMock()
    return pool


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


class TestValidation:
    def test_missing_sql_rejected(self) -> None:
        with pytest.raises(ValueError, match="sql"):
            OracleADBLoader(sql="", content_column="body", dsn="x")

    def test_missing_content_column_rejected(self) -> None:
        with pytest.raises(ValueError, match="content_column"):
            OracleADBLoader(sql="SELECT 1 FROM dual", content_column="", dsn="x")

    def test_bind_params_non_dict_rejected(self) -> None:
        with pytest.raises(TypeError, match="bind_params"):
            OracleADBLoader(
                sql="SELECT 1 FROM dual",
                content_column="body",
                bind_params=["not", "a", "dict"],  # type: ignore[arg-type]
                dsn="x",
            )

    def test_bind_param_key_unsafe_rejected(self) -> None:
        with pytest.raises(ValueError, match="bind_params key"):
            OracleADBLoader(
                sql="SELECT 1 FROM dual WHERE x = :x",
                content_column="body",
                bind_params={"x; DROP TABLE t --": 1},
                dsn="x",
            )

    def test_empty_bind_params_accepted(self) -> None:
        loader = OracleADBLoader(
            sql="SELECT body FROM t",
            content_column="body",
            bind_params={},
            dsn="x",
        )
        assert loader.bind_params == {}

    @pytest.mark.asyncio
    async def test_content_column_not_in_select_raises(self) -> None:
        cursor = _make_cursor(
            rows=[("hello",)],
            description=[("ID", None), ("BODY", None)],  # no "content"
        )
        pool = _make_pool(cursor)
        with patch("oracledb.create_pool_async", return_value=pool):
            loader = OracleADBLoader(
                sql="SELECT id, body FROM t",
                content_column="content",  # column actually named body
                dsn="x",
                user="u",
                password="p",  # noqa: S105, S106
            )
            with pytest.raises(ValueError, match="content_column"):
                async for _ in loader.lazy_load():
                    pass


# --------------------------------------------------------------------------- #
# Execution + mapping
# --------------------------------------------------------------------------- #


class TestLazyLoad:
    @pytest.mark.asyncio
    async def test_sql_and_binds_passed_through(self) -> None:
        cursor = _make_cursor(
            rows=[("1", "hello world", "alice")],
            description=[("ID", None), ("BODY", None), ("AUTHOR", None)],
        )
        pool = _make_pool(cursor)

        with patch("oracledb.create_pool_async", return_value=pool) as mock_pool_ctor:
            loader = OracleADBLoader(
                sql="SELECT id, body, author FROM articles WHERE topic = :topic",
                bind_params={"topic": "oracle"},
                content_column="body",
                id_column="id",
                metadata_columns=["author"],
                dsn="mydb_low",
                user="locus_app",
                password="secret",  # noqa: S105, S106
            )
            docs = [d async for d in loader.lazy_load()]

        # Pool was constructed exactly once with the connection envelope.
        assert mock_pool_ctor.call_count == 1
        kwargs = mock_pool_ctor.call_args.kwargs
        assert kwargs["user"] == "locus_app"
        assert kwargs["password"] == "secret"  # noqa: S105
        assert kwargs["dsn"] == "mydb_low"

        # execute() got our verbatim SQL plus the bind dict.
        cursor.execute.assert_awaited_once_with(
            "SELECT id, body, author FROM articles WHERE topic = :topic",
            {"topic": "oracle"},
        )
        # arraysize was set to the configured batch.
        assert cursor.arraysize == 100

        # Row → Document mapping is faithful.
        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].id == "1"
        assert docs[0].content == "hello world"
        assert docs[0].metadata == {"author": "alice"}

    @pytest.mark.asyncio
    async def test_default_metadata_includes_all_unmapped_columns(self) -> None:
        """When metadata_columns is omitted, everything not content/id flows
        into metadata — matches langchain-oracle's default behaviour."""
        cursor = _make_cursor(
            rows=[("42", "the body", "alice", "2026-01-01")],
            description=[
                ("ID", None),
                ("BODY", None),
                ("AUTHOR", None),
                ("CREATED", None),
            ],
        )
        pool = _make_pool(cursor)

        with patch("oracledb.create_pool_async", return_value=pool):
            loader = OracleADBLoader(
                sql="SELECT id, body, author, created FROM t",
                content_column="body",
                id_column="id",
                dsn="x",
                user="u",
                password="p",  # noqa: S105, S106
            )
            docs = [d async for d in loader.lazy_load()]

        assert docs[0].metadata == {"AUTHOR": "alice", "CREATED": "2026-01-01"}
        assert "BODY" not in docs[0].metadata  # content excluded
        assert "ID" not in docs[0].metadata  # id excluded

    @pytest.mark.asyncio
    async def test_clob_columns_are_read(self) -> None:
        """CLOB locators must be ``.read()`` awaited before reaching Document."""
        body_lob = _FakeCLOB("multi-megabyte article body")
        notes_lob = _FakeCLOB("clob in metadata too")
        cursor = _make_cursor(
            rows=[("doc-1", body_lob, notes_lob)],
            description=[("ID", None), ("BODY", None), ("NOTES", None)],
        )
        pool = _make_pool(cursor)

        with patch("oracledb.create_pool_async", return_value=pool):
            loader = OracleADBLoader(
                sql="SELECT id, body, notes FROM t",
                content_column="body",
                id_column="id",
                metadata_columns=["notes"],
                dsn="x",
                user="u",
                password="p",  # noqa: S105, S106
            )
            docs = [d async for d in loader.lazy_load()]

        # Both LOBs were awaited exactly once each.
        assert body_lob.read_called == 1
        assert notes_lob.read_called == 1
        assert docs[0].content == "multi-megabyte article body"
        assert docs[0].metadata == {"notes": "clob in metadata too"}

    @pytest.mark.asyncio
    async def test_missing_id_column_generates_uuid(self) -> None:
        cursor = _make_cursor(
            rows=[("the body", "alice")],
            description=[("BODY", None), ("AUTHOR", None)],
        )
        pool = _make_pool(cursor)

        with patch("oracledb.create_pool_async", return_value=pool):
            loader = OracleADBLoader(
                sql="SELECT body, author FROM t",
                content_column="body",
                dsn="x",
                user="u",
                password="p",  # noqa: S105, S106
            )
            docs = [d async for d in loader.lazy_load()]

        # No id_column supplied → uuid4 hex (32 chars).
        assert len(docs[0].id) == 32
        assert docs[0].content == "the body"

    @pytest.mark.asyncio
    async def test_load_eager_wrapper(self) -> None:
        cursor = _make_cursor(
            rows=[("1", "a"), ("2", "b")],
            description=[("ID", None), ("BODY", None)],
        )
        pool = _make_pool(cursor)

        with patch("oracledb.create_pool_async", return_value=pool):
            loader = OracleADBLoader(
                sql="SELECT id, body FROM t",
                content_column="body",
                id_column="id",
                dsn="x",
                user="u",
                password="p",  # noqa: S105, S106
            )
            docs = await loader.load()

        assert [d.id for d in docs] == ["1", "2"]
        assert [d.content for d in docs] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_close_releases_pool(self) -> None:
        cursor = _make_cursor(
            rows=[("1", "a")],
            description=[("ID", None), ("BODY", None)],
        )
        pool = _make_pool(cursor)

        with patch("oracledb.create_pool_async", return_value=pool):
            loader = OracleADBLoader(
                sql="SELECT id, body FROM t",
                content_column="body",
                id_column="id",
                dsn="x",
                user="u",
                password="p",  # noqa: S105, S106
            )
            _ = await loader.load()
            await loader.close()

        pool.close.assert_awaited_once()
        assert loader._pool is None


# --------------------------------------------------------------------------- #
# Hygiene
# --------------------------------------------------------------------------- #


class TestZeroLangchainDeps:
    def test_no_langchain_or_langgraph_imports(self) -> None:
        """The loader is locus-native — must not *import* langchain*/langgraph.

        We parse the module's AST and walk every Import/ImportFrom node
        rather than grepping the source so docstring mentions of
        "langchain-oracle" (which are deliberate — we cite the upstream
        analog) don't trigger a false positive.
        """
        import ast

        import locus.rag.loaders.oracle as mod

        with open(mod.__file__) as f:
            tree = ast.parse(f.read())

        forbidden_prefixes = ("langchain", "langgraph")
        offenders: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in forbidden_prefixes:
                        offenders.append(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in forbidden_prefixes:
                    offenders.append(node.module)

        assert offenders == [], f"Loader must not import langchain/langgraph; found: {offenders}"
