# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for OracleInDBChunker — server-side chunking via
DBMS_VECTOR_CHAIN.UTL_TO_CHUNKS. Mocked pool — no DB required.

Verifies the JSON params blob, the SELECT shape, identifier safety
for the column-mode call, and the value coercion on the way out.
"""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest


def _stub_oracledb(monkeypatch):
    """Install a stub ``oracledb`` module with a controllable async pool."""
    pool = MagicMock()
    cursor = MagicMock()

    # Async context-manager glue
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=None)
    cursor.execute = AsyncMock()
    cursor.fetchall = AsyncMock(return_value=[])
    cursor.setinputsizes = MagicMock()

    conn = MagicMock()
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=None)
    conn.cursor = MagicMock(return_value=cursor)

    pool.acquire = MagicMock(return_value=conn)
    pool.close = AsyncMock()

    mod = types.ModuleType("oracledb")
    mod.create_pool_async = MagicMock(return_value=pool)
    mod.makedsn = MagicMock(return_value="host:1521/svc")
    mod.AsyncConnectionPool = MagicMock
    monkeypatch.setitem(sys.modules, "oracledb", mod)
    return mod, pool, cursor


@pytest.fixture
def stub(monkeypatch):
    return _stub_oracledb(monkeypatch)


from locus.rag.chunkers import OracleInDBChunker  # noqa: E402


class TestConstructor:
    def test_rejects_bad_by(self) -> None:
        with pytest.raises(ValueError, match=r"^by must be one of"):
            OracleInDBChunker(dsn="x", user="u", password="p", by="bogus")  # noqa: S105, S106

    def test_rejects_bad_normalize(self) -> None:
        with pytest.raises(ValueError, match=r"^normalize must be one of"):
            OracleInDBChunker(dsn="x", user="u", password="p", normalize="lowercase")  # noqa: S105, S106

    def test_rejects_zero_max_tokens(self) -> None:
        with pytest.raises(ValueError, match=r"^max_tokens must be"):
            OracleInDBChunker(dsn="x", user="u", password="p", max_tokens=0)  # noqa: S105, S106

    def test_rejects_negative_overlap(self) -> None:
        with pytest.raises(ValueError, match=r"^overlap must be"):
            OracleInDBChunker(dsn="x", user="u", password="p", overlap=-1)  # noqa: S105, S106

    def test_defaults_match_oracle_recommended(self) -> None:
        c = OracleInDBChunker(dsn="x", user="u", password="p")  # noqa: S105, S106
        params = json.loads(c._params_json())
        assert params == {
            "by": "words",
            "max": 100,
            "overlap": 0,
            "split": "recursively",
            "normalize": "all",
        }


class TestChunkText:
    @pytest.mark.asyncio
    async def test_empty_text_returns_empty_list(self, stub) -> None:
        _, _, cursor = stub
        c = OracleInDBChunker(dsn="x", user="u", password="p")  # noqa: S105, S106
        out = await c.chunk_text("")
        assert out == []
        cursor.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_string_rejected(self, stub) -> None:
        c = OracleInDBChunker(dsn="x", user="u", password="p")  # noqa: S105, S106
        with pytest.raises(TypeError, match=r"text must be str"):
            await c.chunk_text(123)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_binds_text_and_params_json(self, stub) -> None:
        _, _, cursor = stub
        cursor.fetchall = AsyncMock(return_value=[])

        c = OracleInDBChunker(
            dsn="x",
            user="u",
            password="p",  # noqa: S105, S106
            max_tokens=42,
            overlap=4,
            by="chars",
        )
        await c.chunk_text("hello world")

        assert cursor.execute.await_count == 1
        call = cursor.execute.await_args
        sql, params = call.args
        assert "DBMS_VECTOR_CHAIN.UTL_TO_CHUNKS" in sql
        assert params["text"] == "hello world"
        # params blob is JSON with our overrides applied.
        parsed = json.loads(params["params"])
        assert parsed["max"] == 42
        assert parsed["overlap"] == 4
        assert parsed["by"] == "chars"

    @pytest.mark.asyncio
    async def test_parses_rows_into_dicts(self, stub) -> None:
        _, _, cursor = stub
        # Each row mirrors what JSON_VALUE on UTL_TO_CHUNKS yields:
        # (chunk_id, offset, length, text). Pass as strings — that's
        # what JSON_VALUE returns by default.
        cursor.fetchall = AsyncMock(
            return_value=[("1", "0", "5", "hello"), ("2", "6", "5", "world")]
        )

        c = OracleInDBChunker(dsn="x", user="u", password="p")  # noqa: S105, S106
        out = await c.chunk_text("hello world")

        assert out == [
            {"chunk_id": 1, "offset": 0, "length": 5, "text": "hello"},
            {"chunk_id": 2, "offset": 6, "length": 5, "text": "world"},
        ]


class TestChunkColumn:
    @pytest.mark.asyncio
    async def test_validates_identifiers(self, stub) -> None:
        c = OracleInDBChunker(dsn="x", user="u", password="p")  # noqa: S105, S106
        with pytest.raises(ValueError, match=r"^Invalid table_name"):
            async for _ in c.chunk_column(table_name="bad.name", text_column="t"):
                pass
        with pytest.raises(ValueError, match=r"^Invalid text_column"):
            async for _ in c.chunk_column(table_name="t", text_column="; DROP"):
                pass

    @pytest.mark.asyncio
    async def test_rejects_where_with_semicolon(self, stub) -> None:
        c = OracleInDBChunker(dsn="x", user="u", password="p")  # noqa: S105, S106
        with pytest.raises(ValueError, match=r"where clause must not contain"):
            async for _ in c.chunk_column(
                table_name="docs", text_column="body", where="topic = 'x'; DROP TABLE docs"
            ):
                pass


class TestClose:
    @pytest.mark.asyncio
    async def test_close_releases_pool(self, stub) -> None:
        _, pool, _ = stub
        c = OracleInDBChunker(dsn="x", user="u", password="p")  # noqa: S105, S106
        await c._get_pool()
        assert c._pool is pool
        await c.close()
        pool.close.assert_awaited()
        assert c._pool is None


class TestNoLangchain:
    def test_no_langchain_or_langgraph_imports(self) -> None:
        """Belt-and-braces grep: ensure the new module hasn't sprouted
        a langchain dep. The package promises native-only Oracle
        integration."""
        import ast
        from pathlib import Path

        src = Path(__file__).resolve().parents[2] / "src/locus/rag/chunkers/oracle_indb.py"
        tree = ast.parse(src.read_text())
        bad = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith(("langchain", "langgraph")):
                    bad.append(node.module)
            elif isinstance(node, ast.Import):
                for n in node.names:
                    if n.name.startswith(("langchain", "langgraph")):
                        bad.append(n.name)
        assert bad == [], f"forbidden imports: {bad}"
