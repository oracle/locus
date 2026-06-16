# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for the sync-API variants of the Oracle primitives.

Covers:

* :func:`locus._sync.run_sync` — both "no running loop" and
  "inside-a-running-loop" branches (the second exercises the
  background-thread fallback ``langgraph-oracledb`` uses).
* :func:`locus._sync.drain` — async-iterator → list drain helper.
* Every sync wrapper class:
    - Constructs with the same args as its async counterpart.
    - A representative sync method routes the call through the async
      instance (verified via ``AsyncMock.await_count`` / ``await_args``).
    - Async generators surface as drained lists.
* AST import check — every new sync file is free of langchain /
  langgraph imports, same pattern as ``test_oracle_adb_loader.py``.

The tests don't need a real Oracle: we either stub ``oracledb`` (so the
async pool builder is happy if it runs) or — for most cases — we swap
the wrapper's ``_async`` attribute with an :class:`AsyncMock` and
verify the routing.
"""

from __future__ import annotations

import ast
import asyncio
import importlib
import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from locus._sync import drain, run_sync


# ---------------------------------------------------------------------------
# oracledb stub (only needed for constructors that touch nothing — actual
# pool construction is lazy inside the async classes, so it's enough to
# install a stub for tests that *might* exercise it).
# ---------------------------------------------------------------------------


def _install_oracledb_stub(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Drop a minimal oracledb stub into sys.modules."""
    fake = types.ModuleType("oracledb")
    fake.DB_TYPE_CLOB = "CLOB-SENTINEL"  # type: ignore[attr-defined]
    fake.create_pool_async = MagicMock()  # type: ignore[attr-defined]
    fake.makedsn = MagicMock(return_value="dsn-string")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "oracledb", fake)
    return fake


# ---------------------------------------------------------------------------
# _sync.run_sync — both branches
# ---------------------------------------------------------------------------


class TestRunSync:
    def test_no_running_loop(self) -> None:
        """The ``asyncio.run`` branch fires when called from plain sync code."""
        calls: list[int] = []

        async def coro() -> int:
            calls.append(1)
            await asyncio.sleep(0)
            return 42

        result = run_sync(coro())
        assert result == 42
        assert calls == [1]

    def test_inside_running_loop_uses_background_thread(self) -> None:
        """When a loop is already running, run_sync spins one on a thread.

        Drive run_sync *from inside* a running loop so the
        ``get_running_loop`` succeeds and the threading branch
        executes. We can't ``await`` from within a synchronous test —
        instead we build our own loop and call run_until_complete on a
        coroutine that calls run_sync.
        """

        async def inner() -> int:
            await asyncio.sleep(0)
            return 7

        async def outer() -> int:
            # Inside this coroutine there IS a running loop — run_sync
            # must take the threading branch.
            return run_sync(inner())

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(outer())
        finally:
            loop.close()
        assert result == 7

    def test_propagates_exception(self) -> None:
        async def boom() -> None:
            raise ValueError("nope")

        with pytest.raises(ValueError, match="nope"):
            run_sync(boom())

    def test_propagates_exception_in_threaded_branch(self) -> None:
        async def boom() -> None:
            raise RuntimeError("threaded-boom")

        async def outer() -> None:
            run_sync(boom())

        loop = asyncio.new_event_loop()
        try:
            with pytest.raises(RuntimeError, match="threaded-boom"):
                loop.run_until_complete(outer())
        finally:
            loop.close()


class TestDrain:
    def test_drains_async_iterator(self) -> None:
        async def gen():
            for i in range(3):
                yield i

        result = run_sync(drain(gen()))
        assert result == [0, 1, 2]

    def test_empty_iterator(self) -> None:
        async def gen():
            if False:
                yield 0

        assert run_sync(drain(gen())) == []


# ---------------------------------------------------------------------------
# Helpers for swapping ._async with an AsyncMock
# ---------------------------------------------------------------------------


def _patch_async_with_mock(wrapper: Any, **method_returns: Any) -> AsyncMock:
    """Swap wrapper._async with a fresh AsyncMock.

    Returns the mock so callers can assert ``.method.await_count`` etc.
    Any keyword arg becomes a pre-configured AsyncMock method on the
    inner mock with that return value.
    """
    mock = AsyncMock()
    for name, retval in method_returns.items():
        setattr(mock, name, AsyncMock(return_value=retval))
    wrapper._async = mock
    return mock


# ---------------------------------------------------------------------------
# OracleSyncBackend
# ---------------------------------------------------------------------------


class TestOracleSyncBackend:
    def test_constructs_with_async_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_oracledb_stub(monkeypatch)
        from locus.memory.backends.oracle_sync import OracleSyncBackend

        wrapper = OracleSyncBackend(
            dsn="mydb_low",
            user="u",
            password="p",  # noqa: S106
            wallet_location="/wallets/mydb",
        )
        # Same config envelope as the async class.
        assert wrapper._async.config.dsn == "mydb_low"
        assert wrapper._async.config.user == "u"
        assert wrapper._async.config.wallet_location == "/wallets/mydb"

    def test_save_routes_through_async(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_oracledb_stub(monkeypatch)
        from locus.memory.backends.oracle_sync import OracleSyncBackend

        wrapper = OracleSyncBackend(dsn="x", user="u", password="p")  # noqa: S106
        mock = _patch_async_with_mock(wrapper, save="ckpt-id-1")

        out = wrapper.save({"x": 1}, "thread-A", metadata={"tag": "t1"})

        assert out == "ckpt-id-1"
        mock.save.assert_awaited_once()
        # Positional args + kwargs preserved.
        args, kwargs = mock.save.await_args
        assert args == ({"x": 1}, "thread-A")
        assert kwargs == {"checkpoint_id": None, "metadata": {"tag": "t1"}}

    def test_load_returns_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_oracledb_stub(monkeypatch)
        from locus.memory.backends.oracle_sync import OracleSyncBackend

        wrapper = OracleSyncBackend(dsn="x", user="u", password="p")  # noqa: S106
        mock = _patch_async_with_mock(wrapper, load={"step": 2})

        assert wrapper.load("thread-A") == {"step": 2}
        mock.load.assert_awaited_once_with("thread-A", checkpoint_id=None)

    def test_close_drives_async(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_oracledb_stub(monkeypatch)
        from locus.memory.backends.oracle_sync import OracleSyncBackend

        wrapper = OracleSyncBackend(dsn="x", user="u", password="p")  # noqa: S106
        mock = _patch_async_with_mock(wrapper, close=None)

        wrapper.close()
        mock.close.assert_awaited_once()

    def test_list_threads_threads_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_oracledb_stub(monkeypatch)
        from locus.memory.backends.oracle_sync import OracleSyncBackend

        wrapper = OracleSyncBackend(dsn="x", user="u", password="p")  # noqa: S106
        mock = _patch_async_with_mock(wrapper, list_threads=["t1", "t2"])

        out = wrapper.list_threads(limit=50, offset=10, pattern="prod-%")
        assert out == ["t1", "t2"]
        mock.list_threads.assert_awaited_once_with(limit=50, offset=10, pattern="prod-%")


# ---------------------------------------------------------------------------
# OracleSyncCheckpointSaver
# ---------------------------------------------------------------------------


class TestOracleSyncCheckpointSaver:
    def test_constructs_with_async_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_oracledb_stub(monkeypatch)
        from locus.memory.backends.oracle_sync import OracleSyncCheckpointSaver

        wrapper = OracleSyncCheckpointSaver(
            dsn="x",
            user="u",
            password="p",  # noqa: S105, S106
            table_name="myapp",  # noqa: S106
        )
        # Sanity: the underlying saver picked up the table prefix.
        assert wrapper._async._table_prefix == "myapp"

    def test_put_routes_through_async(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_oracledb_stub(monkeypatch)
        from locus.memory.backends.oracle_sync import OracleSyncCheckpointSaver

        wrapper = OracleSyncCheckpointSaver(dsn="x", user="u", password="p")  # noqa: S106
        mock = _patch_async_with_mock(wrapper, put=None)

        wrapper.put(thread_id="t1", checkpoint_id="c1", checkpoint_data={"step": 0})
        mock.put.assert_awaited_once_with(
            thread_id="t1",
            checkpoint_id="c1",
            checkpoint_data={"step": 0},
            checkpoint_ns="",
            parent_checkpoint_id=None,
            metadata=None,
        )

    def test_get_writes_routes_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_oracledb_stub(monkeypatch)
        from locus.memory.backends.oracle_sync import OracleSyncCheckpointSaver

        wrapper = OracleSyncCheckpointSaver(dsn="x", user="u", password="p")  # noqa: S106
        mock = _patch_async_with_mock(
            wrapper, get_writes=[{"task_id": "n1", "idx": 0, "channel": "x", "value": 1}]
        )

        out = wrapper.get_writes(thread_id="t1", checkpoint_id="c1", task_id="n1")
        assert out[0]["channel"] == "x"
        mock.get_writes.assert_awaited_once_with(
            thread_id="t1", checkpoint_id="c1", checkpoint_ns="", task_id="n1"
        )

    def test_delete_thread_routes_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_oracledb_stub(monkeypatch)
        from locus.memory.backends.oracle_sync import OracleSyncCheckpointSaver

        wrapper = OracleSyncCheckpointSaver(dsn="x", user="u", password="p")  # noqa: S106
        mock = _patch_async_with_mock(wrapper, delete_thread=None)
        wrapper.delete_thread("t1")
        mock.delete_thread.assert_awaited_once_with("t1")


# ---------------------------------------------------------------------------
# OracleSyncStore
# ---------------------------------------------------------------------------


class TestOracleSyncStore:
    def test_constructs_with_async_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_oracledb_stub(monkeypatch)
        from locus.memory.store_backends.oracle_sync import OracleSyncStore

        wrapper = OracleSyncStore(
            dsn="x",
            user="u",
            password="p",  # noqa: S105, S106
            dimension=1024,  # noqa: S106
        )
        assert wrapper._async.config.dimension == 1024
        # Capability passthrough.
        assert wrapper.capabilities.semantic_search is True

    def test_put_routes_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_oracledb_stub(monkeypatch)
        from locus.memory.store_backends.oracle_sync import OracleSyncStore

        wrapper = OracleSyncStore(dsn="x", user="u", password="p")  # noqa: S106
        mock = _patch_async_with_mock(wrapper, put=None)
        wrapper.put(("memory", "u1"), "theme", {"value": "dark"})
        mock.put.assert_awaited_once_with(("memory", "u1"), "theme", {"value": "dark"}, None)

    def test_search_by_embedding_routes_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_oracledb_stub(monkeypatch)
        from locus.memory.store_backends.oracle_sync import OracleSyncStore

        wrapper = OracleSyncStore(
            dsn="x",
            user="u",
            password="p",  # noqa: S105, S106
            dimension=4,  # noqa: S106
        )
        mock = _patch_async_with_mock(wrapper, search_by_embedding=[])
        out = wrapper.search_by_embedding(("ns",), [0.1, 0.2, 0.3, 0.4], limit=5)
        assert out == []
        mock.search_by_embedding.assert_awaited_once_with(
            ("ns",),
            [0.1, 0.2, 0.3, 0.4],
            limit=5,
            threshold=None,
            metadata_filter=None,
        )

    def test_close_routes_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_oracledb_stub(monkeypatch)
        from locus.memory.store_backends.oracle_sync import OracleSyncStore

        wrapper = OracleSyncStore(dsn="x", user="u", password="p")  # noqa: S106
        mock = _patch_async_with_mock(wrapper, close=None)
        wrapper.close()
        mock.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# OracleSyncVectorStore
# ---------------------------------------------------------------------------


class TestOracleSyncVectorStore:
    def test_constructs_with_async_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_oracledb_stub(monkeypatch)
        from locus.rag.stores.oracle_sync import OracleSyncVectorStore

        wrapper = OracleSyncVectorStore(
            dsn="x",
            user="u",
            password="p",  # noqa: S105, S106
            dimension=1536,  # noqa: S106
        )
        assert wrapper._async.oracle_config.dimension == 1536
        assert wrapper.config.dimension == 1536

    def test_add_routes_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_oracledb_stub(monkeypatch)
        from locus.rag.stores.base import Document
        from locus.rag.stores.oracle_sync import OracleSyncVectorStore

        wrapper = OracleSyncVectorStore(dsn="x", user="u", password="p")  # noqa: S106
        mock = _patch_async_with_mock(wrapper, add="doc-abc")
        doc = Document(id="d1", content="hello", embedding=[0.1] * 1024, metadata={})
        out = wrapper.add(doc)
        assert out == "doc-abc"
        mock.add.assert_awaited_once_with(doc)

    def test_search_routes_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_oracledb_stub(monkeypatch)
        from locus.rag.stores.oracle_sync import OracleSyncVectorStore

        wrapper = OracleSyncVectorStore(dsn="x", user="u", password="p")  # noqa: S106
        mock = _patch_async_with_mock(wrapper, search=[])
        wrapper.search(
            [0.1] * 1024,
            limit=7,
            metadata_filter={"category": "x"},
            mmr=True,
            mmr_lambda=0.3,
        )
        mock.search.assert_awaited_once_with(
            [0.1] * 1024,
            limit=7,
            threshold=None,
            metadata_filter={"category": "x"},
            mmr=True,
            mmr_lambda=0.3,
            mmr_candidate_pool=None,
        )

    def test_build_index_routes_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_oracledb_stub(monkeypatch)
        from locus.rag.stores.oracle_sync import OracleSyncVectorStore

        wrapper = OracleSyncVectorStore(dsn="x", user="u", password="p")  # noqa: S106
        mock = _patch_async_with_mock(wrapper, build_index=None)
        wrapper.build_index(rebuild=True)
        mock.build_index.assert_awaited_once_with(rebuild=True)


# ---------------------------------------------------------------------------
# OracleSyncADBLoader
# ---------------------------------------------------------------------------


class TestOracleSyncADBLoader:
    def test_constructs_with_async_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_oracledb_stub(monkeypatch)
        from locus.rag.loaders.oracle_sync import OracleSyncADBLoader

        wrapper = OracleSyncADBLoader(
            sql="SELECT id, body FROM t",
            content_column="body",
            id_column="id",
            dsn="x",
            user="u",
            password="p",  # noqa: S106
        )
        assert wrapper._async.sql == "SELECT id, body FROM t"
        assert wrapper._async.content_column == "body"
        assert wrapper._async.id_column == "id"

    def test_load_routes_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_oracledb_stub(monkeypatch)
        from locus.rag.loaders.oracle_sync import OracleSyncADBLoader
        from locus.rag.stores.base import Document

        wrapper = OracleSyncADBLoader(
            sql="SELECT id, body FROM t",
            content_column="body",
            dsn="x",
            user="u",
            password="p",  # noqa: S106
        )
        docs = [Document(id="1", content="a", embedding=None, metadata={})]
        mock = _patch_async_with_mock(wrapper, load=docs)
        out = wrapper.load()
        assert out == docs
        mock.load.assert_awaited_once_with()

    def test_lazy_load_drains_async_generator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_oracledb_stub(monkeypatch)
        from locus.rag.loaders.oracle_sync import OracleSyncADBLoader
        from locus.rag.stores.base import Document

        wrapper = OracleSyncADBLoader(
            sql="SELECT id, body FROM t",
            content_column="body",
            dsn="x",
            user="u",
            password="p",  # noqa: S106
        )

        docs = [
            Document(id="1", content="a", embedding=None, metadata={}),
            Document(id="2", content="b", embedding=None, metadata={}),
        ]

        async def fake_lazy():
            for d in docs:
                yield d

        # Underlying async class is a Pydantic model so direct attribute
        # set is blocked. Drop into __dict__ to mask the method.
        wrapper._async.__dict__["lazy_load"] = fake_lazy

        out = wrapper.lazy_load()
        assert isinstance(out, list)
        assert out == docs


# ---------------------------------------------------------------------------
# OracleSyncInDBChunker
# ---------------------------------------------------------------------------


class TestOracleSyncInDBChunker:
    def test_constructs_with_async_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_oracledb_stub(monkeypatch)
        from locus.rag.chunkers.oracle_sync import OracleSyncInDBChunker

        wrapper = OracleSyncInDBChunker(
            dsn="x",
            user="u",
            password="p",  # noqa: S106
            max_tokens=256,
            overlap=16,
        )
        assert wrapper._async.params.max == 256
        assert wrapper._async.params.overlap == 16

    def test_chunk_text_routes_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_oracledb_stub(monkeypatch)
        from locus.rag.chunkers.oracle_sync import OracleSyncInDBChunker

        wrapper = OracleSyncInDBChunker(dsn="x", user="u", password="p")  # noqa: S106
        mock = _patch_async_with_mock(
            wrapper,
            chunk_text=[{"chunk_id": 1, "text": "hello", "offset": 0, "length": 5}],
        )
        out = wrapper.chunk_text("hello world")
        assert out[0]["text"] == "hello"
        mock.chunk_text.assert_awaited_once_with("hello world")

    def test_chunk_column_drains_async_generator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_oracledb_stub(monkeypatch)
        from locus.rag.chunkers.oracle_sync import OracleSyncInDBChunker

        wrapper = OracleSyncInDBChunker(dsn="x", user="u", password="p")  # noqa: S106

        async def fake_chunk_column(**_kwargs: Any):
            for cid in range(3):
                yield {"source_id": "row-1", "chunk_id": cid, "text": f"c{cid}"}

        # Underlying async class is a Pydantic model so direct attribute
        # set is blocked. Drop into __dict__ to mask the method.
        wrapper._async.__dict__["chunk_column"] = fake_chunk_column

        out = wrapper.chunk_column(table_name="docs", text_column="body")
        assert isinstance(out, list)
        assert len(out) == 3
        assert out[0]["chunk_id"] == 0
        assert out[2]["chunk_id"] == 2


# ---------------------------------------------------------------------------
# OracleSyncInDBEmbeddings
# ---------------------------------------------------------------------------


class TestOracleSyncInDBEmbeddings:
    def test_constructs_with_async_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_oracledb_stub(monkeypatch)
        from locus.rag.embeddings.oracle_sync import OracleSyncInDBEmbeddings

        wrapper = OracleSyncInDBEmbeddings(
            model_name="ALL_MINILM_L12_V2",
            dimension=384,
            dsn="x",
            user="u",
            password="p",  # noqa: S106
        )
        assert wrapper.model_name == "ALL_MINILM_L12_V2"
        assert wrapper.config.dimension == 384

    def test_embed_routes_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_oracledb_stub(monkeypatch)
        from locus.rag.embeddings.base import EmbeddingResult
        from locus.rag.embeddings.oracle_sync import OracleSyncInDBEmbeddings

        wrapper = OracleSyncInDBEmbeddings(
            model_name="ALL_MINILM_L12_V2",
            dimension=4,
            dsn="x",
            user="u",
            password="p",  # noqa: S106
        )
        expected = EmbeddingResult(
            embedding=[0.1, 0.2, 0.3, 0.4],
            text="hello",
            model="ALL_MINILM_L12_V2",
            tokens=None,
        )
        mock = _patch_async_with_mock(wrapper, embed=expected)
        out = wrapper.embed("hello")
        assert out is expected
        mock.embed.assert_awaited_once_with("hello")

    def test_embed_batch_routes_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_oracledb_stub(monkeypatch)
        from locus.rag.embeddings.oracle_sync import OracleSyncInDBEmbeddings

        wrapper = OracleSyncInDBEmbeddings(
            model_name="ALL_MINILM_L12_V2",
            dimension=4,
            dsn="x",
            user="u",
            password="p",  # noqa: S106
        )
        mock = _patch_async_with_mock(wrapper, embed_batch=[])
        wrapper.embed_batch(["a", "b"])
        mock.embed_batch.assert_awaited_once_with(["a", "b"])

    def test_embed_query_routes_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_oracledb_stub(monkeypatch)
        from locus.rag.embeddings.base import EmbeddingResult
        from locus.rag.embeddings.oracle_sync import OracleSyncInDBEmbeddings

        wrapper = OracleSyncInDBEmbeddings(
            model_name="ALL_MINILM_L12_V2",
            dimension=4,
            dsn="x",
            user="u",
            password="p",  # noqa: S106
        )
        expected = EmbeddingResult(
            embedding=[0.0] * 4, text="q", model="ALL_MINILM_L12_V2", tokens=None
        )
        mock = _patch_async_with_mock(wrapper, embed_query=expected)
        out = wrapper.embed_query("q")
        assert out is expected
        mock.embed_query.assert_awaited_once_with("q")


# ---------------------------------------------------------------------------
# Hygiene — zero langchain/langgraph imports in every new sync module
# ---------------------------------------------------------------------------


class TestZeroLangchainDeps:
    """AST scan: every new sync file is langchain/langgraph-free."""

    _SYNC_MODULES = [
        "locus._sync",
        "locus.memory.backends.oracle_sync",
        "locus.memory.store_backends.oracle_sync",
        "locus.rag.stores.oracle_sync",
        "locus.rag.loaders.oracle_sync",
        "locus.rag.chunkers.oracle_sync",
        "locus.rag.embeddings.oracle_sync",
    ]

    @pytest.mark.parametrize("module_name", _SYNC_MODULES)
    def test_no_langchain_or_langgraph_imports(self, module_name: str) -> None:
        mod = importlib.import_module(module_name)
        with open(mod.__file__ or "") as f:  # type: ignore[arg-type]
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

        assert offenders == [], (
            f"{module_name} must not import langchain/langgraph; found: {offenders}"
        )


# ---------------------------------------------------------------------------
# Public-export smoke test mirroring the verification block in the spec
# ---------------------------------------------------------------------------


class TestPublicExports:
    def test_all_seven_classes_importable(self) -> None:
        from locus.memory.backends import OracleSyncBackend, OracleSyncCheckpointSaver
        from locus.memory.store_backends import OracleSyncStore
        from locus.rag.chunkers import OracleSyncInDBChunker
        from locus.rag.embeddings import OracleSyncInDBEmbeddings
        from locus.rag.loaders import OracleSyncADBLoader
        from locus.rag.stores import OracleSyncVectorStore

        for cls in (
            OracleSyncBackend,
            OracleSyncCheckpointSaver,
            OracleSyncStore,
            OracleSyncVectorStore,
            OracleSyncADBLoader,
            OracleSyncInDBChunker,
            OracleSyncInDBEmbeddings,
        ):
            assert isinstance(cls, type), f"{cls!r} is not a class"
