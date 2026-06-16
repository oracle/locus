# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for ``locus.rag.embeddings.oracle_indb`` (OracleInDBEmbeddings).

Stubs ``oracledb`` so we never need a real Oracle. The cursor stub
records every execute() and returns canned rows shaped like what
``VECTOR_SERIALIZE`` would actually produce — that's enough to pin the
SELECT shapes, the bind names (``:text`` for single, ``:t0``/``:t1``
for batch), the order preservation, and the ``[…]`` text parser.

Coverage matrix:

- Config: model_name + dimension stored correctly; bad identifier
  rejected; zero / negative dimension rejected.
- SQL shape: single uses ``UTL_TO_EMBEDDING`` with ``:text`` bind +
  ``FROM dual``; batch uses ``UTL_TO_EMBEDDINGS`` over ``JSON_ARRAY``
  with ``:t0..:tN`` binds and ``ORDER BY r``.
- Parse: ``[1.0, 2.0, 3.0]`` → ``[1.0, 2.0, 3.0]``; whitespace tolerant;
  empty array; bad input rejected.
- embed(): one row in, EmbeddingResult out with model + text echoed.
- embed_batch(): preserves input order, binds each text under
  ``t{i}``.
- embed_query() is an alias of embed().
- capabilities reflect ``use_batch_function``.
- close() releases the pool.
- Fallback: ``use_batch_function=False`` loops single calls.
"""

from __future__ import annotations

import sys
import types
from contextlib import asynccontextmanager
from typing import Any

import pytest

from locus.rag.embeddings.base import EmbeddingResult
from locus.rag.embeddings.oracle_indb import (
    OracleInDBEmbeddings,
    OracleInDBEmbeddingsConfig,
)


# ---------------------------------------------------------------------------
# Oracledb stub (mirrors tests/unit/test_oracle_store.py)
# ---------------------------------------------------------------------------


class _StubCursor:
    """Records every execute() and returns canned rows.

    ``fetchone_value`` and ``fetchall_value`` are static; tests that
    need to drive multiple distinct results should set
    ``fetchall_queue`` instead, which pops one entry per fetchall.
    """

    def __init__(
        self,
        *,
        fetchone: Any | None = None,
        fetchall: list[Any] | None = None,
    ) -> None:
        self.fetchone_value = fetchone
        self.fetchall_value = fetchall or []
        self.fetchall_queue: list[list[Any]] = []
        self.fetchone_queue: list[Any] = []
        self.execute_calls: list[tuple[str, dict[str, Any]]] = []

    async def execute(self, sql: str, params: dict[str, Any] | None = None) -> None:
        self.execute_calls.append((sql, params or {}))

    async def fetchone(self) -> Any:
        if self.fetchone_queue:
            return self.fetchone_queue.pop(0)
        return self.fetchone_value

    async def fetchall(self) -> list[Any]:
        if self.fetchall_queue:
            return self.fetchall_queue.pop(0)
        return self.fetchall_value

    async def __aenter__(self) -> _StubCursor:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _StubConn:
    def __init__(self, cursor: _StubCursor) -> None:
        self._cursor = cursor

    def cursor(self) -> _StubCursor:
        return self._cursor

    async def __aenter__(self) -> _StubConn:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _StubPool:
    def __init__(self, conn: _StubConn) -> None:
        self._conn = conn
        self.closed = False

    @asynccontextmanager
    async def acquire(self) -> Any:
        yield self._conn

    async def close(self) -> None:
        self.closed = True


def _install_oracledb_stub(
    monkeypatch: pytest.MonkeyPatch,
    cursor: _StubCursor,
) -> _StubPool:
    """Install a fake ``oracledb`` module that returns our stub pool."""
    pool = _StubPool(_StubConn(cursor))

    def fake_create_pool_async(*args: Any, **kwargs: Any) -> _StubPool:
        return pool

    def fake_makedsn(host: str, port: int, *, service_name: str) -> str:
        return f"(HOST={host})(PORT={port})(SERVICE_NAME={service_name})"

    fake = types.ModuleType("oracledb")
    fake.create_pool_async = fake_create_pool_async  # type: ignore[attr-defined]
    fake.makedsn = fake_makedsn  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "oracledb", fake)
    return pool


def _make_emb(**kwargs: Any) -> OracleInDBEmbeddings:
    """Factory: minimal-args constructor with sensible defaults for tests."""
    base: dict[str, Any] = {
        "model_name": "ALL_MINILM_L12_V2",
        "dimension": 384,
        "dsn": "x",
        "user": "u",
        "password": "p",  # noqa: S106
    }
    base.update(kwargs)
    return OracleInDBEmbeddings(**base)


# ---------------------------------------------------------------------------
# Config + constructor validation
# ---------------------------------------------------------------------------


class TestConfig:
    def test_constructor_stores_model_name_and_dimension(self) -> None:
        emb = _make_emb()
        assert emb.model_name == "ALL_MINILM_L12_V2"
        assert emb.config.dimension == 384
        # dimension also surfaced via the BaseEmbedding property
        assert emb.dimension == 384

    def test_bad_model_name_rejected(self) -> None:
        # Identifiers can't contain dots, spaces, or SQL metacharacters —
        # both as defence-in-depth against injection inside the JSON
        # literal and because Oracle model names must match that shape.
        with pytest.raises(ValueError, match="Invalid model_name"):
            OracleInDBEmbeddingsConfig(
                model_name="bad model",
                dimension=384,
            )

    def test_sql_injection_model_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid model_name"):
            OracleInDBEmbeddingsConfig(
                model_name="X','injected','y",
                dimension=384,
            )

    def test_zero_dimension_rejected(self) -> None:
        with pytest.raises(ValueError, match="dimension"):
            OracleInDBEmbeddingsConfig(model_name="ALL_MINILM_L12_V2", dimension=0)

    def test_negative_dimension_rejected(self) -> None:
        with pytest.raises(ValueError, match="dimension"):
            OracleInDBEmbeddingsConfig(model_name="ALL_MINILM_L12_V2", dimension=-1)


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_default_capabilities_advertise_no_batching(self) -> None:
        # Default is now ``use_batch_function=False`` because the batch
        # TABLE path is unreliable on Autonomous Database 23.26 — see
        # the OracleInDBEmbeddingsConfig docstring.
        emb = _make_emb()
        caps = emb.capabilities
        assert caps.supports_batching is False
        assert caps.max_batch_size == 1
        assert caps.supports_query_vs_doc is False
        assert caps.supports_multimodal is False

    def test_batch_enabled_flag_flips_capabilities(self) -> None:
        emb = _make_emb(use_batch_function=True)
        caps = emb.capabilities
        assert caps.supports_batching is True
        assert caps.max_batch_size == 96


# ---------------------------------------------------------------------------
# SQL shape — single + batch
# ---------------------------------------------------------------------------


class TestSqlShape:
    def test_single_sql_uses_utl_to_embedding_with_text_bind(self) -> None:
        emb = _make_emb()
        sql = emb._single_sql()
        assert "DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDING" in sql
        assert ":text" in sql
        assert "FROM dual" in sql
        # Model name is splatted into the JSON literal (not bound).
        assert '"model":"ALL_MINILM_L12_V2"' in sql
        assert '"provider":"database"' in sql
        # VECTOR_SERIALIZE wraps the result so we can fetch it as text.
        assert "VECTOR_SERIALIZE" in sql

    def test_batch_sql_uses_utl_to_embeddings_with_chunk_objects(self) -> None:
        # The batch path builds chunk objects with ``chunk_id`` +
        # ``chunk_data`` keys; ``UTL_TO_EMBEDDINGS`` returns one CLOB per
        # input which ``JSON_TABLE`` decomposes into ``embed_vector``
        # rows ordered by ``embed_id`` — guarantees output order matches
        # input order.
        emb = _make_emb()
        sql = emb._batch_sql(3)
        assert "DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDINGS" in sql
        assert "JSON_OBJECT('chunk_id' VALUE 1, 'chunk_data' VALUE :t0)" in sql
        assert "JSON_OBJECT('chunk_id' VALUE 2, 'chunk_data' VALUE :t1)" in sql
        assert "JSON_OBJECT('chunk_id' VALUE 3, 'chunk_data' VALUE :t2)" in sql
        assert "JSON_TABLE(" in sql
        assert "embed_vector CLOB PATH '$.embed_vector'" in sql
        assert "ORDER BY et.embed_id" in sql
        assert '"model":"ALL_MINILM_L12_V2"' in sql


# ---------------------------------------------------------------------------
# VECTOR_SERIALIZE parser
# ---------------------------------------------------------------------------


class TestVectorParser:
    def test_parses_simple_array(self) -> None:
        assert OracleInDBEmbeddings._parse_serialized_vector("[1.0, 2.0, 3.0]") == [
            1.0,
            2.0,
            3.0,
        ]

    def test_parses_negative_and_scientific(self) -> None:
        out = OracleInDBEmbeddings._parse_serialized_vector("[-0.5, 1.5e-3, 0.0]")
        assert out == [-0.5, 0.0015, 0.0]

    def test_tolerates_whitespace_and_newline(self) -> None:
        # VECTOR_SERIALIZE often emits a trailing newline through TO_CLOB.
        out = OracleInDBEmbeddings._parse_serialized_vector("  [ 1.0 , 2.0 ]\n")
        assert out == [1.0, 2.0]

    def test_empty_array(self) -> None:
        assert OracleInDBEmbeddings._parse_serialized_vector("[]") == []

    def test_missing_brackets_rejected(self) -> None:
        with pytest.raises(ValueError, match="no brackets"):
            OracleInDBEmbeddings._parse_serialized_vector("1.0, 2.0")

    def test_null_rejected(self) -> None:
        with pytest.raises(ValueError, match="NULL"):
            OracleInDBEmbeddings._parse_serialized_vector(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# embed() — single-text path
# ---------------------------------------------------------------------------


class TestEmbedSingle:
    @pytest.mark.asyncio
    async def test_embed_invokes_single_sql_with_text_bind(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cursor = _StubCursor(fetchone=("[1.0, 2.0, 3.0]",))
        _install_oracledb_stub(monkeypatch, cursor)

        emb = _make_emb(dimension=3)
        result = await emb.embed("hello world")

        # SQL shape + bind
        sql, params = cursor.execute_calls[-1]
        assert "DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDING" in sql
        assert "DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDINGS" not in sql
        assert params == {"text": "hello world"}

        # Result shape
        assert isinstance(result, EmbeddingResult)
        assert result.embedding == [1.0, 2.0, 3.0]
        assert result.text == "hello world"
        assert result.model == "ALL_MINILM_L12_V2"
        assert result.tokens is None

    @pytest.mark.asyncio
    async def test_embed_raises_when_no_row_returned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cursor = _StubCursor(fetchone=None)
        _install_oracledb_stub(monkeypatch, cursor)
        emb = _make_emb()

        with pytest.raises(RuntimeError, match="no rows"):
            await emb.embed("anything")


# ---------------------------------------------------------------------------
# embed_batch() — batch path
# ---------------------------------------------------------------------------


class TestEmbedBatch:
    @pytest.mark.asyncio
    async def test_embed_batch_binds_each_text_and_preserves_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # UTL_TO_EMBEDDINGS via JSON_TABLE yields
        # (embed_vector CLOB, embed_id INT) per chunk; embed_id is
        # consumed only for ordering, we don't read column [1].
        cursor = _StubCursor(
            fetchall=[
                ("[0.1, 0.2]", 1),
                ("[0.3, 0.4]", 2),
                ("[0.5, 0.6]", 3),
            ]
        )
        _install_oracledb_stub(monkeypatch, cursor)

        # Explicitly opt in to the batch path — default is now
        # sequential per the OracleInDBEmbeddingsConfig change.
        emb = _make_emb(dimension=2, use_batch_function=True)
        results = await emb.embed_batch(["alpha", "beta", "gamma"])

        # SQL + binds
        sql, params = cursor.execute_calls[-1]
        assert "DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDINGS" in sql
        assert params == {"t0": "alpha", "t1": "beta", "t2": "gamma"}

        # Results in input order, each carrying the right text + vector.
        assert [r.text for r in results] == ["alpha", "beta", "gamma"]
        assert results[0].embedding == [0.1, 0.2]
        assert results[1].embedding == [0.3, 0.4]
        assert results[2].embedding == [0.5, 0.6]
        assert all(r.model == "ALL_MINILM_L12_V2" for r in results)

    @pytest.mark.asyncio
    async def test_embed_batch_empty_input_short_circuits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cursor = _StubCursor()
        _install_oracledb_stub(monkeypatch, cursor)
        emb = _make_emb()

        result = await emb.embed_batch([])
        assert result == []
        # Critical: empty input must NOT issue SQL — JSON_ARRAY() with
        # zero binds is malformed and would 1) explode on the DB and
        # 2) waste a round-trip.
        assert cursor.execute_calls == []

    @pytest.mark.asyncio
    async def test_embed_batch_fallback_loops_single_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Queue one fetchone per text — fallback path issues N single
        # SELECTs back-to-back on the same connection.
        cursor = _StubCursor()
        cursor.fetchone_queue = [("[1.0]",), ("[2.0]",)]
        _install_oracledb_stub(monkeypatch, cursor)

        emb = _make_emb(dimension=1, use_batch_function=False)
        results = await emb.embed_batch(["a", "b"])

        assert len(cursor.execute_calls) == 2
        # Each call uses the *single* SQL shape, with a distinct text bind.
        for call_sql, call_params in cursor.execute_calls:
            assert "UTL_TO_EMBEDDING" in call_sql
            assert "UTL_TO_EMBEDDINGS" not in call_sql
            assert "text" in call_params

        assert [r.embedding for r in results] == [[1.0], [2.0]]
        assert [r.text for r in results] == ["a", "b"]


# ---------------------------------------------------------------------------
# embed_query alias
# ---------------------------------------------------------------------------


class TestEmbedQuery:
    @pytest.mark.asyncio
    async def test_embed_query_uses_single_sql_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cursor = _StubCursor(fetchone=("[9.0, 8.0]",))
        _install_oracledb_stub(monkeypatch, cursor)
        emb = _make_emb(dimension=2)

        result = await emb.embed_query("find me")

        # Same SQL as embed() — no query/doc differentiation for in-DB ONNX.
        sql, params = cursor.execute_calls[-1]
        assert "UTL_TO_EMBEDDING" in sql
        assert "UTL_TO_EMBEDDINGS" not in sql
        assert params == {"text": "find me"}
        assert result.embedding == [9.0, 8.0]
        assert result.text == "find me"


# ---------------------------------------------------------------------------
# CLOB handling
# ---------------------------------------------------------------------------


class _StubAsyncLOB:
    """Mimics oracledb.AsyncLOB — .read() returns an awaitable yielding str."""

    def __init__(self, payload: str) -> None:
        self._payload = payload

    def read(self) -> Any:
        async def _read() -> str:
            return self._payload

        return _read()


class TestClobHandling:
    @pytest.mark.asyncio
    async def test_embed_resolves_async_lob(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Pretend TO_CLOB came back as an AsyncLOB rather than a str —
        # the wrapper must .read() it transparently.
        cursor = _StubCursor(fetchone=(_StubAsyncLOB("[7.0, 8.0]"),))
        _install_oracledb_stub(monkeypatch, cursor)
        emb = _make_emb(dimension=2)

        result = await emb.embed("clob path")
        assert result.embedding == [7.0, 8.0]


# ---------------------------------------------------------------------------
# Lifecycle — pool close()
# ---------------------------------------------------------------------------


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_close_releases_pool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cursor = _StubCursor(fetchone=("[1.0]",))
        pool = _install_oracledb_stub(monkeypatch, cursor)
        emb = _make_emb(dimension=1)

        # Force the pool to materialise.
        await emb.embed("x")
        assert emb._pool is pool

        await emb.close()
        assert pool.closed is True
        assert emb._pool is None

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cursor = _StubCursor()
        _install_oracledb_stub(monkeypatch, cursor)
        emb = _make_emb()
        # No pool yet — close() should be a no-op, not an error.
        await emb.close()
        assert emb._pool is None


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_shape(self) -> None:
        emb = _make_emb()
        r = repr(emb)
        assert "OracleInDBEmbeddings" in r
        assert "ALL_MINILM_L12_V2" in r
        assert "384" in r
