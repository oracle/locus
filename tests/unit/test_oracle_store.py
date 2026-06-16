# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for ``locus.memory.store_backends.oracle`` (OracleStore).

Stubs ``oracledb`` so we never need a real Oracle. The cursor stub
records the SQL it sees and returns canned rows — that's enough to
pin the MERGE / SELECT / DELETE shapes and the namespace flattening
round-trip.

Coverage matrix:

- Config: SQL identifier validation, distance-metric check, dimension
  positive-int guard.
- Namespace helpers: flatten/parse round-trip, separator-in-part guard,
  empty-tuple case.
- DDL: VECTOR column emitted when dimension is set, *omitted* when
  dimension is None (text-only mode).
- MERGE shape: upsert grammar covers both with-embedding and
  without-embedding variants.
- put / get / delete / list_keys / search / list_namespaces:
  the bind dict carries the flattened namespace string, not the tuple.
- list_namespaces: prefix builds ``LIKE prefix/ %`` pattern.
- Capabilities: ``semantic_search`` flips with ``dimension``.
- Async-prefix aliases exist for langgraph-oracledb parity.
"""

from __future__ import annotations

import sys
import types
from contextlib import asynccontextmanager
from typing import Any

import pytest

from locus.memory.store_backends.oracle import (
    OracleStore,
    OracleStoreConfig,
    _flatten_namespace,
    _parse_namespace,
)


# ---------------------------------------------------------------------------
# Oracledb stub
# ---------------------------------------------------------------------------


class _StubCursor:
    """Records every execute() call and returns canned rows."""

    def __init__(
        self,
        *,
        fetchone: Any | None = None,
        fetchall: list[Any] | None = None,
    ) -> None:
        self.fetchone_value = fetchone
        self.fetchall_value = fetchall or []
        self.execute_calls: list[tuple[str, dict[str, Any]]] = []
        self.input_sizes: dict[str, Any] = {}
        self.rowcount: int = 0

    async def execute(self, sql: str, params: dict[str, Any] | None = None) -> None:
        self.execute_calls.append((sql, params or {}))
        # Simulate a successful DELETE / UPDATE returning rowcount=1
        # when the SQL is a DELETE — keeps the delete() return-value
        # assertions honest.
        if sql.lstrip().upper().startswith("DELETE"):
            self.rowcount = 1

    async def fetchone(self) -> Any:
        return self.fetchone_value

    async def fetchall(self) -> list[Any]:
        return self.fetchall_value

    def setinputsizes(self, **kwargs: Any) -> None:
        self.input_sizes.update(kwargs)

    async def __aenter__(self) -> _StubCursor:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _StubConn:
    def __init__(self, cursor: _StubCursor) -> None:
        self._cursor = cursor
        self.committed = False

    def cursor(self) -> _StubCursor:
        # In real oracledb this returns an awaitable context manager.
        # Our stub cursor *is* the context manager.
        return self._cursor

    async def commit(self) -> None:
        self.committed = True

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
    """Install a fake ``oracledb`` module returning our stub pool."""
    pool = _StubPool(_StubConn(cursor))

    def fake_create_pool_async(*args: Any, **kwargs: Any) -> _StubPool:
        return pool

    def fake_makedsn(host: str, port: int, *, service_name: str) -> str:
        return f"(DESCRIPTION=(ADDRESS=(HOST={host})(PORT={port}))(SERVICE_NAME={service_name}))"

    fake = types.ModuleType("oracledb")
    fake.create_pool_async = fake_create_pool_async  # type: ignore[attr-defined]
    fake.makedsn = fake_makedsn  # type: ignore[attr-defined]
    fake.DB_TYPE_CLOB = "CLOB-SENTINEL"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "oracledb", fake)
    return pool


def _make_store(**kwargs: Any) -> OracleStore:
    base: dict[str, Any] = {"dsn": "x", "user": "u", "password": "p"}  # noqa: S106
    base.update(kwargs)
    return OracleStore(**base)


# ---------------------------------------------------------------------------
# Namespace flattening
# ---------------------------------------------------------------------------


class TestNamespaceHelpers:
    def test_round_trip_simple(self) -> None:
        ns = ("memory", "user-42", "facts")
        assert _parse_namespace(_flatten_namespace(ns)) == ns

    def test_round_trip_empty(self) -> None:
        assert _flatten_namespace(()) == ""
        assert _parse_namespace("") == ()

    def test_round_trip_single(self) -> None:
        ns = ("global",)
        assert _parse_namespace(_flatten_namespace(ns)) == ns

    def test_separator_in_part_rejected(self) -> None:
        with pytest.raises(ValueError, match="separator"):
            _flatten_namespace(("memory", "user/42"))

    def test_non_str_part_rejected(self) -> None:
        with pytest.raises(TypeError, match="must be str"):
            _flatten_namespace(("memory", 42))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = OracleStoreConfig()
        assert cfg.table_name == "locus_store"
        assert cfg.dimension is None
        assert cfg.distance_metric == "COSINE"
        assert cfg.auto_create_table is True

    def test_bad_table_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid table_name"):
            OracleStoreConfig(table_name="bad table")

    def test_sql_injection_table_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid table_name"):
            OracleStoreConfig(table_name="locus; DROP TABLE users--")

    def test_bad_schema_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid schema_name"):
            OracleStoreConfig(schema_name="1bad")

    def test_unknown_distance_metric_rejected(self) -> None:
        with pytest.raises(ValueError, match="distance_metric"):
            OracleStoreConfig(distance_metric="BOGUS")

    def test_zero_dimension_rejected(self) -> None:
        with pytest.raises(ValueError, match="dimension"):
            OracleStoreConfig(dimension=0)

    def test_dimension_none_allowed(self) -> None:
        cfg = OracleStoreConfig(dimension=None)
        assert cfg.dimension is None


# ---------------------------------------------------------------------------
# DDL generation
# ---------------------------------------------------------------------------


class TestDDL:
    def test_create_table_with_vector(self) -> None:
        store = _make_store(dimension=1024)
        ddl = store._create_table_ddl()
        assert "CREATE TABLE locus_store" in ddl
        assert "embedding VECTOR(1024, FLOAT32)" in ddl
        assert "PRIMARY KEY (namespace, key)" in ddl
        assert "value      CLOB CHECK (value IS JSON)" in ddl

    def test_create_table_without_vector_when_dimension_none(self) -> None:
        store = _make_store(dimension=None)
        ddl = store._create_table_ddl()
        # Critical: text-only mode must skip the VECTOR column so the
        # DDL is valid on Oracle versions without native vector support.
        assert "VECTOR" not in ddl
        assert "embedding" not in ddl
        # The other columns + PK constraint still appear.
        assert "namespace" in ddl
        assert "key" in ddl
        assert "PRIMARY KEY (namespace, key)" in ddl

    def test_create_table_respects_schema(self) -> None:
        store = _make_store(schema_name="LOCUS_APP", table_name="my_store")
        ddl = store._create_table_ddl()
        assert "CREATE TABLE LOCUS_APP.my_store" in ddl
        # PK constraint name uses the table name (not the qualified one).
        assert "CONSTRAINT pk_my_store" in ddl

    def test_namespace_index_ddl(self) -> None:
        store = _make_store()
        ddl = store._ns_index_ddl()
        assert ddl == "CREATE INDEX idx_locus_store_ns ON locus_store (namespace)"


# ---------------------------------------------------------------------------
# MERGE SQL shape
# ---------------------------------------------------------------------------


class TestMergeSql:
    def test_merge_without_embedding_shape(self) -> None:
        sql = _make_store()._merge_sql(with_embedding=False)
        # MERGE grammar landmarks
        assert "MERGE INTO locus_store t" in sql
        assert "USING (SELECT :namespace AS namespace, :key AS key FROM dual) s" in sql
        assert "ON (t.namespace = s.namespace AND t.key = s.key)" in sql
        assert "WHEN MATCHED THEN UPDATE SET value = :value" in sql
        assert (
            "WHEN NOT MATCHED THEN INSERT (namespace, key, value) VALUES (:namespace, :key, :value)"
            in sql
        )
        # NO embedding bind in the text-only variant.
        assert "embedding" not in sql

    def test_merge_with_embedding_shape(self) -> None:
        sql = _make_store(dimension=1024)._merge_sql(with_embedding=True)
        assert "TO_VECTOR(:embedding)" in sql
        # Both the UPDATE branch and the INSERT branch write the column.
        assert "embedding = TO_VECTOR(:embedding)" in sql
        assert "(namespace, key, value, embedding)" in sql

    def test_merge_uses_schema_qualified_table(self) -> None:
        sql = _make_store(schema_name="APP", table_name="store_x")._merge_sql(with_embedding=False)
        assert "MERGE INTO APP.store_x t" in sql


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_text_mode_no_semantic_search(self) -> None:
        store = _make_store(dimension=None)
        caps = store.capabilities
        assert caps.semantic_search is False
        assert caps.embedding_dimension is None
        assert caps.search is True
        assert caps.list_namespaces is True

    def test_vector_mode_advertises_semantic_search(self) -> None:
        store = _make_store(dimension=1024)
        caps = store.capabilities
        assert caps.semantic_search is True
        assert caps.embedding_dimension == 1024


# ---------------------------------------------------------------------------
# put / get / delete — binds + commit + SQL shape
# ---------------------------------------------------------------------------


class TestPutGetDelete:
    @pytest.mark.asyncio
    async def test_put_flattens_namespace_and_pins_clob(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # First execute is the user_tables existence check (returns 0 → not
        # exists), then CREATE TABLE, CREATE INDEX, then the MERGE.
        cursor = _StubCursor(fetchone=(0,))
        _install_oracledb_stub(monkeypatch, cursor)

        store = _make_store()
        await store.put(("memory", "user-42"), "theme", {"value": "dark"})

        # MERGE call is the last one
        merge_sql, merge_params = cursor.execute_calls[-1]
        assert "MERGE INTO" in merge_sql
        assert merge_params["namespace"] == "memory/user-42"
        assert merge_params["key"] == "theme"
        # Value is JSON-encoded.
        assert '"dark"' in merge_params["value"]
        # CLOB binding hint was applied.
        assert cursor.input_sizes.get("value") == "CLOB-SENTINEL"

    @pytest.mark.asyncio
    async def test_put_with_metadata_wraps_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cursor = _StubCursor(fetchone=(1,))  # table already exists
        _install_oracledb_stub(monkeypatch, cursor)

        store = _make_store()
        await store.put(("u",), "k", {"a": 1}, metadata={"src": "test"})

        _, params = cursor.execute_calls[-1]
        import json

        payload = json.loads(params["value"])
        assert payload == {"value": {"a": 1}, "_meta": {"src": "test"}}

    @pytest.mark.asyncio
    async def test_get_returns_none_when_row_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # table-exists check returns 1 → no DDL; then SELECT returns None.
        cursor = _StubCursor(fetchone=None)
        _install_oracledb_stub(monkeypatch, cursor)

        store = _make_store()
        # First call to `_ensure_table` consumes fetchone; we need a
        # second fetchone for the actual SELECT. Easier: pre-seed the
        # table as existing by toggling _initialized.
        store._initialized = True
        result = await store.get(("u",), "missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_decodes_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cursor = _StubCursor(fetchone=('{"theme":"dark"}',))
        _install_oracledb_stub(monkeypatch, cursor)
        store = _make_store()
        store._initialized = True

        result = await store.get(("u",), "theme")
        assert result == {"theme": "dark"}

        sql, params = cursor.execute_calls[-1]
        assert "SELECT value FROM locus_store" in sql
        assert params == {"namespace": "u", "key": "theme"}

    @pytest.mark.asyncio
    async def test_delete_returns_true_on_rowcount(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cursor = _StubCursor()
        _install_oracledb_stub(monkeypatch, cursor)
        store = _make_store()
        store._initialized = True

        ok = await store.delete(("u",), "theme")
        assert ok is True
        sql, params = cursor.execute_calls[-1]
        assert sql.startswith("DELETE FROM locus_store")
        assert params == {"namespace": "u", "key": "theme"}


# ---------------------------------------------------------------------------
# list_keys / list_namespaces — prefix LIKE pattern
# ---------------------------------------------------------------------------


class TestListing:
    @pytest.mark.asyncio
    async def test_list_keys_passes_flat_namespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cursor = _StubCursor(fetchall=[("k1",), ("k2",)])
        _install_oracledb_stub(monkeypatch, cursor)
        store = _make_store()
        store._initialized = True

        keys = await store.list_keys(("memory", "u1"), limit=50)
        assert keys == ["k1", "k2"]

        sql, params = cursor.execute_calls[-1]
        assert "SELECT key FROM locus_store" in sql
        assert "FETCH FIRST :lim ROWS ONLY" in sql
        assert params == {"namespace": "memory/u1", "lim": 50}

    @pytest.mark.asyncio
    async def test_list_namespaces_builds_like_pattern(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cursor = _StubCursor(fetchall=[("memory/u1",), ("memory/u2",)])
        _install_oracledb_stub(monkeypatch, cursor)
        store = _make_store()
        store._initialized = True

        result = await store.list_namespaces(prefix=("memory",), limit=10)
        assert result == [("memory", "u1"), ("memory", "u2")]

        sql, params = cursor.execute_calls[-1]
        assert "SELECT DISTINCT namespace FROM locus_store" in sql
        # The prefix-match clause uses bind variables, not string interp.
        assert ":pfx" in sql
        assert ":pfx_sep" in sql
        assert params["pfx"] == "memory"
        assert params["pfx_sep"] == "memory/%"
        assert params["lim"] == 10

    @pytest.mark.asyncio
    async def test_list_namespaces_no_prefix_omits_where(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cursor = _StubCursor(fetchall=[])
        _install_oracledb_stub(monkeypatch, cursor)
        store = _make_store()
        store._initialized = True

        await store.list_namespaces(prefix=None, limit=5)
        sql, params = cursor.execute_calls[-1]
        # No WHERE clause means no LIKE pattern bind.
        assert "WHERE" not in sql
        assert "pfx" not in params


# ---------------------------------------------------------------------------
# search — LIKE pattern + namespace filter
# ---------------------------------------------------------------------------


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_with_query_builds_like(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cursor = _StubCursor(fetchall=[])
        _install_oracledb_stub(monkeypatch, cursor)
        store = _make_store()
        store._initialized = True

        await store.search(("memory", "u1"), query="dark", limit=5)
        sql, params = cursor.execute_calls[-1]
        assert "LOWER(value) LIKE LOWER(:pattern)" in sql
        assert params["pattern"] == "%dark%"
        assert params["namespace"] == "memory/u1"
        assert params["lim"] == 5

    @pytest.mark.asyncio
    async def test_search_without_query_no_pattern(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cursor = _StubCursor(fetchall=[])
        _install_oracledb_stub(monkeypatch, cursor)
        store = _make_store()
        store._initialized = True

        await store.search(("u",), query=None, limit=3)
        sql, params = cursor.execute_calls[-1]
        assert "LIKE" not in sql
        assert "pattern" not in params


# ---------------------------------------------------------------------------
# Async-prefix aliases — langgraph-oracledb parity
# ---------------------------------------------------------------------------


class TestAsyncAliases:
    def test_aliases_exist(self) -> None:
        store = _make_store()
        # The methods are class-level aliases, so identity-equal to the
        # canonical implementations.
        assert OracleStore.aput is OracleStore.put
        assert OracleStore.aget is OracleStore.get
        assert OracleStore.adelete is OracleStore.delete
        assert OracleStore.asearch is OracleStore.search
        assert OracleStore.alist_namespaces is OracleStore.list_namespaces
        # Instance-level bound-method lookup also works.
        assert callable(store.aput)
        assert callable(store.asearch)


# ---------------------------------------------------------------------------
# Vector path — dimension guard + embedding-length check
# ---------------------------------------------------------------------------


class TestVectorPath:
    @pytest.mark.asyncio
    async def test_put_with_embedding_dimension_mismatch_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cursor = _StubCursor(fetchone=(1,))
        _install_oracledb_stub(monkeypatch, cursor)
        store = _make_store(dimension=8)
        store._initialized = True

        with pytest.raises(ValueError, match="dims"):
            await store.put_with_embedding(("u",), "k", {"a": 1}, embedding=[0.1, 0.2])

    @pytest.mark.asyncio
    async def test_put_with_embedding_text_mode_raises_capability_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from locus.memory.store import StoreCapabilityError

        cursor = _StubCursor()
        _install_oracledb_stub(monkeypatch, cursor)
        store = _make_store(dimension=None)

        with pytest.raises(StoreCapabilityError):
            await store.put_with_embedding(("u",), "k", {"a": 1}, embedding=[0.1])

    def test_vector_to_string_format(self) -> None:
        s = OracleStore._vector_to_string([0.1, 0.2, 0.3])
        assert s.startswith("[")
        assert s.endswith("]")
        assert "0.1" in s
