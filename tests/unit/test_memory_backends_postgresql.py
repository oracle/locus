# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for ``locus.memory.backends.postgresql`` (PostgreSQLBackend).

Stubs ``asyncpg`` so we never need a real Postgres. The backend's
``async with pool.acquire() as conn`` pattern means the stub conn
needs to record the SQL it sees and return canned rows. Coverage:

- config defaults + DSN vs host/port/etc construction paths
- ``_get_pool`` lazy init for both DSN and component-args paths
- ``_ensure_table`` runs DDL once + idempotent on second call
- save / load / delete / exists / list_threads / get_metadata
- query_by_metadata / search_data JSONB paths
- count + vacuum row-count parsing
- close + __repr__
- missing-asyncpg-package import error
"""

from __future__ import annotations

import json
import sys
import types
from contextlib import asynccontextmanager
from typing import Any

import pytest

from locus.memory.backends.postgresql import PostgreSQLBackend, PostgreSQLConfig


# ---------------------------------------------------------------------------
# Stub asyncpg
# ---------------------------------------------------------------------------


class _StubConn:
    """Records ``execute`` calls and returns canned rows."""

    def __init__(
        self,
        *,
        fetchrow: Any | None = None,
        fetch: list[Any] | None = None,
        execute_results: list[str] | None = None,
    ) -> None:
        self.fetchrow_value = fetchrow
        self.fetch_value = fetch or []
        self.execute_results = execute_results or []
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, sql: str, *args: Any) -> str:
        self.execute_calls.append((sql, args))
        return self.execute_results.pop(0) if self.execute_results else "INSERT 0 1"

    async def fetchrow(self, sql: str, *args: Any) -> Any:
        return self.fetchrow_value

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        return self.fetch_value


class _StubPool:
    def __init__(self, conn: _StubConn) -> None:
        self._conn = conn
        self.closed = False

    @asynccontextmanager
    async def acquire(self) -> Any:
        yield self._conn

    async def close(self) -> None:
        self.closed = True


def _stub_asyncpg(monkeypatch: pytest.MonkeyPatch, conn: _StubConn) -> dict[str, Any]:
    """Install a fake ``asyncpg`` module that returns ``_StubPool``."""
    probes: dict[str, Any] = {}

    pool = _StubPool(conn)

    async def fake_create_pool(*args: Any, **kwargs: Any) -> _StubPool:
        probes["create_pool_args"] = args
        probes["create_pool_kwargs"] = kwargs
        return pool

    fake_module = types.ModuleType("asyncpg")
    fake_module.create_pool = fake_create_pool  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "asyncpg", fake_module)
    probes["pool"] = pool
    return probes


# ---------------------------------------------------------------------------
# Config + construction
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default_values(self) -> None:
        cfg = PostgreSQLConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 5432
        assert cfg.schema_name == "public"
        assert cfg.table_name == "checkpoints"

    def test_invalid_table_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid table_name"):
            PostgreSQLConfig(table_name="bad table")

    def test_invalid_schema_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid schema_name"):
            PostgreSQLConfig(schema_name="bad schema")

    def test_constructor_dsn_path(self) -> None:
        backend = PostgreSQLBackend(dsn="postgres://u:p@h/db")
        assert backend.config.dsn == "postgres://u:p@h/db"

    def test_constructor_component_args(self) -> None:
        backend = PostgreSQLBackend(
            host="db.host",
            port=6543,
            database="mydb",
            user="u",
            password="p",  # noqa: S106
        )
        assert backend.config.host == "db.host"
        assert backend.config.port == 6543
        assert backend.config.database == "mydb"
        assert backend.config.password.get_secret_value() == "p"


# ---------------------------------------------------------------------------
# _get_pool — lazy init + asyncpg missing
# ---------------------------------------------------------------------------


class TestGetPool:
    @pytest.mark.asyncio
    async def test_dsn_path_invokes_create_pool_with_dsn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        probes = _stub_asyncpg(monkeypatch, _StubConn())
        backend = PostgreSQLBackend(dsn="postgres://u:p@h/db")
        await backend._get_pool()
        assert probes["create_pool_args"][0] == "postgres://u:p@h/db"

    @pytest.mark.asyncio
    async def test_component_path_invokes_create_pool_with_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        probes = _stub_asyncpg(monkeypatch, _StubConn())
        backend = PostgreSQLBackend(
            host="h",
            port=1234,
            database="d",
            user="u",
            password="p",  # noqa: S106
        )
        await backend._get_pool()
        kw = probes["create_pool_kwargs"]
        assert kw["host"] == "h"
        assert kw["port"] == 1234
        assert kw["database"] == "d"
        assert kw["user"] == "u"
        assert kw["password"] == "p"  # noqa: S105

    @pytest.mark.asyncio
    async def test_pool_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_asyncpg(monkeypatch, _StubConn())
        backend = PostgreSQLBackend()
        p1 = await backend._get_pool()
        p2 = await backend._get_pool()
        assert p1 is p2

    @pytest.mark.asyncio
    async def test_missing_asyncpg_raises_clear_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Block import.
        monkeypatch.setitem(sys.modules, "asyncpg", None)
        backend = PostgreSQLBackend()
        with pytest.raises(ImportError, match="PostgreSQLBackend requires the 'asyncpg' package"):
            await backend._get_pool()


# ---------------------------------------------------------------------------
# _ensure_table — runs DDL once
# ---------------------------------------------------------------------------


class TestEnsureTable:
    @pytest.mark.asyncio
    async def test_runs_ddl_first_call_and_caches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = _StubConn()
        _stub_asyncpg(monkeypatch, conn)
        backend = PostgreSQLBackend()
        await backend._ensure_table()
        first_count = len(conn.execute_calls)
        # Second call short-circuits — no further DDL.
        await backend._ensure_table()
        assert len(conn.execute_calls) == first_count

    @pytest.mark.asyncio
    async def test_full_table_name_includes_schema(self) -> None:
        backend = PostgreSQLBackend()
        assert backend._full_table_name == "public.checkpoints"


# ---------------------------------------------------------------------------
# save / load / delete / exists
# ---------------------------------------------------------------------------


class TestCrudOperations:
    @pytest.mark.asyncio
    async def test_save_returns_generated_checkpoint_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        conn = _StubConn()
        _stub_asyncpg(monkeypatch, conn)
        backend = PostgreSQLBackend()
        cid = await backend.save("thread-1", {"x": 1})
        assert isinstance(cid, str)
        assert cid

    @pytest.mark.asyncio
    async def test_save_uses_explicit_checkpoint_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = _StubConn()
        _stub_asyncpg(monkeypatch, conn)
        backend = PostgreSQLBackend()
        cid = await backend.save("thread-1", {"x": 1}, checkpoint_id="cp-99")
        assert cid == "cp-99"

    @pytest.mark.asyncio
    async def test_save_passes_metadata_as_json_arg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = _StubConn()
        _stub_asyncpg(monkeypatch, conn)
        backend = PostgreSQLBackend()
        await backend.save("t", {"x": 1}, metadata={"tier": "gold"})
        # Last execute call has the INSERT — last positional arg is metadata.
        sql, args = conn.execute_calls[-1]
        assert "INSERT INTO" in sql
        assert json.loads(args[-1]) == {"tier": "gold"}

    @pytest.mark.asyncio
    async def test_load_parses_jsonb_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = _StubConn(fetchrow={"data": json.dumps({"x": 1})})
        _stub_asyncpg(monkeypatch, conn)
        backend = PostgreSQLBackend()
        result = await backend.load("t")
        assert result == {"x": 1}

    @pytest.mark.asyncio
    async def test_load_returns_none_for_missing_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_asyncpg(monkeypatch, _StubConn(fetchrow=None))
        backend = PostgreSQLBackend()
        assert await backend.load("missing") is None

    @pytest.mark.asyncio
    async def test_delete_returns_true_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 4 DDL statements (CREATE SCHEMA + CREATE TABLE + 2 CREATE INDEX)
        # then the DELETE.
        conn = _StubConn(execute_results=["", "", "", "", "DELETE 1"])
        _stub_asyncpg(monkeypatch, conn)
        backend = PostgreSQLBackend()
        assert await backend.delete("t") is True

    @pytest.mark.asyncio
    async def test_delete_returns_false_when_no_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = _StubConn(execute_results=["", "", "", "", "DELETE 0"])
        _stub_asyncpg(monkeypatch, conn)
        backend = PostgreSQLBackend()
        assert await backend.delete("t") is False

    @pytest.mark.asyncio
    async def test_exists_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_asyncpg(monkeypatch, _StubConn(fetchrow={"?column?": 1}))
        backend = PostgreSQLBackend()
        assert await backend.exists("t") is True

    @pytest.mark.asyncio
    async def test_exists_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_asyncpg(monkeypatch, _StubConn(fetchrow=None))
        backend = PostgreSQLBackend()
        assert await backend.exists("t") is False


# ---------------------------------------------------------------------------
# Query operations
# ---------------------------------------------------------------------------


class TestQueries:
    @pytest.mark.asyncio
    async def test_list_threads_returns_thread_ids(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = [{"thread_id": "t1"}, {"thread_id": "t2"}]
        _stub_asyncpg(monkeypatch, _StubConn(fetch=rows))
        backend = PostgreSQLBackend()
        result = await backend.list_threads()
        assert result == ["t1", "t2"]

    @pytest.mark.asyncio
    async def test_get_metadata_parses_iso_timestamps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from datetime import UTC, datetime

        ts = datetime(2026, 1, 1, tzinfo=UTC)
        row = {
            "checkpoint_id": "cp-1",
            "created_at": ts,
            "updated_at": ts,
            "metadata": json.dumps({"tier": "gold"}),
        }
        _stub_asyncpg(monkeypatch, _StubConn(fetchrow=row))
        backend = PostgreSQLBackend()
        meta = await backend.get_metadata("t")
        assert meta is not None
        assert meta["checkpoint_id"] == "cp-1"
        assert meta["metadata"] == {"tier": "gold"}
        assert "T" in meta["created_at"]  # ISO-formatted

    @pytest.mark.asyncio
    async def test_get_metadata_missing_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_asyncpg(monkeypatch, _StubConn(fetchrow=None))
        backend = PostgreSQLBackend()
        assert await backend.get_metadata("missing") is None

    @pytest.mark.asyncio
    async def test_get_metadata_handles_null_metadata_column(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from datetime import UTC, datetime

        ts = datetime(2026, 1, 1, tzinfo=UTC)
        row = {
            "checkpoint_id": "cp-1",
            "created_at": ts,
            "updated_at": ts,
            "metadata": None,
        }
        _stub_asyncpg(monkeypatch, _StubConn(fetchrow=row))
        backend = PostgreSQLBackend()
        meta = await backend.get_metadata("t")
        assert meta is not None
        assert meta["metadata"] == {}

    @pytest.mark.asyncio
    async def test_query_by_metadata_returns_decoded_rows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from datetime import UTC, datetime

        ts = datetime(2026, 1, 1, tzinfo=UTC)
        rows = [
            {
                "thread_id": "t1",
                "data": json.dumps({"x": 1}),
                "updated_at": ts,
            }
        ]
        _stub_asyncpg(monkeypatch, _StubConn(fetch=rows))
        backend = PostgreSQLBackend()
        result = await backend.query_by_metadata("tier", "gold")
        assert result[0]["thread_id"] == "t1"
        assert result[0]["data"] == {"x": 1}

    @pytest.mark.asyncio
    async def test_search_data_uses_data_jsonb_operator(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from datetime import UTC, datetime

        ts = datetime(2026, 1, 1, tzinfo=UTC)
        rows = [
            {
                "thread_id": "t1",
                "data": json.dumps({"agent_id": "a-1"}),
                "updated_at": ts,
            }
        ]
        _stub_asyncpg(monkeypatch, _StubConn(fetch=rows))
        backend = PostgreSQLBackend()
        result = await backend.search_data("agent_id", "a-1")
        assert result[0]["data"] == {"agent_id": "a-1"}

    @pytest.mark.asyncio
    async def test_count_returns_row_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_asyncpg(monkeypatch, _StubConn(fetchrow={"cnt": 7}))
        backend = PostgreSQLBackend()
        assert await backend.count() == 7

    @pytest.mark.asyncio
    async def test_count_returns_zero_when_no_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_asyncpg(monkeypatch, _StubConn(fetchrow=None))
        backend = PostgreSQLBackend()
        assert await backend.count() == 0


# ---------------------------------------------------------------------------
# Vacuum
# ---------------------------------------------------------------------------


class TestVacuum:
    @pytest.mark.asyncio
    async def test_vacuum_returns_parsed_row_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 4 DDL statements then the DELETE.
        conn = _StubConn(execute_results=["", "", "", "", "DELETE 5"])
        _stub_asyncpg(monkeypatch, conn)
        backend = PostgreSQLBackend()
        assert await backend.vacuum(older_than_days=7) == 5

    @pytest.mark.asyncio
    async def test_vacuum_unparseable_result_returns_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        conn = _StubConn(execute_results=["", "", "", "", ""])
        _stub_asyncpg(monkeypatch, conn)
        backend = PostgreSQLBackend()
        assert await backend.vacuum() == 0


# ---------------------------------------------------------------------------
# close + __repr__
# ---------------------------------------------------------------------------


class TestClose:
    @pytest.mark.asyncio
    async def test_close_calls_pool_close(self, monkeypatch: pytest.MonkeyPatch) -> None:
        probes = _stub_asyncpg(monkeypatch, _StubConn())
        backend = PostgreSQLBackend()
        await backend._get_pool()
        await backend.close()
        assert probes["pool"].closed is True
        # After close, the pool is None and a fresh ``_get_pool`` would
        # reinitialise it.
        assert backend._pool is None

    @pytest.mark.asyncio
    async def test_close_without_open_pool_is_noop(self) -> None:
        backend = PostgreSQLBackend()
        await backend.close()  # No pool ever opened — must not raise.


class TestRepr:
    def test_repr_with_dsn_redacts(self) -> None:
        backend = PostgreSQLBackend(dsn="postgres://user:secret@host/db")
        rep = repr(backend)
        assert "secret" not in rep
        assert "PostgreSQLBackend" in rep

    def test_repr_with_components_shows_host_db(self) -> None:
        backend = PostgreSQLBackend(host="myhost", database="mydb")
        rep = repr(backend)
        assert "myhost" in rep
        assert "mydb" in rep
