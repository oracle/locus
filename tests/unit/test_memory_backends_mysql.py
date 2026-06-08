# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Unit tests for ``tulip.memory.backends.mysql`` (MySQLBackend).

Stubs ``mysql.connector.aio`` so we never need a real MySQL instance.
Coverage mirrors the PostgreSQL backend tests:

- config defaults + DSN vs host/port/etc construction paths
- ``_get_pool`` lazy init for both DSN and component-args paths
- ``_ensure_table`` runs DDL once + idempotent on second call
- save / load / delete / exists / list_threads / get_metadata
- query_by_metadata / search_data JSON paths
- count + vacuum row-count parsing
- close + __repr__
- missing-mysql-connector-package import error
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from datetime import UTC, datetime
from typing import Any

import pytest

from tulip.memory.backends.mysql import (
    MySQLBackend,
    MySQLConfig,
    _decode_json,
    _MySQLConnectionPool,
    _parse_dsn,
)


# ---------------------------------------------------------------------------
# Stub mysql.connector.aio
# ---------------------------------------------------------------------------


class _StubCursor:
    """Records ``execute`` calls and returns canned rows."""

    def __init__(
        self,
        *,
        fetchone: Any | None = None,
        fetchall: list[Any] | None = None,
        rowcount: int = 1,
    ) -> None:
        self.fetchone_value = fetchone
        self.fetchall_value = fetchall or []
        self.rowcount = rowcount
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def __aenter__(self) -> _StubCursor:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    async def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None:
        self.execute_calls.append((sql, params or ()))

    async def fetchone(self) -> Any:
        return self.fetchone_value

    async def fetchall(self) -> list[Any]:
        return self.fetchall_value


class _BrokenCursor(_StubCursor):
    async def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None:
        raise RuntimeError("connection is stale")


class _StubConn:
    """Connection stub with a single reusable cursor."""

    def __init__(self, cursor: _StubCursor | None = None) -> None:
        self.cursor_obj = cursor or _StubCursor()
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    async def cursor(self) -> _StubCursor:
        return self.cursor_obj

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1

    async def close(self) -> None:
        self.closed = True


def _stub_mysql_connector(monkeypatch: pytest.MonkeyPatch, conn: _StubConn) -> dict[str, Any]:
    """Install fake ``mysql.connector.aio`` modules returning ``conn``."""
    probes: dict[str, Any] = {}

    async def fake_connect(**kwargs: Any) -> _StubConn:
        probes["connect_kwargs"] = kwargs
        probes["connect_count"] = probes.get("connect_count", 0) + 1
        return conn

    mysql_mod = types.ModuleType("mysql")
    connector_mod = types.ModuleType("mysql.connector")
    aio_mod = types.ModuleType("mysql.connector.aio")
    aio_mod.connect = fake_connect  # type: ignore[attr-defined]
    connector_mod.aio = aio_mod  # type: ignore[attr-defined]
    mysql_mod.connector = connector_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "mysql", mysql_mod)
    monkeypatch.setitem(sys.modules, "mysql.connector", connector_mod)
    monkeypatch.setitem(sys.modules, "mysql.connector.aio", aio_mod)
    probes["conn"] = conn
    return probes


# ---------------------------------------------------------------------------
# Config + construction
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default_values(self) -> None:
        cfg = MySQLConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 3306
        assert cfg.database == "tulip"
        assert cfg.table_name == "checkpoints"

    def test_invalid_table_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid table_name"):
            MySQLConfig(table_name="bad table")

    def test_invalid_database_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid database"):
            MySQLConfig(database="bad database")

    def test_constructor_dsn_path(self) -> None:
        backend = MySQLBackend(dsn="mysql://u:p@h:3306/db")
        assert backend.config.dsn == "mysql://u:p@h:3306/db"

    def test_constructor_component_args(self) -> None:
        backend = MySQLBackend(
            host="db.host",
            port=3307,
            database="mydb",
            user="u",
            password="p",  # noqa: S106
        )
        assert backend.config.host == "db.host"
        assert backend.config.port == 3307
        assert backend.config.database == "mydb"
        assert backend.config.password.get_secret_value() == "p"


class TestHelpers:
    def test_decode_json_handles_none_bytes_and_native_values(self) -> None:
        assert _decode_json(None) is None
        assert _decode_json(b'{"x": 1}') == {"x": 1}
        native = {"already": "decoded"}
        assert _decode_json(native) is native

    def test_parse_dsn_rejects_non_mysql_scheme(self) -> None:
        with pytest.raises(ValueError, match="mysql://"):
            _parse_dsn("postgresql://u:p@h/db")

    def test_parse_dsn_accepts_minimal_mysql_url_and_query_options(self) -> None:
        parsed = _parse_dsn("mysql+connector:///?ssl_disabled=true&charset=utf8mb4")
        assert parsed == {"ssl_disabled": "true", "charset": "utf8mb4"}


class TestConnectionPool:
    def test_rejects_invalid_pool_sizes(self) -> None:
        with pytest.raises(ValueError, match="min_pool_size"):
            _MySQLConnectionPool(MySQLConfig(min_pool_size=-1))
        with pytest.raises(ValueError, match="max_pool_size"):
            _MySQLConnectionPool(MySQLConfig(max_pool_size=0))
        with pytest.raises(ValueError, match="min_pool_size must be <="):
            _MySQLConnectionPool(MySQLConfig(min_pool_size=2, max_pool_size=1))

    @pytest.mark.asyncio
    async def test_acquire_rejects_closed_pool(self) -> None:
        pool = _MySQLConnectionPool(MySQLConfig(min_pool_size=0))
        await pool.close()
        with pytest.raises(RuntimeError, match="closed"):
            async with pool.acquire():
                pass

    @pytest.mark.asyncio
    async def test_acquire_closes_connection_if_pool_closes_during_context(self) -> None:
        conn = _StubConn()
        pool = _MySQLConnectionPool(MySQLConfig(min_pool_size=0))
        await pool._available.put(conn)
        pool._created = 1

        async with pool.acquire():
            pool._closed = True

        assert conn.closed is True
        assert pool._created == 0

    @pytest.mark.asyncio
    async def test_acquire_discards_stale_idle_connection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stale = _StubConn(_BrokenCursor())
        replacement = _StubConn()
        _stub_mysql_connector(monkeypatch, replacement)
        pool = _MySQLConnectionPool(MySQLConfig(min_pool_size=0, max_pool_size=1))
        pool._created = 1
        await pool._available.put(stale)

        conn = await pool._acquire()

        assert conn is replacement
        assert stale.closed is True
        assert pool._created == 1

    @pytest.mark.asyncio
    async def test_acquire_discards_connection_after_use_error(self) -> None:
        conn = _StubConn()
        pool = _MySQLConnectionPool(MySQLConfig(min_pool_size=0))
        pool._created = 1
        await pool._available.put(conn)

        with pytest.raises(RuntimeError, match="query failed"):
            async with pool.acquire():
                raise RuntimeError("query failed")

        assert conn.closed is True
        assert pool._created == 0
        assert pool._available.empty()

    @pytest.mark.asyncio
    async def test_acquire_waits_when_pool_is_at_capacity(self) -> None:
        conn = _StubConn()
        pool = _MySQLConnectionPool(MySQLConfig(min_pool_size=0, max_pool_size=1))
        pool._created = 1

        task = asyncio.create_task(pool._acquire())
        await asyncio.sleep(0)
        assert not task.done()

        await pool._available.put(conn)
        assert await task is conn

    @pytest.mark.asyncio
    async def test_release_rolls_back_open_transaction(self) -> None:
        # autocommit=False: a returned connection must be rolled back so it is
        # not pooled "idle in transaction" (which would hold a shared MDL and
        # block later DDL such as DROP/ALTER). See ``_release``.
        conn = _StubConn()
        pool = _MySQLConnectionPool(MySQLConfig(min_pool_size=0))
        pool._created = 1
        await pool._available.put(conn)

        async with pool.acquire():
            pass

        assert conn.rollbacks == 1
        assert conn.closed is False
        assert pool._available.qsize() == 1

    @pytest.mark.asyncio
    async def test_release_discards_when_rollback_fails(self) -> None:
        class _RollbackBrokenConn(_StubConn):
            async def rollback(self) -> None:
                raise RuntimeError("connection went away")

        conn = _RollbackBrokenConn()
        pool = _MySQLConnectionPool(MySQLConfig(min_pool_size=0))
        pool._created = 1
        await pool._available.put(conn)

        async with pool.acquire():
            pass

        assert conn.closed is True
        assert pool._created == 0
        assert pool._available.empty()


# ---------------------------------------------------------------------------
# _get_pool — lazy init + mysql connector missing
# ---------------------------------------------------------------------------


class TestGetPool:
    @pytest.mark.asyncio
    async def test_dsn_path_invokes_connect_with_parsed_dsn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        probes = _stub_mysql_connector(monkeypatch, _StubConn())
        backend = MySQLBackend(dsn="mysql://u:p@h:3307/db")
        await backend._get_pool()
        kw = probes["connect_kwargs"]
        assert kw["host"] == "h"
        assert kw["port"] == 3307
        assert kw["database"] == "db"
        assert kw["user"] == "u"
        assert kw["password"] == "p"  # noqa: S105

    @pytest.mark.asyncio
    async def test_component_path_invokes_connect_with_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        probes = _stub_mysql_connector(monkeypatch, _StubConn())
        backend = MySQLBackend(
            host="h",
            port=1234,
            database="d",
            user="u",
            password="p",  # noqa: S106
        )
        await backend._get_pool()
        kw = probes["connect_kwargs"]
        assert kw["host"] == "h"
        assert kw["port"] == 1234
        assert kw["database"] == "d"
        assert kw["user"] == "u"
        assert kw["password"] == "p"  # noqa: S105

    @pytest.mark.asyncio
    async def test_pool_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_mysql_connector(monkeypatch, _StubConn())
        backend = MySQLBackend()
        p1 = await backend._get_pool()
        p2 = await backend._get_pool()
        assert p1 is p2

    @pytest.mark.asyncio
    async def test_missing_mysql_connector_raises_clear_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "mysql.connector.aio", None)
        backend = MySQLBackend()
        with pytest.raises(ImportError, match="MySQLBackend requires"):
            await backend._get_pool()


# ---------------------------------------------------------------------------
# _ensure_table — runs DDL once
# ---------------------------------------------------------------------------


class TestEnsureTable:
    @pytest.mark.asyncio
    async def test_runs_ddl_first_call_and_caches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cur = _StubCursor(fetchone=(0,))
        conn = _StubConn(cur)
        _stub_mysql_connector(monkeypatch, conn)
        backend = MySQLBackend()
        await backend._ensure_table()
        first_count = len(cur.execute_calls)
        await backend._ensure_table()
        assert len(cur.execute_calls) == first_count
        assert conn.commits == 1
        ddl_calls = [
            (sql, params)
            for sql, params in cur.execute_calls
            if "CREATE TABLE IF NOT EXISTS" in sql
        ]
        assert len(ddl_calls) == 1
        sql, params = ddl_calls[0]
        assert "KEY `idx_checkpoints_updated` (updated_at DESC)" in sql
        assert "information_schema.statistics" not in sql
        assert params == ()

    @pytest.mark.asyncio
    async def test_table_name_is_quoted(self) -> None:
        backend = MySQLBackend()
        assert backend._quoted_table_name == "`checkpoints`"


# ---------------------------------------------------------------------------
# save / load / delete / exists
# ---------------------------------------------------------------------------


class TestCrudOperations:
    @pytest.mark.asyncio
    async def test_save_returns_generated_checkpoint_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_mysql_connector(monkeypatch, _StubConn(_StubCursor(fetchone=(1,))))
        backend = MySQLBackend()
        cid = await backend.save("thread-1", {"x": 1})
        assert isinstance(cid, str)
        assert cid

    @pytest.mark.asyncio
    async def test_save_uses_explicit_checkpoint_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_mysql_connector(monkeypatch, _StubConn(_StubCursor(fetchone=(1,))))
        backend = MySQLBackend()
        cid = await backend.save("thread-1", {"x": 1}, checkpoint_id="cp-99")
        assert cid == "cp-99"

    @pytest.mark.asyncio
    async def test_save_passes_metadata_as_json_arg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cur = _StubCursor(fetchone=(1,))
        _stub_mysql_connector(monkeypatch, _StubConn(cur))
        backend = MySQLBackend()
        await backend.save("t", {"x": 1}, metadata={"tier": "gold"})
        sql, args = cur.execute_calls[-1]
        assert "INSERT INTO" in sql
        assert json.loads(args[-1]) == {"tier": "gold"}

    @pytest.mark.asyncio
    async def test_load_parses_json_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cur = _StubCursor(fetchone=(json.dumps({"x": 1}),))
        _stub_mysql_connector(monkeypatch, _StubConn(cur))
        backend = MySQLBackend()
        result = await backend.load("t")
        assert result == {"x": 1}

    @pytest.mark.asyncio
    async def test_load_returns_none_for_missing_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_mysql_connector(monkeypatch, _StubConn(_StubCursor(fetchone=None)))
        backend = MySQLBackend()
        assert await backend.load("missing") is None

    @pytest.mark.asyncio
    async def test_delete_returns_true_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cur = _StubCursor(fetchone=(1,), rowcount=1)
        _stub_mysql_connector(monkeypatch, _StubConn(cur))
        backend = MySQLBackend()
        assert await backend.delete("t") is True

    @pytest.mark.asyncio
    async def test_delete_returns_false_when_no_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cur = _StubCursor(fetchone=(1,), rowcount=0)
        _stub_mysql_connector(monkeypatch, _StubConn(cur))
        backend = MySQLBackend()
        assert await backend.delete("t") is False

    @pytest.mark.asyncio
    async def test_exists_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_mysql_connector(monkeypatch, _StubConn(_StubCursor(fetchone=(1,))))
        backend = MySQLBackend()
        assert await backend.exists("t") is True

    @pytest.mark.asyncio
    async def test_exists_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_mysql_connector(monkeypatch, _StubConn(_StubCursor(fetchone=None)))
        backend = MySQLBackend()
        assert await backend.exists("t") is False


# ---------------------------------------------------------------------------
# Query operations
# ---------------------------------------------------------------------------


class TestQueries:
    @pytest.mark.asyncio
    async def test_list_threads_returns_thread_ids(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = [("t1",), ("t2",)]
        _stub_mysql_connector(monkeypatch, _StubConn(_StubCursor(fetchone=(1,), fetchall=rows)))
        backend = MySQLBackend()
        result = await backend.list_threads()
        assert result == ["t1", "t2"]

    @pytest.mark.asyncio
    async def test_get_metadata_parses_iso_timestamps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        row = ("cp-1", ts, ts, json.dumps({"tier": "gold"}))
        _stub_mysql_connector(monkeypatch, _StubConn(_StubCursor(fetchone=row)))
        backend = MySQLBackend()
        meta = await backend.get_metadata("t")
        assert meta is not None
        assert meta["checkpoint_id"] == "cp-1"
        assert meta["metadata"] == {"tier": "gold"}
        assert "T" in meta["created_at"]

    @pytest.mark.asyncio
    async def test_get_metadata_missing_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_mysql_connector(monkeypatch, _StubConn(_StubCursor(fetchone=None)))
        backend = MySQLBackend()
        assert await backend.get_metadata("missing") is None

    @pytest.mark.asyncio
    async def test_get_metadata_handles_null_metadata_column(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        row = ("cp-1", ts, ts, None)
        _stub_mysql_connector(monkeypatch, _StubConn(_StubCursor(fetchone=row)))
        backend = MySQLBackend()
        meta = await backend.get_metadata("t")
        assert meta is not None
        assert meta["metadata"] == {}

    @pytest.mark.asyncio
    async def test_query_by_metadata_returns_decoded_rows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        rows = [("t1", json.dumps({"x": 1}), ts)]
        cur = _StubCursor(fetchone=(1,), fetchall=rows)
        _stub_mysql_connector(monkeypatch, _StubConn(cur))
        backend = MySQLBackend()
        result = await backend.query_by_metadata("tier", "gold")
        assert result[0]["thread_id"] == "t1"
        assert result[0]["data"] == {"x": 1}

    @pytest.mark.asyncio
    async def test_search_data_uses_json_contains(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        rows = [("t1", json.dumps({"agent_id": "a-1"}), ts)]
        cur = _StubCursor(fetchone=(1,), fetchall=rows)
        _stub_mysql_connector(monkeypatch, _StubConn(cur))
        backend = MySQLBackend()
        result = await backend.search_data("agent_id", "a-1")
        assert result[0]["data"] == {"agent_id": "a-1"}

    @pytest.mark.asyncio
    async def test_count_returns_row_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_mysql_connector(monkeypatch, _StubConn(_StubCursor(fetchone=(7,))))
        backend = MySQLBackend()
        assert await backend.count() == 7

    @pytest.mark.asyncio
    async def test_count_returns_zero_when_no_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_mysql_connector(monkeypatch, _StubConn(_StubCursor(fetchone=None)))
        backend = MySQLBackend()
        assert await backend.count() == 0


# ---------------------------------------------------------------------------
# Vacuum
# ---------------------------------------------------------------------------


class TestVacuum:
    @pytest.mark.asyncio
    async def test_vacuum_returns_row_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cur = _StubCursor(fetchone=(1,), rowcount=5)
        _stub_mysql_connector(monkeypatch, _StubConn(cur))
        backend = MySQLBackend()
        assert await backend.vacuum(older_than_days=7) == 5

    @pytest.mark.asyncio
    async def test_vacuum_zero_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cur = _StubCursor(fetchone=(1,), rowcount=0)
        _stub_mysql_connector(monkeypatch, _StubConn(cur))
        backend = MySQLBackend()
        assert await backend.vacuum() == 0


# ---------------------------------------------------------------------------
# close + __repr__
# ---------------------------------------------------------------------------


class TestClose:
    @pytest.mark.asyncio
    async def test_close_calls_connection_close(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = _StubConn()
        _stub_mysql_connector(monkeypatch, conn)
        backend = MySQLBackend()
        await backend._get_pool()
        await backend.close()
        assert conn.closed is True
        assert backend._pool is None
        assert backend._initialized is False

    @pytest.mark.asyncio
    async def test_close_without_open_pool_is_noop(self) -> None:
        backend = MySQLBackend()
        await backend.close()


class TestRepr:
    def test_repr_with_dsn_redacts(self) -> None:
        backend = MySQLBackend(dsn="mysql://user:secret@host/db")
        rep = repr(backend)
        assert "secret" not in rep
        assert "MySQLBackend" in rep

    def test_repr_with_components_shows_host_db(self) -> None:
        backend = MySQLBackend(host="myhost", database="mydb")
        rep = repr(backend)
        assert "myhost" in rep
        assert "mydb" in rep
