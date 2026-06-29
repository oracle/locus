# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Coverage tests for ``tulip.memory.backends.mysql._MySQLConnectionPool``.

Targets the pool-internal branches the main MySQL suite leaves uncovered:
the MySQL-specific error tuple, the wait-then-reconnect path when a queued
connection turns out to be stale, and the closed-connection health probe.
"""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any

import pytest

from tulip.memory.backends.mysql import MySQLConfig, _MySQLConnectionPool


# ---------------------------------------------------------------------------
# Minimal connector stubs (mirrors test_memory_backends_mysql.py)
# ---------------------------------------------------------------------------


class _StubCursor:
    async def __aenter__(self) -> _StubCursor:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    async def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None:
        return None

    async def fetchone(self) -> Any:
        return (1,)


class _BrokenCursor(_StubCursor):
    async def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None:
        raise RuntimeError("connection is stale")


class _StubConn:
    def __init__(self, cursor: _StubCursor | None = None) -> None:
        self.cursor_obj = cursor or _StubCursor()
        self.closed = False

    async def cursor(self) -> _StubCursor:
        return self.cursor_obj

    async def rollback(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True


def _stub_mysql_connector(monkeypatch: pytest.MonkeyPatch, conn: _StubConn) -> None:
    async def fake_connect(**kwargs: Any) -> _StubConn:
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


# ---------------------------------------------------------------------------
# _connection_errors — MySQL error included when the driver is importable
# ---------------------------------------------------------------------------


def test_connection_errors_includes_mysql_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _MySQLError(Exception):
        pass

    mysql_mod = types.ModuleType("mysql")
    connector_mod = types.ModuleType("mysql.connector")
    connector_mod.Error = _MySQLError  # type: ignore[attr-defined]
    mysql_mod.connector = connector_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mysql", mysql_mod)
    monkeypatch.setitem(sys.modules, "mysql.connector", connector_mod)

    errors = _MySQLConnectionPool._connection_errors()
    assert _MySQLError in errors
    assert OSError in errors


# ---------------------------------------------------------------------------
# _acquire — wait for a queued connection, discard it when stale, reconnect
# ---------------------------------------------------------------------------


async def test_acquire_waits_then_discards_stale_and_reconnects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = _StubConn(_BrokenCursor())
    replacement = _StubConn()
    _stub_mysql_connector(monkeypatch, replacement)

    pool = _MySQLConnectionPool(MySQLConfig(min_pool_size=0, max_pool_size=1))
    pool._created = 1  # at capacity, queue empty → second wait loop is taken

    task = asyncio.create_task(pool._acquire())
    await asyncio.sleep(0)
    assert not task.done()

    await pool._available.put(stale)
    conn = await task

    assert conn is replacement
    assert stale.closed is True
    assert pool._created == 1


# ---------------------------------------------------------------------------
# _is_healthy — closed connection is unusable
# ---------------------------------------------------------------------------


async def test_is_healthy_false_for_closed_connection() -> None:
    pool = _MySQLConnectionPool(MySQLConfig(min_pool_size=0))
    conn = _StubConn()
    conn.closed = True
    assert await pool._is_healthy(conn) is False
