# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""First-time table creation on PostgreSQL is serialised, so concurrent creators cannot race."""

from __future__ import annotations

import sys
import types
from contextlib import asynccontextmanager
from typing import Any

import pytest

from locus.memory.backends.postgresql import PostgreSQLBackend


class _Conn:
    """Records statements; ``to_regclass`` answers None until a CREATE TABLE ran."""

    def __init__(self, *, table_exists: bool = False) -> None:
        self.table_exists = table_exists
        self.calls: list[str] = []
        self.in_transaction = False
        self.ddl_in_transaction: list[bool] = []

    @asynccontextmanager
    async def transaction(self) -> Any:
        self.in_transaction = True
        try:
            yield
        finally:
            self.in_transaction = False

    async def fetchval(self, sql: str, *args: Any) -> Any:
        self.calls.append(sql.strip())
        if "to_regclass" in sql:
            return "t" if self.table_exists else None
        return None  # the advisory lock

    async def execute(self, sql: str, *args: Any) -> str:
        self.calls.append(sql.strip())
        self.ddl_in_transaction.append(self.in_transaction)
        if "CREATE TABLE" in sql:
            self.table_exists = True
        return "OK"


class _Pool:
    def __init__(self, conn: _Conn) -> None:
        self.conn = conn

    @asynccontextmanager
    async def acquire(self) -> Any:
        yield self.conn

    async def close(self) -> None:
        return None


def _backend(monkeypatch: pytest.MonkeyPatch, conn: _Conn) -> PostgreSQLBackend:
    pool = _Pool(conn)

    async def create_pool(*args: Any, **kwargs: Any) -> _Pool:
        return pool

    fake = types.ModuleType("asyncpg")
    fake.create_pool = create_pool  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "asyncpg", fake)
    return PostgreSQLBackend(dsn="postgresql://x/y", table_name="race_t")


@pytest.mark.asyncio
async def test_first_time_ddl_runs_under_an_advisory_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _Conn()
    backend = _backend(monkeypatch, conn)
    await backend._ensure_table()
    lock_at = next(i for i, c in enumerate(conn.calls) if "pg_advisory_xact_lock" in c)
    first_ddl = next(i for i, c in enumerate(conn.calls) if c.startswith("CREATE"))
    assert lock_at < first_ddl, "the lock is taken before any DDL"
    assert all(conn.ddl_in_transaction), "every DDL statement ran inside the locked transaction"
    assert sum("to_regclass" in c for c in conn.calls) == 2, (
        "probed once before and once under the lock"
    )


@pytest.mark.asyncio
async def test_a_creator_that_lost_the_lock_sees_the_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """The second probe, under the lock, finds the table the winner made: no DDL, no error."""
    conn = _Conn()
    backend = _backend(monkeypatch, conn)

    original = conn.fetchval

    async def fetchval(sql: str, *args: Any) -> Any:
        if "pg_advisory_xact_lock" in sql:
            conn.table_exists = True  # the other process created it while we waited
        return await original(sql, *args)

    conn.fetchval = fetchval  # type: ignore[method-assign]
    await backend._ensure_table()
    assert not any(c.startswith("CREATE") for c in conn.calls)
    assert backend._initialized is True
