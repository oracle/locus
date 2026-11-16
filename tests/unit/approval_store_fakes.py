# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Driver fakes for the database-backed approval stores.

Each fake understands exactly the statements its store issues and keeps rows in
a dict shared by every connection to the same "server", so the conformance
suite (including "another process sees the decision") runs with no service.
They record every statement, so a test can also pin the SQL shape -- the row
locks, the conditional DDL -- the way the checkpointer stub tests do.
"""

from __future__ import annotations

import asyncio
import builtins
import json
import re
import sys
import types
from contextlib import asynccontextmanager
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# A table that both SQL fakes share the logic of
# ---------------------------------------------------------------------------


class FakeTable:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.created = False


class FakeServer:
    """Rows per table, shared across every connection of one fake server.

    ``parsed_json=True`` hands the ``record`` column back as a dict instead of a
    LOB or text, as python-oracledb does for a ``CLOB CHECK (IS JSON)`` column
    on Oracle AI Database 26ai.
    """

    def __init__(self, *, parsed_json: bool = False) -> None:
        self.tables: dict[str, FakeTable] = {}
        self.statements: list[str] = []
        self.parsed_json = parsed_json

    def table(self, name: str) -> FakeTable:
        return self.tables.setdefault(name.lower(), FakeTable())

    def snapshot(self) -> dict[str, dict[str, dict[str, Any]]]:
        return {name: {k: dict(v) for k, v in t.rows.items()} for name, t in self.tables.items()}

    def restore(self, snapshot: dict[str, dict[str, dict[str, Any]]]) -> None:
        for name, rows in snapshot.items():
            self.tables[name].rows = rows
        for name in set(self.tables) - set(snapshot):
            self.tables[name].rows = {}


_TABLE_IN = re.compile(r"(?:FROM|INTO|UPDATE|TABLE(?: IF NOT EXISTS)?)\s+([\w.]+)", re.IGNORECASE)


def _table_name(sql: str) -> str:
    match = _TABLE_IN.search(sql)
    assert match, sql
    return match.group(1)


def _query(table: FakeTable, sql: str, params: dict[str, Any]) -> list[Any]:
    """Rows for one of the store's SELECTs, as the driver would return them."""
    if "WHERE approval_id" in sql:
        row = table.rows.get(params["approval_id"])
        return [row] if row is not None else []
    if "WHERE digest" in sql:
        return [r for r in table.rows.values() if r["digest"] == params["digest"]]
    if "WHERE status = 'pending'" in sql:
        rows = [r for r in table.rows.values() if r["status"] == "pending"]
        return sorted(rows, key=lambda r: r["created_at"])
    raise AssertionError(f"unexpected query: {sql}")


# ---------------------------------------------------------------------------
# asyncpg
# ---------------------------------------------------------------------------


class _PgConn:
    def __init__(self, server: FakeServer) -> None:
        self._server = server
        self._in_transaction = False

    def _args(self, sql: str, args: tuple[Any, ...]) -> dict[str, Any]:
        if "WHERE approval_id = $1" in sql:
            return {"approval_id": args[0]}
        if "WHERE digest = $1" in sql:
            return {"digest": args[0]}
        return {}

    async def fetchval(self, sql: str, *args: Any) -> Any:
        self._server.statements.append(sql)
        assert "to_regclass" in sql, sql
        table = self._server.table(args[0])
        return args[0] if table.created else None

    async def fetchrow(self, sql: str, *args: Any) -> Any:
        self._server.statements.append(sql)
        rows = _query(self._server.table(_table_name(sql)), sql, self._args(sql, args))
        return {"record": rows[0]["record"]} if rows else None

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        self._server.statements.append(sql)
        rows = _query(self._server.table(_table_name(sql)), sql, self._args(sql, args))
        return [{"record": r["record"]} for r in rows]

    async def execute(self, sql: str, *args: Any) -> str:
        self._server.statements.append(sql)
        head = sql.strip().split()[0].upper()
        if head == "CREATE":
            if "TABLE" in sql.upper():
                self._server.table(_table_name(sql)).created = True
            return "CREATE"
        table = self._server.table(_table_name(sql))
        if head == "INSERT":
            approval_id, digest, status, created_at, record = args
            if approval_id in table.rows:
                return "INSERT 0 0"
            table.rows[approval_id] = {
                "approval_id": approval_id,
                "digest": digest,
                "status": status,
                "created_at": created_at,
                "record": record,
            }
            return "INSERT 0 1"
        if head == "UPDATE":
            approval_id, status, record = args
            table.rows[approval_id].update(status=status, record=record)
            return "UPDATE 1"
        raise AssertionError(f"unexpected statement: {sql}")

    @asynccontextmanager
    async def transaction(self) -> Any:
        assert not self._in_transaction, "nested transaction"
        self._in_transaction = True
        before = self._server.snapshot()
        try:
            yield
        except BaseException:
            self._server.restore(before)  # an exception rolls the transaction back
            raise
        finally:
            self._in_transaction = False


class _PgPool:
    def __init__(self, server: FakeServer) -> None:
        self._server = server
        self.closed = False

    @asynccontextmanager
    async def acquire(self) -> Any:
        yield _PgConn(self._server)

    async def close(self) -> None:
        self.closed = True


def install_asyncpg(monkeypatch: pytest.MonkeyPatch, server: FakeServer) -> list[dict[str, Any]]:
    """A fake ``asyncpg`` whose pools all serve ``server``; returns the create_pool calls."""
    calls: list[dict[str, Any]] = []

    async def create_pool(dsn: str | None = None, **kwargs: Any) -> _PgPool:
        calls.append({"dsn": dsn, **kwargs})
        return _PgPool(server)

    fake = types.ModuleType("asyncpg")
    fake.create_pool = create_pool  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "asyncpg", fake)
    return calls


# ---------------------------------------------------------------------------
# oracledb
# ---------------------------------------------------------------------------


class FakeIntegrityError(Exception):
    pass


class _Lob:
    def __init__(self) -> None:
        self.value = ""

    async def write(self, value: str) -> None:
        self.value = value

    async def read(self) -> str:
        return self.value


class _OracleCursor:
    def __init__(self, server: FakeServer) -> None:
        self._server = server
        self._rows: list[Any] = []

    def _fetched(self, stored: Any) -> Any:
        if not self._server.parsed_json:
            return stored
        return json.loads(stored.value if isinstance(stored, _Lob) else stored)

    async def execute(self, sql: str, params: dict[str, Any] | None = None) -> None:
        self._server.statements.append(sql)
        params = params or {}
        head = sql.strip().split()[0].upper()
        if "user_tables" in sql:
            table = self._server.table(params["table_name"])
            self._rows = [(1 if table.created else 0,)]
            return
        if head == "CREATE":
            if "TABLE" in sql.upper():
                self._server.table(_table_name(sql)).created = True
            return
        table = self._server.table(_table_name(sql))
        if head == "SELECT":
            self._rows = [(self._fetched(r["record"]),) for r in _query(table, sql, params)]
            return
        if head == "INSERT":
            if params["approval_id"] in table.rows:
                raise FakeIntegrityError("ORA-00001: unique constraint violated")
            table.rows[params["approval_id"]] = {
                "approval_id": params["approval_id"],
                "digest": params["digest"],
                "status": params["status"],
                "created_at": params["created_at"],
                "record": params["record"],
            }
            return
        if head == "UPDATE":
            table.rows[params["approval_id"]].update(
                status=params["status"], record=params["record"]
            )
            return
        raise AssertionError(f"unexpected statement: {sql}")

    async def fetchone(self) -> Any:
        return self._rows[0] if self._rows else None

    async def fetchall(self) -> list[Any]:
        return list(self._rows)

    async def __aenter__(self) -> _OracleCursor:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _OracleConn:
    def __init__(self, server: FakeServer) -> None:
        self._server = server
        self._committed = server.snapshot()
        self.commits = 0
        self.rollbacks = 0

    def cursor(self) -> _OracleCursor:
        return _OracleCursor(self._server)

    async def createlob(self, kind: Any) -> _Lob:
        assert kind == "CLOB-SENTINEL"
        return _Lob()

    async def commit(self) -> None:
        self.commits += 1
        self._committed = self._server.snapshot()

    async def rollback(self) -> None:
        self.rollbacks += 1
        self._server.restore(self._committed)

    async def __aenter__(self) -> _OracleConn:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _OraclePool:
    def __init__(self, server: FakeServer) -> None:
        self._server = server
        self.connections: list[_OracleConn] = []
        self.closed = False

    @asynccontextmanager
    async def acquire(self) -> Any:
        conn = _OracleConn(self._server)
        self.connections.append(conn)
        yield conn

    async def close(self) -> None:
        self.closed = True


def install_oracledb(
    monkeypatch: pytest.MonkeyPatch, server: FakeServer
) -> tuple[list[dict[str, Any]], list[_OraclePool]]:
    """A fake ``oracledb``; returns the create_pool_async calls and the pools made."""
    calls: list[dict[str, Any]] = []
    pools: list[_OraclePool] = []

    def create_pool_async(**kwargs: Any) -> _OraclePool:
        calls.append(kwargs)
        pool = _OraclePool(server)
        pools.append(pool)
        return pool

    def makedsn(host: str, port: int, *, service_name: str) -> str:
        return f"(HOST={host})(PORT={port})(SERVICE={service_name})"

    fake = types.ModuleType("oracledb")
    fake.create_pool_async = create_pool_async  # type: ignore[attr-defined]
    fake.makedsn = makedsn  # type: ignore[attr-defined]
    fake.DB_TYPE_CLOB = "CLOB-SENTINEL"  # type: ignore[attr-defined]
    fake.IntegrityError = FakeIntegrityError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "oracledb", fake)
    return calls, pools


# ---------------------------------------------------------------------------
# redis.asyncio
# ---------------------------------------------------------------------------


class FakeRedisServer:
    def __init__(self) -> None:
        self.strings: dict[str, str] = {}
        self.sets: dict[str, set[str]] = {}
        self.locked: str | None = None
        self.commands: list[str] = []


class _FakeLock:
    def __init__(self, server: FakeRedisServer, name: str) -> None:
        self._server = server
        self._name = name

    async def __aenter__(self) -> _FakeLock:
        # Polled, not an asyncio.Lock: every store owns its own event loop,
        # and the fake server is shared across them, as a real Redis is.
        while self._server.locked is not None:  # noqa: ASYNC110
            await asyncio.sleep(0)
        self._server.locked = self._name
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self._server.locked = None


class _FakeRedis:
    def __init__(self, server: FakeRedisServer) -> None:
        self._server = server
        self.closed = False

    def lock(self, name: str, **kwargs: Any) -> _FakeLock:
        self._server.commands.append(f"LOCK {name} {sorted(kwargs)}")
        return _FakeLock(self._server, name)

    async def get(self, key: str) -> str | None:
        self._server.commands.append(f"GET {key}")
        return self._server.strings.get(key)

    async def mget(self, keys: list[str]) -> list[str | None]:
        self._server.commands.append(f"MGET {len(keys)}")
        return [self._server.strings.get(k) for k in keys]

    async def set(self, key: str, value: str, nx: bool = False) -> bool | None:
        self._server.commands.append(f"SET {key}{' NX' if nx else ''}")
        if nx and key in self._server.strings:
            return None
        self._server.strings[key] = value
        return True

    async def sadd(self, key: str, *members: str) -> int:
        self._server.commands.append(f"SADD {key}")
        target = self._server.sets.setdefault(key, set())
        before = len(target)
        target.update(members)
        return len(target) - before

    async def srem(self, key: str, *members: str) -> int:
        self._server.commands.append(f"SREM {key}")
        target = self._server.sets.setdefault(key, set())
        before = len(target)
        target.difference_update(members)
        return before - len(target)

    async def smembers(self, key: str) -> builtins.set[str]:
        # ``builtins.set``: this class's ``set`` method shadows the type name.
        self._server.commands.append(f"SMEMBERS {key}")
        return builtins.set(self._server.sets.get(key, builtins.set()))

    async def aclose(self) -> None:
        self.closed = True


def install_redis(monkeypatch: pytest.MonkeyPatch, server: FakeRedisServer) -> list[dict[str, Any]]:
    """A fake ``redis.asyncio`` whose clients all serve ``server``; returns the from_url calls."""
    calls: list[dict[str, Any]] = []

    class Redis:
        @staticmethod
        def from_url(url: str, **kwargs: Any) -> _FakeRedis:
            calls.append({"url": url, **kwargs})
            return _FakeRedis(server)

    fake = types.ModuleType("redis.asyncio")
    fake.Redis = Redis  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "redis.asyncio", fake)
    return calls


def records_in(server: FakeServer, table: str) -> list[dict[str, Any]]:
    """The JSON records a SQL fake holds for ``table`` (Oracle's are LOBs)."""
    documents = [r["record"] for r in server.table(table).rows.values()]
    return [json.loads(d.value if isinstance(d, _Lob) else d) for d in documents]
