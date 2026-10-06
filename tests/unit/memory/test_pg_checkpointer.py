# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""PgCheckpointer without a database: the statements it sends, and the tenant it sends them for.

``asyncpg`` is replaced by a recorder, so these pin the contract — one data
statement per save, the tenant pinned in the same transaction, no DDL against
an existing table — while ``tests/integration/test_pg_checkpointer.py`` proves
the behaviour (and Row-Level Security) on a real Postgres.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tulip.core.messages import Message
from tulip.core.state import AgentState
from tulip.memory.backends import PgCheckpointer
from tulip.memory.retention import MESSAGE_TIME_KEY


_COLUMNS = [
    "tenant",
    "thread_id",
    "checkpoint_id",
    "seq",
    "data",
    "metadata",
    "oldest_at",
    "created_at",
    "updated_at",
]


class _Conn:
    """Records every statement; answers the schema probe from ``columns``."""

    def __init__(self, columns: list[str], value: Any = None) -> None:
        self.columns = columns
        self.value = value
        self.calls: list[tuple[str, str, tuple[Any, ...]]] = []
        self.transactions = 0

    @asynccontextmanager
    async def transaction(self) -> Any:
        self.transactions += 1
        yield

    async def execute(self, sql: str, *args: Any) -> str:
        self.calls.append(("execute", sql, args))
        return "DELETE 2" if sql.lstrip().startswith("DELETE") else "INSERT 0 1"

    async def executemany(self, sql: str, args: list[Any]) -> None:
        self.calls.append(("executemany", sql, tuple(args)))

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        self.calls.append(("fetch", sql, args))
        if "information_schema" in sql:
            return [{"column_name": c} for c in self.columns]
        return []

    async def fetchval(self, sql: str, *args: Any) -> Any:
        self.calls.append(("fetchval", sql, args))
        return self.value

    async def fetchrow(self, sql: str, *args: Any) -> Any:
        self.calls.append(("fetchrow", sql, args))
        return None

    def data_calls(self) -> list[tuple[str, str, tuple[Any, ...]]]:
        """Everything but the schema probe and the tenant pin."""
        return [
            c for c in self.calls if "information_schema" not in c[1] and "set_config" not in c[1]
        ]

    def pins(self) -> list[Any]:
        return [c[2][0] for c in self.calls if "set_config" in c[1]]


class _Pool:
    def __init__(self, conn: _Conn) -> None:
        self.conn = conn

    @asynccontextmanager
    async def acquire(self) -> Any:
        yield self.conn

    async def close(self) -> None:
        return None


@pytest.fixture
def conn(monkeypatch: pytest.MonkeyPatch) -> _Conn:
    """A recorder behind a fake ``asyncpg``, over a table that already exists."""
    asyncpg = pytest.importorskip("asyncpg")
    recorder = _Conn(list(_COLUMNS))

    async def create_pool(*_args: Any, **_kwargs: Any) -> _Pool:
        return _Pool(recorder)

    monkeypatch.setattr(asyncpg, "create_pool", create_pool)
    return recorder


def _state(*texts: str) -> AgentState:
    messages = [Message.system("you help")]
    for t in texts:
        messages += [Message.user(t), Message.assistant(f"re: {t}")]
    return AgentState(messages=tuple(messages))


# ── one statement per save ───────────────────────────────────────────────────


async def test_a_save_is_one_statement_in_one_transaction(conn: _Conn) -> None:
    cp = PgCheckpointer("postgresql://stub", keep_checkpoints=1)
    checkpoint_id = await cp.save(_state("hi"), "t1")

    data = conn.data_calls()
    assert len(data) == 1, data
    kind, sql, args = data[0]
    assert kind == "execute"
    # The upsert and the prune travel together.
    assert sql.startswith("WITH saved AS (INSERT INTO public.tulip_checkpoints")
    assert "DELETE FROM public.tulip_checkpoints" in sql
    assert args[:3] == ("default", "t1", checkpoint_id)
    assert args[-1] == 0  # keep_checkpoints=1 keeps none of the others
    assert conn.pins() == ["default"]


async def test_without_keep_checkpoints_a_save_is_a_plain_upsert(conn: _Conn) -> None:
    cp = PgCheckpointer("postgresql://stub")
    await cp.save(_state("hi"), "t1", checkpoint_id="cp-1")
    ((_, sql, args),) = conn.data_calls()
    assert sql.startswith("INSERT INTO public.tulip_checkpoints")
    assert "DELETE" not in sql
    assert args[2] == "cp-1"


async def test_no_ddl_against_an_existing_table(conn: _Conn) -> None:
    cp = PgCheckpointer("postgresql://stub")
    await cp.save(_state("hi"), "t1")
    assert not [c for c in conn.calls if c[1].lstrip().upper().startswith(("CREATE", "ALTER"))]


async def test_a_missing_table_is_created_with_rls(conn: _Conn) -> None:
    conn.columns = []
    cp = PgCheckpointer("postgresql://stub", table="threads")
    await cp.save(_state("hi"), "t1")
    ddl = [c[1] for c in conn.calls if c[1].lstrip().upper().startswith(("CREATE", "ALTER"))]
    assert any("CREATE TABLE IF NOT EXISTS public.threads" in s for s in ddl)
    assert any("FORCE ROW LEVEL SECURITY" in s for s in ddl)
    assert any("current_setting('tulip.tenant', true)" in s for s in ddl)
    # The public schema is never created: that needs CREATE on the database.
    assert not any(s.startswith("CREATE SCHEMA") for s in ddl)


async def test_create_schema_false_never_runs_ddl(conn: _Conn) -> None:
    conn.columns = []
    cp = PgCheckpointer("postgresql://stub", create_schema=False)
    with pytest.raises(RuntimeError, match="create_schema=False"):
        await cp.save(_state("hi"), "t1")
    assert not [c for c in conn.calls if "CREATE" in c[1]]


async def test_a_table_of_another_layout_is_refused(conn: _Conn) -> None:
    conn.columns = ["thread_id", "checkpoint_id", "data", "created_at", "updated_at", "metadata"]
    cp = PgCheckpointer("postgresql://stub")
    with pytest.raises(RuntimeError, match="another layout"):
        await cp.load("t1")


# ── which tenant ─────────────────────────────────────────────────────────────


async def test_scope_beats_tenant_of_beats_the_default(conn: _Conn) -> None:
    cp = PgCheckpointer("postgresql://stub", tenant="fallback", tenant_of=lambda t: t.split("/")[0])
    await cp.load("acme/1")
    with cp.tenant_scope("globex"):
        await cp.load("acme/1")
    assert conn.pins() == ["acme", "globex"]
    assert cp.tenant_for() == "fallback"  # no thread: nothing for tenant_of to read


async def test_a_tenant_of_that_cannot_place_a_thread_fails_closed(conn: _Conn) -> None:
    def broken(thread_id: str) -> str:
        raise KeyError(thread_id)

    for tenant_of in (broken, lambda _t: ""):
        cp = PgCheckpointer("postgresql://stub", tenant_of=tenant_of)
        with pytest.raises(ValueError, match="tenant_of"):
            await cp.save(_state("hi"), "who-knows")
    assert conn.data_calls() == []


async def test_scopes_are_context_local() -> None:
    import asyncio

    cp = PgCheckpointer("postgresql://stub")
    seen: dict[str, str] = {}

    async def run(name: str) -> None:
        with cp.tenant_scope(name):
            await asyncio.sleep(0)
            seen[name] = cp.tenant_for("x")

    await asyncio.gather(run("acme"), run("globex"))
    assert seen == {"acme": "acme", "globex": "globex"}


async def test_copy_thread_refuses_another_tenant(conn: _Conn) -> None:
    cp = PgCheckpointer("postgresql://stub", tenant_of=lambda t: t.split("/")[0])
    with pytest.raises(ValueError, match="another tenant"):
        await cp.copy_thread("acme/1", "globex/1")


# ── what a save writes ───────────────────────────────────────────────────────


async def test_a_save_stamps_and_trims_messages(conn: _Conn) -> None:
    cp = PgCheckpointer("postgresql://stub", message_retention=timedelta(days=30))
    old = (datetime.now(UTC) - timedelta(days=40)).isoformat()
    first = _state("a long time ago")
    aged = tuple(
        m if m.role == "system" else m.model_copy(update={"metadata": {MESSAGE_TIME_KEY: old}})
        for m in first.messages
    )
    state = first.model_copy(
        update={"messages": (*aged, Message.user("today"), Message.assistant("hello"))}
    )
    await cp.save(state, "t1")
    ((_, _, args),) = conn.data_calls()
    written = json.loads(args[3])
    texts = [m["content"] for m in written["messages"]]
    assert texts == ["you help", "today", "hello"]
    assert all(MESSAGE_TIME_KEY in m["metadata"] for m in written["messages"])
    assert args[5] is not None  # oldest_at, what purge_messages looks rows up by


def test_arguments_are_checked() -> None:
    with pytest.raises(ValueError, match="keep_checkpoints"):
        PgCheckpointer("postgresql://stub", keep_checkpoints=0)
    with pytest.raises(ValueError, match="message_retention"):
        PgCheckpointer("postgresql://stub", message_retention=timedelta(0))
    with pytest.raises(ValueError, match="invalid table"):
        PgCheckpointer("postgresql://stub", table="threads; DROP TABLE x")
    with pytest.raises(ValueError, match="tenant"):
        PgCheckpointer("postgresql://stub", tenant="")


def test_capabilities_and_deletes() -> None:
    cp = PgCheckpointer("postgresql://stub")
    assert cp.capabilities.vacuum
    assert cp.capabilities.list_threads
    # The agent checkpoints every iteration only where the turn's final save
    # can delete those saves again.
    assert cp.deletes_single_checkpoints is True


def test_ddl_is_available_for_a_migration() -> None:
    statements = PgCheckpointer("postgresql://stub", schema="agents", table="threads").ddl()
    assert statements[0] == "CREATE SCHEMA IF NOT EXISTS agents"
    assert any("agents.threads" in s for s in statements)
