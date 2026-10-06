# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""PgCheckpointer on a real Postgres: threads, retention, and tenants kept apart.

Runs only when ``POSTGRES_HOST`` / ``_PORT`` / ``_USER`` / ``_DB`` are set.
Isolation is proven the only way it can be: through an unprivileged app role
(a superuser bypasses Row-Level Security), which here also has no ``CREATE``
anywhere — the table is made by the admin, as a migration would.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest

from tulip.agent import Agent
from tulip.core.messages import Message
from tulip.core.state import AgentState
from tulip.memory.backends import PgCheckpointer
from tulip.memory.retention import MESSAGE_TIME_KEY
from tulip.testing import ScriptedModel, text


try:
    from tests.integration.conftest import skip_without_postgres
except ImportError:  # pragma: no cover - conftest is always importable under pytest
    skip_without_postgres = pytest.mark.skipif(True, reason="conftest unavailable")

pytestmark = [pytest.mark.integration, skip_without_postgres]

_TABLE = "tulip_checkpoints_it"
_ROLE = "tulip_cp_rls_test"
_ROLE_PW = "cp_rls_test_pw"  # noqa: S105 - ephemeral local test role, dropped on teardown


def _dsn(user: str | None = None, password: str | None = None) -> str:
    host = os.environ["POSTGRES_HOST"]
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.environ["POSTGRES_DB"]
    user = user or os.environ["POSTGRES_USER"]
    password = password if password is not None else os.getenv("POSTGRES_PASSWORD", "")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


async def _admin(sql: str) -> None:
    import asyncpg

    conn = await asyncpg.connect(_dsn())
    try:
        await conn.execute(sql)
    finally:
        await conn.close()


@pytest.fixture
async def app_dsn() -> AsyncIterator[str]:
    """The table made by the admin; an app role with DML on it and nothing else."""
    await _admin(f"DROP TABLE IF EXISTS {_TABLE} CASCADE")
    for statement in PgCheckpointer(_dsn(), table=_TABLE).ddl():
        await _admin(statement)
    await _admin(
        f"DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='{_ROLE}') THEN "
        f"CREATE ROLE {_ROLE} LOGIN PASSWORD '{_ROLE_PW}' NOSUPERUSER NOBYPASSRLS; "
        f"END IF; END $$"
    )
    await _admin(f"REVOKE CREATE ON SCHEMA public FROM {_ROLE}")
    await _admin(f"GRANT USAGE ON SCHEMA public TO {_ROLE}")
    await _admin(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {_TABLE} TO {_ROLE}")
    try:
        yield _dsn(_ROLE, _ROLE_PW)
    finally:
        await _admin(f"DROP TABLE IF EXISTS {_TABLE} CASCADE")
        await _admin(f"REVOKE ALL ON SCHEMA public FROM {_ROLE}")
        await _admin(f"DROP ROLE IF EXISTS {_ROLE}")


def _state(*texts: str) -> AgentState:
    messages = [Message.system("you help")]
    for t in texts:
        messages += [Message.user(t), Message.assistant(f"re: {t}")]
    return AgentState(messages=tuple(messages))


async def test_an_agent_keeps_its_thread_as_an_app_role_with_no_create(app_dsn: str) -> None:
    cp = PgCheckpointer(app_dsn, table=_TABLE, keep_checkpoints=1)
    try:
        agent = Agent(model=ScriptedModel([text("one"), text("two")]), checkpointer=cp)
        with cp.tenant_scope("acme"):
            await agent.arun("first", thread_id="t")
            await agent.arun("second", thread_id="t")
            loaded = await cp.load("t")
            assert loaded is not None
            talk = [m.content for m in loaded.messages if m.role != "system"]
            assert talk == ["first", "one", "second", "two"]
            # keep_checkpoints=1: the thread is one row, however many turns.
            assert len(await cp.list_checkpoints("t", limit=100)) == 1
    finally:
        await cp.close()


async def test_keep_checkpoints_prunes_in_the_save(app_dsn: str) -> None:
    cp = PgCheckpointer(app_dsn, table=_TABLE, keep_checkpoints=2)
    try:
        ids = [await cp.save(_state(f"turn {i}"), "t") for i in range(4)]
        assert await cp.list_checkpoints("t") == [ids[3], ids[2]]
        # Re-saving an id moves it to the front; it is not counted twice.
        await cp.save(_state("again"), "t", checkpoint_id=ids[2])
        assert await cp.list_checkpoints("t") == [ids[2], ids[3]]
        loaded = await cp.load("t")
        assert loaded is not None
        assert loaded.messages[1].content == "again"
        assert await cp.delete("t", ids[3]) is True
        assert await cp.list_checkpoints("t") == [ids[2]]
    finally:
        await cp.close()


async def test_tenants_never_see_each_others_threads(app_dsn: str) -> None:
    cp = PgCheckpointer(app_dsn, table=_TABLE, tenant_of=lambda t: t.split("/")[0])
    try:
        await cp.save(_state("acme secret"), "acme/support")
        await cp.save(_state("globex secret"), "globex/support")

        # Same thread name, other tenant: nothing there.
        with cp.tenant_scope("globex"):
            assert await cp.load("acme/support") is None
            assert await cp.list_threads() == ["globex/support"]
            assert await cp.delete("acme/support") is False
            assert await cp.vacuum(0) == 1  # only globex's thread is old enough to see
        loaded = await cp.load("acme/support")
        assert loaded is not None
        assert loaded.messages[1].content == "acme secret"
        with cp.tenant_scope("acme"):
            assert await cp.forget_tenant() == 1
        assert await cp.load("acme/support") is None
    finally:
        await cp.close()


async def test_rls_confines_a_raw_scan_and_refuses_a_cross_tenant_write(app_dsn: str) -> None:
    import asyncpg

    cp = PgCheckpointer(app_dsn, table=_TABLE)
    try:
        with cp.tenant_scope("acme"):
            await cp.save(_state("acme"), "t")
        with cp.tenant_scope("globex"):
            await cp.save(_state("globex"), "t")
    finally:
        await cp.close()

    conn = await asyncpg.connect(app_dsn)
    try:
        # No tenant pinned: RLS shows nothing at all.
        assert await conn.fetch(f"SELECT tenant FROM {_TABLE}") == []  # noqa: S608 - test constant
        async with conn.transaction():
            await conn.execute("SELECT set_config('tulip.tenant', 'acme', true)")
            rows = await conn.fetch(f"SELECT tenant FROM {_TABLE}")  # noqa: S608 - test constant
            assert {r["tenant"] for r in rows} == {"acme"}
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await conn.execute(
                "SET LOCAL tulip.tenant = 'acme'; "  # noqa: S608 - test constant
                f"INSERT INTO {_TABLE} (tenant, thread_id, checkpoint_id, data) "
                "VALUES ('globex', 't', 'evil', '{}'::jsonb)"
            )
    finally:
        await conn.close()


async def test_retention_on_save_and_purge_of_idle_threads(app_dsn: str) -> None:
    old = (datetime.now(UTC) - timedelta(days=40)).isoformat()

    def aged(state: AgentState) -> AgentState:
        return state.model_copy(
            update={
                "messages": tuple(
                    m.model_copy(update={"metadata": {MESSAGE_TIME_KEY: old}})
                    for m in state.messages
                )
            }
        )

    keep_all = PgCheckpointer(app_dsn, table=_TABLE)
    retained = PgCheckpointer(app_dsn, table=_TABLE, message_retention=timedelta(days=30))
    try:
        # An idle thread, written long ago with no retention, two checkpoints.
        first = await keep_all.save(aged(_state("ancient")), "idle")
        await keep_all.save(aged(_state("ancient", "also ancient")), "idle")
        # An active thread: the save itself trims.
        stale = aged(_state("ancient"))
        await retained.save(
            stale.model_copy(update={"messages": (*stale.messages, Message.user("today"))}),
            "active",
        )
        active = await retained.load("active")
        assert active is not None
        assert [m.content for m in active.messages] == ["you help", "today"]

        assert await retained.purge_messages(timedelta(days=30)) == 6
        for checkpoint_id in await keep_all.list_checkpoints("idle"):
            state = await keep_all.load("idle", checkpoint_id)
            assert state is not None
            assert [m.content for m in state.messages] == ["you help"]
        # In place: ids and order unchanged, and a second purge has nothing to do.
        assert (await keep_all.list_checkpoints("idle"))[-1] == first
        assert await retained.purge_messages(timedelta(days=30)) == 0
    finally:
        await keep_all.close()
        await retained.close()


async def test_an_old_layout_table_is_refused(app_dsn: str) -> None:
    await _admin("DROP TABLE IF EXISTS tulip_checkpoints_old_it")
    await _admin(
        "CREATE TABLE tulip_checkpoints_old_it (thread_id text PRIMARY KEY, "
        "checkpoint_id text, data jsonb NOT NULL)"
    )
    cp = PgCheckpointer(_dsn(), table="tulip_checkpoints_old_it")
    try:
        with pytest.raises(RuntimeError, match="another layout"):
            await cp.load("t")
    finally:
        await cp.close()
        await _admin("DROP TABLE IF EXISTS tulip_checkpoints_old_it")
