# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Two writers on one thread keep every index entry, on a live store (issue #301).

The adapter's checkpoint index used to be a read-modify-write that only a
lock inside one process serialised; two adapter instances over one store —
two processes, in effect — interleaved it and lost entries (19 of 20 in
the issue's model; far fewer on a real store). Each backend now keeps the
index in an operation its store makes atomic, so here two instances, and
then two processes, write one thread at once and the index is whole.

Redis needs ``REDIS_URL``; PostgreSQL ``POSTGRES_USER`` and ``POSTGRES_DB``
(``POSTGRES_HOST`` / ``POSTGRES_PORT`` optional). The threads written are
deleted after each test; the PostgreSQL table is dropped when the module ends.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from locus.core.state import AgentState


pytestmark = pytest.mark.integration

#: Checkpoints per run, half from each writer.
N = 20
_PG_TABLE = "locus_p301_index"


def _postgres_dsn() -> str | None:
    user, db = os.getenv("POSTGRES_USER"), os.getenv("POSTGRES_DB")
    if not user or not db:
        return None
    host = os.getenv("POSTGRES_HOST", "127.0.0.1")
    port = os.getenv("POSTGRES_PORT", "5432")
    password = os.getenv("POSTGRES_PASSWORD", "")
    auth = f"{user}:{password}" if password else user
    return f"postgresql://{auth}@{host}:{port}/{db}"


def _make(kind: str, target: str, prefix: str) -> Any:
    """One adapter over the store at ``target``: what one process would hold."""
    if kind == "redis":
        from locus.memory.backends import redis_checkpointer

        return redis_checkpointer(url=target, prefix=prefix)
    from locus.memory.backends import postgresql_checkpointer

    return postgresql_checkpointer(dsn=target, table_name=_PG_TABLE)


async def _two_instances(make: Callable[[], Any], thread: str) -> list[str]:
    """Two adapters (two locks) over one store write ``thread`` at once; what one lists."""
    first, second = make(), make()
    state = AgentState(agent_id="p301")
    try:
        # The table / keyspace exists before the writers race: a cold-start
        # CREATE TABLE from two connections is a different matter.
        await first.save(state, thread, "warm-up")
        await first.delete(thread)
        await asyncio.gather(
            *(first.save(state, thread, f"a-{i}") for i in range(N // 2)),
            *(second.save(state, thread, f"b-{i}") for i in range(N // 2)),
        )
        return await second.list_checkpoints(thread, limit=1000)
    finally:
        await first.delete(thread)
        await first.close()
        await second.close()


def _write_half(kind: str, target: str, prefix: str, thread: str, tag: str) -> None:
    """Child process: its own adapter writes ``N // 2`` checkpoints to ``thread`` at once."""

    async def run() -> None:
        cp = _make(kind, target, prefix)
        state = AgentState(agent_id="p301")
        try:
            await asyncio.gather(*(cp.save(state, thread, f"{tag}-{i}") for i in range(N // 2)))
        finally:
            await cp.close()

    asyncio.run(run())


async def _two_processes(kind: str, target: str, prefix: str, thread: str) -> list[str]:
    """Two processes write ``thread`` at once; what this one lists afterwards."""
    cp = _make(kind, target, prefix)
    state = AgentState(agent_id="p301")
    try:
        await cp.save(state, thread, "warm-up")
        await cp.delete(thread)
        context = multiprocessing.get_context("spawn")
        workers = [
            context.Process(target=_write_half, args=(kind, target, prefix, thread, tag))
            for tag in ("p1", "p2")
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=120)
        assert [worker.exitcode for worker in workers] == [0, 0]
        return await cp.list_checkpoints(thread, limit=1000)
    finally:
        await cp.delete(thread)
        await cp.close()


# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------


@pytest.mark.requires_redis
@pytest.mark.skipif(not os.getenv("REDIS_URL"), reason="REDIS_URL not set")
class TestRedisIndexConcurrency:
    @pytest.fixture
    def prefix(self) -> str:
        return f"locus:p301:{uuid.uuid4().hex[:8]}:"

    async def test_two_instances_keep_every_entry(self, prefix: str) -> None:
        url = os.environ["REDIS_URL"]

        listed = await _two_instances(lambda: _make("redis", url, prefix), "hot")

        assert len(set(listed)) == N

    async def test_two_processes_keep_every_entry(self, prefix: str) -> None:
        listed = await _two_processes("redis", os.environ["REDIS_URL"], prefix, "hot")

        assert len(set(listed)) == N


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _drop_postgres_table() -> Iterator[None]:
    yield
    dsn = _postgres_dsn()
    if dsn is None:
        return
    import asyncpg

    async def drop() -> None:
        conn = await asyncpg.connect(dsn)
        try:
            await conn.execute(f"DROP TABLE IF EXISTS {_PG_TABLE}")
        finally:
            await conn.close()

    asyncio.run(drop())


@pytest.mark.requires_postgres
@pytest.mark.skipif(_postgres_dsn() is None, reason="POSTGRES_USER / POSTGRES_DB not set")
@pytest.mark.usefixtures("_drop_postgres_table")
class TestPostgresIndexConcurrency:
    async def test_two_instances_keep_every_entry(self) -> None:
        dsn = _postgres_dsn()
        assert dsn is not None

        listed = await _two_instances(lambda: _make("postgresql", dsn, ""), "hot")

        assert len(set(listed)) == N

    async def test_two_processes_keep_every_entry(self) -> None:
        dsn = _postgres_dsn()
        assert dsn is not None

        listed = await _two_processes("postgresql", dsn, "", "hot")

        assert len(set(listed)) == N
