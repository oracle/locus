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
(``POSTGRES_HOST`` / ``POSTGRES_PORT`` optional); Oracle ``ORACLE_DSN`` and
``ORACLE_PASSWORD`` (``ORACLE_USER`` and a wallet in ``ORACLE_WALLET``
optional). The threads written are deleted after each test; the PostgreSQL
and Oracle tables, each named for this run alone so that two runs sharing a
database never meet, are dropped when the module ends.
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


def _postgres_dsn() -> str | None:
    user, db = os.getenv("POSTGRES_USER"), os.getenv("POSTGRES_DB")
    if not user or not db:
        return None
    host = os.getenv("POSTGRES_HOST", "127.0.0.1")
    port = os.getenv("POSTGRES_PORT", "5432")
    password = os.getenv("POSTGRES_PASSWORD", "")
    auth = f"{user}:{password}" if password else user
    return f"postgresql://{auth}@{host}:{port}/{db}"


def _oracle_kwargs() -> dict[str, Any] | None:
    """``oracle_checkpointer()`` arguments from ``ORACLE_*``; ``None`` without a database."""
    dsn, password = os.getenv("ORACLE_DSN"), os.getenv("ORACLE_PASSWORD")
    if not dsn or not password:
        return None
    kwargs: dict[str, Any] = {
        "dsn": dsn,
        "user": os.getenv("ORACLE_USER", "ADMIN"),
        "password": password,
    }
    if wallet := os.getenv("ORACLE_WALLET"):
        kwargs["wallet_location"] = wallet
        kwargs["wallet_password"] = os.getenv("ORACLE_WALLET_PASSWORD") or None
    return kwargs


def _make(kind: str, target: str, scope: str) -> Any:
    """One adapter over the store at ``target``: what one process would hold.

    ``scope`` keeps runs apart: the Redis key prefix, or the SQL table name.
    Oracle takes its connection from ``ORACLE_*`` and ignores ``target``.
    """
    if kind == "redis":
        from locus.memory.backends import redis_checkpointer

        return redis_checkpointer(url=target, prefix=scope)
    if kind == "oracle":
        from locus.memory.backends import oracle_checkpointer

        kwargs = _oracle_kwargs()
        assert kwargs is not None
        return oracle_checkpointer(table_name=scope, **kwargs)
    from locus.memory.backends import postgresql_checkpointer

    return postgresql_checkpointer(dsn=target, table_name=scope)


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


def _write_half(kind: str, target: str, scope: str, thread: str, tag: str) -> None:
    """Child process: its own adapter writes ``N // 2`` checkpoints to ``thread`` at once."""

    async def run() -> None:
        cp = _make(kind, target, scope)
        state = AgentState(agent_id="p301")
        try:
            await asyncio.gather(*(cp.save(state, thread, f"{tag}-{i}") for i in range(N // 2)))
        finally:
            await cp.close()

    asyncio.run(run())


async def _two_processes(kind: str, target: str, scope: str, thread: str) -> list[str]:
    """Two processes write ``thread`` at once; what this one lists afterwards."""
    cp = _make(kind, target, scope)
    state = AgentState(agent_id="p301")
    try:
        await cp.save(state, thread, "warm-up")
        await cp.delete(thread)
        context = multiprocessing.get_context("spawn")
        workers = [
            context.Process(target=_write_half, args=(kind, target, scope, thread, tag))
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
def pg_table() -> Iterator[str]:
    """A checkpoint table for this run alone, dropped when the module ends."""
    table = f"locus_p301_index_{uuid.uuid4().hex[:8]}"
    yield table
    dsn = _postgres_dsn()
    if dsn is None:
        return
    import asyncpg

    async def drop() -> None:
        conn = await asyncpg.connect(dsn)
        try:
            await conn.execute(f"DROP TABLE IF EXISTS {table}")
        finally:
            await conn.close()

    asyncio.run(drop())


@pytest.mark.requires_postgres
@pytest.mark.skipif(_postgres_dsn() is None, reason="POSTGRES_USER / POSTGRES_DB not set")
class TestPostgresIndexConcurrency:
    async def test_two_instances_keep_every_entry(self, pg_table: str) -> None:
        dsn = _postgres_dsn()
        assert dsn is not None

        listed = await _two_instances(lambda: _make("postgresql", dsn, pg_table), "hot")

        assert len(set(listed)) == N

    async def test_two_processes_keep_every_entry(self, pg_table: str) -> None:
        dsn = _postgres_dsn()
        assert dsn is not None

        listed = await _two_processes("postgresql", dsn, pg_table, "hot")

        assert len(set(listed)) == N


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def oracle_table() -> Iterator[str]:
    """A checkpoint table for this run alone, dropped when the module ends."""
    table = f"locus_p301_index_{uuid.uuid4().hex[:8]}"
    yield table
    kwargs = _oracle_kwargs()
    if kwargs is None:
        return
    import oracledb

    connect: dict[str, Any] = {k: kwargs[k] for k in ("user", "password", "dsn")}
    if "wallet_location" in kwargs:
        connect["config_dir"] = connect["wallet_location"] = kwargs["wallet_location"]
        connect["wallet_password"] = kwargs["wallet_password"]
    with oracledb.connect(**connect) as conn, conn.cursor() as cursor:
        cursor.execute(f"DROP TABLE {table} PURGE")
        conn.commit()


@pytest.mark.requires_oracle
@pytest.mark.skipif(_oracle_kwargs() is None, reason="ORACLE_DSN / ORACLE_PASSWORD not set")
class TestOracleIndexConcurrency:
    async def test_two_instances_keep_every_entry(self, oracle_table: str) -> None:
        listed = await _two_instances(lambda: _make("oracle", "", oracle_table), "hot")

        assert len(set(listed)) == N

    async def test_two_processes_keep_every_entry(self, oracle_table: str) -> None:
        listed = await _two_processes("oracle", "", oracle_table, "hot")

        assert len(set(listed)) == N
