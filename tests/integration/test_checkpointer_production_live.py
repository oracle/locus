# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""What only the live stores can show about the production settings.

The behaviours every checkpointer shares run in the conformance suite
(``tests/integration/test_checkpointer_time_travel_live.py`` mixes
``ProductionSettingsConformance`` into the PostgreSQL and Redis classes).
This file looks at the stores themselves: on Redis (``REDIS_URL``) every key
a thread writes — the checkpoint, its ``latest`` copy, the index hash and
the tool-effect ledger's sidecar — carries the store's own ``EXPIRE``; on
PostgreSQL (``POSTGRES_USER`` / ``POSTGRES_DB``, ``POSTGRES_HOST`` /
``POSTGRES_PORT`` optional) the rows hold ciphertext in ``data`` with the
listing columns in the clear, the ``expires_at`` column is added to a table
made before it existed, and the scope is the prefix of ``thread_id``.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

from locus.core.messages import Message
from locus.core.state import AgentState
from locus.memory import CheckpointScope, EncryptedSerializer
from locus.memory.ledger import ToolEffectLedger
from tests.unit.checkpointer_time_travel_conformance import _agent, _record_calls


pytestmark = pytest.mark.integration

SENSITIVE = "the customer's card ends in 4242"
KEY = secrets.token_bytes(32)
SCOPE = CheckpointScope(app_id="billing", user_id="u-1")


def _postgres_dsn() -> str | None:
    user, db = os.getenv("POSTGRES_USER"), os.getenv("POSTGRES_DB")
    if not user or not db:
        return None
    host = os.getenv("POSTGRES_HOST", "127.0.0.1")
    port = os.getenv("POSTGRES_PORT", "5432")
    password = os.getenv("POSTGRES_PASSWORD", "")
    auth = f"{user}:{password}" if password else user
    return f"postgresql://{auth}@{host}:{port}/{db}"


def _secret_state() -> AgentState:
    return AgentState(agent_id="prod").with_message(Message.user(SENSITIVE))


# ---------------------------------------------------------------------------
# Redis: the store's own EXPIRE on every key
# ---------------------------------------------------------------------------


@pytest.mark.requires_redis
@pytest.mark.skipif(not os.getenv("REDIS_URL"), reason="REDIS_URL not set")
class TestRedisNativeExpiry:
    @pytest.fixture
    async def redis_prefix(self) -> AsyncIterator[str]:
        from redis.asyncio import Redis

        prefix = f"locus:prod:{uuid.uuid4().hex[:8]}:"
        yield prefix
        client = Redis.from_url(os.environ["REDIS_URL"])
        try:
            keys = await client.keys(f"{prefix}*")
            if keys:
                await client.delete(*keys)
        finally:
            await client.aclose()

    async def test_every_key_a_thread_writes_carries_the_ttl(self, redis_prefix: str) -> None:
        from redis.asyncio import Redis

        from locus.memory.backends import redis_checkpointer

        cp = redis_checkpointer(
            url=os.environ["REDIS_URL"],
            prefix=redis_prefix,
            ttl=90,
            encryption=EncryptedSerializer(KEY),
            scope=SCOPE,
        )
        agent = _agent(cp)
        await agent.arun("first", thread_id="t")
        # The tool-effect ledger's sidecar goes through the same checkpointer.
        with SCOPE.active():
            await _record_calls(ToolEffectLedger(cp), "t", range(1, 3))
        client = Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
        try:
            keys = sorted(await client.keys(f"{redis_prefix}*"))
            assert keys, "the run wrote keys"
            assert all(k.startswith(f"{redis_prefix}app=billing/user=u-1/t") for k in keys), keys
            assert any(":_checkpoints:hash" in k for k in keys), "the index hash is there"
            assert any("~ledger" in k for k in keys), "the ledger sidecar is there"
            for key in keys:
                ttl = await client.ttl(key)
                assert 0 < ttl <= 90, (key, ttl)
            # The state at rest is ciphertext; the index fields are plain.
            raw = await client.get(f"{redis_prefix}app=billing/user=u-1/t:latest")
            assert raw is not None
            assert SENSITIVE not in raw
            document = json.loads(raw)
            assert document["alg"] == "A256GCM"
            assert "_expires_at" in document
            [hash_key] = [k for k in keys if k.endswith("t:_checkpoints:hash")]
            entries = [json.loads(v) for v in (await client.hgetall(hash_key)).values()]
            assert all(e["expires_at"] for e in entries)
            assert all(isinstance(e["iteration"], int) for e in entries)
        finally:
            await client.aclose()
        assert await cp.load("t") is not None
        assert (
            await redis_checkpointer(url=os.environ["REDIS_URL"], prefix=redis_prefix).load("t")
            is None
        ), "another scope (none) is a miss"
        await cp.delete("t")
        await cp.close()

    async def test_a_short_ttl_lets_the_store_drop_the_keys(self, redis_prefix: str) -> None:
        from redis.asyncio import Redis

        from locus.memory.backends import redis_checkpointer

        cp = redis_checkpointer(url=os.environ["REDIS_URL"], prefix=redis_prefix, ttl=1)
        await cp.save(AgentState(), "t", "cp-0")
        client = Redis.from_url(os.environ["REDIS_URL"])
        try:
            assert await client.keys(f"{redis_prefix}*")
            await asyncio.sleep(1.6)
            assert await client.keys(f"{redis_prefix}*") == [], "Redis expired them itself"
        finally:
            await client.aclose()
        assert await cp.load("t") is None
        assert await cp.expire() == 0, "nothing left for the sweep"
        await cp.close()

    async def test_mixed_ttl_writers_keep_the_longest_index_expiry(self, redis_prefix: str) -> None:
        """A short-lived save after a long-lived one must not take the thread's index with it."""
        from redis.asyncio import Redis

        from locus.memory.backends import redis_checkpointer

        long_lived = redis_checkpointer(url=os.environ["REDIS_URL"], prefix=redis_prefix, ttl=600)
        short_lived = redis_checkpointer(url=os.environ["REDIS_URL"], prefix=redis_prefix, ttl=1)
        forever = redis_checkpointer(url=os.environ["REDIS_URL"], prefix=redis_prefix)
        client = Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
        try:
            await long_lived.save(AgentState().with_iteration(1), "m", "cp-1")
            await short_lived.save(AgentState().with_iteration(2), "m", "cp-2")
            index = f"{redis_prefix}m:_checkpoints:hash"
            assert await client.ttl(index) > 500, "only ever lengthened"
            assert await client.ttl(f"{redis_prefix}m:cp-2") <= 1, "each checkpoint keeps its own"
            await asyncio.sleep(1.6)
            assert await long_lived.list_checkpoints("m") == ["cp-1"]
            fallen_back = await long_lived.load("m")
            assert fallen_back is not None
            assert fallen_back.iteration == 1, "latest expired: the newest live checkpoint"
            assert await forever.exists("m")
            # A writer without a ttl makes the shared index persist.
            await forever.save(AgentState().with_iteration(3), "m", "cp-3")
            assert await client.ttl(index) == -1
            await short_lived.save(AgentState().with_iteration(4), "m", "cp-4")
            assert await client.ttl(index) == -1, "a later ttl writer leaves it persisted"
        finally:
            await client.aclose()
            await long_lived.delete("m")
            for cp in (long_lived, short_lived, forever):
                await cp.close()


# ---------------------------------------------------------------------------
# PostgreSQL: ciphertext in the rows, the expires_at column, the scoped thread_id
# ---------------------------------------------------------------------------

_PG_TABLES = (
    "locus_prod_adapter",
    "locus_prod_native",
    "locus_prod_old",
    "locus_prod_nottl",
    "locus_prod_tenant",
)


@pytest.fixture(scope="module")
def _drop_postgres_tables() -> Iterator[None]:
    yield
    dsn = _postgres_dsn()
    if dsn is None:
        return
    import asyncpg

    async def drop() -> None:
        conn = await asyncpg.connect(dsn)
        try:
            for table in _PG_TABLES:
                await conn.execute(f"DROP TABLE IF EXISTS {table}")
        finally:
            await conn.close()

    asyncio.run(drop())


async def _rows(dsn: str, table: str, columns: str) -> list[Any]:
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        return list(await conn.fetch(f"SELECT {columns} FROM {table} ORDER BY thread_id"))  # noqa: S608
    finally:
        await conn.close()


#: The ttl the Postgres at-rest cases save with: long enough that a first save
#: which also creates the table does not expire before the asserts, short
#: enough to wait out.
SHORT_TTL = 1.5


@pytest.mark.requires_postgres
@pytest.mark.skipif(_postgres_dsn() is None, reason="POSTGRES_USER / POSTGRES_DB not set")
@pytest.mark.usefixtures("_drop_postgres_tables")
class TestPostgresAtRest:
    async def test_adapter_rows_hold_ciphertext_expiry_and_the_scoped_key(self) -> None:
        from locus.memory.backends import postgresql_checkpointer

        dsn = _postgres_dsn()
        assert dsn is not None
        # The first save on a fresh table runs the DDL under an advisory lock
        # and stamps the expiry before it, so a sub-second ttl can run out
        # under CI load before the asserts read the row back.
        cp = postgresql_checkpointer(
            dsn=dsn,
            table_name="locus_prod_adapter",
            ttl=SHORT_TTL,
            encryption=EncryptedSerializer(KEY),
            scope=SCOPE,
        )
        try:
            await cp.save(_secret_state().with_iteration(2), "t", "cp-0", metadata={"case": "1"})
            rows = await _rows(dsn, "locus_prod_adapter", "thread_id, data::text, expires_at")
            keys = {r["thread_id"] for r in rows}
            assert keys == {
                "app=billing/user=u-1/t:cp-0",
                "app=billing/user=u-1/t:latest",
                "app=billing/user=u-1/t:_checkpoints",
            }
            for row in rows:
                assert SENSITIVE not in row["data"]
                if not row["thread_id"].endswith(":_checkpoints"):
                    assert row["expires_at"] is not None, "the column, not just the document"
                    assert json.loads(row["data"])["alg"] == "A256GCM"
            [info] = await cp.list("t")
            assert (info.iteration, info.metadata) == (2, {"case": "1"})
            loaded = await cp.load("t")
            assert loaded is not None
            assert SENSITIVE in str(loaded.messages[0].content)
            other = postgresql_checkpointer(dsn=dsn, table_name="locus_prod_adapter")
            assert await other.load("t") is None, "no scope: a miss, not a leak"
            await other.close()

            await asyncio.sleep(SHORT_TTL * 1.5)
            assert await cp.load("t") is None
            assert await cp.expire() == 1
            left = await _rows(dsn, "locus_prod_adapter", "thread_id")
            assert [r["thread_id"] for r in left] == ["app=billing/user=u-1/t:_checkpoints"]
        finally:
            await cp.delete("t")
            await cp.close()

    async def test_without_a_ttl_an_old_table_gets_no_ddl(self) -> None:
        """The default path never alters a table it did not create; a ttl adds the column once."""
        import asyncpg

        from locus.memory.backends import postgresql_checkpointer

        dsn = _postgres_dsn()
        assert dsn is not None
        conn = await asyncpg.connect(dsn)
        try:
            await conn.execute(
                "CREATE TABLE locus_prod_nottl (thread_id TEXT PRIMARY KEY, checkpoint_id TEXT, "
                "data JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), "
                "updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), metadata JSONB DEFAULT '{}'::jsonb)"
            )
        finally:
            await conn.close()

        async def columns() -> set[str]:
            conn = await asyncpg.connect(dsn)
            try:
                rows = await conn.fetch(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'locus_prod_nottl'"
                )
            finally:
                await conn.close()
            return {r["column_name"] for r in rows}

        plain = postgresql_checkpointer(dsn=dsn, table_name="locus_prod_nottl")
        try:
            await plain.save(_secret_state(), "p", "cp-0")
            assert await plain.load("p") is not None
            assert await plain.exists("p")
            assert "expires_at" not in await columns(), "no DDL without a ttl"
            expiring = postgresql_checkpointer(dsn=dsn, table_name="locus_prod_nottl", ttl=60)
            await expiring.save(_secret_state(), "q", "cp-0")
            assert "expires_at" in await columns(), "the first save with a ttl adds it"
            assert await expiring.load("q") is not None
            await expiring.delete("q")
            await expiring.close()
        finally:
            await plain.delete("p")
            await plain.close()

    async def test_pg_checkpointer_tenant_of_places_a_view_write_and_a_direct_read_together(
        self,
    ) -> None:
        from locus.memory.backends import PgCheckpointer

        dsn = _postgres_dsn()
        assert dsn is not None
        seen: list[str] = []

        def tenant_of(thread_id: str) -> str:
            seen.append(thread_id)
            return "tenant-" + thread_id.split("-", maxsplit=1)[0]

        cp = PgCheckpointer(dsn, table="locus_prod_tenant", scope=SCOPE, tenant_of=tenant_of)
        try:
            await cp.save(_secret_state().with_iteration(1), "order-1", "cp-d")
            view = cp.with_scope(SCOPE)
            await view.save(_secret_state().with_iteration(2), "order-1", "cp-v")
            assert set(seen) == {"order-1"}
            assert await cp.load("order-1", "cp-v") is not None, (
                "the direct read sees the view's save"
            )
            assert await view.load("order-1", "cp-d") is not None
            rows = await _rows(dsn, "locus_prod_tenant", "tenant, thread_id, checkpoint_id")
            assert {(r["tenant"], r["thread_id"]) for r in rows} == {
                ("tenant-order", "app=billing/user=u-1/order-1")
            }
        finally:
            await cp.delete("order-1")
            await cp.close()

    async def test_pg_checkpointer_rows_and_the_column_added_to_an_old_table(self) -> None:
        import asyncpg

        from locus.memory.backends import PgCheckpointer

        dsn = _postgres_dsn()
        assert dsn is not None
        # A table from before checkpoints could expire: the layout, minus the column.
        conn = await asyncpg.connect(dsn)
        try:
            probe = PgCheckpointer(dsn, table="locus_prod_old")
            for statement in probe.ddl():
                if "_expires" in statement:
                    continue  # the index on the column that is not there
                await conn.execute(statement.replace("expires_at timestamptz, ", ""))
            columns = {
                r["column_name"]
                for r in await conn.fetch(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'locus_prod_old'"
                )
            }
            assert "expires_at" not in columns
        finally:
            await conn.close()

        cp = PgCheckpointer(
            dsn,
            table="locus_prod_old",
            ttl=SHORT_TTL,  # see test_adapter_rows_hold_ciphertext_expiry_and_the_scoped_key
            encryption=EncryptedSerializer(KEY),
            scope=SCOPE,
        )
        try:
            await cp.save(_secret_state().with_iteration(1), "t", "cp-0")
            rows = await _rows(dsn, "locus_prod_old", "thread_id, data::text, expires_at")
            assert [r["thread_id"] for r in rows] == ["app=billing/user=u-1/t"]
            assert SENSITIVE not in rows[0]["data"]
            assert rows[0]["expires_at"] is not None, "the column was added on first use"
            [info] = await cp.list("t")
            assert info.iteration == 1
            plain = PgCheckpointer(dsn, table="locus_prod_old")
            assert await plain.load("t") is None
            assert await plain.list_threads() == []
            assert await plain.list_threads(scope=SCOPE) == ["t"]
            await plain.close()
            await asyncio.sleep(SHORT_TTL * 1.5)
            assert await cp.load("t") is None
            assert await cp.expire() == 1
        finally:
            await cp.delete("t")
            await cp.close()
