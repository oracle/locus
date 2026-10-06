# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""PgCheckpointer — conversation threads in Postgres, one tenant's rows apart from another's.

The checkpointer counterpart of :class:`~tulip.memory.store_backends.postgresql.PgMemory`:
the same tenant boundary, enforced the same way.

* **Every row carries its tenant** and Row-Level Security admits only the
  tenant pinned for the transaction (``tulip.tenant``, the GUC ``PgMemory``
  uses, so one role setting serves both), on read AND write. ``FORCE`` applies
  it to the table owner too, and every query *also* filters on the tenant.
* **One statement per save.** The checkpoint is upserted and, with
  ``keep_checkpoints``, the thread's older checkpoints are pruned in the same
  statement — instead of the four round trips the generic
  ``postgresql_checkpointer()`` adapter makes (the checkpoint, a ``latest``
  copy, and a read-modify-write of an index row).
* **Retention per message.** With ``message_retention`` each save drops the
  exchanges older than that (see :mod:`tulip.memory.retention`), so a thread
  that is used every day stops growing; :meth:`PgCheckpointer.purge_messages`
  does the same for threads nobody is writing to, and :meth:`vacuum` deletes
  threads idle past a cut-off.
* **No DDL when the table exists.** The schema is probed first; ``CREATE`` runs
  only for what is missing, so an application role with no ``CREATE`` on the
  database (or the schema) works against a table a migration made.

Which tenant a call belongs to, in order:

1. a :meth:`PgCheckpointer.tenant_scope` block around the call (context-local,
   so concurrent runs each keep their own);
2. ``tenant_of(thread_id)``, for thread ids that carry the tenant;
3. ``tenant=``, fixed for the instance (``"default"`` when nothing else is
   given — a single-tenant install needs no configuration).

A ``tenant_of`` that raises or returns nothing fails the call: it is never
silently filed under another tenant.
"""

from __future__ import annotations

import asyncio
import contextvars
import importlib.util
import json
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from tulip.core.loop_bound import loop_bound_async
from tulip.core.protocols import CheckpointerCapabilities
from tulip.memory.checkpointer import BaseCheckpointer
from tulip.memory.retention import oldest_message_time, stamp_messages, trim_messages


if TYPE_CHECKING:
    from asyncpg import Pool

    from tulip.core.state import AgentState


__all__ = ["PgCheckpointer", "TENANT_GUC"]

#: The GUC that carries the active tenant to the RLS policy — the one
#: ``PgMemory`` uses. ``set_config(..., true)`` scopes it to the transaction,
#: so a pooled connection never hands one tenant's id to the next checkout.
TENANT_GUC = "tulip.tenant"

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")

#: The columns this layout needs; an existing table without them is refused.
_COLUMNS = frozenset(
    {"tenant", "thread_id", "checkpoint_id", "seq", "data", "metadata", "created_at", "updated_at"}
)


def _ident(value: str, field: str) -> str:
    """Guard an identifier interpolated into SQL (identifiers cannot be bound)."""
    if not _IDENT.match(value):
        raise ValueError(f"invalid {field}: {value!r}")
    return value


def _count(status: str) -> int:
    """The row count of an asyncpg status line (``"DELETE 3"``)."""
    try:
        return int(status.rsplit(maxsplit=1)[-1])
    except (IndexError, ValueError):
        return 0


class PgCheckpointer(BaseCheckpointer):
    """Agent threads in Postgres, isolated per tenant by Row-Level Security.

    Args:
        dsn: A libpq connection string. Its role should not be a superuser or
            have ``BYPASSRLS``: those skip Row-Level Security altogether.
        table: The checkpoint table.
        schema: The schema holding it.
        tenant: The tenant of calls that no scope or ``tenant_of`` places.
        tenant_of: ``(thread_id) -> tenant``, for thread ids that carry it.
        keep_checkpoints: Checkpoints kept per thread, newest first; older
            ones are deleted by the save that makes them surplus. ``None``
            keeps all of them (every save is a restore point).
        message_retention: Drop exchanges older than this at every save
            (:mod:`tulip.memory.retention`). ``None`` keeps every message; they
            are stamped regardless, so :meth:`purge_messages` has times to go by.
        create_schema: Create the schema, table, index and policy when they
            are missing. ``False`` never runs DDL: the table must exist (made by
            a migration), and is checked for the expected columns.
        min_pool_size / max_pool_size: The asyncpg pool bounds.

    Example::

        cp = PgCheckpointer(
            dsn,
            table="agent_threads",
            keep_checkpoints=1,
            message_retention=timedelta(days=30),
        )
        agent = Agent(model=model, checkpointer=cp)

        with cp.tenant_scope("acme"):
            await agent.arun("hello", thread_id="support/42")
    """

    def __init__(
        self,
        dsn: str,
        *,
        table: str = "tulip_checkpoints",
        schema: str = "public",
        tenant: str = "default",
        tenant_of: Callable[[str], str] | None = None,
        keep_checkpoints: int | None = None,
        message_retention: timedelta | None = None,
        create_schema: bool = True,
        min_pool_size: int = 1,
        max_pool_size: int = 10,
    ) -> None:
        if importlib.util.find_spec("asyncpg") is None:
            raise ImportError(
                "PgCheckpointer needs the asyncpg driver, which ships as an optional "
                "dependency. Install it with:\n\n"
                "    pip install 'tulip-agents[postgresql]'\n"
            )
        if not tenant:
            raise ValueError("tenant must be a non-empty string")
        if keep_checkpoints is not None and keep_checkpoints < 1:
            raise ValueError("keep_checkpoints must be at least 1 (or None to keep all)")
        if message_retention is not None and message_retention <= timedelta(0):
            raise ValueError("message_retention must be positive")
        self._dsn = dsn
        self._schema = _ident(schema, "schema")
        self._table_name = _ident(table, "table")
        self._table = f"{self._schema}.{self._table_name}"
        self._tenant = tenant
        self._tenant_of = tenant_of
        self.keep_checkpoints = keep_checkpoints
        self.message_retention = message_retention
        self._create_schema = create_schema
        self._min_pool = min_pool_size
        self._max_pool = max_pool_size
        self._pool: Pool | None = None
        self._pool_lock = asyncio.Lock()
        self._scope: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            f"tulip_pg_checkpointer_tenant_{id(self)}", default=None
        )

    # ------------------------------------------------------------------
    # Tenancy
    # ------------------------------------------------------------------

    @contextmanager
    def tenant_scope(self, tenant: str) -> Iterator[None]:
        """File every call inside the block under ``tenant``.

        Context-local: an agent run started inside the block (and any task it
        spawns) keeps the tenant; a concurrent run outside it does not see it.
        """
        if not tenant:
            raise ValueError("tenant must be a non-empty string")
        token = self._scope.set(tenant)
        try:
            yield
        finally:
            self._scope.reset(token)

    def tenant_for(self, thread_id: str | None = None) -> str:
        """The tenant a call on ``thread_id`` belongs to (see the module docs)."""
        scoped = self._scope.get()
        if scoped:
            return scoped
        if self._tenant_of is not None and thread_id is not None:
            try:
                found = self._tenant_of(thread_id)
            except Exception as exc:
                raise ValueError(f"tenant_of could not place thread {thread_id!r}") from exc
            if not found:
                raise ValueError(f"tenant_of placed thread {thread_id!r} in no tenant")
            return str(found)
        return self._tenant

    # ------------------------------------------------------------------
    # Pool and schema
    # ------------------------------------------------------------------

    async def _get_pool(self) -> Pool:
        """The pool, built (and the schema checked or created) on first use per loop."""

        async def build() -> Pool:
            import asyncpg  # noqa: PLC0415

            pool = await asyncpg.create_pool(
                self._dsn, min_size=self._min_pool, max_size=self._max_pool
            )
            try:
                await self._ensure_schema(pool)
            except BaseException:
                await pool.close()
                raise
            return pool

        async with self._pool_lock:
            return await loop_bound_async(self, "_pool", build)

    async def _ensure_schema(self, pool: Pool) -> None:
        """Probe first; create only what is missing (never needs CREATE when it all exists)."""
        async with pool.acquire() as conn:
            columns = {
                r["column_name"]
                for r in await conn.fetch(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = $1 AND table_name = $2",
                    self._schema,
                    self._table_name,
                )
            }
            if columns:
                missing = _COLUMNS - columns
                if missing:
                    raise RuntimeError(
                        f"PgCheckpointer: table {self._table} exists without the columns "
                        f"{sorted(missing)}. It looks like another layout (the generic "
                        f"postgresql_checkpointer() keeps one row per key with no tenant). "
                        f"Pass a new table= or migrate it."
                    )
                return
            if not self._create_schema:
                raise RuntimeError(
                    f"PgCheckpointer: table {self._table} does not exist and "
                    f"create_schema=False. Create it with a migration (see "
                    f"PgCheckpointer.ddl()) or allow create_schema."
                )
            for statement in self.ddl():
                await conn.execute(statement)

    def ddl(self) -> list[str]:
        """The statements that create this checkpointer's schema, for a migration."""
        t, name = self._table, self._table_name
        schema_exists = (
            [] if self._schema == "public" else [f"CREATE SCHEMA IF NOT EXISTS {self._schema}"]
        )
        return [
            *schema_exists,
            f"CREATE TABLE IF NOT EXISTS {t} ("
            "tenant text NOT NULL, thread_id text NOT NULL, checkpoint_id text NOT NULL, "
            "seq bigint GENERATED BY DEFAULT AS IDENTITY, data jsonb NOT NULL, "
            "metadata jsonb NOT NULL DEFAULT '{}'::jsonb, oldest_at timestamptz, "
            "created_at timestamptz NOT NULL DEFAULT now(), "
            "updated_at timestamptz NOT NULL DEFAULT now(), "
            "PRIMARY KEY (tenant, thread_id, checkpoint_id))",
            f"CREATE INDEX IF NOT EXISTS idx_{name}_latest ON {t} (tenant, thread_id, seq DESC)",
            f"CREATE INDEX IF NOT EXISTS idx_{name}_oldest ON {t} (tenant, oldest_at)",
            f"ALTER TABLE {t} ENABLE ROW LEVEL SECURITY",
            f"ALTER TABLE {t} FORCE ROW LEVEL SECURITY",
            f"DROP POLICY IF EXISTS tenant_isolation ON {t}",
            f"CREATE POLICY tenant_isolation ON {t} "
            f"USING (tenant = current_setting('{TENANT_GUC}', true)) "
            f"WITH CHECK (tenant = current_setting('{TENANT_GUC}', true))",
        ]

    async def _pinned(self, conn: Any, tenant: str) -> None:
        await conn.execute(f"SELECT set_config('{TENANT_GUC}', $1, true)", tenant)

    @property
    def capabilities(self) -> CheckpointerCapabilities:
        return CheckpointerCapabilities(
            metadata_query=True,
            vacuum=True,
            branching=True,
            list_threads=True,
            persistent_checkpoint_ids=True,
        )

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def prepare(self, state: AgentState, now: datetime | None = None) -> AgentState:
        """``state`` as a save writes it: messages stamped, and trimmed to ``message_retention``."""
        now = now or datetime.now(UTC)
        state = stamp_messages(state, now)
        if self.message_retention is not None:
            state, _ = trim_messages(state, now - self.message_retention)
        return state

    async def save(
        self,
        state: AgentState,
        thread_id: str,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Upsert the checkpoint and prune the thread, in one statement."""
        tenant = self.tenant_for(thread_id)
        checkpoint_id = checkpoint_id or uuid4().hex
        state = self.prepare(state)
        args: list[Any] = [
            tenant,
            thread_id,
            checkpoint_id,
            json.dumps(state.to_checkpoint()),
            json.dumps(metadata or {}),
            oldest_message_time(state),
        ]
        upsert = (
            f"INSERT INTO {self._table} "
            "(tenant, thread_id, checkpoint_id, data, metadata, oldest_at) "
            "VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6) "
            "ON CONFLICT (tenant, thread_id, checkpoint_id) DO UPDATE SET "
            "data = EXCLUDED.data, metadata = EXCLUDED.metadata, "
            "oldest_at = EXCLUDED.oldest_at, seq = DEFAULT, updated_at = now()"
        )
        if self.keep_checkpoints is None:
            sql = upsert
        else:
            # The DELETE reads the snapshot from before the INSERT, so it sees
            # the thread's other checkpoints only: keep the newest keep-1 of
            # them, plus the one being written.
            sql = (
                f"WITH saved AS ({upsert} RETURNING 1) "
                f"DELETE FROM {self._table} WHERE tenant = $1 AND thread_id = $2 "
                "AND checkpoint_id IN ("
                f"SELECT checkpoint_id FROM {self._table} "
                "WHERE tenant = $1 AND thread_id = $2 AND checkpoint_id <> $3 "
                "ORDER BY seq DESC OFFSET $7)"
            )
            args.append(self.keep_checkpoints - 1)
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            await self._pinned(conn, tenant)
            await conn.execute(sql, *args)
        return checkpoint_id

    async def load(self, thread_id: str, checkpoint_id: str | None = None) -> AgentState | None:
        from tulip.core.state import AgentState  # noqa: PLC0415

        tenant = self.tenant_for(thread_id)
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            await self._pinned(conn, tenant)
            if checkpoint_id is None:
                raw = await conn.fetchval(
                    f"SELECT data FROM {self._table} WHERE tenant = $1 AND thread_id = $2 "
                    "ORDER BY seq DESC LIMIT 1",
                    tenant,
                    thread_id,
                )
            else:
                raw = await conn.fetchval(
                    f"SELECT data FROM {self._table} "
                    "WHERE tenant = $1 AND thread_id = $2 AND checkpoint_id = $3",
                    tenant,
                    thread_id,
                    checkpoint_id,
                )
        if raw is None:
            return None
        return AgentState.from_checkpoint(json.loads(raw) if isinstance(raw, str) else raw)

    async def list_checkpoints(self, thread_id: str, limit: int = 10) -> list[str]:
        tenant = self.tenant_for(thread_id)
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            await self._pinned(conn, tenant)
            rows = await conn.fetch(
                f"SELECT checkpoint_id FROM {self._table} WHERE tenant = $1 AND thread_id = $2 "
                "ORDER BY seq DESC LIMIT $3",
                tenant,
                thread_id,
                limit,
            )
        return [r["checkpoint_id"] for r in rows]

    async def delete(self, thread_id: str, checkpoint_id: str | None = None) -> bool:
        tenant = self.tenant_for(thread_id)
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            await self._pinned(conn, tenant)
            if checkpoint_id is None:
                status = await conn.execute(
                    f"DELETE FROM {self._table} WHERE tenant = $1 AND thread_id = $2",
                    tenant,
                    thread_id,
                )
            else:
                status = await conn.execute(
                    f"DELETE FROM {self._table} "
                    "WHERE tenant = $1 AND thread_id = $2 AND checkpoint_id = $3",
                    tenant,
                    thread_id,
                    checkpoint_id,
                )
        return _count(status) > 0

    async def exists(self, thread_id: str, checkpoint_id: str | None = None) -> bool:
        if checkpoint_id is None:
            return bool(await self.list_checkpoints(thread_id, limit=1))
        tenant = self.tenant_for(thread_id)
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            await self._pinned(conn, tenant)
            found = await conn.fetchval(
                f"SELECT 1 FROM {self._table} "
                "WHERE tenant = $1 AND thread_id = $2 AND checkpoint_id = $3",
                tenant,
                thread_id,
                checkpoint_id,
            )
        return found is not None

    # ------------------------------------------------------------------
    # Extended
    # ------------------------------------------------------------------

    async def get_metadata(
        self, thread_id: str, checkpoint_id: str | None = None
    ) -> dict[str, Any] | None:
        tenant = self.tenant_for(thread_id)
        cols = "checkpoint_id, metadata, created_at, updated_at"
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            await self._pinned(conn, tenant)
            if checkpoint_id is None:
                row = await conn.fetchrow(
                    f"SELECT {cols} FROM {self._table} WHERE tenant = $1 AND thread_id = $2 "
                    "ORDER BY seq DESC LIMIT 1",
                    tenant,
                    thread_id,
                )
            else:
                row = await conn.fetchrow(
                    f"SELECT {cols} FROM {self._table} "
                    "WHERE tenant = $1 AND thread_id = $2 AND checkpoint_id = $3",
                    tenant,
                    thread_id,
                    checkpoint_id,
                )
        if row is None:
            return None
        meta = row["metadata"]
        return {
            "checkpoint_id": row["checkpoint_id"],
            "metadata": json.loads(meta) if isinstance(meta, str) else dict(meta or {}),
            "created_at": row["created_at"].isoformat(),
            "updated_at": row["updated_at"].isoformat(),
        }

    async def query_by_metadata(
        self, key: str, value: Any, limit: int = 100
    ) -> list[dict[str, Any]]:
        """The current tenant's latest checkpoints whose metadata has ``key: value``."""
        tenant = self.tenant_for()
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            await self._pinned(conn, tenant)
            rows = await conn.fetch(
                f"SELECT thread_id, checkpoint_id, updated_at FROM {self._table} "
                "WHERE tenant = $1 AND metadata @> $2::jsonb ORDER BY seq DESC LIMIT $3",
                tenant,
                json.dumps({key: value}),
                limit,
            )
        return [
            {
                "thread_id": r["thread_id"],
                "checkpoint_id": r["checkpoint_id"],
                "updated_at": r["updated_at"].isoformat(),
            }
            for r in rows
        ]

    async def list_threads(self, limit: int = 100, pattern: str = "*") -> list[str]:
        """The current tenant's threads, most recently saved first.

        ``pattern`` is a glob (``*`` and ``?``), matched in SQL.
        """
        tenant = self.tenant_for()
        like = (
            pattern.replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
            .replace("*", "%")
            .replace("?", "_")
        )
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            await self._pinned(conn, tenant)
            rows = await conn.fetch(
                f"SELECT thread_id FROM {self._table} WHERE tenant = $1 AND thread_id LIKE $2 "
                "GROUP BY thread_id ORDER BY max(seq) DESC LIMIT $3",
                tenant,
                like,
                limit,
            )
        return [r["thread_id"] for r in rows]

    async def copy_thread(self, source_thread_id: str, dest_thread_id: str) -> bool:
        """Copy every checkpoint of a thread to another thread of the same tenant."""
        tenant = self.tenant_for(source_thread_id)
        if self.tenant_for(dest_thread_id) != tenant:
            raise ValueError("copy_thread cannot copy a thread to another tenant")
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            await self._pinned(conn, tenant)
            status = await conn.execute(
                f"INSERT INTO {self._table} "
                "(tenant, thread_id, checkpoint_id, data, metadata, oldest_at) "
                "SELECT tenant, $3, checkpoint_id, data, metadata, oldest_at "
                f"FROM {self._table} WHERE tenant = $1 AND thread_id = $2 ORDER BY seq "
                "ON CONFLICT (tenant, thread_id, checkpoint_id) DO NOTHING",
                tenant,
                source_thread_id,
                dest_thread_id,
            )
        return _count(status) > 0

    async def vacuum(self, older_than_days: int = 30) -> int:
        """Delete the current tenant's threads nobody has saved to for ``older_than_days``.

        Whole threads, every checkpoint in them; an active thread is left alone
        however old its first checkpoint is (that is what ``message_retention``
        and :meth:`purge_messages` are for). Returns the checkpoints deleted.
        """
        tenant = self.tenant_for()
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            await self._pinned(conn, tenant)
            status = await conn.execute(
                f"DELETE FROM {self._table} WHERE tenant = $1 AND thread_id IN ("
                f"SELECT thread_id FROM {self._table} WHERE tenant = $1 GROUP BY thread_id "
                "HAVING max(updated_at) < now() - make_interval(days => $2))",
                tenant,
                older_than_days,
            )
        return _count(status)

    async def purge_messages(
        self, older_than: timedelta, *, thread_id: str | None = None, batch: int = 500
    ) -> int:
        """Drop exchanges older than ``older_than`` from the current tenant's checkpoints.

        For the threads nobody is writing to, which a save never trims: every
        checkpoint holding an exchange older than the cut-off is rewritten in
        place, keeping its id and its place in the thread. Rows are locked
        while they are rewritten, so a run saving at the same time is not
        lost. Returns the messages dropped.

        Args:
            older_than: The age past which an exchange is dropped.
            thread_id: Only this thread (else every thread of the tenant).
            batch: Checkpoints rewritten per transaction.
        """
        from tulip.core.state import AgentState  # noqa: PLC0415

        if older_than <= timedelta(0):
            raise ValueError("older_than must be positive")
        tenant = self.tenant_for(thread_id)
        cutoff = datetime.now(UTC) - older_than
        where = "tenant = $1 AND oldest_at < $2" + (" AND thread_id = $4" if thread_id else "")
        dropped = 0
        pool = await self._get_pool()
        while True:
            async with pool.acquire() as conn, conn.transaction():
                await self._pinned(conn, tenant)
                rows = await conn.fetch(
                    f"SELECT thread_id, checkpoint_id, data FROM {self._table} WHERE {where} "
                    "ORDER BY oldest_at LIMIT $3 FOR UPDATE SKIP LOCKED",
                    tenant,
                    cutoff,
                    batch,
                    *([thread_id] if thread_id else []),
                )
                updates = []
                for r in rows:
                    raw = r["data"]
                    state = AgentState.from_checkpoint(
                        json.loads(raw) if isinstance(raw, str) else raw
                    )
                    trimmed, n = trim_messages(state, cutoff)
                    dropped += n
                    # Unstamped or recent exchanges ahead of the old ones stop
                    # the trim; the row's oldest_at moves on regardless, so a
                    # row is visited once per cut-off, not forever.
                    oldest = oldest_message_time(trimmed)
                    updates.append(
                        (
                            tenant,
                            r["thread_id"],
                            r["checkpoint_id"],
                            json.dumps(trimmed.to_checkpoint()),
                            oldest if oldest is not None and oldest >= cutoff else None,
                        )
                    )
                if updates:
                    await conn.executemany(
                        f"UPDATE {self._table} SET data = $4::jsonb, oldest_at = $5 "
                        "WHERE tenant = $1 AND thread_id = $2 AND checkpoint_id = $3",
                        updates,
                    )
            if len(rows) < batch:
                return dropped

    async def forget_tenant(self) -> int:
        """Delete every checkpoint of the current tenant (a company leaving); returns the count."""
        tenant = self.tenant_for()
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            await self._pinned(conn, tenant)
            status = await conn.execute(f"DELETE FROM {self._table} WHERE tenant = $1", tenant)
        return _count(status)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def __repr__(self) -> str:
        return (
            f"PgCheckpointer(table={self._table!r}, keep_checkpoints={self.keep_checkpoints!r}, "
            f"message_retention={self.message_retention!r})"
        )
