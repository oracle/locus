# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""PgCheckpointer — conversation threads in Postgres, one tenant's rows apart from another's.

The checkpointer counterpart of :class:`~locus.memory.store_backends.postgresql.PgMemory`:
the same tenant boundary, enforced the same way.

* **Every row carries its tenant** and Row-Level Security admits only the
  tenant pinned for the transaction (``locus.tenant``, the GUC ``PgMemory``
  uses, so one role setting serves both), on read AND write. ``FORCE`` applies
  it to the table owner too, and every query *also* filters on the tenant.
* **One statement per save.** The checkpoint is upserted and, with
  ``keep_checkpoints``, the thread's older checkpoints are pruned in the same
  statement — instead of the four round trips the generic
  ``postgresql_checkpointer()`` adapter makes (the checkpoint, a ``latest``
  copy, and a read-modify-write of an index row).
* **Retention per message.** With ``message_retention`` each save drops the
  exchanges older than that (see :mod:`locus.memory.retention`), so a thread
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

from locus.core.loop_bound import loop_bound_async
from locus.core.protocols import CheckpointerCapabilities
from locus.memory.checkpointer import BaseCheckpointer
from locus.memory.ledger import LEDGER_SUFFIX
from locus.memory.retention import oldest_message_time, stamp_messages, trim_messages
from locus.memory.scope import StoredThreadId
from locus.memory.time_travel import CheckpointInfo, info_from, link_parents


if TYPE_CHECKING:
    from asyncpg import Pool

    from locus.core.state import AgentState
    from locus.memory.encryption import EncryptedSerializer
    from locus.memory.scope import CheckpointScope


__all__ = ["PgCheckpointer", "TENANT_GUC"]

#: The GUC that carries the active tenant to the RLS policy — the one
#: ``PgMemory`` uses. ``set_config(..., true)`` scopes it to the transaction,
#: so a pooled connection never hands one tenant's id to the next checkout.
TENANT_GUC = "locus.tenant"

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")

#: The columns this layout needs; an existing table without them is refused.
_COLUMNS = frozenset(
    {"tenant", "thread_id", "checkpoint_id", "seq", "data", "metadata", "created_at", "updated_at"}
)
#: The column a ``ttl`` writes; added to an existing table on first use.
_EXPIRES_COLUMN = "expires_at"
#: The filter every read applies: rows whose expiry has not passed (or that have none).
_LIVE = "AND (expires_at IS NULL OR expires_at > now())"


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
            keeps all of them (every save is a restore point). The
            tool-effect ledger's sidecar thread is never pruned.
        message_retention: Drop exchanges older than this at every save
            (:mod:`locus.memory.retention`). ``None`` keeps every message; they
            are stamped regardless, so :meth:`purge_messages` has times to go by.
        create_schema: Create the schema, table, index and policy when they
            are missing. ``False`` never runs DDL: the table must exist (made by
            a migration), and is checked for the expected columns.
        min_pool_size / max_pool_size: The asyncpg pool bounds.
        ttl: How long a checkpoint stays loadable after its save (seconds or
            a ``timedelta``), kept in the row's ``expires_at`` column (added
            to an existing table with ``ALTER TABLE ... ADD COLUMN IF NOT
            EXISTS`` on first use); every read filters on it and
            :meth:`expire` deletes the rows whose time has passed.
        encryption: Seals the state in the ``data`` column
            (:class:`~locus.memory.encryption.EncryptedSerializer`); the
            other columns stay plaintext.
        scope: The application and user the threads belong to
            (:class:`~locus.memory.scope.CheckpointScope`), stored as the
            prefix of the ``thread_id`` column; the tenant stays the
            Row-Level Security boundary.

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
        table: str = "locus_checkpoints",
        schema: str = "public",
        tenant: str = "default",
        tenant_of: Callable[[str], str] | None = None,
        keep_checkpoints: int | None = None,
        message_retention: timedelta | None = None,
        create_schema: bool = True,
        min_pool_size: int = 1,
        max_pool_size: int = 10,
        ttl: float | timedelta | None = None,
        encryption: EncryptedSerializer | None = None,
        scope: CheckpointScope | None = None,
    ) -> None:
        if importlib.util.find_spec("asyncpg") is None:
            raise ImportError(
                "PgCheckpointer needs the asyncpg driver, which ships as an optional "
                "dependency. Install it with:\n\n"
                "    pip install 'locus-sdk[postgresql]'\n"
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
        self._configure(ttl=ttl, encryption=encryption, scope=scope)
        self._create_schema = create_schema
        self._min_pool = min_pool_size
        self._max_pool = max_pool_size
        self._pool: Pool | None = None
        self._pool_lock = asyncio.Lock()
        # Whether the table has the ``expires_at`` column: a table from before
        # checkpoints could expire goes without it until a ``ttl`` needs it.
        self._has_expires = True
        self._scope: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            f"locus_pg_checkpointer_tenant_{id(self)}", default=None
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
        """The tenant a call on ``thread_id`` belongs to (see the module docs).

        ``tenant_of`` is given the id the caller named, never the stored
        form a scope makes of it, so a save through a scoped run and a
        direct read of the same thread place it in the same tenant.
        """
        scoped = self._scope.get()
        if scoped:
            return scoped
        if self._tenant_of is not None and thread_id is not None:
            named = thread_id.thread_id if isinstance(thread_id, StoredThreadId) else thread_id
            try:
                found = self._tenant_of(named)
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
                self._has_expires = _EXPIRES_COLUMN in columns
                if not self._has_expires and self.ttl is not None:
                    # A table from before checkpoints could expire, and a ttl
                    # that needs the column: add it, where DDL is allowed.
                    if not self._create_schema:
                        raise RuntimeError(
                            f"PgCheckpointer: table {self._table} has no {_EXPIRES_COLUMN} "
                            f"column and create_schema=False. Add it with a migration: "
                            f"{self._expires_ddl()}"
                        )
                    await conn.execute(self._expires_ddl())
                    self._has_expires = True
                return
            if not self._create_schema:
                raise RuntimeError(
                    f"PgCheckpointer: table {self._table} does not exist and "
                    f"create_schema=False. Create it with a migration (see "
                    f"PgCheckpointer.ddl()) or allow create_schema."
                )
            # Serialise concurrent first-time creators (see PostgreSQLBackend).
            async with conn.transaction():
                await conn.fetchval("SELECT pg_advisory_xact_lock(hashtext($1))", self._table)
                already = await conn.fetchval(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_schema = $1 AND table_name = $2 LIMIT 1",
                    self._schema,
                    self._table_name,
                )
                if already is None:
                    for statement in self.ddl():
                        await conn.execute(statement)
            # Whoever created it ran this release's DDL, which carries the column.
            self._has_expires = True

    def _expires_ddl(self) -> str:
        """The statement that adds the ``expires_at`` column to an existing table."""
        return f"ALTER TABLE {self._table} ADD COLUMN IF NOT EXISTS {_EXPIRES_COLUMN} timestamptz"

    @property
    def _live(self) -> str:
        """The expiry filter every read applies; nothing on a table without the column."""
        return _LIVE if self._has_expires else ""

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
            f"{_EXPIRES_COLUMN} timestamptz, "
            "PRIMARY KEY (tenant, thread_id, checkpoint_id))",
            f"CREATE INDEX IF NOT EXISTS idx_{name}_latest ON {t} (tenant, thread_id, seq DESC)",
            f"CREATE INDEX IF NOT EXISTS idx_{name}_oldest ON {t} (tenant, oldest_at)",
            f"CREATE INDEX IF NOT EXISTS idx_{name}_expires ON {t} (tenant, {_EXPIRES_COLUMN})",
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
        key = self._scoped(thread_id)
        checkpoint_id = checkpoint_id or uuid4().hex
        state = self.prepare(state)
        pool = await self._get_pool()  # first: the schema probe decides the columns
        args: list[Any] = [
            tenant,
            key,
            checkpoint_id,
            json.dumps(self._encode_state(state, key, checkpoint_id)),
            json.dumps(metadata or {}),
            oldest_message_time(state),
        ]
        columns, values, updates = "", "", ""
        if self._has_expires:
            args.append(self._expires_at())
            columns, values = ", expires_at", ", $7"
            updates = ", expires_at = EXCLUDED.expires_at"
        upsert = (
            f"INSERT INTO {self._table} "
            f"(tenant, thread_id, checkpoint_id, data, metadata, oldest_at{columns}) "
            f"VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6{values}) "
            "ON CONFLICT (tenant, thread_id, checkpoint_id) DO UPDATE SET "
            "data = EXCLUDED.data, metadata = EXCLUDED.metadata, "
            f"oldest_at = EXCLUDED.oldest_at{updates}, seq = DEFAULT, updated_at = now()"
        )
        if self.keep_checkpoints is None or thread_id.endswith(LEDGER_SUFFIX):
            sql = upsert
        else:
            # The DELETE reads the snapshot from before the INSERT, so it sees
            # the thread's other checkpoints only: keep the newest keep-1 of
            # them, plus the one being written.
            args.append(self.keep_checkpoints - 1)
            sql = (
                f"WITH saved AS ({upsert} RETURNING 1) "
                f"DELETE FROM {self._table} WHERE tenant = $1 AND thread_id = $2 "
                "AND checkpoint_id IN ("
                f"SELECT checkpoint_id FROM {self._table} "
                "WHERE tenant = $1 AND thread_id = $2 AND checkpoint_id <> $3 "
                f"ORDER BY seq DESC OFFSET ${len(args)})"
            )
        async with pool.acquire() as conn, conn.transaction():
            await self._pinned(conn, tenant)
            await conn.execute(sql, *args)
        return checkpoint_id

    async def load(self, thread_id: str, checkpoint_id: str | None = None) -> AgentState | None:
        tenant = self.tenant_for(thread_id)
        key = self._scoped(thread_id)
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            await self._pinned(conn, tenant)
            if checkpoint_id is None:
                row = await conn.fetchrow(
                    f"SELECT checkpoint_id, data FROM {self._table} "
                    f"WHERE tenant = $1 AND thread_id = $2 {self._live} ORDER BY seq DESC LIMIT 1",
                    tenant,
                    key,
                )
            else:
                row = await conn.fetchrow(
                    f"SELECT checkpoint_id, data FROM {self._table} "
                    f"WHERE tenant = $1 AND thread_id = $2 AND checkpoint_id = $3 {self._live}",
                    tenant,
                    key,
                    checkpoint_id,
                )
        if row is None:
            return None
        raw = row["data"]
        return self._decode_state(
            json.loads(raw) if isinstance(raw, str) else raw, key, str(row["checkpoint_id"])
        )

    async def list_checkpoints(self, thread_id: str, limit: int = 10) -> list[str]:
        tenant = self.tenant_for(thread_id)
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            await self._pinned(conn, tenant)
            rows = await conn.fetch(
                f"SELECT checkpoint_id FROM {self._table} WHERE tenant = $1 AND thread_id = $2 "
                f"{self._live} ORDER BY seq DESC LIMIT $3",
                tenant,
                self._scoped(thread_id),
                limit,
            )
        return [r["checkpoint_id"] for r in rows]

    async def delete(self, thread_id: str, checkpoint_id: str | None = None) -> bool:
        tenant = self.tenant_for(thread_id)
        key = self._scoped(thread_id)
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            await self._pinned(conn, tenant)
            if checkpoint_id is None:
                status = await conn.execute(
                    f"DELETE FROM {self._table} WHERE tenant = $1 AND thread_id = $2",
                    tenant,
                    key,
                )
            else:
                status = await conn.execute(
                    f"DELETE FROM {self._table} "
                    "WHERE tenant = $1 AND thread_id = $2 AND checkpoint_id = $3",
                    tenant,
                    key,
                    checkpoint_id,
                )
        return _count(status) > 0

    async def expire(self) -> int:
        """Delete the current tenant's checkpoints whose ``expires_at`` has passed; returns the count.

        A read never returns an expired row whether or not this has run.
        """
        tenant = self.tenant_for()
        pool = await self._get_pool()
        if not self._has_expires:
            return 0
        async with pool.acquire() as conn, conn.transaction():
            await self._pinned(conn, tenant)
            status = await conn.execute(
                f"DELETE FROM {self._table} WHERE tenant = $1 "
                "AND expires_at IS NOT NULL AND expires_at <= now()",
                tenant,
            )
        return _count(status)

    async def exists(self, thread_id: str, checkpoint_id: str | None = None) -> bool:
        if checkpoint_id is None:
            return bool(await self.list_checkpoints(thread_id, limit=1))
        tenant = self.tenant_for(thread_id)
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            await self._pinned(conn, tenant)
            found = await conn.fetchval(
                f"SELECT 1 FROM {self._table} "
                f"WHERE tenant = $1 AND thread_id = $2 AND checkpoint_id = $3 {self._live}",
                tenant,
                self._scoped(thread_id),
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
        key = self._scoped(thread_id)
        cols = "checkpoint_id, metadata, created_at, updated_at"
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            await self._pinned(conn, tenant)
            if checkpoint_id is None:
                row = await conn.fetchrow(
                    f"SELECT {cols} FROM {self._table} WHERE tenant = $1 AND thread_id = $2 "
                    f"{self._live} ORDER BY seq DESC LIMIT 1",
                    tenant,
                    key,
                )
            else:
                row = await conn.fetchrow(
                    f"SELECT {cols} FROM {self._table} "
                    f"WHERE tenant = $1 AND thread_id = $2 AND checkpoint_id = $3 {self._live}",
                    tenant,
                    key,
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
                f"WHERE tenant = $1 AND metadata @> $2::jsonb {self._live} ORDER BY seq DESC LIMIT $3",
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

    async def list_threads(
        self, limit: int = 100, pattern: str = "*", *, scope: CheckpointScope | None = None
    ) -> list[str]:
        """The current tenant's threads of one scope, most recently saved first.

        ``pattern`` is a glob (``*`` and ``?``), matched in SQL.
        """
        tenant = self.tenant_for()
        listing = self._listing_scope(scope)
        like = (
            f"{listing.prefix}{pattern}".replace("\\", "\\\\")
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
                f"{self._live} GROUP BY thread_id ORDER BY max(seq) DESC LIMIT $3",
                tenant,
                like,
                limit,
            )
        return listing.strip(r["thread_id"] for r in rows)

    async def copy_thread(self, source_thread_id: str, dest_thread_id: str) -> bool:
        """Copy every checkpoint of a thread to another thread of the same tenant.

        One statement when the rows can be copied as they are; an encrypted
        checkpoint is bound to its thread, so with ``encryption`` each row
        is unsealed and sealed again under the new thread.
        """
        tenant = self.tenant_for(source_thread_id)
        if self.tenant_for(dest_thread_id) != tenant:
            raise ValueError("copy_thread cannot copy a thread to another tenant")
        source_key, dest_key = self._scoped(source_thread_id), self._scoped(dest_thread_id)
        pool = await self._get_pool()
        expires = ", expires_at" if self._has_expires else ""
        async with pool.acquire() as conn, conn.transaction():
            await self._pinned(conn, tenant)
            if self.encryption is None:
                status = await conn.execute(
                    f"INSERT INTO {self._table} "
                    f"(tenant, thread_id, checkpoint_id, data, metadata, oldest_at{expires}) "
                    f"SELECT tenant, $3, checkpoint_id, data, metadata, oldest_at{expires} "
                    f"FROM {self._table} WHERE tenant = $1 AND thread_id = $2 {self._live} "
                    "ORDER BY seq ON CONFLICT (tenant, thread_id, checkpoint_id) DO NOTHING",
                    tenant,
                    source_key,
                    dest_key,
                )
                return _count(status) > 0
            rows = await conn.fetch(
                f"SELECT checkpoint_id, data, metadata, oldest_at{expires} FROM {self._table} "
                f"WHERE tenant = $1 AND thread_id = $2 {self._live} ORDER BY seq",
                tenant,
                source_key,
            )
            copied = 0
            for r in rows:
                raw = r["data"]
                state = self._decode_state(
                    json.loads(raw) if isinstance(raw, str) else raw, source_key, r["checkpoint_id"]
                )
                args: list[Any] = [
                    tenant,
                    dest_key,
                    r["checkpoint_id"],
                    json.dumps(self._encode_state(state, dest_key, r["checkpoint_id"])),
                    r["metadata"] if isinstance(r["metadata"], str) else json.dumps(r["metadata"]),
                    r["oldest_at"],
                ]
                values = "$1, $2, $3, $4::jsonb, $5::jsonb, $6"
                if self._has_expires:
                    args.append(r["expires_at"])
                    values += ", $7"
                status = await conn.execute(
                    f"INSERT INTO {self._table} "
                    f"(tenant, thread_id, checkpoint_id, data, metadata, oldest_at{expires}) "
                    f"VALUES ({values}) ON CONFLICT (tenant, thread_id, checkpoint_id) DO NOTHING",
                    *args,
                )
                copied += _count(status)
        return copied > 0

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
        if older_than <= timedelta(0):
            raise ValueError("older_than must be positive")
        tenant = self.tenant_for(thread_id)
        key = self._scoped(thread_id) if thread_id else None
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
                    *([key] if key else []),
                )
                updates = []
                for r in rows:
                    raw = r["data"]
                    state = self._decode_state(
                        json.loads(raw) if isinstance(raw, str) else raw,
                        r["thread_id"],
                        r["checkpoint_id"],
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
                            json.dumps(
                                self._encode_state(trimmed, r["thread_id"], r["checkpoint_id"])
                            ),
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

    # Last on purpose: a method named ``list`` shadows the builtin for the
    # annotations of anything defined after it.
    async def list(self, thread_id: str, limit: int = 10) -> list[CheckpointInfo]:
        """The thread's checkpoints, newest first, in one query that never returns the state.

        ``size`` is the stored row's ``jsonb`` as text — the checkpoint as
        this backend keeps it, not the compact JSON the others report.
        """
        tenant = self.tenant_for(thread_id)
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            await self._pinned(conn, tenant)
            # One row past the window names the parent of the oldest in it.
            rows = await conn.fetch(
                "SELECT checkpoint_id, metadata, created_at, "
                "(data->>'iteration')::int AS iteration, octet_length(data::text) AS size "
                f"FROM {self._table} WHERE tenant = $1 AND thread_id = $2 {self._live} "
                "ORDER BY seq DESC LIMIT $3",
                tenant,
                self._scoped(thread_id),
                limit + 1,
            )
        older = rows[limit]["checkpoint_id"] if len(rows) > limit else None
        infos: list[CheckpointInfo] = []
        for r in rows[:limit]:
            meta = r["metadata"]
            infos.append(
                info_from(
                    r["checkpoint_id"],
                    thread_id,
                    created_at=r["created_at"],
                    metadata=json.loads(meta) if isinstance(meta, str) else dict(meta or {}),
                    iteration=r["iteration"],
                    size=r["size"],
                )
            )
        return link_parents(infos, older)
