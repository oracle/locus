# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Approval store in PostgreSQL: one row per held call, decided under a row lock.

Same connection conventions as
:class:`~locus.memory.backends.postgresql.PostgreSQLBackend`: a libpq DSN or
host / port / database / user / password, an ``asyncpg`` pool, and a table that
is created on first use only when it does not exist, so a role without
``CREATE`` works against a table a migration made::

    from locus.control import PostgresApprovals

    store = PostgresApprovals(dsn="postgresql://locus@db.internal:5432/locus")
    refund = gate_tool(
        issue_refund, policy=policy, approval=store, on_refusal="interrupt"
    )

Every submit, decision and consumption is one transaction that locks the rows
it reads (``SELECT ... FOR UPDATE``), so replicas sharing the table never both
redeem one approval. Install with ``pip install locus-sdk[postgresql]``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from locus.control.approvals_db import (
    DatabaseApprovals,
    record_from_json,
    record_to_json,
    sql_identifier,
    without_password,
)
from locus.core.loop_bound import loop_bound_async


if TYPE_CHECKING:
    from asyncpg import Connection, Pool

    from locus.control.approvals import ApprovalAuthority, ApprovalRecord


__all__ = ["PostgresApprovals"]


class _PgTransaction:
    def __init__(self, conn: Connection, table: str) -> None:
        self._conn = conn
        self._table = table

    async def get(self, approval_id: str, *, lock: bool = False) -> ApprovalRecord | None:
        row = await self._conn.fetchrow(
            f"SELECT record FROM {self._table} WHERE approval_id = $1"
            + (" FOR UPDATE" if lock else ""),
            approval_id,
        )
        return record_from_json(row["record"]) if row is not None else None

    async def same_call(self, digest: str, *, lock: bool = False) -> list[ApprovalRecord]:
        rows = await self._conn.fetch(
            f"SELECT record FROM {self._table} WHERE digest = $1" + (" FOR UPDATE" if lock else ""),
            digest,
        )
        return [record_from_json(row["record"]) for row in rows]

    async def insert(self, record: ApprovalRecord) -> bool:
        outcome: str = await self._conn.execute(
            f"""
            INSERT INTO {self._table} (approval_id, digest, status, created_at, record)
            VALUES ($1, $2, $3, $4, $5::jsonb)
            ON CONFLICT (approval_id) DO NOTHING
            """,
            record.approval_id,
            record.digest,
            record.status,
            record.created_at,
            record_to_json(record),
        )
        return outcome.endswith(" 1")

    async def update(self, record: ApprovalRecord) -> None:
        await self._conn.execute(
            f"UPDATE {self._table} SET status = $2, record = $3::jsonb WHERE approval_id = $1",
            record.approval_id,
            record.status,
            record_to_json(record),
        )

    async def waiting(self) -> list[ApprovalRecord]:
        rows = await self._conn.fetch(
            f"SELECT record FROM {self._table} WHERE status = 'pending' ORDER BY created_at"
        )
        return [record_from_json(row["record"]) for row in rows]


class PostgresApprovals(DatabaseApprovals):
    """An approval store in a PostgreSQL table.

    Args:
        dsn: A libpq connection string. Overrides the individual parameters.
        host: The server, when no ``dsn``.
        port: Its port.
        database: The database.
        user: The role to connect as.
        password: Its password.
        table_name: The table; created on first use if missing.
        schema_name: Its schema.
        min_pool_size: The ``asyncpg`` pool's lower bound.
        max_pool_size: Its upper bound.
        authority: Who may decide, as for
            :class:`~locus.control.approvals.InMemoryApprovals`.
    """

    def __init__(
        self,
        dsn: str | None = None,
        *,
        host: str = "localhost",
        port: int = 5432,
        database: str = "locus",
        user: str = "postgres",
        password: str = "",
        table_name: str = "locus_approvals",
        schema_name: str = "public",
        min_pool_size: int = 1,
        max_pool_size: int = 5,
        authority: ApprovalAuthority
        | Callable[[ApprovalRecord], ApprovalAuthority | None]
        | None = None,
    ) -> None:
        super().__init__(authority)
        self.dsn = dsn
        self.host, self.port, self.database, self.user = host, port, database, user
        self._password = password
        self.table_name = sql_identifier(table_name, "table_name")
        self.schema_name = sql_identifier(schema_name, "schema_name")
        self._pool_bounds = (min_pool_size, max_pool_size)
        self._pool: Pool | None = None
        self._pool_loop: Any = None
        self._ready = False

    @property
    def spec(self) -> str:
        where = (
            without_password(self.dsn)
            if self.dsn
            else f"postgresql://{self.host}:{self.port}/{self.database}"
        )
        joiner = "&" if "?" in where else "?"
        return f"{where}{joiner}table={self.schema_name}.{self.table_name}"

    @property
    def _table(self) -> str:
        return f"{self.schema_name}.{self.table_name}"

    async def _build_pool(self) -> Pool:
        try:
            import asyncpg  # noqa: PLC0415
        except ImportError as error:
            raise ImportError(
                "PostgresApprovals needs the 'asyncpg' package: pip install locus-sdk[postgresql]"
            ) from error
        low, high = self._pool_bounds
        self._ready = False
        if self.dsn:
            pool: Pool = await asyncpg.create_pool(self.dsn, min_size=low, max_size=high)
            return pool
        pool = await asyncpg.create_pool(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self._password,
            min_size=low,
            max_size=high,
        )
        return pool

    async def _get_pool(self) -> Pool:
        # The pool lives on the store's own loop, so it is rebuilt only if
        # someone drives the store's coroutines from another loop; the shared
        # cache keeps that case correct rather than mysterious.
        return await loop_bound_async(self, "_pool", self._build_pool)

    async def _ensure_table(self, conn: Any) -> None:
        if self._ready:
            return
        # Probe before any DDL: CREATE ... IF NOT EXISTS checks privileges
        # before existence, so a role without CREATE failed against a table a
        # migration had already made.
        if await conn.fetchval("SELECT to_regclass($1)", self._table) is None:
            await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema_name}")
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    approval_id TEXT PRIMARY KEY,
                    digest TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    record JSONB NOT NULL
                )
            """)
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_digest ON {self._table} (digest)"
            )
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_status "
                f"ON {self._table} (status, created_at)"
            )
        self._ready = True

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[_PgTransaction]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await self._ensure_table(conn)
            async with conn.transaction():
                yield _PgTransaction(conn, self._table)

    async def _disconnect(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._ready = False
