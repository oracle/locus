# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""PostgreSQL checkpoint backend - 100% Pydantic."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, SecretStr

from locus.memory.backends.checkpoint_index import index_key


if TYPE_CHECKING:
    from asyncpg import Pool


logger = logging.getLogger(__name__)

#: The filter a read applies once the table has the column: rows that have not expired.
_LIVE = "AND (expires_at IS NULL OR expires_at > now())"

_SAFE_SQL_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,62}$")


def _validate_sql_identifier(value: str, field_name: str) -> str:
    """Validate that a string is a safe SQL identifier (alphanumeric + underscore only)."""
    if not _SAFE_SQL_IDENTIFIER.match(value):
        msg = (
            f"Invalid {field_name}: {value!r}. "
            "Must start with a letter or underscore and contain only "
            "alphanumeric characters and underscores (max 63 chars)."
        )
        raise ValueError(msg)
    return value


class PostgreSQLConfig(BaseModel):
    """Configuration for PostgreSQL backend."""

    # Reject unknown keys rather than absorbing them. The backends take
    # **kwargs and splat them in here, so a misspelled or wrong-named
    # parameter used to vanish silently — notebook 65 passed namespace=
    # instead of prefix= and every run quietly shared one keyspace.
    model_config = {"extra": "forbid"}

    host: str = "localhost"
    port: int = 5432
    database: str = "locus"
    user: str = "postgres"
    password: SecretStr = SecretStr("")
    table_name: str = "checkpoints"
    schema_name: str = "public"
    min_pool_size: int = 1
    max_pool_size: int = 10
    # Connection string (overrides individual params)
    dsn: str | None = None

    def model_post_init(self, __context: Any) -> None:
        """Validate SQL identifiers to prevent injection."""
        _validate_sql_identifier(self.table_name, "table_name")
        _validate_sql_identifier(self.schema_name, "schema_name")


class PostgreSQLBackend(BaseModel):
    """
    PostgreSQL checkpoint backend.

    Production-grade persistent storage with ACID guarantees.

    Features:
    - Connection pooling
    - Transaction support
    - JSON/JSONB storage
    - Indexing for fast lookups
    - Concurrent access safe

    Example:
        >>> backend = PostgreSQLBackend(
        ...     host="localhost",
        ...     database="myapp",
        ...     user="postgres",
        ...     password="secret",
        ... )
        >>> await backend.save("thread_1", state.model_dump())
        >>> data = await backend.load("thread_1")

    With DSN:
        >>> backend = PostgreSQLBackend(dsn="postgresql://user:pass@localhost:5432/mydb")

    Checkpoint index:
        Through ``postgresql_checkpointer()`` the thread's checkpoint index
        is the ``<thread_id>:_checkpoints`` row's ``data`` column, as it
        always was, but each change is one ``INSERT ... ON CONFLICT DO
        UPDATE`` or ``UPDATE`` that rewrites the JSON array from the row's
        current value under the row lock, so two processes saving to one
        thread never lose an entry.

    Expiry:
        ``save(..., expires_at=...)`` keeps the time in an ``expires_at``
        column, added to an existing table with ``ALTER TABLE ... ADD COLUMN
        IF NOT EXISTS`` on first use (a role that may not alter the table
        goes without the column, and the adapter's own record of the time
        still applies); ``load`` and ``exists`` skip expired rows and
        :meth:`expire` deletes them.
    """

    config: PostgreSQLConfig = Field(default_factory=PostgreSQLConfig)
    _pool: Pool | None = None
    #: Whether the table has the ``expires_at`` column (probed or added on first use).
    _expires_column: bool = True
    #: The loop the cached pool was built on, or ``None`` when the pool did
    #: not come from here — a caller (or a test) that assigns ``_pool`` owns
    #: its lifecycle, and second-guessing that would discard it.
    _pool_loop: asyncio.AbstractEventLoop | None = None
    _initialized: bool = False

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "locus",
        user: str = "postgres",
        password: str | SecretStr = "",
        dsn: str | None = None,
        **kwargs: Any,
    ) -> None:
        config = PostgreSQLConfig(
            host=host,
            port=port,
            database=database,
            user=user,
            password=SecretStr(password) if isinstance(password, str) else password,
            dsn=dsn,
            **kwargs,
        )
        super().__init__(config=config)

    async def _get_pool(self) -> Pool:
        """Get or create the connection pool for the running event loop.

        asyncpg pools are loop-affine: a pool created on one loop cannot be
        used from another, and doing so fails with 'another operation is in
        progress'. Keying the cache on the loop keeps the usual single-loop
        case free and rebuilds only when the loop changed.
        """
        loop = asyncio.get_running_loop()
        stale = self._pool_loop is not None and self._pool_loop is not loop
        if self._pool is None or stale:
            try:
                import asyncpg
            except ImportError as e:
                raise ImportError(
                    "PostgreSQLBackend requires the 'asyncpg' package. "
                    "Install with: pip install locus[postgresql]"
                ) from e

            # Stamp the loop before creating: the pool built below belongs to
            # it, and a later call on another loop must rebuild rather than
            # reuse. Schema setup is per-pool, so it has to run again too.
            self._pool_loop = loop
            self._initialized = False

            if self.config.dsn:
                self._pool = await asyncpg.create_pool(
                    self.config.dsn,
                    min_size=self.config.min_pool_size,
                    max_size=self.config.max_pool_size,
                )
            else:
                self._pool = await asyncpg.create_pool(
                    host=self.config.host,
                    port=self.config.port,
                    database=self.config.database,
                    user=self.config.user,
                    password=self.config.password.get_secret_value(),
                    min_size=self.config.min_pool_size,
                    max_size=self.config.max_pool_size,
                )

        return self._pool

    @property
    def _full_table_name(self) -> str:
        """Get fully qualified table name."""
        return f"{self.config.schema_name}.{self.config.table_name}"

    async def _ensure_table(self) -> None:
        """Create table if not exists."""
        if self._initialized:
            return

        pool = await self._get_pool()

        async with pool.acquire() as conn:
            # An existing table needs no DDL. Probing first matters: CREATE
            # SCHEMA / TABLE / INDEX ... IF NOT EXISTS check privileges before
            # existence, so a role without CREATE on the database (or schema)
            # failed against a table a migration had already made.
            existing = await conn.fetchval("SELECT to_regclass($1)", self._full_table_name)
            if existing is not None:
                await self._ensure_expires_column(conn)
                self._initialized = True
                return

            # Create schema if needed
            await conn.execute(f"""
                CREATE SCHEMA IF NOT EXISTS {self.config.schema_name}
            """)

            # Create table with JSONB for efficient querying
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._full_table_name} (
                    thread_id TEXT PRIMARY KEY,
                    checkpoint_id TEXT,
                    data JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    expires_at TIMESTAMPTZ
                )
            """)

            # Create indexes
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_updated
                ON {self._full_table_name} (updated_at DESC)
            """)

            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_metadata
                ON {self._full_table_name} USING GIN (metadata)
            """)

        self._initialized = True

    async def _ensure_expires_column(self, conn: Any) -> None:
        """Add ``expires_at`` to a table from before checkpoints could expire, when allowed."""
        present = await conn.fetchval(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = $1 AND table_name = $2 AND column_name = 'expires_at'",
            self.config.schema_name,
            self.config.table_name,
        )
        if present is not None:
            self._expires_column = True
            return
        try:
            await conn.execute(
                f"ALTER TABLE {self._full_table_name} ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ"
            )
        except Exception:  # noqa: BLE001 — a role that may not alter the table
            logger.warning(
                "could not add the expires_at column to %s; checkpoint expiry is kept in "
                "the documents only",
                self._full_table_name,
                exc_info=True,
            )
            self._expires_column = False
            return
        self._expires_column = True

    @property
    def _live(self) -> str:
        """The expiry filter for a read, or nothing when the table has no column."""
        return _LIVE if self._expires_column else ""

    async def save(
        self,
        thread_id: str,
        data: dict[str, Any],
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        expires_at: datetime | None = None,
    ) -> str:
        """
        Save checkpoint to PostgreSQL.

        Args:
            thread_id: Thread identifier
            data: Checkpoint data
            checkpoint_id: Optional checkpoint ID
            metadata: Optional metadata for querying
            expires_at: When the row stops being loadable (``None``: never)

        Returns:
            Checkpoint ID
        """
        await self._ensure_table()
        pool = await self._get_pool()

        from uuid import uuid4

        checkpoint_id = checkpoint_id or uuid4().hex
        now = datetime.now(UTC)

        async with pool.acquire() as conn:
            if self._expires_column:
                await conn.execute(
                    f"""
                    INSERT INTO {self._full_table_name}
                        (thread_id, checkpoint_id, data, created_at, updated_at, expires_at,
                         metadata)
                    VALUES ($1, $2, $3::jsonb, $4, $5, $6, $7::jsonb)
                    ON CONFLICT (thread_id) DO UPDATE SET
                        checkpoint_id = EXCLUDED.checkpoint_id,
                        data = EXCLUDED.data,
                        updated_at = EXCLUDED.updated_at,
                        expires_at = EXCLUDED.expires_at,
                        metadata = EXCLUDED.metadata
                    """,
                    thread_id,
                    checkpoint_id,
                    json.dumps(data),
                    now,
                    now,
                    expires_at,
                    json.dumps(metadata or {}),
                )
            else:
                await conn.execute(
                    f"""
                    INSERT INTO {self._full_table_name}
                        (thread_id, checkpoint_id, data, created_at, updated_at, metadata)
                    VALUES ($1, $2, $3::jsonb, $4, $5, $6::jsonb)
                    ON CONFLICT (thread_id) DO UPDATE SET
                        checkpoint_id = EXCLUDED.checkpoint_id,
                        data = EXCLUDED.data,
                        updated_at = EXCLUDED.updated_at,
                        metadata = EXCLUDED.metadata
                    """,
                    thread_id,
                    checkpoint_id,
                    json.dumps(data),
                    now,
                    now,
                    json.dumps(metadata or {}),
                )

        return checkpoint_id

    async def load(self, thread_id: str) -> dict[str, Any] | None:
        """Load checkpoint from PostgreSQL (``None`` for a row whose ``expires_at`` has passed)."""
        await self._ensure_table()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT data FROM {self._full_table_name} WHERE thread_id = $1 {self._live}",
                thread_id,
            )

        if row is None:
            return None

        data: dict[str, Any] = json.loads(row["data"])
        return data

    async def delete(self, thread_id: str) -> bool:
        """Delete checkpoint from PostgreSQL."""
        await self._ensure_table()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result: str = await conn.execute(
                f"DELETE FROM {self._full_table_name} WHERE thread_id = $1",
                thread_id,
            )

        return result == "DELETE 1"

    async def exists(self, thread_id: str) -> bool:
        """Check if checkpoint exists."""
        await self._ensure_table()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT 1 FROM {self._full_table_name} WHERE thread_id = $1 {self._live}",
                thread_id,
            )

        return row is not None

    async def expire(self) -> int:
        """Delete the rows whose ``expires_at`` has passed; returns how many."""
        await self._ensure_table()
        if not self._expires_column:
            return 0
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result: str = await conn.execute(
                f"DELETE FROM {self._full_table_name} "
                "WHERE expires_at IS NOT NULL AND expires_at <= now()"
            )
        try:
            return int(result.split()[1])
        except (IndexError, ValueError):
            return 0

    async def list_threads(
        self,
        pattern: str = "%",
        limit: int = 100,
        offset: int = 0,
    ) -> list[str]:
        """List all thread IDs matching pattern."""
        await self._ensure_table()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT thread_id FROM {self._full_table_name}
                WHERE thread_id LIKE $1
                ORDER BY updated_at DESC
                LIMIT $2 OFFSET $3
                """,
                pattern,
                limit,
                offset,
            )

        return [row["thread_id"] for row in rows]

    async def get_metadata(self, thread_id: str) -> dict[str, Any] | None:
        """Get checkpoint metadata."""
        await self._ensure_table()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT checkpoint_id, created_at, updated_at, metadata
                FROM {self._full_table_name}
                WHERE thread_id = $1
                """,
                thread_id,
            )

        if row is None:
            return None

        return {
            "checkpoint_id": row["checkpoint_id"],
            "created_at": row["created_at"].isoformat(),
            "updated_at": row["updated_at"].isoformat(),
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
        }

    async def query_by_metadata(
        self,
        key: str,
        value: Any,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Query checkpoints by metadata field.

        Uses PostgreSQL JSONB operators for efficient querying.
        """
        await self._ensure_table()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT thread_id, data, updated_at
                FROM {self._full_table_name}
                WHERE metadata @> $1::jsonb
                ORDER BY updated_at DESC
                LIMIT $2
                """,
                json.dumps({key: value}),
                limit,
            )

        return [
            {
                "thread_id": row["thread_id"],
                "data": json.loads(row["data"]),
                "updated_at": row["updated_at"].isoformat(),
            }
            for row in rows
        ]

    async def search_data(
        self,
        path: str,
        value: Any,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Search checkpoints by data field using JSON path.

        Args:
            path: JSON path (e.g., "messages", "confidence")
            value: Value to match
            limit: Maximum results

        Example:
            >>> results = await backend.search_data("agent_id", "agent-123")
        """
        await self._ensure_table()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT thread_id, data, updated_at
                FROM {self._full_table_name}
                WHERE data @> $1::jsonb
                ORDER BY updated_at DESC
                LIMIT $2
                """,
                json.dumps({path: value}),
                limit,
            )

        return [
            {
                "thread_id": row["thread_id"],
                "data": json.loads(row["data"]),
                "updated_at": row["updated_at"].isoformat(),
            }
            for row in rows
        ]

    async def count(self, pattern: str = "%") -> int:
        """Count checkpoints matching pattern."""
        await self._ensure_table()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT COUNT(*) as cnt FROM {self._full_table_name} WHERE thread_id LIKE $1",
                pattern,
            )

        return row["cnt"] if row else 0

    async def vacuum(self, older_than_days: int = 30) -> int:
        """
        Delete old checkpoints.

        Args:
            older_than_days: Delete checkpoints older than this

        Returns:
            Number of deleted rows
        """
        await self._ensure_table()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.execute(
                f"""
                DELETE FROM {self._full_table_name}
                WHERE updated_at < NOW() - make_interval(days => $1)
                """,
                older_than_days,
            )

        # Parse "DELETE N"
        try:
            return int(result.split()[1])
        except (IndexError, ValueError):
            return 0

    # -- checkpoint index (CheckpointIndexStore) -------------------------------

    async def index_add(self, thread_id: str, entry: dict[str, Any]) -> None:
        """Add the entry to the index row in one upsert, replacing one with its id."""
        await self._ensure_table()
        pool = await self._get_pool()
        now = datetime.now(UTC)

        # ``ON CONFLICT DO UPDATE`` evaluates the new array against the row as
        # it is when the row lock is taken, so concurrent upserts queue up and
        # each sees the entries the previous one wrote.
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self._full_table_name} AS idx
                    (thread_id, checkpoint_id, data, created_at, updated_at, metadata)
                VALUES (
                    $1, NULL, jsonb_build_object('checkpoints', jsonb_build_array($2::jsonb)),
                    $3, $3, '{{}}'::jsonb
                )
                ON CONFLICT (thread_id) DO UPDATE SET
                    data = jsonb_build_object(
                        'checkpoints',
                        (
                            SELECT jsonb_agg(e ORDER BY e->>'timestamp' DESC)
                            FROM (
                                SELECT e
                                FROM jsonb_array_elements(
                                    COALESCE(idx.data->'checkpoints', '[]'::jsonb)
                                ) AS elems(e)
                                WHERE e->>'checkpoint_id' IS DISTINCT FROM $4::text
                                UNION ALL
                                SELECT $2::jsonb
                            ) AS entries(e)
                        )
                    ),
                    updated_at = EXCLUDED.updated_at
                """,
                index_key(thread_id),
                json.dumps(entry),
                now,
                str(entry["checkpoint_id"]),
            )

    async def index_remove(self, thread_id: str, checkpoint_id: str) -> bool:
        """Drop the entry from the index row in one ``UPDATE``; ``True`` when it was there."""
        await self._ensure_table()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result: str = await conn.execute(
                f"""
                UPDATE {self._full_table_name} SET
                    data = jsonb_build_object(
                        'checkpoints',
                        COALESCE(
                            (
                                SELECT jsonb_agg(e ORDER BY e->>'timestamp' DESC)
                                FROM jsonb_array_elements(
                                    COALESCE(data->'checkpoints', '[]'::jsonb)
                                ) AS elems(e)
                                WHERE e->>'checkpoint_id' IS DISTINCT FROM $2::text
                            ),
                            '[]'::jsonb
                        )
                    ),
                    updated_at = $3
                WHERE thread_id = $1
                  AND data->'checkpoints' @> jsonb_build_array(
                      jsonb_build_object('checkpoint_id', $2::text)
                  )
                """,
                index_key(thread_id),
                checkpoint_id,
                datetime.now(UTC),
            )

        return result == "UPDATE 1"

    async def index_list(self, thread_id: str) -> list[dict[str, Any]]:
        """The index row's entries; ``[]`` without the row."""
        existing = await self.load(index_key(thread_id))
        return list((existing or {}).get("checkpoints", []))

    async def index_clear(self, thread_id: str) -> bool:
        """Delete the index row."""
        return await self.delete(index_key(thread_id))

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    def __repr__(self) -> str:
        if self.config.dsn:
            return "PostgreSQLBackend(dsn=...)"
        return f"PostgreSQLBackend(host={self.config.host}, database={self.config.database})"
