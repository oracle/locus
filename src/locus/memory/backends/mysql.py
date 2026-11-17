# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""MySQL checkpoint backend using the official Connector/Python asyncio API."""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import parse_qsl, unquote, urlparse

from pydantic import BaseModel, Field, SecretStr

from locus.core.loop_bound import loop_bound_async
from locus.memory.backends.checkpoint_index import index_key


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from mysql.connector.aio.abstracts import MySQLCursorAbstract


_SAFE_SQL_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,63}$")


class _Connection(Protocol):
    """The slice of the Connector/Python async connection API the pool uses.

    ``mysql.connector.aio.connect()`` returns a ``MySQLConnectionAbstract``,
    or a ``PooledMySQLConnection`` proxying one when the connection arguments
    name a connector-level pool. Both satisfy this protocol, so the pool holds
    whichever ``connect()`` handed back.
    """

    async def cursor(self) -> MySQLCursorAbstract: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...

    async def close(self) -> None: ...


def _validate_sql_identifier(value: str, field_name: str) -> str:
    """Validate that a string is a safe MySQL identifier."""
    if not _SAFE_SQL_IDENTIFIER.match(value):
        msg = (
            f"Invalid {field_name}: {value!r}. "
            "Must start with a letter or underscore and contain only "
            "alphanumeric characters and underscores (max 64 chars)."
        )
        raise ValueError(msg)
    return value


def _quote_identifier(value: str) -> str:
    """Quote a validated MySQL identifier."""
    return f"`{value}`"


def _decode_json(value: Any) -> Any:
    """Decode JSON returned by Connector/Python across text/bytes/native paths."""
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    return value


def _like_literal(value: str) -> str:
    """``value`` with ``JSON_SEARCH``'s LIKE wildcards escaped, so it matches itself alone."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


#: The index row's entry array (``[]`` for a row without one).
_INDEX_ARRAY = "COALESCE(JSON_EXTRACT(data, '$.checkpoints'), JSON_ARRAY())"
#: The path of the entry whose ``checkpoint_id`` is the parameter (``NULL`` when none);
#: two parameters: the escaped id and the escape character.
_INDEX_ENTRY_PATH = (
    "REPLACE(JSON_UNQUOTE(JSON_SEARCH(JSON_EXTRACT(data, '$.checkpoints'), 'one', %s, %s,"
    " '$[*].checkpoint_id')), '.checkpoint_id', '')"
)
_LIKE_ESCAPE = "\\"


def _as_str(value: object) -> str:
    """Narrow a text column value to ``str``; the connector may return ``bytes``."""
    if isinstance(value, bytes | bytearray):
        return bytes(value).decode("utf-8")
    if isinstance(value, str):
        return value
    msg = f"Expected a text column value, got {type(value).__name__}"
    raise TypeError(msg)


def _as_int(value: object) -> int:
    """Narrow a numeric column value (``COUNT(*)``) to ``int``."""
    if isinstance(value, int | float | Decimal | str | bytes):
        return int(value)
    msg = f"Expected a numeric column value, got {type(value).__name__}"
    raise TypeError(msg)


def _as_datetime(value: object) -> datetime:
    """Narrow a ``TIMESTAMP`` column value, which the converter returns as ``datetime``."""
    if isinstance(value, datetime):
        return value
    msg = f"Expected a datetime column value, got {type(value).__name__}"
    raise TypeError(msg)


def _parse_dsn(dsn: str) -> dict[str, Any]:
    """Parse a mysql:// DSN into Connector/Python keyword arguments."""
    parsed = urlparse(dsn)
    if parsed.scheme not in {"mysql", "mysql+connector"}:
        msg = "MySQL DSN must use the mysql:// or mysql+connector:// scheme"
        raise ValueError(msg)

    config: dict[str, Any] = {}
    if parsed.hostname:
        config["host"] = parsed.hostname
    if parsed.port:
        config["port"] = parsed.port
    if parsed.username:
        config["user"] = unquote(parsed.username)
    if parsed.password:
        config["password"] = unquote(parsed.password)
    if parsed.path and parsed.path != "/":
        config["database"] = unquote(parsed.path.lstrip("/"))

    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        config[key] = value
    return config


class MySQLConfig(BaseModel):
    """Configuration for the MySQL backend."""

    # Reject unknown keys rather than absorbing them. The backends take
    # **kwargs and splat them in here, so a misspelled or wrong-named
    # parameter used to vanish silently — notebook 65 passed namespace=
    # instead of prefix= and every run quietly shared one keyspace.
    model_config = {"extra": "forbid"}

    host: str = "localhost"
    port: int = 3306
    database: str = "locus"
    user: str = "root"
    password: SecretStr = SecretStr("")
    table_name: str = "checkpoints"
    min_pool_size: int = 1
    max_pool_size: int = 10
    charset: str = "utf8mb4"
    collation: str = "utf8mb4_unicode_ci"
    autocommit: bool = False
    use_pure: bool = True
    dsn: str | None = None

    def model_post_init(self, __context: Any) -> None:
        """Validate SQL identifiers to prevent injection."""
        _validate_sql_identifier(self.database, "database")
        _validate_sql_identifier(self.table_name, "table_name")
        _validate_sql_identifier(self.charset, "charset")
        _validate_sql_identifier(self.collation, "collation")


class _MySQLConnectionPool:
    """Small asyncio pool for official Connector/Python async connections."""

    def __init__(self, config: MySQLConfig) -> None:
        if config.min_pool_size < 0:
            msg = "min_pool_size must be >= 0"
            raise ValueError(msg)
        if config.max_pool_size < 1:
            msg = "max_pool_size must be >= 1"
            raise ValueError(msg)
        if config.min_pool_size > config.max_pool_size:
            msg = "min_pool_size must be <= max_pool_size"
            raise ValueError(msg)

        self._config = config
        self._available: asyncio.Queue[_Connection] = asyncio.Queue()
        self._created = 0
        self._closed = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Open the configured minimum number of connections."""
        for _ in range(self._config.min_pool_size):
            await self._available.put(await self._connect())

    async def _connect(self) -> _Connection:
        try:
            from mysql.connector.aio import connect
        except ImportError as e:
            raise ImportError(
                "MySQLBackend requires the 'mysql-connector-python' package. "
                "Install with: pip install locus[mysql]"
            ) from e

        kwargs = self._connect_kwargs()
        conn: _Connection = await connect(**kwargs)
        self._created += 1
        return conn

    def _connect_kwargs(self) -> dict[str, Any]:
        if self._config.dsn:
            kwargs = _parse_dsn(self._config.dsn)
        else:
            kwargs = {
                "host": self._config.host,
                "port": self._config.port,
                "database": self._config.database,
                "user": self._config.user,
                "password": self._config.password.get_secret_value(),
            }

        kwargs.setdefault("charset", self._config.charset)
        kwargs.setdefault("collation", self._config.collation)
        kwargs.setdefault("autocommit", self._config.autocommit)
        kwargs.setdefault("use_pure", self._config.use_pure)
        return kwargs

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[_Connection]:
        """Acquire a connection and return it to the pool afterwards."""
        if self._closed:
            msg = "MySQL connection pool is closed"
            raise RuntimeError(msg)

        conn = await self._acquire()
        discarded = False
        try:
            yield conn
        except BaseException:
            await self._discard(conn)
            discarded = True
            raise
        finally:
            if discarded:
                pass
            elif self._closed:
                await self._discard(conn)
            elif not getattr(conn, "closed", False):
                await self._release(conn)

    async def _release(self, conn: _Connection) -> None:
        """Return a connection to the pool with no open transaction.

        Under ``autocommit=False`` a bare ``SELECT`` (including read methods
        and the ``SELECT 1`` health check) opens a transaction that holds a
        shared metadata lock on the tables it touched. If the connection were
        pooled in that state it would sit "idle in transaction", block later
        DDL (``DROP``/``ALTER``) on an exclusive MDL, and pin a stale
        REPEATABLE READ snapshot for the next borrower. Roll back first so
        every pooled connection is handed out clean.
        """
        try:
            await conn.rollback()
        except self._connection_errors():
            await self._discard(conn)
        else:
            await self._available.put(conn)

    @staticmethod
    def _connection_errors() -> tuple[type[BaseException], ...]:
        """Errors that mean a pooled connection is no longer usable."""
        error_types: tuple[type[BaseException], ...] = (OSError, RuntimeError, AttributeError)
        try:
            from mysql.connector import Error as MySQLError
        except ImportError:
            return error_types
        return (*error_types, MySQLError)

    async def _acquire(self) -> _Connection:
        while True:
            try:
                conn = self._available.get_nowait()
            except asyncio.QueueEmpty:
                break
            if await self._is_healthy(conn):
                return conn
            await self._discard(conn)

        async with self._lock:
            if self._created < self._config.max_pool_size:
                return await self._connect()

        while True:
            conn = await self._available.get()
            if await self._is_healthy(conn):
                return conn
            await self._discard(conn)

            async with self._lock:
                if self._created < self._config.max_pool_size:
                    return await self._connect()

    async def _is_healthy(self, conn: _Connection) -> bool:
        """Validate an idle connection before handing it out again."""
        if getattr(conn, "closed", False):
            return False

        try:
            async with await conn.cursor() as cur:
                await cur.execute("SELECT 1")
                await cur.fetchone()
        except self._connection_errors():
            return False
        return True

    async def _discard(self, conn: _Connection) -> None:
        """Close a bad connection and make room for a replacement."""
        try:
            if not getattr(conn, "closed", False):
                await conn.close()
        finally:
            async with self._lock:
                self._created = max(0, self._created - 1)

    async def close(self) -> None:
        """Close all idle connections and prevent future acquisitions."""
        self._closed = True
        while not self._available.empty():
            conn = self._available.get_nowait()
            await self._discard(conn)


class MySQLBackend(BaseModel):
    """
    MySQL checkpoint backend.

    Production-grade persistent storage using the official MySQL
    Connector/Python asyncio API.

    Features:
    - Async connection pooling
    - JSON storage
    - Upsert-based checkpoint replacement
    - Metadata queries via MySQL JSON functions

    Example:
        >>> backend = MySQLBackend(
        ...     host="localhost",
        ...     database="myapp",
        ...     user="root",
        ...     password="secret",
        ... )
        >>> await backend.save("thread_1", state.model_dump())
        >>> data = await backend.load("thread_1")

    With DSN:
        >>> backend = MySQLBackend(dsn="mysql://user:pass@localhost:3306/mydb")

    Checkpoint index:
        Through ``mysql_checkpointer()`` the thread's checkpoint index is
        the ``<thread_id>:_checkpoints`` row's ``data`` column, as it always
        was, but each change is one ``INSERT ... ON DUPLICATE KEY UPDATE``
        or ``UPDATE`` that rewrites the JSON array from the row's current
        value under the row lock (``JSON_ARRAY_INSERT`` / ``JSON_REMOVE``),
        so two processes saving to one thread never lose an entry.
    """

    config: MySQLConfig = Field(default_factory=MySQLConfig)
    _pool: _MySQLConnectionPool | None = None
    _initialized: bool = False

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        database: str = "locus",
        user: str = "root",
        password: str | SecretStr = "",
        dsn: str | None = None,
        **kwargs: Any,
    ) -> None:
        config = MySQLConfig(
            host=host,
            port=port,
            database=database,
            user=user,
            password=SecretStr(password) if isinstance(password, str) else password,
            dsn=dsn,
            **kwargs,
        )
        super().__init__(config=config)

    async def _get_pool(self) -> _MySQLConnectionPool:
        """Get or create the connection pool."""

        async def build() -> _MySQLConnectionPool:
            pool = _MySQLConnectionPool(self.config)
            await pool.initialize()
            return pool

        # Bound to the loop that opened it — see locus.core.loop_bound.
        return await loop_bound_async(self, "_pool", build)

    @property
    def _quoted_table_name(self) -> str:
        """Get quoted table name."""
        return _quote_identifier(self.config.table_name)

    async def _ensure_table(self) -> None:
        """Create table if not exists."""
        if self._initialized:
            return

        pool = await self._get_pool()
        index_name = f"idx_{self.config.table_name[:52]}_updated"

        # Define the index inline in CREATE TABLE IF NOT EXISTS so table+index
        # creation is a single atomic, idempotent statement. MySQL has no
        # CREATE INDEX IF NOT EXISTS, so a separate check-then-CREATE INDEX
        # races under concurrent cold start and throws 1061 (Duplicate key
        # name) for the losers.
        async with pool.acquire() as conn:
            async with await conn.cursor() as cur:
                await cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._quoted_table_name} (
                        thread_id VARCHAR(512) PRIMARY KEY,
                        checkpoint_id VARCHAR(128),
                        data JSON NOT NULL,
                        created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
                        updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
                            ON UPDATE CURRENT_TIMESTAMP(6),
                        metadata JSON NULL,
                        KEY `{index_name}` (updated_at DESC)
                    ) ENGINE=InnoDB
                      DEFAULT CHARSET={self.config.charset}
                      COLLATE={self.config.collation}
                    """
                )
            await conn.commit()

        self._initialized = True

    async def save(
        self,
        thread_id: str,
        data: dict[str, Any],
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Save checkpoint to MySQL.

        Args:
            thread_id: Thread identifier
            data: Checkpoint data
            checkpoint_id: Optional checkpoint ID
            metadata: Optional metadata for querying

        Returns:
            Checkpoint ID
        """
        await self._ensure_table()
        pool = await self._get_pool()

        from uuid import uuid4

        checkpoint_id = checkpoint_id or uuid4().hex
        now = datetime.now(UTC).replace(tzinfo=None)

        async with pool.acquire() as conn:
            async with await conn.cursor() as cur:
                await cur.execute(
                    f"""
                    INSERT INTO {self._quoted_table_name}
                        (thread_id, checkpoint_id, data, created_at, updated_at, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        checkpoint_id = VALUES(checkpoint_id),
                        data = VALUES(data),
                        updated_at = VALUES(updated_at),
                        metadata = VALUES(metadata)
                    """,
                    (
                        thread_id,
                        checkpoint_id,
                        json.dumps(data),
                        now,
                        now,
                        json.dumps(metadata or {}),
                    ),
                )
            await conn.commit()

        return checkpoint_id

    async def load(self, thread_id: str) -> dict[str, Any] | None:
        """Load checkpoint from MySQL."""
        await self._ensure_table()
        pool = await self._get_pool()

        async with pool.acquire() as conn, await conn.cursor() as cur:
            await cur.execute(
                f"SELECT data FROM {self._quoted_table_name} WHERE thread_id = %s",
                (thread_id,),
            )
            row = await cur.fetchone()

        if row is None:
            return None

        data: dict[str, Any] = _decode_json(row[0])
        return data

    async def delete(self, thread_id: str) -> bool:
        """Delete checkpoint from MySQL."""
        await self._ensure_table()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            async with await conn.cursor() as cur:
                await cur.execute(
                    f"DELETE FROM {self._quoted_table_name} WHERE thread_id = %s",
                    (thread_id,),
                )
                deleted = int(cur.rowcount)
            await conn.commit()

        return deleted == 1

    async def exists(self, thread_id: str) -> bool:
        """Check if checkpoint exists."""
        await self._ensure_table()
        pool = await self._get_pool()

        async with pool.acquire() as conn, await conn.cursor() as cur:
            await cur.execute(
                f"SELECT 1 FROM {self._quoted_table_name} WHERE thread_id = %s",
                (thread_id,),
            )
            row = await cur.fetchone()

        return row is not None

    async def list_threads(
        self,
        pattern: str = "%",
        limit: int = 100,
        offset: int = 0,
    ) -> list[str]:
        """List all thread IDs matching pattern."""
        await self._ensure_table()
        pool = await self._get_pool()

        async with pool.acquire() as conn, await conn.cursor() as cur:
            await cur.execute(
                f"""
                    SELECT thread_id FROM {self._quoted_table_name}
                    WHERE thread_id LIKE %s
                    ORDER BY updated_at DESC
                    LIMIT %s OFFSET %s
                    """,
                (pattern, limit, offset),
            )
            rows = await cur.fetchall()

        return [_as_str(row[0]) for row in rows]

    async def get_metadata(self, thread_id: str) -> dict[str, Any] | None:
        """Get checkpoint metadata."""
        await self._ensure_table()
        pool = await self._get_pool()

        async with pool.acquire() as conn, await conn.cursor() as cur:
            await cur.execute(
                f"""
                    SELECT checkpoint_id, created_at, updated_at, metadata
                    FROM {self._quoted_table_name}
                    WHERE thread_id = %s
                    """,
                (thread_id,),
            )
            row = await cur.fetchone()

        if row is None:
            return None

        return {
            "checkpoint_id": row[0],
            "created_at": _as_datetime(row[1]).isoformat(),
            "updated_at": _as_datetime(row[2]).isoformat(),
            "metadata": _decode_json(row[3]) or {},
        }

    async def query_by_metadata(
        self,
        key: str,
        value: Any,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Query checkpoints by metadata field.

        Uses MySQL JSON_CONTAINS against a one-key JSON object.
        """
        await self._ensure_table()
        pool = await self._get_pool()

        async with pool.acquire() as conn, await conn.cursor() as cur:
            await cur.execute(
                f"""
                    SELECT thread_id, data, updated_at
                    FROM {self._quoted_table_name}
                    WHERE JSON_CONTAINS(metadata, %s)
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                (json.dumps({key: value}), limit),
            )
            rows = await cur.fetchall()

        return [
            {
                "thread_id": _as_str(row[0]),
                "data": _decode_json(row[1]),
                "updated_at": _as_datetime(row[2]).isoformat(),
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
        Search checkpoints by a top-level data field.

        Args:
            path: Top-level JSON key (for example, "messages" or "agent_id")
            value: Value to match
            limit: Maximum results
        """
        await self._ensure_table()
        pool = await self._get_pool()

        async with pool.acquire() as conn, await conn.cursor() as cur:
            await cur.execute(
                f"""
                    SELECT thread_id, data, updated_at
                    FROM {self._quoted_table_name}
                    WHERE JSON_CONTAINS(data, %s)
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                (json.dumps({path: value}), limit),
            )
            rows = await cur.fetchall()

        return [
            {
                "thread_id": _as_str(row[0]),
                "data": _decode_json(row[1]),
                "updated_at": _as_datetime(row[2]).isoformat(),
            }
            for row in rows
        ]

    async def count(self, pattern: str = "%") -> int:
        """Count checkpoints matching pattern."""
        await self._ensure_table()
        pool = await self._get_pool()

        async with pool.acquire() as conn, await conn.cursor() as cur:
            await cur.execute(
                f"SELECT COUNT(*) FROM {self._quoted_table_name} WHERE thread_id LIKE %s",
                (pattern,),
            )
            row = await cur.fetchone()

        return _as_int(row[0]) if row else 0

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
            async with await conn.cursor() as cur:
                await cur.execute(
                    f"""
                    DELETE FROM {self._quoted_table_name}
                    WHERE updated_at < (UTC_TIMESTAMP(6) - INTERVAL %s DAY)
                    """,
                    (older_than_days,),
                )
                deleted = int(cur.rowcount)
            await conn.commit()

        return int(deleted)

    # -- checkpoint index (CheckpointIndexStore) -------------------------------

    async def index_add(self, thread_id: str, entry: dict[str, Any]) -> None:
        """Add the entry to the index row in one upsert, replacing one with its id.

        The new entry goes first: a save is the thread's newest checkpoint
        in every normal case, and the adapter sorts what it reads anyway.
        """
        await self._ensure_table()
        pool = await self._get_pool()
        now = datetime.now(UTC).replace(tzinfo=None)
        checkpoint_id = _like_literal(str(entry["checkpoint_id"]))
        raw = json.dumps(entry)

        async with pool.acquire() as conn:
            async with await conn.cursor() as cur:
                await cur.execute(
                    f"""
                    INSERT INTO {self._quoted_table_name}
                        (thread_id, checkpoint_id, data, created_at, updated_at, metadata)
                    VALUES (
                        %s, NULL, JSON_OBJECT('checkpoints', JSON_ARRAY(CAST(%s AS JSON))),
                        %s, %s, JSON_OBJECT()
                    )
                    ON DUPLICATE KEY UPDATE
                        data = JSON_OBJECT(
                            'checkpoints',
                            JSON_ARRAY_INSERT(
                                IF(
                                    {_INDEX_ENTRY_PATH} IS NULL,
                                    {_INDEX_ARRAY},
                                    JSON_REMOVE({_INDEX_ARRAY}, {_INDEX_ENTRY_PATH})
                                ),
                                '$[0]',
                                CAST(%s AS JSON)
                            )
                        ),
                        updated_at = VALUES(updated_at)
                    """,
                    (
                        index_key(thread_id),
                        raw,
                        now,
                        now,
                        checkpoint_id,
                        _LIKE_ESCAPE,
                        checkpoint_id,
                        _LIKE_ESCAPE,
                        raw,
                    ),
                )
            await conn.commit()

    async def index_remove(self, thread_id: str, checkpoint_id: str) -> bool:
        """Drop the entry from the index row in one ``UPDATE``; ``True`` when it was there."""
        await self._ensure_table()
        pool = await self._get_pool()
        escaped = _like_literal(checkpoint_id)

        async with pool.acquire() as conn:
            async with await conn.cursor() as cur:
                await cur.execute(
                    f"""
                    UPDATE {self._quoted_table_name} SET
                        data = JSON_OBJECT(
                            'checkpoints', JSON_REMOVE({_INDEX_ARRAY}, {_INDEX_ENTRY_PATH})
                        ),
                        updated_at = %s
                    WHERE thread_id = %s AND {_INDEX_ENTRY_PATH} IS NOT NULL
                    """,
                    (
                        escaped,
                        _LIKE_ESCAPE,
                        datetime.now(UTC).replace(tzinfo=None),
                        index_key(thread_id),
                        escaped,
                        _LIKE_ESCAPE,
                    ),
                )
                removed = int(cur.rowcount)
            await conn.commit()

        return removed == 1

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
            self._initialized = False

    def __repr__(self) -> str:
        if self.config.dsn:
            return "MySQLBackend(dsn=...)"
        return f"MySQLBackend(host={self.config.host}, database={self.config.database})"
