# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Oracle Database versioned checkpoint saver — locus-native LangGraph shape.

Closes the biggest remaining parity gap with ``langgraph-oracledb``: this
module provides a versioned, history-preserving checkpoint saver with
pending-writes durability — the same surface area LangGraph's
``BaseCheckpointSaver`` exposes — but with **zero** langchain / langgraph
imports. locus owns the contract end-to-end.

Differences from the sibling :class:`locus.memory.backends.oracle.OracleBackend`:

* ``OracleBackend`` keeps **one row per ``thread_id``** (MERGE upsert),
  so checkpoint history is destructive — the latest write wins.
* ``OracleCheckpointSaver`` keeps **one row per ``(thread_id,
  checkpoint_ns, checkpoint_id)``** so the full graph-step history is
  preserved. ``parent_checkpoint_id`` walks the lineage. A second table
  (``<table_name>_writes``) holds pending intra-step writes, keyed by
  ``task_id`` plus a monotonic ``idx``.

Schema (two tables, auto-created when ``auto_create_table=True``)::

    CREATE TABLE locus_checkpoints (
        thread_id            VARCHAR2(255) NOT NULL,
        checkpoint_ns        VARCHAR2(255) DEFAULT 'default' NOT NULL,
        checkpoint_id        VARCHAR2(255) NOT NULL,
        parent_checkpoint_id VARCHAR2(255),
        checkpoint_data      CLOB CHECK (checkpoint_data IS JSON),
        metadata             CLOB DEFAULT '{}' CHECK (metadata IS JSON),
        created_at           TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP,
        CONSTRAINT pk_locus_checkpoints PRIMARY KEY
            (thread_id, checkpoint_ns, checkpoint_id)
    );
    CREATE INDEX idx_locus_checkpoints_thread
        ON locus_checkpoints (thread_id, checkpoint_ns, created_at DESC);

    CREATE TABLE locus_checkpoint_writes (
        thread_id     VARCHAR2(255) NOT NULL,
        checkpoint_ns VARCHAR2(255) DEFAULT 'default' NOT NULL,
        checkpoint_id VARCHAR2(255) NOT NULL,
        task_id       VARCHAR2(255) NOT NULL,
        idx           NUMBER(10)    NOT NULL,
        channel       VARCHAR2(255) NOT NULL,
        value         CLOB,
        CONSTRAINT pk_locus_checkpoint_writes PRIMARY KEY
            (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
    );

CLOB binds (``checkpoint_data``, ``metadata``, ``value``) are pinned with
``cursor.setinputsizes(... = oracledb.DB_TYPE_CLOB)`` before execute —
without it, thin-mode oracledb fails with ORA-01461 on large state
payloads. Same fix langgraph-oracledb adopted in
oracle/langchain-oracle#224.

``put_writes`` is idempotent: it deletes any existing rows for the
``(thread, ns, checkpoint, task)`` triple inside one cursor pass before
inserting the new set. Retries are safe.

Uses ``python-oracledb`` in thin mode; the dependency is imported lazily
inside :meth:`_get_pool` so installs without the driver can still import
this module.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, SecretStr

from locus._oracle_pool_cache import safe_acquire
from locus.memory.backends._oracle_config import (
    OracleConfig,
)
from locus.memory.backends._oracle_config import (
    validate_sql_identifier as _validate_sql_identifier,
)


if TYPE_CHECKING:
    import oracledb


class OracleCheckpointSaver(BaseModel):
    """Versioned Oracle checkpoint saver with pending-writes durability.

    locus-native — does **not** inherit from
    ``langgraph.checkpoint.base.BaseCheckpointSaver``. The method surface
    matches the LangGraph shape (``put`` / ``get`` / ``list_checkpoints``
    / ``put_writes`` / ``get_writes`` / ``delete_thread``) so adapter
    layers can wire it into LangGraph runtimes without dragging a
    langchain dependency into locus.

    Example with TNS alias::

        >>> saver = OracleCheckpointSaver(
        ...     dsn="mydb_low",
        ...     user="admin",
        ...     password="secret",
        ...     wallet_location="/path/to/wallet",
        ...     table_name="locus",
        ... )
        >>> await saver.put(
        ...     thread_id="t1",
        ...     checkpoint_id="c1",
        ...     checkpoint_data={"step": 0, "values": {"x": 1}},
        ... )
        >>> latest = await saver.get(thread_id="t1")

    Example with pending writes (intra-step durability)::

        >>> await saver.put_writes(
        ...     thread_id="t1",
        ...     checkpoint_id="c1",
        ...     task_id="node-a",
        ...     writes=[("x", 2), ("y", 3)],
        ... )
    """

    config: OracleConfig = Field(default_factory=OracleConfig)
    auto_create_table: bool = True
    _pool: oracledb.AsyncConnectionPool | None = None
    _pool_loop: Any = None
    _initialized: bool = False

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        dsn: str | None = None,
        user: str = "admin",
        password: str | SecretStr = "",
        wallet_location: str | None = None,
        wallet_password: str | SecretStr | None = None,
        host: str | None = None,
        port: int = 1521,
        service_name: str | None = None,
        table_name: str = "locus",
        schema_name: str | None = None,
        min_pool_size: int = 1,
        max_pool_size: int = 5,
        auto_create_table: bool = True,
        **kwargs: Any,
    ) -> None:
        # The two physical table names are derived from ``table_name``
        # so callers configure one base prefix. Validate the prefix here
        # (re-validate the derived suffixed name too) before letting
        # OracleConfig do its own validation.
        _validate_sql_identifier(table_name, "table_name")
        _validate_sql_identifier(f"{table_name}_checkpoints", "table_name")
        _validate_sql_identifier(f"{table_name}_writes", "table_name")
        if schema_name is not None:
            _validate_sql_identifier(schema_name, "schema_name")

        config = OracleConfig(
            dsn=dsn,
            user=user,
            password=SecretStr(password) if isinstance(password, str) else password,
            wallet_location=wallet_location,
            wallet_password=(
                SecretStr(wallet_password) if isinstance(wallet_password, str) else wallet_password
            ),
            host=host,
            port=port,
            service_name=service_name,
            # OracleConfig validates table_name itself — we pass the
            # checkpoint-table form so its identifier check covers the
            # actual DDL target.
            table_name=f"{table_name}_checkpoints",
            schema_name=schema_name,
            min_pool_size=min_pool_size,
            max_pool_size=max_pool_size,
            **kwargs,
        )
        super().__init__(config=config, auto_create_table=auto_create_table)
        # Stash the base prefix so derived names (_writes table) can
        # share it without re-deriving from the suffixed form.
        self.__dict__["_table_prefix"] = table_name

    # ------------------------------------------------------------------
    # Pool / table helpers
    # ------------------------------------------------------------------

    @property
    def _table_prefix(self) -> str:
        """User-supplied base name (no suffix)."""
        prefix: str = self.__dict__["_table_prefix"]
        return prefix

    @property
    def _checkpoints_table(self) -> str:
        if self.config.schema_name:
            return f"{self.config.schema_name}.{self._table_prefix}_checkpoints"
        return f"{self._table_prefix}_checkpoints"

    @property
    def _writes_table(self) -> str:
        if self.config.schema_name:
            return f"{self.config.schema_name}.{self._table_prefix}_writes"
        return f"{self._table_prefix}_writes"

    @property
    def _checkpoints_table_unqualified(self) -> str:
        return f"{self._table_prefix}_checkpoints"

    @property
    def _writes_table_unqualified(self) -> str:
        return f"{self._table_prefix}_writes"

    async def _get_pool(self) -> oracledb.AsyncConnectionPool:
        """Lazily create the oracledb async pool, rebuilding on loop change."""
        from locus._oracle_pool_cache import get_pool

        def _build() -> oracledb.AsyncConnectionPool:
            try:
                import oracledb
            except ImportError as e:
                raise ImportError(
                    "OracleCheckpointSaver requires the 'oracledb' package. "
                    "Install with: pip install oracledb"
                ) from e

            cfg = self.config
            dsn = cfg.dsn
            if dsn is None and cfg.host and cfg.service_name:
                dsn = oracledb.makedsn(cfg.host, cfg.port, service_name=cfg.service_name)

            params: dict[str, Any] = {}
            if cfg.wallet_location:
                params["config_dir"] = cfg.wallet_location
                params["wallet_location"] = cfg.wallet_location
                # ``is not None``: SecretStr("") is falsy in Pydantic v2, but
                # "" is the auto-login (cwallet.sso) wallet idiom. See #289.
                if cfg.wallet_password is not None:
                    params["wallet_password"] = cfg.wallet_password.get_secret_value()

            return oracledb.create_pool_async(
                user=cfg.user,
                password=cfg.password.get_secret_value(),
                dsn=dsn,
                min=cfg.min_pool_size,
                max=cfg.max_pool_size,
                **params,
            )

        return await get_pool(self, _build)

    def _checkpoints_ddl(self) -> str:
        return (
            f"CREATE TABLE {self._checkpoints_table} (\n"
            f"    thread_id            VARCHAR2(255) NOT NULL,\n"
            f"    checkpoint_ns        VARCHAR2(255) DEFAULT 'default' NOT NULL,\n"
            f"    checkpoint_id        VARCHAR2(255) NOT NULL,\n"
            f"    parent_checkpoint_id VARCHAR2(255),\n"
            f"    checkpoint_data      CLOB CHECK (checkpoint_data IS JSON),\n"
            f"    metadata             CLOB DEFAULT '{{}}' CHECK (metadata IS JSON),\n"
            f"    created_at           TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP,\n"
            f"    CONSTRAINT pk_{self._table_prefix}_checkpoints PRIMARY KEY "
            f"(thread_id, checkpoint_ns, checkpoint_id)\n"
            f")"
        )

    def _checkpoints_index_ddl(self) -> str:
        return (
            f"CREATE INDEX idx_{self._table_prefix}_checkpoints_thread "
            f"ON {self._checkpoints_table} (thread_id, checkpoint_ns, created_at DESC)"
        )

    def _writes_ddl(self) -> str:
        return (
            f"CREATE TABLE {self._writes_table} (\n"
            f"    thread_id     VARCHAR2(255) NOT NULL,\n"
            f"    checkpoint_ns VARCHAR2(255) DEFAULT 'default' NOT NULL,\n"
            f"    checkpoint_id VARCHAR2(255) NOT NULL,\n"
            f"    task_id       VARCHAR2(255) NOT NULL,\n"
            f"    idx           NUMBER(10)    NOT NULL,\n"
            f"    channel       VARCHAR2(255) NOT NULL,\n"
            f"    value         CLOB,\n"
            f"    CONSTRAINT pk_{self._table_prefix}_writes PRIMARY KEY "
            f"(thread_id, checkpoint_ns, checkpoint_id, task_id, idx)\n"
            f")"
        )

    async def _ensure_tables(self) -> None:
        """Create both tables once per instance, idempotently.

        Skipped entirely when ``auto_create_table=False`` (DBA-managed
        mode): the first DML statement surfaces ORA-00942 if the table
        is actually missing.
        """
        if self._initialized:
            return

        if not self.auto_create_table:
            self._initialized = True
            return

        pool = await self._get_pool()
        async with safe_acquire(self, pool) as conn, conn.cursor() as cursor:
            # Check checkpoints table
            await cursor.execute(
                """
                SELECT COUNT(*) FROM user_tables
                WHERE table_name = UPPER(:table_name)
                """,
                {"table_name": self._checkpoints_table_unqualified},
            )
            row = await cursor.fetchone()
            checkpoints_exists = bool(row and row[0] > 0)

            if not checkpoints_exists:
                await cursor.execute(self._checkpoints_ddl())
                await cursor.execute(self._checkpoints_index_ddl())

            # Check writes table
            await cursor.execute(
                """
                SELECT COUNT(*) FROM user_tables
                WHERE table_name = UPPER(:table_name)
                """,
                {"table_name": self._writes_table_unqualified},
            )
            row = await cursor.fetchone()
            writes_exists = bool(row and row[0] > 0)

            if not writes_exists:
                await cursor.execute(self._writes_ddl())

            if not (checkpoints_exists and writes_exists):
                await conn.commit()

        self._initialized = True

    # ------------------------------------------------------------------
    # CLOB binding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pin_checkpoint_clobs(cursor: Any) -> None:
        """Pin ``checkpoint_data`` and ``metadata`` to CLOB.

        Without these hints, thin-mode oracledb can fail with ORA-01461
        (character-to-LOB conversion) on large state payloads. Same
        hardening :class:`OracleBackend` applies for its ``:data`` /
        ``:metadata`` binds.
        """
        import oracledb as _oracledb

        cursor.setinputsizes(
            checkpoint_data=_oracledb.DB_TYPE_CLOB,
            metadata=_oracledb.DB_TYPE_CLOB,
        )

    @staticmethod
    def _pin_value_clob(cursor: Any) -> None:
        """Pin ``value`` to CLOB for writes inserts."""
        import oracledb as _oracledb

        cursor.setinputsizes(value=_oracledb.DB_TYPE_CLOB)

    @staticmethod
    async def _decode_clob(raw: Any) -> Any:
        """Read a CLOB / JSON column into a Python ``dict`` (or raw string).

        oracledb returns three different shapes depending on the column
        type + payload size:

        * ``AsyncLOB`` — async ``.read()`` returns a coroutine that
          must be awaited.
        * Plain ``str`` — short payloads round-trip without a LOB.
        * ``dict`` — Oracle 23ai may decode JSON-constrained columns
          straight into Python dicts.

        ``json.loads`` is best-effort; a value that doesn't parse as
        JSON is returned verbatim (matches the ``value`` column on the
        writes table, which may hold any serialised payload).
        """
        if raw is None:
            return None
        if hasattr(raw, "read"):
            data = raw.read()
            if hasattr(data, "__await__"):
                data = await data
            raw = data
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            if not raw:
                return None
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
        return raw

    # ------------------------------------------------------------------
    # Checkpoint CRUD
    # ------------------------------------------------------------------

    async def put(
        self,
        *,
        thread_id: str,
        checkpoint_id: str,
        checkpoint_data: dict,
        checkpoint_ns: str = "default",
        parent_checkpoint_id: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Persist one checkpoint row.

        History is preserved: ``(thread_id, checkpoint_ns,
        checkpoint_id)`` is the primary key, so re-saving a different
        ``checkpoint_id`` for the same thread inserts a new row rather
        than overwriting.
        """
        await self._ensure_tables()
        pool = await self._get_pool()

        async with safe_acquire(self, pool) as conn, conn.cursor() as cursor:
            self._pin_checkpoint_clobs(cursor)
            await cursor.execute(
                f"""
                INSERT INTO {self._checkpoints_table} (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    parent_checkpoint_id,
                    checkpoint_data,
                    metadata
                ) VALUES (
                    :thread_id,
                    :checkpoint_ns,
                    :checkpoint_id,
                    :parent_checkpoint_id,
                    :checkpoint_data,
                    :metadata
                )
                """,
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                    "parent_checkpoint_id": parent_checkpoint_id,
                    "checkpoint_data": json.dumps(checkpoint_data, default=str),
                    "metadata": json.dumps(metadata or {}, default=str),
                },
            )
            await conn.commit()

    async def get(
        self,
        *,
        thread_id: str,
        checkpoint_id: str | None = None,
        checkpoint_ns: str = "default",
    ) -> dict | None:
        """Fetch one checkpoint.

        ``checkpoint_id=None`` returns the most recent row for the
        ``(thread_id, checkpoint_ns)`` pair, ordered by ``created_at
        DESC`` with ``FETCH FIRST 1 ROWS ONLY``.

        Returns ``None`` when no matching row exists; otherwise a dict
        with keys ``checkpoint_id``, ``parent_checkpoint_id``,
        ``checkpoint``, ``metadata``, ``created_at``.
        """
        await self._ensure_tables()
        pool = await self._get_pool()

        async with safe_acquire(self, pool) as conn, conn.cursor() as cursor:
            if checkpoint_id is None:
                # Latest row for the thread.
                await cursor.execute(
                    f"""
                    SELECT
                        checkpoint_id,
                        parent_checkpoint_id,
                        checkpoint_data,
                        metadata,
                        created_at
                    FROM {self._checkpoints_table}
                    WHERE thread_id = :thread_id
                      AND checkpoint_ns = :checkpoint_ns
                    ORDER BY created_at DESC
                    FETCH FIRST 1 ROWS ONLY
                    """,
                    {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns},
                )
            else:
                await cursor.execute(
                    f"""
                    SELECT
                        checkpoint_id,
                        parent_checkpoint_id,
                        checkpoint_data,
                        metadata,
                        created_at
                    FROM {self._checkpoints_table}
                    WHERE thread_id = :thread_id
                      AND checkpoint_ns = :checkpoint_ns
                      AND checkpoint_id = :checkpoint_id
                    """,
                    {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    },
                )
            row = await cursor.fetchone()

        if row is None:
            return None

        return {
            "checkpoint_id": row[0],
            "parent_checkpoint_id": row[1],
            "checkpoint": await self._decode_clob(row[2]),
            "metadata": await self._decode_clob(row[3]) or {},
            "created_at": (row[4].isoformat() if isinstance(row[4], datetime) else row[4]),
        }

    async def list_checkpoints(
        self,
        *,
        thread_id: str,
        checkpoint_ns: str = "default",
        limit: int = 10,
        before: str | None = None,
    ) -> list[dict]:
        """Return checkpoints for a thread, newest first.

        ``before`` is a ``checkpoint_id``; when supplied, only rows
        strictly older than that row's ``created_at`` are returned —
        the typical "page back through history" pattern LangGraph's
        ``alist`` exposes.
        """
        await self._ensure_tables()
        pool = await self._get_pool()

        async with safe_acquire(self, pool) as conn, conn.cursor() as cursor:
            if before is None:
                await cursor.execute(
                    f"""
                    SELECT
                        checkpoint_id,
                        parent_checkpoint_id,
                        checkpoint_data,
                        metadata,
                        created_at
                    FROM {self._checkpoints_table}
                    WHERE thread_id = :thread_id
                      AND checkpoint_ns = :checkpoint_ns
                    ORDER BY created_at DESC
                    FETCH FIRST :lim ROWS ONLY
                    """,
                    {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "lim": limit,
                    },
                )
            else:
                await cursor.execute(
                    f"""
                    SELECT
                        checkpoint_id,
                        parent_checkpoint_id,
                        checkpoint_data,
                        metadata,
                        created_at
                    FROM {self._checkpoints_table}
                    WHERE thread_id = :thread_id
                      AND checkpoint_ns = :checkpoint_ns
                      AND created_at < (
                          SELECT created_at FROM {self._checkpoints_table}
                          WHERE thread_id = :thread_id
                            AND checkpoint_ns = :checkpoint_ns
                            AND checkpoint_id = :before
                      )
                    ORDER BY created_at DESC
                    FETCH FIRST :lim ROWS ONLY
                    """,
                    {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "before": before,
                        "lim": limit,
                    },
                )
            rows = await cursor.fetchall()

        results: list[dict] = []
        for row in rows:
            results.append(
                {
                    "checkpoint_id": row[0],
                    "parent_checkpoint_id": row[1],
                    "checkpoint": await self._decode_clob(row[2]),
                    "metadata": await self._decode_clob(row[3]) or {},
                    "created_at": (row[4].isoformat() if isinstance(row[4], datetime) else row[4]),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Pending writes
    # ------------------------------------------------------------------

    async def put_writes(
        self,
        *,
        thread_id: str,
        checkpoint_id: str,
        task_id: str,
        writes: list[tuple[str, Any]],
        checkpoint_ns: str = "default",
    ) -> None:
        """Persist pending writes for one ``(checkpoint, task)`` pair.

        Idempotent: existing writes for the same
        ``(thread_id, checkpoint_ns, checkpoint_id, task_id)`` triple
        are deleted first, then the new set is inserted with monotonic
        ``idx`` 0..N-1. Retries are safe.
        """
        await self._ensure_tables()
        pool = await self._get_pool()

        async with safe_acquire(self, pool) as conn, conn.cursor() as cursor:
            # Delete any prior writes for this task. ``put_writes`` is a
            # replace, not an append.
            await cursor.execute(
                f"""
                DELETE FROM {self._writes_table}
                WHERE thread_id = :thread_id
                  AND checkpoint_ns = :checkpoint_ns
                  AND checkpoint_id = :checkpoint_id
                  AND task_id = :task_id
                """,
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                    "task_id": task_id,
                },
            )

            for idx, (channel, value) in enumerate(writes):
                self._pin_value_clob(cursor)
                await cursor.execute(
                    f"""
                    INSERT INTO {self._writes_table} (
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        task_id,
                        idx,
                        channel,
                        value
                    ) VALUES (
                        :thread_id,
                        :checkpoint_ns,
                        :checkpoint_id,
                        :task_id,
                        :idx,
                        :channel,
                        :value
                    )
                    """,
                    {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                        "task_id": task_id,
                        "idx": idx,
                        "channel": channel,
                        "value": json.dumps(value, default=str),
                    },
                )

            await conn.commit()

    async def get_writes(
        self,
        *,
        thread_id: str,
        checkpoint_id: str,
        checkpoint_ns: str = "default",
        task_id: str | None = None,
    ) -> list[dict]:
        """Fetch pending writes, ordered by ``(task_id, idx)``.

        ``task_id=None`` returns writes for every task at the
        checkpoint; otherwise only the named task's rows.
        """
        await self._ensure_tables()
        pool = await self._get_pool()

        async with safe_acquire(self, pool) as conn, conn.cursor() as cursor:
            if task_id is None:
                await cursor.execute(
                    f"""
                    SELECT task_id, idx, channel, value
                    FROM {self._writes_table}
                    WHERE thread_id = :thread_id
                      AND checkpoint_ns = :checkpoint_ns
                      AND checkpoint_id = :checkpoint_id
                    ORDER BY task_id, idx
                    """,
                    {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    },
                )
            else:
                await cursor.execute(
                    f"""
                    SELECT task_id, idx, channel, value
                    FROM {self._writes_table}
                    WHERE thread_id = :thread_id
                      AND checkpoint_ns = :checkpoint_ns
                      AND checkpoint_id = :checkpoint_id
                      AND task_id = :task_id
                    ORDER BY idx
                    """,
                    {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                        "task_id": task_id,
                    },
                )
            rows = await cursor.fetchall()

        results: list[dict] = []
        for row in rows:
            results.append(
                {
                    "task_id": row[0],
                    "idx": row[1],
                    "channel": row[2],
                    "value": await self._decode_clob(row[3]),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Thread deletion
    # ------------------------------------------------------------------

    async def delete_thread(self, thread_id: str) -> None:
        """Cascade-delete every checkpoint + pending write for a thread."""
        await self._ensure_tables()
        pool = await self._get_pool()

        async with safe_acquire(self, pool) as conn, conn.cursor() as cursor:
            await cursor.execute(
                f"DELETE FROM {self._writes_table} WHERE thread_id = :thread_id",
                {"thread_id": thread_id},
            )
            await cursor.execute(
                f"DELETE FROM {self._checkpoints_table} WHERE thread_id = :thread_id",
                {"thread_id": thread_id},
            )
            await conn.commit()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def __repr__(self) -> str:
        if self.config.dsn:
            return (
                f"OracleCheckpointSaver(dsn={self.config.dsn!r}, table_name={self._table_prefix!r})"
            )
        if self.config.host:
            return (
                f"OracleCheckpointSaver(host={self.config.host!r}, "
                f"service={self.config.service_name!r}, "
                f"table_name={self._table_prefix!r})"
            )
        return f"OracleCheckpointSaver(table_name={self._table_prefix!r})"
