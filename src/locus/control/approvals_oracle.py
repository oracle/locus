# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Approval store in Oracle AI Database: one row per held call, decided under a row lock.

Same connection conventions as
:class:`~locus.memory.backends.oracle.OracleBackend`: a TNS alias or host /
port / service name, a wallet for Autonomous Database, ``python-oracledb`` in
thin mode, and a table created on first use when it does not exist::

    from locus.control import OracleApprovals

    store = OracleApprovals(
        dsn="locusdb_low",
        user="LOCUS",
        password=os.environ["ORACLE_PASSWORD"],
        wallet_location="/opt/wallet",
    )

Every submit, decision and consumption is one transaction that locks the rows
it reads (``SELECT ... FOR UPDATE``), so replicas sharing the table never both
redeem one approval. The record is a ``CLOB CHECK (IS JSON)`` column, written
through a temporary LOB as the checkpointer writes its state. Install with
``pip install locus-sdk[oracledb]``.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from pydantic import SecretStr

from locus._oracle_ddl import provisioning_lock, run_ddl
from locus.control.approvals_db import DatabaseApprovals, record_from_json, record_to_json
from locus.core.loop_bound import loop_bound
from locus.memory.backends._oracle_config import OracleConfig


if TYPE_CHECKING:
    import oracledb

    from locus.control.approvals import ApprovalAuthority, ApprovalRecord


__all__ = ["OracleApprovals"]


async def _document(value: Any) -> str | Mapping[str, Any]:
    """The ``record`` column as JSON text or as the parsed document.

    The driver returns a ``CLOB CHECK (IS JSON)`` column as a LOB to read, as
    text, or -- against Oracle AI Database 26ai -- already parsed into a dict.
    """
    if hasattr(value, "read"):
        value = value.read()
        if inspect.isawaitable(value):
            value = await value
    if isinstance(value, str | Mapping):
        return value
    return bytes(value).decode("utf-8")


class _OracleTransaction:
    def __init__(self, conn: Any, cursor: Any, table: str) -> None:
        self._conn = conn
        self._cursor = cursor
        self._table = table

    async def _rows(self, sql: str, binds: dict[str, Any]) -> list[ApprovalRecord]:
        await self._cursor.execute(sql, binds)
        rows = await self._cursor.fetchall()
        return [record_from_json(await _document(row[0])) for row in rows]

    async def get(self, approval_id: str, *, lock: bool = False) -> ApprovalRecord | None:
        records = await self._rows(
            f"SELECT record FROM {self._table} WHERE approval_id = :approval_id"
            + (" FOR UPDATE" if lock else ""),
            {"approval_id": approval_id},
        )
        return records[0] if records else None

    async def same_call(self, digest: str, *, lock: bool = False) -> list[ApprovalRecord]:
        return await self._rows(
            f"SELECT record FROM {self._table} WHERE digest = :digest"
            + (" FOR UPDATE" if lock else ""),
            {"digest": digest},
        )

    async def _lob(self, record: ApprovalRecord) -> Any:
        import oracledb  # noqa: PLC0415

        lob = await self._conn.createlob(oracledb.DB_TYPE_CLOB)
        await lob.write(record_to_json(record))
        return lob

    async def insert(self, record: ApprovalRecord) -> bool:
        import oracledb  # noqa: PLC0415

        try:
            await self._cursor.execute(
                f"""
                INSERT INTO {self._table} (approval_id, digest, status, created_at, record)
                VALUES (:approval_id, :digest, :status, :created_at, :record)
                """,
                {
                    "approval_id": record.approval_id,
                    "digest": record.digest,
                    "status": record.status,
                    "created_at": record.created_at,
                    "record": await self._lob(record),
                },
            )
        except oracledb.IntegrityError:
            return False
        return True

    async def update(self, record: ApprovalRecord) -> None:
        await self._cursor.execute(
            f"UPDATE {self._table} SET status = :status, record = :record "
            "WHERE approval_id = :approval_id",
            {
                "approval_id": record.approval_id,
                "status": record.status,
                "record": await self._lob(record),
            },
        )

    async def waiting(self) -> list[ApprovalRecord]:
        return await self._rows(
            f"SELECT record FROM {self._table} WHERE status = 'pending' ORDER BY created_at", {}
        )


class OracleApprovals(DatabaseApprovals):
    """An approval store in an Oracle AI Database table.

    Args:
        dsn: A TNS alias from the wallet's ``tnsnames.ora``, or a connect string.
        user: The schema to connect as.
        password: Its password.
        wallet_location: The unzipped wallet directory, for Autonomous Database.
        wallet_password: The wallet password, when the wallet is encrypted.
        host: The server, when no ``dsn``.
        port: Its port.
        service_name: The service.
        table_name: The table; created on first use if missing.
        schema_name: Its schema; the user's own when ``None``.
        min_pool_size: The pool's lower bound.
        max_pool_size: Its upper bound.
        authority: Who may decide, as for
            :class:`~locus.control.approvals.InMemoryApprovals`.
    """

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
        *,
        table_name: str = "locus_approvals",
        schema_name: str | None = None,
        min_pool_size: int = 1,
        max_pool_size: int = 5,
        authority: ApprovalAuthority
        | Callable[[ApprovalRecord], ApprovalAuthority | None]
        | None = None,
    ) -> None:
        super().__init__(authority)
        self.config = OracleConfig(
            dsn=dsn,
            user=user,
            password=SecretStr(password) if isinstance(password, str) else password,
            wallet_location=wallet_location,
            wallet_password=SecretStr(wallet_password)
            if isinstance(wallet_password, str)
            else wallet_password,
            host=host,
            port=port,
            service_name=service_name,
            table_name=table_name,
            schema_name=schema_name,
            min_pool_size=min_pool_size,
            max_pool_size=max_pool_size,
        )
        self._pool: oracledb.AsyncConnectionPool | None = None
        self._pool_loop: Any = None
        self._ready = False

    @property
    def spec(self) -> str:
        config = self.config
        if config.dsn:
            where = f"oracle://{config.user}@/?dsn={config.dsn}"
        else:
            where = f"oracle://{config.user}@{config.host}:{config.port}/{config.service_name}?"
        if config.wallet_location:
            where += f"&wallet={config.wallet_location}"
        return f"{where}&table={self._table}".replace("?&", "?")

    @property
    def _table(self) -> str:
        config = self.config
        return (
            f"{config.schema_name}.{config.table_name}" if config.schema_name else config.table_name
        )

    def _build_pool(self) -> oracledb.AsyncConnectionPool:
        try:
            import oracledb  # noqa: PLC0415
        except ImportError as error:
            raise ImportError(
                "OracleApprovals needs the 'oracledb' package: pip install locus-sdk[oracledb]"
            ) from error
        config = self.config
        dsn = config.dsn
        if dsn is None and config.host and config.service_name:
            dsn = oracledb.makedsn(config.host, config.port, service_name=config.service_name)
        params: dict[str, Any] = {}
        if config.wallet_location:
            params["config_dir"] = config.wallet_location
            params["wallet_location"] = config.wallet_location
            # ``is not None``: "" is the idiom for an auto-login wallet.
            if config.wallet_password is not None:
                params["wallet_password"] = config.wallet_password.get_secret_value()
        self._ready = False
        pool: oracledb.AsyncConnectionPool = oracledb.create_pool_async(
            user=config.user,
            password=config.password.get_secret_value(),
            dsn=dsn,
            min=config.min_pool_size,
            max=config.max_pool_size,
            **params,
        )
        return pool

    async def _get_pool(self) -> oracledb.AsyncConnectionPool:
        # The pool lives on the store's own loop; the shared cache keeps a
        # call from another loop correct rather than mysterious.
        return loop_bound(self, "_pool", self._build_pool)

    async def _ensure_table(self, conn: Any, cursor: Any) -> None:
        if self._ready:
            return
        # One provisioning pass per instance, and a CREATE another process
        # won or is still running is handled (see ``locus._oracle_ddl``).
        async with provisioning_lock(self):
            if self._ready:
                return
            await cursor.execute(
                "SELECT COUNT(*) FROM user_tables WHERE table_name = UPPER(:table_name)",
                {"table_name": self.config.table_name},
            )
            row = await cursor.fetchone()
            if not (row and row[0] > 0):
                for statement in (
                    f"""
                    CREATE TABLE {self._table} (
                        approval_id VARCHAR2(255) PRIMARY KEY,
                        digest VARCHAR2(64) NOT NULL,
                        status VARCHAR2(16) NOT NULL,
                        created_at VARCHAR2(40) NOT NULL,
                        record CLOB CHECK (record IS JSON)
                    )
                    """,
                    f"CREATE INDEX idx_{self.config.table_name}_digest ON {self._table} (digest)",
                    f"CREATE INDEX idx_{self.config.table_name}_status "
                    f"ON {self._table} (status, created_at)",
                ):
                    await run_ddl(cursor, statement)
                await conn.commit()
            self._ready = True

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[_OracleTransaction]:
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.cursor() as cursor:
            await self._ensure_table(conn, cursor)
            try:
                yield _OracleTransaction(conn, cursor, self._table)
            except BaseException:
                await conn.rollback()
                raise
            await conn.commit()

    async def _disconnect(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._ready = False
