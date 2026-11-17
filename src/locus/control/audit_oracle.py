# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The audit trail on an Oracle AI Database immutable or blockchain table.

:class:`OracleAuditTrail` is an :class:`~locus.control.audit.AuditTrail` whose
records are rows of a table the database itself refuses to update or delete.
The hash chain is unchanged — every row commits to the one before it — and the
table adds what a chain cannot provide on its own: nobody with the schema
password can rewrite or drop a row inside the retention window, so the agent
(or an operator) cannot edit the record after the fact.

Two table kinds, chosen with ``table_kind``:

* ``"immutable"`` (default) — ``CREATE IMMUTABLE TABLE … NO DROP UNTIL n DAYS
  IDLE NO DELETE UNTIL n DAYS AFTER INSERT``. Rows are insert-only and cannot
  be deleted before ``retention_days``; the table cannot be dropped until it
  has been idle for ``drop_idle_days``.
* ``"blockchain"`` — the same, plus the database chains every row with its own
  SHA2-512 hash (``ORABCTAB_HASH$``) and :meth:`OracleAuditTrail.check` asks
  the database to re-verify those hashes with
  ``DBMS_BLOCKCHAIN_TABLE.VERIFY_ROWS``. Choose it when the auditor wants a
  check the application code did not write.

Same public API as the in-memory trail, so ``admit(trail=…)``,
``gate_tool(trail=…)`` and ``governed_agent(audit_trail=…)`` work unchanged::

    trail = OracleAuditTrail(
        dsn="mydb_low",
        user="AUDIT",
        password=pw,
        wallet_location="~/wallets/mydb",
        table_kind="blockchain",
    )
    await admit(action, perform, policy=policy, trail=trail)
    trail.verify()  # re-reads the rows, recomputes the chain, asks the DB

**Sync and async.** :meth:`record` is synchronous and the row is committed
before it returns — the same durability contract as the file-backed trail,
which fsyncs before returning — because ``admit()`` records a decision without
awaiting. Async code that must not block its loop uses :meth:`arecord`,
:meth:`acheck` and :meth:`averify`, which run the same work on a worker
thread. Connections come from a small synchronous ``oracledb`` pool built with
the connection arguments the Oracle checkpointer takes (``dsn`` or
``host``/``port``/``service_name``, ``wallet_location``, ``wallet_password``).

Needs ``oracledb`` (``pip install "locus-sdk[oracledb]"``).
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal

from locus._oracle_ddl import run_ddl_sync
from locus.control.audit import (
    _GENESIS,
    AuditRecord,
    AuditReport,
    AuditSigner,
    AuditTrail,
    _check,
)


if TYPE_CHECKING:
    from locus.memory.backends._oracle_config import OracleConfig

TableKind = Literal["immutable", "blockchain"]

#: Oracle's minimum for ``NO DELETE UNTIL n DAYS AFTER INSERT``.
MIN_RETENTION_DAYS = 16

_COLUMNS = "seq, ts, event_type, payload, prev_hash, entry_hash, key_id, signature"

#: Binding a string over this size inline risks the TTC packet-boundary
#: corruption the checkpointer documents; larger payloads go through a LOB.
_INLINE_LIMIT = 32_000


class OracleAuditTrail(AuditTrail):
    """An :class:`AuditTrail` whose rows live in an Oracle immutable or blockchain table.

    Args:
        dsn: TNS alias or connect string. Alternatively ``host``, ``port`` and
            ``service_name``.
        user: Schema user. Give the trail its own schema with only
            ``CREATE TABLE`` and ``INSERT`` on it where you can.
        password: The schema user's password.
        wallet_location: Wallet directory, for Autonomous Database with mTLS.
        wallet_password: Wallet password, when the wallet is encrypted.
        host: Database host, when not using ``dsn``.
        port: Listener port (default 1521).
        service_name: Service name, paired with ``host``.
        table_name: Table the rows go to. Created on first use when
            ``auto_create_table`` is on; must be a safe SQL identifier.
        schema_name: Owner of the table when it is not the connecting user's.
        trail_id: Several trails can share one table; rows are chained per
            ``trail_id``. Defaults to ``"default"``.
        table_kind: ``"immutable"`` or ``"blockchain"`` (see the module docs).
        retention_days: ``NO DELETE UNTIL n DAYS AFTER INSERT`` — at least
            16. ``None`` writes ``NO DELETE LOCKED``: rows can never be deleted.
        drop_idle_days: ``NO DROP UNTIL n DAYS IDLE``. ``None`` writes
            ``NO DROP``: the table can never be dropped.
        auto_create_table: Create the table when it does not exist. Turn off
            when a DBA provisions it; the trail then only inserts and reads.
        clock: Timestamp source, as on :class:`AuditTrail`.
        signer: Signs every record, as on :class:`AuditTrail`.
        min_pool_size: Smallest size of the synchronous ``oracledb`` pool.
        max_pool_size: Largest size of the pool; two connections is plenty.
    """

    def __init__(
        self,
        *,
        dsn: str | None = None,
        user: str = "admin",
        password: str = "",
        wallet_location: str | None = None,
        wallet_password: str | None = None,
        host: str | None = None,
        port: int = 1521,
        service_name: str | None = None,
        table_name: str = "locus_audit_trail",
        schema_name: str | None = None,
        trail_id: str = "default",
        table_kind: TableKind = "immutable",
        retention_days: int | None = MIN_RETENTION_DAYS,
        drop_idle_days: int | None = MIN_RETENTION_DAYS,
        auto_create_table: bool = True,
        clock: Callable[[], str] | None = None,
        signer: AuditSigner | None = None,
        min_pool_size: int = 1,
        max_pool_size: int = 2,
    ) -> None:
        super().__init__(clock=clock, signer=signer)
        from pydantic import SecretStr  # noqa: PLC0415

        from locus.memory.backends._oracle_config import OracleConfig  # noqa: PLC0415

        if table_kind not in ("immutable", "blockchain"):
            raise ValueError(f"table_kind must be 'immutable' or 'blockchain', not {table_kind!r}")
        if retention_days is not None and retention_days < MIN_RETENTION_DAYS:
            raise ValueError(
                f"retention_days must be at least {MIN_RETENTION_DAYS} (Oracle's minimum) or None"
            )
        if drop_idle_days is not None and drop_idle_days < 0:
            raise ValueError("drop_idle_days must be 0 or more, or None")
        if not trail_id or len(trail_id) > 255:  # noqa: PLR2004 — the column width
            raise ValueError("trail_id must be 1 to 255 characters")
        self.config: OracleConfig = OracleConfig(
            dsn=dsn,
            user=user,
            password=SecretStr(password),
            wallet_location=wallet_location,
            wallet_password=SecretStr(wallet_password) if wallet_password is not None else None,
            host=host,
            port=port,
            service_name=service_name,
            table_name=table_name,
            schema_name=schema_name,
            min_pool_size=min_pool_size,
            max_pool_size=max_pool_size,
        )
        self.trail_id = trail_id
        self.table_kind: TableKind = table_kind
        self.retention_days = retention_days
        self.drop_idle_days = drop_idle_days
        self._auto_create = auto_create_table
        self._pool: Any = None
        self._table_ready = False
        # The first-time DDL runs once per instance (see ``locus._oracle_ddl``).
        self._provision_lock = threading.Lock()
        self._loaded = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    @property
    def _full_table_name(self) -> str:
        if self.config.schema_name:
            return f"{self.config.schema_name}.{self.config.table_name}"
        return self.config.table_name

    def _build_pool(self) -> Any:
        try:
            import oracledb  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ImportError(
                'OracleAuditTrail needs the oracledb package:\n    pip install "locus-sdk[oracledb]"'
            ) from exc
        cfg = self.config
        dsn = cfg.dsn
        if dsn is None and cfg.host and cfg.service_name:
            dsn = oracledb.makedsn(cfg.host, cfg.port, service_name=cfg.service_name)
        params: dict[str, Any] = {}
        if cfg.wallet_location:
            params["config_dir"] = cfg.wallet_location
            params["wallet_location"] = cfg.wallet_location
            if cfg.wallet_password is not None:
                params["wallet_password"] = cfg.wallet_password.get_secret_value()
        return oracledb.create_pool(
            user=cfg.user,
            password=cfg.password.get_secret_value(),
            dsn=dsn,
            min=cfg.min_pool_size,
            max=cfg.max_pool_size,
            **params,
        )

    def _run(self, op: Callable[[Any], Any]) -> Any:
        """Run ``op(conn)`` on a pooled connection, rebuilding the pool once on a dead socket."""
        from locus._oracle_pool_cache import is_reconnectable  # noqa: PLC0415

        # A synchronous pool: not bound to an event loop, so cached plainly.
        pool = self._pool if self._pool is not None else self._build_pool()
        self._pool = pool
        try:
            with pool.acquire() as conn:
                return op(conn)
        except Exception as exc:
            if not is_reconnectable(exc):
                raise
            self.close()
            self._pool = self._build_pool()
            with self._pool.acquire() as conn:
                return op(conn)

    def close(self) -> None:
        """Close the connection pool. The next call opens a new one."""
        if self._pool is not None:
            try:
                self._pool.close(force=True)
            except Exception:  # noqa: BLE001 — best-effort teardown
                pass
            self._pool = None
        self._table_ready = False

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------

    def ddl(self) -> str:
        """The ``CREATE … TABLE`` statement this trail runs (or expects a DBA to have run)."""
        kind = "BLOCKCHAIN" if self.table_kind == "blockchain" else "IMMUTABLE"
        drop = (
            "NO DROP"
            if self.drop_idle_days is None
            else f"NO DROP UNTIL {self.drop_idle_days} DAYS IDLE"
        )
        delete = (
            "NO DELETE LOCKED"
            if self.retention_days is None
            else f"NO DELETE UNTIL {self.retention_days} DAYS AFTER INSERT"
        )
        hashing = ' HASHING USING "SHA2_512" VERSION "v1"' if kind == "BLOCKCHAIN" else ""
        return (
            f"CREATE {kind} TABLE {self._full_table_name} (\n"
            "    trail_id   VARCHAR2(255) NOT NULL,\n"
            "    seq        NUMBER(19) NOT NULL,\n"
            "    ts         VARCHAR2(64) NOT NULL,\n"
            "    event_type VARCHAR2(255) NOT NULL,\n"
            "    payload    CLOB NOT NULL,\n"
            "    prev_hash  VARCHAR2(64) NOT NULL,\n"
            "    entry_hash VARCHAR2(64) NOT NULL,\n"
            "    key_id     VARCHAR2(255),\n"
            "    signature  VARCHAR2(4000),\n"
            f"    CONSTRAINT {self.config.table_name[:120]}_uq UNIQUE (trail_id, seq)\n"
            f") {drop} {delete}{hashing}"
        )

    def _ensure_table(self, conn: Any) -> None:
        if self._table_ready:
            return
        with self._provision_lock:
            if self._table_ready:
                return
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM all_tables WHERE table_name = UPPER(:t) "
                    "AND owner = NVL(UPPER(:s), SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA'))",
                    {"t": self.config.table_name, "s": self.config.schema_name},
                )
                row = cur.fetchone()
                exists = bool(row and row[0] > 0)
                if not exists:
                    if not self._auto_create:
                        raise RuntimeError(
                            f"audit table {self._full_table_name} does not exist and "
                            "auto_create_table is off; create it with OracleAuditTrail.ddl()"
                        )
                    # A CREATE another process won or is still running is handled.
                    run_ddl_sync(cur, self.ddl())
                    conn.commit()
            self._table_ready = True

    # ------------------------------------------------------------------
    # Reading the chain back
    # ------------------------------------------------------------------

    def _fetch_rows(self, conn: Any) -> list[tuple[Any, ...]]:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {_COLUMNS} FROM {self._full_table_name} "  # noqa: S608 — validated identifier
                "WHERE trail_id = :trail_id ORDER BY seq",
                {"trail_id": self.trail_id},
            )
            rows: list[tuple[Any, ...]] = list(cur.fetchall())
        return rows

    def _load_from(self, conn: Any) -> None:
        """Read the chain into memory; the caller holds ``self._lock``."""
        self._ensure_table(conn)
        records: list[AuditRecord] = []
        unreadable = 0
        for row in self._fetch_rows(conn):
            try:
                records.append(_record_from_row(row))
            except (TypeError, ValueError):
                unreadable += 1
        self._records = records
        self._unreadable = unreadable
        self._loaded = True

    def reload(self) -> None:
        """Replace the in-memory copy of the chain with what the table holds now.

        The lock is taken before the connection, the order ``record`` uses,
        so no thread ever holds a pooled connection while waiting for the
        lock. The other order deadlocked a burst of writers on the
        two-connection pool: one thread held the lock inside ``record`` and
        waited for a connection, while the threads holding both connections
        inside this method waited for the lock.
        """
        with self._lock:
            self._run(self._load_from)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.reload()

    # ------------------------------------------------------------------
    # AuditTrail API
    # ------------------------------------------------------------------

    @property
    def head(self) -> str:
        self._ensure_loaded()
        return self._records[-1].hash if self._records else _GENESIS

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._records)

    def records(self) -> list[AuditRecord]:
        self._ensure_loaded()
        return list(self._records)

    def export_jsonl(self) -> str:
        self._ensure_loaded()
        return super().export_jsonl()

    def record(self, event_type: str, payload: Mapping[str, Any] | None = None) -> AuditRecord:
        """Append a record; the row is committed before this returns.

        When another writer has appended to the same ``trail_id`` since this
        trail last read the table, the unique ``(trail_id, seq)`` constraint
        rejects the row; the trail reloads the chain and tries once more on
        the new head, so two processes can share one trail without one of
        them forking the chain.
        """
        self._ensure_loaded()
        try:
            return super().record(event_type, payload)
        except Exception as exc:
            if "ORA-00001" not in str(exc):
                raise
            self.reload()
            return super().record(event_type, payload)

    def _persist(self, rec: AuditRecord) -> None:
        self._run(lambda conn: self._insert(conn, rec))

    def _insert(self, conn: Any, rec: AuditRecord) -> None:
        self._ensure_table(conn)
        body = json.dumps(rec.payload, default=str)
        payload: Any = body
        if len(body) > _INLINE_LIMIT:
            import oracledb  # noqa: PLC0415

            payload = conn.createlob(oracledb.DB_TYPE_CLOB)
            payload.write(body)
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {self._full_table_name} "  # noqa: S608 — validated identifier
                "(trail_id, seq, ts, event_type, payload, prev_hash, entry_hash, key_id, signature) "
                "VALUES (:trail_id, :seq, :ts, :event_type, :payload, :prev_hash, :entry_hash, "
                ":key_id, :signature)",
                {
                    "trail_id": self.trail_id,
                    "seq": rec.seq,
                    "ts": rec.ts,
                    "event_type": rec.event_type,
                    "payload": payload,
                    "prev_hash": rec.prev_hash,
                    "entry_hash": rec.hash,
                    "key_id": rec.key_id,
                    "signature": rec.signature,
                },
            )
        conn.commit()

    def check(
        self,
        *,
        expected_head: str | None = None,
        keys: Mapping[str, bytes | str] | None = None,
    ) -> AuditReport:
        """Re-read the rows, recompute the chain, and — on a blockchain table — ask the database.

        The in-memory copy is not trusted: every check reads the table again
        and recomputes each hash from the row's contents. On a blockchain
        table the report also fails when any row lacks its database-side hash
        (``ORABCTAB_HASH$``) or when ``DBMS_BLOCKCHAIN_TABLE.VERIFY_ROWS``
        reports fewer verified rows than the table holds, or raises.
        """

        def _do(conn: Any) -> AuditReport:
            self._load_from(conn)
            report = _check(self._records, self._unreadable, expected_head=expected_head, keys=keys)
            if not report.ok or self.table_kind != "blockchain":
                return report
            problem = self._blockchain_problem(conn)
            if problem:
                return AuditReport(
                    ok=False, records=report.records, head=report.head, problem=problem
                )
            return report

        result: AuditReport = self._run(_do)
        return result

    def _blockchain_problem(self, conn: Any) -> str:
        """Why the database-side hashes do not stand up, or ``""`` when they do."""
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM {self._full_table_name} "  # noqa: S608 — validated identifier
                "WHERE trail_id = :trail_id AND ORABCTAB_HASH$ IS NULL",
                {"trail_id": self.trail_id},
            )
            row = cur.fetchone()
            missing = int(row[0]) if row else 0
            if missing:
                return (
                    f"{missing} row(s) carry no database-side hash: this is not a blockchain table"
                )
            cur.execute(f"SELECT COUNT(*) FROM {self._full_table_name}")  # noqa: S608
            row = cur.fetchone()
            total = int(row[0]) if row else 0
            verified = cur.var(int)
            try:
                cur.execute(
                    "BEGIN DBMS_BLOCKCHAIN_TABLE.VERIFY_ROWS("
                    "schema_name => NVL(:s, SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA')), "
                    "table_name => :t, number_of_rows_verified => :n, "
                    "verify_signature => FALSE); END;",
                    {
                        "s": self.config.schema_name.upper() if self.config.schema_name else None,
                        "t": self.config.table_name.upper(),
                        "n": verified,
                    },
                )
            except Exception as exc:  # noqa: BLE001 — the database names the bad row
                return f"the database could not verify its row hashes: {exc}"
            count = verified.getvalue()
            if count is None or int(count) != total:
                return (
                    f"the database verified {count} of {total} row hashes: "
                    "a row was changed underneath the table"
                )
        return ""

    # ------------------------------------------------------------------
    # Async twins
    # ------------------------------------------------------------------

    async def arecord(
        self, event_type: str, payload: Mapping[str, Any] | None = None
    ) -> AuditRecord:
        """:meth:`record` on a worker thread, for code that must not block its loop."""
        return await asyncio.to_thread(self.record, event_type, payload)

    async def acheck(
        self,
        *,
        expected_head: str | None = None,
        keys: Mapping[str, bytes | str] | None = None,
    ) -> AuditReport:
        """:meth:`check` on a worker thread."""
        return await asyncio.to_thread(lambda: self.check(expected_head=expected_head, keys=keys))

    async def averify(
        self,
        *,
        expected_head: str | None = None,
        keys: Mapping[str, bytes | str] | None = None,
    ) -> bool:
        """:meth:`verify` on a worker thread."""
        return (await self.acheck(expected_head=expected_head, keys=keys)).ok

    def __repr__(self) -> str:
        where = f"dsn={self.config.dsn!r}" if self.config.dsn else f"host={self.config.host!r}"
        return (
            f"OracleAuditTrail({where}, table={self._full_table_name!r}, "
            f"trail_id={self.trail_id!r}, kind={self.table_kind!r})"
        )


def _record_from_row(row: tuple[Any, ...]) -> AuditRecord:
    seq, ts, event_type, payload, prev_hash, entry_hash, key_id, signature = row
    if hasattr(payload, "read"):
        payload = payload.read()
    body = json.loads(payload) if isinstance(payload, str | bytes) else payload
    if not isinstance(body, dict):
        raise TypeError("payload is not a JSON object")
    return AuditRecord(
        seq=int(seq),
        ts=str(ts),
        event_type=str(event_type),
        payload=body,
        prev_hash=str(prev_hash),
        hash=str(entry_hash),
        key_id=key_id,
        signature=signature,
    )


__all__ = ["MIN_RETENTION_DAYS", "OracleAuditTrail", "TableKind"]
