# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``OracleAuditTrail`` against a stub ``oracledb`` that keeps the rows.

Same shape as the checkpointer stubs (a fake ``oracledb`` module dropped into
``sys.modules``), but the cursor stores what is inserted and serves it back,
so chain continuity across appends, reload, tamper detection on a mutated row
and the blockchain-table verification path are exercised end to end without a
database.
"""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from typing import Any

import pytest

from locus.control import (
    Action,
    AdmissionError,
    ControlPolicy,
    OracleAuditTrail,
    admit_sync,
    check_jsonl,
)
from locus.control.audit import _GENESIS


class _FakeError(Exception):
    pass


class _FakeVar:
    def __init__(self) -> None:
        self._value: Any = None

    def setvalue(self, _pos: int, value: Any) -> None:
        self._value = value

    def getvalue(self) -> Any:
        return self._value


class _FakeLob:
    def __init__(self) -> None:
        self.text = ""

    def write(self, text: str) -> None:
        self.text += text

    def read(self) -> str:
        return self.text


class _FakeDB:
    """The table, as the fake cursor sees it."""

    def __init__(self) -> None:
        self.table_created = False
        self.ddl = ""
        self.rows: list[dict[str, Any]] = []
        self.pending: list[dict[str, Any]] = []
        self.executed: list[str] = []
        self.missing_hashes = 0
        self.verified_rows: int | None = None
        self.verify_error: str | None = None
        self.acquire_errors: list[str] = []
        self.pools_built = 0


class _FakeCursor:
    def __init__(self, db: _FakeDB) -> None:
        self.db = db
        self._one: Any = None
        self._all: list[tuple[Any, ...]] = []

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> None:
        self.db.executed.append(sql)
        p = params or {}
        s = " ".join(sql.split()).upper()
        if s.startswith("SELECT COUNT(*) FROM ALL_TABLES"):
            self._one = (1 if self.db.table_created else 0,)
        elif s.startswith("CREATE "):
            if self.db.table_created:
                raise _FakeError("ORA-00955: name is already used by an existing object")
            self.db.table_created = True
            self.db.ddl = sql
        elif s.startswith("INSERT INTO"):
            if any(r["trail_id"] == p["trail_id"] and r["seq"] == p["seq"] for r in self.db.rows):
                raise _FakeError("ORA-00001: unique constraint violated")
            row = dict(p)
            if isinstance(row["payload"], _FakeLob):
                row["payload_lob"] = True
                row["payload"] = row["payload"].read()
            self.db.pending.append(row)
        elif "ORABCTAB_HASH$ IS NULL" in s:
            self._one = (self.db.missing_hashes,)
        elif s.startswith("SELECT COUNT(*) FROM"):
            self._one = (len(self.db.rows),)
        elif s.startswith("SELECT SEQ, TS, EVENT_TYPE"):
            rows = sorted(
                (r for r in self.db.rows if r["trail_id"] == p["trail_id"]), key=lambda r: r["seq"]
            )
            self._all = [
                (
                    r["seq"],
                    r["ts"],
                    r["event_type"],
                    r["payload"],
                    r["prev_hash"],
                    r["entry_hash"],
                    r["key_id"],
                    r["signature"],
                )
                for r in rows
            ]
        elif s.startswith("BEGIN DBMS_BLOCKCHAIN_TABLE.VERIFY_ROWS"):
            if self.db.verify_error:
                raise _FakeError(self.db.verify_error)
            n = len(self.db.rows) if self.db.verified_rows is None else self.db.verified_rows
            p["n"].setvalue(0, n)
        else:  # pragma: no cover - a statement the trail does not send
            raise AssertionError(f"unexpected SQL: {sql}")

    def fetchone(self) -> Any:
        return self._one

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._all

    def var(self, _typ: Any) -> _FakeVar:
        return _FakeVar()

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


class _FakeConn:
    def __init__(self, db: _FakeDB) -> None:
        self.db = db
        self.commits = 0

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self.db)

    def commit(self) -> None:
        self.db.rows.extend(self.db.pending)
        self.db.pending.clear()
        self.commits += 1

    def createlob(self, _typ: Any) -> _FakeLob:
        return _FakeLob()


class _FakePool:
    def __init__(self, db: _FakeDB) -> None:
        self.db = db
        self.closed = False

    @contextmanager
    def acquire(self) -> Any:
        if self.db.acquire_errors:
            raise _FakeError(self.db.acquire_errors.pop(0))
        yield _FakeConn(self.db)

    def close(self, force: bool = False) -> None:
        self.closed = True


@pytest.fixture
def db(monkeypatch: pytest.MonkeyPatch) -> _FakeDB:
    database = _FakeDB()
    fake = types.ModuleType("oracledb")

    def create_pool(**kwargs: Any) -> _FakePool:
        database.pools_built += 1
        return _FakePool(database)

    fake.create_pool = create_pool  # type: ignore[attr-defined]
    fake.makedsn = lambda host, port, service_name: f"{host}:{port}/{service_name}"  # type: ignore[attr-defined]
    fake.DB_TYPE_CLOB = "CLOB"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "oracledb", fake)
    return database


def _trail(**kwargs: Any) -> OracleAuditTrail:
    base: dict[str, Any] = {"dsn": "audit_low", "user": "AUDIT", "password": "pw"}  # noqa: S106
    base.update(kwargs)
    return OracleAuditTrail(**base)


# ---------------------------------------------------------------------------
# Construction and DDL
# ---------------------------------------------------------------------------


def test_immutable_table_ddl_carries_retention_clauses(db: _FakeDB) -> None:
    trail = _trail(retention_days=30, drop_idle_days=45)
    ddl = trail.ddl()
    assert ddl.startswith("CREATE IMMUTABLE TABLE locus_audit_trail (")
    assert "NO DROP UNTIL 45 DAYS IDLE NO DELETE UNTIL 30 DAYS AFTER INSERT" in ddl
    assert "HASHING" not in ddl
    assert "CONSTRAINT locus_audit_trail_uq UNIQUE (trail_id, seq)" in ddl


def test_blockchain_table_ddl_and_locked_clauses(db: _FakeDB) -> None:
    trail = _trail(
        table_kind="blockchain", retention_days=None, drop_idle_days=None, schema_name="AUD"
    )
    ddl = trail.ddl()
    assert ddl.startswith("CREATE BLOCKCHAIN TABLE AUD.locus_audit_trail (")
    assert ') NO DROP NO DELETE LOCKED HASHING USING "SHA2_512" VERSION "v1"' in ddl


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"table_name": "bad name"}, "Invalid table_name"),
        ({"retention_days": 5}, "at least 16"),
        ({"drop_idle_days": -1}, "0 or more"),
        ({"table_kind": "ordinary"}, "table_kind"),
        ({"trail_id": ""}, "trail_id"),
    ],
)
def test_rejects_bad_configuration(db: _FakeDB, kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _trail(**kwargs)


def test_construction_touches_no_database(db: _FakeDB) -> None:
    _trail()
    assert db.pools_built == 0
    assert db.executed == []


def test_table_created_once_on_first_use(db: _FakeDB) -> None:
    trail = _trail()
    trail.record("gate", {"n": 1})
    trail.record("gate", {"n": 2})
    assert db.table_created
    assert sum(1 for s in db.executed if s.lstrip().upper().startswith("CREATE")) == 1
    assert db.pools_built == 1


def test_missing_table_without_auto_create_is_an_error(db: _FakeDB) -> None:
    trail = _trail(auto_create_table=False)
    with pytest.raises(RuntimeError, match="does not exist"):
        trail.record("gate", {})
    assert not db.table_created
    assert trail._records == []


def test_host_triplet_builds_a_dsn(db: _FakeDB) -> None:
    trail = OracleAuditTrail(
        host="adb.example.com",
        port=1522,
        service_name="svc_high",
        user="u",
        password="p",  # noqa: S106
    )
    trail.record("gate", {})
    assert "host='adb.example.com'" in repr(trail)
    assert db.rows


# ---------------------------------------------------------------------------
# Appending and the chain
# ---------------------------------------------------------------------------


def test_rows_are_committed_and_chained(db: _FakeDB) -> None:
    trail = _trail()
    first = trail.record("gate", {"tool": "bash", "verdict": "allow"})
    second = trail.record("gate", {"tool": "write", "verdict": "deny"})
    assert [r["seq"] for r in db.rows] == [0, 1]
    assert db.rows[0]["prev_hash"] == _GENESIS
    assert db.rows[1]["prev_hash"] == first.hash
    assert db.rows[1]["entry_hash"] == second.hash
    assert db.pending == []
    assert trail.head == second.hash
    assert len(trail) == 2


def test_reopening_continues_the_chain_from_the_table(db: _FakeDB) -> None:
    first = _trail()
    first.record("gate", {"n": 1})
    head = first.head

    second = _trail()
    assert len(second) == 1
    assert second.head == head
    rec = second.record("gate", {"n": 2})
    assert rec.prev_hash == head
    assert second.verify(expected_head=rec.hash)
    assert check_jsonl(second.export_jsonl(), expected_head=rec.hash).ok


def test_trails_are_chained_per_trail_id(db: _FakeDB) -> None:
    a = _trail(trail_id="session-a")
    b = _trail(trail_id="session-b")
    a.record("gate", {"n": 1})
    b.record("gate", {"n": 1})
    a.record("gate", {"n": 2})
    assert len(a) == 2
    assert len(b) == 1
    assert a.verify()
    assert b.verify()
    assert a.records()[1].prev_hash == a.records()[0].hash


def test_concurrent_writer_is_detected_and_the_chain_continues(db: _FakeDB) -> None:
    writer_a = _trail()
    writer_b = _trail()
    assert len(writer_b) == 0  # b reads the empty table first
    writer_a.record("gate", {"by": "a"})
    assert len(writer_b) == 0  # and is stale now
    rec = writer_b.record("gate", {"by": "b"})
    # The unique (trail_id, seq) constraint rejected seq 0; b reloaded and
    # appended as seq 1 on a's head instead of forking the chain.
    assert rec.seq == 1
    assert rec.prev_hash == writer_a.head
    assert len(writer_a.records()) == 1  # a's cache is older, but what it has is intact
    assert writer_a.verify()  # verify() re-reads the table ...
    assert len(writer_a.records()) == 2  # ... and refreshes the cache


def test_failed_insert_keeps_the_record_out_of_the_chain(db: _FakeDB) -> None:
    trail = _trail()
    trail.record("gate", {"n": 1})
    db.acquire_errors = ["ORA-00942: table or view does not exist"]
    with pytest.raises(_FakeError, match="ORA-00942"):
        trail.record("gate", {"n": 2})
    assert len(trail) == 1
    assert len(db.rows) == 1


def test_dead_connection_rebuilds_the_pool_once(db: _FakeDB) -> None:
    trail = _trail()
    trail.record("gate", {"n": 1})
    db.acquire_errors = ["DPY-4011: the database or network closed the connection"]
    trail.record("gate", {"n": 2})
    assert db.pools_built == 2
    assert trail.verify()


def test_large_payload_goes_through_a_lob(db: _FakeDB) -> None:
    trail = _trail()
    trail.record("gate", {"blob": "x" * 40_000})
    assert db.rows[0].get("payload_lob") is True
    assert trail.verify()
    assert _trail().records()[0].payload["blob"] == "x" * 40_000


def test_signed_records_round_trip(db: _FakeDB) -> None:
    pytest.importorskip("cryptography")
    from locus.control import Ed25519Signer  # noqa: PLC0415

    signer = Ed25519Signer.generate(key_id="audit-2026")
    trail = _trail(signer=signer)
    trail.record("gate", {"n": 1})
    reopened = _trail()
    assert reopened.verify(keys={"audit-2026": signer.public_key_pem()})
    assert not reopened.verify(keys={"other": signer.public_key_pem()})


# ---------------------------------------------------------------------------
# Verification reads the table, not the cache
# ---------------------------------------------------------------------------


def test_mutated_row_is_located(db: _FakeDB) -> None:
    trail = _trail()
    for n in range(4):
        trail.record("gate", {"verdict": "deny", "n": n})
    db.rows[2]["payload"] = '{"verdict": "allow", "n": 2}'
    assert not trail.verify()
    report = trail.check()
    assert report.broken_at == 2
    assert "changed after it was written" in report.problem


def test_removed_row_is_located(db: _FakeDB) -> None:
    trail = _trail()
    for n in range(3):
        trail.record("gate", {"n": n})
    del db.rows[1]
    report = trail.check()
    assert (report.ok, report.broken_at) == (False, 1)
    assert "removed or moved" in report.problem


def test_truncated_table_needs_an_external_head(db: _FakeDB) -> None:
    trail = _trail()
    for n in range(3):
        trail.record("gate", {"n": n})
    head = trail.head
    del db.rows[2]
    assert trail.verify()  # a shorter chain still stands on its own
    report = trail.check(expected_head=head)
    assert not report.ok
    assert "cut off" in report.problem


def test_unreadable_payload_fails_verification(db: _FakeDB) -> None:
    trail = _trail()
    trail.record("gate", {"n": 1})
    db.rows[0]["payload"] = "not json"
    assert not trail.verify()
    assert "not audit records" in trail.check().problem


# ---------------------------------------------------------------------------
# Blockchain table: the database checks its own hashes too
# ---------------------------------------------------------------------------


def test_blockchain_check_asks_the_database(db: _FakeDB) -> None:
    trail = _trail(table_kind="blockchain")
    trail.record("gate", {"n": 1})
    trail.record("gate", {"n": 2})
    assert trail.verify()
    assert any("DBMS_BLOCKCHAIN_TABLE.VERIFY_ROWS" in s for s in db.executed)
    assert any("ORABCTAB_HASH$ IS NULL" in s for s in db.executed)


def test_immutable_check_does_not_call_blockchain_apis(db: _FakeDB) -> None:
    trail = _trail()
    trail.record("gate", {"n": 1})
    assert trail.verify()
    assert not any("DBMS_BLOCKCHAIN_TABLE" in s for s in db.executed)


def test_blockchain_row_without_database_hash_fails(db: _FakeDB) -> None:
    trail = _trail(table_kind="blockchain")
    trail.record("gate", {"n": 1})
    db.missing_hashes = 1
    report = trail.check()
    assert not report.ok
    assert "no database-side hash" in report.problem


def test_blockchain_verify_rows_error_fails(db: _FakeDB) -> None:
    trail = _trail(table_kind="blockchain")
    trail.record("gate", {"n": 1})
    db.verify_error = "ORA-05715: operation not allowed on the blockchain table"
    report = trail.check()
    assert not report.ok
    assert "could not verify" in report.problem
    assert "ORA-05715" in report.problem


def test_blockchain_partial_verification_fails(db: _FakeDB) -> None:
    trail = _trail(table_kind="blockchain")
    trail.record("gate", {"n": 1})
    trail.record("gate", {"n": 2})
    db.verified_rows = 1
    report = trail.check()
    assert not report.ok
    assert "verified 1 of 2" in report.problem


def test_blockchain_check_skips_database_when_chain_already_broken(db: _FakeDB) -> None:
    trail = _trail(table_kind="blockchain")
    trail.record("gate", {"n": 1})
    db.rows[0]["payload"] = '{"n": 9}'
    db.executed.clear()
    report = trail.check()
    assert report.broken_at == 0
    assert not any("DBMS_BLOCKCHAIN_TABLE" in s for s in db.executed)


# ---------------------------------------------------------------------------
# Works where an AuditTrail works
# ---------------------------------------------------------------------------


def test_admit_records_on_the_oracle_trail(db: _FakeDB) -> None:
    trail = _trail()
    policy = ControlPolicy()
    refund = Action(
        name="refund", asset="cust:1", blast_radius=1, kind="payment", environment="production"
    )
    with pytest.raises(AdmissionError):
        admit_sync(refund, lambda: None, policy=policy, trail=trail)
    assert db.rows[0]["event_type"] == "action-admission"
    assert '"action": "refund"' in db.rows[0]["payload"]
    assert trail.verify()


async def test_async_twins_run_the_same_work(db: _FakeDB) -> None:
    trail = _trail(table_kind="blockchain")
    rec = await trail.arecord("gate", {"n": 1})
    assert rec.seq == 0
    assert await trail.averify(expected_head=rec.hash)
    report = await trail.acheck()
    assert report.ok
    assert report.records == 1


def test_close_drops_the_pool(db: _FakeDB) -> None:
    trail = _trail()
    trail.record("gate", {})
    trail.close()
    assert trail._pool is None
    trail.record("gate", {})
    assert db.pools_built == 2
    assert "OracleAuditTrail(dsn='audit_low'" in repr(trail)
