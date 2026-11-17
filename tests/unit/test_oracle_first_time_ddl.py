# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Two Oracle backends that share the first-time DDL helper, under a burst of first writes.

The runner itself is covered in ``test_oracle_ddl.py``; ``OracleBackend``
in ``test_memory_backends_oracle_save.py``. Here an async store and a
sync one show the shape every Oracle class now has: one provisioning pass
per instance however many first operations arrive together, and a second
instance whose ``CREATE`` loses to the first carrying on.
"""

from __future__ import annotations

import asyncio
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any

import pytest

from locus.memory.store_backends.oracle import OracleStore
from tests.unit.test_audit_oracle_trail import _FakeConn, _FakeCursor, _FakeDB, _FakePool, _trail
from tests.unit.test_oracle_store import _install_oracledb_stub, _StubCursor


async def test_twenty_concurrent_first_puts_provision_the_store_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cursor = _StubCursor(fetchone=(0,))  # the table is missing at every probe
    _install_oracledb_stub(monkeypatch, cursor)
    store = OracleStore(dsn="x", user="u", password="p", table_name="t")  # noqa: S106

    await asyncio.gather(*(store.put(("ns",), f"k{i}", {"v": i}) for i in range(20)))

    creates = [sql for sql, _ in cursor.execute_calls if sql.lstrip().upper().startswith("CREATE")]
    assert len(creates) == 2, "one CREATE TABLE and one CREATE INDEX for twenty first puts"
    assert store._initialized is True


class _MissingTableCursor(_FakeCursor):
    """The probe keeps answering "missing": both instances reach CREATE, as in a real race."""

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> None:
        super().execute(sql, params)
        if " ".join(sql.split()).upper().startswith("SELECT COUNT(*) FROM ALL_TABLES"):
            self._one = (0,)


class _MissingTableConn(_FakeConn):
    def cursor(self) -> _FakeCursor:
        return _MissingTableCursor(self.db)


class _MissingTablePool(_FakePool):
    @contextmanager
    def acquire(self) -> Any:
        yield _MissingTableConn(self.db)


def test_two_audit_trails_starting_on_one_fresh_table_both_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _FakeDB()  # raises ORA-00955 on the second CREATE, as the database would
    fake = types.ModuleType("oracledb")
    fake.create_pool = lambda **kwargs: _MissingTablePool(db)  # type: ignore[attr-defined]
    fake.makedsn = lambda host, port, service_name: f"{host}:{port}/{service_name}"  # type: ignore[attr-defined]
    fake.DB_TYPE_CLOB = "CLOB"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "oracledb", fake)
    first, second = _trail(trail_id="a"), _trail(trail_id="b")

    def write(trail: Any, i: int) -> None:
        trail.record("event", {"i": i})

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(write, trail, i) for i in range(10) for trail in (first, second)]
        for future in futures:
            future.result()

    creates = [sql for sql in db.executed if " ".join(sql.split()).upper().startswith("CREATE ")]
    assert len(creates) == 2, "one CREATE per instance; the loser's ORA-00955 read as created"
    assert len(db.rows) == 20
    assert first._table_ready is True
    assert second._table_ready is True
