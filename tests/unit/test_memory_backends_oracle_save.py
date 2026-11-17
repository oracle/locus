# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``OracleBackend`` when two writers start on one table or one key at once.

``save`` is a ``MERGE``; for a key no writer has seen, both take the
``INSERT`` branch and the loser meets the primary key (``ORA-00001``) --
the shared ``<thread_id>:latest`` row, with two adapters saving to one
thread. The row exists once the winner commits, so the statement is
repeated and updates it. The first-time DDL has the same shape one level
up: a ``CREATE`` another session won reads as created, a lock collision
with that session is retried, and within one process the coroutines of a
backend take turns through a lock. Against the fake connection of the
index tests.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.unit.test_checkpoint_index_backends import _oracle, _OracleConn


class _RacingConn(_OracleConn):
    """The first ``failing_merges`` MERGE statements raise ``error``."""

    def __init__(self, *, failing_merges: int, error: str = "ORA-00001") -> None:
        super().__init__()
        self.failing_merges = failing_merges
        self.error = error

    async def run(self, sql: str, binds: dict[str, Any]) -> tuple[Any, int]:
        if sql.upper().startswith("MERGE") and self.failing_merges:
            self.failing_merges -= 1
            self.statements.append(sql)
            raise RuntimeError(
                f"{self.error}: unique constraint (LOCUS.SYS_C007) violated on table "
                "LOCUS.LOCUS_CHECKPOINTS columns (THREAD_ID)"
            )
        return await super().run(sql, binds)


class _ColdStartConn(_RacingConn):
    """The table is missing at the probe; ``errors`` is what each CREATE raises, in order.

    ``None`` lets a statement succeed. ``ddl`` records every attempt.
    """

    def __init__(self, errors: list[str | None]) -> None:
        super().__init__(failing_merges=0)
        self.errors = list(errors)
        self.ddl: list[str] = []

    async def run(self, sql: str, binds: dict[str, Any]) -> tuple[Any, int]:
        head = sql.upper()
        if "USER_TABLES" in head:
            return (0,), 0
        if head.startswith("CREATE"):
            self.ddl.append(head.split()[1])
            error = self.errors.pop(0) if self.errors else None
            if error:
                raise RuntimeError(f"{error}: while another session holds the object")
            return None, 0
        return await super().run(sql, binds)


def _merges(conn: _OracleConn) -> int:
    return sum(s.upper().startswith("MERGE") for s in conn.statements)


async def test_a_lost_first_write_repeats_the_merge_on_the_same_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _RacingConn(failing_merges=1)
    backend = _oracle(monkeypatch, conn)

    checkpoint_id = await backend.save("t:latest", {"a": 1})

    assert checkpoint_id
    assert conn.rows["t:latest"] == {"a": 1}
    assert _merges(conn) == 2
    assert backend._pool is not None, "no pool rebuild: the connection was fine"


async def test_a_violation_that_persists_surfaces_after_the_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _RacingConn(failing_merges=10)
    backend = _oracle(monkeypatch, conn)

    with pytest.raises(RuntimeError, match="ORA-00001"):
        await backend.save("t:latest", {"a": 1})

    assert _merges(conn) == 3


async def test_any_other_database_error_surfaces_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _RacingConn(failing_merges=10, error="ORA-12899")
    backend = _oracle(monkeypatch, conn)

    with pytest.raises(RuntimeError, match="ORA-12899"):
        await backend.save("t:latest", {"a": 1})

    assert _merges(conn) == 1


async def test_losing_the_table_creation_to_another_pool_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _ColdStartConn(errors=["ORA-00955", "ORA-00955"])
    backend = _oracle(monkeypatch, conn)

    await backend.save("t:latest", {"a": 1})

    assert conn.ddl == ["TABLE", "INDEX"], "both statements were attempted and both had lost"
    assert conn.rows["t:latest"] == {"a": 1}
    assert backend._initialized is True


async def test_a_lock_collision_on_the_index_waits_and_tries_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits: list[float] = []

    async def no_sleep(delay: float) -> None:
        waits.append(delay)

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    conn = _ColdStartConn(errors=[None, "ORA-00054", "ORA-04020", None])
    backend = _oracle(monkeypatch, conn)

    await backend.save("t:latest", {"a": 1})

    assert conn.ddl == ["TABLE", "INDEX", "INDEX", "INDEX"]
    assert waits == [0.2, 0.4], "a short, growing wait between attempts"
    assert backend._initialized is True


async def test_a_collision_that_never_clears_surfaces(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    conn = _ColdStartConn(errors=[None, *["ORA-00054"] * 10])
    backend = _oracle(monkeypatch, conn)

    with pytest.raises(RuntimeError, match="ORA-00054"):
        await backend.save("t:latest", {"a": 1})

    assert conn.ddl.count("INDEX") == 5
    assert backend._initialized is False


async def test_concurrent_first_saves_run_the_ddl_once(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _ColdStartConn(errors=[])
    backend = _oracle(monkeypatch, conn)

    await asyncio.gather(*(backend.save(f"t:cp-{i}", {"i": i}) for i in range(20)))

    assert conn.ddl == ["TABLE", "INDEX"], "one provisioning pass for twenty concurrent saves"
    assert len(conn.rows) == 20
