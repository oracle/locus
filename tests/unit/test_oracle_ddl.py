# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The shared first-time DDL runner every Oracle backend provisions through."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from locus import _oracle_ddl as ddl


class _Cursor:
    """Raises ``errors`` in order on ``execute``; ``None`` lets a statement through."""

    def __init__(self, errors: list[str | None]) -> None:
        self.errors = list(errors)
        self.executed: list[str] = []

    def _next(self, statement: str) -> None:
        self.executed.append(statement)
        error = self.errors.pop(0) if self.errors else None
        if error:
            raise RuntimeError(f"{error}: raised by the fake")

    async def execute(self, statement: str) -> None:
        self._next(statement)


class _SyncCursor(_Cursor):
    def execute(self, statement: str) -> None:  # type: ignore[override]
        self._next(statement)


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    waits: list[float] = []

    async def fake_async(delay: float) -> None:
        waits.append(delay)

    def fake_sync(delay: float) -> None:
        waits.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_async)
    monkeypatch.setattr(time, "sleep", fake_sync)
    return waits


class TestPredicates:
    def test_already_exists_is_ora_00955(self) -> None:
        assert ddl.already_exists(RuntimeError("ORA-00955: name is already used"))
        assert not ddl.already_exists(RuntimeError("ORA-00054: resource busy"))

    def test_contended_is_a_lock_collision(self) -> None:
        assert ddl.ddl_contended(RuntimeError("ORA-00054: resource busy"))
        assert ddl.ddl_contended(RuntimeError("ORA-04020: deadlock detected"))
        assert not ddl.ddl_contended(RuntimeError("ORA-00955: name is already used"))
        assert not ddl.ddl_contended(RuntimeError("ORA-00001: unique constraint"))


class TestRunDdl:
    async def test_a_clean_statement_runs_once(self, no_sleep: list[float]) -> None:
        cursor = _Cursor([None])
        await ddl.run_ddl(cursor, "CREATE TABLE t (x NUMBER)")
        assert cursor.executed == ["CREATE TABLE t (x NUMBER)"]
        assert no_sleep == []

    async def test_a_create_another_session_won_is_done(self, no_sleep: list[float]) -> None:
        cursor = _Cursor(["ORA-00955"])
        await ddl.run_ddl(cursor, "CREATE TABLE t (x NUMBER)")
        assert len(cursor.executed) == 1
        assert no_sleep == []

    async def test_a_collision_waits_and_tries_again(self, no_sleep: list[float]) -> None:
        cursor = _Cursor(["ORA-00054", "ORA-04020", None])
        await ddl.run_ddl(cursor, "CREATE INDEX i ON t (x)")
        assert len(cursor.executed) == 3
        assert no_sleep == [0.2, 0.4], "a short, growing wait between attempts"

    async def test_a_collision_that_never_clears_surfaces(self, no_sleep: list[float]) -> None:
        cursor = _Cursor(["ORA-00054"] * 10)
        with pytest.raises(RuntimeError, match="ORA-00054"):
            await ddl.run_ddl(cursor, "CREATE INDEX i ON t (x)")
        assert len(cursor.executed) == ddl.DDL_ATTEMPTS
        assert len(no_sleep) == ddl.DDL_ATTEMPTS - 1

    async def test_anything_else_surfaces_at_once(self, no_sleep: list[float]) -> None:
        cursor = _Cursor(["ORA-01031"])
        with pytest.raises(RuntimeError, match="ORA-01031"):
            await ddl.run_ddl(cursor, "CREATE TABLE t (x NUMBER)")
        assert len(cursor.executed) == 1
        assert no_sleep == []

    def test_the_sync_runner_has_the_same_rules(self, no_sleep: list[float]) -> None:
        won = _SyncCursor(["ORA-00955"])
        ddl.run_ddl_sync(won, "CREATE TABLE t (x NUMBER)")
        assert len(won.executed) == 1

        contended = _SyncCursor(["ORA-00054", None])
        ddl.run_ddl_sync(contended, "CREATE INDEX i ON t (x)")
        assert len(contended.executed) == 2
        assert no_sleep == [0.2]

        other = _SyncCursor(["ORA-01031"])
        with pytest.raises(RuntimeError, match="ORA-01031"):
            ddl.run_ddl_sync(other, "CREATE TABLE t (x NUMBER)")


class TestProvisioningLock:
    async def test_one_lock_per_owner_and_loop(self) -> None:
        class _Owner:
            pass

        owner, other = _Owner(), _Owner()
        lock = ddl.provisioning_lock(owner)
        assert isinstance(lock, asyncio.Lock)
        assert ddl.provisioning_lock(owner) is lock, "the same lock while the loop lives"
        assert ddl.provisioning_lock(other) is not lock

    def test_a_new_loop_gets_a_new_lock(self) -> None:
        class _Owner:
            pass

        owner = _Owner()
        first = asyncio.run(_lock_of(owner))
        second = asyncio.run(_lock_of(owner))
        assert first is not second

    async def test_the_lock_serialises_the_coroutines_of_one_owner(self) -> None:
        class _Owner:
            pass

        owner = _Owner()
        running = 0
        overlaps = 0

        async def provision() -> None:
            nonlocal running, overlaps
            async with ddl.provisioning_lock(owner):
                running += 1
                overlaps += running > 1
                await asyncio.sleep(0)
                running -= 1

        await asyncio.gather(*(provision() for _ in range(20)))
        assert overlaps == 0


async def _lock_of(owner: Any) -> asyncio.Lock:
    return ddl.provisioning_lock(owner)
