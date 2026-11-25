# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The agent loop reads its clock, ids and randomness from ``locus.core.sources``."""

from __future__ import annotations

import ast
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import locus
from locus.agent import model_retry
from locus.agent.result import AgentResult
from locus.core import sources
from locus.core.events import ModelChunkEvent
from locus.core.interrupt import InterruptValue
from locus.core.messages import ToolCall
from locus.core.state import AgentState
from locus.core.termination import TimeLimit


class FixedSources:
    """A clock that moves only when told to, counted ids and a fixed draw."""

    def __init__(self) -> None:
        self.at = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
        self.ids = 0

    def advance(self, seconds: float) -> None:
        self.at += timedelta(seconds=seconds)

    def now(self) -> datetime:
        return self.at

    def time(self) -> float:
        return self.at.timestamp()

    def monotonic(self) -> float:
        return self.at.timestamp()

    def uuid4(self) -> uuid.UUID:
        self.ids += 1
        return uuid.UUID(int=self.ids)

    def uniform(self, a: float, b: float) -> float:
        return (a + b) / 2


@pytest.fixture
def fixed() -> Iterator[FixedSources]:
    installed = FixedSources()
    previous = sources.set_sources(installed)
    try:
        yield installed
    finally:
        sources.set_sources(previous)


def test_system_sources_are_the_default() -> None:
    assert isinstance(sources.get_sources(), sources.SystemSources)
    assert isinstance(sources.get_sources(), sources.RuntimeSources)
    before = datetime.now(UTC)
    now = sources.now()
    assert now.tzinfo is UTC
    assert before <= now <= datetime.now(UTC)
    assert abs(sources.time() - now.timestamp()) < 5
    first = sources.monotonic()
    assert sources.monotonic() >= first
    assert sources.uuid4() != sources.uuid4()
    assert sources.uuid4().version == 4
    assert 1.0 <= sources.uniform(1.0, 2.0) <= 2.0


def test_set_sources_returns_the_previous_and_reset_restores_the_system() -> None:
    system = sources.get_sources()
    installed = FixedSources()
    try:
        assert sources.set_sources(installed) is system
        assert sources.get_sources() is installed
        assert sources.set_sources(FixedSources()) is installed
    finally:
        sources.reset_sources()
    assert sources.get_sources() is system


def test_state_events_and_ids_come_from_the_sources(fixed: FixedSources) -> None:
    state = AgentState()
    assert state.started_at == fixed.at
    assert state.updated_at == fixed.at
    assert state.run_id == uuid.UUID(int=1).hex
    assert ToolCall(name="t").id == f"call_{uuid.UUID(int=2).hex[:12]}"
    assert InterruptValue(payload="approve?").interrupt_id == f"int_{uuid.UUID(int=3).hex[:8]}"
    assert ModelChunkEvent().timestamp == fixed.at
    assert AgentResult(message="done", state=state, stop_reason="complete").started_at == fixed.at

    fixed.advance(30)
    assert state.with_iteration(1).updated_at == fixed.at
    assert state.started_at == fixed.at - timedelta(seconds=30)


def test_time_limit_runs_on_the_sources_clock(fixed: FixedSources) -> None:
    limit = TimeLimit(seconds=60)
    state = AgentState()
    assert limit.check(state) == (False, None)
    fixed.advance(59)
    assert limit.check(state) == (False, None)
    fixed.advance(1)
    assert limit.check(state) == (True, "time_budget")


def test_model_retry_jitter_and_budget_use_the_sources(fixed: FixedSources) -> None:
    assert model_retry._uniform(0.0, 8.0) == 4.0
    started = model_retry._monotonic()
    fixed.advance(12.5)
    assert model_retry._monotonic() - started == 12.5


def test_same_sources_give_the_same_run_twice() -> None:
    """What a replaying host relies on: equal sources, equal state."""

    def build() -> tuple[str, datetime, str]:
        previous = sources.set_sources(FixedSources())
        try:
            state = AgentState()
            return state.run_id, state.started_at, ToolCall(name="t").id
        finally:
            sources.set_sources(previous)

    assert build() == build()


# The modules the agent loop runs. A direct clock, id or random read in one of
# them is invisible to an installed source, so a replaying host would diverge.
_LOOP_PACKAGES = ("core", "agent")
_LOOP_MODULES = ("tools/executor.py",)
_DIRECT_READS = {
    ("time", "time"),
    ("time", "monotonic"),
    ("time", "perf_counter"),
    ("datetime", "now"),
    ("datetime", "utcnow"),
    ("uuid", "uuid4"),
    ("random", "random"),
    ("random", "uniform"),
    ("random", "randint"),
    ("random", "choice"),
}


def _loop_files() -> list[Path]:
    root = Path(locus.__file__).parent
    files = [p for package in _LOOP_PACKAGES for p in sorted((root / package).rglob("*.py"))]
    files += [root / module for module in _LOOP_MODULES]
    return [p for p in files if p.name != "sources.py"]


def _direct_reads(path: Path) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "uuid4":
            found.append(f"{path.name}:{node.lineno} uuid4()")
        elif (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and (func.value.id, func.attr) in _DIRECT_READS
        ):
            found.append(f"{path.name}:{node.lineno} {func.value.id}.{func.attr}()")
    return found


def test_loop_modules_read_only_through_the_sources() -> None:
    files = _loop_files()
    assert len(files) > 20
    direct = [read for path in files for read in _direct_reads(path)]
    assert direct == [], (
        "read the clock, ids and randomness through locus.core.sources in the "
        f"agent loop's modules: {direct}"
    )
