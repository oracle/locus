# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``Orchestrator`` parallel specialist execution.

The audit caught the orchestrator running specialists serially (a
deliberate trade for provider stability before timeouts were
properly pinned). With ``request_timeout`` / ``max_retries`` defaults
now pinned, the
orchestrator fans out to specialists in parallel — bounded by
``max_parallel_specialists``, with per-specialist exception
isolation so one failure can't drop the whole batch.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tulip.multiagent.orchestrator import Orchestrator, RoutingDecision
from tulip.multiagent.specialist import Specialist, SpecialistResult


class _Overlap:
    """Tracks how many ``execute`` bodies run at once, across all specialists.

    Parallelism is asserted structurally instead of by wall-clock (a loaded CI
    runner makes any millisecond threshold flaky): with ``parties`` set, each
    body parks until that many are in flight together — reachable only when
    they really run concurrently — and times out with a clear message
    otherwise.
    """

    parties = 0
    arrived = 0
    in_flight = 0
    peak = 0
    broken = False
    gate: asyncio.Event | None = None

    @classmethod
    def reset(cls, parties: int = 0) -> None:
        cls.parties = parties
        cls.arrived = cls.in_flight = cls.peak = 0
        cls.broken = False
        cls.gate = asyncio.Event() if parties else None

    @classmethod
    async def enter(cls) -> None:
        cls.in_flight += 1
        cls.peak = max(cls.peak, cls.in_flight)
        if cls.gate is None:
            # No barrier: just yield so an overlapping sibling would get in.
            for _ in range(5):
                await asyncio.sleep(0)
            return
        if cls.broken:
            raise RuntimeError("barrier broken by an earlier timeout")
        cls.arrived += 1
        if cls.arrived >= cls.parties:
            cls.gate.set()
        try:
            await asyncio.wait_for(cls.gate.wait(), 10.0)
        except TimeoutError:
            cls.broken = True
            raise RuntimeError(
                f"only {cls.arrived}/{cls.parties} specialists ever ran together"
            ) from None

    @classmethod
    def leave(cls) -> None:
        cls.in_flight -= 1


@pytest.fixture(autouse=True)
def _reset_overlap() -> Any:
    _Overlap.reset()
    yield
    _Overlap.reset()


class _SleepingSpec(Specialist):
    """Specialist that sleeps for ``sleep_ms`` then returns a fixed reply."""

    sleep_ms: int = 50
    reply: str = "ok"
    fail_with: str | None = None
    empty_first: bool = False  # surface the retry path: empty then non-empty
    _calls: int = 0

    async def execute(
        self, *, task: str, context: dict[str, Any] | None = None
    ) -> SpecialistResult:
        await _Overlap.enter()
        try:
            await asyncio.sleep(self.sleep_ms / 1000.0)
        finally:
            _Overlap.leave()
        self._calls += 1
        if self.fail_with:
            msg = self.fail_with
            raise RuntimeError(msg)
        if self.empty_first and self._calls == 1:
            output = ""
        else:
            output = f"{self.id}:{self.reply}"
        return SpecialistResult(
            specialist_id=self.id,
            specialist_type=self.specialist_type,
            output=output,
            success=bool(output),
        )


def _orch(specs: list[Specialist], *, max_parallel: int = 5) -> Orchestrator:
    o = Orchestrator(name="O", specialists={s.id: s for s in specs})
    o.max_parallel_specialists = max_parallel
    return o


@pytest.mark.asyncio
async def test_specialists_run_in_parallel() -> None:
    # All three must be in flight at once for the barrier to open.
    a = _SleepingSpec(
        id="a", name="A", specialist_type="alpha", description="d", system_prompt="d", sleep_ms=50
    )
    b = _SleepingSpec(
        id="b", name="B", specialist_type="beta", description="d", system_prompt="d", sleep_ms=50
    )
    c = _SleepingSpec(
        id="c",
        name="C",
        specialist_type="gamma",
        description="d",
        system_prompt="d",
        sleep_ms=50,
    )
    o = _orch([a, b, c], max_parallel=3)
    _Overlap.reset(parties=3)

    results = await o._invoke_specialists(
        "task", RoutingDecision(decision_type="invoke", specialists=["a", "b", "c"])
    )

    assert set(results) == {"a", "b", "c"}
    assert all(r.output for r in results.values()), results
    assert _Overlap.peak == 3


@pytest.mark.asyncio
async def test_max_parallel_one_is_serial() -> None:
    # ``max_parallel_specialists=1`` reproduces the old serial behaviour
    # for users who hit provider rate limits with the parallel fan-out.
    a = _SleepingSpec(
        id="a", name="A", specialist_type="alpha", description="d", system_prompt="d", sleep_ms=40
    )
    b = _SleepingSpec(
        id="b", name="B", specialist_type="beta", description="d", system_prompt="d", sleep_ms=40
    )
    o = _orch([a, b], max_parallel=1)
    results = await o._invoke_specialists(
        "task", RoutingDecision(decision_type="invoke", specialists=["a", "b"])
    )
    assert all(r.output for r in results.values()), results
    assert _Overlap.peak == 1, f"{_Overlap.peak} specialists overlapped with max_parallel=1"


@pytest.mark.asyncio
async def test_per_specialist_exception_isolated() -> None:
    # One specialist raises; the others must still produce results.
    a = _SleepingSpec(
        id="a", name="A", specialist_type="alpha", description="d", system_prompt="d", sleep_ms=10
    )
    boom = _SleepingSpec(
        id="boom",
        name="Boom",
        specialist_type="explode",
        description="d",
        system_prompt="d",
        fail_with="downstream-out",
        sleep_ms=10,
    )
    c = _SleepingSpec(
        id="c", name="C", specialist_type="gamma", description="d", system_prompt="d", sleep_ms=10
    )
    o = _orch([a, boom, c], max_parallel=3)

    results = await o._invoke_specialists(
        "task", RoutingDecision(decision_type="invoke", specialists=["a", "boom", "c"])
    )
    assert results["a"].output == "a:ok"
    assert results["c"].output == "c:ok"
    assert results["boom"].error is not None
    assert "downstream-out" in (results["boom"].error or "")


@pytest.mark.asyncio
async def test_unknown_specialist_id_yields_typed_error() -> None:
    a = _SleepingSpec(
        id="a", name="A", specialist_type="alpha", description="d", system_prompt="d", sleep_ms=5
    )
    o = _orch([a], max_parallel=2)
    results = await o._invoke_specialists(
        "task",
        RoutingDecision(decision_type="invoke", specialists=["a", "ghost"]),
    )
    assert results["a"].output == "a:ok"
    assert results["ghost"].error == "Specialist not found: ghost"


@pytest.mark.asyncio
async def test_empty_first_response_is_retried() -> None:
    # The retry-on-empty branch is preserved from the pre-parallel
    # implementation — it covers a provider's occasional empty-completion
    # blip. Each specialist gets exactly one retry.
    a = _SleepingSpec(
        id="a",
        name="A",
        specialist_type="alpha",
        description="d",
        system_prompt="d",
        sleep_ms=5,
        empty_first=True,
    )
    o = _orch([a], max_parallel=1)
    results = await o._invoke_specialists(
        "task", RoutingDecision(decision_type="invoke", specialists=["a"])
    )
    assert results["a"].output == "a:ok"
    # ``_calls`` private counter proves the retry actually fired.
    assert a._calls == 2  # noqa: SLF001 — test inspects the recorded call count
