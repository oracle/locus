# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Verify newly-instrumented modules publish their canonical event
sequences on the bus when run under an active ``run_context``.

The audit added emission points to:

* ``Orchestrator`` — routing/decision/specialists_invoked/summary
* ``Specialist`` — specialist.started/completed
* ``Handoff`` — handoff.initiated/completed
* ``SequentialPipeline`` / ``ParallelPipeline`` / ``LoopAgent`` —
  stage/fanout/iteration/terminated

Each test below drives the relevant primitive with a lightweight stub
agent (no real LLM call), reads events out of the bus's per-run
history buffer, and asserts the expected event_type sequence.

The history buffer (``EventBus._history``) is the cleanest read path
for tests because publishes there are synchronous under the bus's
lock — no need to race a subscriber against the publisher.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from typing import Any

import pytest

from locus.agent.composition import LoopAgent, ParallelPipeline, SequentialPipeline
from locus.agent.result import AgentResult
from locus.core.state import AgentState
from locus.multiagent.handoff import Handoff, HandoffAgent, HandoffReason, HandoffResult
from locus.multiagent.orchestrator import Orchestrator
from locus.multiagent.specialist import Specialist, SpecialistResult
from locus.observability import (
    StreamEvent,
    get_event_bus,
    reset_event_bus,
    run_context,
)


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_bus():
    reset_event_bus()
    yield
    reset_event_bus()


def _history(rid: str) -> list[StreamEvent]:
    """Pull a run's event history out of the bus.

    Reads the private ``_history`` deque so we don't have to spin up
    a subscriber and race the publisher.
    """
    bus = get_event_bus()
    return list(bus._history.get(rid, ()))  # type: ignore[attr-defined]


def _kinds(rid: str) -> list[str]:
    return [e.event_type for e in _history(rid)]


class _StubAgent:
    """Minimal agent stand-in that returns a fixed message via ``run_sync``.

    SequentialPipeline / ParallelPipeline call ``agent.run_sync(...)``
    and read ``result.message``. We honour that contract without going
    near the model layer.
    """

    def __init__(self, response: str = "ok") -> None:
        self._response = response

    def run_sync(self, prompt: str) -> Any:  # noqa: ARG002 — match interface
        state = AgentState(agent_id="stub")
        return AgentResult(
            message=self._response,
            state=state,
            stop_reason="complete",
            started_at=datetime.now(UTC),
        )


# ---------------------------------------------------------------------------
# Composition pipelines
# ---------------------------------------------------------------------------


class TestPipelines:
    async def test_sequential_emits_stage_started_completed_per_stage(self):
        pipeline = SequentialPipeline(agents=[_StubAgent("a"), _StubAgent("b")])
        async with run_context("seq-1") as rid:
            result = await pipeline.run("go")

        assert result.success
        # Two stages → started+completed × 2 in publish order.
        assert _kinds(rid) == [
            "composition.stage.started",
            "composition.stage.completed",
            "composition.stage.started",
            "composition.stage.completed",
        ], _kinds(rid)

    async def test_parallel_emits_fanout_started_completed(self):
        pipeline = ParallelPipeline(agents=[_StubAgent("a"), _StubAgent("b")])
        async with run_context("par-1") as rid:
            await pipeline.run("go")

        history = _history(rid)
        kinds = [e.event_type for e in history]
        assert kinds[0] == "composition.fanout.started"
        assert kinds[-1] == "composition.fanout.completed"
        assert history[0].data["agent_count"] == 2

    async def test_loop_emits_iteration_and_terminated(self):
        loop_agent = LoopAgent(
            agent=_StubAgent("done"),
            condition=lambda r: True,  # stop after first iteration
            max_loops=3,
        )
        async with run_context("loop-1") as rid:
            await loop_agent.run("go")

        kinds = _kinds(rid)
        assert "composition.loop.iteration.started" in kinds
        assert "composition.loop.iteration.completed" in kinds
        assert kinds[-1] == "composition.loop.terminated"
        assert _history(rid)[-1].data["terminated_by"] in {"condition", "max_loops"}


# ---------------------------------------------------------------------------
# Multi-agent — Orchestrator + Specialist
# ---------------------------------------------------------------------------


class _StubModel:
    """Bypass the model layer for orchestrator routing — we don't care
    about the prompt; we just need ``complete`` to return JSON-shaped
    text the orchestrator can parse into a routing decision."""

    def __init__(self, payload: str) -> None:
        self._payload = payload

    async def complete(self, *_args, **_kwargs):  # noqa: ANN001, ANN202
        from locus.core.messages import Message  # noqa: PLC0415

        class _Resp:
            def __init__(self, content: str) -> None:
                self.message = Message.assistant(content)

        return _Resp(self._payload)


class _StubSpecialist(Specialist):
    """Specialist with a hard-coded ``execute`` so tests don't need a model."""

    async def execute(  # type: ignore[override]
        self,
        task: str,  # noqa: ARG002
        context: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> SpecialistResult:
        return SpecialistResult(
            specialist_id=self.id,
            specialist_type=self.specialist_type,
            output="stub-output",
            success=True,
            confidence=0.9,
        )


class TestOrchestrator:
    async def test_orchestrator_emits_full_sequence(self):
        spec = _StubSpecialist(
            name="Stub",
            specialist_type="stub",
            description="d",
            system_prompt="p",
        )
        # Routing payload selects exactly the stub specialist.
        payload = (
            f'```json\n{{"specialists": ["{spec.id}"], "reasoning": "go", "subtasks": {{}}}}\n```'
        )
        orch = Orchestrator(model=_StubModel(payload))
        orch.register_specialist(spec)

        async with run_context("orch-1") as rid:
            result = await orch.execute("incident triage")

        assert result.success
        kinds = _kinds(rid)
        for required in (
            "multiagent.orchestrator.routing",
            "multiagent.orchestrator.decision",
            "multiagent.orchestrator.specialists_invoked",
            "multiagent.orchestrator.summary",
        ):
            assert required in kinds, (required, kinds)
        # Routing precedes specialists_invoked; summary terminates.
        assert kinds.index("multiagent.orchestrator.routing") < kinds.index(
            "multiagent.orchestrator.specialists_invoked"
        )
        assert kinds[-1] == "multiagent.orchestrator.summary"


# ---------------------------------------------------------------------------
# Handoff
# ---------------------------------------------------------------------------


class _StubHandoffAgent(HandoffAgent):
    """HandoffAgent override that bypasses the model layer."""

    async def receive_handoff(self, context):  # type: ignore[override]
        return HandoffResult(
            handoff_id=context.handoff_id,
            success=True,
            source_agent_id=context.source_agent_id,
            target_agent_id=self.id,
            output="handoff-output",
            final_confidence=0.9,
        )


class TestHandoff:
    async def test_handoff_emits_initiated_and_completed(self):
        source = _StubHandoffAgent(name="source")
        target = _StubHandoffAgent(name="target")
        handoff = Handoff()
        handoff.register_agents([source, target])

        async with run_context("ho-1") as rid:
            await handoff.execute_handoff(
                source_agent=source,
                target_agent_id=target.id,
                task="check the logs",
                reason=HandoffReason.DELEGATION,
            )

        kinds = _kinds(rid)
        assert "multiagent.handoff.initiated" in kinds
        assert "multiagent.handoff.completed" in kinds
        assert kinds.index("multiagent.handoff.initiated") < kinds.index(
            "multiagent.handoff.completed"
        )


# ---------------------------------------------------------------------------
# SDK-without-SSE invariant — drive the same primitives outside a
# ``run_context`` and confirm the bus stays unborn.
# ---------------------------------------------------------------------------


class TestNoContextNoBus:
    async def test_pipelines_outside_run_context_do_not_create_bus(self):
        bus_mod = sys.modules["locus.observability.event_bus"]
        # Force a clean state.
        reset_event_bus()
        bus_mod._event_bus = None  # type: ignore[attr-defined]

        await SequentialPipeline(agents=[_StubAgent()]).run("go")
        await ParallelPipeline(agents=[_StubAgent()]).run("go")
        await LoopAgent(agent=_StubAgent(), condition=lambda r: True, max_loops=1).run("go")

        assert bus_mod._event_bus is None, (  # type: ignore[attr-defined]
            "Pipelines outside run_context must not instantiate the bus"
        )
