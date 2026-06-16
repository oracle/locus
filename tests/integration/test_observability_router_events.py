# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""End-to-end test that the router emits the canonical event sequence
on the :class:`EventBus` for one cognitive dispatch.

No real provider is needed — we hand the compiler a pre-built
GoalFrame and observe the events the compile path emits before
``runnable.execute`` (which would need an LLM).
"""

from __future__ import annotations

import asyncio

import pytest

from locus.observability import StreamEvent, get_event_bus, reset_event_bus
from locus.router import (
    CapabilityIndex,
    CognitiveCompiler,
    Complexity,
    GoalFrame,
    PolicyGate,
    ProtocolRegistry,
    Risk,
    TaskType,
    builtin_protocols,
)
from locus.router.protocol import NoMatchingProtocolError
from locus.tools.decorator import tool
from locus.tools.registry import create_registry


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@tool
def _echo(text: str) -> str:
    """Echo the input."""
    return text


@pytest.fixture(autouse=True)
def _reset_bus():
    reset_event_bus()
    yield
    reset_event_bus()


def _build_compiler() -> CognitiveCompiler:
    caps = CapabilityIndex(create_registry(_echo))
    caps.annotate("c1", tool_name="_echo", description="echo", domain="d")
    protocols = ProtocolRegistry()
    protocols.register_many(builtin_protocols())
    return CognitiveCompiler(
        protocols=protocols,
        capabilities=caps,
        policy=PolicyGate(),
        model="openai:gpt-4o-mini",  # opaque — never invoked in compile()
    )


async def _collect_events(run_id: str, expected_count: int) -> list[StreamEvent]:
    """Subscribe and collect ``expected_count`` events, then return."""
    bus = get_event_bus()
    received: list[StreamEvent] = []

    async def consumer():
        async for ev in bus.subscribe(run_id):
            received.append(ev)
            if len(received) >= expected_count:
                break

    task = asyncio.create_task(consumer())
    # Give the consumer a moment to register before the publishers fire.
    await asyncio.sleep(0.02)
    return task, received


class TestCompileEventSequence:
    """The compile path emits a deterministic sequence of events when
    a frame matches a registered protocol cleanly."""

    async def test_canonical_compile_emits_three_events(self):
        compiler = _build_compiler()
        run_id = "compile-1"

        # Subscribe FIRST, then run the compile in the same loop. The
        # compiler emits 3 events: protocol.selected, policy.verdict,
        # runnable.compiled.
        task, received = await _collect_events(run_id, expected_count=3)
        await compiler.compile(
            GoalFrame(
                primary_goal=TaskType.ANSWER,
                domain="d",
                complexity=Complexity.LOW,
                risk=Risk.LOW,
            ),
            run_id=run_id,
        )
        await asyncio.wait_for(task, timeout=2.0)

        types = [ev.event_type for ev in received]
        assert types == [
            "router.protocol.selected",
            "router.policy.verdict",
            "router.runnable.compiled",
        ], f"unexpected event sequence: {types}"

        # Check the protocol selection payload carries the canonical-flag
        # that the rank-key uses to disambiguate.
        selected = next(e for e in received if e.event_type == "router.protocol.selected")
        assert selected.data["protocol_id"] == "direct_response"
        assert selected.data["is_canonical"] is True

        # Policy verdict is allow/no-approval for a LOW-risk frame.
        verdict = next(e for e in received if e.event_type == "router.policy.verdict")
        assert verdict.data["allow"] is True
        assert verdict.data["require_approval"] is False

        # Runnable compiled identifies the adapter type — drives the
        # workbench's "this protocol compiled to {AgentRunnable}" badge.
        compiled = next(e for e in received if e.event_type == "router.runnable.compiled")
        assert compiled.data["protocol_id"] == "direct_response"
        assert compiled.data["runnable_type"] == "AgentRunnable"

    async def test_unmatched_frame_emits_no_match_then_raises(self):
        compiler = _build_compiler()
        run_id = "compile-2"
        task, received = await _collect_events(run_id, expected_count=1)

        # HIGH risk + ANSWER has no canonical protocol — registry
        # raises NoMatchingProtocolError after emitting the no-match
        # event for tail-able diagnostics.
        with pytest.raises(NoMatchingProtocolError):
            await compiler.compile(
                GoalFrame(
                    primary_goal=TaskType.ANSWER,
                    domain="d",
                    complexity=Complexity.LOW,
                    risk=Risk.HIGH,
                ),
                run_id=run_id,
            )
        await asyncio.wait_for(task, timeout=2.0)
        assert [e.event_type for e in received] == ["router.protocol.no_match"]

    async def test_no_run_id_means_no_events(self):
        """Compile without a ``run_id`` is silent — backward-compat with
        the pre-telemetry router API."""
        compiler = _build_compiler()
        bus = get_event_bus()

        await compiler.compile(
            GoalFrame(
                primary_goal=TaskType.ANSWER,
                domain="d",
                complexity=Complexity.LOW,
                risk=Risk.LOW,
            ),
        )
        # No run_id passed → no history accumulated for any run.
        assert not bus._history, (
            f"expected empty history when run_id omitted; got {bus._history.keys()}"
        )


class TestEventDataShape:
    """The wire format must match what SSE consumers (workbench,
    third-party monitors) expect to parse."""

    async def test_protocol_selected_carries_metadata(self):
        compiler = _build_compiler()
        run_id = "shape-1"
        task, received = await _collect_events(run_id, expected_count=3)
        await compiler.compile(
            GoalFrame(
                primary_goal=TaskType.PLAN,
                domain="engineering",
                complexity=Complexity.MEDIUM,
                risk=Risk.MEDIUM,
            ),
            run_id=run_id,
        )
        await asyncio.wait_for(task, timeout=2.0)

        selected = received[0]
        for required_key in (
            "protocol_id",
            "protocol_description",
            "primary_goal",
            "is_canonical",
            "cost",
            "latency",
            "risk_max",
            "method",
            "rationale",
        ):
            assert required_key in selected.data, (
                f"protocol.selected event missing field {required_key!r}: {selected.data}"
            )
        assert selected.data["protocol_id"] == "plan_execute_validate"
        assert selected.data["is_canonical"] is True
        assert selected.data["primary_goal"] == "plan"
        # Default compiler has no picker — selection must be rule-based.
        assert selected.data["method"] == "rule_based"
        assert selected.data["rationale"] is None

    async def test_to_dict_is_json_safe(self):
        compiler = _build_compiler()
        run_id = "shape-2"
        task, received = await _collect_events(run_id, expected_count=3)
        await compiler.compile(
            GoalFrame(
                primary_goal=TaskType.ANSWER,
                domain="d",
                complexity=Complexity.LOW,
                risk=Risk.LOW,
            ),
            run_id=run_id,
        )
        await asyncio.wait_for(task, timeout=2.0)

        import json

        for ev in received:
            payload = ev.to_dict()
            # Must round-trip through json without raising. SSE wire
            # format depends on this.
            json.dumps(payload)
            assert payload["run_id"] == run_id
            assert payload["event_type"].startswith("router.")
