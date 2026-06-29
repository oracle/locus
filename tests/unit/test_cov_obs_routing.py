# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Extra coverage for router_events.py and agent_bridge.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tulip.observability.event_bus import reset_event_bus


pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _reset_bus():
    reset_event_bus()
    yield
    reset_event_bus()


# ---------------------------------------------------------------------------
# Helpers — mock GoalFrame, Protocol, PolicyVerdict
# ---------------------------------------------------------------------------


def _mock_frame(
    primary_goal: str = "execute",
    risk: str = "medium",
    secondary_goals: list[str] | None = None,
) -> MagicMock:
    frame = MagicMock()
    frame.primary_goal = MagicMock()
    frame.primary_goal.value = primary_goal
    frame.secondary_goals = []
    frame.domain = "test-domain"
    frame.complexity = MagicMock()
    frame.complexity.value = "moderate"
    frame.risk = MagicMock()
    frame.risk.value = risk
    frame.approval_required = False
    frame.required_capabilities = ["tool_use"]
    frame.success_criteria = ["task_complete"]
    return frame


def _mock_protocol() -> MagicMock:
    proto = MagicMock()
    proto.id = "direct_response"
    proto.description = "Direct response protocol"
    proto.primary_for = set()
    proto.cost = 0.01
    proto.latency = 0.5
    proto.risk_max = MagicMock()
    proto.risk_max.value = "high"
    return proto


def _mock_verdict(allow: bool = True) -> MagicMock:
    verdict = MagicMock()
    verdict.allow = allow
    verdict.require_approval = False
    verdict.reason = "Policy passed"
    return verdict


# ---------------------------------------------------------------------------
# router_events — uncovered emit helpers
# ---------------------------------------------------------------------------


class TestRouterEventsEmitHelpers:
    async def test_emit_frame_failed(self):
        """emit_frame_failed publishes on the bus (line 65)."""
        from tulip.observability.router_events import emit_frame_failed

        await emit_frame_failed("run-1", "schema mismatch")

        from tulip.observability.event_bus import get_event_bus

        bus = get_event_bus()
        hist = list(bus._history.get("run-1", []))
        assert any(e.event_type == "router.frame.failed" for e in hist)
        assert hist[-1].data["error"] == "schema mismatch"

    async def test_emit_protocol_selected(self):
        """emit_protocol_selected publishes on the bus (line 96)."""
        from tulip.observability.router_events import emit_protocol_selected

        frame = _mock_frame()
        proto = _mock_protocol()
        await emit_protocol_selected("run-2", frame, proto, method="rule_based")

        from tulip.observability.event_bus import get_event_bus

        bus = get_event_bus()
        hist = list(bus._history.get("run-2", []))
        assert any(e.event_type == "router.protocol.selected" for e in hist)

    async def test_emit_protocol_selected_with_rationale(self):
        """emit_protocol_selected with llm_picked method and rationale."""
        from tulip.observability.router_events import emit_protocol_selected

        frame = _mock_frame()
        proto = _mock_protocol()
        await emit_protocol_selected(
            "run-2b",
            frame,
            proto,
            method="llm_picked",
            rationale="Best fit for task",
        )

        from tulip.observability.event_bus import get_event_bus

        bus = get_event_bus()
        hist = list(bus._history.get("run-2b", []))
        ev = next(e for e in hist if e.event_type == "router.protocol.selected")
        assert ev.data["method"] == "llm_picked"
        assert ev.data["rationale"] == "Best fit for task"

    async def test_emit_protocol_no_match(self):
        """emit_protocol_no_match publishes on the bus (line 117)."""
        from tulip.observability.router_events import emit_protocol_no_match

        frame = _mock_frame()
        await emit_protocol_no_match("run-3", frame, "no matching protocol")

        from tulip.observability.event_bus import get_event_bus

        bus = get_event_bus()
        hist = list(bus._history.get("run-3", []))
        assert any(e.event_type == "router.protocol.no_match" for e in hist)

    async def test_emit_policy_verdict(self):
        """emit_policy_verdict publishes on the bus (line 158)."""
        from tulip.observability.router_events import emit_policy_verdict

        frame = _mock_frame()
        proto = _mock_protocol()
        verdict = _mock_verdict(allow=True)
        await emit_policy_verdict("run-4", frame, proto, verdict)

        from tulip.observability.event_bus import get_event_bus

        bus = get_event_bus()
        hist = list(bus._history.get("run-4", []))
        ev = next(e for e in hist if e.event_type == "router.policy.verdict")
        assert ev.data["allow"] is True

    async def test_emit_runnable_compiled(self):
        """emit_runnable_compiled publishes on the bus (line 181)."""
        from tulip.observability.router_events import emit_runnable_compiled

        await emit_runnable_compiled("run-5", "direct_response", "AgentRunnable", 3)

        from tulip.observability.event_bus import get_event_bus

        bus = get_event_bus()
        hist = list(bus._history.get("run-5", []))
        ev = next(e for e in hist if e.event_type == "router.runnable.compiled")
        assert ev.data["capability_count"] == 3

    async def test_emit_runnable_failed(self):
        """emit_runnable_failed publishes on the bus (line 221)."""
        from tulip.observability.router_events import emit_runnable_failed

        await emit_runnable_failed("run-6", "direct_response", "timeout")

        from tulip.observability.event_bus import get_event_bus

        bus = get_event_bus()
        hist = list(bus._history.get("run-6", []))
        ev = next(e for e in hist if e.event_type == "router.runnable.failed")
        assert ev.data["error"] == "timeout"

    async def test_emit_frame_extracted(self):
        """emit_frame_extracted is mostly covered; verify it still works with mock frame."""
        from tulip.observability.router_events import emit_frame_extracted

        frame = _mock_frame()
        await emit_frame_extracted("run-7", frame)

        from tulip.observability.event_bus import get_event_bus

        bus = get_event_bus()
        hist = list(bus._history.get("run-7", []))
        assert any(e.event_type == "router.frame.extracted" for e in hist)

    async def test_emit_picker_fallback(self):
        """emit_picker_fallback is a bonus: verify it emits the right type."""
        from tulip.observability.router_events import emit_picker_fallback

        frame = _mock_frame()
        await emit_picker_fallback("run-8", frame, "unknown protocol id")

        from tulip.observability.event_bus import get_event_bus

        bus = get_event_bus()
        hist = list(bus._history.get("run-8", []))
        assert any(e.event_type == "router.protocol.picker_fallback" for e in hist)


# ---------------------------------------------------------------------------
# agent_bridge — uncovered event branches (lines 133, 140, 148, 149)
# ---------------------------------------------------------------------------


class TestAgentBridgeMissingBranches:
    async def _emit_event(self, event_type, **data):
        """Helper: call emit under a run_context to see published events."""
        from tulip.core.events import (
            ModelChunkEvent,
            ModelCompleteEvent,
        )
        from tulip.observability.agent_bridge import bridge_tulip_event
        from tulip.observability.context import run_context
        from tulip.observability.event_bus import get_event_bus

        bus = get_event_bus()
        collected: list[str] = []

        rid = f"bridge-{event_type}"

        async def subscriber():
            async for ev in bus.subscribe(rid):
                collected.append(ev.event_type)
                if len(collected) >= data.get("_expect", 1):
                    return

        task = None

        async with run_context(rid):
            task = __import__("asyncio").create_task(subscriber())
            await __import__("asyncio").sleep(0.01)

            # Build and bridge the event
            if event_type == "model_chunk":
                ev = ModelChunkEvent(content="hello", done=False, tool_calls=[])
            elif event_type == "model_complete_no_usage":
                ev = ModelCompleteEvent(
                    content="final", tool_calls=[], usage={}, stop_reason="end_turn"
                )
            elif event_type == "model_complete_with_usage":
                ev = ModelCompleteEvent(
                    content="final",
                    tool_calls=[],
                    usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    stop_reason="end_turn",
                )
            else:
                raise ValueError(f"Unknown: {event_type}")

            await bridge_tulip_event(ev)

        await __import__("asyncio").wait_for(task, timeout=2.0)
        return collected

    async def test_bridge_model_chunk_event(self):
        """ModelChunkEvent triggers emit(EV_AGENT_MODEL_CHUNK) — line 133."""
        collected = await self._emit_event("model_chunk")
        assert any("agent.model.chunk" in t for t in collected)

    async def test_bridge_model_complete_no_usage(self):
        """ModelCompleteEvent with empty usage emits only the completed event
        (line 140); the usage branch is NOT entered (line 148 → False)."""
        collected = await self._emit_event("model_complete_no_usage")
        assert any("agent.model.completed" in t for t in collected)
        assert not any("tokens.used" in t for t in collected)

    async def test_bridge_model_complete_with_usage(self):
        """ModelCompleteEvent with usage emits both completed and tokens.used
        (lines 140, 148, 149)."""
        collected = await self._emit_event("model_complete_with_usage", _expect=2)
        assert any("agent.model.completed" in t for t in collected)
        assert any("tokens.used" in t for t in collected)
