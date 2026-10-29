# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Two routing rules and the intent guide.

A frame whose risk is above every protocol that handles its intent compiles
onto ``risk_fallback`` (``approval_gated_execution``) and is held for
approval instead of raising. A canonical protocol outranks a closer cost
match when its cost is within one step of the frame's complexity. The
``primary_goal`` schema carries one line per ``TaskType``. Plus the
``router.runnable.failed`` event, which nothing exercised.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from locus.agent.result import AgentResult, ExecutionMetrics
from locus.core.state import AgentState
from locus.router import (
    CapabilityIndex,
    CognitiveCompiler,
    Complexity,
    GoalFrame,
    NoMatchingProtocolError,
    PolicyDeniedError,
    PolicyGate,
    Protocol,
    ProtocolRegistry,
    Risk,
    Router,
    TaskType,
    builtin_protocols,
)
from locus.router.compiler import _ApprovalRunnable
from locus.router.explain import RoutingExplanation
from locus.router.goal_frame import TASK_TYPE_GUIDE
from locus.tools.decorator import tool
from locus.tools.registry import create_registry


@tool
def _echo(text: str) -> str:
    """Echo the input verbatim."""
    return text


def _frame(
    *,
    primary_goal: TaskType = TaskType.ANSWER,
    complexity: Complexity = Complexity.LOW,
    risk: Risk = Risk.LOW,
    required_capabilities: list[str] | None = None,
) -> GoalFrame:
    return GoalFrame(
        primary_goal=primary_goal,
        domain="research",
        complexity=complexity,
        risk=risk,
        required_capabilities=required_capabilities or [],
    )


def _registry(*protocols: Protocol) -> ProtocolRegistry:
    reg = ProtocolRegistry()
    reg.register_many(list(protocols) or builtin_protocols())
    return reg


def _by_id(protocol_id: str) -> Protocol:
    return next(p for p in builtin_protocols() if p.id == protocol_id)


def _compiler(**overrides: Any) -> CognitiveCompiler:
    idx = CapabilityIndex(create_registry(_echo))
    idx.annotate("c1", tool_name="_echo", description="echo", domain="d")
    kwargs: dict[str, Any] = {
        "protocols": _registry(),
        "capabilities": idx,
        "policy": PolicyGate(),
        "model": "openai:gpt-4o-mini",
    }
    kwargs.update(overrides)
    return CognitiveCompiler(**kwargs)


# ---------------------------------------------------------------------------
# Ranking: canonical fit first, when the cost is within one step
# ---------------------------------------------------------------------------


class TestRankingRule:
    def test_canonical_beats_a_closer_cost_match_when_adjacent(self):
        # MEDIUM research: handoff_chain (medium cost, not canonical) used to win
        # on distance alone; specialist_fanout is canonical and one step away.
        chosen = _registry().select(
            _frame(primary_goal=TaskType.RESEARCH, complexity=Complexity.MEDIUM)
        )
        assert chosen.id == "specialist_fanout"

    def test_canonical_does_not_win_when_two_steps_away(self):
        # LOW-complexity research stays cheap: the canonical fan-out is two steps up.
        assert _registry().select(_frame(primary_goal=TaskType.RESEARCH)).id == "direct_response"

    def test_sequencing_stays_sequential_at_high_complexity(self):
        chosen = _registry().select(
            _frame(primary_goal=TaskType.COORDINATE, complexity=Complexity.HIGH)
        )
        assert chosen.id == "handoff_chain"

    def test_the_cheap_canonical_answer_is_unchanged(self):
        assert _registry().select(_frame()).id == "direct_response"


# ---------------------------------------------------------------------------
# Risk fallback: too risky for every handler -> approval-gated, held
# ---------------------------------------------------------------------------


class TestRiskFallback:
    def test_high_risk_plan_is_held_instead_of_raising(self):
        compiler = _compiler()
        frame = _frame(primary_goal=TaskType.PLAN, risk=Risk.HIGH)
        explanation = compiler.explain(frame)
        assert explanation.selected_protocol == "approval_gated_execution"
        assert explanation.method == "risk_fallback"
        assert explanation.requires_approval
        assert [c.protocol_id for c in explanation.candidates] == ["approval_gated_execution"]
        runnable = asyncio.run(compiler.compile(frame))
        assert isinstance(runnable, _ApprovalRunnable)

    def test_the_hold_is_forced_below_the_gate_threshold(self):
        # direct_response caps at LOW. The gate alone lets MEDIUM through
        # unheld; the fallback path always holds and says why.
        explanation = _compiler().explain(_frame(risk=Risk.MEDIUM))
        assert explanation.selected_protocol == "approval_gated_execution"
        assert explanation.requires_approval
        assert explanation.policy is not None
        assert "held for approval" in explanation.policy.reason

    def test_fallback_is_not_used_when_a_handler_fits(self):
        explanation = _compiler().explain(_frame(primary_goal=TaskType.PLAN, risk=Risk.MEDIUM))
        assert explanation.selected_protocol == "plan_execute_validate"
        assert explanation.method == "rule_based"

    def test_unhandled_goal_still_raises(self):
        compiler = _compiler(
            protocols=_registry(_by_id("direct_response"), _by_id("approval_gated_execution"))
        )
        frame = _frame(primary_goal=TaskType.GENERATE_CODE, risk=Risk.HIGH)
        assert compiler.explain(frame).selected_protocol is None
        with pytest.raises(NoMatchingProtocolError):
            asyncio.run(compiler.compile(frame))

    def test_fallback_can_be_disabled(self):
        compiler = _compiler(risk_fallback=None)
        frame = _frame(primary_goal=TaskType.PLAN, risk=Risk.HIGH)
        assert compiler.explain(frame).selected_protocol is None
        with pytest.raises(NoMatchingProtocolError):
            asyncio.run(compiler.compile(frame))

    def test_fallback_needs_a_registered_protocol_that_takes_the_risk(self):
        compiler = _compiler(
            protocols=_registry(_by_id("direct_response"), _by_id("plan_execute_validate"))
        )
        with pytest.raises(NoMatchingProtocolError):
            asyncio.run(compiler.compile(_frame(primary_goal=TaskType.PLAN, risk=Risk.HIGH)))

    def test_gate_ceiling_still_denies_above_max_risk(self):
        compiler = _compiler(policy=PolicyGate(max_risk=Risk.MEDIUM))
        with pytest.raises(PolicyDeniedError):
            asyncio.run(compiler.compile(_frame(primary_goal=TaskType.PLAN, risk=Risk.HIGH)))

    def test_every_goal_complexity_risk_cell_resolves(self):
        compiler = _compiler()
        for goal in TaskType:
            for complexity in Complexity:
                for risk in Risk:
                    frame = _frame(primary_goal=goal, complexity=complexity, risk=risk)
                    explanation = compiler.explain(frame)
                    assert explanation.selected_protocol is not None, (goal, complexity, risk)


# ---------------------------------------------------------------------------
# Intent guide reaches the schema the extractor fills
# ---------------------------------------------------------------------------


class TestIntentGuide:
    def test_every_task_type_has_a_line(self):
        assert set(TASK_TYPE_GUIDE) == set(TaskType)

    def test_guide_is_in_the_primary_goal_description(self):
        description = GoalFrame.model_json_schema()["properties"]["primary_goal"]["description"]
        for task_type, text in TASK_TYPE_GUIDE.items():
            assert f"{task_type.value}: {text}" in description


# ---------------------------------------------------------------------------
# Rendering lines and the failure event that had no test
# ---------------------------------------------------------------------------


def test_rendering_lists_capabilities_and_rationale():
    explanation = RoutingExplanation(
        goal="g",
        goal_frame=_frame(required_capabilities=["c1"]),
        selected_protocol="direct_response",
        method="llm_picked",
        candidates=(),
        rejected=(),
        available_capabilities=("c1",),
        policy=None,
        picker_rationale="cheapest fit",
    )
    text = str(explanation)
    assert "capabilities ['c1']" in text
    assert "chosen by        llm_picked" in text
    assert "rationale        cheapest fit" in text


def test_dispatch_reports_a_runnable_failure_on_the_bus(monkeypatch):
    failed = AsyncMock()
    monkeypatch.setattr("locus.router.runtime.emit_runnable_failed", failed)
    extractor = MagicMock()
    extractor.invoke = MagicMock(
        return_value=AgentResult.from_state(
            state=AgentState(),
            stop_reason="complete",
            metrics=ExecutionMetrics(),
            message="",
            parsed=_frame(),
        )
    )
    router = Router(extractor=extractor, compiler=_compiler())
    boom = MagicMock()
    boom.protocol_id = "direct_response"
    boom.execute = AsyncMock(side_effect=RuntimeError("boom"))
    router.compiler.compile = AsyncMock(return_value=boom)  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(router.dispatch("What is RAG?", run_id="r1"))
    failed.assert_awaited_once_with("r1", "direct_response", "RuntimeError: boom")


def test_a_capability_listed_twice_binds_once():
    compiler = _compiler()
    runnable = asyncio.run(compiler.compile(_frame(required_capabilities=["c1", "c1"])))
    assert len(runnable.agent.config.tools) == 1
