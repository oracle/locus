# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``router.explain()`` — the decision record.

Two properties matter and neither is "it returns something". First, the
explanation must agree with what the compiler actually does: an explanation
that drifts from the real selection is worse than none, because it will be
believed. Second, it must cost nothing — no model call, no execution — or
developers will not reach for it.

Everything here is deterministic: no model is contacted at any point.
"""

from __future__ import annotations

import pytest

from tulip.router import (
    CapabilityIndex,
    CognitiveCompiler,
    GoalFrame,
    PolicyGate,
    ProtocolRegistry,
    TaskType,
    builtin_protocols,
)
from tulip.router.explain import RankedProtocol, RejectedProtocol, RoutingExplanation
from tulip.router.goal_frame import Complexity, Risk
from tulip.tools.registry import create_registry


def _compiler(**kwargs) -> CognitiveCompiler:
    registry = ProtocolRegistry()
    registry.register_many(builtin_protocols())
    return CognitiveCompiler(
        protocols=registry,
        capabilities=CapabilityIndex(create_registry()),
        policy=kwargs.pop("policy", PolicyGate()),
        model="openai:gpt-4o-mini",
        **kwargs,
    )


def _frame(
    goal: TaskType = TaskType.ANSWER,
    complexity: Complexity = Complexity.LOW,
    risk: Risk = Risk.LOW,
) -> GoalFrame:
    return GoalFrame(
        primary_goal=goal,
        domain="ops",
        complexity=complexity,
        risk=risk,
        success_criteria=["done"],
    )


# ------------------------------------------------------- agrees with compile --


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("goal", "complexity", "risk"),
    [
        (TaskType.ANSWER, Complexity.LOW, Risk.LOW),
        (TaskType.DIAGNOSE, Complexity.HIGH, Risk.MEDIUM),
        (TaskType.COMPARE, Complexity.HIGH, Risk.LOW),
        (TaskType.GENERATE_CODE, Complexity.MEDIUM, Risk.LOW),
        (TaskType.REMEDIATE, Complexity.MEDIUM, Risk.HIGH),
        (TaskType.COORDINATE, Complexity.MEDIUM, Risk.LOW),
        (TaskType.PLAN, Complexity.MEDIUM, Risk.MEDIUM),
    ],
)
async def test_explain_names_the_protocol_compile_actually_builds(
    goal: TaskType, complexity: Complexity, risk: Risk
) -> None:
    """The whole value proposition: the explanation must not drift from reality."""
    compiler = _compiler()
    frame = _frame(goal, complexity, risk)

    explanation = compiler.explain(frame)
    runnable = await compiler.compile(frame)

    built = getattr(runnable, "protocol_id", None) or getattr(
        getattr(runnable, "inner", None), "protocol_id", None
    )
    assert explanation.selected_protocol == built


# ------------------------------------------------------------- the evidence --


def test_rejected_names_the_gate_that_stopped_each_protocol() -> None:
    compiler = _compiler()
    # HIGH risk remediation: one survivor, everything else gated.
    explanation = compiler.explain(_frame(TaskType.REMEDIATE, Complexity.MEDIUM, Risk.HIGH))

    by_id = {r.protocol_id: r for r in explanation.rejected}
    assert by_id["debate"].gate == "goal"
    assert "does not handle" in by_id["debate"].reason
    # plan_execute_validate handles REMEDIATE but caps risk at medium
    assert by_id["plan_execute_validate"].gate == "risk"
    assert "caps risk" in by_id["plan_execute_validate"].reason


def test_every_protocol_is_either_a_candidate_or_rejected() -> None:
    """No protocol may silently vanish — that would hide a routing bug."""
    compiler = _compiler()
    explanation = compiler.explain(_frame(TaskType.DIAGNOSE, Complexity.HIGH, Risk.MEDIUM))

    accounted = {c.protocol_id for c in explanation.candidates} | {
        r.protocol_id for r in explanation.rejected
    }
    assert accounted == {p.id for p in compiler.protocols.all()}


def test_candidates_are_ordered_best_first_and_mark_the_winner() -> None:
    compiler = _compiler()
    explanation = compiler.explain(_frame(TaskType.RESEARCH, Complexity.HIGH, Risk.LOW))

    ranks = [c.rank for c in explanation.candidates]
    assert ranks == sorted(ranks), "candidates must be ordered by rank, best first"
    assert explanation.candidates[0].selected is True
    assert explanation.candidates[0].protocol_id == explanation.selected_protocol
    assert sum(1 for c in explanation.candidates if c.selected) == 1


def test_a2a_is_rejected_on_capabilities_when_no_endpoint_is_configured() -> None:
    """The synthetic peer capability is the opt-in; without it, say so."""
    compiler = _compiler()
    explanation = compiler.explain(_frame(TaskType.COORDINATE, Complexity.MEDIUM, Risk.LOW))

    a2a = next(r for r in explanation.rejected if r.protocol_id == "a2a_delegate")
    assert a2a.gate == "capabilities"
    assert "not available" in a2a.reason


def test_configuring_an_endpoint_moves_a2a_into_the_candidates() -> None:
    compiler = _compiler(a2a_endpoint="http://peer.invalid/a2a")
    explanation = compiler.explain(_frame(TaskType.COORDINATE, Complexity.MEDIUM, Risk.LOW))

    assert "a2a_delegate" in {c.protocol_id for c in explanation.candidates}


# ---------------------------------------------------------------- the policy --


def test_high_risk_reports_that_a_human_must_approve() -> None:
    compiler = _compiler()
    explanation = compiler.explain(_frame(TaskType.REMEDIATE, Complexity.MEDIUM, Risk.HIGH))

    assert explanation.requires_approval is True
    assert explanation.allowed is True  # allowed, but held
    assert explanation.policy is not None
    assert explanation.policy.reason


def test_low_risk_needs_no_approval() -> None:
    compiler = _compiler()
    explanation = compiler.explain(_frame(TaskType.ANSWER, Complexity.LOW, Risk.LOW))

    assert explanation.requires_approval is False
    assert explanation.allowed is True


def test_no_matching_protocol_yields_an_empty_selection_not_an_exception() -> None:
    """``explain`` is diagnostic — it must describe a dead end, not raise into one."""
    registry = ProtocolRegistry()
    registry.register_many([p for p in builtin_protocols() if p.id == "debate"])
    compiler = CognitiveCompiler(
        protocols=registry,
        capabilities=CapabilityIndex(create_registry()),
        policy=PolicyGate(),
        model="openai:gpt-4o-mini",
    )
    explanation = compiler.explain(_frame(TaskType.REMEDIATE, Complexity.LOW, Risk.LOW))

    assert explanation.selected_protocol is None
    assert explanation.candidates == ()
    assert explanation.rejected  # and it says why
    assert explanation.policy is None


# ---------------------------------------------------------------- rendering --


def test_str_shows_frame_selection_rejections_and_policy() -> None:
    compiler = _compiler()
    rendered = str(
        compiler.explain(
            _frame(TaskType.REMEDIATE, Complexity.MEDIUM, Risk.HIGH),
            goal="Delete every record older than seven years",
        )
    )

    assert "Delete every record older than seven years" in rendered
    assert "GoalFrame" in rendered
    assert "remediate" in rendered
    assert "approval_gated_execution" in rendered
    assert "Rejected" in rendered
    assert "human approval required" in rendered


def test_no_chain_of_thought_is_exposed() -> None:
    """Structured evidence only — the rationale field stays empty without a picker."""
    compiler = _compiler()
    explanation = compiler.explain(_frame())

    assert explanation.picker_rationale is None


def test_types_are_frozen_so_a_record_cannot_be_edited_after_the_fact() -> None:
    compiler = _compiler()
    explanation = compiler.explain(_frame())

    for obj in (explanation, explanation.candidates[0]):
        with pytest.raises(Exception):  # noqa: B017,PT011 — FrozenInstanceError
            obj.protocol_id = "tampered"  # type: ignore[misc]


def test_dataclasses_render_standalone() -> None:
    rejected = RejectedProtocol("debate", "goal", "does not handle 'remediate'")
    ranked = RankedProtocol("debate", (0, 1, 2, 2), 0, False, "high", 2, selected=True)

    assert "debate" in str(rejected)
    assert "does not handle" in str(rejected)
    assert "→" in str(ranked)
    assert "not canonical" in str(ranked)


# ------------------------------------------------------------------- costs ----


def test_explain_contacts_no_model() -> None:
    """A model string that would explode if instantiated proves nothing is called."""
    registry = ProtocolRegistry()
    registry.register_many(builtin_protocols())
    compiler = CognitiveCompiler(
        protocols=registry,
        capabilities=CapabilityIndex(create_registry()),
        policy=PolicyGate(),
        model="nonexistent-provider:does-not-exist",
    )

    explanation = compiler.explain(_frame(TaskType.DIAGNOSE, Complexity.HIGH, Risk.MEDIUM))

    assert isinstance(explanation, RoutingExplanation)
    assert explanation.selected_protocol == "specialist_fanout"


# ------------------------------------------------------- through the Router --
#
# The compiler tests above cover selection. These cover the Router front door,
# which is where a caller actually arrives. A ScriptedModel stands in for the
# extractor so a GoalFrame is produced without contacting anything.


def _router(frame_goal: str = "diagnose", *, repeat: bool = False, **compiler_kwargs) -> tuple:
    """A Router whose extractor is scripted to return one fixed frame."""
    import json

    from tulip import Agent
    from tulip.router import Router
    from tulip.testing import ScriptedModel, text

    payload = json.dumps(
        {
            "primary_goal": frame_goal,
            "secondary_goals": [],
            "domain": "ops",
            "complexity": "high",
            "risk": "medium",
            "requires_tools": True,
            "requires_memory": False,
            "requires_code_generation": False,
            "requires_multi_agent": True,
            "approval_required": False,
            "success_criteria": ["root cause identified"],
            "required_capabilities": [],
        }
    )
    model = ScriptedModel([text(payload)], repeat_last=repeat)
    extractor = Agent(model=model, system_prompt="extract", output_schema=GoalFrame)
    return Router(extractor=extractor, compiler=_compiler(**compiler_kwargs)), model


@pytest.mark.asyncio
async def test_router_explain_extracts_then_explains() -> None:
    router, model = _router()

    explanation = await router.explain("why did checkout latency double?")

    assert isinstance(explanation, RoutingExplanation)
    assert explanation.goal == "why did checkout latency double?"
    assert explanation.goal_frame.primary_goal is TaskType.DIAGNOSE
    assert explanation.selected_protocol == "specialist_fanout"
    # exactly one extraction call, and nothing else
    assert model.call_count == 1


@pytest.mark.asyncio
async def test_router_explain_with_a_supplied_frame_spends_no_call_at_all() -> None:
    router, model = _router()

    explanation = await router.explain("ignored", frame=_frame(TaskType.ANSWER))

    assert explanation.selected_protocol == "direct_response"
    assert model.call_count == 0


@pytest.mark.asyncio
async def test_router_compile_builds_without_executing() -> None:
    router, model = _router()

    runnable = await router.compile("why did checkout latency double?")

    protocol_id = getattr(runnable, "protocol_id", None) or getattr(
        getattr(runnable, "inner", None), "protocol_id", None
    )
    assert protocol_id == "specialist_fanout"
    # the extractor ran; the compiled topology did not
    assert model.call_count == 1
    assert hasattr(runnable, "execute")


@pytest.mark.asyncio
async def test_router_compile_accepts_a_supplied_frame() -> None:
    router, model = _router()

    runnable = await router.compile("ignored", frame=_frame(TaskType.ANSWER))

    protocol_id = getattr(runnable, "protocol_id", None) or getattr(
        getattr(runnable, "inner", None), "protocol_id", None
    )
    assert protocol_id == "direct_response"
    assert model.call_count == 0


@pytest.mark.asyncio
async def test_router_explain_and_compile_agree_through_the_front_door() -> None:
    """Same property as the compiler-level test, at the surface a caller uses."""
    # explain() and compile() each extract, so the extractor answers twice.
    router, _ = _router(repeat=True)
    goal = "why did checkout latency double?"

    explanation = await router.explain(goal)
    runnable = await router.compile(goal)

    built = getattr(runnable, "protocol_id", None) or getattr(
        getattr(runnable, "inner", None), "protocol_id", None
    )
    assert explanation.selected_protocol == built


# --------------------------------------------------- method, with a picker --
#
# ``explain`` never calls the picker (that would cost a call and could answer
# differently next run), but it must still report which arm of the ladder in
# ``_pick_protocol`` would run, or the ``chosen by`` line would be a lie.


def _picker() -> object:
    """A picker that would explode if called — explain must not call it."""

    class ExplodingPicker:
        async def pick(self, frame, candidates):  # noqa: ANN001, ANN202
            raise AssertionError("explain() must not consult the picker")

    return ExplodingPicker()


def test_method_is_llm_picked_when_a_picker_faces_several_candidates() -> None:
    compiler = _compiler(protocol_picker=_picker())
    # RESEARCH at HIGH complexity leaves more than one survivor
    explanation = compiler.explain(_frame(TaskType.RESEARCH, Complexity.HIGH, Risk.LOW))

    assert len(explanation.candidates) > 1
    assert explanation.method == "llm_picked"
    # and the ranking shown is the fallback the picker would choose among
    assert explanation.selected_protocol == explanation.candidates[0].protocol_id


def test_method_is_single_candidate_when_only_one_survives_the_gates() -> None:
    compiler = _compiler(protocol_picker=_picker())
    # HIGH-risk REMEDIATE gates everything except the approval shape
    explanation = compiler.explain(_frame(TaskType.REMEDIATE, Complexity.MEDIUM, Risk.HIGH))

    assert len(explanation.candidates) == 1
    assert explanation.method == "single_candidate"


def test_method_is_rule_based_without_a_picker() -> None:
    compiler = _compiler()
    explanation = compiler.explain(_frame(TaskType.RESEARCH, Complexity.HIGH, Risk.LOW))

    assert explanation.method == "rule_based"


def test_available_capabilities_reports_the_synthetic_peer_only_when_configured() -> None:
    assert "a2a_peer" not in _compiler().available_capabilities()
    assert "a2a_peer" in _compiler(a2a_endpoint="http://peer.invalid").available_capabilities()
