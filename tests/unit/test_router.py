# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the cognitive router (`tulip.router`).

Covers the deterministic core: enum ordering, GoalFrame validation,
CapabilityIndex behaviour, ProtocolRegistry selection, PolicyGate
verdicts, the three v1 builders' compile paths, the adapter
normalization, and end-to-end Router.dispatch with a mock extractor +
mock primitives.

No network — every "model" is a tiny stub that returns canned text.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from tulip.agent.composition import PipelineResult
from tulip.agent.result import AgentResult, ExecutionMetrics
from tulip.core.state import AgentState
from tulip.multiagent.orchestrator import OrchestratorResult
from tulip.router import (
    HUMAN_SENTINEL,
    A2ARunnable,
    AgentRunnable,
    BuilderContext,
    CapabilityIndex,
    CognitiveCompiler,
    Complexity,
    GoalFrame,
    NoMatchingProtocolError,
    OrchestratorRunnable,
    PipelineRunnable,
    PolicyDeniedError,
    PolicyGate,
    PolicyVerdict,
    Protocol,
    ProtocolRegistry,
    Risk,
    Router,
    RunnableResult,
    SkillIndex,
    TaskType,
    builtin_protocols,
)
from tulip.router.compiler import _ApprovalRunnable, _default_deny
from tulip.router.runtime import FrameExtractionError
from tulip.skills.models import Skill
from tulip.skills.plugin import SkillsPlugin
from tulip.tools.decorator import tool
from tulip.tools.registry import create_registry


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@tool
def _echo(text: str) -> str:
    """Echo the input verbatim."""
    return text


@tool
def _shout(text: str) -> str:
    """Uppercase the input."""
    return text.upper()


def _make_agent_result(message: str, parsed: BaseModel | None = None) -> AgentResult:
    """Build a minimal AgentResult for adapter tests."""
    state = AgentState()
    return AgentResult.from_state(
        state=state,
        stop_reason="complete",
        metrics=ExecutionMetrics(),
        message=message,
        parsed=parsed,
    )


def _frame(
    *,
    primary_goal: TaskType = TaskType.ANSWER,
    domain: str = "research",
    complexity: Complexity = Complexity.LOW,
    risk: Risk = Risk.LOW,
    approval_required: bool = False,
    required_capabilities: list[str] | None = None,
) -> GoalFrame:
    return GoalFrame(
        primary_goal=primary_goal,
        domain=domain,
        complexity=complexity,
        risk=risk,
        approval_required=approval_required,
        required_capabilities=required_capabilities or [],
    )


# ---------------------------------------------------------------------------
# Enum ordering
# ---------------------------------------------------------------------------


class TestRiskOrdering:
    """Risk and Complexity must order by declaration index, not alphabetically."""

    def test_lt(self):
        assert Risk.LOW < Risk.MEDIUM < Risk.HIGH

    def test_gt_does_not_fall_back_to_string(self):
        # If StrEnum's str.__gt__ leaked through, "low" > "high" would be True.
        assert not (Risk.LOW > Risk.HIGH)
        assert Risk.HIGH > Risk.LOW

    def test_le_ge(self):
        assert Risk.LOW <= Risk.LOW
        assert Risk.HIGH >= Risk.HIGH
        assert Risk.MEDIUM <= Risk.HIGH

    def test_complexity_ordering(self):
        assert Complexity.LOW < Complexity.MEDIUM < Complexity.HIGH

    def test_compare_across_enums_is_notimplemented(self):
        # Cross-enum comparison shouldn't accidentally succeed.
        with pytest.raises(TypeError):
            _ = Risk.LOW < Complexity.LOW  # type: ignore[operator]


# ---------------------------------------------------------------------------
# GoalFrame
# ---------------------------------------------------------------------------


class TestGoalFrame:
    def test_minimal(self):
        frame = _frame()
        assert frame.primary_goal is TaskType.ANSWER
        assert frame.secondary_goals == []
        assert frame.required_capabilities == []
        assert not frame.approval_required

    def test_frozen(self):
        frame = _frame()
        with pytest.raises((TypeError, AttributeError, Exception)):
            frame.primary_goal = TaskType.PLAN  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CapabilityIndex
# ---------------------------------------------------------------------------


class TestCapabilityIndex:
    def test_annotate_unknown_tool_raises(self):
        idx = CapabilityIndex(create_registry(_echo))
        with pytest.raises(KeyError, match="unknown tool"):
            idx.annotate("bad", tool_name="nope", description="x", domain="d")

    def test_annotate_duplicate_raises(self):
        idx = CapabilityIndex(create_registry(_echo))
        idx.annotate("c1", tool_name="_echo", description="x", domain="d")
        with pytest.raises(ValueError, match="already registered"):
            idx.annotate("c1", tool_name="_echo", description="y", domain="d")

    def test_lookup_missing_raises(self):
        idx = CapabilityIndex(create_registry(_echo))
        idx.annotate("c1", tool_name="_echo", description="x", domain="d")
        with pytest.raises(KeyError, match="Unknown capability"):
            idx.lookup(["c1", "missing"])

    def test_for_domain(self):
        idx = CapabilityIndex(create_registry(_echo, _shout))
        idx.annotate("a", tool_name="_echo", description="x", domain="d1")
        idx.annotate("b", tool_name="_shout", description="y", domain="d2")
        assert {c.id for c in idx.for_domain("d1")} == {"a"}
        assert {c.id for c in idx.for_domain("d2")} == {"b"}

    def test_human_sentinel(self):
        idx = CapabilityIndex(create_registry(_echo))
        cap = idx.annotate(
            "approve",
            tool_name=HUMAN_SENTINEL,
            description="needs human",
            domain="ops",
            risk=Risk.HIGH,
        )
        assert cap.is_human
        with pytest.raises(ValueError, match="human-approval"):
            idx.resolve_tool(cap)

    def test_resolve_tool_round_trip(self):
        idx = CapabilityIndex(create_registry(_echo))
        cap = idx.annotate("c1", tool_name="_echo", description="x", domain="d")
        tool_obj = idx.resolve_tool(cap)
        assert tool_obj.name == "_echo"

    def test_contains_and_len(self):
        idx = CapabilityIndex(create_registry(_echo))
        idx.annotate("c1", tool_name="_echo", description="x", domain="d")
        assert "c1" in idx
        assert "missing" not in idx
        assert len(idx) == 1


# ---------------------------------------------------------------------------
# ProtocolRegistry
# ---------------------------------------------------------------------------


class TestProtocolRegistry:
    def test_empty_registry_raises(self):
        reg = ProtocolRegistry()
        with pytest.raises(NoMatchingProtocolError, match="No protocols registered"):
            reg.select(_frame())

    def test_register_duplicate_raises(self):
        reg = ProtocolRegistry()
        reg.register_many(builtin_protocols())
        with pytest.raises(ValueError, match="already registered"):
            reg.register(builtin_protocols()[0])

    def test_get_unknown_raises(self):
        reg = ProtocolRegistry()
        with pytest.raises(KeyError, match="Unknown protocol id"):
            reg.get("does-not-exist")

    def test_select_answer_picks_direct_response(self):
        reg = ProtocolRegistry()
        reg.register_many(builtin_protocols())
        assert reg.select(_frame()).id == "direct_response"

    def test_select_diagnose_picks_specialist_fanout(self):
        reg = ProtocolRegistry()
        reg.register_many(builtin_protocols())
        chosen = reg.select(
            _frame(
                primary_goal=TaskType.DIAGNOSE,
                complexity=Complexity.HIGH,
                risk=Risk.MEDIUM,
            ),
        )
        assert chosen.id == "specialist_fanout"

    def test_select_high_risk_no_match_raises(self):
        reg = ProtocolRegistry()
        reg.register_many(builtin_protocols())
        # No builtin protocol caps at HIGH, so a HIGH-risk frame can't match.
        with pytest.raises(NoMatchingProtocolError):
            reg.select(_frame(risk=Risk.HIGH))

    def test_select_filters_by_capability_availability(self):
        reg = ProtocolRegistry()
        # Register one protocol that requires a capability, and one that doesn't.
        reg.register(
            Protocol(
                id="needs_cap",
                description="x",
                handles=[TaskType.ANSWER],
                requires_capabilities=["needed_cap"],
                builder=lambda f, c, ctx: AgentRunnable(
                    agent=MagicMock(invoke=lambda _t: _make_agent_result("x")),
                    protocol_id="needs_cap",
                    frame=f,
                ),
            )
        )
        reg.register_many(builtin_protocols())
        # When needed_cap is missing, must fall back to direct_response.
        assert reg.select(_frame(), available_capabilities=set()).id == "direct_response"
        # When provided, the registry can pick either; just confirm no error.
        chosen = reg.select(_frame(), available_capabilities={"needed_cap"})
        assert chosen.id in {"needs_cap", "direct_response"}


# ---------------------------------------------------------------------------
# PolicyGate
# ---------------------------------------------------------------------------


class TestPolicyGate:
    def test_allow_low(self):
        gate = PolicyGate()
        proto = builtin_protocols()[0]  # direct_response, risk_max=LOW
        verdict = gate.check(_frame(), proto)
        assert verdict.allow
        assert not verdict.require_approval
        assert verdict.reason == "ok"

    def test_deny_when_protocol_caps_lower(self):
        gate = PolicyGate()
        proto = builtin_protocols()[0]  # risk_max=LOW
        verdict = gate.check(_frame(risk=Risk.MEDIUM), proto)
        assert not verdict.allow
        assert "caps risk" in verdict.reason

    def test_deny_when_above_gate_max(self):
        # Build a pretend protocol that allows HIGH so we can isolate the
        # gate-level cap from the protocol-level cap.
        loose = Protocol(
            id="loose",
            description="x",
            handles=[TaskType.ANSWER],
            risk_max=Risk.HIGH,
            builder=lambda f, c, ctx: AgentRunnable(
                agent=MagicMock(invoke=lambda _t: _make_agent_result("x")),
                protocol_id="loose",
                frame=f,
            ),
        )
        gate = PolicyGate(max_risk=Risk.MEDIUM)
        verdict = gate.check(_frame(risk=Risk.HIGH), loose)
        assert not verdict.allow
        assert "exceeds gate max_risk" in verdict.reason

    def test_require_approval_above_threshold(self):
        # plan_execute_validate accepts MEDIUM. With approval threshold at LOW,
        # a MEDIUM frame is allowed but flagged for approval.
        proto = next(p for p in builtin_protocols() if p.id == "plan_execute_validate")
        gate = PolicyGate(require_approval_above=Risk.LOW)
        verdict = gate.check(_frame(primary_goal=TaskType.PLAN, risk=Risk.MEDIUM), proto)
        assert verdict.allow
        assert verdict.require_approval

    def test_require_approval_when_frame_flag_set(self):
        gate = PolicyGate()
        proto = builtin_protocols()[0]
        verdict = gate.check(_frame(approval_required=True), proto)
        assert verdict.allow
        assert verdict.require_approval


# ---------------------------------------------------------------------------
# Adapter normalization
# ---------------------------------------------------------------------------


class TestAdapters:
    def test_agent_runnable_normalizes(self):
        agent = MagicMock()
        agent.invoke = MagicMock(return_value=_make_agent_result("hello"))
        runnable = AgentRunnable(agent=agent, protocol_id="direct_response", frame=_frame())
        result = asyncio.run(runnable.execute("ping"))
        assert isinstance(result, RunnableResult)
        assert result.text == "hello"
        assert result.protocol_id == "direct_response"
        agent.invoke.assert_called_once_with("ping")

    def test_pipeline_runnable_normalizes(self):
        async def fake_run(_task: str) -> PipelineResult:
            return PipelineResult(success=True, outputs=["a", "b"], final_output="b")

        pipeline = MagicMock()
        pipeline.run = fake_run
        runnable = PipelineRunnable(
            pipeline=pipeline, protocol_id="plan_execute_validate", frame=_frame()
        )
        result = asyncio.run(runnable.execute("ping"))
        assert result.text == "b"

    def test_orchestrator_runnable_normalizes(self):
        async def fake_execute(_task: str, **_kw: Any) -> OrchestratorResult:
            return OrchestratorResult(orchestrator_id="o1", success=True, summary="done")

        orch = MagicMock()
        orch.execute = fake_execute
        runnable = OrchestratorRunnable(
            orchestrator=orch, protocol_id="specialist_fanout", frame=_frame()
        )
        result = asyncio.run(runnable.execute("ping"))
        assert result.text == "done"

    def test_a2a_runnable_uses_send_message_task_artifact_text(self):
        client = MagicMock()
        task = MagicMock()
        artifact = MagicMock()
        artifact.parts = [MagicMock(text="older"), MagicMock(text="newest")]
        task.artifacts = [artifact]
        task.status = None

        async def send_message(_message):
            return task

        client.send_message = send_message
        runnable = A2ARunnable(client=client, protocol_id="a2a_delegate", frame=_frame())
        result = asyncio.run(runnable.execute("ping"))

        assert result.text == "newest"
        assert result.raw is task
        assert result.protocol_id == "a2a_delegate"

    def test_a2a_runnable_falls_back_to_legacy_invoke_on_missing_send_method(self):
        client = MagicMock()

        async def send_message(_message):
            raise RuntimeError("A2A error -32601: method not found")

        async def invoke(_task):
            return "legacy answer"

        client.send_message = send_message
        client.invoke = invoke
        runnable = A2ARunnable(client=client, protocol_id="a2a_delegate", frame=_frame())
        result = asyncio.run(runnable.execute("ping"))

        assert result.text == "legacy answer"
        assert result.raw == "legacy answer"

    def test_a2a_runnable_propagates_non_method_not_found_errors(self):
        client = MagicMock()

        async def send_message(_message):
            raise RuntimeError("boom")

        client.send_message = send_message
        runnable = A2ARunnable(client=client, protocol_id="a2a_delegate", frame=_frame())

        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(runnable.execute("ping"))

    def test_a2a_extract_task_text_reads_status_message_and_empty_task(self):
        status_task = MagicMock()
        status_task.artifacts = []
        status = MagicMock()
        message = MagicMock()
        message.parts = [MagicMock(text=""), MagicMock(text="status text")]
        status.message = message
        status_task.status = status

        empty_task = MagicMock()
        empty_task.artifacts = []
        empty_task.status = None

        assert A2ARunnable._extract_task_text(status_task) == "status text"
        assert A2ARunnable._extract_task_text(empty_task) == ""


# ---------------------------------------------------------------------------
# v1 builder compile paths (no network — model is opaque to the builders)
# ---------------------------------------------------------------------------


class TestBuilders:
    def _ctx(self, idx: CapabilityIndex) -> BuilderContext:
        # The builders pass `model` through to Agent/Specialist constructors;
        # those accept a string model spec without resolving until invoke time.
        return BuilderContext(model="openai:gpt-4o-mini", capabilities=idx)

    def test_direct_response_builder(self):
        idx = CapabilityIndex(create_registry(_echo))
        idx.annotate("c1", tool_name="_echo", description="echo it", domain="d")
        proto = next(p for p in builtin_protocols() if p.id == "direct_response")
        runnable = proto.builder(_frame(), idx.lookup(["c1"]), self._ctx(idx))
        assert isinstance(runnable, AgentRunnable)
        # The wrapped agent must carry the resolved tool, not a raw capability id.
        assert runnable.agent.config.tools  # non-empty after resolution

    def test_plan_execute_validate_builder(self):
        idx = CapabilityIndex(create_registry(_echo))
        idx.annotate("c1", tool_name="_echo", description="echo", domain="d")
        proto = next(p for p in builtin_protocols() if p.id == "plan_execute_validate")
        runnable = proto.builder(
            _frame(primary_goal=TaskType.PLAN), idx.lookup(["c1"]), self._ctx(idx)
        )
        assert isinstance(runnable, PipelineRunnable)
        # Sequential: planner, executor, validator.
        assert len(runnable.pipeline.agents) == 3

    def test_specialist_fanout_builder(self):
        idx = CapabilityIndex(create_registry(_echo, _shout))
        idx.annotate("a", tool_name="_echo", description="echo", domain="d")
        idx.annotate("b", tool_name="_shout", description="shout", domain="d")
        proto = next(p for p in builtin_protocols() if p.id == "specialist_fanout")
        runnable = proto.builder(
            _frame(primary_goal=TaskType.DIAGNOSE),
            idx.lookup(["a", "b"]),
            self._ctx(idx),
        )
        # specialist_fanout uses ParallelPipeline of tool-bound Agents (not
        # Orchestrator + Specialist — Specialist's single-turn execute
        # doesn't process tool calls).
        assert isinstance(runnable, PipelineRunnable)
        assert len(runnable.pipeline.agents) == 2

    def test_builder_skips_human_capabilities_when_resolving_tools(self):
        idx = CapabilityIndex(create_registry(_echo))
        idx.annotate("approve", tool_name=HUMAN_SENTINEL, description="x", domain="d")
        proto = next(p for p in builtin_protocols() if p.id == "direct_response")
        runnable = proto.builder(_frame(), idx.lookup(["approve"]), self._ctx(idx))
        # No tools resolved (human-only capability skipped).
        assert runnable.agent.config.tools == []


# ---------------------------------------------------------------------------
# CognitiveCompiler
# ---------------------------------------------------------------------------


def _stub_compiler(
    *,
    policy: PolicyGate | None = None,
    on_approval: Any = None,
) -> tuple[CognitiveCompiler, CapabilityIndex]:
    idx = CapabilityIndex(create_registry(_echo))
    idx.annotate("c1", tool_name="_echo", description="echo", domain="d")
    reg = ProtocolRegistry()
    reg.register_many(builtin_protocols())
    compiler = CognitiveCompiler(
        protocols=reg,
        capabilities=idx,
        policy=policy or PolicyGate(),
        model="openai:gpt-4o-mini",
        on_approval=on_approval,
    )
    return compiler, idx


class TestCognitiveCompiler:
    def test_happy_path_returns_runnable(self):
        compiler, _ = _stub_compiler()
        runnable = asyncio.run(compiler.compile(_frame()))
        assert isinstance(runnable, AgentRunnable)

    def test_policy_denied_raises(self):
        # PLAN at MEDIUM matches plan_execute_validate (cap=MEDIUM), so the
        # registry doesn't reject it. With gate max_risk=LOW, the gate denies.
        compiler, _ = _stub_compiler(policy=PolicyGate(max_risk=Risk.LOW))
        with pytest.raises(PolicyDeniedError):
            asyncio.run(
                compiler.compile(
                    _frame(primary_goal=TaskType.PLAN, risk=Risk.MEDIUM),
                ),
            )

    def test_approval_required_wraps(self):
        async def approved(_f: GoalFrame, _v: PolicyVerdict) -> bool:
            return True

        compiler, _ = _stub_compiler(
            policy=PolicyGate(require_approval_above=Risk.LOW),
            on_approval=approved,
        )
        # plan_execute_validate caps at MEDIUM; PLAN+MEDIUM with threshold LOW
        # produces require_approval=True.
        runnable = asyncio.run(
            compiler.compile(_frame(primary_goal=TaskType.PLAN, risk=Risk.MEDIUM))
        )
        assert isinstance(runnable, _ApprovalRunnable)

    def test_approval_denied_at_runtime_raises(self):
        # Default callback denies — the compiled runnable refuses to execute.
        compiler, _ = _stub_compiler(
            policy=PolicyGate(require_approval_above=Risk.LOW),
        )
        runnable = asyncio.run(
            compiler.compile(_frame(primary_goal=TaskType.PLAN, risk=Risk.MEDIUM))
        )
        with pytest.raises(PolicyDeniedError, match="approval denied"):
            asyncio.run(runnable.execute("go"))

    def test_default_deny_callback(self):
        # Direct test of the module-level helper.
        assert asyncio.run(_default_deny(_frame(), PolicyVerdict(allow=True))) is False

    def test_compile_with_no_required_capabilities_skips_lookup(self):
        compiler, _ = _stub_compiler()
        # No required_capabilities listed → builder gets empty caps list, fine.
        runnable = asyncio.run(compiler.compile(_frame()))
        assert runnable.agent.config.tools == []

    def test_compile_resolves_required_capabilities(self):
        compiler, _ = _stub_compiler()
        runnable = asyncio.run(compiler.compile(_frame(required_capabilities=["c1"])))
        assert len(runnable.agent.config.tools) == 1


# ---------------------------------------------------------------------------
# Router (end-to-end with mock extractor)
# ---------------------------------------------------------------------------


class TestRouter:
    def _build_router(
        self,
        *,
        frame: GoalFrame,
        runnable_text: str = "answer",
        on_frame_calls: list[GoalFrame] | None = None,
    ) -> Router:
        extractor = MagicMock()
        extractor.invoke = MagicMock(return_value=_make_agent_result("", parsed=frame))

        compiler_mock = MagicMock(spec=CognitiveCompiler)

        async def _compile(_f: GoalFrame, run_id: str | None = None) -> AgentRunnable:
            agent = MagicMock()
            agent.invoke = MagicMock(return_value=_make_agent_result(runnable_text))
            return AgentRunnable(agent=agent, protocol_id="direct_response", frame=_f)

        compiler_mock.compile = _compile

        return Router(
            extractor=extractor,
            compiler=compiler_mock,
            on_frame=on_frame_calls.append if on_frame_calls is not None else None,
        )

    def test_dispatch_runs_extractor_then_runnable(self):
        frame = _frame()
        router = self._build_router(frame=frame, runnable_text="42")
        result = asyncio.run(router.dispatch("life?"))
        assert result.text == "42"
        assert result.protocol_id == "direct_response"

    def test_on_frame_callback_fires(self):
        frame = _frame()
        observed: list[GoalFrame] = []
        router = self._build_router(frame=frame, on_frame_calls=observed)
        asyncio.run(router.dispatch("?"))
        assert observed == [frame]

    def test_extractor_failure_raises(self):
        extractor = MagicMock()
        # No parsed -> error.
        extractor.invoke = MagicMock(return_value=_make_agent_result("oops"))
        router = Router(extractor=extractor, compiler=MagicMock(spec=CognitiveCompiler))
        with pytest.raises(FrameExtractionError):
            asyncio.run(router.extract("anything"))


# ---------------------------------------------------------------------------
# SkillIndex
# ---------------------------------------------------------------------------


def _skill(name: str = "triage", description: str = "Use when triaging.") -> Skill:
    return Skill(name=name, description=description, instructions="# Steps\n1. ...")


class TestSkillIndex:
    def test_register_and_get(self):
        idx = SkillIndex()
        sk = _skill("triage")
        idx.register(sk, domain="observability")
        assert idx.get("triage") is sk
        assert "triage" in idx
        assert len(idx) == 1

    def test_register_duplicate_raises(self):
        idx = SkillIndex()
        idx.register(_skill("triage"))
        with pytest.raises(ValueError, match="already registered"):
            idx.register(_skill("triage"))

    def test_get_unknown_raises(self):
        idx = SkillIndex()
        with pytest.raises(KeyError, match="Unknown skill"):
            idx.get("missing")

    def test_for_domain_filters_and_includes_global(self):
        idx = SkillIndex()
        idx.register(_skill("ops-triage"), domain="observability")
        idx.register(_skill("code-review"), domain="engineering")
        idx.register(_skill("safety-checklist"))  # global (domain="")
        observability = {s.name for s in idx.for_domain("observability")}
        engineering = {s.name for s in idx.for_domain("engineering")}
        unknown = {s.name for s in idx.for_domain("nonexistent")}
        # Global skill must appear in every domain's catalogue.
        assert observability == {"ops-triage", "safety-checklist"}
        assert engineering == {"code-review", "safety-checklist"}
        assert unknown == {"safety-checklist"}


# ---------------------------------------------------------------------------
# Router-Skills wiring (compiler attaches SkillsPlugin to compiled Agents)
# ---------------------------------------------------------------------------


class TestRouterSkillsWiring:
    """When a SkillIndex is wired, every compiled Agent gets a
    SkillsPlugin scoped to the frame's domain. When no SkillIndex is
    wired (or the domain has no matching skills), no plugin is attached."""

    def _compile_with_skills(self, idx: SkillIndex, frame: GoalFrame) -> Any:
        capabilities = CapabilityIndex(create_registry(_echo))
        capabilities.annotate("c1", tool_name="_echo", description="echo", domain="d")
        registry = ProtocolRegistry()
        registry.register_many(builtin_protocols())
        compiler = CognitiveCompiler(
            protocols=registry,
            capabilities=capabilities,
            policy=PolicyGate(),
            model="openai:gpt-4o-mini",
            skills=idx,
        )
        return asyncio.run(compiler.compile(frame))

    def test_no_skill_index_yields_no_plugins(self):
        capabilities = CapabilityIndex(create_registry(_echo))
        registry = ProtocolRegistry()
        registry.register_many(builtin_protocols())
        compiler = CognitiveCompiler(
            protocols=registry,
            capabilities=capabilities,
            policy=PolicyGate(),
            model="openai:gpt-4o-mini",
        )
        runnable = asyncio.run(compiler.compile(_frame()))
        assert runnable.agent.config.plugins == []

    def test_domain_match_attaches_skills_plugin(self):
        idx = SkillIndex()
        idx.register(_skill("ops-triage"), domain="observability")
        runnable = self._compile_with_skills(
            idx, _frame(primary_goal=TaskType.ANSWER, domain="observability")
        )
        plugins = runnable.agent.config.plugins
        assert len(plugins) == 1
        assert isinstance(plugins[0], SkillsPlugin)
        assert "ops-triage" in plugins[0].available_skills

    def test_domain_mismatch_yields_no_plugin(self):
        idx = SkillIndex()
        idx.register(_skill("ops-triage"), domain="observability")
        runnable = self._compile_with_skills(
            idx, _frame(primary_goal=TaskType.ANSWER, domain="research")
        )
        # No skill matches 'research' (and none are global), so no plugin.
        assert runnable.agent.config.plugins == []

    def test_global_skill_appears_in_every_domain(self):
        idx = SkillIndex()
        idx.register(_skill("safety-checklist"))  # global
        idx.register(_skill("ops-triage"), domain="observability")
        runnable = self._compile_with_skills(
            idx, _frame(primary_goal=TaskType.ANSWER, domain="research")
        )
        plugins = runnable.agent.config.plugins
        assert len(plugins) == 1
        # The global skill must be present even in domains that have
        # no domain-specific skills.
        assert "safety-checklist" in plugins[0].available_skills
        assert "ops-triage" not in plugins[0].available_skills


# ---------------------------------------------------------------------------
# LLM Protocol Picker — opt-in second mode
# ---------------------------------------------------------------------------


class TestFilterCandidates:
    """`ProtocolRegistry.filter_candidates` returns the same survivors
    that `select()`'s inline filter would produce — additive method."""

    def test_filter_isolates_handles(self):
        reg = ProtocolRegistry()
        reg.register_many(builtin_protocols())
        # ANSWER goal — only protocols with ANSWER in handles + risk_max
        # >= LOW survive. direct_response is the only one.
        survivors = reg.filter_candidates(_frame(primary_goal=TaskType.ANSWER))
        assert [p.id for p in survivors] == ["direct_response"]

    def test_filter_returns_multiple_when_goals_overlap(self):
        reg = ProtocolRegistry()
        reg.register_many(builtin_protocols())
        # COMPARE is handled by specialist_fanout (risk_max=MEDIUM) AND
        # debate (risk_max=LOW). At risk=LOW both survive.
        survivors = reg.filter_candidates(_frame(primary_goal=TaskType.COMPARE))
        ids = {p.id for p in survivors}
        assert {"specialist_fanout", "debate"}.issubset(ids)

    def test_filter_empty_on_no_match(self):
        reg = ProtocolRegistry()
        reg.register_many(builtin_protocols())
        # No protocol handles PLAN at HIGH risk.
        survivors = reg.filter_candidates(_frame(primary_goal=TaskType.PLAN, risk=Risk.HIGH))
        assert survivors == []


def _picker_compiler(picker, *, protocols=None):
    """Build a compiler with the picker plugged in. No skills, no a2a."""
    return CognitiveCompiler(
        protocols=protocols or _full_registry(),
        capabilities=CapabilityIndex(create_registry(_echo)),
        policy=PolicyGate(),
        model=MagicMock(),
        protocol_picker=picker,
    )


def _full_registry() -> ProtocolRegistry:
    reg = ProtocolRegistry()
    reg.register_many(builtin_protocols())
    return reg


class TestLLMProtocolPicker:
    """The opt-in picker path. Filters first, then asks the LLM.
    Falls back to rule-based on any picker failure."""

    def test_picker_disabled_by_default_uses_rule_based(self):
        """Regression: no `protocol_picker=` → unchanged behaviour."""
        compiler = CognitiveCompiler(
            protocols=_full_registry(),
            capabilities=CapabilityIndex(create_registry(_echo)),
            policy=PolicyGate(),
            model=MagicMock(),
        )
        assert compiler.protocol_picker is None

    def test_picker_short_circuits_on_single_candidate(self):
        """One candidate post-filter → picker NOT invoked, method =
        single_candidate. Verified via mock assertion."""
        from tulip.router import LLMProtocolPicker

        picker = MagicMock(spec=LLMProtocolPicker)
        compiler = _picker_compiler(picker)

        async def _drive() -> tuple[Any, str, str | None]:
            # ANSWER → only direct_response survives the filter.
            candidates = compiler.protocols.filter_candidates(
                _frame(primary_goal=TaskType.ANSWER),
                available_capabilities=set(),
            )
            return await compiler._pick_protocol(
                _frame(primary_goal=TaskType.ANSWER),
                candidates,
                run_id=None,
            )

        protocol, method, rationale = asyncio.run(_drive())
        assert protocol.id == "direct_response"
        assert method == "single_candidate"
        assert rationale is None
        picker.pick.assert_not_called()

    def test_picker_happy_path_returns_pick_with_rationale(self):
        """Multiple candidates + picker → picker called, method =
        llm_picked, rationale threaded through."""
        from tulip.router.picker import LLMProtocolPicker

        async def _fake_pick(frame, candidates):
            # Pick the second candidate, with a one-liner rationale.
            return candidates[1], "Picked debate for principled comparison."

        picker = MagicMock(spec=LLMProtocolPicker)
        picker.pick = _fake_pick
        compiler = _picker_compiler(picker)

        async def _drive():
            frame = _frame(primary_goal=TaskType.COMPARE)
            candidates = compiler.protocols.filter_candidates(frame, available_capabilities=set())
            assert len(candidates) >= 2  # precondition
            return await compiler._pick_protocol(frame, candidates, run_id=None)

        protocol, method, rationale = asyncio.run(_drive())
        assert method == "llm_picked"
        assert rationale == "Picked debate for principled comparison."
        # The mock chose candidates[1]; verify it's in the registry.
        assert protocol.id in {p.id for p in builtin_protocols()}

    def test_picker_raises_falls_back_to_rule_based(self):
        """Picker exception → fall back to _rank_key, method =
        rule_based_fallback. Passes ``run_id`` so the
        ``router.protocol.picker_fallback`` event is emitted on the
        bus — verified by the in-process subscriber below."""
        from tulip.observability import get_event_bus
        from tulip.router.picker import LLMProtocolPicker

        async def _boom(frame, candidates):
            raise RuntimeError("model unreachable")

        picker = MagicMock(spec=LLMProtocolPicker)
        picker.pick = _boom
        compiler = _picker_compiler(picker)

        async def _drive():
            frame = _frame(primary_goal=TaskType.COMPARE)
            candidates = compiler.protocols.filter_candidates(frame, available_capabilities=set())
            captured: list[Any] = []

            async def _collect() -> None:
                async for ev in get_event_bus().subscribe("picker-fallback-1"):
                    if ev.event_type == "router.protocol.picker_fallback":
                        captured.append(ev)
                        break

            collect_task = asyncio.create_task(_collect())
            result = await compiler._pick_protocol(frame, candidates, run_id="picker-fallback-1")
            await asyncio.wait_for(collect_task, timeout=0.2)
            return result, captured

        (protocol, method, rationale), captured = asyncio.run(_drive())
        assert method == "rule_based_fallback"
        assert rationale is None
        # The fallback must produce a valid registered protocol.
        assert protocol.id in {p.id for p in builtin_protocols()}
        # The picker_fallback event must fire with the wrapped error message.
        assert len(captured) == 1
        assert captured[0].data["error"].startswith("RuntimeError")
        assert "model unreachable" in captured[0].data["error"]

    def test_picker_unknown_id_falls_back(self):
        """Picker returns id not in candidates → PickerError caught,
        rule-based fallback wins. Passes ``run_id`` so the fallback
        event fires (covers the PickerError branch in the emitter)."""
        from tulip.router.picker import LLMProtocolPicker, PickerError

        async def _hallucinate(frame, candidates):
            raise PickerError("picker hallucinated nonexistent_id")

        picker = MagicMock(spec=LLMProtocolPicker)
        picker.pick = _hallucinate
        compiler = _picker_compiler(picker)

        async def _drive():
            frame = _frame(primary_goal=TaskType.COMPARE)
            candidates = compiler.protocols.filter_candidates(frame, available_capabilities=set())
            return await compiler._pick_protocol(frame, candidates, run_id="picker-fallback-2")

        protocol, method, _ = asyncio.run(_drive())
        assert method == "rule_based_fallback"
        assert protocol.id in {p.id for p in builtin_protocols()}

    def test_compile_zero_candidates_raises_without_calling_picker(self):
        """No protocol fits the filter → NoMatchingProtocolError, picker
        never invoked (saves a token on impossible dispatches)."""
        from tulip.router.picker import LLMProtocolPicker

        picker = MagicMock(spec=LLMProtocolPicker)
        compiler = _picker_compiler(picker)

        async def _drive():
            # PLAN at HIGH risk — no built-in protocol covers this.
            return await compiler.compile(_frame(primary_goal=TaskType.PLAN, risk=Risk.HIGH))

        with pytest.raises(NoMatchingProtocolError):
            asyncio.run(_drive())
        picker.pick.assert_not_called()


class TestLLMProtocolPickerInternals:
    """Direct tests of `LLMProtocolPicker.pick()` — exercises the
    prompt rendering, Agent invocation, and validation paths that
    the higher-level compiler tests mock through."""

    def test_pick_happy_path_returns_tuple(self):
        from unittest.mock import patch

        from tulip.router.picker import LLMProtocolPicker, PickedProtocol

        picker = LLMProtocolPicker(model=MagicMock())
        protocols = builtin_protocols()
        # debate + specialist_fanout both handle COMPARE.
        candidates = [p for p in protocols if p.id in ("debate", "specialist_fanout")]

        with patch("tulip.agent.agent.Agent") as mock_agent_cls:
            mock_agent_cls.return_value.invoke.return_value = _make_agent_result(
                "picker output",
                parsed=PickedProtocol(
                    protocol_id="debate",
                    rationale="Debate is canonical for COMPARE.",
                ),
            )
            picked, rationale = asyncio.run(
                picker.pick(_frame(primary_goal=TaskType.COMPARE), candidates)
            )

        assert picked.id == "debate"
        assert "canonical" in rationale.lower()

    def test_pick_parse_failure_raises_picker_error(self):
        from unittest.mock import patch

        from tulip.router.picker import LLMProtocolPicker, PickerError

        picker = LLMProtocolPicker(model=MagicMock())
        protocols = builtin_protocols()
        candidates = [p for p in protocols if p.id in ("debate", "specialist_fanout")]

        with patch("tulip.agent.agent.Agent") as mock_agent_cls:
            # parsed=None simulates a schema-rejection from the model.
            mock_agent_cls.return_value.invoke.return_value = _make_agent_result(
                "garbage",
                parsed=None,
            )
            with pytest.raises(PickerError, match="did not return PickedProtocol"):
                asyncio.run(picker.pick(_frame(primary_goal=TaskType.COMPARE), candidates))

    def test_pick_unknown_id_raises_picker_error(self):
        from unittest.mock import patch

        from tulip.router.picker import (
            LLMProtocolPicker,
            PickedProtocol,
            PickerError,
        )

        picker = LLMProtocolPicker(model=MagicMock())
        protocols = builtin_protocols()
        candidates = [p for p in protocols if p.id in ("debate", "specialist_fanout")]

        with patch("tulip.agent.agent.Agent") as mock_agent_cls:
            mock_agent_cls.return_value.invoke.return_value = _make_agent_result(
                "x",
                parsed=PickedProtocol(
                    protocol_id="nonexistent_protocol",
                    rationale="I made this up.",
                ),
            )
            with pytest.raises(PickerError, match="unknown protocol_id"):
                asyncio.run(picker.pick(_frame(primary_goal=TaskType.COMPARE), candidates))

    def test_default_system_prompt_is_used(self):
        from tulip.router.picker import _DEFAULT_SYSTEM_PROMPT, LLMProtocolPicker

        picker = LLMProtocolPicker(model=MagicMock())
        assert picker.system_prompt == _DEFAULT_SYSTEM_PROMPT

    def test_custom_system_prompt_override(self):
        from tulip.router.picker import LLMProtocolPicker

        custom = "You are a strict canonical-only picker."
        picker = LLMProtocolPicker(model=MagicMock(), system_prompt=custom)
        assert picker.system_prompt == custom
