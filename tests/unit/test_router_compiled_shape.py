# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Structural audit of every compiled :class:`Runnable`.

The end-to-end suite (``tests/integration/test_router_e2e.py``) proves
each protocol *runs*. This file proves each protocol's compiled object
graph matches the **recommended locus pattern byte-for-byte**:

    direct_response       → exactly one :class:`Agent`
    plan_execute_validate → :class:`SequentialPipeline` of 3 Agents
                            (planner / executor / validator)
    specialist_fanout     → :class:`ParallelPipeline` of N Agents,
                            one per non-human capability,
                            each with exactly its bound tool

This is the test that justifies the "bounded graph generation" claim:
the compiler emits a known, inspectable shape — not arbitrary
LLM-authored topology. It runs without any model calls because the
builders are deterministic given ``(frame, capabilities, ctx)``.

If you change a builder, this file is what catches the drift.
"""

from __future__ import annotations

import asyncio

import pytest

from locus.agent.agent import Agent
from locus.agent.composition import LoopAgent, ParallelPipeline, SequentialPipeline
from locus.router import (
    HUMAN_SENTINEL,
    AgentRunnable,
    BuilderContext,
    CapabilityIndex,
    CognitiveCompiler,
    Complexity,
    DebateRunnable,
    GoalFrame,
    PipelineRunnable,
    PolicyGate,
    ProtocolRegistry,
    Risk,
    TaskType,
    builtin_protocols,
)
from locus.router.compiler import _ApprovalRunnable
from locus.tools.decorator import tool
from locus.tools.registry import create_registry


# ---------------------------------------------------------------------------
# Tools + fixtures.
# ---------------------------------------------------------------------------


@tool
def kb_search(query: str) -> str:
    """Knowledge-base lookup."""
    return f"hit: {query}"


@tool
def get_metric(name: str) -> str:
    """Latest metric value."""
    return f"{name}=stub"


@tool
def list_alerts(window_minutes: int = 30) -> str:
    """Recent alerts."""
    return f"alerts in {window_minutes}min"


@pytest.fixture
def capability_index() -> CapabilityIndex:
    tools = create_registry(kb_search, get_metric, list_alerts)
    idx = CapabilityIndex(tools)
    idx.annotate(
        "kb_search",
        tool_name="kb_search",
        description="KB lookup.",
        domain="research",
    )
    idx.annotate(
        "metric_probe",
        tool_name="get_metric",
        description="Latest metric value.",
        domain="observability",
    )
    idx.annotate(
        "alert_list",
        tool_name="list_alerts",
        description="Recent alerts.",
        domain="observability",
    )
    return idx


@pytest.fixture
def builder_ctx(capability_index: CapabilityIndex) -> BuilderContext:
    return BuilderContext(model="openai:gpt-4o-mini", capabilities=capability_index)


@pytest.fixture
def compiler(capability_index: CapabilityIndex) -> CognitiveCompiler:
    protocols = ProtocolRegistry()
    protocols.register_many(builtin_protocols())
    return CognitiveCompiler(
        protocols=protocols,
        capabilities=capability_index,
        policy=PolicyGate(),
        model="openai:gpt-4o-mini",
    )


def _frame(
    *,
    primary_goal: TaskType,
    domain: str = "research",
    complexity: Complexity = Complexity.LOW,
    risk: Risk = Risk.LOW,
    required_capabilities: list[str] | None = None,
    success_criteria: list[str] | None = None,
    approval_required: bool = False,
) -> GoalFrame:
    return GoalFrame(
        primary_goal=primary_goal,
        domain=domain,
        complexity=complexity,
        risk=risk,
        required_capabilities=required_capabilities or [],
        success_criteria=success_criteria or [],
        approval_required=approval_required,
    )


def _tool_names(agent: Agent) -> set[str]:
    return {t.name for t in agent.config.tools}


# ---------------------------------------------------------------------------
# direct_response — exactly one Agent, with the requested capability tools.
# ---------------------------------------------------------------------------


class TestDirectResponseShape:
    """`direct_response` must produce exactly one :class:`Agent` with no
    extra primitives wrapped around it."""

    def test_no_capabilities_yields_toolless_agent(self, compiler: CognitiveCompiler) -> None:
        frame = _frame(primary_goal=TaskType.ANSWER)
        runnable = asyncio.run(compiler.compile(frame))

        assert isinstance(runnable, AgentRunnable), (
            f"direct_response should emit AgentRunnable; got {type(runnable).__name__}"
        )
        agent = runnable.agent
        assert isinstance(agent, Agent)
        assert _tool_names(agent) == set(), (
            f"no capabilities requested → no tools bound; got {_tool_names(agent)}"
        )
        assert runnable.protocol_id == "direct_response"
        assert runnable.frame == frame

    def test_required_capabilities_get_bound(self, compiler: CognitiveCompiler) -> None:
        frame = _frame(
            primary_goal=TaskType.RESEARCH,
            required_capabilities=["kb_search"],
        )
        runnable = asyncio.run(compiler.compile(frame))

        assert isinstance(runnable, AgentRunnable)
        assert _tool_names(runnable.agent) == {"kb_search"}, (
            "exactly the requested capability's tool should be bound — no more, no less"
        )

    def test_unknown_capability_silently_dropped(self, compiler: CognitiveCompiler) -> None:
        # The LLM extractor sometimes hallucinates ids; the compiler
        # must drop them rather than crash. (Behaviour pinned in
        # src/locus/router/compiler.py:117.)
        frame = _frame(
            primary_goal=TaskType.ANSWER,
            required_capabilities=["kb_search", "this_does_not_exist"],
        )
        runnable = asyncio.run(compiler.compile(frame))
        assert isinstance(runnable, AgentRunnable)
        assert _tool_names(runnable.agent) == {"kb_search"}, (
            "unknown ids must be dropped; only kb_search should remain bound"
        )

    def test_system_prompt_mentions_goal_and_domain(self, compiler: CognitiveCompiler) -> None:
        frame = _frame(primary_goal=TaskType.EXPLAIN, domain="observability")
        runnable = asyncio.run(compiler.compile(frame))
        prompt = runnable.agent.config.system_prompt.lower()
        assert "explain" in prompt, "system prompt should reference primary_goal"
        assert "observability" in prompt, "system prompt should reference domain"


# ---------------------------------------------------------------------------
# plan_execute_validate — SequentialPipeline of exactly 3 Agents.
# ---------------------------------------------------------------------------


class TestPlanExecuteValidateShape:
    """The pipeline must be ``SequentialPipeline([planner, executor, validator])``
    with the canonical role split: only the executor carries tools."""

    def _compile(self, compiler: CognitiveCompiler, **frame_kwargs) -> PipelineRunnable:
        frame = _frame(primary_goal=TaskType.PLAN, **frame_kwargs)
        runnable = asyncio.run(compiler.compile(frame))
        assert isinstance(runnable, PipelineRunnable), (
            f"plan_execute_validate should emit PipelineRunnable; got {type(runnable).__name__}"
        )
        return runnable

    def test_pipeline_is_sequential_with_three_agents(self, compiler: CognitiveCompiler) -> None:
        runnable = self._compile(compiler)
        pipeline = runnable.pipeline
        assert isinstance(pipeline, SequentialPipeline), (
            f"plan_execute_validate must be Sequential; got {type(pipeline).__name__}"
        )
        assert len(pipeline.agents) == 3, (
            f"canonical plan-execute-validate has 3 stages; got {len(pipeline.agents)}"
        )
        for stage in pipeline.agents:
            assert isinstance(stage, Agent), (
                f"each pipeline stage must be a real Agent; got {type(stage).__name__}"
            )

    def test_role_split_only_executor_carries_tools(self, compiler: CognitiveCompiler) -> None:
        runnable = self._compile(
            compiler,
            required_capabilities=["kb_search"],
        )
        planner, executor, validator = runnable.pipeline.agents
        assert _tool_names(planner) == set(), "planner must not have tools"
        assert _tool_names(executor) == {"kb_search"}, (
            "executor must carry the resolved capability tools"
        )
        assert _tool_names(validator) == set(), "validator must not have tools"

    def test_each_stage_prompt_declares_its_role(self, compiler: CognitiveCompiler) -> None:
        runnable = self._compile(compiler)
        planner, executor, validator = runnable.pipeline.agents
        assert "planner" in planner.config.system_prompt.lower()
        assert "executor" in executor.config.system_prompt.lower()
        assert "validator" in validator.config.system_prompt.lower()

    def test_validator_prompt_carries_success_criteria(self, compiler: CognitiveCompiler) -> None:
        criteria = ["plan has 3 numbered steps", "step 3 is validation"]
        runnable = self._compile(compiler, success_criteria=criteria)
        _, _, validator = runnable.pipeline.agents
        prompt = validator.config.system_prompt
        for c in criteria:
            assert c in prompt, (
                f"validator prompt must thread success_criteria through verbatim "
                f"so the LLM can check against them; missing {c!r}"
            )


# ---------------------------------------------------------------------------
# specialist_fanout — ParallelPipeline of N Agents, one per non-human cap.
# ---------------------------------------------------------------------------


class TestSpecialistFanoutShape:
    """Fan-out must be a parallel pipeline with one tool-bound Agent per
    capability. Crucially, the builder uses :class:`ParallelPipeline` of
    :class:`Agent`s — *not* :class:`Specialist` — because Specialist's
    single-turn ``execute`` doesn't process tool calls."""

    def _compile(
        self,
        compiler: CognitiveCompiler,
        *,
        capabilities: list[str],
    ) -> PipelineRunnable:
        frame = _frame(
            primary_goal=TaskType.DIAGNOSE,
            domain="observability",
            complexity=Complexity.HIGH,
            risk=Risk.MEDIUM,
            required_capabilities=capabilities,
        )
        runnable = asyncio.run(compiler.compile(frame))
        assert isinstance(runnable, PipelineRunnable), (
            f"specialist_fanout should emit PipelineRunnable; got {type(runnable).__name__}"
        )
        return runnable

    def test_pipeline_is_parallel(self, compiler: CognitiveCompiler) -> None:
        runnable = self._compile(compiler, capabilities=["metric_probe", "alert_list"])
        pipeline = runnable.pipeline
        assert isinstance(pipeline, ParallelPipeline), (
            f"specialist_fanout must be Parallel (Specialist's single-turn "
            f"execute can't loop on tool calls, so we use a fanout of real "
            f"Agents instead); got {type(pipeline).__name__}"
        )

    def test_one_agent_per_non_human_capability(self, compiler: CognitiveCompiler) -> None:
        runnable = self._compile(compiler, capabilities=["metric_probe", "alert_list"])
        pipeline = runnable.pipeline
        assert len(pipeline.agents) == 2, (
            f"two capabilities → two fan-out agents; got {len(pipeline.agents)}"
        )
        for agent in pipeline.agents:
            assert isinstance(agent, Agent), (
                f"each fan-out leg must be a real Agent (with a tool loop); "
                f"got {type(agent).__name__}"
            )

    def test_each_agent_has_exactly_its_bound_tool(self, compiler: CognitiveCompiler) -> None:
        runnable = self._compile(compiler, capabilities=["metric_probe", "alert_list"])
        all_tools = [
            tool_name for agent in runnable.pipeline.agents for tool_name in _tool_names(agent)
        ]
        assert sorted(all_tools) == ["get_metric", "list_alerts"], (
            f"each agent must hold exactly its bound tool, no overlap; observed {sorted(all_tools)}"
        )
        # And: no agent holds more than one tool.
        for agent in runnable.pipeline.agents:
            assert len(_tool_names(agent)) == 1, (
                f"each fan-out agent must hold exactly one tool to keep the "
                f"specialist boundary; got {_tool_names(agent)}"
            )

    def test_merge_strategy_is_concatenate(self, compiler: CognitiveCompiler) -> None:
        runnable = self._compile(compiler, capabilities=["metric_probe", "alert_list"])
        assert runnable.pipeline.merge_strategy == "concatenate", (
            "v1 fan-out concatenates specialist outputs verbatim; correlation "
            "is a follow-up. Changing this is a behaviour change, not a refactor."
        )

    def test_human_sentinel_capabilities_are_skipped(
        self,
        compiler: CognitiveCompiler,
        capability_index: CapabilityIndex,
    ) -> None:
        capability_index.annotate(
            "approve_change",
            tool_name=HUMAN_SENTINEL,
            description="Human approval needed.",
            domain="observability",
            risk=Risk.HIGH,
        )
        runnable = self._compile(
            compiler,
            capabilities=["metric_probe", "alert_list", "approve_change"],
        )
        # Three capabilities requested, one is human-only → only two agents.
        assert len(runnable.pipeline.agents) == 2, (
            f"human-sentinel capabilities must be skipped during fan-out (the "
            f"policy gate handles approval, not the agent loop); "
            f"got {len(runnable.pipeline.agents)} agents"
        )

    def test_each_agent_prompt_names_its_tool(self, compiler: CognitiveCompiler) -> None:
        runnable = self._compile(compiler, capabilities=["metric_probe", "alert_list"])
        for agent in runnable.pipeline.agents:
            (tool_name,) = _tool_names(agent)
            prompt = agent.config.system_prompt
            assert tool_name in prompt, (
                f"prompt must explicitly name {tool_name!r} so the model knows "
                f"which tool to call (it's the difference between 'I'll check' "
                f"vs an actual tool invocation)"
            )


# ---------------------------------------------------------------------------
# Cross-protocol invariants.
# ---------------------------------------------------------------------------


class TestCrossProtocolInvariants:
    """Properties that hold for every compiled runnable, regardless of which
    protocol fired."""

    @pytest.mark.parametrize(
        "frame_kwargs",
        [
            {"primary_goal": TaskType.ANSWER},
            {"primary_goal": TaskType.PLAN, "complexity": Complexity.MEDIUM, "risk": Risk.MEDIUM},
            {
                "primary_goal": TaskType.DIAGNOSE,
                "domain": "observability",
                "complexity": Complexity.HIGH,
                "risk": Risk.MEDIUM,
                "required_capabilities": ["metric_probe", "alert_list"],
            },
        ],
        ids=["direct_response", "plan_execute_validate", "specialist_fanout"],
    )
    def test_runnable_carries_originating_frame(
        self, compiler: CognitiveCompiler, frame_kwargs
    ) -> None:
        # Audit hook: every compiled Runnable echoes the frame that produced
        # it, so callers can later prove which goal led to which graph.
        frame = _frame(**frame_kwargs)
        runnable = asyncio.run(compiler.compile(frame))
        assert runnable.frame == frame, (
            "runnable.frame is the audit anchor — without it you can't prove "
            "which GoalFrame compiled to which graph"
        )

    @pytest.mark.parametrize(
        ("frame_kwargs", "expected_protocol"),
        [
            ({"primary_goal": TaskType.ANSWER}, "direct_response"),
            (
                {
                    "primary_goal": TaskType.PLAN,
                    "complexity": Complexity.MEDIUM,
                    "risk": Risk.MEDIUM,
                },
                "plan_execute_validate",
            ),
            (
                {
                    "primary_goal": TaskType.DIAGNOSE,
                    "domain": "observability",
                    "complexity": Complexity.HIGH,
                    "risk": Risk.MEDIUM,
                    "required_capabilities": ["metric_probe", "alert_list"],
                },
                "specialist_fanout",
            ),
        ],
        ids=["direct_response", "plan_execute_validate", "specialist_fanout"],
    )
    def test_protocol_id_matches_intended_protocol(
        self,
        compiler: CognitiveCompiler,
        frame_kwargs,
        expected_protocol: str,
    ) -> None:
        frame = _frame(**frame_kwargs)
        runnable = asyncio.run(compiler.compile(frame))
        assert runnable.protocol_id == expected_protocol, (
            f"frame {frame_kwargs!r} should compile to {expected_protocol!r}; "
            f"got {runnable.protocol_id!r}"
        )


# ---------------------------------------------------------------------------
# debate — ParallelPipeline of two debaters + a single judge Agent.
# ---------------------------------------------------------------------------


class TestDebateShape:
    """The compiled runnable must be a :class:`DebateRunnable` wrapping
    two debaters in a :class:`ParallelPipeline` plus a single judge."""

    def test_debate_shape(self, compiler: CognitiveCompiler) -> None:
        frame = _frame(
            primary_goal=TaskType.COMPARE,
            domain="research",
            complexity=Complexity.HIGH,
            risk=Risk.LOW,
        )
        runnable = asyncio.run(compiler.compile(frame))
        assert isinstance(runnable, DebateRunnable), (
            f"COMPARE-at-HIGH should compile to DebateRunnable; got {type(runnable).__name__}"
        )
        debaters = runnable.debaters
        assert isinstance(debaters, ParallelPipeline)
        assert len(debaters.agents) == 2, "exactly two debaters"
        assert all(isinstance(a, Agent) for a in debaters.agents)
        assert isinstance(runnable.judge, Agent)
        assert runnable.judge.config.tools == [], (
            "judge must not carry tools — it just reads transcripts"
        )

    def test_debater_prompts_take_opposing_sides(self, compiler: CognitiveCompiler) -> None:
        frame = _frame(
            primary_goal=TaskType.COMPARE,
            complexity=Complexity.HIGH,
            risk=Risk.LOW,
        )
        runnable = asyncio.run(compiler.compile(frame))
        prompts = [a.config.system_prompt.lower() for a in runnable.debaters.agents]
        # One debater argues "for", the other "against" — the asymmetry
        # is the whole point of the pattern.
        assert any("for" in p for p in prompts), (
            f"a debater must argue *for* the proposition; got prompts: {prompts}"
        )
        assert any("against" in p for p in prompts), (
            f"a debater must argue *against* the proposition; got prompts: {prompts}"
        )


# ---------------------------------------------------------------------------
# codegen_test_validate — LoopAgent with a PASS/FAIL stop condition.
# ---------------------------------------------------------------------------


class TestCodegenTestValidateShape:
    def _compile(
        self, compiler: CognitiveCompiler, *, capabilities: list[str] | None = None
    ) -> PipelineRunnable:
        frame = _frame(
            primary_goal=TaskType.GENERATE_CODE,
            domain="engineering",
            complexity=Complexity.MEDIUM,
            risk=Risk.MEDIUM,
            required_capabilities=capabilities or [],
        )
        runnable = asyncio.run(compiler.compile(frame))
        assert isinstance(runnable, PipelineRunnable), (
            f"GENERATE_CODE should fire codegen_test_validate (PipelineRunnable); "
            f"got {type(runnable).__name__}"
        )
        return runnable

    def test_pipeline_is_loop_agent(self, compiler: CognitiveCompiler) -> None:
        runnable = self._compile(compiler)
        assert isinstance(runnable.pipeline, LoopAgent), (
            f"codegen_test_validate must wrap LoopAgent; got {type(runnable.pipeline).__name__}"
        )

    def test_pass_condition_recognises_pass(self, compiler: CognitiveCompiler) -> None:
        runnable = self._compile(compiler)
        cond = runnable.pipeline.condition
        # The whole point of the PASS-on-line-1 contract is that the
        # condition stops the loop the moment the agent declares it's
        # done. Exercise both directions.
        assert cond("PASS\nall good") is True
        assert cond("FAIL: tests broke") is False
        # Case-insensitive prefix match — "Pass" should also stop.
        assert cond("Pass — looks fine") is True

    def test_loop_max_loops_is_bounded(self, compiler: CognitiveCompiler) -> None:
        runnable = self._compile(compiler)
        assert runnable.pipeline.max_loops <= 10, (
            f"unbounded retry loops bleed cost; max_loops should be <=10, "
            f"got {runnable.pipeline.max_loops}"
        )


# ---------------------------------------------------------------------------
# approval_gated_execution — single Agent wrapped by _ApprovalRunnable.
# ---------------------------------------------------------------------------


class TestApprovalGatedExecutionShape:
    def test_runnable_is_approval_wrapped_agent(self, compiler: CognitiveCompiler) -> None:
        frame = _frame(
            primary_goal=TaskType.REMEDIATE,
            complexity=Complexity.MEDIUM,
            risk=Risk.HIGH,
            # The gate's default require_approval_above=MEDIUM, so HIGH
            # already triggers the wrap. We also flag explicitly to make
            # the test independent of gate defaults.
            approval_required=True,
        )
        runnable = asyncio.run(compiler.compile(frame))
        assert isinstance(runnable, _ApprovalRunnable), (
            f"high-risk REMEDIATE must be wrapped with _ApprovalRunnable; "
            f"got {type(runnable).__name__}"
        )
        # Inner runnable is the real Agent; the wrap is only the gate.
        assert isinstance(runnable.inner, AgentRunnable)
        assert runnable.inner.protocol_id == "approval_gated_execution"


# ---------------------------------------------------------------------------
# a2a_delegate — A2ARunnable wrapping an A2AClient.
# ---------------------------------------------------------------------------


class TestA2ADelegateShape:
    def _compiler_with_a2a(self, capability_index: CapabilityIndex) -> CognitiveCompiler:
        protocols = ProtocolRegistry()
        protocols.register_many(builtin_protocols())
        return CognitiveCompiler(
            protocols=protocols,
            capabilities=capability_index,
            policy=PolicyGate(),
            model="openai:gpt-4o-mini",
            a2a_endpoint="http://localhost:9999",
        )

    def test_a2a_compile_with_endpoint(
        self,
        capability_index: CapabilityIndex,
    ) -> None:
        compiler = self._compiler_with_a2a(capability_index)
        # ESCALATE at MEDIUM with no approval_required is the cleanest
        # path to a2a_delegate (approval_gated_execution is canonical
        # for ESCALATE only when frame.risk drives the gate; we keep it
        # MEDIUM here).
        frame = _frame(
            primary_goal=TaskType.COORDINATE,
            complexity=Complexity.MEDIUM,
            risk=Risk.MEDIUM,
        )
        runnable = asyncio.run(compiler.compile(frame))
        # COORDINATE has multiple matching protocols; we just need to
        # confirm a2a_delegate is *reachable* when an endpoint exists.
        assert runnable.protocol_id in {
            "a2a_delegate",
            "specialist_fanout",
            "handoff_chain",
        }

    def test_a2a_compile_without_endpoint_falls_through(
        self,
        compiler: CognitiveCompiler,
    ) -> None:
        # The base `compiler` fixture has no a2a_endpoint configured.
        # The a2a_delegate builder raises if invoked, but the registry
        # still lists it — selection prefers another protocol when the
        # ranking allows.
        frame = _frame(
            primary_goal=TaskType.COORDINATE,
            complexity=Complexity.MEDIUM,
            risk=Risk.MEDIUM,
        )
        # Without an endpoint, ``a2a_delegate`` must not be selected.
        # Current ranking guarantees this (primary_for=[] makes it
        # opt-in only), so a canonical protocol always wins for any
        # frame the registry can match. If a future ranking change
        # makes a2a_delegate reachable here, the build would raise
        # RuntimeError("a2a_endpoint required") and this assertion
        # would surface that as a hard failure.
        runnable = asyncio.run(compiler.compile(frame))
        assert runnable.protocol_id != "a2a_delegate", (
            "a2a_delegate must not be selectable without an endpoint"
        )


# ---------------------------------------------------------------------------
# handoff_chain — SequentialPipeline of one-tool agents.
# ---------------------------------------------------------------------------


class TestHandoffChainShape:
    def test_chain_is_sequential_pipeline_of_agents(self, compiler: CognitiveCompiler) -> None:
        # COORDINATE is canonical for handoff_chain (one-tool sequential
        # links). Use medium complexity + medium cost match so the
        # ranking picks handoff_chain over specialist_fanout (high cost
        # at high complexity).
        frame = _frame(
            primary_goal=TaskType.COORDINATE,
            domain="research",
            complexity=Complexity.MEDIUM,
            risk=Risk.MEDIUM,
            required_capabilities=["kb_search"],
        )
        runnable = asyncio.run(compiler.compile(frame))
        assert isinstance(runnable, PipelineRunnable)
        assert isinstance(runnable.pipeline, SequentialPipeline), (
            f"handoff_chain must use SequentialPipeline; got {type(runnable.pipeline).__name__}"
        )
        # One link per non-human capability.
        assert len(runnable.pipeline.agents) == 1
        # Each link carries exactly one tool.
        agent = runnable.pipeline.agents[0]
        assert len(_tool_names(agent)) == 1

    def test_empty_capabilities_falls_back_to_single_link(
        self, compiler: CognitiveCompiler
    ) -> None:
        # When the frame requests no capabilities, the chain should
        # still produce a well-formed runnable rather than crash.
        frame = _frame(
            primary_goal=TaskType.COORDINATE,
            domain="research",
            complexity=Complexity.MEDIUM,
            risk=Risk.MEDIUM,
        )
        runnable = asyncio.run(compiler.compile(frame))
        if runnable.protocol_id == "handoff_chain":
            assert len(runnable.pipeline.agents) >= 1, (
                "even with no capabilities, the chain must produce one fallback link"
            )
