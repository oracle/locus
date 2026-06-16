# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""End-to-end integration tests for the cognitive router (`locus.router`).

Four scenarios, each one a load-bearing claim from the design:

A. **Bounded routing** — no matter what the LLM produces, every
   dispatched runnable's ``protocol_id`` is in the registered set.
   This is the framework's central guarantee: the LLM cannot escape
   the protocol catalogue.

B. **Routing accuracy** — for a labelled corpus of prompts, the
   extractor's ``GoalFrame.primary_goal`` matches the expected goal
   and the registry picks the expected protocol at least
   ``ROUTING_ACCURACY_FLOOR`` of the time. This is the headline
   paper claim.

C. **Real execution per protocol** — each v1 protocol
   (``direct_response`` / ``plan_execute_validate`` /
   ``specialist_fanout``) actually runs end-to-end against a real
   model and returns a non-empty :class:`RunnableResult` with the
   correct ``protocol_id``.

D. **Capability binding** — when a frame requires a tool-backed
   capability, the underlying ``Tool`` actually fires during
   execution. We use a recording tool to count invocations.

All tests use the session-scoped ``model`` fixture (OCI / OpenAI
auto-detected) and are marked ``requires_model`` so they skip
cleanly when no provider is configured.
"""

from __future__ import annotations

from typing import Any

import pytest

from locus import Agent, tool
from locus.router import (
    CapabilityIndex,
    CognitiveCompiler,
    Complexity,
    GoalFrame,
    PolicyGate,
    ProtocolRegistry,
    Risk,
    Router,
    TaskType,
    builtin_protocols,
)
from locus.router.protocol import NoMatchingProtocolError
from locus.router.runnable import (
    PipelineRunnable,
    RunnableResult,
)
from locus.router.runtime import FrameExtractionError
from locus.tools.registry import create_registry


pytestmark = [pytest.mark.integration, pytest.mark.requires_model]


# ---------------------------------------------------------------------------
# Labelled corpus — used by Scenarios A and B.
# ---------------------------------------------------------------------------

# Each entry: (prompt, expected_primary_goal, expected_protocol_id).
# Picked to span the three v1 protocols. The expected_goal is the
# goal we *prefer* the extractor to pick; ROUTING_ACCURACY_FLOOR
# accommodates honest disagreements (e.g. EXPLAIN vs ANSWER).
LABELLED_CORPUS: tuple[tuple[str, TaskType, str], ...] = (
    # --- direct_response (single-call) ---
    (
        "What does the locus.router module do in the locus SDK?",
        TaskType.ANSWER,
        "direct_response",
    ),
    (
        "Explain in two sentences how a Goal Frame differs from a free-form prompt.",
        TaskType.EXPLAIN,
        "direct_response",
    ),
    (
        "Look up our knowledge-base entry on retrieval-augmented generation.",
        TaskType.RESEARCH,
        "direct_response",
    ),
    # --- plan_execute_validate (linear pipeline) ---
    (
        "Write a three-step plan to migrate our checkout service to a new payment provider, "
        "including a validation step that checks the test coverage.",
        TaskType.PLAN,
        "plan_execute_validate",
    ),
    (
        "Generate Python code for a function that computes the SHA-256 of a file in 64KB chunks, "
        "then add doctests that verify it on the empty file and a 1MB random file.",
        TaskType.GENERATE_CODE,
        "plan_execute_validate",
    ),
    (
        "Build an OpenAPI 3.1 schema for a /users endpoint that supports GET, POST, and DELETE, "
        "with a validation step that lints the schema.",
        TaskType.BUILD,
        "plan_execute_validate",
    ),
    # --- specialist_fanout (parallel fan-out — canonical for DIAGNOSE / MONITOR) ---
    (
        "Diagnose what's slowing down checkout right now — pull recent alerts AND the latency_p99 "
        "metric, then correlate them.",
        TaskType.DIAGNOSE,
        "specialist_fanout",
    ),
    (
        "Monitor the system: list any active alerts and report the latest CPU value side-by-side.",
        TaskType.MONITOR,
        "specialist_fanout",
    ),
    # --- debate (canonical for COMPARE — beats specialist_fanout in ranking) ---
    (
        "Compare CPU and 5xx error metrics across the catalog and checkout services, then say "
        "which service is hotter.",
        TaskType.COMPARE,
        "debate",
    ),
)

# Per-axis accuracy floors. Set conservatively — we'd rather catch
# real regressions than flake the suite; tune up once we have
# stable numbers from CI.
ROUTING_ACCURACY_FLOOR_GOAL = 0.7
ROUTING_ACCURACY_FLOOR_PROTOCOL = 0.7

# How many of the 9 corpus prompts to actually dispatch end-to-end
# in scenarios C/D — full execution is expensive against the real
# provider. The rest still exercise the cheap extract-and-select
# path in scenarios A and B.
LIVE_EXECUTION_BUDGET = 3


# ---------------------------------------------------------------------------
# Tools + capabilities used across scenarios.
# ---------------------------------------------------------------------------


class _ToolCallRecorder:
    """Side-effect counter for Scenario D — records each tool invocation."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def reset(self) -> None:
        self.calls.clear()

    def record(self, name: str, **kwargs: Any) -> None:
        self.calls.append((name, kwargs))


_RECORDER = _ToolCallRecorder()


@tool
def kb_search(query: str) -> str:
    """Search the knowledge base for a topic and return a short summary."""
    _RECORDER.record("kb_search", query=query)
    canned = {
        "rag": "RAG = retrieve documents from a vector store and condition the LLM on them.",
        "router": "locus.router compiles a typed GoalFrame onto bounded orchestration shapes.",
        "goal frame": "A GoalFrame is a typed Pydantic schema the LLM extracts; "
        "it never authors orchestration topology.",
    }
    for k, v in canned.items():
        if k in query.lower():
            return v
    return f"No KB entry for {query!r}; suggest a richer query."


@tool
def get_metric(name: str) -> str:
    """Return the latest value of a named metric (mocked for tests)."""
    _RECORDER.record("get_metric", name=name)
    metrics = {
        "cpu": "cpu=87% (warn threshold 80%)",
        "latency_p99": "latency_p99=420ms (slo 300ms — breach)",
        "errors_5xx": "errors_5xx=0.4% (within 1% budget)",
    }
    return metrics.get(name.lower(), f"No metric named {name!r}.")


@tool
def list_alerts(window_minutes: int = 30) -> str:
    """List recent alerts in the given window (mocked)."""
    _RECORDER.record("list_alerts", window_minutes=window_minutes)
    return (
        "alert_id=A-101 sev=high svc=checkout latency_p99 breach\n"
        "alert_id=A-102 sev=medium svc=catalog cpu_warn"
    )


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def registered_protocol_ids() -> set[str]:
    return {p.id for p in builtin_protocols()}


@pytest.fixture
def router_under_test(model) -> Router:
    """Standard Router config used across every scenario."""
    _RECORDER.reset()

    tools = create_registry(kb_search, get_metric, list_alerts)
    capabilities = CapabilityIndex(tools)
    capabilities.annotate(
        "kb_search",
        tool_name="kb_search",
        description="Look up a topic in the knowledge base.",
        domain="research",
    )
    capabilities.annotate(
        "metric_probe",
        tool_name="get_metric",
        description="Return the latest value of a named metric.",
        domain="observability",
    )
    capabilities.annotate(
        "alert_list",
        tool_name="list_alerts",
        description="List recent alerts in a time window.",
        domain="observability",
    )

    protocols = ProtocolRegistry()
    protocols.register_many(builtin_protocols())

    extractor = Agent(
        model=model,
        system_prompt=(
            "You are a goal-frame extractor for the cognitive router. Given the "
            "user's request, fill the provided schema.\n\n"
            "Rules:\n"
            "1. Choose a single primary_goal that best matches the verb in "
            "the request.\n"
            "2. Risk:\n"
            "   - LOW   for read-only / informational tasks\n"
            "   - MEDIUM for build / modify / plan tasks (default for most things)\n"
            "   - HIGH  only for irreversible operations on production "
            "(deletes, migrations actually being executed)\n"
            "   Writing a plan or generating code is NOT high risk — the "
            "execution is.\n"
            "3. Complexity: LOW for one-step tasks, MEDIUM for multi-step, "
            "HIGH for tasks needing fan-out across multiple specialists.\n"
            "4. required_capabilities must come from this exact set; do NOT "
            "invent new ids:\n"
            "     - kb_search   (knowledge-base lookup)\n"
            "     - metric_probe (latest value of a named metric)\n"
            "     - alert_list  (recent alerts)\n"
            "If the request needs a tool that isn't in this list, leave "
            "required_capabilities empty."
        ),
        output_schema=GoalFrame,
    )
    compiler = CognitiveCompiler(
        protocols=protocols,
        capabilities=capabilities,
        policy=PolicyGate(),
        model=model,
    )
    return Router(extractor=extractor, compiler=compiler)


# ---------------------------------------------------------------------------
# Scenario A — bounded routing.
# ---------------------------------------------------------------------------


class TestBoundedRouting:
    """Every dispatch lands on one of the registered protocol ids — period."""

    @pytest.mark.asyncio
    async def test_every_dispatch_uses_a_registered_protocol(
        self,
        router_under_test: Router,
        registered_protocol_ids: set[str],
    ) -> None:
        observed: list[str] = []
        constrained_failures = 0
        for prompt, _expected_goal, _expected_protocol in LABELLED_CORPUS:
            try:
                frame = await router_under_test.extract(prompt)
                runnable = await router_under_test.compiler.compile(frame)
            except (FrameExtractionError, NoMatchingProtocolError):
                # Both are *constrained* outcomes: the LLM didn't author
                # arbitrary topology, the system refused. That's exactly
                # the boundedness guarantee we want.
                constrained_failures += 1
                continue
            protocol_id = _runnable_protocol_id(runnable)
            assert protocol_id in registered_protocol_ids, (
                f"prompt={prompt!r} produced unknown protocol_id={protocol_id!r}; "
                f"registered={sorted(registered_protocol_ids)}"
            )
            observed.append(protocol_id)
        # At least one prompt must have made it through — otherwise the
        # assertion is vacuous.
        assert observed, (
            f"every prompt produced a constrained failure ({constrained_failures} "
            f"of {len(LABELLED_CORPUS)}); cannot verify the success path"
        )


# ---------------------------------------------------------------------------
# Scenario B — routing accuracy on the labelled corpus.
# ---------------------------------------------------------------------------


class TestRoutingAccuracy:
    """The extractor + registry pick the expected goal/protocol most of the time."""

    @pytest.mark.asyncio
    async def test_corpus_accuracy(
        self,
        router_under_test: Router,
    ) -> None:
        goal_hits = 0
        protocol_hits = 0
        rows: list[str] = []
        total = len(LABELLED_CORPUS)

        for prompt, expected_goal, expected_protocol in LABELLED_CORPUS:
            try:
                frame = await router_under_test.extract(prompt)
            except FrameExtractionError as exc:
                rows.append(f"  EXTRACT-FAIL  {prompt[:60]!r}  {exc}")
                continue
            runnable = await router_under_test.compiler.compile(frame)
            protocol_id = _runnable_protocol_id(runnable)
            goal_ok = frame.primary_goal == expected_goal
            protocol_ok = protocol_id == expected_protocol
            goal_hits += int(goal_ok)
            protocol_hits += int(protocol_ok)
            tag_g = "✓" if goal_ok else "✗"
            tag_p = "✓" if protocol_ok else "✗"
            rows.append(
                f"  goal {tag_g} ({frame.primary_goal.value}/"
                f"{expected_goal.value})  protocol {tag_p} "
                f"({protocol_id}/{expected_protocol})  {prompt[:60]!r}",
            )

        report = "\n".join(["routing audit:", *rows])
        goal_acc = goal_hits / total
        protocol_acc = protocol_hits / total
        assert goal_acc >= ROUTING_ACCURACY_FLOOR_GOAL, (
            f"goal accuracy {goal_acc:.2f} below floor {ROUTING_ACCURACY_FLOOR_GOAL:.2f}\n{report}"
        )
        assert protocol_acc >= ROUTING_ACCURACY_FLOOR_PROTOCOL, (
            f"protocol accuracy {protocol_acc:.2f} below floor "
            f"{ROUTING_ACCURACY_FLOOR_PROTOCOL:.2f}\n{report}"
        )


# ---------------------------------------------------------------------------
# Scenario C — each v1 protocol actually runs end-to-end.
# ---------------------------------------------------------------------------


# One representative prompt per v1 protocol. The `frame_override` lets
# us bypass the extractor in case it picks a different goal — we want
# this scenario to verify the *builder + adapter*, not the extractor.
EXECUTION_FIXTURES: tuple[tuple[str, str, GoalFrame], ...] = (
    (
        "direct_response",
        "What does the locus.router module do in the locus SDK?",
        GoalFrame(
            primary_goal=TaskType.ANSWER,
            domain="research",
            complexity=Complexity.LOW,
            risk=Risk.LOW,
        ),
    ),
    (
        "plan_execute_validate",
        "Write a three-step plan to add a new health endpoint to our API, "
        "with the third step being a validation that returns 200.",
        GoalFrame(
            primary_goal=TaskType.PLAN,
            domain="engineering",
            complexity=Complexity.MEDIUM,
            risk=Risk.LOW,
            success_criteria=["plan has 3 numbered steps", "step 3 is validation"],
        ),
    ),
    (
        "specialist_fanout",
        "Diagnose checkout: pull recent alerts and the latency_p99 metric, then correlate.",
        GoalFrame(
            primary_goal=TaskType.DIAGNOSE,
            domain="observability",
            complexity=Complexity.HIGH,
            risk=Risk.MEDIUM,
            required_capabilities=["metric_probe", "alert_list"],
        ),
    ),
)


class TestProtocolExecution:
    """Each v1 protocol runs end-to-end and produces a normalized result."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("protocol_id", "prompt", "frame"),
        EXECUTION_FIXTURES,
        ids=[fx[0] for fx in EXECUTION_FIXTURES],
    )
    async def test_protocol_runs_end_to_end(
        self,
        router_under_test: Router,
        protocol_id: str,
        prompt: str,
        frame: GoalFrame,
    ) -> None:
        # Skip the extractor step — Scenario B already asserts that path.
        # Here we want a deterministic compile-and-execute on the chosen
        # protocol.
        runnable = await router_under_test.compiler.compile(frame)
        observed_id = _runnable_protocol_id(runnable)
        assert observed_id == protocol_id, (
            f"compiler picked {observed_id!r}, fixture asked for {protocol_id!r}"
        )

        result = await runnable.execute(prompt)

        assert isinstance(result, RunnableResult), f"got {type(result).__name__}"
        assert result.protocol_id == protocol_id
        assert result.frame == frame
        assert result.text.strip(), f"empty result text for protocol {protocol_id!r}"


# ---------------------------------------------------------------------------
# Scenario D — capability binding actually fires real tools.
# ---------------------------------------------------------------------------


class TestCapabilityBinding:
    """When a frame requires a tool-backed capability, the underlying Tool
    actually gets invoked during execution."""

    @pytest.mark.asyncio
    async def test_specialist_fanout_invokes_resolved_tools(
        self,
        router_under_test: Router,
    ) -> None:
        _RECORDER.reset()
        frame = GoalFrame(
            primary_goal=TaskType.DIAGNOSE,
            domain="observability",
            complexity=Complexity.HIGH,
            risk=Risk.MEDIUM,
            required_capabilities=["metric_probe", "alert_list"],
        )
        runnable = await router_under_test.compiler.compile(frame)
        # specialist_fanout uses a ParallelPipeline of tool-bound Agents
        # (Specialist's single-turn execute can't process tool calls).
        assert isinstance(runnable, PipelineRunnable), (
            f"DIAGNOSE should fire specialist_fanout (PipelineRunnable); "
            f"got {type(runnable).__name__}"
        )

        prompt = (
            "We're seeing checkout slow down. Pull the latest latency_p99 metric "
            "and the list of active alerts, then correlate them."
        )
        await runnable.execute(prompt)

        invoked = {name for name, _ in _RECORDER.calls}
        assert invoked, (
            "no tools were invoked; the orchestrator either skipped specialists "
            "or the specialists never called their bound tools. recorder=[]"
        )
        # We don't insist both tools fire — model behaviour varies — but at
        # least one of the two bound capabilities must have been used.
        assert invoked & {"get_metric", "list_alerts"}, (
            f"expected at least one of get_metric / list_alerts to fire; "
            f"observed tools: {sorted(invoked)}"
        )


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _runnable_protocol_id(runnable: Any) -> str:
    """Pull the protocol_id off any adapter shape — including the
    follow-up adapters (DebateRunnable, A2ARunnable) that were added
    when the catalogue grew from 3 to 8 protocols."""
    # Every concrete adapter exposes ``.protocol_id`` directly. The
    # _ApprovalRunnable wrapper exposes its wrapped runnable as
    # ``.inner`` and delegates.
    direct = getattr(runnable, "protocol_id", None)
    if direct is not None:
        return direct
    inner = getattr(runnable, "inner", None)
    if inner is not None:
        return _runnable_protocol_id(inner)
    raise AssertionError(f"Cannot extract protocol_id from {type(runnable).__name__}")


# ---------------------------------------------------------------------------
# Scenario E — emergent picker (opt-in LLM protocol picker)
# ---------------------------------------------------------------------------


@pytest.fixture
def router_with_picker(model) -> Router:
    """Variant of ``router_under_test`` with the LLM picker enabled."""
    from locus.router import LLMProtocolPicker

    _RECORDER.reset()
    tools = create_registry(kb_search, get_metric, list_alerts)
    capabilities = CapabilityIndex(tools)
    capabilities.annotate(
        "kb_search",
        tool_name="kb_search",
        description="Look up a topic in the knowledge base.",
        domain="research",
    )
    capabilities.annotate(
        "metric_probe",
        tool_name="get_metric",
        description="Return the latest value of a named metric.",
        domain="observability",
    )
    capabilities.annotate(
        "alert_list",
        tool_name="list_alerts",
        description="List recent alerts in a time window.",
        domain="observability",
    )

    protocols = ProtocolRegistry()
    protocols.register_many(builtin_protocols())

    extractor = Agent(
        model=model,
        system_prompt=(
            "Fill the GoalFrame schema based on the user's verb. "
            "required_capabilities can include: kb_search, metric_probe, alert_list."
        ),
        output_schema=GoalFrame,
    )
    compiler = CognitiveCompiler(
        protocols=protocols,
        capabilities=capabilities,
        policy=PolicyGate(),
        model=model,
        protocol_picker=LLMProtocolPicker(model=model),
    )
    return Router(extractor=extractor, compiler=compiler)


class TestEmergentPicker:
    """The opt-in LLM-driven protocol picker — live wire.

    Asserts the picker dispatches to a registered protocol on an
    ambiguous prompt, and that the resulting protocol_id is part of
    the built-in catalogue (no hallucinations escape the membership
    check + fallback).
    """

    @pytest.mark.asyncio
    async def test_picker_routes_compare_prompt_to_valid_protocol(
        self,
        router_with_picker: Router,
        registered_protocol_ids: set[str],
    ) -> None:
        # COMPARE leaves both ``debate`` and ``specialist_fanout`` in
        # the candidate set — this is exactly where the picker earns
        # its keep.
        result = await router_with_picker.dispatch(
            "Compare swarm vs orchestrator patterns for open-ended research.",
        )
        assert result.protocol_id in registered_protocol_ids, (
            f"picker produced unknown protocol_id={result.protocol_id!r}; "
            f"expected one of {sorted(registered_protocol_ids)}"
        )
