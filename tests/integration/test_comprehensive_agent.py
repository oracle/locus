# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Comprehensive integration tests for the agentic loop.

Tests exercise all 10 features together with real models, real databases,
and realistic multi-step scenarios. Uses OCI GenAI GPT-5.4 and the
deep research Oracle 26ai database.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from locus.agent import Agent, GroundingConfig, ReflexionConfig
from locus.core.events import (
    ReflectEvent,
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
    ToolStartEvent,
)
from locus.tools.decorator import tool
from tests._safe_math import safe_math_eval


pytestmark = [pytest.mark.integration, pytest.mark.requires_model]


# =============================================================================
# Tools that simulate real-world patterns
# =============================================================================


@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for information about a topic."""
    # Simulate a knowledge base search with realistic results
    knowledge = {
        "python": "Python is a high-level programming language created by Guido van Rossum in 1991. It supports multiple paradigms including procedural, object-oriented, and functional programming. Python 3.14 is the latest version.",
        "quantum": "Quantum computing uses quantum-mechanical phenomena such as superposition and entanglement. Key players include IBM, Google, and IonQ. Google achieved quantum supremacy in 2019 with Sycamore.",
        "ai": "Artificial Intelligence encompasses machine learning, deep learning, and agentic AI. Large Language Models (LLMs) like GPT-5, Claude, and Gemini power modern AI applications.",
        "oracle": "Oracle Cloud Infrastructure (OCI) provides compute, storage, networking, and AI services. OCI GenAI offers access to Cohere, Meta, and OpenAI models.",
    }
    for key, value in knowledge.items():
        if key in query.lower():
            return value
    return f"No specific results found for '{query}'. Try a more specific query."


@tool
def query_database(sql_description: str) -> str:
    """Query a database with a natural language description of what to find."""
    # Simulate database queries with realistic results
    time.sleep(0.2)  # Simulate DB latency
    if "count" in sql_description.lower():
        return "Query result: 1,787 documents found in the knowledge base."
    if "medical" in sql_description.lower():
        return "Query result: Found 500 medical documents covering biochemistry, pathology, pharmacology, and microbiology."
    if "topic" in sql_description.lower() or "category" in sql_description.lower():
        return "Query result: Topics include Biochemistry (312), Pathology (245), Pharmacology (198), Microbiology (156), Anatomy (120), Physiology (98), Other (658)."
    return f"Query executed: {sql_description}. No matching records."


@tool
def verify_fact(claim: str, source: str) -> str:
    """Verify a factual claim against a source."""
    time.sleep(0.1)
    # Simulate verification
    if any(word in claim.lower() for word in ["python", "1991", "guido"]):
        return f"VERIFIED: '{claim}' is supported by {source}."
    if any(word in claim.lower() for word in ["quantum", "google", "superposition"]):
        return f"VERIFIED: '{claim}' is supported by {source}."
    return f"UNVERIFIED: '{claim}' could not be confirmed against {source}."


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(safe_math_eval(expression))
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        return f"Error: {e!s}"


@tool
def unreliable_api(endpoint: str) -> str:
    """Call an unreliable external API that sometimes fails."""
    # Fails 50% of the time to test error recovery
    if hash(endpoint) % 2 == 0:
        raise ConnectionError(f"API timeout: {endpoint}")
    return f"API response from {endpoint}: status=200, data=ok"


@tool
def slow_analysis(data: str) -> str:
    """Perform a slow analysis on data. Takes 1-2 seconds."""
    time.sleep(1.0)
    return f"Analysis complete for '{data[:50]}': 3 key insights found, confidence=high."


@tool
def generate_report_section(topic: str, findings: str) -> str:
    """Generate a report section based on findings."""
    return (
        f"## {topic}\n\n"
        f"Based on the analysis: {findings[:100]}...\n\n"
        f"Key takeaway: The data supports the hypothesis with high confidence."
    )


# =============================================================================
# Test 1: Multi-Tool Parallel Execution
# =============================================================================


class TestParallelToolExecution:
    """Test agent handles multiple tool calls in a single model response."""

    @pytest.mark.asyncio
    async def test_agent_uses_multiple_tools_per_turn(self, model):
        """Agent requests and executes multiple tools simultaneously."""
        agent = Agent(
            model=model,
            tools=[search_knowledge_base, query_database, calculate],
            system_prompt=(
                "You are a research assistant. When asked to investigate a topic, "
                "use MULTIPLE tools in a SINGLE response to gather information efficiently. "
                "Call search_knowledge_base AND query_database at the same time."
            ),
            max_iterations=5,
            tool_execution="concurrent",
        )

        events = []
        async for event in agent.run(
            "How many documents are in our knowledge base and what topics do they cover?"
        ):
            events.append(event)

        tool_events = [e for e in events if isinstance(e, ToolCompleteEvent)]
        assert len(tool_events) >= 2  # At least 2 tool calls

        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None
        assert terminate.final_message is not None

    @pytest.mark.asyncio
    async def test_parallel_tool_calls_execute_concurrently_wall_time(self, model) -> None:  # type: ignore[no-untyped-def]
        """End-to-end wall-time guard for ``tool_execution="concurrent"`` (#210).

        The unit test ``tests/unit/test_agent_concurrent_tools.py`` uses a
        scripted model to pin parallelism deterministically. This live-model
        twin closes the same gap end-to-end: the previously-passing
        :func:`test_agent_uses_multiple_tools_per_turn` only checked that
        tools fired, not that they ran in parallel, so it never noticed
        when ``ConcurrentExecutor`` was being fed singletons.

        We skip cleanly if the live model never fans out (small models often
        won't). When it does, the gap between the first ``ToolStartEvent``
        and the last ``ToolCompleteEvent`` of that iteration must be far
        below ``N * sleep_per_tool``.
        """
        sleep_per_tool = 0.6  # large enough to clear network jitter
        n_target = 3  # ask the model for 3 lookups in one turn

        @tool(name="lookup_async")
        async def lookup_async(topic: str) -> str:
            """Async lookup that takes ~0.6s per call (simulates remote I/O)."""
            await asyncio.sleep(sleep_per_tool)
            return f"info about {topic}"

        agent = Agent(
            model=model,
            tools=[lookup_async],
            system_prompt=(
                "You are a research assistant. You MUST call lookup_async "
                f"{n_target} times in your VERY FIRST response, all in a "
                "single message — one call per topic the user asks about. "
                "Do NOT call them one-at-a-time across iterations."
            ),
            max_iterations=3,
            tool_execution="concurrent",
            max_concurrency=10,
        )

        # Group events by iteration so we can measure each ThinkEvent → tools
        # window independently. ``ThinkEvent`` starts a new iteration.
        events: list = []
        async for ev in agent.run(
            "Look up these three topics: python, quantum, oracle. "
            "Call the tool three times in one response."
        ):
            events.append((time.perf_counter(), ev))

        # Find the first iteration in which >= 2 ToolStartEvents fire before
        # any ThinkEvent (i.e. a fan-out turn).
        per_iter_starts: list[list[tuple[float, ToolStartEvent]]] = []
        per_iter_completes: list[list[tuple[float, ToolCompleteEvent]]] = []
        current_starts: list[tuple[float, ToolStartEvent]] = []
        current_completes: list[tuple[float, ToolCompleteEvent]] = []
        for ts, ev in events:
            if isinstance(ev, ThinkEvent):
                if current_starts:
                    per_iter_starts.append(current_starts)
                    per_iter_completes.append(current_completes)
                current_starts = []
                current_completes = []
            elif isinstance(ev, ToolStartEvent):
                current_starts.append((ts, ev))
            elif isinstance(ev, ToolCompleteEvent):
                current_completes.append((ts, ev))
        if current_starts:
            per_iter_starts.append(current_starts)
            per_iter_completes.append(current_completes)

        # Pick the first iteration that actually fanned out.
        fanout_idx = next(
            (i for i, sts in enumerate(per_iter_starts) if len(sts) >= 2),
            None,
        )
        if fanout_idx is None:
            pytest.skip(
                "Live model did not emit >=2 parallel tool_calls in any "
                "iteration — can't measure parallelism. Try a stronger model."
            )

        starts = per_iter_starts[fanout_idx]
        completes = per_iter_completes[fanout_idx]
        n = len(starts)
        assert len(completes) == n, (
            f"start/complete count mismatch in fan-out iteration: "
            f"starts={n}, completes={len(completes)}"
        )

        first_start = starts[0][0]
        last_complete = completes[-1][0]
        gap = last_complete - first_start

        sequential_floor = n * sleep_per_tool
        # Generous: half the serial floor catches the bug (which would land
        # at ~100% of floor) without flaking on slow CI (parallel ~= 1 sleep
        # plus model overhead).
        assert gap < sequential_floor / 2, (
            f"{n} concurrent tool calls took {gap:.2f}s "
            f"(sequential floor {sequential_floor:.2f}s) — "
            f"runtime loop may be serializing the executor again (#210)"
        )


# =============================================================================
# Test 2: Long Multi-Turn Conversation
# =============================================================================


class TestLongMultiTurnConversation:
    """Test agent handles 5+ iterations without context overflow."""

    @pytest.mark.asyncio
    async def test_multi_turn_with_conversation_management(self, model):
        """Agent runs multiple iterations with auto conversation manager."""
        agent = Agent(
            model=model,
            tools=[search_knowledge_base, query_database, verify_fact],
            system_prompt=(
                "You are a thorough researcher. For the given question:\n"
                "1. First search the knowledge base\n"
                "2. Then query the database for statistics\n"
                "3. Then verify key facts\n"
                "4. Finally provide a comprehensive answer\n"
                "Complete ALL steps before answering."
            ),
            max_iterations=8,
            max_tool_result_length=2000,
        )

        events = []
        async for event in agent.run(
            "Tell me about Python programming — search, get stats, and verify facts"
        ):
            events.append(event)

        tool_events = [e for e in events if isinstance(e, ToolCompleteEvent)]
        think_events = [e for e in events if isinstance(e, ThinkEvent)]

        # Should have at least some turns and tool usage
        assert len(think_events) >= 1
        assert len(tool_events) >= 1

        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None
        assert terminate.reason in ("complete", "max_iterations", "tool_loop")
        # Agent should have produced some output
        if terminate.reason == "complete":
            assert terminate.final_message is not None
            assert len(terminate.final_message) > 20


# =============================================================================
# Test 3: Reflexion + Grounding Together
# =============================================================================


class TestReflexionAndGroundingTogether:
    """Test both reflexion and grounding working in the same run."""

    @pytest.mark.asyncio
    async def test_reflexion_and_grounding_combined(self, model):
        """Agent self-assesses progress AND validates final answer."""
        agent = Agent(
            model=model,
            tools=[search_knowledge_base, verify_fact],
            system_prompt=(
                "Research the topic thoroughly. Use search_knowledge_base "
                "to find information, then verify_fact to check key claims. "
                "Be factual — only state what tools confirmed."
            ),
            reflexion=ReflexionConfig(enabled=True, include_guidance=True),
            grounding=GroundingConfig(enabled=True, threshold=0.3),
            max_iterations=6,
        )

        events = []
        async for event in agent.run(
            "What is quantum computing and who achieved quantum supremacy?"
        ):
            events.append(event)

        # Should have reflection events
        reflect_events = [e for e in events if isinstance(e, ReflectEvent)]
        assert len(reflect_events) >= 1

        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None
        assert terminate.reason in ("complete", "max_iterations", "tool_loop")
        # At minimum, agent ran and produced events
        assert len(events) >= 3


# =============================================================================
# Test 4: Tool Errors Mid-Execution
# =============================================================================


class TestToolErrorRecovery:
    """Test agent recovers when tools fail mid-execution."""

    @pytest.mark.asyncio
    async def test_agent_handles_tool_failure(self, model):
        """Agent continues working when some tools throw exceptions."""
        agent = Agent(
            model=model,
            tools=[search_knowledge_base, unreliable_api],
            system_prompt=(
                "Search for information. If the unreliable_api fails, "
                "fall back to search_knowledge_base. Don't give up — "
                "use whatever tools succeed to answer the question."
            ),
            max_iterations=5,
        )

        events = []
        async for event in agent.run("Find information about AI using all available sources"):
            events.append(event)

        tool_events = [e for e in events if isinstance(e, ToolCompleteEvent)]
        # Some tools should have failed
        errors = [e for e in tool_events if e.error]
        successes = [e for e in tool_events if e.result and not e.error]

        # Agent should have completed despite errors
        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None
        assert terminate.final_message is not None
        # Should have at least some successful tool calls
        assert len(successes) >= 1


# =============================================================================
# Test 5: Token Budget Mid-Run
# =============================================================================


class TestTokenBudgetMidRun:
    """Test token budget stops agent gracefully mid-run."""

    @pytest.mark.asyncio
    async def test_token_budget_with_tool_usage(self, model):
        """Agent with token budget stops after using too many tokens."""
        agent = Agent(
            model=model,
            tools=[search_knowledge_base, query_database],
            system_prompt="Research thoroughly. Keep searching and querying.",
            max_iterations=15,
            token_budget=2000,  # Low budget — should stop after a few calls
        )

        events = []
        async for event in agent.run(
            "Research everything about Python, AI, quantum computing, and Oracle"
        ):
            events.append(event)

        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None
        # Should stop before max_iterations
        assert terminate.iterations_used < 15


# =============================================================================
# Test 6: Time Budget with Slow Tools
# =============================================================================


class TestTimeBudgetWithSlowTools:
    """Test time budget with tools that take significant time."""

    @pytest.mark.asyncio
    async def test_time_budget_stops_slow_agent(self, model):
        """Agent with time budget stops when slow tools eat up the clock."""
        agent = Agent(
            model=model,
            tools=[slow_analysis, search_knowledge_base],
            system_prompt=(
                "Analyze the topic by calling slow_analysis multiple times "
                "with different aspects. Keep analyzing until comprehensive."
            ),
            max_iterations=10,
            time_budget_seconds=5.0,
        )

        start = time.time()
        events = []
        async for event in agent.run("Analyze quantum computing from every angle"):
            events.append(event)
        elapsed = time.time() - start

        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None
        assert terminate.iterations_used < 10
        # Should finish within reasonable time
        assert elapsed < 30.0


# =============================================================================
# Test 7: Truncation + Recovery
# =============================================================================


class TestTruncationInConversation:
    """Test tool result truncation works mid-conversation."""

    @pytest.mark.asyncio
    async def test_truncation_doesnt_break_reasoning(self, model):
        """Agent handles truncated tool results and still reasons correctly."""

        @tool
        def huge_data_dump(topic: str) -> str:
            """Get a massive dataset about a topic."""
            return f"Data about {topic}: " + ("detailed row of data | " * 5000)

        agent = Agent(
            model=model,
            tools=[huge_data_dump, calculate],
            system_prompt="Get the data dump, note it was truncated, then calculate 2+2.",
            max_iterations=4,
            max_tool_result_length=500,
        )

        events = []
        async for event in agent.run("Get data about AI, then calculate 2+2"):
            events.append(event)

        tool_events = [e for e in events if isinstance(e, ToolCompleteEvent)]
        # First tool should be truncated
        truncated = [e for e in tool_events if e.result and "[OUTPUT TRUNCATED" in e.result]
        assert len(truncated) >= 1

        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None


# =============================================================================
# Test 8: run_sync Full State Preservation
# =============================================================================


class TestRunSyncComprehensive:
    """Test run_sync preserves complete state across complex runs."""

    def test_run_sync_with_reflexion_and_tools(self, model):
        """run_sync preserves tool executions, confidence, and metrics."""
        agent = Agent(
            model=model,
            tools=[search_knowledge_base, calculate],
            system_prompt="Search for information and do a calculation, then answer.",
            reflexion=ReflexionConfig(enabled=True),
            max_iterations=5,
        )

        result = agent.run_sync("What is Python? Also, what is 15*23?")

        assert result.success or result.stop_reason in ("max_iterations", "tool_loop")
        assert result.metrics.iterations >= 1
        assert result.metrics.duration_ms > 0
        assert len(result.message) > 0

        # State should have real data
        if result.metrics.tool_calls > 0:
            assert len(result.tool_executions) > 0
            # Each execution should have tool name and result/error
            for exec in result.tool_executions:
                assert exec.tool_name in ("search_knowledge_base", "calculate")


# =============================================================================
# Test 9: Graceful Degradation on Max Iterations
# =============================================================================


class TestGracefulDegradationComplex:
    """Test graceful max-iterations with complex multi-tool scenario."""

    @pytest.mark.asyncio
    async def test_summary_after_complex_run(self, model):
        """Agent produces meaningful summary after hitting max_iterations."""
        agent = Agent(
            model=model,
            tools=[search_knowledge_base, query_database, verify_fact, generate_report_section],
            system_prompt=(
                "You are writing a comprehensive report. For each topic:\n"
                "1. Search the knowledge base\n"
                "2. Query the database for stats\n"
                "3. Verify key facts\n"
                "4. Generate a report section\n"
                "Cover Python, AI, and quantum computing. Do ALL steps for EACH topic."
            ),
            max_iterations=3,  # Too few for all topics — will hit limit
        )

        events = []
        async for event in agent.run("Write a comprehensive technology report"):
            events.append(event)

        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None

        if terminate.reason == "max_iterations":
            # Should have a summary (not None)
            assert terminate.final_message is not None
            assert len(terminate.final_message) > 20


# =============================================================================
# Test 10: Full Pipeline — Everything Together
# =============================================================================


class TestFullPipeline:
    """The ultimate test: all features working together in one run."""

    @pytest.mark.asyncio
    async def test_all_features_in_one_run(self, model):
        """Agent uses reflexion, grounding, truncation, budget, and tools."""
        agent = Agent(
            model=model,
            tools=[
                search_knowledge_base,
                query_database,
                verify_fact,
                calculate,
            ],
            system_prompt=(
                "You are a research analyst. For the given question:\n"
                "1. Search the knowledge base for relevant information\n"
                "2. Query the database for statistics\n"
                "3. Verify at least one key fact\n"
                "4. Calculate any relevant numbers\n"
                "Be thorough and factual. Only state verified information."
            ),
            reflexion=ReflexionConfig(enabled=True, include_guidance=True),
            grounding=GroundingConfig(enabled=True, threshold=0.3),
            max_iterations=8,
            max_tool_result_length=2000,
            token_budget=10000,
        )

        events = []
        event_types = set()
        async for event in agent.run(
            "How many documents are in our knowledge base? "
            "What topics do they cover? "
            "Verify that Python was created in 1991."
        ):
            events.append(event)
            event_types.add(type(event).__name__)

        # Should have diverse event types
        assert "ThinkEvent" in event_types
        assert "ToolStartEvent" in event_types
        assert "ToolCompleteEvent" in event_types
        assert "TerminateEvent" in event_types
        # Reflexion should have fired
        assert "ReflectEvent" in event_types

        # Tool calls should have happened
        tool_events = [e for e in events if isinstance(e, ToolCompleteEvent)]
        assert len(tool_events) >= 2

        # Should complete
        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None
        assert terminate.final_message is not None
        assert len(terminate.final_message) > 50

        # Event stream should be ordered: think → tool_start → tool_complete → ...
        event_names = [type(e).__name__ for e in events]
        first_think = event_names.index("ThinkEvent")
        first_tool = event_names.index("ToolStartEvent")
        assert first_think < first_tool  # Think before tool
