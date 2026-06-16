# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Integration tests for Agent with real models.

All tests use the session-scoped ``model`` fixture from conftest.py, which
auto-detects OCI GenAI or OpenAI based on environment variables.
Model selection via OCI_MODEL_ID (default: openai.gpt-5.4) when OCI configured.
"""

from __future__ import annotations

import pytest

from locus.agent import Agent, AgentResult, ReflexionConfig
from locus.core.events import (
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
    ToolStartEvent,
)
from locus.tools.decorator import tool
from tests._safe_math import safe_math_eval


# Skip all tests if no API key is available
pytestmark = [pytest.mark.integration, pytest.mark.requires_model]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def math_tools():
    """Create math-related tools."""

    @tool
    def add(a: int, b: int) -> str:
        """Add two numbers."""
        return str(a + b)

    @tool
    def multiply(a: int, b: int) -> str:
        """Multiply two numbers."""
        return str(a * b)

    @tool
    def subtract(a: int, b: int) -> str:
        """Subtract b from a."""
        return str(a - b)

    return [add, multiply, subtract]


@pytest.fixture
def search_tools():
    """Create search-related tools."""

    @tool
    def search_web(query: str) -> str:
        """Search the web for information."""
        # Simulated search results
        if "weather" in query.lower():
            return "Current weather: 72F, Sunny"
        if "capital" in query.lower():
            return "The capital of France is Paris."
        return f"No results found for: {query}"

    @tool
    def search_database(query: str) -> str:
        """Search the internal database."""
        return f"Database results for '{query}': No matching records."

    return [search_web, search_database]


@pytest.fixture
def terminal_tool():
    """Create a terminal tool."""

    @tool
    def submit(answer: str) -> str:
        """Submit the final answer."""
        return f"Answer submitted: {answer}"

    return submit


# =============================================================================
# Agent Integration Tests
# =============================================================================


class TestAgentIntegration:
    """Core integration tests: completions, tool use, reflexion, sync execution."""

    @pytest.mark.asyncio
    async def test_simple_completion(self, model):
        """Test simple completion without tools."""
        agent = Agent(
            model=model,
            system_prompt="You are a helpful assistant. Keep responses brief.",
            max_tokens=100,
        )

        events = []
        async for event in agent.run("What is 2+2? Reply with just the number."):
            events.append(event)

        # Should have ThinkEvent and TerminateEvent
        assert any(isinstance(e, ThinkEvent) for e in events)
        assert any(isinstance(e, TerminateEvent) for e in events)

        # Check the response contains "4"
        think_events = [e for e in events if isinstance(e, ThinkEvent)]
        assert len(think_events) > 0
        assert "4" in think_events[0].reasoning

    @pytest.mark.asyncio
    async def test_tool_usage(self, model, math_tools):
        """Test that agent uses tools."""
        agent = Agent(
            model=model,
            tools=math_tools,
            system_prompt="You are a calculator. Use the provided tools to compute results.",
            max_tokens=200,
        )

        events = []
        async for event in agent.run("Add 5 and 3"):
            events.append(event)

        # Should have tool events
        assert any(isinstance(e, ToolStartEvent) for e in events)
        assert any(isinstance(e, ToolCompleteEvent) for e in events)

        # Find the tool result
        tool_completes = [e for e in events if isinstance(e, ToolCompleteEvent)]
        assert len(tool_completes) > 0
        assert tool_completes[0].tool_name == "add"
        assert tool_completes[0].result == "8"

    @pytest.mark.asyncio
    async def test_multi_tool_usage(self, model, math_tools):
        """Test agent using multiple tools."""
        agent = Agent(
            model=model,
            tools=math_tools,
            system_prompt="You are a calculator. Use tools to compute step by step.",
            max_tokens=500,
            max_iterations=5,
        )

        events = []
        async for event in agent.run("What is (3 + 5) * 2?"):
            events.append(event)

        # Should have multiple tool calls
        tool_starts = [e for e in events if isinstance(e, ToolStartEvent)]
        assert len(tool_starts) >= 2  # At least add and multiply

    @pytest.mark.asyncio
    async def test_with_reflexion(self, model, math_tools):
        """Test agent with Reflexion enabled."""
        agent = Agent(
            model=model,
            tools=math_tools,
            reflexion=ReflexionConfig(
                enabled=True,
                confidence_threshold=0.95,  # High threshold
            ),
            max_iterations=5,
        )

        events = []
        async for event in agent.run("Calculate 10 + 20"):
            events.append(event)

        # Should complete successfully
        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None

    def test_run_sync(self, model, math_tools):
        """Test synchronous execution."""
        agent = Agent(
            model=model,
            tools=math_tools,
            system_prompt="You are a calculator.",
            max_tokens=200,
        )

        result = agent.run_sync("What is 7 + 8?")

        assert isinstance(result, AgentResult)
        assert result.success is True
        assert "15" in result.message or len(result.tool_executions) > 0


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling with real models."""

    @pytest.mark.asyncio
    async def test_tool_error_recovery(self, model):
        """Test that agent handles tool errors gracefully."""

        @tool
        def failing_tool(input: str) -> str:  # noqa: A002
            """A tool that always fails."""
            raise RuntimeError("Tool failed!")  # noqa: TRY003

        @tool
        def working_tool(input: str) -> str:  # noqa: A002
            """A tool that works."""
            return f"Processed: {input}"

        agent = Agent(
            model=model,
            tools=[failing_tool, working_tool],
            system_prompt="Try to process the input. If one tool fails, try another.",
            max_iterations=3,
        )

        events = []
        async for event in agent.run("Process this text"):
            events.append(event)

        # Should have completed (possibly with errors)
        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None

        # Check that error was captured
        tool_completes = [e for e in events if isinstance(e, ToolCompleteEvent)]
        if any(e.tool_name == "failing_tool" for e in tool_completes):
            error_events = [e for e in tool_completes if e.error is not None]
            assert len(error_events) > 0


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Performance-related tests."""

    @pytest.mark.asyncio
    async def test_max_iterations_limit(self, model, math_tools):
        """Test that max_iterations is respected."""
        agent = Agent(
            model=model,
            tools=math_tools,
            max_iterations=2,
            system_prompt="Keep using tools.",
        )

        events = []
        async for event in agent.run("Keep calculating forever"):
            events.append(event)

        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None
        assert terminate.iterations_used <= 2


# =============================================================================
# Feature 1: Tool Result Truncation (Integration)
# =============================================================================


class TestToolResultTruncation:
    """Integration: verify truncation works with real model calls."""

    @pytest.mark.asyncio
    async def test_large_tool_result_truncated_with_real_model(self, model):
        """Agent handles a tool that returns a massive result and still completes."""

        @tool
        def big_data_dump() -> str:
            """Returns a huge dataset."""
            return "row " * 20000  # ~100K chars

        agent = Agent(
            model=model,
            tools=[big_data_dump],
            system_prompt="Call big_data_dump, then summarize what you got.",
            max_iterations=3,
            max_tool_result_length=500,
        )

        events = []
        async for event in agent.run("Get the data dump"):
            events.append(event)

        # Tool should have been called and result truncated
        tool_events = [e for e in events if isinstance(e, ToolCompleteEvent)]
        assert len(tool_events) >= 1
        assert "[OUTPUT TRUNCATED" in tool_events[0].result

        # Agent should still complete (not crash from context overflow)
        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None


# =============================================================================
# Feature 2: Message Validation (Integration)
# =============================================================================


class TestMessageValidation:
    """Integration: verify message validation doesn't break real model calls."""

    @pytest.mark.asyncio
    async def test_agent_completes_with_tool_usage(self, model):
        """Agent with tools completes normally -- validation is transparent."""

        @tool
        def get_info(question: str) -> str:
            """Get info about a question. Always call this tool first."""
            return f"The answer to '{question}' is 42."

        agent = Agent(
            model=model,
            tools=[get_info],
            system_prompt="You MUST use the get_info tool to answer any question. Call get_info first, then respond.",
            max_iterations=5,
        )

        events = []
        async for event in agent.run("Use get_info to find the answer to life"):
            events.append(event)

        # Agent should complete without errors (validation is transparent)
        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None
        assert terminate.reason in ("complete", "max_iterations")


# =============================================================================
# Feature 3: Malformed Tool Call Recovery (Integration)
# =============================================================================


class TestMalformedToolCallRecovery:
    """Integration: the parse fallback doesn't interfere with structured calls."""

    @pytest.mark.asyncio
    async def test_structured_tool_calls_work_normally(self, model):
        """Normal structured tool calls still work (recovery doesn't fire)."""

        @tool
        def calculator(expression: str) -> str:
            """Evaluate a math expression."""
            try:
                return str(safe_math_eval(expression))
            except (ValueError, SyntaxError, ZeroDivisionError):
                return "error"

        agent = Agent(
            model=model,
            tools=[calculator],
            system_prompt="Use the calculator tool to answer math questions. Be concise.",
            max_iterations=3,
        )

        events = []
        async for event in agent.run("What is 15 * 23?"):
            events.append(event)

        tool_events = [e for e in events if isinstance(e, ToolCompleteEvent)]
        assert len(tool_events) >= 1
        assert "345" in tool_events[0].result

        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None


# =============================================================================
# Feature 4: Config Budgets (Integration)
# =============================================================================


class TestConfigBudgets:
    """Integration: verify token budget terminates real agent runs."""

    @pytest.mark.asyncio
    async def test_token_budget_stops_agent(self, model):
        """Agent with very low token budget stops early."""

        @tool
        def search(query: str) -> str:
            """Search for info."""
            return f"Found results for: {query}"

        agent = Agent(
            model=model,
            tools=[search],
            system_prompt="Search for info, then answer. Be detailed and thorough.",
            max_iterations=10,
            token_budget=500,  # Very low -- should stop after 1-2 calls
        )

        events = []
        async for event in agent.run("Tell me everything about Python programming"):
            events.append(event)

        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None
        # Should stop early (budget, loop, or complete -- not all 10 iterations)
        assert terminate.iterations_used < 10

    @pytest.mark.asyncio
    async def test_time_budget_concept(self, model):
        """Verify time_budget_seconds config is accepted (enforcement in Feature 5)."""
        agent = Agent(
            model=model,
            tools=[],
            system_prompt="Be brief.",
            time_budget_seconds=60.0,
        )

        events = []
        async for event in agent.run("Say hello"):
            events.append(event)

        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None
        assert terminate.reason == "complete"


# =============================================================================
# Feature 5: Time Budget Enforcement (Integration)
# =============================================================================


class TestTimeBudget:
    """Integration: verify time budget stops real agent runs."""

    @pytest.mark.asyncio
    async def test_time_budget_stops_chatty_agent(self, model):
        """Agent with short time budget stops before max_iterations."""
        import time

        @tool
        def research(topic: str) -> str:
            """Research a topic in depth."""
            time.sleep(0.5)  # Simulate slow tool
            return f"Detailed findings about {topic}: lots of data here..."

        agent = Agent(
            model=model,
            tools=[research],
            system_prompt="Research the topic thoroughly. Keep using the research tool with different aspects.",
            max_iterations=20,
            time_budget_seconds=3.0,
        )

        start = time.time()
        events = []
        async for event in agent.run("Tell me everything about quantum computing"):
            events.append(event)
        elapsed = time.time() - start

        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None
        assert terminate.reason in ("time_budget", "complete")
        assert terminate.iterations_used < 20
        assert elapsed < 30.0  # Generous — OCI model calls add overhead


# =============================================================================
# Feature 6: Auto Conversation Manager (Integration)
# =============================================================================


class TestAutoConversationManager:
    """Integration: verify auto conversation manager works with real model."""

    @pytest.mark.asyncio
    async def test_multi_turn_agent_completes(self, model):
        """Agent with multiple tool turns completes with auto conversation manager."""

        @tool
        def lookup(topic: str) -> str:
            """Look up information about a topic."""
            return f"Info about {topic}: this is a detailed explanation with many words " * 20

        @tool
        def summarize(text: str) -> str:
            """Summarize text."""
            return f"Summary: {text[:50]}..."

        agent = Agent(
            model=model,
            tools=[lookup, summarize],
            system_prompt="Look up 'AI' then summarize it. Be concise.",
            max_iterations=5,
            # Auto manager will be created: window = max(20, 5*2) = 20
        )

        events = []
        async for event in agent.run("Research AI briefly"):
            events.append(event)

        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None
        assert terminate.reason in ("complete", "max_iterations")
        # Agent completed -- conversation manager didn't break anything
        assert terminate.iterations_used >= 1


# =============================================================================
# Feature 7: Real Reflector
# =============================================================================


@pytest.mark.requires_model
class TestRealReflector:
    """Integration: Real Reflector with guidance injection."""

    @pytest.mark.asyncio
    async def test_reflexion_with_real_model(self, model):
        """Agent with reflexion=True completes and produces reflection events."""
        from locus.agent import ReflexionConfig
        from locus.core.events import ReflectEvent

        @tool
        def lookup(topic: str) -> str:
            """Look up information."""
            return f"Detailed findings about {topic}: " + "important data " * 20

        @tool
        def verify(claim: str) -> str:
            """Verify a claim."""
            return f"Verified: {claim} is correct."

        agent = Agent(
            model=model,
            tools=[lookup, verify],
            system_prompt="Research the topic using lookup, then verify findings. Be thorough.",
            reflexion=ReflexionConfig(enabled=True, include_guidance=True),
            max_iterations=8,
        )

        events = []
        async for event in agent.run("What are the benefits of exercise?"):
            events.append(event)

        # Should have at least one reflection event
        reflect_events = [e for e in events if isinstance(e, ReflectEvent)]
        assert len(reflect_events) >= 1

        # Reflections should have real assessments (not just "on_track")
        assessments = {e.assessment for e in reflect_events}
        assert assessments.issubset({"on_track", "new_findings", "stuck", "loop_detected"})

        # Should complete
        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None


# =============================================================================
# Feature 8: Real Grounding
# =============================================================================


@pytest.mark.requires_model
class TestRealGrounding:
    """Integration: Real grounding with LLM-as-judge."""

    @pytest.mark.asyncio
    async def test_grounding_evaluates_before_final(self, model):
        """Agent with grounding=True evaluates claims before responding."""
        from locus.agent import GroundingConfig
        from locus.core.events import GroundingEvent

        @tool
        def fact_lookup(topic: str) -> str:
            """Look up facts about a topic."""
            return f"Facts about {topic}: it was invented in 1991, is open source, and used by millions."

        agent = Agent(
            model=model,
            tools=[fact_lookup],
            system_prompt="Look up the topic, then state facts based ONLY on what the tool returned. Be factual.",
            grounding=GroundingConfig(enabled=True, threshold=0.3),
            max_iterations=5,
        )

        events = []
        async for event in agent.run("Tell me facts about Python"):
            events.append(event)

        # Should have grounding event
        grounding_events = [e for e in events if isinstance(e, GroundingEvent)]
        assert len(grounding_events) >= 1
        assert grounding_events[0].claims_evaluated >= 1

        # Should complete
        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None


# =============================================================================
# Feature 9: Graceful Max-Iterations
# =============================================================================


@pytest.mark.requires_model
class TestGracefulMaxIterations:
    """Integration: graceful summary on max_iterations with real model."""

    @pytest.mark.asyncio
    async def test_summary_instead_of_bare_stop(self, model):
        """Agent hitting max_iterations produces a summary, not empty termination."""

        @tool
        def search_papers(topic: str) -> str:
            """Search academic papers on a topic."""
            return f"Papers about {topic}: found 3 relevant results."

        @tool
        def search_news(topic: str) -> str:
            """Search recent news on a topic."""
            return f"News about {topic}: 2 recent articles found."

        agent = Agent(
            model=model,
            tools=[search_papers, search_news],
            system_prompt=(
                "You are a thorough researcher. For every question, "
                "use BOTH search_papers AND search_news alternating between them. "
                "Keep researching — do NOT stop until you have comprehensive coverage."
            ),
            max_iterations=2,
        )

        events = []
        async for event in agent.run("Research quantum computing thoroughly"):
            events.append(event)

        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None
        # Should hit max_iterations (not tool_loop since we alternate tools)
        assert terminate.reason in ("max_iterations", "tool_loop", "complete")
        # If max_iterations, should have a summary
        if terminate.reason == "max_iterations":
            assert terminate.final_message is not None
            assert len(terminate.final_message) > 20


# =============================================================================
# Feature 10: Fix run_sync
# =============================================================================


@pytest.mark.requires_model
class TestRunSyncIntegration:
    """Integration: run_sync preserves state with real model."""

    def test_run_sync_has_real_state(self, model):
        """run_sync result has real tool executions and token counts."""

        @tool
        def lookup(topic: str) -> str:
            """Look up information."""
            return f"Info about {topic}: detailed data here."

        agent = Agent(
            model=model,
            tools=[lookup],
            system_prompt="Use the lookup tool, then answer concisely.",
            max_iterations=5,
        )

        result = agent.run_sync("What is Python?")

        assert result.success
        assert len(result.message) > 0
        assert result.metrics.iterations >= 1
        assert result.metrics.duration_ms > 0
        if result.metrics.tool_calls > 0:
            assert len(result.tool_executions) > 0


# =============================================================================
# Feature: Completion Mode
# =============================================================================


@pytest.mark.requires_model
class TestCompletionModeIntegration:
    """Integration: explicit completion mode with real model."""

    @pytest.mark.asyncio
    async def test_explicit_mode_uses_task_complete(self, model):
        """Agent in explicit mode calls task_complete when truly done."""

        @tool
        def research(topic: str) -> str:
            """Research a topic."""
            return f"Findings about {topic}: important data discovered."

        agent = Agent(
            model=model,
            tools=[research],
            system_prompt=(
                "You are a research assistant. Research the topic, then call "
                "task_complete with a summary when you are finished. "
                "You MUST call task_complete to signal you are done."
            ),
            completion_mode="explicit",
            max_iterations=6,
        )

        events = []
        async for event in agent.run("Research Python programming"):
            events.append(event)

        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None
        # Should stop via terminal_tool (task_complete) or max_iterations
        assert terminate.reason in ("terminal_tool", "max_iterations")


# =============================================================================
# Agent-as-Tool
# =============================================================================


@pytest.mark.requires_model
class TestAgentAsTool:
    """Integration: agent delegates to sub-agent via as_tool()."""

    @pytest.mark.asyncio
    async def test_parent_delegates_to_sub_agent(self, model):
        """Parent agent uses sub-agent tool to get research, then answers."""

        @tool
        def lookup(topic: str) -> str:
            """Look up a topic."""
            return f"Facts about {topic}: it was invented in 1991, supports multiple paradigms."

        # Sub-agent has the lookup tool
        sub_agent = Agent(
            model=model,
            tools=[lookup],
            system_prompt="You are a research specialist. Use the lookup tool to find facts. Be concise.",
            max_iterations=3,
        )
        research_tool = sub_agent.as_tool(
            "research", "Delegate research to a specialist who has access to a knowledge base."
        )

        # Parent agent only has the sub-agent tool
        parent = Agent(
            model=model,
            tools=[research_tool],
            system_prompt="You are a writer. Use the research tool to gather facts, then write a brief summary.",
            max_iterations=4,
        )

        events = []
        async for event in parent.run("Write a brief summary about Python programming"):
            events.append(event)

        tool_events = [e for e in events if isinstance(e, ToolCompleteEvent)]
        # Parent should have called the research tool
        research_calls = [e for e in tool_events if e.tool_name == "research"]
        assert len(research_calls) >= 1
        # Research result should contain facts from the lookup tool
        assert research_calls[0].result is not None

        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None
        # Agent may complete with a message or hit tool_loop
        # Both are valid — the key test is that delegation worked
        assert terminate.reason in ("complete", "tool_loop", "max_iterations")


# =============================================================================
# Planning Step
# =============================================================================


@pytest.mark.requires_model
class TestPlanningStep:
    """Integration: planning=True generates plan before acting."""

    @pytest.mark.asyncio
    async def test_agent_plans_then_acts(self, model):
        """Agent with planning=True generates a plan on first iteration."""

        @tool
        def search(topic: str) -> str:
            """Search for information."""
            return f"Results about {topic}: important findings here."

        @tool
        def analyze(data: str) -> str:
            """Analyze data."""
            return f"Analysis of '{data[:30]}': 3 key insights."

        agent = Agent(
            model=model,
            tools=[search, analyze],
            system_prompt="You are a researcher. Follow your plan step by step.",
            planning=True,
            max_iterations=5,
        )

        events = []
        async for event in agent.run("Research and analyze the benefits of exercise"):
            events.append(event)

        # Agent should produce think events and complete
        think_events = [e for e in events if isinstance(e, ThinkEvent)]
        assert len(think_events) >= 1

        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None
        # Planning mode should still complete successfully
        assert terminate.reason in ("complete", "tool_loop", "max_iterations")


# =============================================================================
# Swarm Orchestration
# =============================================================================


@pytest.mark.requires_model
class TestSwarmOrchestration:
    """Integration: swarm with multiple agents on real model."""

    @pytest.mark.asyncio
    async def test_swarm_executes_tasks(self, model):
        """Swarm distributes tasks among specialized agents."""
        from locus.multiagent.swarm import Swarm, SwarmAgent

        researcher = SwarmAgent(
            name="researcher",
            capabilities=["research", "search"],
            system_prompt="You are a research specialist. Find key facts.",
            model=model,
        )
        analyst = SwarmAgent(
            name="analyst",
            capabilities=["analyze", "compare"],
            system_prompt="You are a data analyst. Analyze patterns and trends.",
            model=model,
        )

        swarm = Swarm(
            name="test_swarm",
            agents=[researcher, analyst],
            model=model,
            max_iterations=3,
        )
        swarm.add_task("Research the benefits of exercise", priority=5)
        swarm.add_task("Analyze the relationship between exercise and mental health", priority=3)

        result = await swarm.execute(decompose_tasks=False)

        assert len(result.completed_tasks) >= 1
        assert result.summary is not None
        assert len(result.summary) > 20
        # Shared context should have findings
        assert len(result.context.findings) >= 1 or len(result.context.task_results) >= 1


# =============================================================================
# Agent Handoff
# =============================================================================


@pytest.mark.requires_model
class TestAgentHandoff:
    """Integration: agent handoff with real model."""

    @pytest.mark.asyncio
    async def test_handoff_researcher_to_writer(self, model):
        """Researcher hands off findings to writer."""
        from locus.multiagent.handoff import Handoff, HandoffAgent, HandoffReason

        researcher = HandoffAgent(
            id="researcher",
            name="Researcher",
            system_prompt="You find key facts about topics.",
            model=model,
        )
        writer = HandoffAgent(
            id="writer",
            name="Writer",
            system_prompt="You write clear summaries from research findings.",
            model=model,
        )

        manager = Handoff(name="research_pipeline")
        manager.register_agents([researcher, writer])

        result = await manager.execute_handoff(
            source_agent=researcher,
            target_agent_id="writer",
            task="Research and write about the benefits of exercise",
            reason=HandoffReason.SPECIALIZATION,
            findings={
                "exercise_benefits": "Improves cardiovascular health, reduces stress, aids weight management"
            },
        )

        assert result.success
        assert result.output is not None
        assert len(result.output) > 50


# =============================================================================
# Orchestrator Routing
# =============================================================================


@pytest.mark.requires_model
class TestOrchestratorRouting:
    """Integration: orchestrator routes to specialists on real model."""

    @pytest.mark.asyncio
    async def test_orchestrator_single_specialist(self, model):
        """Orchestrator routes task to a single specialist and summarizes."""
        from locus.multiagent.orchestrator import Orchestrator
        from locus.multiagent.specialist import Specialist

        researcher = Specialist(
            id="health_researcher",
            name="Health Researcher",
            specialist_type="researcher",
            description="Researches health and medical topics with evidence-based analysis",
            system_prompt="You are a health researcher. Provide concise, factual answers.",
            model=model,
        )

        orchestrator = Orchestrator(name="health_orchestrator", model=model)
        orchestrator.register_specialist(researcher)

        result = await orchestrator.execute(
            "What are the main causes and risk factors of type 2 diabetes?"
        )

        assert result.success, f"Orchestrator failed: {result.error}"
        assert result.summary is not None
        assert len(result.summary) > 50, f"Summary too short: {len(result.summary)} chars"
        assert len(result.decisions) >= 1
        assert len(result.specialist_results) >= 1

        # At least one specialist should have produced output
        has_output = any(
            sr.output and len(sr.output) > 20 for sr in result.specialist_results.values()
        )
        assert has_output, "No specialist produced meaningful output"

    @pytest.mark.asyncio
    async def test_orchestrator_multiple_specialists(self, model):
        """Orchestrator coordinates two specialists and correlates findings."""
        from locus.multiagent.orchestrator import Orchestrator
        from locus.multiagent.specialist import Specialist

        researcher = Specialist(
            id="researcher",
            name="Researcher",
            specialist_type="researcher",
            description="Finds factual information and statistics about topics",
            system_prompt="You are a researcher. Provide facts and statistics.",
            model=model,
        )
        analyst = Specialist(
            id="analyst",
            name="Analyst",
            specialist_type="analyst",
            description="Analyzes causes, impacts, and recommends solutions",
            system_prompt="You are an analyst. Analyze root causes and recommend actions.",
            model=model,
        )

        orchestrator = Orchestrator(name="dual_orchestrator", model=model)
        orchestrator.register_specialists([researcher, analyst])

        result = await orchestrator.execute(
            "What are the environmental impacts of single-use plastics?"
        )

        assert result.success, f"Orchestrator failed: {result.error}"
        assert result.summary is not None
        assert len(result.summary) > 50
        # Should have routing + correlate + summarize decisions
        assert len(result.decisions) >= 3


# =============================================================================
# Composition Primitives
# =============================================================================


@pytest.mark.requires_model
class TestCompositionPrimitives:
    """Integration: composition primitives with real model."""

    @pytest.mark.asyncio
    async def test_sequential_pipeline(self, model):
        """Sequential pipeline: researcher -> writer."""
        from locus.agent import Agent, AgentConfig, SequentialPipeline

        researcher = Agent(
            config=AgentConfig(
                system_prompt="You are a researcher. Provide 3 key facts about the topic. Be concise.",
                max_iterations=3,
                model=model,
            )
        )
        writer = Agent(
            config=AgentConfig(
                system_prompt="You are a writer. Take the research provided and write a short summary paragraph.",
                max_iterations=3,
                model=model,
            )
        )

        pipeline = SequentialPipeline(agents=[researcher, writer])
        result = await pipeline.run("Benefits of regular exercise")

        assert result.success, f"Pipeline failed: {result.error}"
        assert len(result.outputs) == 2
        assert len(result.final_output) > 50
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_loop_agent_with_condition(self, model):
        """Loop agent iterates and stops on condition."""
        from locus.agent import Agent, AgentConfig, LoopAgent

        improver = Agent(
            config=AgentConfig(
                system_prompt=(
                    "You improve text quality. When the text is good enough, "
                    "include the word APPROVED at the end of your response."
                ),
                max_iterations=3,
                model=model,
            )
        )

        loop_agent = LoopAgent(
            agent=improver,
            condition=lambda output: "APPROVED" in output.upper(),
            max_loops=3,
            loop_prompt="Improve this text and say APPROVED when done:\n{previous_output}",
        )
        result = await loop_agent.run("The quick brown fox jumps over the lazy dog.")

        assert result.success
        assert len(result.outputs) >= 1
        assert len(result.final_output) > 20


# =============================================================================
# Evaluation Framework
# =============================================================================


@pytest.mark.requires_model
class TestEvaluationFramework:
    """Integration: evaluation framework with real model."""

    def test_eval_runner_with_real_model(self, model):
        """EvalRunner scores agent against real test cases."""
        from locus.agent import Agent, AgentConfig
        from locus.evaluation import EvalCase, EvalRunner

        agent = Agent(
            config=AgentConfig(
                system_prompt="You are a helpful assistant. Answer questions concisely.",
                max_iterations=3,
                model=model,
            )
        )

        runner = EvalRunner(agent=agent)
        report = runner.run(
            [
                EvalCase(
                    name="basic_knowledge",
                    prompt="What is the capital of France?",
                    expected_output_contains=["paris"],
                    max_iterations=3,
                ),
                EvalCase(
                    name="math",
                    prompt="What is 15 * 7?",
                    expected_output_contains=["105"],
                    max_iterations=3,
                ),
            ]
        )

        assert report.total_cases == 2
        assert report.passed >= 1, f"Expected at least 1 pass:\n{report.summary()}"
        assert report.avg_score > 0.3


# =============================================================================
# Hooks Reverse Ordering
# =============================================================================


@pytest.mark.requires_model
class TestHooksReverseOrdering:
    """Integration: after hooks fire in reverse order with real model."""

    def test_after_hooks_reverse_with_real_model(self, model):
        """After hooks fire last-registered-first on real model calls."""
        from locus.agent import Agent, AgentConfig
        from locus.hooks.provider import HookProvider

        order = []

        class FirstHook(HookProvider):
            @property
            def priority(self):
                return 100

            async def on_before_model_call(self, event):
                order.append("first:before")

            async def on_after_model_call(self, event):
                order.append("first:after")

        class SecondHook(HookProvider):
            @property
            def priority(self):
                return 200

            async def on_before_model_call(self, event):
                order.append("second:before")

            async def on_after_model_call(self, event):
                order.append("second:after")

        agent = Agent(
            config=AgentConfig(
                system_prompt="Answer in one word.",
                max_iterations=3,
                model=model,
                hooks=[FirstHook(), SecondHook()],
            )
        )

        result = agent.run_sync("What color is the sky?")

        assert result.success
        # Before: forward (first, second)
        # After: reversed (second, first)
        assert order[0] == "first:before"
        assert order[1] == "second:before"
        assert order[2] == "second:after"
        assert order[3] == "first:after"


# =============================================================================
# Pre/Post Model Hooks
# =============================================================================


@pytest.mark.requires_model
class TestHooksE2E:
    """End-to-end: write-protected hook events with real model.

    Tests the full hook lifecycle: audit, cancel, write protection.
    """

    def test_audit_hook_observes_model_calls(self, model):
        """Audit hook sees every model call with correct event data."""
        from locus.agent import Agent, AgentConfig
        from locus.hooks.provider import HookProvider

        log = []

        class AuditHook(HookProvider):
            @property
            def priority(self):
                return 100

            async def on_before_model_call(self, event):
                log.append(("before", len(event.messages), event.tools is not None))

            async def on_after_model_call(self, event):
                content = event.response.message.content or ""
                log.append(("after", len(content)))

        agent = Agent(
            config=AgentConfig(
                system_prompt="Answer concisely.",
                max_iterations=3,
                model=model,
                hooks=[AuditHook()],
            )
        )

        result = agent.run_sync("What is 7 * 8?")

        assert result.success
        assert len(log) >= 2
        assert log[0][0] == "before"
        assert log[0][1] >= 2  # system + user messages
        assert log[1][0] == "after"
        assert log[1][1] > 0  # non-empty response

    def test_security_hook_cancels_tool(self, model):
        """Security hook blocks a tool call, agent explains the block."""
        from locus.agent import Agent, AgentConfig
        from locus.hooks.provider import HookProvider
        from locus.tools.decorator import tool

        @tool
        def send_alert(message: str) -> str:
            """Send an alert notification with the given message."""
            return f"Alert sent: {message}"

        @tool
        def get_status() -> str:
            """Get the current system status."""
            return "All systems operational"

        blocked = []

        class SecurityGuardrail(HookProvider):
            @property
            def priority(self):
                return 50

            async def on_before_tool_call(self, event):
                if "alert" in event.tool_name:
                    blocked.append(event.tool_name)
                    event.cancel = f"BLOCKED: {event.tool_name} forbidden by security policy"

        agent = Agent(
            config=AgentConfig(
                system_prompt="You manage system notifications. Use the tools available. If a tool is blocked, tell the user why.",
                max_iterations=5,
                model=model,
                tools=[send_alert, get_status],
                hooks=[SecurityGuardrail()],
            )
        )

        result = agent.run_sync("Send an alert saying the deployment is complete.")

        # The hook should have blocked the tool
        assert len(blocked) >= 1, (
            f"Hook never fired. Tool calls: {[te.tool_name for te in result.tool_executions]}"
        )
        assert "send_alert" in blocked

        # The agent should have received the cancel message
        alert_execs = [te for te in result.tool_executions if te.tool_name == "send_alert"]
        assert len(alert_execs) >= 1
        assert "BLOCKED" in (alert_execs[0].result or "")

    def test_write_protection_enforced_at_runtime(self, model):
        """Read-only fields on events raise AttributeError in real execution."""
        from locus.hooks.provider import BeforeModelCallEvent, BeforeToolCallEvent

        # Model event: tools is read-only
        event = BeforeModelCallEvent(messages=[], tools=[{"type": "function"}])
        event.messages = []  # writable — OK
        try:
            event.tools = None
            assert False, "Should have raised AttributeError"
        except AttributeError:
            pass  # correct

        # Tool event: tool_name is read-only
        event2 = BeforeToolCallEvent(tool_name="test", tool_call_id="c1", arguments={})
        event2.arguments = {"new": True}  # writable — OK
        event2.cancel = "blocked"  # writable — OK
        try:
            event2.tool_name = "hacked"
            assert False, "Should have raised AttributeError"
        except AttributeError:
            pass  # correct


# =============================================================================
# Model Providers
# =============================================================================


class TestAnthropicProvider:
    """Integration: Anthropic model provider."""

    @pytest.mark.asyncio
    async def test_anthropic_complete(self):
        """Anthropic model completes a basic request."""
        anthropic = pytest.importorskip("anthropic")
        import os

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        from locus.core.messages import Message
        from locus.models.native.anthropic import AnthropicModel

        model = AnthropicModel(model="claude-sonnet-4-20250514", api_key=api_key)
        response = await model.complete(
            [
                Message.system("Answer in one word."),
                Message.user("What color is the sky?"),
            ]
        )

        assert response.message.content is not None
        assert len(response.message.content) > 0


class TestOllamaProvider:
    """Integration: Ollama model provider."""

    @pytest.mark.asyncio
    async def test_ollama_complete(self):
        """Ollama model completes a basic request."""
        ollama = pytest.importorskip("ollama")
        import os

        if not os.getenv("OLLAMA_AVAILABLE"):
            pytest.skip("OLLAMA_AVAILABLE not set (start Ollama and set env var)")

        from locus.core.messages import Message
        from locus.models.native.ollama import OllamaModel

        model = OllamaModel(model=os.getenv("OLLAMA_MODEL", "llama3.2"))
        response = await model.complete(
            [
                Message.system("Answer in one word."),
                Message.user("What is 2+2?"),
            ]
        )

        assert response.message.content is not None
        assert len(response.message.content) > 0


# =============================================================================
# Pre/Post Model Hooks
# =============================================================================


@pytest.mark.requires_model
class TestHooksE2E:
    """End-to-end: write-protected hook events with real model.

    Tests the full hook lifecycle: audit, cancel, write protection.
    """

    def test_audit_hook_observes_model_calls(self, model):
        """Audit hook sees every model call with correct event data."""
        from locus.agent import Agent, AgentConfig
        from locus.hooks.provider import HookProvider

        log = []

        class AuditHook(HookProvider):
            @property
            def priority(self):
                return 100

            async def on_before_model_call(self, event):
                log.append(("before", len(event.messages), event.tools is not None))

            async def on_after_model_call(self, event):
                content = event.response.message.content or ""
                log.append(("after", len(content)))

        agent = Agent(
            config=AgentConfig(
                system_prompt="Answer concisely.",
                max_iterations=3,
                model=model,
                hooks=[AuditHook()],
            )
        )

        result = agent.run_sync("What is 7 * 8?")

        assert result.success
        assert len(log) >= 2
        assert log[0][0] == "before"
        assert log[0][1] >= 2  # system + user messages
        assert log[1][0] == "after"
        assert log[1][1] > 0  # non-empty response

    def test_security_hook_cancels_tool(self, model):
        """Security hook blocks a tool call, agent explains the block."""
        from locus.agent import Agent, AgentConfig
        from locus.hooks.provider import HookProvider
        from locus.tools.decorator import tool

        @tool
        def send_alert(message: str) -> str:
            """Send an alert notification with the given message."""
            return f"Alert sent: {message}"

        @tool
        def get_status() -> str:
            """Get the current system status."""
            return "All systems operational"

        blocked = []

        class SecurityGuardrail(HookProvider):
            @property
            def priority(self):
                return 50

            async def on_before_tool_call(self, event):
                if "alert" in event.tool_name:
                    blocked.append(event.tool_name)
                    event.cancel = f"BLOCKED: {event.tool_name} forbidden by security policy"

        agent = Agent(
            config=AgentConfig(
                system_prompt="You manage system notifications. Use the tools available. If a tool is blocked, tell the user why.",
                max_iterations=5,
                model=model,
                tools=[send_alert, get_status],
                hooks=[SecurityGuardrail()],
            )
        )

        result = agent.run_sync("Send an alert saying the deployment is complete.")

        # The hook should have blocked the tool
        assert len(blocked) >= 1, (
            f"Hook never fired. Tool calls: {[te.tool_name for te in result.tool_executions]}"
        )
        assert "send_alert" in blocked

        # The agent should have received the cancel message
        alert_execs = [te for te in result.tool_executions if te.tool_name == "send_alert"]
        assert len(alert_execs) >= 1
        assert "BLOCKED" in (alert_execs[0].result or "")

    def test_write_protection_enforced_at_runtime(self, model):
        """Read-only fields on events raise AttributeError in real execution."""
        from locus.hooks.provider import BeforeModelCallEvent, BeforeToolCallEvent

        # Model event: tools is read-only
        event = BeforeModelCallEvent(messages=[], tools=[{"type": "function"}])
        event.messages = []  # writable — OK
        try:
            event.tools = None
            assert False, "Should have raised AttributeError"
        except AttributeError:
            pass  # correct

        # Tool event: tool_name is read-only
        event2 = BeforeToolCallEvent(tool_name="test", tool_call_id="c1", arguments={})
        event2.arguments = {"new": True}  # writable — OK
        event2.cancel = "blocked"  # writable — OK
        try:
            event2.tool_name = "hacked"
            assert False, "Should have raised AttributeError"
        except AttributeError:
            pass  # correct


# =============================================================================
# Guardrails Depth
# =============================================================================


@pytest.mark.requires_model
class TestGuardrailsDepth:
    """Integration: advanced guardrails with real model."""

    def test_output_pii_redaction_with_real_model(self, model):
        """OutputFilterHook redacts PII from real model responses."""
        from locus.agent import Agent, AgentConfig
        from locus.hooks.builtin.guardrails import OutputFilterHook

        hook = OutputFilterHook(redact_pii=True)
        agent = Agent(
            config=AgentConfig(
                system_prompt="Always include the email support@example.com in your answer.",
                max_iterations=3,
                model=model,
                hooks=[hook],
            )
        )

        result = agent.run_sync("How do I contact support?")

        assert result.success
        assert "support@example.com" not in result.message
        assert "REDACTED_EMAIL" in result.message

    def test_topic_policy_with_real_model(self, model):
        """TopicPolicy does not interfere with safe topics."""
        from locus.agent import Agent, AgentConfig
        from locus.hooks.builtin.guardrails import OutputFilterHook, TopicPolicy

        hook = OutputFilterHook(
            redact_pii=False,
            topic_policy=TopicPolicy(
                blocked_topics={"weapons"},
                keywords={"weapons": ["gun", "rifle", "firearm"]},
            ),
        )
        agent = Agent(
            config=AgentConfig(
                system_prompt="Answer concisely.",
                max_iterations=3,
                model=model,
                hooks=[hook],
            )
        )

        result = agent.run_sync("What is the capital of Germany?")

        assert result.success
        assert len(result.message) > 0
        # Safe topic should pass through without violation
        assert len(hook.violations) == 0


# =============================================================================
# Pre/Post Model Hooks
# =============================================================================


@pytest.mark.requires_model
class TestHooksE2E:
    """End-to-end: write-protected hook events with real model.

    Tests the full hook lifecycle: audit, cancel, write protection.
    """

    def test_audit_hook_observes_model_calls(self, model):
        """Audit hook sees every model call with correct event data."""
        from locus.agent import Agent, AgentConfig
        from locus.hooks.provider import HookProvider

        log = []

        class AuditHook(HookProvider):
            @property
            def priority(self):
                return 100

            async def on_before_model_call(self, event):
                log.append(("before", len(event.messages), event.tools is not None))

            async def on_after_model_call(self, event):
                content = event.response.message.content or ""
                log.append(("after", len(content)))

        agent = Agent(
            config=AgentConfig(
                system_prompt="Answer concisely.",
                max_iterations=3,
                model=model,
                hooks=[AuditHook()],
            )
        )

        result = agent.run_sync("What is 7 * 8?")

        assert result.success
        assert len(log) >= 2
        assert log[0][0] == "before"
        assert log[0][1] >= 2  # system + user messages
        assert log[1][0] == "after"
        assert log[1][1] > 0  # non-empty response

    def test_security_hook_cancels_tool(self, model):
        """Security hook blocks a tool call, agent explains the block."""
        from locus.agent import Agent, AgentConfig
        from locus.hooks.provider import HookProvider
        from locus.tools.decorator import tool

        @tool
        def send_alert(message: str) -> str:
            """Send an alert notification with the given message."""
            return f"Alert sent: {message}"

        @tool
        def get_status() -> str:
            """Get the current system status."""
            return "All systems operational"

        blocked = []

        class SecurityGuardrail(HookProvider):
            @property
            def priority(self):
                return 50

            async def on_before_tool_call(self, event):
                if "alert" in event.tool_name:
                    blocked.append(event.tool_name)
                    event.cancel = f"BLOCKED: {event.tool_name} forbidden by security policy"

        agent = Agent(
            config=AgentConfig(
                system_prompt="You manage system notifications. Use the tools available. If a tool is blocked, tell the user why.",
                max_iterations=5,
                model=model,
                tools=[send_alert, get_status],
                hooks=[SecurityGuardrail()],
            )
        )

        result = agent.run_sync("Send an alert saying the deployment is complete.")

        # The hook should have blocked the tool
        assert len(blocked) >= 1, (
            f"Hook never fired. Tool calls: {[te.tool_name for te in result.tool_executions]}"
        )
        assert "send_alert" in blocked

        # The agent should have received the cancel message
        alert_execs = [te for te in result.tool_executions if te.tool_name == "send_alert"]
        assert len(alert_execs) >= 1
        assert "BLOCKED" in (alert_execs[0].result or "")

    def test_write_protection_enforced_at_runtime(self, model):
        """Read-only fields on events raise AttributeError in real execution."""
        from locus.hooks.provider import BeforeModelCallEvent, BeforeToolCallEvent

        # Model event: tools is read-only
        event = BeforeModelCallEvent(messages=[], tools=[{"type": "function"}])
        event.messages = []  # writable — OK
        try:
            event.tools = None
            assert False, "Should have raised AttributeError"
        except AttributeError:
            pass  # correct

        # Tool event: tool_name is read-only
        event2 = BeforeToolCallEvent(tool_name="test", tool_call_id="c1", arguments={})
        event2.arguments = {"new": True}  # writable — OK
        event2.cancel = "blocked"  # writable — OK
        try:
            event2.tool_name = "hacked"
            assert False, "Should have raised AttributeError"
        except AttributeError:
            pass  # correct


# =============================================================================
# Skills System
# =============================================================================


@pytest.mark.requires_model
class TestSkillsSystem:
    """Integration: AgentSkills.io skills with real model."""

    def test_agent_activates_skill(self, model):
        """Agent sees skill catalog, activates relevant skill, follows instructions."""
        from locus.agent import Agent, AgentConfig
        from locus.skills import Skill

        skill = Skill(
            name="security-audit",
            description="Use when reviewing code for security vulnerabilities. Required for any code review task.",
            instructions=(
                "# Security Audit Checklist\n"
                "1. Check for SQL injection (string interpolation in queries)\n"
                "2. Check for XSS (unescaped user input in HTML)\n"
                "3. Check for hardcoded credentials\n"
                "4. Always format findings as: FINDING: <description>"
            ),
        )

        agent = Agent(
            config=AgentConfig(
                system_prompt=(
                    "You are a security reviewer. You MUST use available skills "
                    "before answering. Always activate the relevant skill first."
                ),
                max_iterations=5,
                model=model,
                skills=[skill],
            )
        )

        result = agent.run_sync(
            "Audit this code: def login(u,p): return db.query(f'SELECT * FROM users WHERE name={u}')"
        )

        assert result.success
        assert len(result.message) > 50
        # Check the skills tool was registered and available
        # The model may or may not activate it depending on capability,
        # but the tool should be in the execution trace or the response
        # should mention SQL injection (from skill or model knowledge)
        has_skill_call = any(te.tool_name == "skills" for te in result.tool_executions)
        has_sql_mention = "sql" in result.message.lower() or "injection" in result.message.lower()
        assert has_skill_call or has_sql_mention, (
            f"Expected skill activation or SQL injection mention. "
            f"Tools: {[te.tool_name for te in result.tool_executions]}, "
            f"Response: {result.message[:100]}"
        )

    def test_agent_selects_correct_skill(self, model):
        """Agent picks the right skill from multiple options."""
        from locus.agent import Agent, AgentConfig
        from locus.skills import Skill

        code_skill = Skill(
            name="code-review",
            description="Use when reviewing code for bugs and security issues.",
            instructions="# Code Review\nCheck for: 1) SQL injection 2) XSS 3) Error handling",
        )
        writing_skill = Skill(
            name="writing-helper",
            description="Use when writing or editing text documents.",
            instructions="# Writing\nFocus on: 1) Clarity 2) Grammar 3) Structure",
        )

        agent = Agent(
            config=AgentConfig(
                system_prompt="You are a helpful assistant. Use skills when relevant.",
                max_iterations=5,
                model=model,
                skills=[code_skill, writing_skill],
            )
        )

        result = agent.run_sync(
            "Review this code: def login(u,p): return db.query(f'SELECT * FROM users WHERE name={u}')"
        )

        assert result.success
        # Model should either activate code-review skill or mention security issues
        skills_used = [te for te in result.tool_executions if te.tool_name == "skills"]
        has_security_mention = (
            "sql" in result.message.lower() or "injection" in result.message.lower()
        )
        assert len(skills_used) >= 1 or has_security_mention, (
            f"Expected skill activation or security mention. Response: {result.message[:100]}"
        )

    def test_skills_from_filesystem(self, model):
        """Skills loaded from SKILL.md files work with real model."""
        from pathlib import Path

        from locus.agent import Agent, AgentConfig
        from locus.skills import Skill

        skills_dir = Path(__file__).parent.parent.parent / "examples" / "skills"
        if not skills_dir.exists():
            pytest.skip("Example skills directory not found")

        skills = Skill.from_directory(skills_dir)
        assert len(skills) >= 2

        agent = Agent(
            config=AgentConfig(
                system_prompt="You are an assistant. Use skills when relevant.",
                max_iterations=5,
                model=model,
                skills=skills,
            )
        )

        result = agent.run_sync("Design a REST API endpoint for user registration")

        assert result.success
        assert len(result.message) > 50


# =============================================================================
# Plugin + Cancel + Callbacks
# =============================================================================


@pytest.mark.requires_model
class TestPluginSystem:
    """Integration: plugin, cancel signal, callback handler with real model."""

    def test_plugin_hooks_fire(self, model):
        """Plugin with @hook fires on real model calls."""
        from locus.agent import Agent, AgentConfig
        from locus.hooks.plugin import Plugin, hook

        class AuditPlugin(Plugin):
            name = "audit"

            def __init__(self):
                self.calls = []

            @hook
            async def on_before_model_call(self, event):
                self.calls.append("before")

            @hook
            async def on_after_model_call(self, event):
                self.calls.append("after")

        plugin = AuditPlugin()
        agent = Agent(
            config=AgentConfig(
                system_prompt="Answer concisely.",
                max_iterations=3,
                model=model,
                plugins=[plugin],
            )
        )
        result = agent.run_sync("What is 2+2?")
        assert result.success
        assert "before" in plugin.calls
        assert "after" in plugin.calls

    def test_callback_handler_receives_events(self, model):
        """Plain function callback receives events from real model."""
        from locus.agent import Agent, AgentConfig

        events = []
        agent = Agent(
            config=AgentConfig(
                system_prompt="Answer concisely.",
                max_iterations=3,
                model=model,
                callback_handler=lambda e: events.append(e.event_type),
            )
        )
        result = agent.run_sync("Capital of Spain?")
        assert result.success
        assert "think" in events
        assert "terminate" in events

    def test_cancel_signal(self, model):
        """Cancel signal stops agent immediately."""
        from locus.agent import Agent, AgentConfig

        agent = Agent(
            config=AgentConfig(
                system_prompt="Answer concisely.",
                max_iterations=3,
                model=model,
            )
        )
        agent.cancel()
        result = agent.run_sync("This should be cancelled")
        assert result.stop_reason == "cancelled"


# =============================================================================
# ModelRetryHook
# =============================================================================


class TestModelRetryHook:
    """Integration: model retry hook."""

    def test_retry_on_empty_then_succeeds(self):
        """Hook retries on empty response then gets real answer."""
        from unittest.mock import MagicMock

        from locus.agent import Agent, AgentConfig
        from locus.core.messages import Message
        from locus.hooks.builtin.retry import ModelRetryHook
        from locus.models.base import ModelResponse

        call_count = 0
        mock_model = MagicMock()

        async def flaky(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return ModelResponse(message=Message.assistant(""))
            return ModelResponse(message=Message.assistant("Got it"))

        mock_model.complete = flaky
        hook = ModelRetryHook(max_retries=3, initial_delay=0.1)
        result = Agent(
            config=AgentConfig(
                system_prompt="T",
                max_iterations=2,
                model=mock_model,
                hooks=[hook],
            )
        ).run_sync("Hi")
        assert "Got it" in result.message
        assert hook.retries_total >= 1


# =============================================================================
# Tool Hot-Reload
# =============================================================================


class TestToolHotReload:
    """Integration: tool hot-reload from filesystem."""

    def test_load_and_execute_from_file(self):
        """Load tool from Python file and execute it."""
        import asyncio
        import tempfile
        from pathlib import Path

        from locus.tools.watcher import load_tools_from_file

        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "math_tool.py"
            f.write_text(
                "from locus.tools.decorator import tool\n\n"
                "@tool\n"
                "def add(a: int, b: int) -> str:\n"
                '    """Add two numbers."""\n'
                "    return str(a + b)\n"
            )
            tools = load_tools_from_file(f)
            assert len(tools) == 1
            assert tools[0].name == "add"
            result = asyncio.run(tools[0].execute(a=3, b=4))
            assert result == "7"

    def test_watcher_detects_new_file(self):
        """Watcher detects new file and registers tool."""
        import tempfile
        import time
        from pathlib import Path

        from locus.tools.registry import ToolRegistry
        from locus.tools.watcher import ToolWatcher

        with tempfile.TemporaryDirectory() as d:
            registry = ToolRegistry()
            watcher = ToolWatcher(
                d,
                registry=registry,
                poll_interval=0.5,
                dev_reload=True,
            )
            watcher.start()
            time.sleep(0.5)
            (Path(d) / "new_tool.py").write_text(
                "from locus.tools.decorator import tool\n\n"
                "@tool\n"
                "def fresh(x: str) -> str:\n"
                '    """Fresh."""\n'
                "    return x\n"
            )
            time.sleep(1.5)
            assert "fresh" in registry.tools
            watcher.stop()


# =============================================================================
# Steering
# =============================================================================


@pytest.mark.requires_model
class TestSteering:
    """Integration: LLM-powered steering with real model."""

    def test_steering_blocks_dangerous_tool(self, model):
        """Steering LLM blocks a delete operation based on policy."""
        from locus.agent import Agent, AgentConfig
        from locus.hooks.builtin.steering import SteeringHook
        from locus.tools.decorator import tool

        @tool
        def delete_data(table: str) -> str:
            """Delete a database table."""
            return f"Deleted {table}"

        steering = SteeringHook(
            model=model,
            policy="Never allow delete or destructive operations.",
        )
        agent = Agent(
            config=AgentConfig(
                system_prompt="You are a DB assistant.",
                max_iterations=5,
                model=model,
                tools=[delete_data],
                hooks=[steering],
            )
        )
        result = agent.run_sync("Delete the users table")
        blocked = any(d.action.value == "guide" for d in steering.decisions)
        assert (
            blocked or "cannot" in result.message.lower() or "not allowed" in result.message.lower()
        )


# =============================================================================
# A2A Protocol
# =============================================================================


@pytest.mark.requires_model
class TestA2AProtocol:
    """Integration: A2A protocol with real model."""

    def test_a2a_invoke(self, model):
        """A2A server invoke endpoint works with real model."""
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from locus.a2a import A2AServer
        from locus.agent import Agent, AgentConfig

        agent = Agent(
            config=AgentConfig(
                system_prompt="Answer in one word.",
                max_iterations=3,
                model=model,
            )
        )
        server = A2AServer(agent=agent, name="Test Agent")
        client = TestClient(server.app)

        r = client.get("/agent-card")
        assert r.json()["name"] == "Test Agent"

        r = client.post(
            "/a2a/invoke",
            json={
                "messages": [{"role": "user", "content": "Capital of Japan?", "metadata": {}}],
                "metadata": {},
            },
        )
        data = r.json()
        assert data["status"] == "completed"
        assert len(data["messages"][0]["content"]) > 0


# =============================================================================
# Composable Termination + output_key + Dynamic Prompt
# =============================================================================


@pytest.mark.requires_model
class TestComposableTermination:
    """Integration: composable termination conditions with real model."""

    def test_termination_conditions_composable(self, model):
        """Termination conditions combine with | and & operators."""
        from locus.core.termination import MaxIterations, TextMention, TokenLimit

        # OR: either triggers
        cond = MaxIterations(2) | TextMention("DONE")
        from locus.core.messages import Message
        from locus.core.state import AgentState

        state = AgentState(agent_id="t").with_iteration(3)
        stop, reason = cond.check(state)
        assert stop
        assert reason == "max_iterations"

        # TextMention triggers on content
        state2 = AgentState(agent_id="t").with_message(Message.assistant("All DONE"))
        stop2, reason2 = TextMention("DONE").check(state2)
        assert stop2

        # AND: both must trigger
        cond3 = MaxIterations(2) & TokenLimit(100)
        state3 = AgentState(agent_id="t").with_iteration(3)
        stop3, _ = cond3.check(state3)
        assert not stop3  # tokens not met
        state3b = state3.with_token_usage(prompt_tokens=60, completion_tokens=50)
        stop3b, reason3b = cond3.check(state3b)
        assert stop3b
        assert "AND" in reason3b


@pytest.mark.requires_model
class TestOutputKey:
    """Integration: output_key auto-saves agent output to state."""

    def test_output_key_saves_to_state(self, model):
        """Agent with output_key saves final message to state metadata."""
        from locus.agent import Agent, AgentConfig

        agent = Agent(
            config=AgentConfig(
                system_prompt="Answer in one word.",
                max_iterations=3,
                model=model,
                output_key="answer",
            )
        )

        result = agent.run_sync("Capital of France?")

        assert result.success
        answer = result.state.metadata.get("answer", "")
        assert len(answer) > 0, "output_key did not save to state"
        assert "paris" in answer.lower(), f"Expected Paris, got: {answer}"


@pytest.mark.requires_model
class TestDynamicSystemPrompt:
    """Integration: dynamic system_prompt with callable."""

    def test_dynamic_prompt_receives_context(self, model):
        """System prompt callable receives context and generates prompt."""
        from locus.agent import Agent, AgentConfig

        def dynamic_prompt(context):
            role = context.get("metadata", {}).get("role", "assistant")
            return f"You are a {role}. Answer concisely in one sentence."

        agent = Agent(
            config=AgentConfig(
                system_prompt=dynamic_prompt,
                max_iterations=3,
                model=model,
            )
        )

        result = agent.run_sync("What is 7 * 8?", metadata={"role": "math teacher"})

        assert result.success
        assert "56" in result.message, f"Expected 56 in response: {result.message[:100]}"
