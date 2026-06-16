# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Comprehensive tests for the ReAct loop implementation."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from locus.core.events import (
    ReflectEvent,
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
    ToolStartEvent,
)
from locus.core.messages import Message, ToolCall
from locus.core.state import AgentState, ToolExecution
from locus.loop.nodes import ExecuteNode, ReflectNode, ThinkNode
from locus.loop.react import ReActLoop, ReActLoopConfig, create_react_loop
from locus.loop.router import ConditionalRouter, NodeType, RouteDecision, Router
from locus.loop.runner import BatchRunner, LoopRunner, StreamingCollector, create_runner
from locus.models.base import ModelResponse
from locus.tools.decorator import tool
from locus.tools.registry import ToolRegistry, create_registry
from tests._safe_math import safe_math_eval


# =============================================================================
# Test Fixtures and Mocks
# =============================================================================


@pytest.fixture
def mock_model() -> AsyncMock:
    """Create a mock model."""
    model = AsyncMock()
    model.complete = AsyncMock()
    return model


@pytest.fixture
def sample_tools() -> ToolRegistry:
    """Create sample tools for testing."""

    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Results for: {query}"

    @tool
    def calculate(expression: str) -> str:
        """Calculate a mathematical expression."""
        return str(safe_math_eval(expression))

    @tool
    def done(result: str) -> str:
        """Mark task as complete."""
        return f"Completed: {result}"

    return create_registry(search, calculate, done)


@pytest.fixture
def empty_registry() -> ToolRegistry:
    """Create an empty tool registry."""
    return ToolRegistry()


def create_model_response(
    content: str | None = None,
    tool_calls: list[ToolCall] | None = None,
) -> ModelResponse:
    """Helper to create ModelResponse."""
    message = Message.assistant(content=content, tool_calls=tool_calls)
    return ModelResponse(message=message)


# =============================================================================
# Router Tests
# =============================================================================


class TestRouter:
    """Tests for the Router class."""

    def test_route_from_think_with_tool_calls(self):
        """Route to Execute when tool calls exist."""
        router = Router()
        state = AgentState().with_message(
            Message.assistant(
                content="Let me search for that",
                tool_calls=[ToolCall(name="search", arguments={"query": "test"})],
            )
        )

        decision = router.route_from_think(state)

        assert decision.next_node == NodeType.EXECUTE
        assert "tool call" in decision.reason.lower()

    def test_route_from_think_no_tools_terminates(self):
        """Route to Terminate when no tool calls and iteration > 0."""
        router = Router()
        state = AgentState(iteration=1).with_message(Message.assistant(content="Here is my answer"))

        decision = router.route_from_think(state)

        assert decision.next_node == NodeType.TERMINATE
        assert "no_tools" in decision.reason.lower()

    def test_route_from_think_max_iterations(self):
        """Route to Terminate at max iterations."""
        router = Router()
        state = AgentState(max_iterations=5, iteration=5)

        decision = router.route_from_think(state)

        assert decision.next_node == NodeType.TERMINATE
        assert decision.metadata.get("termination_reason") == "max_iterations"

    def test_route_from_think_confidence_met(self):
        """Route to Terminate when confidence threshold met."""
        router = Router()
        state = AgentState(confidence=0.9, confidence_threshold=0.85)

        decision = router.route_from_think(state)

        assert decision.next_node == NodeType.TERMINATE
        assert decision.metadata.get("termination_reason") == "confidence_met"

    def test_route_from_execute_to_reflect(self):
        """Route to Reflect after execution when enabled."""
        router = Router(enable_reflection=True, reflect_interval=1)
        state = AgentState(iteration=1).with_tool_execution(
            ToolExecution(
                tool_name="search",
                tool_call_id="call_1",
                arguments={"query": "test"},
                result="Found results",
            )
        )

        decision = router.route_from_execute(state)

        assert decision.next_node == NodeType.REFLECT

    def test_route_from_execute_to_think_no_reflection(self):
        """Route to Think after execution when reflection disabled."""
        router = Router(enable_reflection=False)
        state = AgentState()

        decision = router.route_from_execute(state)

        assert decision.next_node == NodeType.THINK

    def test_route_from_reflect_to_think(self):
        """Route to Think after reflection."""
        router = Router()
        state = AgentState()

        decision = router.route_from_reflect(state)

        assert decision.next_node == NodeType.THINK

    def test_route_from_reflect_with_termination(self):
        """Route to Terminate from Reflect when conditions met."""
        router = Router()
        state = AgentState(confidence=0.9, confidence_threshold=0.85)

        decision = router.route_from_reflect(state)

        assert decision.next_node == NodeType.TERMINATE

    def test_route_generic(self):
        """Test the generic route method."""
        router = Router()
        state = AgentState()

        decision = router.route(NodeType.REFLECT, state)

        assert decision.next_node == NodeType.THINK

    def test_route_from_terminate(self):
        """Route from Terminate node stays terminated."""
        router = Router()
        state = AgentState()

        decision = router.route(NodeType.TERMINATE, state)

        assert decision.next_node == NodeType.TERMINATE
        assert "terminated" in decision.reason.lower()

    def test_route_from_think_reflect_without_tools(self):
        """Route to Reflect when no tools and skip_reflect_without_tools=False."""
        router = Router(enable_reflection=True, skip_reflect_without_tools=False)
        # Use iteration=0 to avoid "no_tools" termination condition
        state = AgentState(iteration=0).with_message(Message.assistant(content="Here is my answer"))

        decision = router.route_from_think(state)

        assert decision.next_node == NodeType.REFLECT
        assert "no tool calls" in decision.reason.lower()

    def test_should_reflect_on_error(self):
        """Reflection triggered on tool execution error."""
        router = Router(enable_reflection=True)
        state = AgentState(iteration=0).with_tool_execution(
            ToolExecution(
                tool_name="search",
                tool_call_id="call_1",
                arguments={"query": "test"},
                result=None,
                error="Connection failed",
            )
        )

        decision = router.route_from_execute(state)

        assert decision.next_node == NodeType.REFLECT

    def test_should_reflect_checks_interval(self):
        """_should_reflect respects reflect_interval."""
        router = Router(enable_reflection=True, reflect_interval=2)
        state = AgentState(iteration=2)

        # At iteration 2 with interval 2, should reflect
        decision = router.route_from_execute(state)

        assert decision.next_node == NodeType.REFLECT


class TestConditionalRouter:
    """Tests for ConditionalRouter with custom conditions."""

    def test_add_condition_returns_new_router(self):
        """Adding a condition returns a new router instance."""
        router = ConditionalRouter()

        def custom_cond(state: AgentState) -> RouteDecision | None:
            return None

        new_router = router.add_condition("custom", custom_cond)

        assert router is not new_router
        assert len(router.custom_conditions) == 0
        assert len(new_router.custom_conditions) == 1

    def test_custom_condition_overrides_default(self):
        """Custom condition can override default routing."""
        router = ConditionalRouter()

        def force_terminate(state: AgentState) -> RouteDecision | None:
            if state.iteration > 0:
                return RouteDecision(
                    next_node=NodeType.TERMINATE,
                    reason="Custom termination",
                )
            return None

        router = router.add_condition("force_terminate", force_terminate)
        state = AgentState(iteration=1)

        decision = router.route(NodeType.THINK, state)

        assert decision.next_node == NodeType.TERMINATE
        assert decision.metadata.get("custom_condition") == "force_terminate"

    def test_custom_condition_fallback_to_default(self):
        """Falls back to default routing when custom condition returns None."""
        router = ConditionalRouter()

        def no_override(state: AgentState) -> RouteDecision | None:
            return None

        router = router.add_condition("no_override", no_override)
        state = AgentState()

        decision = router.route(NodeType.REFLECT, state)

        # Default routing from REFLECT goes to THINK
        assert decision.next_node == NodeType.THINK

    def test_custom_condition_exception_continues(self):
        """Exception in custom condition is caught and next condition is tried."""
        router = ConditionalRouter()

        def failing_condition(state: AgentState) -> RouteDecision | None:
            raise RuntimeError("Condition failed")

        def working_condition(state: AgentState) -> RouteDecision | None:
            return RouteDecision(
                next_node=NodeType.TERMINATE,
                reason="Working condition",
            )

        router = router.add_condition("failing", failing_condition)
        router = router.add_condition("working", working_condition)
        state = AgentState()

        decision = router.route(NodeType.THINK, state)

        assert decision.next_node == NodeType.TERMINATE
        assert decision.metadata.get("custom_condition") == "working"


# =============================================================================
# Node Tests
# =============================================================================


class TestThinkNode:
    """Tests for ThinkNode."""

    @pytest.mark.asyncio
    async def test_execute_with_reasoning(self, mock_model, empty_registry):
        """ThinkNode produces reasoning."""
        mock_model.complete.return_value = create_model_response(
            content="I need to think about this"
        )

        node = ThinkNode(model=mock_model, registry=empty_registry)
        state = AgentState().with_message(Message.user("Hello"))

        result = await node.execute(state)

        assert len(result.events) == 1
        assert isinstance(result.events[0], ThinkEvent)
        assert result.events[0].reasoning == "I need to think about this"

    @pytest.mark.asyncio
    async def test_execute_with_tool_calls(self, mock_model, sample_tools):
        """ThinkNode produces tool calls."""
        tool_call = ToolCall(name="search", arguments={"query": "test"})
        mock_model.complete.return_value = create_model_response(
            content="Let me search",
            tool_calls=[tool_call],
        )

        node = ThinkNode(model=mock_model, registry=sample_tools)
        state = AgentState().with_message(Message.user("Search for test"))

        result = await node.execute(state)

        assert len(result.events) == 1
        event = result.events[0]
        assert isinstance(event, ThinkEvent)
        assert len(event.tool_calls) == 1
        assert event.tool_calls[0].name == "search"

    @pytest.mark.asyncio
    async def test_execute_adds_system_prompt(self, mock_model, empty_registry):
        """ThinkNode adds system prompt if provided."""
        mock_model.complete.return_value = create_model_response(content="OK")

        node = ThinkNode(
            model=mock_model,
            registry=empty_registry,
            system_prompt="You are a helpful assistant",
        )
        state = AgentState().with_message(Message.user("Hello"))

        await node.execute(state)

        # Check that system message was added
        call_args = mock_model.complete.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        assert messages[0].role.value == "system"
        assert "helpful assistant" in messages[0].content

    @pytest.mark.asyncio
    async def test_execute_updates_state(self, mock_model, empty_registry):
        """ThinkNode updates state with assistant message."""
        mock_model.complete.return_value = create_model_response(content="Response")

        node = ThinkNode(model=mock_model, registry=empty_registry)
        state = AgentState().with_message(Message.user("Hello"))

        result = await node.execute(state)

        assert len(result.state.messages) == 2
        assert result.state.messages[-1].role.value == "assistant"


class TestExecuteNode:
    """Tests for ExecuteNode."""

    @pytest.mark.asyncio
    async def test_execute_tool_call(self, sample_tools):
        """ExecuteNode executes tool calls."""
        node = ExecuteNode(registry=sample_tools)

        # Create state with tool call
        tool_call = ToolCall(name="search", arguments={"query": "test"})
        state = AgentState().with_message(
            Message.assistant(content="Searching", tool_calls=[tool_call])
        )

        result = await node.execute(state)

        # Should have start and complete events
        assert len(result.events) == 2
        assert isinstance(result.events[0], ToolStartEvent)
        assert isinstance(result.events[1], ToolCompleteEvent)
        assert result.events[1].tool_name == "search"
        assert "Results for: test" in result.events[1].result

    @pytest.mark.asyncio
    async def test_execute_multiple_tools(self, sample_tools):
        """ExecuteNode can execute multiple tools."""
        node = ExecuteNode(registry=sample_tools)

        tool_calls = [
            ToolCall(name="search", arguments={"query": "test"}),
            ToolCall(name="calculate", arguments={"expression": "2+2"}),
        ]
        state = AgentState().with_message(Message.assistant(tool_calls=tool_calls))

        result = await node.execute(state)

        # 2 start events + 2 complete events
        assert len(result.events) == 4

    @pytest.mark.asyncio
    async def test_execute_records_tool_execution(self, sample_tools):
        """ExecuteNode records tool executions in state."""
        node = ExecuteNode(registry=sample_tools)

        tool_call = ToolCall(name="search", arguments={"query": "test"})
        state = AgentState().with_message(Message.assistant(tool_calls=[tool_call]))

        result = await node.execute(state)

        assert len(result.state.tool_executions) == 1
        assert result.state.tool_executions[0].tool_name == "search"
        assert result.state.tool_executions[0].success

    @pytest.mark.asyncio
    async def test_execute_handles_unknown_tool(self, empty_registry):
        """ExecuteNode handles unknown tools gracefully."""
        node = ExecuteNode(registry=empty_registry)

        tool_call = ToolCall(name="nonexistent", arguments={})
        state = AgentState().with_message(Message.assistant(tool_calls=[tool_call]))

        result = await node.execute(state)

        # Should complete with error
        complete_event = result.events[-1]
        assert isinstance(complete_event, ToolCompleteEvent)
        assert complete_event.error is not None
        assert "Unknown tool" in complete_event.error

    @pytest.mark.asyncio
    async def test_execute_no_tool_calls(self, sample_tools):
        """ExecuteNode handles state with no tool calls."""
        node = ExecuteNode(registry=sample_tools)
        state = AgentState().with_message(Message.assistant(content="Just text, no tools"))

        result = await node.execute(state)

        assert len(result.events) == 0
        assert result.state == state


class TestReflectNode:
    """Tests for ReflectNode."""

    @pytest.mark.asyncio
    async def test_reflect_on_success(self):
        """ReflectNode assesses progress positively on success."""
        node = ReflectNode()

        # State with successful tool execution
        state = (
            AgentState(iteration=1)
            .with_tool_execution(
                ToolExecution(
                    tool_name="search",
                    tool_call_id="call_1",
                    arguments={"query": "test"},
                    result="Found 10 results for your query",
                )
            )
            .with_message(
                Message.assistant(tool_calls=[ToolCall(name="search", arguments={"query": "test"})])
            )
        )

        result = await node.execute(state)

        assert len(result.events) == 1
        event = result.events[0]
        assert isinstance(event, ReflectEvent)
        assert event.assessment in ("on_track", "new_findings")
        assert event.confidence_delta > 0

    @pytest.mark.asyncio
    async def test_reflect_on_error(self):
        """ReflectNode assesses negatively on errors."""
        node = ReflectNode()

        # State with failed tool execution
        state = AgentState(iteration=1).with_tool_execution(
            ToolExecution(
                tool_name="search",
                tool_call_id="call_1",
                arguments={"query": "test"},
                error="Connection failed",
            )
        )

        result = await node.execute(state)

        event = result.events[0]
        assert isinstance(event, ReflectEvent)
        assert event.assessment in ("stuck", "error")
        assert event.confidence_delta < 0

    @pytest.mark.asyncio
    async def test_reflect_on_multiple_errors(self):
        """ReflectNode assesses 'error' when multiple recent errors."""
        node = ReflectNode()

        # State with multiple failed tool executions
        state = AgentState(iteration=2)
        for i in range(3):
            state = state.with_tool_execution(
                ToolExecution(
                    tool_name=f"tool_{i}",
                    tool_call_id=f"call_{i}",
                    arguments={"x": i},
                    error=f"Error {i}",
                )
            )

        result = await node.execute(state)

        event = result.events[0]
        assert isinstance(event, ReflectEvent)
        assert event.assessment == "error"

    @pytest.mark.asyncio
    async def test_reflect_on_success_short_result(self):
        """ReflectNode returns on_track when results are short."""
        node = ReflectNode()

        # State with successful tool execution but short result
        tc = ToolCall(name="check", arguments={})
        state = (
            AgentState(iteration=1)
            .with_tool_execution(
                ToolExecution(
                    tool_name="check",
                    tool_call_id="call_1",
                    arguments={},
                    result="OK",  # Very short result
                )
            )
            .model_copy(update={"last_tool_calls": (tc,)})
        )

        result = await node.execute(state)

        event = result.events[0]
        assert isinstance(event, ReflectEvent)
        assert event.assessment == "on_track"

    @pytest.mark.asyncio
    async def test_reflect_updates_confidence(self):
        """ReflectNode updates state confidence."""
        node = ReflectNode()
        state = (
            AgentState(confidence=0.5, iteration=1)
            .with_tool_execution(
                ToolExecution(
                    tool_name="search",
                    tool_call_id="call_1",
                    arguments={},
                    result="Good results here",
                )
            )
            .with_message(Message.assistant(tool_calls=[ToolCall(name="search", arguments={})]))
        )

        result = await node.execute(state)

        # Confidence should have changed
        assert result.state.confidence != state.confidence

    @pytest.mark.asyncio
    async def test_reflect_detects_loop(self):
        """ReflectNode detects tool loops."""
        node = ReflectNode()

        # Create state with tool loop across iterations
        from locus.core.messages import ToolCall
        from locus.core.state import ReasoningStep

        state = AgentState(tool_loop_threshold=3)
        for i in range(3):
            step = ReasoningStep(
                iteration=i + 1,
                thought=f"Search {i}",
                tool_calls=[ToolCall(name="search", arguments={"query": "test"})],
            )
            state = state.with_reasoning_step(step)
            state = state.with_tool_execution(
                ToolExecution(
                    tool_name="search",
                    tool_call_id=f"call_{i}",
                    arguments={"query": "test"},
                    result="Same result",
                )
            )
            state = state.next_iteration()

        result = await node.execute(state)

        event = result.events[0]
        assert event.assessment == "loop_detected"
        assert event.confidence_delta < 0

    @pytest.mark.asyncio
    async def test_reflect_adds_reasoning_step(self):
        """ReflectNode adds reasoning step to state."""
        node = ReflectNode()
        state = AgentState(iteration=1)

        result = await node.execute(state)

        assert len(result.state.reasoning_steps) == 1
        step = result.state.reasoning_steps[0]
        assert step.iteration == 1


# =============================================================================
# ReActLoop Tests
# =============================================================================


class TestReActLoopConfig:
    """Tests for ReActLoopConfig."""

    def test_default_config(self):
        """Default configuration values."""
        config = ReActLoopConfig()

        assert config.max_iterations == 20
        assert config.confidence_threshold == 0.85
        assert config.enable_reflection is True
        assert config.reflect_interval == 1

    def test_custom_config(self):
        """Custom configuration values."""
        config = ReActLoopConfig(
            max_iterations=10,
            confidence_threshold=0.9,
            enable_reflection=False,
            system_prompt="You are a bot",
        )

        assert config.max_iterations == 10
        assert config.confidence_threshold == 0.9
        assert config.enable_reflection is False
        assert config.system_prompt == "You are a bot"

    def test_config_validation(self):
        """Configuration validates bounds."""
        with pytest.raises(ValueError):
            ReActLoopConfig(max_iterations=0)

        with pytest.raises(ValueError):
            ReActLoopConfig(confidence_threshold=1.5)


class TestReActLoop:
    """Tests for ReActLoop."""

    @pytest.mark.asyncio
    async def test_simple_completion(self, mock_model, empty_registry):
        """Loop completes with simple response."""
        # Model responds without tool calls (triggers termination)
        mock_model.complete.return_value = create_model_response(content="Here is my answer")

        loop = ReActLoop(model=mock_model, registry=empty_registry)

        events = []
        async for event in loop.run("Hello"):
            events.append(event)

        # Should have Think and Terminate events
        assert any(isinstance(e, ThinkEvent) for e in events)
        assert any(isinstance(e, TerminateEvent) for e in events)

    @pytest.mark.asyncio
    async def test_tool_execution_cycle(self, mock_model, sample_tools):
        """Loop executes think -> execute -> reflect cycle."""
        # First call: model requests tool
        tool_call = ToolCall(name="search", arguments={"query": "test"})
        # Subsequent calls: model responds without tools
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return create_model_response(content="Searching", tool_calls=[tool_call])
            return create_model_response(content="Found the answer")

        mock_model.complete.side_effect = side_effect

        loop = ReActLoop(model=mock_model, registry=sample_tools)

        events = []
        async for event in loop.run("Search for test"):
            events.append(event)

        # Should have: Think, ToolStart, ToolComplete, Reflect, Think, Terminate
        event_types = [type(e).__name__ for e in events]
        assert "ThinkEvent" in event_types
        assert "ToolStartEvent" in event_types
        assert "ToolCompleteEvent" in event_types
        assert "TerminateEvent" in event_types

    @pytest.mark.asyncio
    async def test_max_iterations_termination(self, mock_model, sample_tools):
        """Loop terminates at max iterations."""
        # Model always requests tools
        tool_call = ToolCall(name="search", arguments={"query": "test"})
        mock_model.complete.return_value = create_model_response(
            content="Searching", tool_calls=[tool_call]
        )

        config = ReActLoopConfig(max_iterations=2, enable_reflection=False)
        loop = ReActLoop(model=mock_model, registry=sample_tools, config=config)

        events = []
        async for event in loop.run("Keep searching"):
            events.append(event)

        terminate_event = next(e for e in events if isinstance(e, TerminateEvent))
        assert terminate_event.reason == "max_iterations"

    @pytest.mark.asyncio
    async def test_terminal_tool_termination(self, mock_model, sample_tools):
        """Loop terminates when terminal tool is called."""
        tool_call = ToolCall(name="done", arguments={"result": "finished"})
        mock_model.complete.return_value = create_model_response(
            content="Done", tool_calls=[tool_call]
        )

        loop = ReActLoop(model=mock_model, registry=sample_tools)

        events = []
        async for event in loop.run("Complete the task"):
            events.append(event)

        terminate_event = next(e for e in events if isinstance(e, TerminateEvent))
        assert terminate_event.reason == "terminal_tool"

    @pytest.mark.asyncio
    async def test_run_to_completion(self, mock_model, empty_registry):
        """run_to_completion returns state and events."""
        mock_model.complete.return_value = create_model_response(content="Done")

        loop = ReActLoop(model=mock_model, registry=empty_registry)

        state, events = await loop.run_to_completion("Hello")

        assert isinstance(state, AgentState)
        assert len(events) > 0
        assert any(isinstance(e, TerminateEvent) for e in events)

    @pytest.mark.asyncio
    async def test_with_initial_state(self, mock_model, empty_registry):
        """Loop can start with pre-configured state."""
        mock_model.complete.return_value = create_model_response(content="Continuing")

        initial = AgentState(
            iteration=5,
            confidence=0.5,
        ).with_message(Message.system("Previous context"))

        loop = ReActLoop(model=mock_model, registry=empty_registry)

        state, events = await loop.run_to_completion("Continue", initial_state=initial)

        # State should have the initial message plus new ones
        assert len(state.messages) >= 2

    @pytest.mark.asyncio
    async def test_run_generator_with_initial_state(self, mock_model, empty_registry):
        """Loop.run() generator can accept initial_state."""
        mock_model.complete.return_value = create_model_response(content="Done!")

        initial = AgentState(
            iteration=3,
            confidence=0.6,
        )

        loop = ReActLoop(model=mock_model, registry=empty_registry)

        events = []
        async for event in loop.run("Continue from here", initial_state=initial):
            events.append(event)

        assert len(events) > 0

    def test_with_config(self, mock_model, empty_registry):
        """with_config returns new loop with updated config."""
        loop = ReActLoop(model=mock_model, registry=empty_registry)

        new_loop = loop.with_config(max_iterations=5, enable_reflection=False)

        assert loop.config.max_iterations == 20  # Original unchanged
        assert new_loop.config.max_iterations == 5
        assert new_loop.config.enable_reflection is False


class TestCreateReactLoop:
    """Tests for create_react_loop factory function."""

    def test_creates_loop(self, mock_model, sample_tools):
        """Factory creates configured loop."""
        loop = create_react_loop(
            model=mock_model,
            registry=sample_tools,
            max_iterations=10,
            confidence_threshold=0.9,
            enable_reflection=False,
            system_prompt="Be helpful",
        )

        assert isinstance(loop, ReActLoop)
        assert loop.config.max_iterations == 10
        assert loop.config.confidence_threshold == 0.9
        assert loop.config.enable_reflection is False
        assert loop.config.system_prompt == "Be helpful"


# =============================================================================
# Runner Tests
# =============================================================================


class TestLoopRunner:
    """Tests for LoopRunner."""

    @pytest.mark.asyncio
    async def test_run_with_callback(self, mock_model, empty_registry):
        """Runner calls event callback."""
        mock_model.complete.return_value = create_model_response(content="Done")

        loop = ReActLoop(model=mock_model, registry=empty_registry)
        received_events = []

        runner = LoopRunner(
            loop=loop,
            on_event=lambda e: received_events.append(e),
        )

        events = []
        async for event in runner.run("Hello"):
            events.append(event)

        assert len(received_events) == len(events)

    @pytest.mark.asyncio
    async def test_run_with_timeout(self, mock_model, empty_registry):
        """Runner handles timeout."""

        # Make model hang
        async def slow_complete(*args, **kwargs):
            import asyncio

            await asyncio.sleep(10)
            return create_model_response(content="Done")

        mock_model.complete = slow_complete

        loop = ReActLoop(model=mock_model, registry=empty_registry)
        runner = LoopRunner(loop=loop, timeout=0.1)

        events = []
        async for event in runner.run("Hello"):
            events.append(event)

        # Should have timeout termination
        terminate_event = next(e for e in events if isinstance(e, TerminateEvent))
        assert terminate_event.reason == "timeout"


class TestStreamingCollector:
    """Tests for StreamingCollector."""

    def test_collect_events(self):
        """Collector categorizes events."""
        collector = StreamingCollector()

        collector.collect(ThinkEvent(iteration=0, reasoning="Thinking"))
        collector.collect(ToolStartEvent(tool_name="search", tool_call_id="1", arguments={}))
        collector.collect(ToolCompleteEvent(tool_name="search", tool_call_id="1", result="Done"))
        collector.collect(
            ReflectEvent(
                iteration=0, assessment="on_track", confidence_delta=0.1, new_confidence=0.1
            )
        )
        collector.collect(
            TerminateEvent(
                reason="complete", iterations_used=1, final_confidence=0.1, total_tool_calls=1
            )
        )

        assert len(collector.events) == 5
        assert len(collector.think_events) == 1
        assert len(collector.tool_events) == 2
        assert len(collector.reflect_events) == 1
        assert collector.terminate_event is not None

    def test_is_complete(self):
        """Collector tracks completion state."""
        collector = StreamingCollector()

        assert not collector.is_complete

        collector.collect(
            TerminateEvent(
                reason="done", iterations_used=1, final_confidence=0.5, total_tool_calls=0
            )
        )

        assert collector.is_complete
        assert collector.iterations == 1
        assert collector.final_confidence == 0.5

    def test_reset(self):
        """Collector can be reset."""
        collector = StreamingCollector()
        collector.collect(ThinkEvent(iteration=0, reasoning="Test"))
        collector.collect(
            TerminateEvent(
                reason="done", iterations_used=1, final_confidence=0.5, total_tool_calls=0
            )
        )

        collector.reset()

        assert len(collector.events) == 0
        assert not collector.is_complete


class TestBatchRunner:
    """Tests for BatchRunner."""

    @pytest.mark.asyncio
    async def test_run_batch(self, mock_model, empty_registry):
        """BatchRunner processes multiple prompts."""
        mock_model.complete.return_value = create_model_response(content="Response")

        loop = ReActLoop(model=mock_model, registry=empty_registry)
        runner = BatchRunner(loop=loop, max_concurrency=2)

        prompts = ["Hello 1", "Hello 2", "Hello 3"]
        results = await runner.run_batch(prompts)

        assert len(results) == 3
        for prompt, state, events in results:
            assert prompt in prompts
            assert isinstance(state, AgentState)
            assert any(isinstance(e, TerminateEvent) for e in events)


class TestCreateRunner:
    """Tests for create_runner factory function."""

    def test_creates_runner(self, mock_model, sample_tools):
        """Factory creates configured runner."""
        events_received = []

        runner = create_runner(
            model=mock_model,
            registry=sample_tools,
            max_iterations=10,
            timeout=30.0,
            on_event=lambda e: events_received.append(e),
        )

        assert isinstance(runner, LoopRunner)
        assert runner.loop.config.max_iterations == 10
        assert runner.timeout == 30.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestReActLoopIntegration:
    """Integration tests for the full ReAct loop."""

    @pytest.mark.asyncio
    async def test_full_cycle_with_reflection(self, mock_model, sample_tools):
        """Test complete cycle with reflection enabled."""
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First: request search
                return create_model_response(
                    content="I'll search for that",
                    tool_calls=[ToolCall(name="search", arguments={"query": "test"})],
                )
            else:
                # Second: complete
                return create_model_response(
                    content="Based on the search results, here's the answer"
                )

        mock_model.complete.side_effect = side_effect

        loop = create_react_loop(
            model=mock_model,
            registry=sample_tools,
            enable_reflection=True,
        )

        state, events = await loop.run_to_completion("Find information about test")

        # Verify event sequence
        event_types = [e.event_type for e in events]
        assert "think" in event_types
        assert "tool_start" in event_types
        assert "tool_complete" in event_types
        assert "reflect" in event_types
        assert "terminate" in event_types

        # Verify state - tools were executed
        assert len(state.tool_executions) >= 1
        assert len(state.reasoning_steps) >= 1

    @pytest.mark.asyncio
    async def test_loop_with_multiple_tools(self, mock_model, sample_tools):
        """Test loop with multiple tool calls in sequence."""
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return create_model_response(
                    content="First search",
                    tool_calls=[ToolCall(name="search", arguments={"query": "first"})],
                )
            elif call_count == 2:
                return create_model_response(
                    content="Calculate",
                    tool_calls=[ToolCall(name="calculate", arguments={"expression": "2+2"})],
                )
            else:
                return create_model_response(content="Done with both tasks")

        mock_model.complete.side_effect = side_effect

        loop = create_react_loop(
            model=mock_model,
            registry=sample_tools,
            enable_reflection=False,  # Simplify for this test
        )

        state, events = await loop.run_to_completion("Do two things")

        # Should have executed both tools at some point
        tool_names = [e.tool_name for e in state.tool_executions]
        assert "search" in tool_names
        assert "calculate" in tool_names

    @pytest.mark.asyncio
    async def test_confidence_buildup(self, mock_model, sample_tools):
        """Test confidence builds up through successful actions."""
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return create_model_response(
                    content=f"Search {call_count}",
                    tool_calls=[ToolCall(name="search", arguments={"query": f"query{call_count}"})],
                )
            else:
                return create_model_response(content="Final answer")

        mock_model.complete.side_effect = side_effect

        loop = create_react_loop(
            model=mock_model,
            registry=sample_tools,
            enable_reflection=True,
            confidence_threshold=0.99,  # High threshold to ensure multiple iterations
            max_iterations=10,
        )

        state, events = await loop.run_to_completion("Complex task")

        # Confidence should have increased through reflections
        reflect_events = [e for e in events if isinstance(e, ReflectEvent)]
        assert len(reflect_events) >= 1

        # Each successful reflection should increase confidence
        confidences = [e.new_confidence for e in reflect_events]
        if len(confidences) >= 2:
            assert confidences[-1] >= confidences[0]
