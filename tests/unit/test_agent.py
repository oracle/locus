# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for Agent class."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from locus.agent import (
    Agent,
    AgentConfig,
    AgentResult,
    ExecutionMetrics,
    GroundingConfig,
    ReflexionConfig,
)
from locus.core.events import (
    ReflectEvent,
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
)
from locus.core.messages import Message, ToolCall, ToolResult
from locus.core.state import AgentState, ToolExecution
from locus.models.base import ModelResponse
from locus.tools.decorator import tool
from tests._safe_math import safe_math_eval


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_model():
    """Create a mock model."""
    model = MagicMock()
    model.complete = AsyncMock()
    return model


@pytest.fixture
def sample_tool():
    """Create a sample tool."""

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a mathematical expression."""
        return str(safe_math_eval(expression))

    return calculator


@pytest.fixture
def sample_tools():
    """Create multiple sample tools."""

    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Results for: {query}"

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a mathematical expression."""
        return str(safe_math_eval(expression))

    return [search, calculator]


@pytest.fixture
def mock_hook():
    """Create a mock hook."""
    hook = MagicMock()
    hook.on_before_invocation = AsyncMock(side_effect=lambda p, s: s)
    hook.on_after_invocation = AsyncMock()
    hook.on_before_tool_call = AsyncMock(side_effect=lambda event: None)
    hook.on_after_tool_call = AsyncMock(side_effect=lambda event: None)
    hook.on_before_model_call = AsyncMock(side_effect=lambda event: None)
    hook.on_after_model_call = AsyncMock(side_effect=lambda event: None)
    hook.priority = 100
    return hook


# =============================================================================
# Agent Configuration Tests
# =============================================================================


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_create_minimal_config(self):
        """Test creating config with minimal params."""
        config = AgentConfig(model="openai:gpt-4o")
        assert config.model == "openai:gpt-4o"
        assert config.tools == []
        assert config.max_iterations == 20

    def test_create_full_config(self, sample_tools):
        """Test creating config with all params."""
        config = AgentConfig(
            model="openai:gpt-4o",
            tools=sample_tools,
            system_prompt="You are a helpful assistant.",
            max_iterations=50,
            reflexion=ReflexionConfig(confidence_threshold=0.9),
            grounding=GroundingConfig(threshold=0.7),
            terminal_tools={"done", "submit"},
            temperature=0.5,
        )
        assert config.model == "openai:gpt-4o"
        assert len(config.tools) == 2
        assert config.max_iterations == 50
        assert config.reflexion is not None
        assert config.reflexion.confidence_threshold == 0.9
        assert config.grounding is not None
        assert config.grounding.threshold == 0.7

    def test_invalid_model_string(self):
        """Test that invalid model string raises error."""
        with pytest.raises(ValueError, match="must be 'provider:model'"):
            AgentConfig(model="invalid-model")

    def test_with_reflexion(self):
        """Test with_reflexion helper."""
        config = AgentConfig(model="openai:gpt-4o")
        new_config = config.with_reflexion(enabled=True, confidence_threshold=0.8)
        assert new_config.reflexion is not None
        assert new_config.reflexion.enabled is True
        assert new_config.reflexion.confidence_threshold == 0.8

    def test_with_grounding(self):
        """Test with_grounding helper."""
        config = AgentConfig(model="openai:gpt-4o")
        new_config = config.with_grounding(enabled=True, threshold=0.5)
        assert new_config.grounding is not None
        assert new_config.grounding.enabled is True
        assert new_config.grounding.threshold == 0.5


class TestReflexionConfig:
    """Tests for ReflexionConfig."""

    def test_defaults(self):
        """Test default values."""
        config = ReflexionConfig()
        assert config.enabled is True
        assert config.confidence_threshold == 0.85
        assert config.diminishing_returns is True

    def test_custom_values(self):
        """Test custom values."""
        config = ReflexionConfig(
            enabled=False,
            confidence_threshold=0.5,
            diminishing_returns=False,
            evaluate_every_n_iterations=3,
        )
        assert config.enabled is False
        assert config.confidence_threshold == 0.5
        assert config.evaluate_every_n_iterations == 3


class TestGroundingConfig:
    """Tests for GroundingConfig."""

    def test_defaults(self):
        """Test default values."""
        config = GroundingConfig()
        assert config.enabled is True
        assert config.threshold == 0.65
        assert config.max_replans == 2

    def test_custom_values(self):
        """Test custom values."""
        config = GroundingConfig(
            enabled=False,
            threshold=0.8,
            max_replans=5,
        )
        assert config.enabled is False
        assert config.threshold == 0.8
        assert config.max_replans == 5


# =============================================================================
# Agent Initialization Tests
# =============================================================================


class TestAgentInitialization:
    """Tests for Agent initialization."""

    def test_init_with_model_string(self, mock_model, sample_tools, monkeypatch):
        """Test initialization with model string."""
        monkeypatch.setattr("locus.agent.agent.get_model", lambda m: mock_model)

        agent = Agent(
            model="openai:gpt-4o",
            tools=sample_tools,
            system_prompt="Test prompt",
        )

        assert agent.config.model == "openai:gpt-4o"
        assert agent.system_prompt == "Test prompt"
        assert len(agent.tools) == 2

    def test_init_with_model_instance(self, mock_model, sample_tools):
        """Test initialization with model instance."""
        agent = Agent(
            model=mock_model,
            tools=sample_tools,
        )

        assert agent.model == mock_model
        assert len(agent.tools) == 2

    def test_init_with_reflexion_bool(self, mock_model, monkeypatch):
        """Test initialization with reflexion=True."""
        monkeypatch.setattr("locus.agent.agent.get_model", lambda m: mock_model)

        agent = Agent(
            model="openai:gpt-4o",
            reflexion=True,
        )

        assert agent.config.reflexion is not None
        assert agent.config.reflexion.enabled is True

    def test_init_with_grounding_bool(self, mock_model, monkeypatch):
        """Test initialization with grounding=True."""
        monkeypatch.setattr("locus.agent.agent.get_model", lambda m: mock_model)

        agent = Agent(
            model="openai:gpt-4o",
            grounding=True,
        )

        assert agent.config.grounding is not None
        assert agent.config.grounding.enabled is True

    def test_init_with_reflexion_config(self, mock_model, monkeypatch):
        """Test initialization with ReflexionConfig object."""
        monkeypatch.setattr("locus.agent.agent.get_model", lambda m: mock_model)

        config = ReflexionConfig(confidence_threshold=0.75)
        agent = Agent(
            model="openai:gpt-4o",
            reflexion=config,
        )

        assert agent.config.reflexion is not None
        assert agent.config.reflexion.confidence_threshold == 0.75

    def test_init_with_grounding_config(self, mock_model, monkeypatch):
        """Test initialization with GroundingConfig object."""
        monkeypatch.setattr("locus.agent.agent.get_model", lambda m: mock_model)

        config = GroundingConfig(threshold=0.8)
        agent = Agent(
            model="openai:gpt-4o",
            grounding=config,
        )

        assert agent.config.grounding is not None
        assert agent.config.grounding.threshold == 0.8

    def test_init_with_config_object(self, mock_model, sample_tools, monkeypatch):
        """Test initialization with AgentConfig object."""
        monkeypatch.setattr("locus.agent.agent.get_model", lambda m: mock_model)

        config = AgentConfig(
            model="openai:gpt-4o",
            tools=sample_tools,
            max_iterations=10,
        )

        agent = Agent(config=config)

        assert agent.config.max_iterations == 10
        assert len(agent.tools) == 2

    def test_tool_registration(self, mock_model, sample_tools):
        """Test that tools are properly registered."""
        agent = Agent(
            model=mock_model,
            tools=sample_tools,
        )

        assert "search" in agent.tools
        assert "calculator" in agent.tools
        assert agent.tools.get("search") is not None
        assert agent.tools.get("calculator") is not None

    def test_invalid_tool_raises_error(self, mock_model):
        """Test that non-Tool objects raise TypeError."""
        with pytest.raises(TypeError, match="Expected Tool instance"):
            Agent(
                model=mock_model,
                tools=[{"not": "a tool"}],
            )

    # ------ add_tool / add_tools (B2) ---------------------------------------

    def test_add_tool_registers_post_construct(self, mock_model):
        """``Agent.add_tool`` adds a tool to the live registry.

        Regression: mutating ``agent.config.tools`` directly after
        construction is a silent no-op because ``_initialize()`` runs
        in ``__init__`` and compiles the tool list once. Verify that
        the supported API actually attaches the tool to the registry.
        """

        @tool
        def search(query: str) -> str:
            """Search."""
            return f"r:{query}"

        agent = Agent(model=mock_model, tools=[])
        # Pre-condition: registry is empty.
        assert "search" not in agent.tools

        agent.add_tool(search)

        assert "search" in agent.tools
        assert agent.tools.get("search") is not None
        # config is mirrored so a re-init sees the same surface.
        assert search in agent.config.tools

    def test_silent_config_mutation_is_documented_no_op(self, mock_model):
        """Directly assigning to config.tools does NOT update the registry.

        This is the footgun ``add_tool`` exists to solve. Locking the
        existing behaviour in a test prevents accidental regressions
        (and surfaces the design choice if someone tries to "fix" it
        by re-initialising on mutation).
        """

        @tool
        def slow_search(query: str) -> str:
            """Search slowly."""
            return query

        agent = Agent(model=mock_model, tools=[])
        agent.config.tools.append(slow_search)
        # Registry is NOT updated by direct mutation.
        assert "slow_search" not in agent.tools

    def test_add_tool_rejects_non_tool(self, mock_model):
        """Non-Tool input fails fast with TypeError."""
        agent = Agent(model=mock_model, tools=[])
        with pytest.raises(TypeError, match="Expected Tool instance"):
            agent.add_tool({"not": "a tool"})  # type: ignore[arg-type]

    def test_add_tool_rejects_duplicates(self, mock_model):
        """Duplicate names propagate ToolRegistry's ValueError."""

        @tool
        def search(query: str) -> str:
            """Search."""
            return query

        agent = Agent(model=mock_model, tools=[search])
        with pytest.raises(ValueError, match="already registered"):
            agent.add_tool(search)

    def test_add_tools_batch(self, mock_model, sample_tools):
        """``add_tools`` registers each entry in order."""
        agent = Agent(model=mock_model, tools=[])
        agent.add_tools(sample_tools)
        assert "search" in agent.tools
        assert "calculator" in agent.tools

    def test_sequential_executor_config(self, mock_model, sample_tools, monkeypatch):
        """Test sequential executor is used when configured."""
        from locus.tools.executor import SequentialExecutor

        monkeypatch.setattr("locus.agent.agent.get_model", lambda m: mock_model)

        config = AgentConfig(
            model="openai:gpt-4o",
            tools=sample_tools,
            tool_execution="sequential",
        )
        agent = Agent(config=config)

        assert isinstance(agent._executor, SequentialExecutor)


# =============================================================================
# Agent Result Tests
# =============================================================================


class TestAgentResult:
    """Tests for AgentResult."""

    def test_from_state_success(self):
        """Test creating result from successful state."""
        state = AgentState(
            iteration=3,
            confidence=0.9,
        )
        state = state.with_message(Message.system("You are helpful."))
        state = state.with_message(Message.user("Hello"))
        state = state.with_message(Message.assistant("Hi there!"))

        result = AgentResult.from_state(
            state=state,
            stop_reason="complete",
        )

        assert result.success is True
        assert result.message == "Hi there!"
        assert result.stop_reason == "complete"
        assert result.confidence == 0.9
        assert result.iterations == 3

    def test_from_state_error(self):
        """Test creating result from error state."""
        state = AgentState()
        state = state.with_error("Something went wrong")

        result = AgentResult.from_state(
            state=state,
            stop_reason="error",
            error="Something went wrong",
        )

        assert result.success is False
        assert result.stop_reason == "error"
        assert result.error == "Something went wrong"

    def test_computed_fields(self):
        """Test computed fields."""
        state = AgentState(
            iteration=5,
            confidence=0.75,
        )
        state = state.with_message(Message.assistant("Done"))

        result = AgentResult.from_state(
            state=state,
            stop_reason="terminal_tool",
            metrics=ExecutionMetrics(
                iterations=5,
                tool_calls=10,
                tool_errors=2,
            ),
        )

        assert result.success is True
        assert result.iterations == 5
        assert result.metrics.tools_success_rate == 0.8


class TestExecutionMetrics:
    """Tests for ExecutionMetrics."""

    def test_default_values(self):
        """Test default values."""
        metrics = ExecutionMetrics()
        assert metrics.iterations == 0
        assert metrics.tool_calls == 0
        assert metrics.tools_success_rate == 1.0

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = ExecutionMetrics(
            tool_calls=10,
            tool_errors=3,
        )
        assert metrics.tools_success_rate == 0.7

    def test_tokens_per_iteration(self):
        """Test tokens per iteration calculation."""
        metrics = ExecutionMetrics(
            iterations=5,
            total_tokens=1000,
        )
        assert metrics.tokens_per_iteration == 200.0


# =============================================================================
# Agent Run Tests
# =============================================================================


class TestAgentRun:
    """Tests for Agent.run()."""

    @pytest.mark.asyncio
    async def test_simple_completion(self, mock_model, monkeypatch):
        """Test simple completion without tools."""
        monkeypatch.setattr("locus.agent.agent.get_model", lambda m: mock_model)

        # Setup mock response
        mock_model.complete.return_value = ModelResponse(
            message=Message.assistant("Hello! How can I help?"),
            usage={"total_tokens": 100},
            stop_reason="end_turn",
        )

        agent = Agent(
            model="openai:gpt-4o",
            tools=[],
        )

        events = []
        async for event in agent.run("Hi"):
            events.append(event)

        # Should have ThinkEvent and TerminateEvent
        assert len(events) == 2
        assert isinstance(events[0], ThinkEvent)
        assert events[0].reasoning == "Hello! How can I help?"
        assert isinstance(events[1], TerminateEvent)
        assert events[1].reason == "complete"

    @pytest.mark.asyncio
    async def test_tool_execution(self, mock_model, sample_tool, monkeypatch):
        """Test execution with tool call."""
        monkeypatch.setattr("locus.agent.agent.get_model", lambda m: mock_model)

        # First response: tool call
        tool_call = ToolCall(id="call_1", name="calculator", arguments={"expression": "2+2"})
        first_response = ModelResponse(
            message=Message.assistant(
                content="Let me calculate that.",
                tool_calls=[tool_call],
            ),
            usage={"total_tokens": 50},
        )

        # Second response: final answer
        second_response = ModelResponse(
            message=Message.assistant("The result is 4."),
            usage={"total_tokens": 30},
        )

        mock_model.complete.side_effect = [first_response, second_response]

        agent = Agent(
            model="openai:gpt-4o",
            tools=[sample_tool],
        )

        events = []
        async for event in agent.run("What is 2+2?"):
            events.append(event)

        # Should have: ThinkEvent, ToolStartEvent, ToolCompleteEvent, ThinkEvent, TerminateEvent
        event_types = [type(e).__name__ for e in events]
        assert "ThinkEvent" in event_types
        assert "ToolStartEvent" in event_types
        assert "ToolCompleteEvent" in event_types
        assert "TerminateEvent" in event_types

        # Check tool execution
        tool_complete = next(e for e in events if isinstance(e, ToolCompleteEvent))
        assert tool_complete.tool_name == "calculator"
        assert tool_complete.result == "4"
        assert tool_complete.error is None

    @pytest.mark.asyncio
    async def test_max_iterations(self, mock_model, sample_tool, monkeypatch):
        """Test that max_iterations is respected."""
        monkeypatch.setattr("locus.agent.agent.get_model", lambda m: mock_model)

        # Always return a tool call
        tool_call = ToolCall(id="call_1", name="calculator", arguments={"expression": "1+1"})
        mock_model.complete.return_value = ModelResponse(
            message=Message.assistant(content="Thinking...", tool_calls=[tool_call]),
            usage={"total_tokens": 10},
        )

        agent = Agent(
            model="openai:gpt-4o",
            tools=[sample_tool],
            max_iterations=3,
        )

        events = []
        async for event in agent.run("Keep calculating"):
            events.append(event)

        # Should terminate due to max_iterations
        terminate = next(e for e in events if isinstance(e, TerminateEvent))
        assert terminate.reason == "max_iterations"
        assert terminate.iterations_used == 3

    @pytest.mark.asyncio
    async def test_with_hooks(self, mock_model, mock_hook, monkeypatch):
        """Test that hooks are called."""
        monkeypatch.setattr("locus.agent.agent.get_model", lambda m: mock_model)

        mock_model.complete.return_value = ModelResponse(
            message=Message.assistant("Done"),
            usage={"total_tokens": 10},
        )

        agent = Agent(
            model="openai:gpt-4o",
            hooks=[mock_hook],
        )

        events = []
        async for event in agent.run("Test"):
            events.append(event)

        # Verify hooks were called
        mock_hook.on_before_invocation.assert_called_once()
        mock_hook.on_after_invocation.assert_called_once()

    @pytest.mark.asyncio
    async def test_reflexion_integration(self, mock_model, sample_tool, monkeypatch):
        """Test Reflexion integration."""
        monkeypatch.setattr("locus.agent.agent.get_model", lambda m: mock_model)

        # First response: tool call
        tool_call = ToolCall(id="call_1", name="calculator", arguments={"expression": "2+2"})
        first_response = ModelResponse(
            message=Message.assistant(content="Calculating...", tool_calls=[tool_call]),
            usage={"total_tokens": 50},
        )

        # Second response: final answer
        second_response = ModelResponse(
            message=Message.assistant("The result is 4."),
            usage={"total_tokens": 30},
        )

        mock_model.complete.side_effect = [first_response, second_response]

        agent = Agent(
            model="openai:gpt-4o",
            tools=[sample_tool],
            reflexion=True,
        )

        events = []
        async for event in agent.run("Calculate 2+2"):
            events.append(event)

        # Should have a ReflectEvent
        reflect_events = [e for e in events if isinstance(e, ReflectEvent)]
        assert len(reflect_events) > 0
        assert reflect_events[0].assessment in [
            "on_track",
            "new_findings",
            "stuck",
            "loop_detected",
        ]


# =============================================================================
# Agent Sync Run Tests
# =============================================================================


class TestAgentRunSync:
    """Tests for Agent.run_sync() and invoke()."""

    def test_run_sync(self, mock_model, monkeypatch):
        """Test synchronous execution."""
        monkeypatch.setattr("locus.agent.agent.get_model", lambda m: mock_model)

        mock_model.complete.return_value = ModelResponse(
            message=Message.assistant("Hello!"),
            usage={"total_tokens": 50},
        )

        agent = Agent(
            model="openai:gpt-4o",
        )

        result = agent.run_sync("Hi")

        assert isinstance(result, AgentResult)
        assert result.success is True
        assert result.stop_reason == "complete"

    def test_invoke_alias(self, mock_model, monkeypatch):
        """Test that invoke() is an alias for run_sync()."""
        monkeypatch.setattr("locus.agent.agent.get_model", lambda m: mock_model)

        mock_model.complete.return_value = ModelResponse(
            message=Message.assistant("Hello!"),
            usage={"total_tokens": 50},
        )

        agent = Agent(
            model="openai:gpt-4o",
        )

        result = agent.invoke("Hi")

        assert isinstance(result, AgentResult)
        assert result.success is True


# =============================================================================
# Tool Loop Detection Tests
# =============================================================================


class TestToolLoopDetection:
    """Tests for tool loop detection."""

    @pytest.mark.asyncio
    async def test_detects_tool_loop(self, mock_model, sample_tool, monkeypatch):
        """Test that tool loops are detected."""
        monkeypatch.setattr("locus.agent.agent.get_model", lambda m: mock_model)

        # Always return the same tool call
        tool_call = ToolCall(id="call_1", name="calculator", arguments={"expression": "1+1"})
        mock_model.complete.return_value = ModelResponse(
            message=Message.assistant(content="Let me try again...", tool_calls=[tool_call]),
            usage={"total_tokens": 10},
        )

        agent = Agent(
            model="openai:gpt-4o",
            tools=[sample_tool],
            tool_loop_threshold=3,  # Detect after 3 consecutive same tool calls
            max_iterations=10,
        )

        events = []
        async for event in agent.run("Keep trying"):
            events.append(event)

        # Should eventually terminate
        terminate = next(e for e in events if isinstance(e, TerminateEvent))
        # Could be tool_loop or max_iterations depending on timing
        assert terminate.reason in ["tool_loop", "max_iterations"]


# =============================================================================
# Event Streaming Tests
# =============================================================================


class TestEventStreaming:
    """Tests for event streaming."""

    @pytest.mark.asyncio
    async def test_event_order(self, mock_model, sample_tool, monkeypatch):
        """Test that events are emitted in correct order."""
        monkeypatch.setattr("locus.agent.agent.get_model", lambda m: mock_model)

        tool_call = ToolCall(id="call_1", name="calculator", arguments={"expression": "1+1"})
        first_response = ModelResponse(
            message=Message.assistant(content="Calculating...", tool_calls=[tool_call]),
            usage={"total_tokens": 50},
        )
        second_response = ModelResponse(
            message=Message.assistant("Result: 2"),
            usage={"total_tokens": 30},
        )
        mock_model.complete.side_effect = [first_response, second_response]

        agent = Agent(
            model="openai:gpt-4o",
            tools=[sample_tool],
        )

        events = []
        async for event in agent.run("Calculate 1+1"):
            events.append(event)

        # Verify order
        event_types = [type(e).__name__ for e in events]

        # ThinkEvent should come before tool events
        think_idx = event_types.index("ThinkEvent")
        tool_start_idx = event_types.index("ToolStartEvent")
        tool_complete_idx = event_types.index("ToolCompleteEvent")

        assert think_idx < tool_start_idx < tool_complete_idx

        # TerminateEvent should be last
        assert event_types[-1] == "TerminateEvent"

    @pytest.mark.asyncio
    async def test_event_timestamps(self, mock_model, monkeypatch):
        """Test that events have timestamps."""
        monkeypatch.setattr("locus.agent.agent.get_model", lambda m: mock_model)

        mock_model.complete.return_value = ModelResponse(
            message=Message.assistant("Hello!"),
            usage={"total_tokens": 50},
        )

        agent = Agent(
            model="openai:gpt-4o",
        )

        async for event in agent.run("Hi"):
            assert hasattr(event, "timestamp")
            assert event.timestamp is not None


# =============================================================================
# Checkpointer Integration Tests
# =============================================================================


class TestAgentCheckpointer:
    """Tests for agent with checkpointer."""

    @pytest.fixture
    def mock_checkpointer(self):
        """Create mock checkpointer."""
        cp = AsyncMock()
        cp.save = AsyncMock()
        cp.load = AsyncMock(return_value=None)
        return cp

    @pytest.mark.asyncio
    async def test_agent_saves_checkpoint(self, mock_model, mock_checkpointer, monkeypatch):
        """Test agent saves checkpoint after iteration."""
        monkeypatch.setattr("locus.agent.agent.get_model", lambda m: mock_model)

        mock_model.complete.return_value = ModelResponse(
            message=Message.assistant("Done!"),
            usage={"total_tokens": 50},
        )

        agent = Agent(
            model="openai:gpt-4o",
            checkpointer=mock_checkpointer,
            checkpoint_every_n_iterations=1,
        )

        async for _ in agent.run("Do something", thread_id="test-thread"):
            pass

        # Should have saved at least once
        assert mock_checkpointer.save.called

    @pytest.mark.asyncio
    async def test_agent_loads_existing_checkpoint(
        self, mock_model, mock_checkpointer, monkeypatch
    ):
        """Test agent loads existing checkpoint on thread continuation."""
        monkeypatch.setattr("locus.agent.agent.get_model", lambda m: mock_model)

        # Create existing state
        from locus.core.state import AgentState

        existing_state = AgentState(
            run_id="existing",
            messages=[Message.system("Previous context")],
        )
        mock_checkpointer.load.return_value = existing_state

        mock_model.complete.return_value = ModelResponse(
            message=Message.assistant("Continuing..."),
            usage={"total_tokens": 50},
        )

        agent = Agent(
            model="openai:gpt-4o",
            checkpointer=mock_checkpointer,
        )

        async for _ in agent.run("Continue", thread_id="test-thread"):
            pass

        mock_checkpointer.load.assert_called_once_with("test-thread")


# =============================================================================
# Conversation Manager Tests
# =============================================================================


class TestAgentConversationManager:
    """Tests for agent with conversation manager."""

    @pytest.mark.asyncio
    async def test_agent_uses_conversation_manager(self, mock_model, monkeypatch):
        """Test agent uses conversation manager."""
        monkeypatch.setattr("locus.agent.agent.get_model", lambda m: mock_model)

        mock_model.complete.return_value = ModelResponse(
            message=Message.assistant("Managed response"),
            usage={"total_tokens": 50},
        )

        from locus.memory.conversation import SlidingWindowManager

        manager = SlidingWindowManager(window_size=10)

        agent = Agent(
            model="openai:gpt-4o",
            conversation_manager=manager,
        )

        async for _ in agent.run("Test message"):
            pass

        # Model should have been called with messages processed by manager
        assert mock_model.complete.called


# =============================================================================
# Additional Agent Configuration Tests
# =============================================================================


class TestAgentAdditionalConfig:
    """Additional tests for agent configuration."""

    def test_invoke_alias(self, mock_model, monkeypatch):
        """Test invoke is alias for run_sync."""
        monkeypatch.setattr("locus.agent.agent.get_model", lambda m: mock_model)

        mock_model.complete.return_value = ModelResponse(
            message=Message.assistant("Done"),
            usage={"total_tokens": 50},
        )

        agent = Agent(model="openai:gpt-4o")
        result = agent.invoke("Test")

        assert result.message == "Done"

    def test_agent_with_custom_temperature(self, mock_model, monkeypatch):
        """Test agent with custom temperature."""
        monkeypatch.setattr("locus.agent.agent.get_model", lambda m: mock_model)

        mock_model.complete.return_value = ModelResponse(
            message=Message.assistant("Response"),
            usage={"total_tokens": 50},
        )

        agent = Agent(
            model="openai:gpt-4o",
            temperature=0.9,
        )

        _result = agent.run_sync("Test")

        # Verify temperature was passed
        call_kwargs = mock_model.complete.call_args[1]
        assert call_kwargs["temperature"] == 0.9

    def test_agent_with_max_tokens(self, mock_model, monkeypatch):
        """Test agent with max_tokens."""
        monkeypatch.setattr("locus.agent.agent.get_model", lambda m: mock_model)

        mock_model.complete.return_value = ModelResponse(
            message=Message.assistant("Response"),
            usage={"total_tokens": 50},
        )

        agent = Agent(
            model="openai:gpt-4o",
            max_tokens=500,
        )

        _result = agent.run_sync("Test")

        call_kwargs = mock_model.complete.call_args[1]
        assert call_kwargs["max_tokens"] == 500


# =============================================================================
# Tool Execution Error Tests
# =============================================================================


class TestToolExecutionErrors:
    """Tests for tool execution error handling."""

    @pytest.mark.asyncio
    async def test_tool_exception_is_caught(self, mock_model, monkeypatch):
        """Test that exceptions during tool execution are caught."""
        monkeypatch.setattr("locus.agent.agent.get_model", lambda m: mock_model)

        @tool
        async def failing_tool() -> str:
            """A tool that always fails."""
            raise ValueError("Tool execution failed!")

        tool_call = ToolCall(id="call_1", name="failing_tool", arguments={})
        first_response = ModelResponse(
            message=Message.assistant(content="Let me try...", tool_calls=[tool_call]),
            usage={"total_tokens": 50},
        )
        second_response = ModelResponse(
            message=Message.assistant("I see there was an error."),
            usage={"total_tokens": 30},
        )
        mock_model.complete.side_effect = [first_response, second_response]

        agent = Agent(
            model="openai:gpt-4o",
            tools=[failing_tool],
        )

        events = []
        async for event in agent.run("Use the tool"):
            events.append(event)

        # Should have ToolCompleteEvent with error
        tool_complete = next((e for e in events if isinstance(e, ToolCompleteEvent)), None)
        assert tool_complete is not None
        assert tool_complete.error is not None
        assert "Tool execution failed!" in tool_complete.error

    @pytest.mark.asyncio
    async def test_tool_error_count_tracked(self, mock_model, monkeypatch):
        """Test that tool errors are counted in metrics."""
        monkeypatch.setattr("locus.agent.agent.get_model", lambda m: mock_model)

        @tool
        async def failing_tool() -> str:
            """A tool that fails."""
            raise RuntimeError("Error!")

        tool_call = ToolCall(id="call_1", name="failing_tool", arguments={})
        first_response = ModelResponse(
            message=Message.assistant(content="Trying...", tool_calls=[tool_call]),
            usage={"total_tokens": 50},
        )
        second_response = ModelResponse(
            message=Message.assistant("Done."),
            usage={"total_tokens": 30},
        )
        mock_model.complete.side_effect = [first_response, second_response]

        agent = Agent(
            model="openai:gpt-4o",
            tools=[failing_tool],
        )

        events = []
        async for event in agent.run("Use the tool"):
            events.append(event)

        terminate_event = next(e for e in events if isinstance(e, TerminateEvent))
        # The tool error should be counted
        assert terminate_event.total_tool_calls >= 1


# =============================================================================
# Model Error Tests
# =============================================================================


class TestModelErrors:
    """Tests for model error handling."""

    @pytest.mark.asyncio
    async def test_model_exception_is_raised(self, mock_model, monkeypatch):
        """Test that model exceptions are raised and wrapped."""
        monkeypatch.setattr("locus.agent.agent.get_model", lambda m: mock_model)

        mock_model.complete.side_effect = RuntimeError("Model API error")

        agent = Agent(model="openai:gpt-4o")

        async def consume_events():
            async for _event in agent.run("Hi"):
                pass

        with pytest.raises(RuntimeError, match="Model API error"):
            await consume_events()


# =============================================================================
# Agent State Management Tests
# =============================================================================


class TestAgentStateManagement:
    """Tests for agent state management."""

    @pytest.mark.asyncio
    async def test_initial_state_has_system_message(self, mock_model, monkeypatch):
        """Test initial state contains system message."""
        monkeypatch.setattr("locus.agent.agent.get_model", lambda m: mock_model)

        mock_model.complete.return_value = ModelResponse(
            message=Message.assistant("Hello!"),
            usage={"total_tokens": 50},
        )

        agent = Agent(
            model="openai:gpt-4o",
            system_prompt="You are a helpful assistant.",
        )

        events = []
        async for event in agent.run("Hi"):
            events.append(event)

        # Model should receive system message
        call_args = mock_model.complete.call_args[1]
        messages = call_args["messages"]
        assert any(m.role == "system" for m in messages)

    @pytest.mark.asyncio
    async def test_state_tracks_iterations(self, mock_model, sample_tool, monkeypatch):
        """Test that state tracks iterations correctly."""
        monkeypatch.setattr("locus.agent.agent.get_model", lambda m: mock_model)

        tool_call = ToolCall(id="call_1", name="calculator", arguments={"expression": "1+1"})
        first_response = ModelResponse(
            message=Message.assistant(content="Calculating...", tool_calls=[tool_call]),
            usage={"total_tokens": 50},
        )
        second_response = ModelResponse(
            message=Message.assistant("Result: 2"),
            usage={"total_tokens": 30},
        )
        mock_model.complete.side_effect = [first_response, second_response]

        agent = Agent(
            model="openai:gpt-4o",
            tools=[sample_tool],
        )

        events = []
        async for event in agent.run("Calculate 1+1"):
            events.append(event)

        terminate_event = next(e for e in events if isinstance(e, TerminateEvent))
        assert terminate_event.iterations_used >= 1


# =============================================================================
# Terminal Tool Tests
# =============================================================================


class TestTerminalTools:
    """Tests for terminal tool handling."""

    @pytest.mark.asyncio
    async def test_terminal_tool_stops_execution(self, mock_model, monkeypatch):
        """Test that calling a terminal tool stops execution."""
        monkeypatch.setattr("locus.agent.agent.get_model", lambda m: mock_model)

        @tool
        def done(result: str) -> str:
            """Signal completion with result."""
            return result

        tool_call = ToolCall(id="call_1", name="done", arguments={"result": "Task completed"})
        response = ModelResponse(
            message=Message.assistant(content="Finishing...", tool_calls=[tool_call]),
            usage={"total_tokens": 50},
        )
        mock_model.complete.return_value = response

        agent = Agent(
            model="openai:gpt-4o",
            tools=[done],
            terminal_tools={"done"},
        )

        events = []
        async for event in agent.run("Do the task"):
            events.append(event)

        terminate_event = next(e for e in events if isinstance(e, TerminateEvent))
        assert terminate_event.reason == "terminal_tool"


# =============================================================================
# Hook Execution Tests
# =============================================================================


class TestHookExecution:
    """Tests for hook execution during agent run."""

    @pytest.mark.asyncio
    async def test_hooks_are_called(self, mock_model, mock_hook, monkeypatch):
        """Test that hooks are called during agent run."""
        monkeypatch.setattr("locus.agent.agent.get_model", lambda m: mock_model)

        mock_model.complete.return_value = ModelResponse(
            message=Message.assistant("Hello!"),
            usage={"total_tokens": 50},
        )

        agent = Agent(
            model="openai:gpt-4o",
            hooks=[mock_hook],
        )

        async for _ in agent.run("Hi"):
            pass

        # Hooks should have been called
        mock_hook.on_before_invocation.assert_called()
        mock_hook.on_after_invocation.assert_called()

    @pytest.mark.asyncio
    async def test_tool_hooks_are_called(self, mock_model, sample_tool, mock_hook, monkeypatch):
        """Test that tool hooks are called during tool execution."""
        monkeypatch.setattr("locus.agent.agent.get_model", lambda m: mock_model)

        tool_call = ToolCall(id="call_1", name="calculator", arguments={"expression": "1+1"})
        first_response = ModelResponse(
            message=Message.assistant(content="Calculating...", tool_calls=[tool_call]),
            usage={"total_tokens": 50},
        )
        second_response = ModelResponse(
            message=Message.assistant("Result: 2"),
            usage={"total_tokens": 30},
        )
        mock_model.complete.side_effect = [first_response, second_response]

        agent = Agent(
            model="openai:gpt-4o",
            tools=[sample_tool],
            hooks=[mock_hook],
        )

        async for _ in agent.run("Calculate 1+1"):
            pass

        # Tool hooks should have been called
        mock_hook.on_before_tool_call.assert_called()
        mock_hook.on_after_tool_call.assert_called()


# =============================================================================
# Tool Result Truncation Tests
# =============================================================================


class TestToolResultTruncation:
    """Tests for tool result truncation to prevent context window blowup."""

    @pytest.mark.asyncio
    async def test_long_tool_result_truncated(self, mock_model):
        """Tool results exceeding max_tool_result_length are truncated."""

        @tool
        def big_tool() -> str:
            """Returns a very large result."""
            return "x" * 100_000

        # First call: model requests tool, second call: model gives final answer
        first_response = ModelResponse(
            message=Message.assistant(
                "Let me call the tool.",
                tool_calls=[ToolCall(id="call_1", name="big_tool", arguments={})],
            ),
        )
        second_response = ModelResponse(
            message=Message.assistant("Done."),
        )
        mock_model.complete = AsyncMock(side_effect=[first_response, second_response])

        agent = Agent(model=mock_model, tools=[big_tool], max_tool_result_length=1000)

        events = []
        async for event in agent.run("Do it"):
            events.append(event)

        # Find the ToolCompleteEvent
        tool_events = [e for e in events if isinstance(e, ToolCompleteEvent)]
        assert len(tool_events) == 1
        result_content = tool_events[0].result
        assert result_content is not None
        assert len(result_content) < 1200  # 1000 + truncation notice
        assert "[OUTPUT TRUNCATED" in result_content
        assert "100000 chars" in result_content

    @pytest.mark.asyncio
    async def test_short_tool_result_not_truncated(self, mock_model):
        """Tool results under the limit are not modified."""

        @tool
        def small_tool() -> str:
            """Returns a small result."""
            return "small result"

        first_response = ModelResponse(
            message=Message.assistant(
                "Calling tool.",
                tool_calls=[ToolCall(id="call_1", name="small_tool", arguments={})],
            ),
        )
        second_response = ModelResponse(
            message=Message.assistant("Done."),
        )
        mock_model.complete = AsyncMock(side_effect=[first_response, second_response])

        agent = Agent(model=mock_model, tools=[small_tool], max_tool_result_length=32000)

        events = []
        async for event in agent.run("Do it"):
            events.append(event)

        tool_events = [e for e in events if isinstance(e, ToolCompleteEvent)]
        assert len(tool_events) == 1
        assert tool_events[0].result == "small result"
        assert "[OUTPUT TRUNCATED" not in tool_events[0].result

    @pytest.mark.asyncio
    async def test_truncation_disabled_with_zero(self, mock_model):
        """Setting max_tool_result_length=0 disables truncation."""

        @tool
        def big_tool() -> str:
            """Returns a very large result."""
            return "x" * 100_000

        first_response = ModelResponse(
            message=Message.assistant(
                "Calling tool.",
                tool_calls=[ToolCall(id="call_1", name="big_tool", arguments={})],
            ),
        )
        second_response = ModelResponse(
            message=Message.assistant("Done."),
        )
        mock_model.complete = AsyncMock(side_effect=[first_response, second_response])

        agent = Agent(model=mock_model, tools=[big_tool], max_tool_result_length=0)

        events = []
        async for event in agent.run("Do it"):
            events.append(event)

        tool_events = [e for e in events if isinstance(e, ToolCompleteEvent)]
        assert len(tool_events) == 1
        assert len(tool_events[0].result) == 100_000
        assert "[OUTPUT TRUNCATED" not in tool_events[0].result

    def test_config_default(self):
        """Default max_tool_result_length is 32000."""
        config = AgentConfig(model="openai:gpt-4o")
        assert config.max_tool_result_length == 32000


# =============================================================================
# Message Validation Tests
# =============================================================================


class TestMessageValidation:
    """Tests for message validation / orphan cleanup."""

    def test_valid_messages_unchanged(self):
        """Well-formed message sequences pass through unchanged."""
        messages = [
            Message.system("You are helpful."),
            Message.user("Hello"),
            Message.assistant(
                "Let me search.",
                tool_calls=[ToolCall(id="tc_1", name="search", arguments={"q": "test"})],
            ),
            Message.tool(ToolResult(tool_call_id="tc_1", name="search", content="found it")),
            Message.assistant("Here are the results."),
        ]
        result = Agent._validate_messages(messages)
        assert len(result) == 5

    def test_orphaned_tool_call_removed(self):
        """Assistant message with tool_call but no matching tool result is cleaned."""
        messages = [
            Message.system("You are helpful."),
            Message.user("Hello"),
            Message.assistant(
                "Let me search.",
                tool_calls=[ToolCall(id="tc_orphan", name="search", arguments={})],
            ),
            # No tool result for tc_orphan
            Message.assistant("Never mind."),
        ]
        result = Agent._validate_messages(messages)
        # system + user + assistant (text-only, tool calls stripped) + assistant
        assert len(result) == 4
        # The tool_calls should be stripped from the orphaned message
        assistant_msgs = [m for m in result if m.role.value == "assistant"]
        assert len(assistant_msgs[0].tool_calls) == 0  # tool calls removed
        assert assistant_msgs[0].content == "Let me search."  # content preserved

    def test_orphaned_tool_result_removed(self):
        """Tool result without matching tool_call is removed."""
        messages = [
            Message.system("You are helpful."),
            Message.user("Hello"),
            Message.tool(
                ToolResult(tool_call_id="tc_nonexistent", name="search", content="result")
            ),
            Message.assistant("Done."),
        ]
        result = Agent._validate_messages(messages)
        # Tool result should be dropped, rest kept
        assert len(result) == 3  # system + user + assistant
        assert all(m.role.value != "tool" for m in result)

    def test_partial_orphan_keeps_valid(self):
        """When some tool calls have results and some don't, keep only valid pairs."""
        messages = [
            Message.system("System."),
            Message.user("Hi"),
            Message.assistant(
                "Calling two tools.",
                tool_calls=[
                    ToolCall(id="tc_valid", name="search", arguments={}),
                    ToolCall(id="tc_orphan", name="calc", arguments={}),
                ],
            ),
            Message.tool(ToolResult(tool_call_id="tc_valid", name="search", content="found")),
            # No result for tc_orphan
            Message.assistant("Done."),
        ]
        result = Agent._validate_messages(messages)
        # Assistant message should keep only tc_valid
        assistant_with_tools = [m for m in result if m.role.value == "assistant" and m.tool_calls]
        assert len(assistant_with_tools) == 1
        assert len(assistant_with_tools[0].tool_calls) == 1
        assert assistant_with_tools[0].tool_calls[0].id == "tc_valid"

    def test_empty_messages(self):
        """Empty message list returns empty."""
        assert Agent._validate_messages([]) == []

    def test_no_tool_messages(self):
        """Messages without tools pass through unchanged."""
        messages = [
            Message.system("System."),
            Message.user("Hi"),
            Message.assistant("Hello!"),
        ]
        result = Agent._validate_messages(messages)
        assert len(result) == 3


# =============================================================================
# Malformed Tool Call Recovery Tests
# =============================================================================


class TestMalformedToolCallRecovery:
    """Tests for parsing tool calls from model text output."""

    def _make_agent_with_tools(self, mock_model):
        """Create an agent with registered tools for testing _parse_text_tool_calls."""

        @tool
        def search_web(query: str) -> str:
            """Search the web."""
            return f"Results for: {query}"

        @tool
        def get_weather(city: str, units: str = "celsius") -> str:
            """Get weather for a city."""
            return f"Weather in {city}"

        agent = Agent(model=mock_model, tools=[search_web, get_weather])
        return agent

    def test_parse_simple_tool_call(self, mock_model):
        """Parse a simple tool call from text."""
        agent = self._make_agent_with_tools(mock_model)
        result = agent._parse_text_tool_calls('I will search_web(query="python notebooks")')
        assert len(result) == 1
        assert result[0].name == "search_web"
        assert result[0].arguments["query"] == "python notebooks"

    def test_parse_single_quotes(self, mock_model):
        """Parse tool call with single-quoted arguments."""
        agent = self._make_agent_with_tools(mock_model)
        result = agent._parse_text_tool_calls("search_web(query='hello world')")
        assert len(result) == 1
        assert result[0].arguments["query"] == "hello world"

    def test_parse_multiple_args(self, mock_model):
        """Parse tool call with multiple arguments."""
        agent = self._make_agent_with_tools(mock_model)
        result = agent._parse_text_tool_calls('get_weather(city="London", units="fahrenheit")')
        assert len(result) == 1
        assert result[0].name == "get_weather"
        assert result[0].arguments["city"] == "London"
        assert result[0].arguments["units"] == "fahrenheit"

    def test_parse_case_insensitive(self, mock_model):
        """Tool name matching is case-insensitive."""
        agent = self._make_agent_with_tools(mock_model)
        result = agent._parse_text_tool_calls('SearchWeb(query="test")')
        assert len(result) == 1
        assert result[0].name == "search_web"  # Resolved to real name

    def test_parse_ignores_unknown_tools(self, mock_model):
        """Unknown tool names are ignored."""
        agent = self._make_agent_with_tools(mock_model)
        result = agent._parse_text_tool_calls('unknown_tool(arg="test")')
        assert len(result) == 0

    def test_parse_no_tool_calls_in_text(self, mock_model):
        """Regular text without tool patterns returns empty."""
        agent = self._make_agent_with_tools(mock_model)
        result = agent._parse_text_tool_calls("Just a normal response with no tool calls.")
        assert len(result) == 0

    def test_parse_empty_text(self, mock_model):
        """Empty text returns empty."""
        agent = self._make_agent_with_tools(mock_model)
        assert agent._parse_text_tool_calls("") == []
        assert agent._parse_text_tool_calls(None) == []

    @pytest.mark.asyncio
    async def test_recovery_executes_parsed_tool(self, mock_model):
        """End-to-end: model outputs text tool call, agent parses and executes it."""

        @tool
        def calculator(expression: str) -> str:
            """Calculate."""
            return "42"

        # First response: model outputs tool call as TEXT (no structured tool_calls)
        first_response = ModelResponse(
            message=Message.assistant('I need to calculate. calculator(expression="6*7")'),
        )
        # Second response: after tool execution, model gives final answer
        second_response = ModelResponse(
            message=Message.assistant("The answer is 42."),
        )
        mock_model.complete = AsyncMock(side_effect=[first_response, second_response])

        agent = Agent(model=mock_model, tools=[calculator])

        events = []
        async for event in agent.run("What is 6*7?"):
            events.append(event)

        # Should have executed the tool (parsed from text)
        tool_events = [e for e in events if isinstance(e, ToolCompleteEvent)]
        assert len(tool_events) == 1
        assert tool_events[0].tool_name == "calculator"
        assert tool_events[0].result == "42"

        # Should terminate with final answer
        term_events = [e for e in events if isinstance(e, TerminateEvent)]
        assert len(term_events) == 1
        assert term_events[0].reason == "complete"
        assert "42" in term_events[0].final_message


# =============================================================================
# Auto Conversation Manager Tests
# =============================================================================


class TestAutoConversationManager:
    """Tests for automatic conversation manager creation."""

    def test_auto_created_for_default_config(self, mock_model):
        """SlidingWindowManager auto-created when max_iterations > 10."""
        agent = Agent(model=mock_model, tools=[])
        assert agent._conversation_manager is not None
        from locus.memory.conversation import SlidingWindowManager

        assert isinstance(agent._conversation_manager, SlidingWindowManager)
        assert agent._conversation_manager.window_size == 40  # max(20, 20*2)

    def test_auto_created_with_large_iterations(self, mock_model):
        """Window scales with max_iterations."""
        agent = Agent(model=mock_model, tools=[], max_iterations=100)
        assert agent._conversation_manager is not None
        from locus.memory.conversation import SlidingWindowManager

        assert isinstance(agent._conversation_manager, SlidingWindowManager)
        assert agent._conversation_manager.window_size == 200  # max(20, 100*2)

    def test_not_created_for_small_iterations(self, mock_model):
        """No auto-manager when max_iterations <= 10."""
        agent = Agent(model=mock_model, tools=[], max_iterations=5)
        assert agent._conversation_manager is None

    def test_explicit_manager_used(self, mock_model):
        """Explicit conversation_manager overrides auto-creation."""
        from locus.memory.conversation import NullManager

        null_mgr = NullManager()
        agent = Agent(model=mock_model, tools=[], conversation_manager=null_mgr)
        assert agent._conversation_manager is null_mgr

    @pytest.mark.asyncio
    async def test_sliding_window_trims_long_conversations(self, mock_model):
        """Auto-manager keeps agent working through many iterations."""

        @tool
        def step_a() -> str:
            """Step A."""
            return "ok a"

        @tool
        def step_b() -> str:
            """Step B."""
            return "ok b"

        call_count = 0

        async def multi_turn_complete(messages, tools=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 5:
                # Alternate tools to avoid tool loop detection
                t = "step_a" if call_count % 2 == 1 else "step_b"
                return ModelResponse(
                    message=Message.assistant(
                        f"Turn {call_count}.",
                        tool_calls=[ToolCall(id=f"c{call_count}", name=t, arguments={})],
                    ),
                )
            return ModelResponse(message=Message.assistant("Done after many turns."))

        mock_model.complete = multi_turn_complete

        agent = Agent(
            model=mock_model, tools=[step_a, step_b], max_iterations=10, max_tool_result_length=0
        )

        events = []
        async for event in agent.run("Do 5 turns"):
            events.append(event)

        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None
        assert terminate.reason == "complete"
        assert call_count == 6


# =============================================================================
# SummarizingManager Async Tests
# =============================================================================


class TestSummarizingManagerAsync:
    """Tests for async_apply on SummarizingManager."""

    @pytest.mark.asyncio
    async def test_async_apply_with_async_summarize_fn(self):
        """async_apply calls async summarize_fn properly."""
        from locus.memory.conversation import SummarizingManager

        summarize_called = False

        async def mock_summarize(messages):
            nonlocal summarize_called
            summarize_called = True
            return f"Summary of {len(messages)} messages"

        manager = SummarizingManager(threshold=5, keep_recent=2, summarize_fn=mock_summarize)
        messages = [Message.user(f"Message {i}") for i in range(10)]

        result = await manager.async_apply(messages)

        assert summarize_called
        assert len(result) == 3  # summary + 2 recent
        assert "Summary of 8 messages" in result[0].content

    @pytest.mark.asyncio
    async def test_async_apply_under_threshold(self):
        """async_apply returns all messages when under threshold."""
        from locus.memory.conversation import SummarizingManager

        manager = SummarizingManager(threshold=20, keep_recent=5)
        messages = [Message.user(f"Msg {i}") for i in range(3)]

        result = await manager.async_apply(messages)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_async_apply_fallback_to_sync(self):
        """async_apply falls back to sync when no async summarize_fn."""
        from locus.memory.conversation import SummarizingManager

        manager = SummarizingManager(threshold=5, keep_recent=2)
        messages = [Message.user(f"Message {i}") for i in range(10)]

        result = await manager.async_apply(messages)
        assert len(result) == 3
        assert "[Summary of previous conversation" in result[0].content


# =============================================================================
# Real Grounding Tests
# =============================================================================


class TestRealGrounding:
    """Tests for real GroundingEvaluator integration."""

    def test_extract_claims(self):
        """Claims extracted from response text correctly."""
        from locus.agent.agent import Agent

        response = (
            "Python is a popular programming language. "
            "It was created by Guido van Rossum in 1991. "
            "How does it work? "
            "I think it's great. "
            "The language supports multiple programming paradigms."
        )
        claims = Agent._extract_claims(response)
        # Should exclude questions and "I think" statements
        assert len(claims) >= 2
        assert any("Python" in c for c in claims)
        assert not any(c.endswith("?") for c in claims)

    def test_gather_evidence(self):
        """Evidence gathered from tool executions correctly."""
        from locus.agent.agent import Agent

        state = AgentState()
        state = state.with_tool_execution(
            ToolExecution(
                tool_name="search",
                tool_call_id="c1",
                arguments={"q": "test"},
                result="Found: Python is a programming language created in 1991.",
            )
        )
        state = state.with_tool_execution(
            ToolExecution(
                tool_name="search",
                tool_call_id="c2",
                arguments={"q": "test2"},
                result=None,
                error="Not found",
            )
        )
        evidence = Agent._gather_evidence(state)
        assert len(evidence) == 1  # Only successful execution
        assert "[search]" in evidence[0]
        assert "Python" in evidence[0]

    @pytest.mark.asyncio
    async def test_grounding_not_active_when_disabled(self, mock_model):
        """No grounding events when grounding is disabled."""

        @tool
        def lookup(q: str) -> str:
            """Lookup."""
            return "data"

        mock_model.complete = AsyncMock(
            side_effect=[
                ModelResponse(
                    message=Message.assistant(
                        "Looking up.",
                        tool_calls=[ToolCall(id="c1", name="lookup", arguments={"q": "test"})],
                    ),
                ),
                ModelResponse(
                    message=Message.assistant(
                        "The answer based on my research is that Python is great and widely used in many applications."
                    )
                ),
            ]
        )

        agent = Agent(model=mock_model, tools=[lookup], max_iterations=5)  # No grounding

        events = []
        async for event in agent.run("Tell me about Python"):
            events.append(event)

        from locus.core.events import GroundingEvent

        grounding_events = [e for e in events if isinstance(e, GroundingEvent)]
        assert len(grounding_events) == 0

    @pytest.mark.asyncio
    async def test_grounding_runs_before_final_response(self, mock_model):
        """Grounding evaluator runs when grounding is enabled."""

        @tool
        def research(topic: str) -> str:
            """Research a topic."""
            return f"Detailed research about {topic}: it is very important and widely used."

        call_count = 0

        async def grounding_model(messages, tools=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    message=Message.assistant(
                        "Researching.",
                        tool_calls=[ToolCall(id="c1", name="research", arguments={"topic": "AI"})],
                    ),
                )
            if call_count == 2:
                return ModelResponse(
                    message=Message.assistant(
                        "Based on my research, AI is very important and widely used in many industries today."
                    ),
                )
            # Grounding judge call — return evaluation
            return ModelResponse(
                message=Message.assistant("CLAIM 1: 0.9 - Supported by research tool output"),
            )

        mock_model.complete = grounding_model

        from locus.agent import GroundingConfig

        agent = Agent(
            model=mock_model,
            tools=[research],
            grounding=GroundingConfig(enabled=True, threshold=0.3),
            max_iterations=5,
        )

        events = []
        async for event in agent.run("Tell me about AI"):
            events.append(event)

        from locus.core.events import GroundingEvent

        grounding_events = [e for e in events if isinstance(e, GroundingEvent)]
        assert len(grounding_events) >= 1
        assert grounding_events[0].claims_evaluated >= 1


# =============================================================================
# Real Reflector Tests
# =============================================================================


class TestRealReflector:
    """Tests for real Reflector integration (replaces fake reflexion)."""

    @pytest.mark.asyncio
    async def test_reflector_detects_loop_and_injects_guidance(self, mock_model):
        """When agent repeats the same tool across iterations, loop is detected.

        Note: state.has_tool_loop catches the loop at the top of the next
        iteration (via should_terminate), so the agent terminates with
        reason='tool_loop'. The Reflector may or may not flag it first
        depending on timing.
        """

        @tool
        def search(query: str) -> str:
            """Search."""
            return "same result"

        call_count = 0

        async def looping_model(messages, tools=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 5:
                return ModelResponse(
                    message=Message.assistant(
                        f"Searching again {call_count}.",
                        tool_calls=[
                            ToolCall(
                                id=f"c{call_count}", name="search", arguments={"query": "test"}
                            )
                        ],
                    ),
                )
            return ModelResponse(message=Message.assistant("Done."))

        mock_model.complete = looping_model

        from locus.agent import ReflexionConfig

        agent = Agent(
            model=mock_model,
            tools=[search],
            reflexion=ReflexionConfig(enabled=True, include_guidance=True),
            max_iterations=10,
        )

        events = []
        async for event in agent.run("Search for something"):
            events.append(event)

        # Should terminate due to tool_loop (detected by state.has_tool_loop)
        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None
        assert terminate.reason == "tool_loop"

        # Should have had reflection events during the run
        from locus.core.events import ReflectEvent

        reflect_events = [e for e in events if isinstance(e, ReflectEvent)]
        assert len(reflect_events) >= 1

    @pytest.mark.asyncio
    async def test_reflector_on_track_with_good_results(self, mock_model):
        """Agent making progress gets on_track assessment."""

        @tool
        def step_a(query: str) -> str:
            """Step A."""
            return "Found important information about the topic with detailed analysis " * 5

        @tool
        def step_b() -> str:
            """Step B."""
            return "Additional findings confirming the hypothesis with evidence " * 5

        call_count = 0

        async def progressing_model(messages, tools=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    message=Message.assistant(
                        "Researching.",
                        tool_calls=[ToolCall(id="c1", name="step_a", arguments={"query": "test"})],
                    ),
                )
            if call_count == 2:
                return ModelResponse(
                    message=Message.assistant(
                        "More research.",
                        tool_calls=[ToolCall(id="c2", name="step_b", arguments={})],
                    ),
                )
            return ModelResponse(message=Message.assistant("All done with findings."))

        mock_model.complete = progressing_model

        from locus.agent import ReflexionConfig

        agent = Agent(
            model=mock_model,
            tools=[step_a, step_b],
            reflexion=ReflexionConfig(enabled=True),
            max_iterations=10,
        )

        events = []
        async for event in agent.run("Research topic"):
            events.append(event)

        from locus.core.events import ReflectEvent

        reflect_events = [e for e in events if isinstance(e, ReflectEvent)]
        assert len(reflect_events) >= 1
        # With good results (long content), should be new_findings or on_track
        assert reflect_events[0].assessment in ("on_track", "new_findings")
        assert reflect_events[0].confidence_delta >= 0

    @pytest.mark.asyncio
    async def test_reflector_not_active_when_disabled(self, mock_model):
        """No reflection events when reflexion is disabled."""

        @tool
        def noop() -> str:
            """Noop."""
            return "ok"

        mock_model.complete = AsyncMock(
            side_effect=[
                ModelResponse(
                    message=Message.assistant(
                        "Calling.",
                        tool_calls=[ToolCall(id="c1", name="noop", arguments={})],
                    ),
                ),
                ModelResponse(message=Message.assistant("Done.")),
            ]
        )

        agent = Agent(model=mock_model, tools=[noop], max_iterations=5)  # No reflexion

        events = []
        async for event in agent.run("Do it"):
            events.append(event)

        from locus.core.events import ReflectEvent

        reflect_events = [e for e in events if isinstance(e, ReflectEvent)]
        assert len(reflect_events) == 0

    @pytest.mark.asyncio
    async def test_guidance_injected_when_findings_made(self, mock_model):
        """Reflector guidance appears in events when agent makes findings."""

        @tool
        def search(query: str) -> str:
            """Search."""
            return "Important finding: detailed analysis of the topic " * 5

        @tool
        def analyze(data: str) -> str:
            """Analyze."""
            return "Analysis complete: 3 key insights discovered " * 5

        call_count = 0
        tools_cycle = ["search", "analyze", "search"]

        async def progressing_model(messages, tools=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                t = tools_cycle[call_count - 1]
                return ModelResponse(
                    message=Message.assistant(
                        f"Step {call_count}.",
                        tool_calls=[
                            ToolCall(
                                id=f"c{call_count}",
                                name=t,
                                arguments={"query": f"topic{call_count}"}
                                if t == "search"
                                else {"data": f"data{call_count}"},
                            )
                        ],
                    ),
                )
            return ModelResponse(message=Message.assistant("Done with all research."))

        mock_model.complete = progressing_model

        from locus.agent import ReflexionConfig

        agent = Agent(
            model=mock_model,
            tools=[search, analyze],
            reflexion=ReflexionConfig(enabled=True, include_guidance=True),
            max_iterations=10,
        )

        events = []
        async for event in agent.run("Research topics"):
            events.append(event)

        from locus.core.events import ReflectEvent

        reflect_events = [e for e in events if isinstance(e, ReflectEvent)]
        # Should have reflection events with new_findings assessment
        assert len(reflect_events) >= 1
        # With substantial tool results, should get new_findings or on_track
        assessments = {e.assessment for e in reflect_events}
        assert assessments.issubset({"on_track", "new_findings", "stuck", "loop_detected"})

        # Agent should complete normally (different queries = no loop)
        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None
        assert terminate.reason == "complete"


# =============================================================================
# Graceful Max-Iterations Tests
# =============================================================================


class TestGracefulMaxIterations:
    """Tests for graceful summary on max_iterations."""

    @pytest.mark.asyncio
    async def test_summary_on_max_iterations(self, mock_model):
        """Agent produces summary instead of bare stop on max_iterations."""

        @tool
        def research(topic: str) -> str:
            """Research."""
            return f"Data about {topic}"

        call_count = 0

        async def always_tool_model(messages, tools=None, **kwargs):
            nonlocal call_count
            call_count += 1
            # Check if we got the summary request
            has_summary_request = any(
                m.content and "[Iteration Limit Reached]" in m.content
                for m in messages
                if m.role.value == "system"
            )
            if has_summary_request:
                return ModelResponse(
                    message=Message.assistant(
                        "Based on my research, here is a summary of all findings."
                    ),
                )
            # Always call a tool (will hit max_iterations)
            t = "research"
            return ModelResponse(
                message=Message.assistant(
                    f"Researching turn {call_count}.",
                    tool_calls=[
                        ToolCall(
                            id=f"c{call_count}", name=t, arguments={"topic": f"topic{call_count}"}
                        )
                    ],
                ),
            )

        mock_model.complete = always_tool_model

        agent = Agent(model=mock_model, tools=[research], max_iterations=3)

        events = []
        async for event in agent.run("Research everything"):
            events.append(event)

        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None
        assert terminate.reason == "max_iterations"
        # Should have a summary as final message (not None)
        assert terminate.final_message is not None
        assert (
            "summary" in terminate.final_message.lower()
            or "findings" in terminate.final_message.lower()
        )

    @pytest.mark.asyncio
    async def test_other_stop_reasons_unaffected(self, mock_model):
        """Non-max_iterations stops still work normally (no grace iteration)."""

        @tool
        def done_tool() -> str:
            """Signal completion."""
            return "done"

        mock_model.complete = AsyncMock(
            side_effect=[
                ModelResponse(
                    message=Message.assistant(
                        "Finishing.",
                        tool_calls=[ToolCall(id="c1", name="done_tool", arguments={})],
                    ),
                ),
                ModelResponse(message=Message.assistant("All done.")),
            ]
        )

        agent = Agent(
            model=mock_model,
            tools=[done_tool],
            max_iterations=20,
        )

        events = []
        async for event in agent.run("Do something"):
            events.append(event)

        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None
        assert terminate.reason == "complete"  # Not max_iterations


# =============================================================================
# Time Budget Tests
# =============================================================================


class TestTimeBudget:
    """Tests for time budget enforcement."""

    @pytest.mark.asyncio
    async def test_time_budget_terminates(self, mock_model):
        """Agent stops when time budget is exceeded."""
        import asyncio

        @tool
        def slow_tool(query: str) -> str:
            """A slow tool."""
            return "result"

        call_count = 0

        async def slow_complete(messages, tools=None, **kwargs):
            nonlocal call_count
            call_count += 1
            # First call: trigger tool, with a delay
            if call_count == 1:
                await asyncio.sleep(0.3)  # Burn time
                return ModelResponse(
                    message=Message.assistant(
                        "Searching.",
                        tool_calls=[
                            ToolCall(id="c1", name="slow_tool", arguments={"query": "test"})
                        ],
                    ),
                )
            # Subsequent calls: keep calling tools to burn more time
            await asyncio.sleep(0.3)
            return ModelResponse(
                message=Message.assistant(
                    "More searching.",
                    tool_calls=[
                        ToolCall(id=f"c{call_count}", name="slow_tool", arguments={"query": "test"})
                    ],
                ),
            )

        mock_model.complete = slow_complete

        agent = Agent(
            model=mock_model,
            tools=[slow_tool],
            max_iterations=20,
            time_budget_seconds=0.5,  # 500ms — should stop after 1-2 iterations
        )

        events = []
        async for event in agent.run("Search a lot"):
            events.append(event)

        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None
        assert terminate.reason == "time_budget"
        assert terminate.iterations_used < 20  # Stopped early

    @pytest.mark.asyncio
    async def test_no_time_budget_runs_normally(self, mock_model):
        """Without time_budget, agent runs to completion."""

        @tool
        def fast_tool() -> str:
            """Fast tool."""
            return "done"

        first = ModelResponse(
            message=Message.assistant(
                "Calling tool.",
                tool_calls=[ToolCall(id="c1", name="fast_tool", arguments={})],
            ),
        )
        second = ModelResponse(message=Message.assistant("All done."))
        mock_model.complete = AsyncMock(side_effect=[first, second])

        agent = Agent(model=mock_model, tools=[fast_tool])  # No time_budget

        events = []
        async for event in agent.run("Do it"):
            events.append(event)

        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None
        assert terminate.reason == "complete"


# =============================================================================
# Fix run_sync State Preservation Tests
# =============================================================================


class TestRunSyncStatePreservation:
    """Tests for run_sync preserving actual final state."""

    def test_run_sync_preserves_tool_executions(self, mock_model):
        """run_sync result contains actual tool executions from the run."""

        @tool
        def calc(expr: str) -> str:
            """Calculate."""
            return "42"

        mock_model.complete = AsyncMock(
            side_effect=[
                ModelResponse(
                    message=Message.assistant(
                        "Calculating.",
                        tool_calls=[ToolCall(id="c1", name="calc", arguments={"expr": "6*7"})],
                    ),
                ),
                ModelResponse(message=Message.assistant("The answer is 42.")),
            ]
        )

        agent = Agent(model=mock_model, tools=[calc], max_iterations=5)
        result = agent.run_sync("What is 6*7?")

        assert result.stop_reason == "complete"
        assert result.message == "The answer is 42."
        assert len(result.tool_executions) == 1
        assert result.tool_executions[0].tool_name == "calc"
        assert result.tool_executions[0].result == "42"

    def test_run_sync_preserves_metrics(self, mock_model):
        """run_sync metrics reflect actual execution."""

        @tool
        def step() -> str:
            """Step."""
            return "done"

        mock_model.complete = AsyncMock(
            side_effect=[
                ModelResponse(
                    message=Message.assistant(
                        "Step 1.",
                        tool_calls=[ToolCall(id="c1", name="step", arguments={})],
                    ),
                    usage={"prompt_tokens": 100, "completion_tokens": 50},
                ),
                ModelResponse(
                    message=Message.assistant("All done."),
                    usage={"prompt_tokens": 200, "completion_tokens": 30},
                ),
            ]
        )

        agent = Agent(model=mock_model, tools=[step], max_iterations=5)
        result = agent.run_sync("Do it")

        assert result.metrics.iterations >= 1
        assert result.metrics.tool_calls == 1
        assert result.metrics.total_tokens == 380
        assert result.metrics.duration_ms > 0

    def test_run_sync_preserves_confidence(self, mock_model):
        """run_sync state has correct confidence when reflexion is used."""
        from locus.agent import ReflexionConfig

        @tool
        def research(q: str) -> str:
            """Research."""
            return "Important findings about the topic with detailed analysis " * 5

        mock_model.complete = AsyncMock(
            side_effect=[
                ModelResponse(
                    message=Message.assistant(
                        "Researching.",
                        tool_calls=[ToolCall(id="c1", name="research", arguments={"q": "test"})],
                    ),
                ),
                ModelResponse(message=Message.assistant("Done with findings.")),
            ]
        )

        agent = Agent(
            model=mock_model,
            tools=[research],
            reflexion=ReflexionConfig(enabled=True),
            max_iterations=5,
        )
        result = agent.run_sync("Research something")

        assert result.confidence > 0.0


# =============================================================================
# Completion Mode Tests
# =============================================================================


class TestCompletionMode:
    """Tests for explicit completion mode."""

    @pytest.mark.asyncio
    async def test_explicit_mode_ignores_confidence(self, mock_model):
        """Agent in explicit mode keeps going even when confidence=1.0."""

        @tool
        def work() -> str:
            """Do work."""
            return "work done " * 50  # Lots of content to boost confidence

        call_count = 0

        async def persistent_model(messages, tools=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return ModelResponse(
                    message=Message.assistant(
                        f"Working step {call_count}.",
                        tool_calls=[ToolCall(id=f"c{call_count}", name="work", arguments={})],
                    ),
                )
            # Call task_complete on step 4
            return ModelResponse(
                message=Message.assistant(
                    "All done.",
                    tool_calls=[
                        ToolCall(
                            id="cdone",
                            name="task_complete",
                            arguments={"summary": "All 3 steps done", "status": "success"},
                        )
                    ],
                ),
            )

        mock_model.complete = persistent_model

        from locus.agent import ReflexionConfig

        agent = Agent(
            model=mock_model,
            tools=[work],
            completion_mode="explicit",
            reflexion=ReflexionConfig(enabled=True),
            max_iterations=10,
        )

        events = []
        async for event in agent.run("Do 3 steps then signal done"):
            events.append(event)

        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None
        # Should stop because task_complete was called, NOT confidence_met
        assert terminate.reason == "terminal_tool"
        # Should have done all 4 calls (3 work + 1 task_complete)
        assert call_count == 4

    @pytest.mark.asyncio
    async def test_explicit_mode_ignores_no_tools(self, mock_model):
        """Agent in explicit mode doesn't stop when model returns no tool calls."""

        @tool
        def work() -> str:
            """Do work."""
            return "done"

        call_count = 0

        async def thinking_model(messages, tools=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    message=Message.assistant(
                        "Working.",
                        tool_calls=[ToolCall(id="c1", name="work", arguments={})],
                    ),
                )
            if call_count == 2:
                # Model "thinks" without calling tools — should NOT terminate
                return ModelResponse(
                    message=Message.assistant("Let me think about the results..."),
                )
            if call_count == 3:
                return ModelResponse(
                    message=Message.assistant(
                        "Now completing.",
                        tool_calls=[
                            ToolCall(
                                id="cdone", name="task_complete", arguments={"summary": "Done"}
                            )
                        ],
                    ),
                )
            return ModelResponse(message=Message.assistant("Unexpected."))

        mock_model.complete = thinking_model

        agent = Agent(model=mock_model, tools=[work], completion_mode="explicit", max_iterations=10)

        events = []
        async for event in agent.run("Work and think"):
            events.append(event)

        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None
        assert terminate.reason == "terminal_tool"
        # Should have reached call 3 (not stopped at call 2's no-tools)
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_auto_mode_still_works(self, mock_model):
        """Default auto mode behavior unchanged."""

        @tool
        def work() -> str:
            """Do work."""
            return "done"

        mock_model.complete = AsyncMock(
            side_effect=[
                ModelResponse(
                    message=Message.assistant(
                        "Working.",
                        tool_calls=[ToolCall(id="c1", name="work", arguments={})],
                    ),
                ),
                ModelResponse(message=Message.assistant("All done.")),
            ]
        )

        agent = Agent(model=mock_model, tools=[work], max_iterations=10)  # Default auto mode

        events = []
        async for event in agent.run("Do work"):
            events.append(event)

        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None
        assert terminate.reason == "complete"  # Stops on no_tools (auto mode)

    def test_task_complete_registered_in_explicit_mode(self, mock_model):
        """task_complete tool is auto-registered in explicit mode."""
        agent = Agent(model=mock_model, tools=[], completion_mode="explicit")
        assert "task_complete" in agent._tool_registry.tools

    def test_task_complete_not_registered_in_auto_mode(self, mock_model):
        """task_complete tool is NOT auto-registered in auto mode."""
        agent = Agent(model=mock_model, tools=[])  # Default auto
        assert "task_complete" not in agent._tool_registry.tools

    def test_config_completion_mode_default(self):
        """Default completion_mode is auto."""
        config = AgentConfig(model="openai:gpt-4o")
        assert config.completion_mode == "auto"

    def test_config_completion_mode_explicit(self):
        """Can set completion_mode to explicit."""
        config = AgentConfig(model="openai:gpt-4o", completion_mode="explicit")
        assert config.completion_mode == "explicit"


# =============================================================================
# Verification Reminder Tests
# =============================================================================


class TestVerificationReminder:
    """Tests for verification reminder injection."""

    @pytest.mark.asyncio
    async def test_reminder_injected_after_write(self, mock_model):
        """System message injected when a write-like tool is used."""

        @tool
        def write_file(path: str, content: str) -> str:
            """Write a file."""
            return f"Written to {path}"

        messages_seen = []

        async def capturing_model(messages, tools=None, **kwargs):
            messages_seen.append(list(messages))
            if len(messages_seen) == 1:
                return ModelResponse(
                    message=Message.assistant(
                        "Writing file.",
                        tool_calls=[
                            ToolCall(
                                id="c1",
                                name="write_file",
                                arguments={"path": "test.py", "content": "hello"},
                            )
                        ],
                    ),
                )
            return ModelResponse(message=Message.assistant("Done."))

        mock_model.complete = capturing_model

        agent = Agent(model=mock_model, tools=[write_file], max_iterations=5)

        async for _ in agent.run("Write a file"):
            pass

        # Second model call should have received the verification reminder
        assert len(messages_seen) >= 2
        second_call_msgs = messages_seen[1]
        reminder_msgs = [
            m
            for m in second_call_msgs
            if m.role.value == "system" and "Verification Reminder" in (m.content or "")
        ]
        assert len(reminder_msgs) >= 1

    @pytest.mark.asyncio
    async def test_no_reminder_for_read_tools(self, mock_model):
        """No reminder when only read-like tools are used."""

        @tool
        def read_file(path: str) -> str:
            """Read a file."""
            return "file contents"

        messages_seen = []

        async def capturing_model(messages, tools=None, **kwargs):
            messages_seen.append(list(messages))
            if len(messages_seen) == 1:
                return ModelResponse(
                    message=Message.assistant(
                        "Reading.",
                        tool_calls=[
                            ToolCall(id="c1", name="read_file", arguments={"path": "test.py"})
                        ],
                    ),
                )
            return ModelResponse(message=Message.assistant("Done."))

        mock_model.complete = capturing_model

        agent = Agent(model=mock_model, tools=[read_file], max_iterations=5)

        async for _ in agent.run("Read a file"):
            pass

        # No verification reminder should appear
        for call_msgs in messages_seen:
            for m in call_msgs:
                if m.role.value == "system" and m.content:
                    assert "Verification Reminder" not in m.content


# =============================================================================
# Interrupt/Resume Tests
# =============================================================================


class TestInterruptResume:
    """Tests for ask_user interrupt and resume."""

    def test_ask_user_registered_in_explicit_mode(self, mock_model):
        """ask_user tool registered in explicit completion mode."""
        agent = Agent(model=mock_model, tools=[], completion_mode="explicit")
        assert "ask_user" in agent._tool_registry.tools
        assert "task_complete" in agent._tool_registry.tools

    def test_ask_user_not_registered_in_auto_mode(self, mock_model):
        """ask_user not registered in auto mode."""
        agent = Agent(model=mock_model, tools=[])
        assert "ask_user" not in agent._tool_registry.tools

    @pytest.mark.asyncio
    async def test_interrupt_yields_interrupt_event(self, mock_model):
        """When ask_user is called, agent yields InterruptEvent and pauses."""
        from locus.core.events import InterruptEvent

        call_count = 0

        async def interrupting_model(messages, tools=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    message=Message.assistant(
                        "I need to ask the user.",
                        tool_calls=[
                            ToolCall(
                                id="c1",
                                name="ask_user",
                                arguments={
                                    "question": "Should I use JWT or session auth?",
                                    "options": "JWT,session,OAuth",
                                },
                            )
                        ],
                    ),
                )
            return ModelResponse(message=Message.assistant("Unreachable."))

        mock_model.complete = interrupting_model

        agent = Agent(
            model=mock_model,
            tools=[],
            completion_mode="explicit",
            max_iterations=5,
        )

        events = []
        async for event in agent.run("Build an auth system"):
            events.append(event)

        # Should have yielded an InterruptEvent (not TerminateEvent)
        interrupt_events = [e for e in events if isinstance(e, InterruptEvent)]
        assert len(interrupt_events) == 1
        assert "JWT" in interrupt_events[0].question
        assert interrupt_events[0].options is not None

        # Should NOT have a TerminateEvent (agent is paused, not done)
        terminate_events = [e for e in events if isinstance(e, TerminateEvent)]
        assert len(terminate_events) == 0

        # Agent should have saved interrupt state
        assert agent._interrupt_state is not None


# =============================================================================
# Verification Gate Tests
# =============================================================================


class TestVerificationGate:
    """Tests for task_complete verification gate."""

    @pytest.mark.asyncio
    async def test_task_complete_blocked_without_verification(self, mock_model):
        """task_complete returns BLOCKED when writes happened but no tests ran."""

        @tool
        def write_file(path: str, content: str) -> str:
            """Write a file."""
            return f"Written to {path}"

        call_count = 0

        async def eager_model(messages, tools=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    message=Message.assistant(
                        "Writing file.",
                        tool_calls=[
                            ToolCall(
                                id="c1",
                                name="write_file",
                                arguments={"path": "test.py", "content": "pass"},
                            )
                        ],
                    ),
                )
            if call_count == 2:
                # Try to complete without running tests
                return ModelResponse(
                    message=Message.assistant(
                        "Done!",
                        tool_calls=[
                            ToolCall(
                                id="c2", name="task_complete", arguments={"summary": "All done"}
                            )
                        ],
                    ),
                )
            if call_count == 3:
                # After being blocked, model should see BLOCKED message and try again
                # This time just complete (gate resets after one block)
                return ModelResponse(
                    message=Message.assistant(
                        "Ok completing for real.",
                        tool_calls=[
                            ToolCall(
                                id="c3",
                                name="task_complete",
                                arguments={"summary": "Done after block"},
                            )
                        ],
                    ),
                )
            return ModelResponse(message=Message.assistant("Unexpected."))

        mock_model.complete = eager_model

        agent = Agent(
            model=mock_model,
            tools=[write_file],
            completion_mode="explicit",
            max_iterations=10,
        )

        events = []
        async for event in agent.run("Write and complete"):
            events.append(event)

        tool_events = [e for e in events if isinstance(e, ToolCompleteEvent)]
        # The task_complete call should have returned BLOCKED
        blocked = [e for e in tool_events if e.result and "BLOCKED" in e.result]
        assert len(blocked) >= 1

        # Agent should eventually complete (gate resets after block)
        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None
        assert terminate.reason == "terminal_tool"

    @pytest.mark.asyncio
    async def test_task_complete_allowed_after_verification(self, mock_model):
        """task_complete succeeds when verification ran after writes."""

        @tool
        def write_file(path: str, content: str) -> str:
            """Write a file."""
            return f"Written to {path}"

        @tool
        def run_command(command: str, working_dir: str) -> str:
            """Run a command."""
            return "2 passed"

        call_count = 0

        async def proper_model(messages, tools=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    message=Message.assistant(
                        "Writing.",
                        tool_calls=[
                            ToolCall(
                                id="c1",
                                name="write_file",
                                arguments={"path": "app.py", "content": "code"},
                            )
                        ],
                    ),
                )
            if call_count == 2:
                return ModelResponse(
                    message=Message.assistant(
                        "Testing.",
                        tool_calls=[
                            ToolCall(
                                id="c2",
                                name="run_command",
                                arguments={"command": "pytest", "working_dir": "."},
                            )
                        ],
                    ),
                )
            if call_count == 3:
                return ModelResponse(
                    message=Message.assistant(
                        "All tests pass.",
                        tool_calls=[
                            ToolCall(
                                id="c3", name="task_complete", arguments={"summary": "Tests pass"}
                            )
                        ],
                    ),
                )
            return ModelResponse(message=Message.assistant("Unexpected."))

        mock_model.complete = proper_model

        agent = Agent(
            model=mock_model,
            tools=[write_file, run_command],
            completion_mode="explicit",
            max_iterations=10,
        )

        events = []
        async for event in agent.run("Write, test, complete"):
            events.append(event)

        tool_events = [e for e in events if isinstance(e, ToolCompleteEvent)]
        # No BLOCKED messages — verification was done
        blocked = [e for e in tool_events if e.result and "BLOCKED" in e.result]
        assert len(blocked) == 0

        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None
        assert terminate.reason == "terminal_tool"

    @pytest.mark.asyncio
    async def test_gate_disabled_when_require_verification_false(self, mock_model):
        """Gate doesn't fire when require_verification=False."""

        @tool
        def write_file(path: str, content: str) -> str:
            """Write a file."""
            return f"Written to {path}"

        mock_model.complete = AsyncMock(
            side_effect=[
                ModelResponse(
                    message=Message.assistant(
                        "Writing.",
                        tool_calls=[
                            ToolCall(
                                id="c1",
                                name="write_file",
                                arguments={"path": "x.py", "content": "y"},
                            )
                        ],
                    ),
                ),
                ModelResponse(
                    message=Message.assistant(
                        "Done.",
                        tool_calls=[
                            ToolCall(
                                id="c2", name="task_complete", arguments={"summary": "Wrote file"}
                            )
                        ],
                    ),
                ),
            ]
        )

        agent = Agent(
            model=mock_model,
            tools=[write_file],
            completion_mode="explicit",
            require_verification=False,
        )

        events = []
        async for event in agent.run("Write and complete"):
            events.append(event)

        # Should complete without BLOCKED
        tool_events = [e for e in events if isinstance(e, ToolCompleteEvent)]
        blocked = [e for e in tool_events if e.result and "BLOCKED" in e.result]
        assert len(blocked) == 0

        terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
        assert terminate is not None
        assert terminate.reason == "terminal_tool"


# =============================================================================
# Agent-as-Tool Tests
# =============================================================================


class TestAgentAsTool:
    """Tests for Agent.as_tool() — wrapping an agent as a tool."""

    def test_as_tool_returns_tool(self, mock_model):
        """as_tool() returns a Tool instance."""
        from locus.tools.decorator import Tool

        mock_model.complete = AsyncMock(
            return_value=ModelResponse(message=Message.assistant("I'm a sub-agent."))
        )

        sub_agent = Agent(model=mock_model, tools=[], system_prompt="I help.")
        t = sub_agent.as_tool("helper", "A helpful sub-agent")

        assert isinstance(t, Tool)
        assert t.name == "helper"

    def test_as_tool_default_name(self, mock_model):
        """as_tool() uses agent_id or 'sub_agent' as default name."""
        mock_model.complete = AsyncMock(return_value=ModelResponse(message=Message.assistant("ok")))

        agent = Agent(model=mock_model, tools=[], agent_id="researcher")
        t = agent.as_tool()
        assert t.name == "researcher"

        agent2 = Agent(model=mock_model, tools=[])
        t2 = agent2.as_tool()
        assert t2.name == "sub_agent"

    def test_parent_calls_sub_agent(self, mock_model):
        """Parent agent can call sub-agent tool and get response."""

        # Sub-agent model
        sub_model = MagicMock()
        sub_model.complete = AsyncMock(
            return_value=ModelResponse(
                message=Message.assistant("Quantum computing uses superposition and entanglement."),
            )
        )

        sub_agent = Agent(
            model=sub_model,
            tools=[],
            system_prompt="You are a research specialist.",
        )
        research_tool = sub_agent.as_tool("research", "Research a topic in depth")

        # Parent agent model — calls the sub-agent tool, then answers
        call_count = 0

        async def parent_model(messages, tools=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    message=Message.assistant(
                        "Let me research this.",
                        tool_calls=[
                            ToolCall(
                                id="c1",
                                name="research",
                                arguments={"prompt": "What is quantum computing?"},
                            )
                        ],
                    ),
                )
            return ModelResponse(
                message=Message.assistant(
                    "Based on my research: quantum computing uses superposition."
                ),
            )

        mock_model.complete = parent_model

        parent = Agent(model=mock_model, tools=[research_tool], max_iterations=5)
        result = parent.run_sync("Tell me about quantum computing")

        assert result.success
        assert "quantum" in result.message.lower() or "superposition" in result.message.lower()
        # Sub-agent should have been called
        sub_model.complete.assert_called_once()

    def test_sub_agent_failure_returns_status(self, mock_model):
        """When sub-agent hits max_iterations, parent sees the status."""

        sub_model = MagicMock()

        async def looping_sub(messages, tools=None, **kwargs):
            return ModelResponse(
                message=Message.assistant(
                    "Still working...",
                    tool_calls=[ToolCall(id="cx", name="nonexistent", arguments={})],
                ),
            )

        sub_model.complete = looping_sub

        sub_agent = Agent(model=sub_model, tools=[], max_iterations=2)
        sub_tool = sub_agent.as_tool("worker")

        # Parent calls sub-agent
        call_count = 0

        async def parent_model(messages, tools=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    message=Message.assistant(
                        "Delegating.",
                        tool_calls=[
                            ToolCall(id="c1", name="worker", arguments={"prompt": "Do something"})
                        ],
                    ),
                )
            return ModelResponse(message=Message.assistant("The worker had issues."))

        mock_model.complete = parent_model

        parent = Agent(model=mock_model, tools=[sub_tool], max_iterations=5)
        result = parent.run_sync("Do work")

        # Parent should still complete (sub-agent failure is just a tool result)
        assert result.stop_reason in ("complete", "max_iterations")


# =============================================================================
# Planning Step Tests
# =============================================================================


class TestPlanningStep:
    """Tests for planning=True — plan before acting."""

    @pytest.mark.asyncio
    async def test_planning_injects_prompt_on_first_iteration(self, mock_model):
        """Planning prompt injected on iteration 1."""

        messages_seen = []

        async def capturing_model(messages, tools=None, **kwargs):
            messages_seen.append(list(messages))
            if len(messages_seen) == 1:
                # First call: model sees planning prompt, responds with plan + tool call
                return ModelResponse(
                    message=Message.assistant(
                        "Plan:\n1. Search for info\n2. Analyze results\n3. Summarize\n\nStarting step 1.",
                        tool_calls=[ToolCall(id="c1", name="search", arguments={"q": "test"})],
                    ),
                )
            return ModelResponse(message=Message.assistant("Done. Summary based on findings."))

        mock_model.complete = capturing_model

        @tool
        def search(q: str) -> str:
            """Search."""
            return "found data"

        agent = Agent(
            model=mock_model,
            tools=[search],
            planning=True,
            max_iterations=5,
        )

        events = []
        async for event in agent.run("Research and summarize"):
            events.append(event)

        # First model call should have received the planning prompt
        first_call = messages_seen[0]
        planning_msgs = [
            m
            for m in first_call
            if m.role.value == "system" and "Planning Phase" in (m.content or "")
        ]
        assert len(planning_msgs) == 1
        assert "step-by-step plan" in planning_msgs[0].content.lower()

    @pytest.mark.asyncio
    async def test_plan_stored_in_metadata(self, mock_model):
        """Plan from first iteration stored in state metadata."""

        mock_model.complete = AsyncMock(
            side_effect=[
                ModelResponse(
                    message=Message.assistant(
                        "Plan:\n1. Lookup facts\n2. Write summary",
                        tool_calls=[ToolCall(id="c1", name="lookup", arguments={"q": "test"})],
                    ),
                ),
                ModelResponse(message=Message.assistant("Summary complete.")),
            ]
        )

        @tool
        def lookup(q: str) -> str:
            """Lookup."""
            return "facts"

        agent = Agent(model=mock_model, tools=[lookup], planning=True, max_iterations=5)
        result = agent.run_sync("Research topic")

        # Plan should be in state metadata
        assert "plan" in result.state.metadata
        assert "Lookup facts" in result.state.metadata["plan"]

    @pytest.mark.asyncio
    async def test_no_planning_when_disabled(self, mock_model):
        """No planning prompt when planning=False (default)."""

        messages_seen = []

        async def capturing_model(messages, tools=None, **kwargs):
            messages_seen.append(list(messages))
            return ModelResponse(message=Message.assistant("Done."))

        mock_model.complete = capturing_model

        agent = Agent(model=mock_model, tools=[], max_iterations=5)  # Default: planning=False

        async for _ in agent.run("Do something"):
            pass

        # No planning prompt should be in any call
        for call_msgs in messages_seen:
            for m in call_msgs:
                if m.role.value == "system" and m.content:
                    assert "Planning Phase" not in m.content

    @pytest.mark.asyncio
    async def test_replan_injected_when_stuck(self, mock_model):
        """Replan suggestion injected when reflexion detects stuck + planning enabled."""
        from locus.agent import ReflexionConfig

        @tool
        def broken_tool(q: str) -> str:
            """A tool that always fails."""
            raise RuntimeError("Connection failed")

        call_count = 0
        messages_seen = []

        async def stuck_model(messages, tools=None, **kwargs):
            nonlocal call_count
            messages_seen.append(list(messages))
            call_count += 1
            if call_count <= 4:
                return ModelResponse(
                    message=Message.assistant(
                        f"Trying {call_count}.",
                        tool_calls=[
                            ToolCall(
                                id=f"c{call_count}",
                                name="broken_tool",
                                arguments={"q": f"query{call_count}"},
                            )
                        ],
                    ),
                )
            return ModelResponse(message=Message.assistant("Giving up."))

        mock_model.complete = stuck_model

        agent = Agent(
            model=mock_model,
            tools=[broken_tool],
            planning=True,
            reflexion=ReflexionConfig(enabled=True, include_guidance=True),
            max_iterations=6,
        )

        async for _ in agent.run("Find something"):
            pass

        # Check if replan message was injected
        found_replan = False
        for call_msgs in messages_seen:
            for m in call_msgs:
                if m.role.value == "system" and m.content and "[Replan]" in m.content:
                    found_replan = True
                    break

        assert found_replan, "Replan message was not injected when agent was stuck"

    def test_config_planning_default_false(self):
        """Default planning is False."""
        config = AgentConfig(model="openai:gpt-4o")
        assert config.planning is False

    def test_config_planning_true(self):
        """Can set planning=True."""
        config = AgentConfig(model="openai:gpt-4o", planning=True)
        assert config.planning is True


# =============================================================================
# Swarm Orchestration Tests
# =============================================================================


class TestSwarmOrchestration:
    """Tests for Swarm execution with multiple agents."""

    @pytest.mark.asyncio
    async def test_swarm_distributes_tasks(self):
        """Swarm distributes tasks among agents."""
        from locus.multiagent.swarm import Swarm, SwarmAgent

        mock_model = MagicMock()
        mock_model.complete = AsyncMock(
            return_value=ModelResponse(
                message=Message.assistant(
                    "### Findings\nFound important data.\n\n### Analysis\nAnalysis complete."
                ),
            )
        )

        agent1 = SwarmAgent(name="researcher", capabilities=["research"], model=mock_model)
        agent2 = SwarmAgent(name="analyst", capabilities=["market", "data"], model=mock_model)

        swarm = Swarm(name="test_swarm", agents=[agent1, agent2], model=mock_model)
        swarm.add_task("Research the topic of AI", priority=5)
        swarm.add_task("Analyze the market data trends", priority=3)

        result = await swarm.execute(decompose_tasks=False)

        assert result.success
        assert len(result.completed_tasks) == 2
        assert result.summary is not None

    @pytest.mark.asyncio
    async def test_swarm_shared_context(self):
        """Agents share findings through SharedContext."""
        from locus.multiagent.swarm import SharedContext

        ctx = SharedContext()
        await ctx.add_finding("key1", "value1", "agent_1")
        await ctx.add_finding("key2", "value2", "agent_2")
        await ctx.post_to_blackboard("msg1", "Need help with X", "agent_1")

        assert ctx.findings["key1"] == "value1"
        assert ctx.findings["key2"] == "value2"
        assert "Need help" in ctx.blackboard["msg1"]
        assert len(ctx.discovery_log) == 3
        summary = ctx.get_summary()
        assert "key1" in summary

    @pytest.mark.asyncio
    async def test_swarm_capability_matching(self):
        """Agents only claim tasks matching their capabilities."""
        from locus.multiagent.swarm import Swarm, SwarmAgent

        mock_model = MagicMock()
        mock_model.complete = AsyncMock(
            return_value=ModelResponse(
                message=Message.assistant("### Findings\nDone.\n\n### Analysis\nDone."),
            )
        )

        # Each agent handles tasks matching its capabilities
        researcher = SwarmAgent(name="researcher", capabilities=["research"], model=mock_model)
        writer = SwarmAgent(name="writer", capabilities=["write", "report"], model=mock_model)

        swarm = Swarm(name="test", agents=[researcher, writer], model=mock_model)
        swarm.add_task("Research quantum computing")
        swarm.add_task("Write a report about findings")

        result = await swarm.execute(decompose_tasks=False)

        # Both tasks should complete (each agent handles what it can)
        assert len(result.completed_tasks) == 2

    @pytest.mark.asyncio
    async def test_swarm_handles_agent_failure(self):
        """Swarm handles agent failures gracefully."""
        from locus.multiagent.swarm import Swarm, SwarmAgent

        failing_model = MagicMock()
        failing_model.complete = AsyncMock(side_effect=RuntimeError("Model crashed"))

        agent = SwarmAgent(name="broken", model=failing_model)

        swarm = Swarm(name="test", agents=[agent])
        swarm.add_task("Do something")

        result = await swarm.execute(decompose_tasks=False)

        # Task should be failed, not crash the swarm
        assert len(result.failed_tasks) == 1
        assert result.failed_tasks[0].error is not None


# =============================================================================
# Agent Handoff Tests
# =============================================================================


class TestAgentHandoff:
    """Tests for agent-to-agent handoff."""

    @pytest.mark.asyncio
    async def test_handoff_transfers_context(self):
        """Handoff creates context and target agent receives it."""
        from locus.multiagent.handoff import (
            Handoff,
            HandoffAgent,
            HandoffReason,
        )

        mock_model = MagicMock()
        mock_model.complete = AsyncMock(
            return_value=ModelResponse(
                message=Message.assistant(
                    "I received the handoff. Based on the findings, here is my analysis: the data shows clear trends."
                ),
            )
        )

        source = HandoffAgent(id="researcher", name="Researcher", model=mock_model)
        target = HandoffAgent(id="analyst", name="Analyst", model=mock_model)

        manager = Handoff(name="test")
        manager.register_agents([source, target])

        result = await manager.execute_handoff(
            source_agent=source,
            target_agent_id="analyst",
            task="Analyze the research findings",
            reason=HandoffReason.SPECIALIZATION,
            findings={"key_finding": "Diabetes affects 537M people"},
        )

        assert result.success
        assert result.source_agent_id == "researcher"
        assert result.target_agent_id == "analyst"
        assert result.output is not None

    @pytest.mark.asyncio
    async def test_handoff_chain(self):
        """Chain of handoffs: A → B → C."""
        from locus.multiagent.handoff import Handoff, HandoffAgent

        mock_model = MagicMock()
        mock_model.complete = AsyncMock(
            return_value=ModelResponse(
                message=Message.assistant("Processed and passing along."),
            )
        )

        agent_a = HandoffAgent(id="a", name="Agent A", model=mock_model)
        agent_b = HandoffAgent(id="b", name="Agent B", model=mock_model)
        agent_c = HandoffAgent(id="c", name="Agent C", model=mock_model)

        manager = Handoff(name="chain_test")
        manager.register_agents([agent_a, agent_b, agent_c])

        results = await manager.chain_handoff(
            agent_chain=["a", "b", "c"],
            task="Process this data through the pipeline",
        )

        assert len(results) >= 2  # At least 2 handoffs in A→B→C
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_handoff_to_unknown_agent_fails(self):
        """Handoff to non-existent agent returns error."""
        from locus.multiagent.handoff import Handoff, HandoffAgent, HandoffReason

        source = HandoffAgent(id="src", name="Source", model=MagicMock())
        manager = Handoff(name="test")
        manager.register_agent(source)

        result = await manager.execute_handoff(
            source_agent=source,
            target_agent_id="nonexistent",
            task="Do something",
            reason=HandoffReason.ESCALATION,
        )

        assert not result.success
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_handoff_context_has_findings(self):
        """HandoffContext includes findings from source agent."""
        from locus.multiagent.handoff import HandoffContext, HandoffReason

        context = HandoffContext(
            handoff_id="h1",
            source_agent_id="researcher",
            target_agent_id="writer",
            original_task="Write a report",
            reason=HandoffReason.SPECIALIZATION,
            findings={"research": "Found important data about AI"},
            progress_summary="Completed research phase",
        )

        prompt = context.to_prompt()
        assert "researcher" in prompt
        assert "Write a report" in prompt
        assert "Found important data" in prompt


# =============================================================================
# Orchestrator Routing Tests
# =============================================================================


class TestOrchestratorRouting:
    """Tests for orchestrator routing tasks to specialists."""

    @pytest.mark.asyncio
    async def test_orchestrator_routes_and_executes(self):
        """Orchestrator routes task to specialists and produces summary."""
        from locus.multiagent.orchestrator import Orchestrator
        from locus.multiagent.specialist import Specialist

        mock_model = MagicMock()

        async def smart_model(messages, tools=None, **kwargs):
            content = messages[-1].content or "" if messages else ""

            # Routing decision — return JSON with specialist IDs
            if "specialist" in content.lower() or "select" in content.lower():
                return ModelResponse(
                    message=Message.assistant(
                        '```json\n{"specialists": ["medical_researcher"], '
                        '"reasoning": "Need medical research", '
                        '"subtasks": {"medical_researcher": "Find causes of diabetes"}}\n```'
                    ),
                )
            # Specialist work
            if "medical" in content.lower() or "diabetes" in content.lower():
                return ModelResponse(
                    message=Message.assistant(
                        "Found: diabetes affects 537M people globally. "
                        "Key risk factors include obesity and genetics."
                    ),
                )
            # Correlation / summary
            return ModelResponse(
                message=Message.assistant(
                    "Summary: Research shows diabetes is a global health challenge."
                ),
            )

        mock_model.complete = smart_model

        specialist = Specialist(
            id="medical_researcher",
            name="Medical Researcher",
            specialist_type="researcher",
            description="Researches medical topics",
            system_prompt="You are a medical researcher.",
            model=mock_model,
        )

        orchestrator = Orchestrator(name="test_orchestrator", model=mock_model)
        orchestrator.register_specialist(specialist)

        result = await orchestrator.execute("What are the global impacts of diabetes?")

        assert result.success
        assert result.summary is not None
        assert len(result.summary) > 20
        assert len(result.decisions) >= 1

    @pytest.mark.asyncio
    async def test_orchestrator_multiple_specialists(self):
        """Orchestrator can invoke multiple specialists."""
        from locus.multiagent.orchestrator import Orchestrator
        from locus.multiagent.specialist import Specialist

        mock_model = MagicMock()

        async def multi_model(messages, tools=None, **kwargs):
            content = messages[-1].content or "" if messages else ""
            if "specialist" in content.lower() or "select" in content.lower():
                return ModelResponse(
                    message=Message.assistant(
                        '```json\n{"specialists": ["researcher", "analyst"], '
                        '"reasoning": "Both needed"}\n```'
                    ),
                )
            return ModelResponse(
                message=Message.assistant("Findings from specialist analysis."),
            )

        mock_model.complete = multi_model

        researcher = Specialist(
            id="researcher",
            name="Researcher",
            specialist_type="researcher",
            description="Finds facts",
            system_prompt="Find facts.",
            model=mock_model,
        )
        analyst = Specialist(
            id="analyst",
            name="Analyst",
            specialist_type="analyst",
            description="Analyzes data",
            system_prompt="Analyze data.",
            model=mock_model,
        )

        orchestrator = Orchestrator(name="test", model=mock_model)
        orchestrator.register_specialists([researcher, analyst])

        result = await orchestrator.execute("Research and analyze diabetes")

        assert result.success
        assert len(result.specialist_results) == 2
        assert "researcher" in result.specialist_results
        assert "analyst" in result.specialist_results

    @pytest.mark.asyncio
    async def test_orchestrator_without_model_invokes_all(self):
        """Orchestrator without routing model invokes all specialists."""
        from locus.multiagent.orchestrator import Orchestrator
        from locus.multiagent.specialist import Specialist

        spec_model = MagicMock()
        spec_model.complete = AsyncMock(
            return_value=ModelResponse(message=Message.assistant("Analysis done"))
        )

        spec1 = Specialist(
            id="s1",
            name="S1",
            specialist_type="t1",
            description="First",
            system_prompt="Test",
            model=spec_model,
        )
        spec2 = Specialist(
            id="s2",
            name="S2",
            specialist_type="t2",
            description="Second",
            system_prompt="Test",
            model=spec_model,
        )

        orchestrator = Orchestrator(name="no_model")
        orchestrator.register_specialists([spec1, spec2])

        result = await orchestrator.execute("Analyze this")

        assert result.success
        assert "s1" in result.specialist_results
        assert "s2" in result.specialist_results

    @pytest.mark.asyncio
    async def test_orchestrator_retries_empty_specialist(self):
        """Orchestrator retries when specialist returns empty output."""
        from locus.multiagent.orchestrator import Orchestrator
        from locus.multiagent.specialist import Specialist

        call_count = 0

        async def flaky_complete(messages, tools=None, **kwargs):
            nonlocal call_count
            call_count += 1
            # First call returns empty, second returns content
            if call_count == 1:
                return ModelResponse(message=Message.assistant(""))
            return ModelResponse(message=Message.assistant("Real output after retry"))

        mock_model = MagicMock()
        mock_model.complete = flaky_complete

        spec = Specialist(
            id="flaky_spec",
            name="Flaky",
            specialist_type="test",
            description="Test",
            system_prompt="Test",
            model=mock_model,
        )

        # No routing model — invokes all specialists directly
        orchestrator = Orchestrator(name="retry_test")
        orchestrator.register_specialist(spec)

        result = await orchestrator.execute("Test task")

        assert result.success
        assert result.specialist_results["flaky_spec"].output == "Real output after retry"
        assert call_count == 2  # First empty + retry


# =============================================================================
# Composition Primitives Tests
# =============================================================================


class TestSequentialPipeline:
    """Tests for SequentialPipeline."""

    @pytest.mark.asyncio
    async def test_sequential_chains_output(self):
        """Sequential pipeline passes output from one agent to the next."""
        from locus.agent.composition import SequentialPipeline

        outputs_seen = []

        class FakeAgent:
            def __init__(self, name):
                self.name = name

            def run_sync(self, prompt):
                outputs_seen.append(prompt)
                return AgentResult(
                    message=f"{self.name} processed: {prompt[:30]}",
                    state=AgentState(agent_id=self.name),
                    stop_reason="complete",
                )

        pipeline = SequentialPipeline(agents=[FakeAgent("A"), FakeAgent("B"), FakeAgent("C")])
        result = await pipeline.run("initial task")

        assert result.success
        assert len(result.outputs) == 3
        # First agent gets the original task
        assert outputs_seen[0] == "initial task"
        # Second agent gets output from first
        assert "A processed" in outputs_seen[1]
        # Third agent gets output from second
        assert "B processed" in outputs_seen[2]
        # Final output is from last agent
        assert "C processed" in result.final_output

    @pytest.mark.asyncio
    async def test_sequential_custom_template(self):
        """Sequential pipeline uses custom prompt template."""
        from locus.agent.composition import SequentialPipeline

        prompts = []

        class FakeAgent:
            def run_sync(self, prompt):
                prompts.append(prompt)
                return AgentResult(
                    message="output",
                    state=AgentState(agent_id="a"),
                    stop_reason="complete",
                )

        pipeline = SequentialPipeline(
            agents=[FakeAgent(), FakeAgent()],
            prompt_template="Previous: {previous_output} | Task: {task}",
        )
        result = await pipeline.run("my task")

        assert result.success
        assert prompts[1] == "Previous: output | Task: my task"

    @pytest.mark.asyncio
    async def test_sequential_single_agent(self):
        """Sequential pipeline works with a single agent."""
        from locus.agent.composition import SequentialPipeline

        class FakeAgent:
            def run_sync(self, prompt):
                return AgentResult(
                    message="done",
                    state=AgentState(agent_id="solo"),
                    stop_reason="complete",
                )

        pipeline = SequentialPipeline(agents=[FakeAgent()])
        result = await pipeline.run("task")

        assert result.success
        assert result.final_output == "done"
        assert len(result.outputs) == 1

    @pytest.mark.asyncio
    async def test_sequential_handles_error(self):
        """Sequential pipeline handles agent errors gracefully."""
        from locus.agent.composition import SequentialPipeline

        class FailingAgent:
            def run_sync(self, prompt):
                raise RuntimeError("Agent crashed")

        pipeline = SequentialPipeline(agents=[FailingAgent()])
        result = await pipeline.run("task")

        assert not result.success
        assert "crashed" in result.error


class TestParallelPipeline:
    """Tests for ParallelPipeline."""

    @pytest.mark.asyncio
    async def test_parallel_runs_all_agents(self):
        """Parallel pipeline runs all agents and collects results."""
        from locus.agent.composition import ParallelPipeline

        class FakeAgent:
            def __init__(self, name):
                self.name = name

            def run_sync(self, prompt):
                return AgentResult(
                    message=f"{self.name}: {prompt[:20]}",
                    state=AgentState(agent_id=self.name),
                    stop_reason="complete",
                )

        pipeline = ParallelPipeline(agents=[FakeAgent("A"), FakeAgent("B"), FakeAgent("C")])
        result = await pipeline.run("analyze this")

        assert result.success
        assert len(result.outputs) == 3
        assert "A:" in result.outputs[0]
        assert "B:" in result.outputs[1]
        assert "C:" in result.outputs[2]
        # Default merge = concatenate
        assert "A:" in result.final_output
        assert "B:" in result.final_output

    @pytest.mark.asyncio
    async def test_parallel_merge_last(self):
        """Parallel pipeline with 'last' merge strategy."""
        from locus.agent.composition import ParallelPipeline

        class FakeAgent:
            def __init__(self, val):
                self.val = val

            def run_sync(self, prompt):
                return AgentResult(
                    message=self.val,
                    state=AgentState(agent_id="a"),
                    stop_reason="complete",
                )

        pipeline = ParallelPipeline(
            agents=[FakeAgent("first"), FakeAgent("second"), FakeAgent("third")],
            merge_strategy="last",
        )
        result = await pipeline.run("task")

        assert result.success
        assert result.final_output == "third"

    @pytest.mark.asyncio
    async def test_parallel_custom_task_map(self):
        """Parallel pipeline with per-agent custom tasks."""
        from locus.agent.composition import ParallelPipeline

        prompts = {}

        class FakeAgent:
            def __init__(self, idx):
                self.idx = idx

            def run_sync(self, prompt):
                prompts[self.idx] = prompt
                return AgentResult(
                    message="ok",
                    state=AgentState(agent_id="a"),
                    stop_reason="complete",
                )

        pipeline = ParallelPipeline(agents=[FakeAgent(0), FakeAgent(1)])
        result = await pipeline.run(
            "default",
            task_map={0: "custom task for agent 0"},
        )

        assert result.success
        assert prompts[0] == "custom task for agent 0"
        assert prompts[1] == "default"


class TestLoopAgent:
    """Tests for LoopAgent."""

    @pytest.mark.asyncio
    async def test_loop_stops_on_condition(self):
        """Loop stops when condition returns True."""
        from locus.agent.composition import LoopAgent

        call_count = 0

        class FakeAgent:
            def run_sync(self, prompt):
                nonlocal call_count
                call_count += 1
                msg = "DONE" if call_count >= 3 else "not yet"
                return AgentResult(
                    message=msg,
                    state=AgentState(agent_id="a"),
                    stop_reason="complete",
                )

        loop_agent = LoopAgent(
            agent=FakeAgent(),
            condition=lambda output: "DONE" in output,
            max_loops=10,
        )
        result = await loop_agent.run("iterate until done")

        assert result.success
        assert len(result.outputs) == 3
        assert result.final_output == "DONE"

    @pytest.mark.asyncio
    async def test_loop_respects_max_loops(self):
        """Loop stops at max_loops even if condition never met."""
        from locus.agent.composition import LoopAgent

        class FakeAgent:
            def run_sync(self, prompt):
                return AgentResult(
                    message="still going",
                    state=AgentState(agent_id="a"),
                    stop_reason="complete",
                )

        loop_agent = LoopAgent(
            agent=FakeAgent(),
            condition=lambda output: False,  # Never stops
            max_loops=3,
        )
        result = await loop_agent.run("infinite task")

        assert result.success
        assert len(result.outputs) == 3

    @pytest.mark.asyncio
    async def test_loop_custom_prompt(self):
        """Loop uses custom prompt template for iterations."""
        from locus.agent.composition import LoopAgent

        prompts = []

        class FakeAgent:
            def run_sync(self, prompt):
                prompts.append(prompt)
                return AgentResult(
                    message="iteration output",
                    state=AgentState(agent_id="a"),
                    stop_reason="complete",
                )

        loop_agent = LoopAgent(
            agent=FakeAgent(),
            condition=lambda output: False,
            max_loops=2,
            loop_prompt="Improve: {previous_output}",
        )
        result = await loop_agent.run("start")

        assert result.success
        assert prompts[0] == "start"
        assert prompts[1] == "Improve: iteration output"


class TestCompositionHelpers:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_sequential_helper(self):
        """sequential() creates a SequentialPipeline."""
        from locus.agent.composition import SequentialPipeline, sequential

        class FakeAgent:
            def run_sync(self, prompt):
                return AgentResult(
                    message="ok",
                    state=AgentState(agent_id="a"),
                    stop_reason="complete",
                )

        pipeline = sequential(FakeAgent(), FakeAgent())
        assert isinstance(pipeline, SequentialPipeline)
        result = await pipeline.run("task")
        assert result.success

    @pytest.mark.asyncio
    async def test_parallel_helper(self):
        """parallel() creates a ParallelPipeline."""
        from locus.agent.composition import ParallelPipeline, parallel

        class FakeAgent:
            def run_sync(self, prompt):
                return AgentResult(
                    message="ok",
                    state=AgentState(agent_id="a"),
                    stop_reason="complete",
                )

        pipeline = parallel(FakeAgent(), FakeAgent())
        assert isinstance(pipeline, ParallelPipeline)
        result = await pipeline.run("task")
        assert result.success

    @pytest.mark.asyncio
    async def test_loop_helper(self):
        """loop() creates a LoopAgent."""
        from locus.agent.composition import LoopAgent, loop

        class FakeAgent:
            def run_sync(self, prompt):
                return AgentResult(
                    message="STOP",
                    state=AgentState(agent_id="a"),
                    stop_reason="complete",
                )

        agent = loop(FakeAgent(), condition=lambda o: "STOP" in o, max_loops=3)
        assert isinstance(agent, LoopAgent)
        result = await agent.run("task")
        assert result.success
        assert len(result.outputs) == 1  # Stops on first iteration


# =============================================================================
# Evaluation Framework Tests
# =============================================================================


class TestEvalCase:
    """Tests for EvalCase."""

    def test_create_basic_case(self):
        """Create a basic eval case."""
        from locus.evaluation import EvalCase

        case = EvalCase(
            name="test_basic",
            prompt="What is 2+2?",
            expected_output_contains=["4"],
        )
        assert case.name == "test_basic"
        assert case.prompt == "What is 2+2?"
        assert case.expected_output_contains == ["4"]

    def test_create_full_case(self):
        """Create a case with all fields."""
        from locus.evaluation import EvalCase

        case = EvalCase(
            name="complex",
            prompt="Search and summarize",
            expected_tools=["search", "summarize"],
            expected_output_contains=["result"],
            expected_output_not_contains=["error"],
            max_iterations=5,
            max_duration_ms=10000,
            tags=["search", "complex"],
        )
        assert len(case.expected_tools) == 2
        assert case.max_iterations == 5


class TestEvalRunner:
    """Tests for EvalRunner."""

    def test_run_passing_case(self):
        """Runner evaluates a passing case."""
        from locus.evaluation import EvalCase, EvalRunner

        class FakeAgent:
            def run_sync(self, prompt):
                return AgentResult(
                    message="The answer is 42.",
                    state=AgentState(agent_id="test"),
                    stop_reason="complete",
                )

        runner = EvalRunner(agent=FakeAgent())
        report = runner.run(
            [
                EvalCase(
                    name="answer_check",
                    prompt="What is the answer?",
                    expected_output_contains=["42"],
                ),
            ]
        )

        assert report.total_cases == 1
        assert report.passed == 1
        assert report.failed == 0
        assert report.results[0].passed
        assert report.results[0].score == 1.0

    def test_run_failing_case(self):
        """Runner evaluates a failing case."""
        from locus.evaluation import EvalCase, EvalRunner

        class FakeAgent:
            def run_sync(self, prompt):
                return AgentResult(
                    message="I don't know.",
                    state=AgentState(agent_id="test"),
                    stop_reason="complete",
                )

        runner = EvalRunner(agent=FakeAgent())
        report = runner.run(
            [
                EvalCase(
                    name="missing_answer",
                    prompt="What is 2+2?",
                    expected_output_contains=["4"],
                ),
            ]
        )

        assert report.total_cases == 1
        assert report.passed == 0
        assert report.failed == 1
        assert not report.results[0].passed

    def test_run_tool_check(self):
        """Runner checks tool usage."""
        from locus.evaluation import EvalCase, EvalRunner

        class FakeAgent:
            def run_sync(self, prompt):
                state = AgentState(agent_id="test")
                state = state.with_tool_execution(
                    ToolExecution(
                        tool_name="search",
                        tool_call_id="call_search_1",
                        arguments={"q": "test"},
                        result="found",
                    )
                )
                return AgentResult(
                    message="Found results using search.",
                    state=state,
                    stop_reason="complete",
                )

        runner = EvalRunner(agent=FakeAgent())
        report = runner.run(
            [
                EvalCase(
                    name="tool_usage",
                    prompt="Search for something",
                    expected_tools=["search"],
                    expected_output_contains=["results"],
                ),
            ]
        )

        assert report.passed == 1
        assert report.results[0].tools_called == ["search"]
        assert report.results[0].checks["tool_called:search"]

    def test_run_not_contains_check(self):
        """Runner checks output does NOT contain excluded strings."""
        from locus.evaluation import EvalCase, EvalRunner

        class FakeAgent:
            def run_sync(self, prompt):
                return AgentResult(
                    message="The operation succeeded.",
                    state=AgentState(agent_id="test"),
                    stop_reason="complete",
                )

        runner = EvalRunner(agent=FakeAgent())
        report = runner.run(
            [
                EvalCase(
                    name="no_error",
                    prompt="Do something",
                    expected_output_not_contains=["error", "failed"],
                ),
            ]
        )

        assert report.passed == 1

    def test_run_iteration_budget(self):
        """Runner checks iteration budget."""
        from locus.evaluation import EvalCase, EvalRunner

        class FakeAgent:
            def run_sync(self, prompt):
                state = AgentState(agent_id="test")
                # Simulate 3 iterations
                state = state.with_iteration(3)
                return AgentResult(
                    message="Done.",
                    state=state,
                    stop_reason="complete",
                )

        runner = EvalRunner(agent=FakeAgent())

        # Within budget
        report = runner.run(
            [
                EvalCase(name="within", prompt="task", max_iterations=5),
            ]
        )
        assert report.passed == 1

        # Over budget
        report = runner.run(
            [
                EvalCase(name="over", prompt="task", max_iterations=2),
            ]
        )
        assert report.passed == 0

    def test_run_multiple_cases(self):
        """Runner evaluates multiple cases."""
        from locus.evaluation import EvalCase, EvalRunner

        call_count = 0

        class FakeAgent:
            def run_sync(self, prompt):
                nonlocal call_count
                call_count += 1
                return AgentResult(
                    message=f"Response {call_count}",
                    state=AgentState(agent_id="test"),
                    stop_reason="complete",
                )

        runner = EvalRunner(agent=FakeAgent())
        report = runner.run(
            [
                EvalCase(name="case1", prompt="p1", expected_output_contains=["response"]),
                EvalCase(name="case2", prompt="p2", expected_output_contains=["response"]),
                EvalCase(name="case3", prompt="p3", expected_output_contains=["missing"]),
            ]
        )

        assert report.total_cases == 3
        assert report.passed == 2
        assert report.failed == 1
        assert 0.5 < report.avg_score < 1.0

    def test_run_handles_agent_error(self):
        """Runner handles agent exceptions gracefully."""
        from locus.evaluation import EvalCase, EvalRunner

        class CrashingAgent:
            def run_sync(self, prompt):
                raise RuntimeError("Agent exploded")

        runner = EvalRunner(agent=CrashingAgent())
        report = runner.run(
            [
                EvalCase(name="crash", prompt="boom"),
            ]
        )

        assert report.failed == 1
        assert report.results[0].error == "Agent exploded"

    def test_report_summary(self):
        """Report generates human-readable summary."""
        from locus.evaluation import EvalReport, EvalResult

        report = EvalReport(
            results=[
                EvalResult(case_name="pass1", passed=True, score=1.0, duration_ms=100),
                EvalResult(
                    case_name="fail1",
                    passed=False,
                    score=0.5,
                    duration_ms=200,
                    checks={"output_contains:foo": False, "tool_called:bar": True},
                ),
            ],
            total_cases=2,
            passed=1,
            failed=1,
            avg_score=0.75,
            total_duration_ms=300,
        )

        summary = report.summary()
        assert "1/2 passed" in summary
        assert "PASS" in summary
        assert "FAIL" in summary
        assert "output_contains:foo" in summary


# =============================================================================
# Pre/Post Model Hooks Tests
# =============================================================================


class TestModelHooks:
    """Tests for pre/post model call hooks."""

    @pytest.mark.asyncio
    async def test_before_model_hook_called(self):
        """on_before_model_call hook is invoked before model.complete()."""
        from locus.hooks.provider import HookProvider

        hook_calls = []

        class TrackingHook(HookProvider):
            @property
            def priority(self):
                return 100

            async def on_before_model_call(self, event):
                hook_calls.append(("before", len(event.messages)))

            async def on_after_model_call(self, event):
                hook_calls.append(("after", event.response.message.content))

        mock_model = MagicMock()
        mock_model.complete = AsyncMock(
            return_value=ModelResponse(
                message=Message.assistant("Test response"),
            )
        )

        agent = Agent(
            config=AgentConfig(
                system_prompt="Test",
                max_iterations=1,
                model=mock_model,
                hooks=[TrackingHook()],
            )
        )

        result = agent.run_sync("Hello")

        assert len(hook_calls) >= 2
        assert hook_calls[0][0] == "before"
        assert hook_calls[1][0] == "after"
        assert hook_calls[1][1] == "Test response"

    @pytest.mark.asyncio
    async def test_before_model_hook_modifies_messages(self):
        """on_before_model_call can modify messages before sending to model."""
        from locus.hooks.provider import HookProvider

        captured_messages = []

        class TrimHook(HookProvider):
            @property
            def priority(self):
                return 100

            async def on_before_model_call(self, event):
                # Keep only last 2 messages (system + last user)
                event.messages = (
                    [event.messages[0], event.messages[-1]]
                    if len(event.messages) > 2
                    else event.messages
                )

        mock_model = MagicMock()

        async def capture_complete(messages, **kwargs):
            captured_messages.extend(messages)
            return ModelResponse(message=Message.assistant("Done"))

        mock_model.complete = capture_complete

        agent = Agent(
            config=AgentConfig(
                system_prompt="System prompt",
                max_iterations=1,
                model=mock_model,
                hooks=[TrimHook()],
            )
        )

        result = agent.run_sync("User message")

        # The hook should have trimmed to 2 messages
        assert len(captured_messages) == 2

    @pytest.mark.asyncio
    async def test_after_model_hook_modifies_response(self):
        """on_after_model_call can modify the model response."""
        from locus.hooks.provider import HookProvider

        class FilterHook(HookProvider):
            @property
            def priority(self):
                return 100

            async def on_after_model_call(self, event):
                # Replace content in response
                new_msg = Message.assistant("Filtered: " + (event.response.message.content or ""))
                event.response = ModelResponse(message=new_msg)

        mock_model = MagicMock()
        mock_model.complete = AsyncMock(
            return_value=ModelResponse(message=Message.assistant("Original"))
        )

        agent = Agent(
            config=AgentConfig(
                system_prompt="Test",
                max_iterations=1,
                model=mock_model,
                hooks=[FilterHook()],
            )
        )

        result = agent.run_sync("Hello")

        assert "Filtered: Original" in result.message

    @pytest.mark.asyncio
    async def test_multiple_model_hooks_chain(self):
        """Multiple hooks chain in priority order."""
        from locus.hooks.provider import HookProvider

        order = []

        class HookA(HookProvider):
            @property
            def priority(self):
                return 50

            async def on_before_model_call(self, event):
                order.append("A")

        class HookB(HookProvider):
            @property
            def priority(self):
                return 100

            async def on_before_model_call(self, event):
                order.append("B")

        mock_model = MagicMock()
        mock_model.complete = AsyncMock(
            return_value=ModelResponse(message=Message.assistant("Done"))
        )

        agent = Agent(
            config=AgentConfig(
                system_prompt="Test",
                max_iterations=1,
                model=mock_model,
                hooks=[HookA(), HookB()],
            )
        )

        result = agent.run_sync("Hello")

        # Hooks execute in insertion order
        assert order[0] == "A"
        assert order[1] == "B"

    def test_hook_provider_has_model_hooks(self):
        """HookProvider base class has model hook methods."""
        from locus.hooks.provider import HookProvider

        class MinimalHook(HookProvider):
            @property
            def priority(self):
                return 100

        hook = MinimalHook()
        hooks = hook.register_hooks()
        assert "on_before_model_call" in hooks
        assert "on_after_model_call" in hooks

    @pytest.mark.asyncio
    async def test_hook_registry_model_hooks(self):
        """HookRegistry dispatches model hook events."""
        from locus.hooks.provider import HookProvider
        from locus.hooks.registry import HookRegistry

        calls = []

        class TestHook(HookProvider):
            @property
            def priority(self):
                return 100

            async def on_before_model_call(self, event):
                calls.append("before")
                event.messages = event.messages + [Message.system("injected")]

            async def on_after_model_call(self, event):
                calls.append("after")

        registry = HookRegistry()
        registry.add_provider(TestHook())

        messages = [Message.user("Hello")]
        result = await registry.emit_before_model_call(messages, None)

        assert len(result) == 2  # Original + injected
        assert calls == ["before"]

        response = ModelResponse(message=Message.assistant("Hi"))
        await registry.emit_after_model_call(response, result)
        assert calls == ["before", "after"]


class TestHookControlFlow:
    """Tests for hook control flow via write-protected events."""

    @pytest.mark.asyncio
    async def test_cancel_tool_via_event(self):
        """Hook cancels a tool call via event.cancel."""
        from locus.hooks.provider import HookProvider

        class BlockDangerousTool(HookProvider):
            @property
            def priority(self):
                return 50

            async def on_before_tool_call(self, event):
                if event.tool_name == "dangerous_tool":
                    event.cancel = "Tool blocked by security policy"

        @tool
        def dangerous_tool(x: str) -> str:
            """A dangerous tool."""
            return f"executed: {x}"

        @tool
        def safe_tool(x: str) -> str:
            """A safe tool."""
            return f"safe: {x}"

        mock_model = MagicMock()
        call_count = 0

        async def model_fn(messages, tools=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    message=Message.assistant(
                        content="",
                        tool_calls=[
                            ToolCall(id="c1", name="dangerous_tool", arguments={"x": "hack"})
                        ],
                    ),
                )
            return ModelResponse(message=Message.assistant("Done"))

        mock_model.complete = model_fn

        agent = Agent(
            config=AgentConfig(
                system_prompt="Test",
                max_iterations=3,
                model=mock_model,
                tools=[dangerous_tool, safe_tool],
                hooks=[BlockDangerousTool()],
            )
        )

        result = agent.run_sync("Do something dangerous")

        # The dangerous tool should have been cancelled, not executed
        tool_results = [te for te in result.tool_executions if te.tool_name == "dangerous_tool"]
        assert len(tool_results) == 1
        assert tool_results[0].result == "Tool blocked by security policy"
        assert tool_results[0].error is None  # Not an error, just cancelled

    @pytest.mark.asyncio
    async def test_retry_model_via_event(self):
        """Hook retries model call via event.retry = True."""
        from locus.hooks.provider import HookProvider

        retry_count = 0

        class RetryOnEmpty(HookProvider):
            @property
            def priority(self):
                return 100

            async def on_after_model_call(self, event):
                nonlocal retry_count
                if not event.response.message.content and retry_count == 0:
                    retry_count += 1
                    event.retry = True

        model_calls = 0
        mock_model = MagicMock()

        async def model_fn(messages, **kwargs):
            nonlocal model_calls
            model_calls += 1
            if model_calls == 1:
                return ModelResponse(message=Message.assistant(""))
            return ModelResponse(message=Message.assistant("Real answer"))

        mock_model.complete = model_fn

        agent = Agent(
            config=AgentConfig(
                system_prompt="Test",
                max_iterations=2,
                model=mock_model,
                hooks=[RetryOnEmpty()],
            )
        )

        result = agent.run_sync("Hello")

        assert model_calls == 2  # First empty + retry
        assert "Real answer" in result.message

    @pytest.mark.asyncio
    async def test_retry_tool_via_event(self):
        """Hook retries tool call via event.retry = True."""
        from locus.hooks.provider import HookProvider

        tool_attempts = 0

        class RetryFailedTool(HookProvider):
            @property
            def priority(self):
                return 100

            async def on_after_tool_call(self, event):
                nonlocal tool_attempts
                tool_attempts += 1
                if event.error and tool_attempts == 1:
                    event.retry = True

        exec_count = 0

        @tool
        def flaky_tool(x: str) -> str:
            """A flaky tool."""
            nonlocal exec_count
            exec_count += 1
            if exec_count == 1:
                raise RuntimeError("Transient failure")
            return f"success: {x}"

        mock_model = MagicMock()
        call_count = 0

        async def model_fn(messages, tools=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    message=Message.assistant(
                        content="",
                        tool_calls=[ToolCall(id="c1", name="flaky_tool", arguments={"x": "test"})],
                    ),
                )
            return ModelResponse(message=Message.assistant("Done"))

        mock_model.complete = model_fn

        agent = Agent(
            config=AgentConfig(
                system_prompt="Test",
                max_iterations=3,
                model=mock_model,
                tools=[flaky_tool],
                hooks=[RetryFailedTool()],
            )
        )

        result = agent.run_sync("Use flaky tool")

        assert exec_count == 2  # First fail + retry

    def test_write_protection_blocks_readonly_fields(self):
        """Setting a read-only field on an event raises AttributeError."""
        from locus.hooks.provider import BeforeToolCallEvent

        event = BeforeToolCallEvent(tool_name="test", tool_call_id="c1", arguments={"x": 1})

        # Writable fields work
        event.arguments = {"x": 2}
        event.cancel = "blocked"
        assert event.arguments == {"x": 2}
        assert event.cancel == "blocked"

        # Read-only fields raise
        with pytest.raises(AttributeError, match="read-only"):
            event.tool_name = "hacked"

        with pytest.raises(AttributeError, match="read-only"):
            event.tool_call_id = "fake"

    def test_write_protection_on_model_events(self):
        """Model events protect read-only fields."""
        from locus.hooks.provider import AfterModelCallEvent, BeforeModelCallEvent

        before = BeforeModelCallEvent(messages=[], tools=None)
        before.messages = [Message.user("ok")]  # writable
        with pytest.raises(AttributeError, match="read-only"):
            before.tools = [{"fake": True}]

        after = AfterModelCallEvent(response="resp", messages=[])
        after.retry = True  # writable
        after.response = "new"  # writable
        with pytest.raises(AttributeError, match="read-only"):
            after.messages = []

    @pytest.mark.asyncio
    async def test_after_hooks_run_in_reverse_order(self):
        """After hooks fire in reverse order (last-registered-first)."""
        from locus.hooks.provider import HookProvider

        order = []

        class HookA(HookProvider):
            @property
            def priority(self):
                return 100

            async def on_after_model_call(self, event):
                order.append("A")

        class HookB(HookProvider):
            @property
            def priority(self):
                return 200

            async def on_after_model_call(self, event):
                order.append("B")

        mock_model = MagicMock()
        mock_model.complete = AsyncMock(
            return_value=ModelResponse(message=Message.assistant("Done"))
        )

        agent = Agent(
            config=AgentConfig(
                system_prompt="Test",
                max_iterations=1,
                model=mock_model,
                hooks=[HookA(), HookB()],
            )
        )

        result = agent.run_sync("Hello")

        # After hooks: B first (last registered), then A
        assert order == ["B", "A"]


# =============================================================================
# Security Hardening Tests
# =============================================================================


class TestErrorSanitization:
    """Tests for error message sanitization in tool execution."""

    def test_sanitize_connection_string(self):
        """Connection strings are redacted from error messages."""
        from locus.tools.executor import _sanitize_error

        error = (
            "OperationalError: could not connect to postgresql://admin:s3cret@db.internal:5432/prod"
        )
        sanitized = _sanitize_error(error)
        assert "s3cret" not in sanitized
        assert "admin" not in sanitized
        assert "[REDACTED]" in sanitized

    def test_sanitize_file_path(self):
        """Home directory paths are redacted."""
        from locus.tools.executor import _sanitize_error

        error = "FileNotFoundError: /Users/john.doe/Projects/secret/config.yaml"
        sanitized = _sanitize_error(error)
        assert "john.doe" not in sanitized
        assert "[REDACTED]" in sanitized

    def test_sanitize_api_key(self):
        """API keys in errors are redacted."""
        from locus.tools.executor import _sanitize_error

        error = "AuthError: invalid api_key=sk-proj-abc123def456"
        sanitized = _sanitize_error(error)
        assert "sk-proj" not in sanitized
        assert "[REDACTED]" in sanitized

    def test_sanitize_oci_ocid(self):
        """OCI resource IDs are redacted."""
        from locus.tools.executor import _sanitize_error

        error = "NotFound: ocid1.compartment.oc1..aaaaaaaa00000000000000000000000000000000000000000000000000000000"
        sanitized = _sanitize_error(error)
        assert "aaaaaa" not in sanitized
        assert "[REDACTED]" in sanitized

    def test_safe_error_passes_through(self):
        """Normal errors pass through unchanged."""
        from locus.tools.executor import _sanitize_error

        error = "ValueError: expected int, got str"
        sanitized = _sanitize_error(error)
        assert sanitized == error

    def test_multiline_error_first_line_only(self):
        """Only first line of error is kept."""
        from locus.tools.executor import _sanitize_error

        error = "Error: something\nTraceback details\nMore internal info"
        sanitized = _sanitize_error(error)
        assert "\n" not in sanitized
        assert "Traceback" not in sanitized


class TestTextToolCallValidation:
    """Tests for _parse_text_tool_calls schema validation."""

    @pytest.mark.asyncio
    async def test_parsed_args_validated_against_schema(self):
        """Parsed text tool calls only keep args declared in schema."""
        mock_model = MagicMock()
        mock_model.complete = AsyncMock(
            return_value=ModelResponse(message=Message.assistant("Done")),
        )

        @tool
        def search(query: str) -> str:
            """Search for something."""
            return f"results for {query}"

        agent = Agent(
            config=AgentConfig(
                system_prompt="Test",
                max_iterations=1,
                model=mock_model,
                tools=[search],
            )
        )
        agent._initialize()

        # Simulate model text with injected extra args
        parsed = agent._parse_text_tool_calls(
            'search(query="test", evil_param="DROP TABLE", __import__="os")'
        )

        assert len(parsed) == 1
        # Only "query" should survive — evil_param and __import__ filtered out
        assert "query" in parsed[0].arguments
        assert "evil_param" not in parsed[0].arguments
        assert "__import__" not in parsed[0].arguments

    @pytest.mark.asyncio
    async def test_unregistered_tool_ignored(self):
        """Text tool calls for unregistered tools are ignored."""
        mock_model = MagicMock()
        mock_model.complete = AsyncMock(
            return_value=ModelResponse(message=Message.assistant("Done")),
        )

        @tool
        def safe_tool(x: str) -> str:
            """A safe tool."""
            return x

        agent = Agent(
            config=AgentConfig(
                system_prompt="Test",
                max_iterations=1,
                model=mock_model,
                tools=[safe_tool],
            )
        )
        agent._initialize()

        parsed = agent._parse_text_tool_calls('os.system("rm -rf /") and safe_tool(x="hello")')

        # Only safe_tool should be parsed, os.system ignored
        assert len(parsed) == 1
        assert parsed[0].name == "safe_tool"


# =============================================================================
# Agent Server Tests
# =============================================================================


class TestAgentServer:
    """Tests for AgentServer HTTP wrapper."""

    def test_server_creates_app(self):
        """AgentServer creates a FastAPI app."""
        pytest.importorskip("fastapi")
        from locus.server import AgentServer

        mock_agent = MagicMock()
        server = AgentServer(agent=mock_agent, title="Test Server")

        app = server.app
        assert app is not None
        assert app.title == "Test Server"

    def test_health_endpoint(self):
        """Health endpoint returns ok."""
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from locus.server import AgentServer

        mock_agent = MagicMock()
        server = AgentServer(agent=mock_agent)
        client = TestClient(server.app)

        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_invoke_endpoint(self):
        """Invoke endpoint iterates agent.run() and returns the final message."""
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from locus.core.events import TerminateEvent
        from locus.server import AgentServer

        async def fake_run(*_args, **_kwargs):
            yield TerminateEvent(
                final_message="Hello!",
                reason="complete",
                iterations_used=1,
                final_confidence=1.0,
                total_tool_calls=0,
            )

        mock_agent = MagicMock()
        mock_agent.run.side_effect = fake_run

        server = AgentServer(agent=mock_agent)
        client = TestClient(server.app)

        response = client.post("/invoke", json={"prompt": "Hi"})
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Hello!"
        assert data["success"] is True
        assert data["stop_reason"] == "complete"
        mock_agent.run.assert_called_once()

    def test_invoke_with_thread_id(self):
        """Invoke scopes thread_id with the caller principal before passing."""
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from locus.core.events import TerminateEvent
        from locus.server import AgentServer

        async def fake_run(*_args, **_kwargs):
            yield TerminateEvent(
                final_message="Ok",
                reason="complete",
                iterations_used=1,
                final_confidence=1.0,
                total_tool_calls=0,
            )

        mock_agent = MagicMock()
        mock_agent.run.side_effect = fake_run

        server = AgentServer(agent=mock_agent)
        client = TestClient(server.app)

        response = client.post(
            "/invoke",
            json={"prompt": "Hi", "thread_id": "thread-123"},
        )
        assert response.status_code == 200
        call_kwargs = mock_agent.run.call_args
        # Anonymous principal scoping means thread ids are prefixed.
        assert call_kwargs.kwargs.get("thread_id") == "anon:thread-123"
