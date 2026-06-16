# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for agent result classes."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from locus.agent.result import (
    AgentResult,
    ExecutionMetrics,
    StreamingResult,
)
from locus.core.messages import Message, Role
from locus.core.state import AgentState


class TestExecutionMetrics:
    """Tests for ExecutionMetrics."""

    def test_default_metrics(self):
        """Test creating metrics with defaults."""
        metrics = ExecutionMetrics()
        assert metrics.iterations == 0
        assert metrics.tool_calls == 0
        assert metrics.tool_errors == 0
        assert metrics.total_tokens == 0
        assert metrics.prompt_tokens == 0
        assert metrics.completion_tokens == 0
        assert metrics.duration_ms == 0.0

    def test_custom_metrics(self):
        """Test creating metrics with custom values."""
        metrics = ExecutionMetrics(
            iterations=5,
            tool_calls=10,
            tool_errors=2,
            total_tokens=1000,
            prompt_tokens=600,
            completion_tokens=400,
            duration_ms=500.0,
            reflexion_evaluations=3,
            grounding_evaluations=1,
        )
        assert metrics.iterations == 5
        assert metrics.tool_calls == 10
        assert metrics.tool_errors == 2

    def test_tools_success_rate_no_calls(self):
        """Test success rate when no tool calls."""
        metrics = ExecutionMetrics(tool_calls=0)
        assert metrics.tools_success_rate == 1.0

    def test_tools_success_rate_all_success(self):
        """Test success rate with all successful calls."""
        metrics = ExecutionMetrics(tool_calls=10, tool_errors=0)
        assert metrics.tools_success_rate == 1.0

    def test_tools_success_rate_with_errors(self):
        """Test success rate with some errors."""
        metrics = ExecutionMetrics(tool_calls=10, tool_errors=3)
        assert metrics.tools_success_rate == 0.7

    def test_tokens_per_iteration_no_iterations(self):
        """Test tokens per iteration when no iterations."""
        metrics = ExecutionMetrics(iterations=0, total_tokens=100)
        assert metrics.tokens_per_iteration == 0.0

    def test_tokens_per_iteration_normal(self):
        """Test tokens per iteration calculation."""
        metrics = ExecutionMetrics(iterations=5, total_tokens=1000)
        assert metrics.tokens_per_iteration == 200.0

    def test_metrics_are_frozen(self):
        """Test that metrics are immutable."""
        metrics = ExecutionMetrics()
        with pytest.raises(ValidationError):
            metrics.iterations = 10


class TestAgentResult:
    """Tests for AgentResult."""

    @pytest.fixture
    def state(self):
        """Create test state."""
        return AgentState()

    @pytest.fixture
    def state_with_messages(self):
        """Create state with messages."""
        state = AgentState()
        state = state.with_message(Message(role=Role.USER, content="Hello"))
        state = state.with_message(Message(role=Role.ASSISTANT, content="Hi there!"))
        return state

    def test_minimal_result(self, state):
        """Test creating result with minimal fields."""
        result = AgentResult(
            message="Hello",
            state=state,
            stop_reason="complete",
        )
        assert result.message == "Hello"
        assert result.stop_reason == "complete"

    def test_text_alias_mirrors_message(self, state):
        """``.text`` is an alias for ``.message`` for SDK ergonomics parity."""
        result = AgentResult(message="ping", state=state, stop_reason="complete")
        assert result.text == "ping"
        assert result.text == result.message

    def test_full_result(self, state):
        """Test creating result with all fields."""
        metrics = ExecutionMetrics(iterations=5)
        now = datetime.now(UTC)

        result = AgentResult(
            message="Done",
            state=state,
            stop_reason="terminal_tool",
            metrics=metrics,
            started_at=now,
            completed_at=now,
            error=None,
            grounding_score=0.9,
            ungrounded_claims=["claim1"],
        )
        assert result.metrics.iterations == 5
        assert result.grounding_score == 0.9

    def test_success_complete(self, state):
        """Test success property for complete."""
        result = AgentResult(message="", state=state, stop_reason="complete")
        assert result.success is True

    def test_success_terminal_tool(self, state):
        """Test success property for terminal_tool."""
        result = AgentResult(message="", state=state, stop_reason="terminal_tool")
        assert result.success is True

    def test_success_confidence_met(self, state):
        """Test success property for confidence_met."""
        result = AgentResult(message="", state=state, stop_reason="confidence_met")
        assert result.success is True

    def test_success_error(self, state):
        """Test success property for error."""
        result = AgentResult(message="", state=state, stop_reason="error")
        assert result.success is False

    def test_success_max_iterations(self, state):
        """Test success property for max_iterations."""
        result = AgentResult(message="", state=state, stop_reason="max_iterations")
        assert result.success is False

    def test_confidence_property(self, state):
        """Test confidence property."""
        state = state.with_confidence(0.85)
        result = AgentResult(message="", state=state, stop_reason="complete")
        assert result.confidence == 0.85

    def test_iterations_property(self, state):
        """Test iterations property."""
        for _ in range(3):
            state = state.next_iteration()
        result = AgentResult(message="", state=state, stop_reason="complete")
        assert result.iterations == 3

    def test_messages_property(self, state_with_messages):
        """Test messages property."""
        result = AgentResult(message="", state=state_with_messages, stop_reason="complete")
        assert len(result.messages) == 2

    def test_last_assistant_message(self, state_with_messages):
        """Test last_assistant_message property."""
        result = AgentResult(message="", state=state_with_messages, stop_reason="complete")
        assert result.last_assistant_message == "Hi there!"

    def test_last_assistant_message_none(self, state):
        """Test last_assistant_message with no assistant messages."""
        result = AgentResult(message="", state=state, stop_reason="complete")
        assert result.last_assistant_message is None

    def test_to_dict(self, state):
        """Test to_dict export."""
        result = AgentResult(message="Hello", state=state, stop_reason="complete")
        d = result.to_dict()
        assert d["message"] == "Hello"
        assert d["stop_reason"] == "complete"

    def test_from_state(self, state_with_messages):
        """Test from_state factory method."""
        result = AgentResult.from_state(
            state=state_with_messages,
            stop_reason="complete",
        )
        assert result.message == "Hi there!"
        assert result.stop_reason == "complete"

    def test_from_state_with_metrics(self, state):
        """Test from_state with metrics."""
        metrics = ExecutionMetrics(iterations=5)
        result = AgentResult.from_state(
            state=state,
            stop_reason="error",
            metrics=metrics,
            error="Something went wrong",
        )
        assert result.metrics.iterations == 5
        assert result.error == "Something went wrong"

    def test_from_state_with_grounding(self, state):
        """Test from_state with grounding info."""
        result = AgentResult.from_state(
            state=state,
            stop_reason="grounding_failed",
            grounding_score=0.4,
            ungrounded_claims=["claim1", "claim2"],
        )
        assert result.grounding_score == 0.4
        assert result.ungrounded_claims == ["claim1", "claim2"]

    def test_result_is_frozen(self, state):
        """Test that result is immutable."""
        result = AgentResult(message="", state=state, stop_reason="complete")
        with pytest.raises(ValidationError):
            result.message = "New message"


class TestStreamingResult:
    """Tests for StreamingResult."""

    @pytest.fixture
    def state(self):
        """Create test state."""
        return AgentState()

    def test_default_streaming_result(self, state):
        """Test creating streaming result with defaults."""
        result = StreamingResult(state=state)
        assert result.partial_content == ""
        assert result.iteration == 0
        assert result.is_complete is False
        assert result.final is None

    def test_streaming_result_with_content(self, state):
        """Test creating streaming result with content."""
        result = StreamingResult(
            state=state,
            partial_content="Hello world",
            iteration=2,
        )
        assert result.partial_content == "Hello world"
        assert result.iteration == 2

    def test_streaming_result_complete(self, state):
        """Test creating complete streaming result."""
        final = AgentResult(message="Done", state=state, stop_reason="complete")
        result = StreamingResult(
            state=state,
            is_complete=True,
            final=final,
        )
        assert result.is_complete is True
        assert result.final is not None
        assert result.final.message == "Done"

    def test_streaming_result_is_frozen(self, state):
        """Test that streaming result is immutable."""
        result = StreamingResult(state=state)
        with pytest.raises(ValidationError):
            result.iteration = 5
