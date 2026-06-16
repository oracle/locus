# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Tests for agent state."""

import pytest
from pydantic import ValidationError

from locus.core.messages import Message, ToolCall
from locus.core.state import AgentState, ReasoningStep, ToolExecution


class TestAgentState:
    """Tests for AgentState."""

    def test_create_default_state(self):
        """Create state with defaults."""
        state = AgentState()

        assert state.iteration == 0
        assert state.max_iterations == 20
        assert state.confidence == 0.0
        assert state.messages == ()
        assert state.errors == ()

    def test_state_is_frozen(self):
        """State is immutable."""
        state = AgentState()

        with pytest.raises(ValidationError):
            state.iteration = 5  # type: ignore[misc]

    def test_with_message(self):
        """Add a message to state."""
        state = AgentState()
        msg = Message.user("Hello!")

        new_state = state.with_message(msg)

        # Original unchanged
        assert state.messages == ()

        # New state has message
        assert len(new_state.messages) == 1
        assert new_state.messages[0].content == "Hello!"

    def test_with_messages(self):
        """Add multiple messages."""
        state = AgentState()
        messages = [Message.user("Hello!"), Message.assistant("Hi!")]

        new_state = state.with_messages(messages)

        assert len(new_state.messages) == 2

    def test_next_iteration(self):
        """Increment iteration counter."""
        state = AgentState()

        state = state.next_iteration()
        assert state.iteration == 1

        state = state.next_iteration()
        assert state.iteration == 2

    def test_with_confidence(self):
        """Update confidence score."""
        state = AgentState()

        new_state = state.with_confidence(0.75)

        assert state.confidence == 0.0  # Original unchanged
        assert new_state.confidence == 0.75
        assert new_state.confidence_history == (0.75,)

    def test_confidence_clamped(self):
        """Confidence is clamped to [0, 1]."""
        state = AgentState()

        assert state.with_confidence(-0.5).confidence == 0.0
        assert state.with_confidence(1.5).confidence == 1.0

    def test_adjust_confidence_diminishing(self):
        """Diminishing returns for confidence adjustment."""
        state = AgentState().with_confidence(0.8)

        # With diminishing returns, delta is scaled by (1 - current)
        new_state = state.adjust_confidence(0.5, diminishing=True)

        # Expected: 0.8 + 0.5 * (1 - 0.8) = 0.8 + 0.1 = 0.9
        assert new_state.confidence == pytest.approx(0.9)

    def test_adjust_confidence_no_diminishing(self):
        """No diminishing returns for confidence adjustment."""
        state = AgentState().with_confidence(0.8)

        new_state = state.adjust_confidence(0.1, diminishing=False)

        assert new_state.confidence == pytest.approx(0.9)

    def test_with_error(self):
        """Record an error."""
        state = AgentState()

        new_state = state.with_error("Something went wrong")

        assert len(new_state.errors) == 1
        assert new_state.errors[0] == "Something went wrong"

    def test_with_metadata(self):
        """Set metadata value."""
        state = AgentState()

        new_state = state.with_metadata("user_id", "123")

        assert new_state.metadata["user_id"] == "123"

    def test_tool_loop_detection(self):
        """Detect tool loops across iterations (not within parallel calls)."""
        state = AgentState(tool_loop_threshold=3)

        # Simulate 3 iterations each calling the same single tool
        for i in range(3):
            step = ReasoningStep(
                iteration=i + 1,
                thought=f"Search {i}",
                tool_calls=[ToolCall(name="search", arguments={"q": "test"})],
            )
            state = state.with_reasoning_step(step)
            exec_ = ToolExecution(
                tool_name="search",
                tool_call_id=f"call_{i}",
                arguments={"q": "test"},
            )
            state = state.with_tool_execution(exec_)

        assert state.has_tool_loop is True

    def test_parallel_calls_not_a_loop(self):
        """Multiple calls to the same tool in ONE iteration is NOT a loop."""
        state = AgentState(tool_loop_threshold=3)

        # ONE iteration with 5 parallel calls to the same tool
        step = ReasoningStep(
            iteration=1,
            thought="Searching multiple topics",
            tool_calls=[ToolCall(name="search", arguments={"q": f"topic{i}"}) for i in range(5)],
        )
        state = state.with_reasoning_step(step)
        for i in range(5):
            state = state.with_tool_execution(
                ToolExecution(
                    tool_name="search", tool_call_id=f"c{i}", arguments={"q": f"topic{i}"}
                )
            )

        assert state.has_tool_loop is False

    def test_no_tool_loop_with_variety(self):
        """No loop with varied tools across iterations."""
        state = AgentState(tool_loop_threshold=3)

        for i, name in enumerate(["search", "calculate", "search"]):
            step = ReasoningStep(
                iteration=i + 1,
                thought=f"Step {i}",
                tool_calls=[ToolCall(name=name, arguments={})],
            )
            state = state.with_reasoning_step(step)
            exec_ = ToolExecution(
                tool_name=name,
                tool_call_id=f"call_{i}",
                arguments={},
            )
            state = state.with_tool_execution(exec_)

        assert state.has_tool_loop is False

    def test_no_tool_loop_with_same_name_different_args(self):
        """Same tool with different arguments is forward progress, not a loop.

        Regression for the false-positive that fired on paged-discovery
        patterns: e.g. three sequential calls to ``list_metric_names``
        with three different ``regex`` values are all making progress
        and must not be flagged.
        """
        state = AgentState(tool_loop_threshold=3)

        for i, regex in enumerate(["fusion:database_.*", "fusion:db_.*", "fusion:dbnode_.*"]):
            step = ReasoningStep(
                iteration=i + 1,
                thought=f"Discovery batch {i}",
                tool_calls=[ToolCall(name="list_metric_names", arguments={"regex": regex})],
            )
            state = state.with_reasoning_step(step)
            exec_ = ToolExecution(
                tool_name="list_metric_names",
                tool_call_id=f"call_{i}",
                arguments={"regex": regex},
            )
            state = state.with_tool_execution(exec_)

        assert state.has_tool_loop is False

    def test_tool_loop_canonical_arg_order(self):
        """Loop detection ignores dict-key order in arguments.

        ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}`` represent the same
        call and must hash to the same signature so the detector
        catches a real loop even when the model emits keys in different
        orders across iterations.
        """
        state = AgentState(tool_loop_threshold=3)

        arg_variants = [
            {"a": 1, "b": 2},
            {"b": 2, "a": 1},
            {"a": 1, "b": 2},
        ]
        for i, args in enumerate(arg_variants):
            step = ReasoningStep(
                iteration=i + 1,
                thought=f"Step {i}",
                tool_calls=[ToolCall(name="search", arguments=args)],
            )
            state = state.with_reasoning_step(step)
            state = state.with_tool_execution(
                ToolExecution(tool_name="search", tool_call_id=f"call_{i}", arguments=args)
            )

        assert state.has_tool_loop is True

    def test_should_terminate_max_iterations(self):
        """Terminate at max iterations."""
        state = AgentState(max_iterations=5, iteration=5)

        should_stop, reason = state.should_terminate
        assert should_stop is True
        assert reason == "max_iterations"

    def test_should_terminate_confidence(self):
        """Terminate when confidence threshold met."""
        state = AgentState(confidence_threshold=0.85, confidence=0.9)

        should_stop, reason = state.should_terminate
        assert should_stop is True
        assert reason == "confidence_met"

    def test_should_not_terminate_early(self):
        """Don't terminate when conditions not met."""
        state = AgentState(
            iteration=1,
            max_iterations=10,
            confidence=0.5,
            confidence_threshold=0.85,
        )

        should_stop, reason = state.should_terminate
        assert should_stop is False
        assert reason is None

    def test_no_tools_terminates_after_assistant_reply(self):
        """Terminate with 'no_tools' when last message is an assistant reply without tool calls."""
        state = AgentState(iteration=1, max_iterations=10)
        state = state.with_message(Message.user("hi"))
        state = state.with_message(Message.assistant("Hello!"))

        should_stop, reason = state.should_terminate
        assert should_stop is True
        assert reason == "no_tools"

    def test_no_tools_does_not_fire_on_unanswered_user_message(self):
        """Multi-turn regression: when the checkpointer replays prior history and
        appends a new user message, `should_terminate` must NOT fire 'no_tools'
        — otherwise the model is never called for the new turn."""
        state = AgentState(iteration=1, max_iterations=10)
        state = state.with_message(Message.user("hi"))
        state = state.with_message(Message.assistant("Hello!"))
        state = state.with_message(Message.user("whats your job"))

        should_stop, reason = state.should_terminate
        assert should_stop is False, (
            f"Expected agent to keep running to answer the new user message, "
            f"got should_stop={should_stop}, reason={reason}"
        )
        assert reason is None

    def test_checkpoint_roundtrip(self):
        """State can be serialized and restored."""
        state = AgentState(
            iteration=5,
            confidence=0.75,
        ).with_message(Message.user("Hello!"))

        data = state.to_checkpoint()
        restored = AgentState.from_checkpoint(data)

        assert restored.iteration == 5
        assert restored.confidence == 0.75
        assert len(restored.messages) == 1

    def test_total_tokens(self):
        """Estimate total tokens from messages."""
        state = AgentState()

        # Empty state should have 0 tokens
        assert state.total_tokens == 0

        # Add a message with content
        msg = Message.user("Hello world!")  # 12 chars -> ~3 tokens
        state = state.with_message(msg)
        assert state.total_tokens > 0

        # Add message with tool calls
        tc = ToolCall(name="search", arguments={"query": "test"})
        msg_with_tools = Message.assistant(content="Searching", tool_calls=[tc])
        state = state.with_message(msg_with_tools)
        assert state.total_tokens > 3


class TestToolExecution:
    """Tests for ToolExecution."""

    def test_successful_execution(self):
        """Create successful execution record."""
        exec_ = ToolExecution(
            tool_name="search",
            tool_call_id="call_123",
            arguments={"query": "test"},
            result="Found 5 results",
            duration_ms=150.5,
        )

        assert exec_.success is True
        assert exec_.error is None

    def test_failed_execution(self):
        """Create failed execution record."""
        exec_ = ToolExecution(
            tool_name="search",
            tool_call_id="call_123",
            arguments={"query": "test"},
            error="Connection timeout",
        )

        assert exec_.success is False


class TestReasoningStep:
    """Tests for ReasoningStep."""

    def test_create_step(self):
        """Create a reasoning step."""
        tc = ToolCall(name="search", arguments={"q": "test"})
        step = ReasoningStep(
            iteration=1,
            thought="I need to search for information",
            tool_calls=[tc],
            confidence_delta=0.1,
        )

        assert step.iteration == 1
        assert step.thought == "I need to search for information"
        assert len(step.tool_calls) == 1


# =============================================================================
# Token Tracking Tests
# =============================================================================


class TestTokenTracking:
    """Tests for token usage tracking and budget enforcement."""

    def test_with_token_usage(self):
        """Token usage accumulates correctly."""
        state = AgentState()
        state = state.with_token_usage(100, 50)
        assert state.total_tokens_used == 150
        assert state.prompt_tokens_used == 100
        assert state.completion_tokens_used == 50

        state = state.with_token_usage(200, 75)
        assert state.total_tokens_used == 425
        assert state.prompt_tokens_used == 300
        assert state.completion_tokens_used == 125

    def test_total_tokens_prefers_real_count(self):
        """total_tokens property returns real count when available."""
        state = AgentState()
        state = state.with_message(Message.user("hello world"))
        # Before any real tracking, falls back to estimate
        assert state.total_tokens > 0

        # After real tracking, returns real count
        state = state.with_token_usage(500, 200)
        assert state.total_tokens == 700

    def test_token_budget_terminates(self):
        """should_terminate returns True when token budget exceeded."""
        state = AgentState(token_budget=1000)
        state = state.with_token_usage(600, 500)  # 1100 > 1000

        should_stop, reason = state.should_terminate
        assert should_stop is True
        assert reason == "token_budget"

    def test_no_token_budget_no_termination(self):
        """Without token_budget, high usage doesn't trigger termination."""
        state = AgentState()  # token_budget=None
        state = state.with_token_usage(999999, 999999)

        should_stop, _reason = state.should_terminate
        assert should_stop is False

    def test_token_budget_under_limit(self):
        """Under budget doesn't trigger termination."""
        state = AgentState(token_budget=1000)
        state = state.with_token_usage(300, 200)  # 500 < 1000

        should_stop, _reason = state.should_terminate
        assert should_stop is False

    def test_default_token_fields(self):
        """Default state has zero token counts and no budget."""
        state = AgentState()
        assert state.total_tokens_used == 0
        assert state.prompt_tokens_used == 0
        assert state.completion_tokens_used == 0
        assert state.token_budget is None


# =============================================================================
# Config Budget Tests
# =============================================================================


class TestConfigBudgets:
    """Tests for budget fields in AgentConfig."""

    def test_max_iterations_raised_to_500(self):
        """max_iterations cap is now 500."""
        from locus.agent.config import AgentConfig

        config = AgentConfig(model="openai:gpt-4o", max_iterations=200)
        assert config.max_iterations == 200

        config = AgentConfig(model="openai:gpt-4o", max_iterations=500)
        assert config.max_iterations == 500

        with pytest.raises(ValidationError):
            AgentConfig(model="openai:gpt-4o", max_iterations=501)

    def test_token_budget(self):
        """token_budget accepts positive int or None."""
        from locus.agent.config import AgentConfig

        config = AgentConfig(model="openai:gpt-4o", token_budget=50000)
        assert config.token_budget == 50000

        config = AgentConfig(model="openai:gpt-4o")
        assert config.token_budget is None

    def test_time_budget(self):
        """time_budget_seconds accepts positive float or None."""
        from locus.agent.config import AgentConfig

        config = AgentConfig(model="openai:gpt-4o", time_budget_seconds=30.0)
        assert config.time_budget_seconds == 30.0

        config = AgentConfig(model="openai:gpt-4o")
        assert config.time_budget_seconds is None

    def test_stop_reason_includes_budgets(self):
        """StopReason literal includes budget reasons."""
        from locus.agent.result import StopReason

        # These should be valid StopReason values
        reasons: list[StopReason] = ["token_budget", "time_budget"]
        assert len(reasons) == 2
