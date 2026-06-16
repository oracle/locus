# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for ``AgentConfig.termination`` wired into the agent loop.

The composable termination algebra (``MaxIterations``, ``ToolCalled``, etc.)
existed in source for a while but was never read by ``Agent.run()``. These
tests guard the wiring: each condition fires, the OR / AND combinators work,
and the ``stop_reason`` arrives correctly on ``AgentResult``.
"""

from __future__ import annotations

from typing import Any

from locus.agent import Agent
from locus.core.messages import Message, ToolCall
from locus.core.termination import (
    ConfidenceMet,
    CustomCondition,
    MaxIterations,
    NoToolCalls,
    TextMention,
    TokenLimit,
    ToolCalled,
)
from locus.models.base import ModelResponse
from locus.tools.decorator import tool


class _ScriptedModel:
    """Returns a scripted sequence of model responses, looping the last one."""

    def __init__(self, responses: list[ModelResponse]):
        self._responses = list(responses)
        self.calls: int = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self.calls += 1
        if not self._responses:
            raise AssertionError("ScriptedModel exhausted")
        # Repeat the last response if the loop keeps going past the script.
        if len(self._responses) == 1:
            return self._responses[0]
        return self._responses.pop(0)

    async def stream(self, *args: Any, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError


def _assistant(content: str, *, tool_calls: list[ToolCall] | None = None) -> ModelResponse:
    return ModelResponse(
        message=Message.assistant(content=content, tool_calls=tool_calls or []),
        usage={"prompt_tokens": 5, "completion_tokens": 5},
    )


@tool
def book_flight(destination: str) -> str:
    """Book a flight."""
    return f"Booked flight to {destination}"


@tool
def search(query: str) -> str:
    """Search for something."""
    return f"Found results for {query}"


# =============================================================================
# Individual conditions
# =============================================================================


class TestMaxIterations:
    def test_fires_at_threshold(self):
        # Model wants to keep tool-calling forever; MaxIterations should stop it.
        loop_response = _assistant(
            None,
            tool_calls=[ToolCall(name="search", arguments={"query": "x"})],
        )
        model = _ScriptedModel([loop_response])
        agent = Agent(
            model=model,
            tools=[search],
            termination=MaxIterations(2),
            max_iterations=20,  # built-in cap is far higher; condition wins.
        )
        result = agent.run_sync("Search forever.")
        assert result.stop_reason == "max_iterations"
        # Should have stopped early (well under built-in 20).
        assert result.iterations <= 3


class TestTextMention:
    def test_fires_when_text_appears(self):
        # Model emits a tool call alongside text containing "DONE" — the agent
        # executes the tool, loops, and the next-iteration termination check
        # picks up the mention from the previous assistant message.
        responses = [
            _assistant(
                "still working...",
                tool_calls=[ToolCall(name="search", arguments={"query": "x"})],
            ),
            _assistant(
                "Final answer: DONE.",
                tool_calls=[ToolCall(name="search", arguments={"query": "y"})],
            ),
            _assistant(
                "should never reach this",
                tool_calls=[ToolCall(name="search", arguments={"query": "z"})],
            ),
        ]
        model = _ScriptedModel(responses)
        agent = Agent(
            model=model,
            tools=[search],
            termination=TextMention("DONE"),
            max_iterations=20,
        )
        result = agent.run_sync("Eventually say DONE.")
        assert result.stop_reason == "complete"
        # Final assistant content includes the DONE mention.
        assert "DONE" in result.message


class TestToolCalled:
    def test_fires_when_named_tool_called(self):
        # First iteration: call ``search`` — does not satisfy.
        # Second iteration: call ``book_flight`` — satisfies.
        responses = [
            _assistant(None, tool_calls=[ToolCall(name="search", arguments={"query": "Paris"})]),
            _assistant(
                None,
                tool_calls=[ToolCall(name="book_flight", arguments={"destination": "Paris"})],
            ),
        ]
        model = _ScriptedModel(responses)
        agent = Agent(
            model=model,
            tools=[search, book_flight],
            termination=ToolCalled("book_flight"),
            max_iterations=10,
        )
        result = agent.run_sync("Plan a trip.")
        assert result.stop_reason == "terminal_tool"


class TestNoToolCalls:
    def test_fires_when_model_returns_text_only(self):
        # Single text response with no tool calls.
        model = _ScriptedModel([_assistant("All done.")])
        agent = Agent(
            model=model,
            tools=[],
            termination=NoToolCalls(),
            max_iterations=10,
        )
        result = agent.run_sync("Wrap up.")
        # NoToolCalls reason is "no_tools" which is in the StopReason literal.
        assert result.stop_reason in ("no_tools", "complete")


# =============================================================================
# Combinators
# =============================================================================


class TestOrCombinator:
    def test_either_branch_fires(self):
        # MaxIterations(5) | TextMention("DONE") — text mention should fire first.
        responses = [
            _assistant("step 1"),
            _assistant("step 2 DONE"),
            _assistant("never reached"),
        ]
        model = _ScriptedModel(responses)
        agent = Agent(
            model=model,
            tools=[],
            termination=MaxIterations(5) | TextMention("DONE"),
            max_iterations=20,
        )
        result = agent.run_sync("Eventually say DONE.")
        # Either branch — should have stopped well before MaxIterations.
        assert result.iterations <= 3


class TestAndCombinator:
    def test_both_must_fire(self):
        # ConfidenceMet(0.0) is always true; ToolCalled("book_flight") fires only
        # after that tool is invoked. AND should require both — i.e. wait for the
        # tool call.
        responses = [
            _assistant(None, tool_calls=[ToolCall(name="search", arguments={"query": "x"})]),
            _assistant(
                None,
                tool_calls=[ToolCall(name="book_flight", arguments={"destination": "Paris"})],
            ),
        ]
        model = _ScriptedModel(responses)
        agent = Agent(
            model=model,
            tools=[search, book_flight],
            termination=ConfidenceMet(0.0) & ToolCalled("book_flight"),
            max_iterations=10,
        )
        result = agent.run_sync("Plan a trip.")
        assert result.stop_reason == "terminal_tool"


# =============================================================================
# CustomCondition
# =============================================================================


class TestCustomCondition:
    def test_lambda_predicate(self):
        # Stop when iteration counter >= 2 for a custom reason.
        cond = CustomCondition(lambda state, **ctx: (state.iteration >= 2, "iteration_threshold"))
        loop = _assistant(None, tool_calls=[ToolCall(name="search", arguments={"query": "x"})])
        model = _ScriptedModel([loop])
        agent = Agent(model=model, tools=[search], termination=cond, max_iterations=20)
        result = agent.run_sync("Loop a few times.")
        # Custom reasons normalise to ``complete``.
        assert result.stop_reason == "complete"
        assert result.iterations <= 3


# =============================================================================
# TokenLimit
# =============================================================================


class TestTokenLimit:
    def test_token_budget_fires(self):
        # Each call consumes 10 tokens (5 prompt + 5 completion). Budget=15
        # means the second call should trip the limit.
        loop = _assistant(None, tool_calls=[ToolCall(name="search", arguments={"query": "x"})])
        model = _ScriptedModel([loop])
        agent = Agent(
            model=model,
            tools=[search],
            termination=TokenLimit(15),
            max_iterations=20,
        )
        result = agent.run_sync("Burn tokens.")
        assert result.stop_reason == "token_budget"


# =============================================================================
# No-op when termination is None
# =============================================================================


class TestUnsetTerminationIsNoOp:
    def test_unset_uses_builtin_termination(self):
        # No user termination — built-in should handle the no-tool-calls case.
        model = _ScriptedModel([_assistant("Done.")])
        agent = Agent(model=model, tools=[], max_iterations=5)
        result = agent.run_sync("Hi.")
        assert result.stop_reason in ("complete", "no_tools")
