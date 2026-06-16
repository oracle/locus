# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for ``@tool(idempotent=True)`` dedup in ``Agent.run()``.

The dedup logic exists in ``loop/nodes.py`` (used by ``ExecuteNode``) but
``Agent.run()`` has its own inline executor block. These tests guard that
the inline block correctly looks up prior executions before invoking the
tool body, so the README hero example (idempotent booking) is real.
"""

from __future__ import annotations

from typing import Any

from locus.agent import Agent
from locus.core.messages import Message, ToolCall
from locus.core.termination import MaxIterations
from locus.models.base import ModelResponse
from locus.tools.decorator import tool


class _ScriptedModel:
    def __init__(self, responses: list[ModelResponse]):
        self._responses = list(responses)

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        if not self._responses:
            raise AssertionError("ScriptedModel exhausted")
        if len(self._responses) == 1:
            return self._responses[0]
        return self._responses.pop(0)

    async def stream(self, *args: Any, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError


def _assistant(content: str | None, *, tool_calls: list[ToolCall] | None = None) -> ModelResponse:
    return ModelResponse(
        message=Message.assistant(content=content, tool_calls=tool_calls or []),
        usage={"prompt_tokens": 1, "completion_tokens": 1},
    )


# Side-effect counter that lets us observe whether the body actually ran.
class _Counter:
    def __init__(self) -> None:
        self.calls = 0


_book_counter = _Counter()
_log_counter = _Counter()


@tool(idempotent=True)
def book_flight(destination: str) -> str:
    """Book a flight (idempotent — same destination must not double-charge)."""
    _book_counter.calls += 1
    return f"Booked to {destination}. (call {_book_counter.calls})"


@tool
def write_log(line: str) -> str:
    """Write a log line (NOT idempotent — should re-execute every time)."""
    _log_counter.calls += 1
    return f"Logged: {line}"


class TestIdempotentDedup:
    def setup_method(self) -> None:
        # Reset shared counters between tests.
        _book_counter.calls = 0
        _log_counter.calls = 0

    def test_repeat_call_uses_cache(self):
        """Same tool, same args, twice in a row — body fires once."""
        # Iteration 1: book Paris. Iteration 2: book Paris again. Iteration 3:
        # book Tokyo (different args — must re-execute).
        responses = [
            _assistant(
                None,
                tool_calls=[ToolCall(name="book_flight", arguments={"destination": "Paris"})],
            ),
            _assistant(
                None,
                tool_calls=[ToolCall(name="book_flight", arguments={"destination": "Paris"})],
            ),
            _assistant(
                None,
                tool_calls=[ToolCall(name="book_flight", arguments={"destination": "Tokyo"})],
            ),
        ]
        model = _ScriptedModel(responses)
        agent = Agent(
            model=model,
            tools=[book_flight],
            termination=MaxIterations(3),
            max_iterations=10,
        )
        result = agent.run_sync("Plan a trip.")

        # Body ran twice: once for Paris, once for Tokyo. The duplicate Paris
        # call must have been short-circuited.
        assert _book_counter.calls == 2

        # Three executions are recorded on state — the middle one is a cache hit.
        executions = list(result.state.tool_executions)
        assert len(executions) == 3
        assert executions[0].idempotent_cache_hit is False
        assert executions[1].idempotent_cache_hit is True
        assert executions[2].idempotent_cache_hit is False

        # Cached result reuses the prior content verbatim (call counter is 1).
        assert "(call 1)" in (executions[1].result or "")

    def test_different_args_skip_cache(self):
        """Same tool, different args — both executions fire."""
        responses = [
            _assistant(
                None,
                tool_calls=[ToolCall(name="book_flight", arguments={"destination": "Paris"})],
            ),
            _assistant(
                None,
                tool_calls=[ToolCall(name="book_flight", arguments={"destination": "Tokyo"})],
            ),
        ]
        model = _ScriptedModel(responses)
        agent = Agent(
            model=model,
            tools=[book_flight],
            termination=MaxIterations(2),
            max_iterations=10,
        )
        agent.run_sync("Trip.")
        assert _book_counter.calls == 2

    def test_non_idempotent_tool_always_reruns(self):
        """A tool without ``idempotent=True`` must fire every call, even with same args."""
        responses = [
            _assistant(None, tool_calls=[ToolCall(name="write_log", arguments={"line": "hi"})]),
            _assistant(None, tool_calls=[ToolCall(name="write_log", arguments={"line": "hi"})]),
        ]
        model = _ScriptedModel(responses)
        agent = Agent(
            model=model,
            tools=[write_log],
            termination=MaxIterations(2),
            max_iterations=10,
        )
        result = agent.run_sync("Log twice.")
        assert _log_counter.calls == 2
        executions = list(result.state.tool_executions)
        assert all(not e.idempotent_cache_hit for e in executions)
