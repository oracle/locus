# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Idempotent-tool deduplication in the Execute node.

When a tool declares ``idempotent=True`` the ReAct loop reuses its prior
result if the same (tool_name, arguments) combination has already been
executed in the current agent run. This prevents side-effect-bearing tools
(bookings, transfers, writes) from firing twice when a model re-issues the
same call.
"""

from __future__ import annotations

import pytest

from tulip.core.messages import Message, ToolCall
from tulip.core.state import AgentState, ToolExecution
from tulip.loop.nodes import ExecuteNode, _find_matching_execution
from tulip.tools.decorator import tool
from tulip.tools.registry import ToolRegistry


def _state_with_pending_calls(base: AgentState, calls: list[ToolCall]) -> AgentState:
    """Attach ``calls`` as pending tool calls on an assistant message so
    ``state.last_tool_calls`` returns them. That's what ExecuteNode reads."""
    return base.with_message(Message.assistant("", tool_calls=calls))


class TestFindMatchingExecution:
    def test_returns_none_when_no_history(self):
        state = AgentState()
        assert _find_matching_execution(state, "book", {"x": 1}) is None

    def test_matches_name_and_args(self):
        state = AgentState().with_tool_execution(
            ToolExecution(
                tool_name="book",
                tool_call_id="1",
                arguments={"x": 1},
                result="ok",
            )
        )
        match = _find_matching_execution(state, "book", {"x": 1})
        assert match is not None
        assert match.result == "ok"

    def test_different_args_is_not_a_match(self):
        state = AgentState().with_tool_execution(
            ToolExecution(
                tool_name="book",
                tool_call_id="1",
                arguments={"x": 1},
                result="ok",
            )
        )
        assert _find_matching_execution(state, "book", {"x": 2}) is None

    def test_different_tool_is_not_a_match(self):
        state = AgentState().with_tool_execution(
            ToolExecution(
                tool_name="book",
                tool_call_id="1",
                arguments={"x": 1},
                result="ok",
            )
        )
        assert _find_matching_execution(state, "cancel", {"x": 1}) is None

    def test_returns_most_recent_match(self):
        state = (
            AgentState()
            .with_tool_execution(
                ToolExecution(
                    tool_name="book", tool_call_id="1", arguments={"x": 1}, result="first"
                )
            )
            .with_tool_execution(
                ToolExecution(
                    tool_name="book", tool_call_id="2", arguments={"x": 1}, result="second"
                )
            )
        )
        match = _find_matching_execution(state, "book", {"x": 1})
        assert match is not None
        assert match.result == "second"


class TestExecuteNodeIdempotentDedup:
    @pytest.mark.asyncio
    async def test_idempotent_tool_is_not_re_run(self):
        """A second call to an idempotent tool reuses the prior result."""
        call_count = 0

        @tool(idempotent=True)
        def book_flight(flight_id: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"booked {flight_id} (call {call_count})"

        registry = ToolRegistry()
        registry.register(book_flight)
        node = ExecuteNode(registry=registry)

        # First call — executes for real.
        state = _state_with_pending_calls(
            AgentState(), [ToolCall(id="c1", name="book_flight", arguments={"flight_id": "FL-001"})]
        )
        r1 = await node.execute(state)
        assert call_count == 1
        assert "booked FL-001 (call 1)" in r1.state.tool_executions[-1].result

        # Second call with identical args — cache hit, body never runs again.
        state2 = _state_with_pending_calls(
            r1.state, [ToolCall(id="c2", name="book_flight", arguments={"flight_id": "FL-001"})]
        )
        r2 = await node.execute(state2)
        assert call_count == 1, f"idempotent tool should not re-execute; call_count={call_count}"
        # The reused result carries the same payload as the first run.
        assert "call 1" in r2.state.tool_executions[-1].result

    @pytest.mark.asyncio
    async def test_non_idempotent_tool_runs_every_time(self):
        """Tools without ``idempotent=True`` keep their previous behavior."""
        call_count = 0

        @tool
        def search(query: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"results {call_count}"

        registry = ToolRegistry()
        registry.register(search)
        node = ExecuteNode(registry=registry)

        state = _state_with_pending_calls(
            AgentState(), [ToolCall(id="c1", name="search", arguments={"query": "q"})]
        )
        r1 = await node.execute(state)
        state2 = _state_with_pending_calls(
            r1.state, [ToolCall(id="c2", name="search", arguments={"query": "q"})]
        )
        await node.execute(state2)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_different_args_bypass_the_cache(self):
        """Same idempotent tool with different args is still re-run."""
        call_count = 0

        @tool(idempotent=True)
        def book_flight(flight_id: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"{flight_id} -> {call_count}"

        registry = ToolRegistry()
        registry.register(book_flight)
        node = ExecuteNode(registry=registry)

        state = _state_with_pending_calls(
            AgentState(), [ToolCall(id="c1", name="book_flight", arguments={"flight_id": "FL-001"})]
        )
        r1 = await node.execute(state)
        state2 = _state_with_pending_calls(
            r1.state, [ToolCall(id="c2", name="book_flight", arguments={"flight_id": "FL-002"})]
        )
        await node.execute(state2)
        assert call_count == 2


class TestBuiltinsGetTodayDate:
    def test_get_today_date_is_marked_idempotent(self):
        """The built-in date tool should be cache-safe across a single run."""
        from tulip.tools.builtins import get_today_date

        assert get_today_date.idempotent is True

    def test_get_today_date_returns_expected_keys(self):
        from tulip.tools.builtins import get_today_date

        result = get_today_date.fn()
        assert {"today", "weekday", "year", "tomorrow", "next_7_days_by_weekday"} <= result.keys()


class TestToolFuncAlias:
    """``.fn`` and ``.func`` both point at the wrapped callable.

    Some downstream samples and frameworks (LangChain/LangGraph idiom) reach
    for ``.func``; the Pydantic field is named ``fn``. Surfacing both names
    avoids the ``getattr(t, "fn", None) or getattr(t, "func", t)`` dance.
    """

    def test_func_returns_same_callable_as_fn(self):
        @tool(description="t")
        async def my_tool(x: int) -> int:
            return x + 1

        assert my_tool.func is my_tool.fn

    def test_func_works_for_sync_tool(self):
        @tool
        def my_sync_tool(x: int) -> int:
            """Doubles x."""
            return x * 2

        assert my_sync_tool.func(3) == 6
        assert my_sync_tool.func is my_sync_tool.fn
