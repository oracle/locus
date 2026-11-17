# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Idempotent-tool deduplication: the match the executor looks up.

When a tool declares ``idempotent=True`` the agent loop reuses its prior
result if the same (tool_name, arguments) combination has already been
executed in the current agent run. This prevents side-effect-bearing tools
(bookings, transfers, writes) from firing twice when a model re-issues the
same call. The agent-level behaviour is in ``test_agent_idempotent.py``;
this file covers the lookup itself and the built-in tools that rely on it.
"""

from __future__ import annotations

from locus.core.state import AgentState, ToolExecution
from locus.tools.decorator import tool
from locus.tools.executor import find_matching_execution


class TestFindMatchingExecution:
    def test_returns_none_when_no_history(self):
        state = AgentState()
        assert find_matching_execution(state, "book", {"x": 1}) is None

    def test_matches_name_and_args(self):
        state = AgentState().with_tool_execution(
            ToolExecution(
                tool_name="book",
                tool_call_id="1",
                arguments={"x": 1},
                result="ok",
            )
        )
        match = find_matching_execution(state, "book", {"x": 1})
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
        assert find_matching_execution(state, "book", {"x": 2}) is None

    def test_different_tool_is_not_a_match(self):
        state = AgentState().with_tool_execution(
            ToolExecution(
                tool_name="book",
                tool_call_id="1",
                arguments={"x": 1},
                result="ok",
            )
        )
        assert find_matching_execution(state, "cancel", {"x": 1}) is None

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
        match = find_matching_execution(state, "book", {"x": 1})
        assert match is not None
        assert match.result == "second"


class TestBuiltinsGetTodayDate:
    def test_get_today_date_is_marked_idempotent(self):
        """The built-in date tool should be cache-safe across a single run."""
        from locus.tools.builtins import get_today_date

        assert get_today_date.idempotent is True

    def test_get_today_date_returns_expected_keys(self):
        from locus.tools.builtins import get_today_date

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
