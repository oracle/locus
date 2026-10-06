# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""An oversized tool result keeps its head and tail, not just its head.

A test runner prints the failure summary last, so a head-only cut hands the
model pages of passing dots and none of the verdict.
"""

from __future__ import annotations

from typing import Any

import pytest

from tulip.agent.agent import Agent
from tulip.agent.config import AgentConfig
from tulip.agent.runtime_loop import truncate_tool_output
from tulip.core.messages import Message, Role, ToolCall
from tulip.models import ModelResponse
from tulip.tools.decorator import tool


class TestTruncateToolOutput:
    def test_text_within_the_limit_is_unchanged(self) -> None:
        assert truncate_tool_output("short", 10) == "short"
        assert truncate_tool_output("x" * 10, 10) == "x" * 10

    def test_zero_limit_means_unlimited(self) -> None:
        assert truncate_tool_output("x" * 100, 0) == "x" * 100

    def test_keeps_head_and_tail_split_by_fraction(self) -> None:
        text = "H" * 500 + "M" * 9_000 + "T" * 500
        out = truncate_tool_output(text, 1_000, 0.4)
        head, marker, tail = out.split("\n")
        assert head == "H" * 400
        assert tail == "M" * 100 + "T" * 500
        assert marker == (
            "[OUTPUT TRUNCATED — 9000 of 10000 chars cut; first 400 and last 600 kept]"
        )

    def test_head_fraction_one_keeps_only_the_head(self) -> None:
        out = truncate_tool_output("a" * 50 + "b" * 50, 10, 1.0)
        assert (
            out == "a" * 10 + "\n[OUTPUT TRUNCATED — 90 of 100 chars cut; first 10 and last 0 kept]"
        )

    def test_head_fraction_zero_keeps_only_the_tail(self) -> None:
        out = truncate_tool_output("a" * 50 + "b" * 50, 10, 0.0)
        assert (
            out == "[OUTPUT TRUNCATED — 90 of 100 chars cut; first 0 and last 10 kept]\n" + "b" * 10
        )

    @pytest.mark.parametrize("fraction", [-1.0, 2.0])
    def test_out_of_range_fraction_is_clamped(self, fraction: float) -> None:
        out = truncate_tool_output("x" * 100, 10, fraction)
        assert out.count("x") == 10

    def test_config_rejects_out_of_range_fraction(self) -> None:
        with pytest.raises(ValueError):
            AgentConfig(model="openai:gpt-4o", tool_result_head_fraction=1.5)

    def test_config_default_keeps_more_tail_than_head(self) -> None:
        assert AgentConfig(model="openai:gpt-4o").tool_result_head_fraction == 0.4


_PYTEST_OUTPUT = (
    "============ test session starts ============\n"
    "tests/test_mod.py "
    + "." * 40_000
    + "\n"
    + "FAILED tests/test_mod.py::test_parse - AssertionError: expected 3, got 4\n"
    + "======= 1 failed, 39999 passed in 12.34s =======\n"
)


@tool
def run_tests() -> str:
    """Run the test suite."""
    return _PYTEST_OUTPUT


class _ScriptedModel:
    name = "scripted"

    def __init__(self) -> None:
        self._responses = [
            ModelResponse(
                message=Message.assistant(
                    tool_calls=[ToolCall(id="t1", name="run_tests", arguments={})]
                )
            ),
            ModelResponse(message=Message.assistant("one failure")),
        ]

    async def complete(
        self, messages: list[Message], tools: Any = None, **kwargs: Any
    ) -> ModelResponse:
        return self._responses.pop(0)


def _tool_message(**config: Any) -> str:
    agent = Agent(
        config=AgentConfig(
            model=_ScriptedModel(),
            tools=[run_tests],
            max_iterations=3,
            max_tool_result_length=2_000,
            **config,
        )
    )
    result = agent.run_sync("run the tests")
    return next(m.content or "" for m in result.state.messages if m.role == Role.TOOL)


class TestLoopTruncation:
    def test_the_failure_summary_at_the_tail_reaches_the_model(self) -> None:
        content = _tool_message()
        assert content.startswith("============ test session starts")
        assert "FAILED tests/test_mod.py::test_parse" in content
        assert "1 failed, 39999 passed" in content
        assert "[OUTPUT TRUNCATED — " in content
        assert f"of {len(_PYTEST_OUTPUT)} chars cut" in content

    def test_head_fraction_one_restores_head_only(self) -> None:
        content = _tool_message(tool_result_head_fraction=1.0)
        assert "FAILED" not in content
        assert content.endswith("first 2000 and last 0 kept]")
