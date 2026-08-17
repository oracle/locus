# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Tests for AgentTestClient and AgentTrace.

Two things are worth testing here and only one is obvious. The first is that a
passing assertion passes. The second — the one that matters at 3am — is that a
*failing* assertion says what actually happened, so most of this file asserts
on the text of the error rather than merely that one was raised.
"""

from __future__ import annotations

import pytest

from tulip import Agent, tool
from tulip.testing import AgentTestClient, AgentTrace, ScriptedModel, text, tool_call


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def refund(order_id: str) -> str:
    """Refund an order."""
    return f"refunded {order_id}"


@tool
def explode() -> str:
    """Always raises, to exercise the failure path."""
    raise RuntimeError("boom")


def _client(*turns, tools=None) -> AgentTestClient:
    model = ScriptedModel(list(turns))
    return AgentTestClient(
        Agent(model=model, tools=tools if tools is not None else [add, refund], system_prompt="x")
    )


# --------------------------------------------------------------- data ----


def test_trace_exposes_the_run_as_plain_data() -> None:
    trace = _client(tool_call("add", a=2, b=2), text("4")).run("2+2?")

    assert trace.message == "4"
    assert trace.tool_calls == [("add", {"a": 2, "b": 2})]
    assert trace.tool_names == ["add"]
    assert trace.model_calls == 2
    assert trace.failed_tools == []


def test_a_run_with_no_tools_reports_empty_rather_than_raising() -> None:
    trace = _client(text("hello")).run("hi")

    assert trace.tool_calls == []
    assert trace.tool_names == []
    assert trace.model_calls == 1


def test_failed_tools_carries_the_error() -> None:
    trace = _client(tool_call("explode"), text("done"), tools=[explode]).run("go")

    assert [name for name, _ in trace.failed_tools] == ["explode"]
    assert "boom" in trace.failed_tools[0][1]


# --------------------------------------------------- passing assertions ----


def test_assertions_chain() -> None:
    trace = _client(tool_call("add", a=2, b=2), text("4")).run("2+2?")

    returned = (
        trace.assert_tool_called("add", a=2, b=2)
        .assert_tool_not_called("refund")
        .assert_tools_called("add")
        .assert_model_calls(2)
        .assert_tool_offered("refund")
        .assert_succeeded()
    )
    assert returned is trace


def test_assert_tool_called_ignores_arguments_it_was_not_given() -> None:
    """Pin the argument that matters without restating the whole call."""
    trace = _client(tool_call("add", a=2, b=2), text("4")).run("2+2?")

    trace.assert_tool_called("add", a=2)  # b unmentioned, still passes


# --------------------------------------------------- failure messages ----


def test_missing_tool_names_what_was_called_instead() -> None:
    trace = _client(tool_call("add", a=2, b=2), text("4")).run("2+2?")

    with pytest.raises(AssertionError, match=r"expected tool 'refund' to be called") as exc:
        trace.assert_tool_called("refund")
    assert "['add']" in str(exc.value)


def test_missing_tool_says_so_when_nothing_ran_at_all() -> None:
    trace = _client(text("hello")).run("hi")

    with pytest.raises(AssertionError, match="no tools at all"):
        trace.assert_tool_called("add")


def test_wrong_arguments_shows_the_actual_call() -> None:
    trace = _client(tool_call("add", a=2, b=2), text("4")).run("2+2?")

    with pytest.raises(AssertionError) as exc:
        trace.assert_tool_called("add", a=99)
    assert "never with" in str(exc.value)
    assert "'a': 2" in str(exc.value)


def test_forbidden_tool_reports_the_full_call_order() -> None:
    """The gate assertion — this is the one a control test hangs on."""
    trace = _client(tool_call("add", a=2, b=2), text("4")).run("2+2?")

    with pytest.raises(AssertionError, match="NOT to be called") as exc:
        trace.assert_tool_not_called("add")
    assert "['add']" in str(exc.value)


def test_order_mismatch_shows_both_sequences() -> None:
    trace = _client(tool_call("add", a=1, b=1), text("2")).run("go")

    with pytest.raises(AssertionError) as exc:
        trace.assert_tools_called("refund", "add")
    assert "expected: ['refund', 'add']" in str(exc.value)
    assert "actual  : ['add']" in str(exc.value)


def test_model_call_count_mismatch_mentions_the_tools() -> None:
    trace = _client(tool_call("add", a=2, b=2), text("4")).run("2+2?")

    with pytest.raises(AssertionError, match=r"expected 5 model call\(s\), got 2"):
        trace.assert_model_calls(5)


def test_unoffered_tool_is_distinguished_from_an_unchosen_one() -> None:
    """A tool the model never saw cannot be called; that is a different bug."""
    trace = _client(text("hi"), tools=[add]).run("hi")

    with pytest.raises(AssertionError, match="never offered to the model") as exc:
        trace.assert_tool_offered("refund")
    assert "['add']" in str(exc.value)


def test_assert_succeeded_surfaces_a_raising_tool() -> None:
    trace = _client(tool_call("explode"), text("done"), tools=[explode]).run("go")

    with pytest.raises(AssertionError, match="tools raised during the run"):
        trace.assert_succeeded()


# ------------------------------------------------------------- client ----


@pytest.mark.asyncio
async def test_arun_returns_the_same_trace_shape() -> None:
    client = _client(tool_call("add", a=3, b=4), text("7"))

    trace = await client.arun("3+4?")

    assert isinstance(trace, AgentTrace)
    trace.assert_tool_called("add", a=3, b=4).assert_model_calls(2)
    assert trace.message == "7"


def test_client_exposes_the_double_for_input_side_assertions() -> None:
    client = _client(text("hi"))
    client.run("hello")

    assert client.model.call_count == 1
    assert client.model.last_prompt == "hello"
