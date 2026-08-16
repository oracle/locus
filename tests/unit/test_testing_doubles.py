# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``tulip.testing``.

The doubles exist because there was no supported way to test an agent: this
suite alone carried thirteen private ``_ScriptedModel`` classes, each slightly
different, and the docs had no testing page at all. These tests pin the
behaviour those private copies kept re-deriving.
"""

from __future__ import annotations

import pytest

from tulip.agent import Agent
from tulip.core.messages import Message
from tulip.core.protocols import ModelProtocol
from tulip.testing import FunctionModel, ScriptedModel, text, tool_call
from tulip.tools.decorator import tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"{city}: 18C, light rain"


# --------------------------------------------------------------------------
# Turn builders
# --------------------------------------------------------------------------


def test_text_builds_a_plain_assistant_turn() -> None:
    response = text("hello")
    assert response.content == "hello"
    assert not response.tool_calls


def test_tool_call_passes_arguments_as_keywords() -> None:
    response = tool_call("issue_refund", order_id="ord-4821", amount=49.0)
    call = response.tool_calls[0]
    assert call.name == "issue_refund"
    assert call.arguments == {"order_id": "ord-4821", "amount": 49.0}


def test_tool_call_id_is_stable() -> None:
    """Assertions must not depend on a random id."""
    assert tool_call("x").tool_calls[0].id == tool_call("x").tool_calls[0].id


def test_tool_call_accepts_accompanying_text() -> None:
    response = tool_call("lookup", content="Let me check.", key="k")
    assert response.content == "Let me check."
    assert response.tool_calls[0].name == "lookup"


# --------------------------------------------------------------------------
# ScriptedModel
# --------------------------------------------------------------------------


def test_scripted_model_satisfies_the_model_protocol() -> None:
    assert isinstance(ScriptedModel([text("hi")]), ModelProtocol)


@pytest.mark.asyncio
async def test_scripted_model_drives_a_tool_call_then_an_answer() -> None:
    model = ScriptedModel([tool_call("get_weather", city="Lisbon"), text("It is 18C and raining.")])
    result = await Agent(model=model, tools=[get_weather]).arun("Weather in Lisbon?")

    assert result.text == "It is 18C and raining."
    assert [t.tool_name for t in result.tool_executions] == ["get_weather"]
    assert model.call_count == 2


@pytest.mark.asyncio
async def test_bare_strings_are_accepted_as_turns() -> None:
    result = await Agent(model=ScriptedModel(["just an answer"])).arun("hi")
    assert result.text == "just an answer"


@pytest.mark.asyncio
async def test_running_out_of_turns_fails_loudly() -> None:
    """Silently improvising a reply would hide a loop that did not stop."""
    model = ScriptedModel([tool_call("get_weather", city="Lisbon")])
    with pytest.raises(AssertionError, match="ran out of turns"):
        await Agent(model=model, tools=[get_weather]).arun("Weather?")


@pytest.mark.asyncio
async def test_repeat_last_keeps_answering() -> None:
    model = ScriptedModel([text("done")], repeat_last=True)
    result = await Agent(model=model).arun("hi")
    assert result.text == "done"


# --------------------------------------------------------------------------
# Recording — assert on what the agent *sent*, not only what came back
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_offered_tools_records_what_the_agent_bound() -> None:
    model = ScriptedModel([text("ok")])
    await Agent(model=model, tools=[get_weather]).arun("hi")
    assert model.offered_tools[0] == ["get_weather"]


@pytest.mark.asyncio
async def test_offered_tools_is_empty_when_no_tools_are_bound() -> None:
    model = ScriptedModel([text("ok")])
    await Agent(model=model).arun("hi")
    assert model.offered_tools[0] == []


@pytest.mark.asyncio
async def test_last_prompt_returns_the_user_turn() -> None:
    model = ScriptedModel([text("ok")])
    await Agent(model=model).arun("what is the capital of Japan?")
    assert model.last_prompt == "what is the capital of Japan?"


def test_last_prompt_is_empty_before_any_call() -> None:
    assert ScriptedModel([text("ok")]).last_prompt == ""


@pytest.mark.asyncio
async def test_received_messages_grows_per_call() -> None:
    model = ScriptedModel([tool_call("get_weather", city="Lisbon"), text("done")])
    await Agent(model=model, tools=[get_weather]).arun("Weather?")
    assert len(model.received_messages) == 2
    # The second call must see the tool result the first one triggered.
    roles = [getattr(m.role, "value", m.role) for m in model.received_messages[1]]
    assert "tool" in roles


# --------------------------------------------------------------------------
# FunctionModel
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_function_model_can_branch_on_the_conversation() -> None:
    def handler(messages: list[Message], tools: list[dict]) -> object:
        if any(getattr(m.role, "value", m.role) == "tool" for m in messages):
            return "The weather was looked up."
        return tool_call("get_weather", city="Berlin")

    model = FunctionModel(handler)
    result = await Agent(model=model, tools=[get_weather]).arun("Berlin weather?")

    assert result.text == "The weather was looked up."
    assert [t.tool_name for t in result.tool_executions] == ["get_weather"]


@pytest.mark.asyncio
async def test_function_model_sees_the_offered_tools() -> None:
    seen: list[list[str]] = []

    def handler(messages: list[Message], tools: list[dict]) -> str:
        seen.append([t["function"]["name"] for t in tools])
        return "ok"

    await Agent(model=FunctionModel(handler), tools=[get_weather]).arun("hi")
    assert seen == [["get_weather"]]


# --------------------------------------------------------------------------
# Streaming — the same double must work for a streaming agent
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_yields_chunks_then_done() -> None:
    model = ScriptedModel([text("hello world")], repeat_last=True)
    chunks = [event async for event in model.stream([Message.user("hi")], None)]

    assert "".join(c.content or "" for c in chunks) == "hello world"
    assert chunks[-1].done is True


@pytest.mark.asyncio
async def test_streamed_doubles_carry_their_tool_calls() -> None:
    """A streaming double must stream the tool calls, not just the prose.

    The agent loop rebuilds a turn from chunk events alone. A double that
    streams only content produces a turn with NO tool calls — and because that
    is a legal turn, nothing raises: the agent simply does not act, and a test
    written to prove it acts passes having exercised nothing. That is the
    failure this asserts against, and it is the reason the assertion is on the
    CHUNKS rather than on the agent's behaviour.
    """
    from tulip.testing import FunctionModel, ScriptedModel, tool_call

    for model in (
        ScriptedModel([tool_call("issue_refund", order_id="ord-1")]),
        FunctionModel(lambda messages, tools: tool_call("issue_refund", order_id="ord-1")),
    ):
        chunks = [c async for c in model.stream([], [])]
        streamed = [tc for c in chunks for tc in (getattr(c, "tool_calls", None) or [])]
        assert [tc.name for tc in streamed] == ["issue_refund"], (
            f"{type(model).__name__}.stream() dropped the tool call; the agent "
            "would make no call at all and nothing would report it"
        )
        assert chunks[-1].done is True
