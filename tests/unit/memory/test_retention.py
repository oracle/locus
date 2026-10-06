# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-message retention: whole exchanges older than the cut-off go, nothing else does."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tulip.agent import Agent
from tulip.core.messages import Message, ToolCall, ToolResult
from tulip.core.state import AgentState
from tulip.memory.backends import MemoryCheckpointer
from tulip.memory.retention import (
    MESSAGE_TIME_KEY,
    RetainedCheckpointer,
    message_time,
    oldest_message_time,
    stamp_messages,
    trim_messages,
)
from tulip.testing import ScriptedModel, text


NOW = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)


def _at(message: Message, days_ago: float) -> Message:
    when = (NOW - timedelta(days=days_ago)).isoformat()
    return message.model_copy(update={"metadata": {MESSAGE_TIME_KEY: when}})


def _thread() -> AgentState:
    """Three exchanges, 40, 20 and 0 days old; the oldest has a tool call."""
    call = ToolCall(id="c1", name="lookup", arguments={})
    return AgentState(
        messages=(
            Message.system("you help"),
            _at(Message.user("old question"), 40),
            _at(Message.assistant(tool_calls=[call]), 40),
            _at(Message.tool(ToolResult(tool_call_id="c1", name="lookup", content="42")), 40),
            _at(Message.assistant("old answer"), 40),
            _at(Message.system("summary of earlier turns"), 25),
            _at(Message.user("middle question"), 20),
            _at(Message.assistant("middle answer"), 20),
            _at(Message.user("new question"), 0),
            _at(Message.assistant("new answer"), 0),
        )
    )


def _texts(state: AgentState) -> list[str | None]:
    return [m.content for m in state.messages]


def test_whole_old_exchanges_go_and_system_messages_stay() -> None:
    trimmed, dropped = trim_messages(_thread(), NOW - timedelta(days=30))
    assert dropped == 4
    assert _texts(trimmed) == [
        "you help",
        "summary of earlier turns",
        "middle question",
        "middle answer",
        "new question",
        "new answer",
    ]
    # The tool result went with the call it answers.
    assert not [m for m in trimmed.messages if m.role == "tool"]


def test_a_tighter_cut_off_drops_more_and_never_the_present() -> None:
    trimmed, dropped = trim_messages(_thread(), NOW - timedelta(days=10))
    assert dropped == 6
    assert _texts(trimmed) == ["you help", "summary of earlier turns", "new question", "new answer"]


def test_an_unstamped_exchange_stops_the_trim() -> None:
    state = _thread()
    messages = list(state.messages)
    messages[1] = Message.user("old question, never stamped")
    trimmed, dropped = trim_messages(
        state.model_copy(update={"messages": tuple(messages)}), NOW - timedelta(days=10)
    )
    assert dropped == 0
    assert len(trimmed.messages) == len(state.messages)


def test_stamping_keeps_existing_stamps() -> None:
    state = _thread()
    fresh = state.model_copy(update={"messages": (*state.messages, Message.user("now"))})
    stamped = stamp_messages(fresh, NOW + timedelta(hours=1))
    assert message_time(stamped.messages[1]) == NOW - timedelta(days=40)
    assert message_time(stamped.messages[-1]) == NOW + timedelta(hours=1)
    assert stamp_messages(stamped) is stamped  # nothing new, nothing copied
    assert oldest_message_time(stamped) == NOW - timedelta(days=40)


def test_message_time_ignores_garbage() -> None:
    assert message_time(Message.user("x")) is None
    assert (
        message_time(Message.user("x").model_copy(update={"metadata": {MESSAGE_TIME_KEY: 3}}))
        is None
    )
    bad = Message.user("x").model_copy(update={"metadata": {MESSAGE_TIME_KEY: "yesterday"}})
    assert message_time(bad) is None


async def test_retained_checkpointer_trims_the_local_default() -> None:
    inner = MemoryCheckpointer()
    cp = RetainedCheckpointer(inner, max_age=timedelta(days=30))
    await cp.save(_thread(), "t1")
    loaded = await cp.load("t1")
    assert loaded is not None
    assert "old question" not in _texts(loaded)
    assert "middle question" in _texts(loaded)
    assert await cp.list_checkpoints("t1")
    assert cp.deletes_single_checkpoints is inner.deletes_single_checkpoints


async def test_an_agent_on_a_retained_thread_stamps_each_turn() -> None:
    cp = RetainedCheckpointer(MemoryCheckpointer(), max_age=timedelta(days=30))
    agent = Agent(model=ScriptedModel([text("one"), text("two")]), checkpointer=cp)
    await agent.arun("first", thread_id="t")
    await agent.arun("second", thread_id="t")
    loaded = await cp.load("t")
    assert loaded is not None
    talk = [m for m in loaded.messages if m.role != "system"]
    assert [m.content for m in talk] == ["first", "one", "second", "two"]
    assert all(message_time(m) is not None for m in talk)


def test_max_age_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_age"):
        RetainedCheckpointer(MemoryCheckpointer(), max_age=timedelta(0))


def test_messages_before_any_user_message_are_one_exchange() -> None:
    state = AgentState(
        messages=(
            Message.system("you help"),
            _at(Message.assistant("hello, how can I help?"), 40),
        )
    )
    trimmed, dropped = trim_messages(state, NOW - timedelta(days=30))
    assert dropped == 1
    assert _texts(trimmed) == ["you help"]
    assert trim_messages(AgentState(), NOW)[1] == 0
    assert oldest_message_time(AgentState(messages=(Message.user("unstamped"),))) is None


async def test_retained_checkpointer_hands_everything_else_to_inner() -> None:
    inner = MemoryCheckpointer()
    cp = RetainedCheckpointer(inner)  # stamps only
    await cp.save(_thread(), "t1")
    assert await cp.exists("t1")
    assert await cp.list_threads() == ["t1"]
    assert cp.capabilities == inner.capabilities
    assert "MemoryCheckpointer" in repr(cp)
    assert await cp.delete("t1") is True
    assert not await cp.exists("t1")
    with pytest.raises(NotImplementedError, match="vacuum"):
        await cp.vacuum(1)
    await cp.close()
