# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Hook events carry their run's identity and can emit UI-only events.

A shared agent serving many users gave hooks no way to tell runs apart (they
reached for context variables), and an ``on_after_tool_call`` hook that wanted
to show the user a widget could only do it by rewriting ``event.result`` —
which is what the MODEL reads.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from tulip.agent import Agent
from tulip.core.events import (
    CustomEvent,
    RunInfo,
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
    TulipEvent,
)
from tulip.hooks.provider import AfterToolCallEvent, BeforeToolCallEvent, HookProvider
from tulip.memory.backends.memory import MemoryCheckpointer
from tulip.testing import _RecordingModel, text, tool_call
from tulip.tools.decorator import tool


@tool
def search(city: str) -> str:
    """Search hotels."""
    return json.dumps({"items": [{"id": "h1"}, {"id": "h2"}], "city": city})


class _SearchThenAnswer(_RecordingModel):
    async def complete(self, messages: Any, tools: Any = None, **kwargs: Any) -> Any:
        self._record(messages, tools)
        await asyncio.sleep(0.01)
        tool_msgs = [m.content for m in messages if m.role.value == "tool"]
        if tool_msgs:
            return text("found " + tool_msgs[-1])
        return tool_call("search", city="Paris")


class _WidgetHook(HookProvider):
    def __init__(self) -> None:
        self.seen: list[tuple[str, RunInfo | None]] = []

    @property
    def priority(self) -> int:
        return 100

    async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        self.seen.append(("before_tool", event.run))

    async def on_after_tool_call(self, event: AfterToolCallEvent) -> None:
        self.seen.append(("after_tool", event.run))
        data = json.loads(event.result)
        assert event.run is not None
        event.emit(
            CustomEvent(
                name="hotel_list",
                data={"count": len(data["items"]), "user": event.run.metadata["user"]},
            )
        )

    async def on_before_model_call(self, event: Any) -> None:
        self.seen.append(("before_model", event.run))


async def test_hooks_see_their_own_run_under_concurrency() -> None:
    hook = _WidgetHook()
    agent = Agent(model=_SearchThenAnswer(), tools=[search], hooks=[hook], reflexion=False)

    async def go(user: str) -> list[TulipEvent]:
        return [e async for e in agent.run("x", thread_id=user, metadata={"user": user})]

    alice, bob = await asyncio.gather(go("alice"), go("bob"))

    for phase, info in hook.seen:
        assert info is not None, phase
        assert info.thread_id == info.metadata["user"]
    assert {info.thread_id for _, info in hook.seen if info} == {"alice", "bob"}
    # run_id is the run's, identical across a run's hook calls.
    alice_run_ids = {info.run_id for _, info in hook.seen if info and info.thread_id == "alice"}
    assert len(alice_run_ids) == 1
    for events, user in ((alice, "alice"), (bob, "bob")):
        customs = [e for e in events if isinstance(e, CustomEvent)]
        assert len(customs) == 1
        assert customs[0].data == {"count": 2, "user": user}
        assert customs[0].thread_id == user
        assert customs[0].run_id in {info.run_id for _, info in hook.seen if info}
        assert customs[0].tool_call_id


async def test_custom_event_follows_its_tool_and_never_reaches_the_model() -> None:
    model = _SearchThenAnswer()
    checkpointer = MemoryCheckpointer()
    agent = Agent(
        model=model,
        tools=[search],
        hooks=[_WidgetHook()],
        checkpointer=checkpointer,
        reflexion=False,
    )
    events = [e async for e in agent.run("x", thread_id="t", metadata={"user": "u"})]

    kinds = [type(e).__name__ for e in events]
    complete_at = kinds.index("ToolCompleteEvent")
    assert kinds[complete_at + 1] == "CustomEvent"
    complete = events[complete_at]
    assert isinstance(complete, ToolCompleteEvent)
    custom = events[complete_at + 1]
    assert isinstance(custom, CustomEvent)
    assert custom.tool_call_id == complete.tool_call_id
    # The model saw the untouched tool result; the widget is stream-only.
    for batch in model.received_messages:
        for message in batch:
            assert "hotel_list" not in (message.content or "")
    state = await checkpointer.load("t")
    assert state is not None
    assert all("hotel_list" not in (m.content or "") for m in state.messages)
    assert isinstance(events[-1], TerminateEvent)
    assert '"items"' in (events[-1].final_message or "")
    assert any(isinstance(e, ThinkEvent) for e in events)
    assert custom.model_dump(mode="json")["event_type"] == "custom"


def test_emit_requires_a_run_and_a_custom_event() -> None:
    event = AfterToolCallEvent("search", "{}", None, tool_call_id="c1")
    assert event.run is None
    with pytest.raises(RuntimeError, match="running agent"):
        event.emit(CustomEvent(name="x"))

    from tulip.agent.run_context import RunContext

    rc = RunContext.create(
        run_id="r1",
        thread_id="t1",
        prompt="p",
        metadata={"user": "u"},
        agent_name=None,
        termination=None,
    )
    bound = AfterToolCallEvent("search", "{}", None, tool_call_id="c1", run=rc)
    with pytest.raises(TypeError, match="CustomEvent"):
        bound.emit(ThinkEvent(iteration=1))  # type: ignore[arg-type]
    bound.emit(CustomEvent(name="x", thread_id="explicit"))
    [queued] = rc.drain()
    assert queued.thread_id == "explicit"
    assert queued.run_id == "r1"
    assert queued.tool_call_id == "c1"
    assert rc.drain() == []


def test_run_info_is_read_only() -> None:
    info = RunInfo.build(run_id="r", thread_id="t", metadata={"user": "u"})
    with pytest.raises(TypeError):
        info.metadata["user"] = "mallory"  # type: ignore[index]
    event = BeforeToolCallEvent("t", "c", {})
    with pytest.raises(AttributeError, match="read-only"):
        event.run = info  # type: ignore[misc]


async def test_resume_path_hooks_get_the_run_context() -> None:
    """The approved call performed by resume() fires hooks with its run."""
    from tulip.control import Action, ControlPolicy, InMemoryApprovals, gate_tool
    from tulip.testing import ScriptedModel

    @tool
    def book(hotel_id: str) -> str:
        """Book."""
        return json.dumps({"items": [{"id": hotel_id}]})

    store = InMemoryApprovals()
    gated = gate_tool(
        book,
        policy=ControlPolicy(
            require_verification_score=0.0, require_human_for=frozenset({"production"})
        ),
        action=lambda n, kw: Action(name=n, asset=kw["hotel_id"], environment="production"),
        approval=store,
        on_refusal="interrupt",
        principal="u1",
    )
    hook = _WidgetHook()
    agent = Agent(
        model=ScriptedModel([tool_call("book", call_id="b1", hotel_id="h9"), text("ok")]),
        tools=[gated],
        hooks=[hook],
        checkpointer=MemoryCheckpointer(),
        reflexion=False,
        grounding=False,
    )
    [e async for e in agent.run("book", thread_id="T", metadata={"user": "u1"})]
    [record] = store.pending()
    store.decide(record.approval_id, "approved", by="advisor")
    hook.seen.clear()
    events = [e async for e in agent.resume("ok", thread_id="T", perform_dangling=True)]
    customs = [e for e in events if isinstance(e, CustomEvent)]
    assert [c.data for c in customs] == [{"count": 1, "user": "u1"}]
    assert all(info is not None and info.thread_id == "T" for _, info in hook.seen)
