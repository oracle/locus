# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""``resume`` is keyed strictly by thread, and refuses to fold a pending hold.

Two regressions, both on the human-approval path a consumer chat depends on:

* ``resume(thread_id="A")`` preferred whatever interrupt the agent held in
  memory — the most recent one, from thread B — and executed thread B's gated
  booking with B's arguments. Security-relevant: one user's approval performed
  another user's action.
* Resuming before the approval was decided re-invoked the gate, which
  interrupted again, and the raw ``__interrupt__`` JSON was folded in as the
  tool's result: the model read "the call returned", and the turn ended with
  the booking silently never made.
"""

from __future__ import annotations

from typing import Any

import pytest

from tulip.agent import Agent
from tulip.control import Action, ControlPolicy, InMemoryApprovals, gate_tool
from tulip.core import errors
from tulip.core.events import InterruptEvent, TerminateEvent, ToolCompleteEvent, TulipEvent
from tulip.memory.backends.memory import MemoryCheckpointer
from tulip.testing import ScriptedModel, text, tool_call
from tulip.tools.decorator import tool


calls: list[dict[str, Any]] = []


@tool
def book_hotel(hotel_id: str, nights: int) -> str:
    """Book a hotel."""
    calls.append({"hotel_id": hotel_id, "nights": nights})
    return f"booked {hotel_id} x{nights}"


def _gated(store: InMemoryApprovals) -> Any:
    return gate_tool(
        book_hotel,
        policy=ControlPolicy(
            require_verification_score=0.0, require_human_for=frozenset({"production"})
        ),
        action=lambda n, kw: Action(name=n, asset=kw["hotel_id"], environment="production"),
        approval=store,
        on_refusal="interrupt",
        principal="u1",
    )


async def _collect(stream: Any) -> list[TulipEvent]:
    return [e async for e in stream]


@pytest.fixture(autouse=True)
def _reset_calls() -> None:
    calls.clear()


async def _two_paused_threads(checkpointer: bool = True) -> tuple[Agent, InMemoryApprovals]:
    """One shared agent; thread A and thread B each paused on a gated booking."""
    store = InMemoryApprovals()
    model = ScriptedModel(
        [tool_call("book_hotel", call_id="cA", hotel_id="hA", nights=1)], repeat_last=True
    )
    agent = Agent(
        model=model,
        tools=[_gated(store)],
        checkpointer=MemoryCheckpointer() if checkpointer else None,
        reflexion=False,
        grounding=False,
    )
    events_a = await _collect(agent.run("book for A", thread_id="thread-A"))
    assert isinstance(events_a[-1], InterruptEvent)
    model._turns = [tool_call("book_hotel", call_id="cB", hotel_id="hB", nights=9)]
    events_b = await _collect(agent.run("book for B", thread_id="thread-B"))
    assert isinstance(events_b[-1], InterruptEvent)
    model._turns = [text("done")]
    return agent, store


@pytest.mark.parametrize("checkpointer", [True, False])
async def test_resume_thread_a_performs_only_thread_a(checkpointer: bool) -> None:
    agent, store = await _two_paused_threads(checkpointer)
    for record in store.pending():
        store.decide(record.approval_id, "approved", by="advisor")

    events = await _collect(agent.resume("ok", thread_id="thread-A", perform_dangling=True))

    assert calls == [{"hotel_id": "hA", "nights": 1}]
    completes = [e for e in events if isinstance(e, ToolCompleteEvent)]
    assert [e.tool_call_id for e in completes] == ["cA"]
    # Thread B is still paused, and resuming it performs B's booking only.
    assert agent.pending_interrupts() == ["thread-B"]
    await _collect(agent.resume("ok", thread_id="thread-B", perform_dangling=True))
    assert calls == [{"hotel_id": "hA", "nights": 1}, {"hotel_id": "hB", "nights": 9}]


async def test_resume_without_thread_id_is_refused_when_ambiguous() -> None:
    agent, _ = await _two_paused_threads()
    with pytest.raises(RuntimeError, match="pass thread_id"):
        await _collect(agent.resume("ok"))
    assert calls == []
    assert sorted(map(str, agent.pending_interrupts())) == ["thread-A", "thread-B"]


async def test_resume_without_thread_id_still_works_for_a_single_pause() -> None:
    store = InMemoryApprovals()
    agent = Agent(
        model=ScriptedModel(
            [tool_call("book_hotel", call_id="c1", hotel_id="h1", nights=2), text("final")]
        ),
        tools=[_gated(store)],
        reflexion=False,
        grounding=False,
    )
    await _collect(agent.run("book"))
    [record] = store.pending()
    store.decide(record.approval_id, "approved", by="advisor")
    events = await _collect(agent.resume("ok", perform_dangling=True))
    assert calls == [{"hotel_id": "h1", "nights": 2}]
    assert isinstance(events[-1], TerminateEvent)
    assert events[-1].final_message == "final"


async def test_resume_of_an_unknown_thread_rehydrates_or_raises() -> None:
    agent, _ = await _two_paused_threads(checkpointer=False)
    with pytest.raises(RuntimeError, match="No interrupt to resume"):
        await _collect(agent.resume("ok", thread_id="thread-C", perform_dangling=True))
    assert calls == []


async def test_resume_while_approval_pending_raises_and_keeps_thread_paused() -> None:
    store = InMemoryApprovals()
    model = ScriptedModel(
        [tool_call("book_hotel", call_id="c1", hotel_id="h1", nights=2), text("final")],
        repeat_last=True,
    )
    checkpointer = MemoryCheckpointer()
    agent = Agent(
        model=model,
        tools=[_gated(store)],
        checkpointer=checkpointer,
        reflexion=False,
        grounding=False,
    )
    await _collect(agent.run("book", thread_id="t1"))
    paused = await checkpointer.load("t1")
    assert paused is not None
    model_calls_before = model.call_count

    # Not decided yet: the typed error, nothing yielded, nothing folded.
    stream = agent.resume("ok", thread_id="t1", perform_dangling=True)
    with pytest.raises(errors.ApprovalPendingError) as info:
        await stream.__anext__()  # raised before any event is yielded
    assert info.value.thread_id == "t1"
    assert info.value.interrupt_id == "c1"
    assert info.value.metadata["approval_id"] == store.pending()[0].approval_id
    assert calls == []
    assert model.call_count == model_calls_before
    after = await checkpointer.load("t1")
    assert after is not None
    assert after.messages == paused.messages
    assert not any("__interrupt__" in (m.content or "") for m in after.messages)
    assert agent.pending_interrupts() == ["t1"]

    # Decide, resume again: the booking now happens and the run completes.
    [record] = store.pending()
    store.decide(record.approval_id, "approved", by="advisor")
    events = await _collect(agent.resume("ok", thread_id="t1", perform_dangling=True))
    assert calls == [{"hotel_id": "h1", "nights": 2}]
    tool_msgs = [m.content for m in model.received_messages[-1] if m.role.value == "tool"]
    assert tool_msgs == ["booked h1 x2"]
    assert isinstance(events[-1], TerminateEvent)
    assert events[-1].reason == "complete"


async def test_pending_check_also_applies_after_a_process_restart() -> None:
    """Rehydrated from the checkpointer (no in-memory interrupt) — same rule."""
    store = InMemoryApprovals()
    checkpointer = MemoryCheckpointer()

    def make() -> Agent:
        return Agent(
            model=ScriptedModel(
                [tool_call("book_hotel", call_id="c1", hotel_id="h1", nights=2), text("final")],
                repeat_last=True,
            ),
            tools=[_gated(store)],
            checkpointer=checkpointer,
            reflexion=False,
            grounding=False,
        )

    await _collect(make().run("book", thread_id="t1"))
    fresh = make()
    with pytest.raises(errors.ApprovalPendingError):
        await _collect(fresh.resume("ok", thread_id="t1", perform_dangling=True))
    assert calls == []
