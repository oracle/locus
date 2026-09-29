# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""A resumed turn reports tools exactly like a fresh one.

After a human approval, ``resume(..., perform_dangling=True)`` performs the
approved call and continues the loop on the resume path. That path built its
``ToolCompleteEvent`` by hand and executed tools without the progress merge,
so a consumer lost the structured result of the approved call and every
progress update for the rest of the turn — a widget-driven UI rendered
nothing after the guest pressed Confirm.
"""

from __future__ import annotations

import asyncio
from typing import Any

from tulip.agent import Agent
from tulip.control import Action, ControlPolicy, InMemoryApprovals, gate_tool
from tulip.core.events import (
    InterruptEvent,
    TerminateEvent,
    ToolCompleteEvent,
    ToolProgressEvent,
)
from tulip.memory.backends.memory import MemoryCheckpointer
from tulip.testing import ScriptedModel, text, tool_call
from tulip.tools import ToolOutput, report_progress, tool


@tool
def book_hotel(hotel_id: str) -> ToolOutput:
    """Book a hotel."""
    return ToolOutput(f"booked {hotel_id}", structured_content={"ref": "R1", "hotel": hotel_id})


@tool(emits_progress=True)
async def price_rooms(rooms: int) -> ToolOutput:
    """Price rooms one by one."""
    for i in range(1, rooms + 1):
        report_progress(i, rooms, f"room {i}")
        await asyncio.sleep(0)
    return ToolOutput("priced", structured_content={"priced": rooms})


def _gated(store: InMemoryApprovals) -> Any:
    return gate_tool(
        book_hotel,
        policy=ControlPolicy(
            require_verification_score=0.0, require_human_for=frozenset({"production"})
        ),
        action=lambda n, kw: Action(name=n, asset=kw["hotel_id"], environment="production"),
        approval=store,
        on_refusal="interrupt",
        principal="guest-1",
    )


async def test_resume_keeps_structured_results_and_streams_progress() -> None:
    store = InMemoryApprovals()
    agent = Agent(
        model=ScriptedModel(
            [
                tool_call("book_hotel", call_id="c1", hotel_id="h1"),
                tool_call("price_rooms", call_id="c2", rooms=3),
                text("all set"),
            ]
        ),
        tools=[_gated(store), price_rooms],
        checkpointer=MemoryCheckpointer(),
        reflexion=False,
        grounding=False,
    )
    first = [e async for e in agent.run("book it", thread_id="t1")]
    assert isinstance(first[-1], InterruptEvent)
    for record in store.pending():
        store.decide(record.approval_id, "approved", by="guest-1")

    events = [e async for e in agent.resume("ok", thread_id="t1", perform_dangling=True)]

    completes = {e.tool_call_id: e for e in events if isinstance(e, ToolCompleteEvent)}
    assert completes["c1"].structured_content == {"ref": "R1", "hotel": "h1"}
    assert completes["c2"].structured_content == {"priced": 3}

    progress = [e for e in events if isinstance(e, ToolProgressEvent)]
    assert [(p.progress, p.total) for p in progress] == [(1, 3), (2, 3), (3, 3)]
    # Progress arrives live: before the tool's completion, not after the turn.
    assert events.index(progress[-1]) < events.index(completes["c2"])
    assert isinstance(events[-1], TerminateEvent)
