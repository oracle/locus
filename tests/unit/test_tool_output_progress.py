# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""``ToolOutput`` and ``report_progress`` for local tools.

The MCP client is the first producer of structured results and progress, but
the plumbing is generic: any tool can return a ``ToolOutput`` and any tool
declared ``emits_progress=True`` can stream progress.
"""

from __future__ import annotations

import asyncio
import copy
import pickle
import time
from typing import Any

from tulip.agent import Agent
from tulip.core.events import ToolCompleteEvent, ToolProgressEvent, ToolStartEvent
from tulip.hooks.provider import AfterToolCallEvent, HookPriority, HookProvider
from tulip.testing import ScriptedModel, text, tool_call
from tulip.tools import ToolOutput, report_progress, tool
from tulip.tools.context import current_tool_context, progress_sink


@tool
def find_rooms(city: str) -> ToolOutput:
    """Find rooms."""
    return ToolOutput(f"2 rooms in {city}", structured_content={"rooms": [1, 2]})


@tool
def book_room(room: int) -> ToolOutput:
    """Book a room."""
    return ToolOutput("room taken", structured_content={"code": "TAKEN"}, is_error=True)


@tool(emits_progress=True)
async def crawl(pages: int) -> str:
    """Crawl pages."""
    ctx = current_tool_context()
    assert ctx is not None
    assert ctx.tool_name == "crawl"
    for page in range(1, pages + 1):
        report_progress(page, pages, f"page {page}")
        await asyncio.sleep(0.01)
    return "crawled"


@tool(emits_progress=True)
def crawl_sync(pages: int) -> str:
    """Crawl pages from a worker thread."""
    for page in range(1, pages + 1):
        report_progress(page, pages)
        time.sleep(0.01)
    return "crawled"


def _agent(model: ScriptedModel, *tools: Any, **config: Any) -> Agent:
    return Agent(model=model, tools=list(tools), reflexion=False, grounding=False, **config)


async def _events(agent: Agent) -> list[Any]:
    return [e async for e in agent.run("go")]


def test_tool_output_is_a_string_that_round_trips() -> None:
    out = ToolOutput(
        "x", structured_content={"a": 1}, content_blocks=[{"type": "image"}], is_error=True
    )
    assert out == "x"
    assert isinstance(out, str)
    assert type(out.text) is str
    for clone in (pickle.loads(pickle.dumps(out)), copy.deepcopy(out)):  # noqa: S301 — our own bytes
        assert (clone.structured_content, clone.content_blocks, clone.is_error) == (
            {"a": 1},
            [{"type": "image"}],
            True,
        )
    assert repr(out) == "ToolOutput('x' [structured, 1 block(s), error])"
    assert repr(ToolOutput("y")) == "ToolOutput('y')"


async def test_structured_content_from_a_local_tool() -> None:
    model = ScriptedModel([tool_call("find_rooms", city="Rome"), text("ok")])
    events = await _events(_agent(model, find_rooms))
    [done] = [e for e in events if isinstance(e, ToolCompleteEvent)]
    assert done.result == "2 rooms in Rome"
    assert done.structured_content == {"rooms": [1, 2]}


async def test_a_returned_error_is_the_calls_error() -> None:
    model = ScriptedModel([tool_call("book_room", room=1), text("sorry")])
    events = await _events(_agent(model, book_room))
    [done] = [e for e in events if isinstance(e, ToolCompleteEvent)]
    assert (done.result, done.error, done.structured_content) == (
        None,
        "room taken",
        {"code": "TAKEN"},
    )
    assert model.received_messages[1][-1].content == "Error: room taken"


async def test_structured_content_survives_truncation_and_completion_order() -> None:
    model = ScriptedModel([tool_call("find_rooms", city="Rome"), text("ok")])
    agent = _agent(model, find_rooms, max_tool_result_length=4, tool_event_order="completion")
    [done] = [e for e in await _events(agent) if isinstance(e, ToolCompleteEvent)]
    assert done.structured_content == {"rooms": [1, 2]}


async def test_structured_content_survives_an_after_hook_replacement() -> None:
    class Redact(HookProvider):
        @property
        def priority(self) -> int:
            return HookPriority.OBSERVABILITY_DEFAULT

        async def on_after_tool_call(self, event: AfterToolCallEvent) -> None:
            event.result = "[redacted]"

    model = ScriptedModel([tool_call("find_rooms", city="Rome"), text("ok")])
    agent = Agent(
        model=model, tools=[find_rooms], hooks=[Redact()], reflexion=False, grounding=False
    )
    [done] = [e for e in await _events(agent) if isinstance(e, ToolCompleteEvent)]
    assert done.result == "[redacted]"
    assert done.structured_content == {"rooms": [1, 2]}


async def test_progress_from_an_async_tool_streams_live() -> None:
    model = ScriptedModel([tool_call("crawl", call_id="c1", pages=3), text("ok")])
    events = await _events(_agent(model, crawl))
    seq = [
        type(e).__name__
        for e in events
        if isinstance(e, (ToolStartEvent, ToolProgressEvent, ToolCompleteEvent))
    ]
    assert seq == [
        "ToolStartEvent",
        "ToolProgressEvent",
        "ToolProgressEvent",
        "ToolProgressEvent",
        "ToolCompleteEvent",
    ]
    progress = [e for e in events if isinstance(e, ToolProgressEvent)]
    assert [(p.progress, p.total, p.message, p.tool_call_id) for p in progress][-1] == (
        3.0,
        3.0,
        "page 3",
        "c1",
    )


async def test_progress_from_a_sync_tool_in_a_worker_thread() -> None:
    model = ScriptedModel([tool_call("crawl_sync", pages=2), text("ok")])
    events = await _events(_agent(model, crawl_sync, tool_execution="sequential"))
    assert [e.progress for e in events if isinstance(e, ToolProgressEvent)] == [1.0, 2.0]


def test_report_progress_without_a_listener_is_a_no_op() -> None:
    assert report_progress(1, 2) is False
    seen: list[ToolProgressEvent] = []
    with progress_sink(seen.append):
        assert report_progress(1) is False  # no tool call in flight
    assert seen == []


def test_lazy_model_exports() -> None:
    from tulip import models

    assert models.FallbackChain.__name__ == "FallbackChain"
    assert models.CircuitBreaker.__name__ == "CircuitBreaker"
