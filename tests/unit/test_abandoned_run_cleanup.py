# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""A run abandoned in the middle of a tool call is finalized without errors.

A host whose event loop has no async-generator hooks (a Temporal workflow's
loop is one) can drop a run while it awaits a tool. The garbage collector then
finalizes the run's generators with no loop to await on.
"""

from __future__ import annotations

import asyncio
import gc
import sys
from collections.abc import AsyncIterator
from typing import Any

import pytest

from locus.agent import Agent
from locus.core.events import ModelChunkEvent
from locus.core.messages import Message, ToolCall
from locus.models.base import ModelResponse
from locus.tools.decorator import tool
from locus.tools.executor import ConcurrentExecutor
from locus.tools.registry import ToolRegistry


@tool
async def wait_forever() -> str:
    """Never returns."""
    await asyncio.Event().wait()
    return "unreachable"


class CallsTheTool:
    """A model whose only answer is a call to ``wait_forever``."""

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        call = ToolCall(id="call_1", name="wait_forever", arguments={})
        return ModelResponse(
            message=Message.assistant(content=None, tool_calls=[call]),
            stop_reason="tool_calls",
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunkEvent]:
        response = await self.complete(messages, tools, **kwargs)
        yield ModelChunkEvent(tool_calls=response.message.tool_calls, done=True)


def _abandon(make_stream: Any, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Start a stream on a loop without async-generator hooks, drop it mid-call,
    and return what Python reported while finalizing it."""
    reported: list[str] = []
    monkeypatch.setattr(
        sys,
        "unraisablehook",
        lambda unraisable: reported.append(
            f"{type(unraisable.exc_value).__name__}: {unraisable.exc_value}"
        ),
    )

    async def start() -> Any:
        sys.set_asyncgen_hooks(firstiter=None, finalizer=None)
        stream = make_stream()

        async def drain() -> None:
            async for _ in stream:
                pass

        task = asyncio.ensure_future(drain())
        await asyncio.sleep(0.05)
        assert not task.done()
        return stream, task

    loop = asyncio.new_event_loop()
    try:
        held = loop.run_until_complete(start())
        del held
        gc.collect()
    finally:
        loop.close()
    gc.collect()
    return reported


def test_abandoned_tool_batch_is_finalized_quietly(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = ToolRegistry()
    registry.register(wait_forever)

    def stream() -> Any:
        return ConcurrentExecutor().execute_streaming([ToolCall(name="wait_forever")], registry)

    assert _abandon(stream, monkeypatch) == []


def test_abandoned_agent_run_is_finalized_quietly(monkeypatch: pytest.MonkeyPatch) -> None:
    def stream() -> Any:
        return Agent(model=CallsTheTool(), tools=[wait_forever]).run("go")

    assert _abandon(stream, monkeypatch) == []


@pytest.mark.asyncio
async def test_closing_a_tool_batch_still_waits_for_its_cancelled_tasks() -> None:
    """On the loop that owns the tasks, closing the stream waits for them."""
    finished = asyncio.Event()

    @tool
    async def slow() -> str:
        """Blocks until cancelled."""
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()
        return "unreachable"

    registry = ToolRegistry()
    registry.register(slow)
    stream = ConcurrentExecutor().execute_streaming([ToolCall(name="slow")], registry)
    pending = asyncio.ensure_future(stream.__anext__())
    await asyncio.sleep(0.05)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    # The cancelled task has run its cleanup by the time the stream is closed.
    assert finished.is_set()
