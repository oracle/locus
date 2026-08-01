# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""``Agent.run(stream_tokens=True)`` surfaces model chunks from the loop.

Without this a streaming UI has to abandon the agent loop and re-implement
ReAct over a raw provider client, losing admission, audit and the tool-loop
guard along the way.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from tulip import Agent
from tulip.core.events import ModelChunkEvent, TerminateEvent
from tulip.core.messages import Message
from tulip.models.base import ModelResponse


class _StreamingModel:
    """Yields chunks, and records whether the loop used stream() or complete()."""

    def __init__(self, pieces: list[str] | None = None) -> None:
        self.pieces = pieces or ["Hello", " ", "world"]
        self.completed = 0
        self.streamed = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self.completed += 1
        return ModelResponse(
            message=Message.assistant(content="".join(self.pieces)),
            usage={"prompt_tokens": 1, "completion_tokens": 3},
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunkEvent]:
        self.streamed += 1
        for piece in self.pieces:
            yield ModelChunkEvent(content=piece)
        yield ModelChunkEvent(reasoning="because")
        yield ModelChunkEvent(
            done=True,
            usage={"prompt_tokens": 1, "completion_tokens": 3},
            stop_reason="stop",
        )


def _agent(model: _StreamingModel) -> Agent:
    return Agent(model=model, tools=[], system_prompt="p")


@pytest.mark.asyncio
async def test_streaming_off_by_default() -> None:
    """Enabling it changes which event types a consumer sees, so it is opt-in."""
    model = _StreamingModel()
    events = [ev async for ev in _agent(model).run("hi")]

    assert not any(isinstance(e, ModelChunkEvent) for e in events)
    assert model.completed == 1
    assert model.streamed == 0


@pytest.mark.asyncio
async def test_stream_tokens_yields_chunks() -> None:
    model = _StreamingModel()
    events = [ev async for ev in _agent(model).run("hi", stream_tokens=True)]

    chunks = [e for e in events if isinstance(e, ModelChunkEvent)]
    assert model.streamed == 1
    assert model.completed == 0
    assert "".join(c.content or "" for c in chunks) == "Hello world"
    assert any(c.reasoning for c in chunks), "chain-of-thought must stream too"


@pytest.mark.asyncio
async def test_streamed_text_matches_the_final_message() -> None:
    """The assembled response must equal what the consumer rendered."""
    model = _StreamingModel()
    streamed: list[str] = []
    final = ""

    async for ev in _agent(model).run("hi", stream_tokens=True):
        if isinstance(ev, ModelChunkEvent) and ev.content:
            streamed.append(ev.content)
        elif isinstance(ev, TerminateEvent):
            final = ev.final_message or ""

    assert "".join(streamed) == final == "Hello world"


@pytest.mark.asyncio
async def test_usage_survives_the_streaming_path() -> None:
    """Metering must not depend on which path the loop took."""
    model = _StreamingModel()
    events = [ev async for ev in _agent(model).run("hi", stream_tokens=True)]
    terminate = next(e for e in events if isinstance(e, TerminateEvent))
    assert terminate.reason == "complete"


@pytest.mark.asyncio
async def test_falls_back_when_the_model_cannot_stream() -> None:
    """A transport without stream() must still work with stream_tokens=True."""

    class _NoStream:
        def __init__(self) -> None:
            self.completed = 0

        async def complete(self, messages, tools=None, **kwargs):  # noqa: ANN001, ANN003
            self.completed += 1
            return ModelResponse(
                message=Message.assistant(content="ok"),
                usage={"prompt_tokens": 1, "completion_tokens": 1},
            )

    model = _NoStream()
    events = [
        ev
        async for ev in Agent(model=model, tools=[], system_prompt="p").run(
            "hi", stream_tokens=True
        )
    ]

    assert model.completed == 1
    assert not any(isinstance(e, ModelChunkEvent) for e in events)
