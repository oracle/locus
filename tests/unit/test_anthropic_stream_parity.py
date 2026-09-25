# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""``AnthropicModel.stream`` sends what ``complete`` sends.

``stream()`` built its own request and dropped ``temperature``, prompt caching
(the ``cache_control`` on the system prompt and tool catalog) and
``response_format``, and its usage lacked the cache counters — so a streaming
agent paid full input price on every cached turn, ran at the server's default
temperature, and could not see that caching was off.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


pytest.importorskip("anthropic")

from tulip.core.messages import Message  # noqa: E402
from tulip.models.native.anthropic import AnthropicModel  # noqa: E402


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look something up.",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }
]


class _TextStream:
    def __init__(self, parts: list[str]) -> None:
        self._parts = list(parts)

    def __aiter__(self) -> _TextStream:
        return self

    async def __anext__(self) -> str:
        if not self._parts:
            raise StopAsyncIteration
        return self._parts.pop(0)


class _StreamCM:
    def __init__(self, final: Any, parts: list[str]) -> None:
        self.text_stream = _TextStream(parts)
        self._final = final

    async def __aenter__(self) -> _StreamCM:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def get_final_message(self) -> Any:
        return self._final


def _usage() -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=100,
        output_tokens=20,
        cache_creation_input_tokens=80,
        cache_read_input_tokens=15,
    )


def _model(
    final: Any, parts: list[str] | None = None, **config: Any
) -> tuple[AnthropicModel, dict[str, Any], dict[str, Any]]:
    model = AnthropicModel(model="claude-sonnet-4-6", api_key="sk-test", **config)  # noqa: S106
    streamed: dict[str, Any] = {}
    created: dict[str, Any] = {}

    def _stream(**kw: Any) -> _StreamCM:
        streamed.update(kw)
        return _StreamCM(final, parts or [])

    async def _create(**kw: Any) -> Any:
        created.update(kw)
        return final

    model._client = SimpleNamespace(  # type: ignore[assignment]
        messages=SimpleNamespace(stream=_stream, create=_create)
    )
    return model, streamed, created


def _final(content: list[Any] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        content=content or [SimpleNamespace(type="text", text="hi")],
        usage=_usage(),
        stop_reason="end_turn",
    )


async def test_stream_request_matches_complete_request() -> None:
    model, streamed, created = _model(_final(), prompt_cache=True, temperature=0.2)
    messages = [Message.system("be brief"), Message.user("hi")]

    await model.complete(messages, TOOLS)
    _ = [chunk async for chunk in model.stream(messages, TOOLS)]

    assert streamed == created
    assert streamed["temperature"] == 0.2
    assert streamed["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert streamed["tools"][-1]["cache_control"] == {"type": "ephemeral"}


async def test_stream_honours_per_call_temperature() -> None:
    model, streamed, _ = _model(_final())
    _ = [c async for c in model.stream([Message.user("hi")], temperature=0.9)]
    assert streamed["temperature"] == 0.9


async def test_stream_reports_cache_usage() -> None:
    model, _, _ = _model(_final(), parts=["hi"], prompt_cache=True)
    chunks = [c async for c in model.stream([Message.user("hi")])]
    assert chunks[-1].done is True
    assert chunks[-1].usage == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "cache_creation_input_tokens": 80,
        "cache_read_input_tokens": 15,
    }


async def test_stream_structured_output_matches_complete() -> None:
    block = SimpleNamespace(type="tool_use", id="tu_1", name="respond_with_schema", input={"a": 1})
    model, streamed, _ = _model(_final([block]))
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "A", "schema": {"type": "object"}},
    }
    chunks = [c async for c in model.stream([Message.user("hi")], response_format=response_format)]
    assert streamed["tool_choice"] == {"type": "tool", "name": "respond_with_schema"}
    assert "".join(c.content or "" for c in chunks) == '{"a": 1}'
    assert not any(c.tool_calls for c in chunks)
