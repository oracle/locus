# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``Target`` primitive — every variant round-trips offline."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from tulip.security import Target
from tulip.security.target import _extract_text


async def test_from_callable_sync() -> None:
    target = Target.from_callable(lambda p: f"echo:{p}", name="sync")
    assert target.kind == "callable"
    assert await target.send("hi") == "echo:hi"


async def test_from_callable_async() -> None:
    async def fn(prompt: str) -> str:
        return prompt.upper()

    target = Target.from_callable(fn, name="async")
    assert await target.send("ok") == "OK"


async def test_agent_target_captures_final_message() -> None:
    class _Event:
        def __init__(self, final_message: str | None) -> None:
            self.final_message = final_message

    class _FakeAgent:
        async def run(self, prompt: str) -> Any:  # async generator
            yield _Event(None)  # non-terminal event
            yield _Event(f"answer to {prompt}")

    target = Target.agent(_FakeAgent(), name="bot")
    assert target.kind == "agent"
    assert await target.send("ping") == "answer to ping"


async def test_a2a_target_wraps_async_sender() -> None:
    async def peer(prompt: str) -> str:
        return f"peer:{prompt}"

    target = Target.a2a(peer, name="peer-1")
    assert target.kind == "a2a"
    assert await target.send("x") == "peer:x"


async def test_endpoint_target_with_mock_transport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": "from-endpoint"})

    target = Target.endpoint(
        "https://bot.example/chat",
        transport=httpx.MockTransport(handler),
        metadata={"model": "test"},
    )
    assert target.kind == "endpoint"
    assert target.metadata["model"] == "test"
    assert await target.send("hello") == "from-endpoint"


async def test_endpoint_target_openai_chat_shape() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "chat-reply"}}]},
        )

    target = Target.endpoint("https://api.example/v1/chat", transport=httpx.MockTransport(handler))
    assert await target.send("hi") == "chat-reply"


def test_extract_text_dotted_path() -> None:
    body = {"choices": [{"message": {"content": "deep"}}]}
    assert _extract_text(body, "choices.0.message.content") == "deep"


def test_extract_text_falls_back_to_raw_json() -> None:
    # No known key and no path -> raw JSON, never an exception.
    out = _extract_text({"weird": {"shape": 1}}, None)
    assert "weird" in out


@pytest.mark.parametrize("key", ["response", "output", "text", "content", "message", "answer"])
def test_extract_text_common_keys(key: str) -> None:
    assert _extract_text({key: "v"}, None) == "v"
