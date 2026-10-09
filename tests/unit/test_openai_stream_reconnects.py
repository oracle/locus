"""OpenAIModel(stream_reconnects=n): a turn whose connection drops part way is asked again.

Inside an NVIDIA OpenShell box a connection is cut whenever the box's policy generation
moves on (its first settings poll after start does so). The OpenAI client retries a
request that failed but never a stream that broke after it began, so a box runner's turn
streaming at that instant ended the run with ``APIConnectionError``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import httpx
import openai
import pytest

from tests.unit.test_models_native_openai import _Chunk, _ChunkChoice, _Delta, _model_with
from tulip.core.messages import Message
from tulip.models.native import openai as openai_module
from tulip.models.native.openai import OpenAIModel, _is_disconnect


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://llm.example/v1/chat/completions")


def _cut() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=_request())


def _broken(chunks: list[_Chunk], exc: BaseException) -> AsyncIterator[_Chunk]:
    """A stream that yields ``chunks`` and then breaks with ``exc``."""

    async def gen() -> AsyncIterator[_Chunk]:
        for c in chunks:
            yield c
        raise exc

    return gen()


def _whole(text: str) -> AsyncIterator[_Chunk]:
    async def gen() -> AsyncIterator[_Chunk]:
        yield _Chunk(choices=[_ChunkChoice(delta=_Delta(content=text))])
        yield _Chunk(choices=[_ChunkChoice(delta=_Delta(), finish_reason="stop")])

    return gen()


def _half() -> _Chunk:
    return _Chunk(choices=[_ChunkChoice(delta=_Delta(content="half "))])


@pytest.fixture(autouse=True)
def _no_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(openai_module, "STREAM_RECONNECT_BASE_DELAY", 0.0)


async def _collect(model: OpenAIModel) -> str:
    return "".join([e.content or "" async for e in model.stream([Message.user("hi")])])


async def test_a_turn_cut_part_way_is_asked_again_and_yielded_once() -> None:
    client = AsyncMock()
    client.chat.completions.create.side_effect = [
        _broken([_half()], _cut()),
        _whole("the whole answer"),
    ]
    model = _model_with(client, stream_reconnects=3)

    text = await _collect(model)

    assert text == "the whole answer"  # never the half turn, never twice
    assert client.chat.completions.create.await_count == 2


async def test_without_reconnects_the_cut_is_raised_as_before() -> None:
    client = AsyncMock()
    client.chat.completions.create.side_effect = [_broken([_half()], _cut())]
    model = _model_with(client)

    with pytest.raises(openai.APIConnectionError):
        await _collect(model)
    assert client.chat.completions.create.await_count == 1


async def test_reconnects_are_bounded() -> None:
    client = AsyncMock()
    client.chat.completions.create.side_effect = [_broken([], _cut()) for _ in range(3)]
    model = _model_with(client, stream_reconnects=2)

    with pytest.raises(openai.APIConnectionError):
        await _collect(model)
    assert client.chat.completions.create.await_count == 3


async def test_a_refusal_is_not_asked_again() -> None:
    """A guard's 403 (``token_budget_exhausted``) is the server's answer, not a cut."""
    refused = openai.PermissionDeniedError(
        "token_budget_exhausted",
        response=httpx.Response(403, request=_request()),
        body={"reason_code": "token_budget_exhausted"},
    )
    client = AsyncMock()
    client.chat.completions.create.side_effect = [refused]
    model = _model_with(client, stream_reconnects=3)

    with pytest.raises(openai.PermissionDeniedError):
        await _collect(model)
    assert client.chat.completions.create.await_count == 1


async def test_a_dropped_httpx_stream_counts_as_a_cut() -> None:
    client = AsyncMock()
    client.chat.completions.create.side_effect = [
        _broken([], httpx.RemoteProtocolError("peer closed connection")),
        _whole("ok"),
    ]
    model = _model_with(client, stream_reconnects=1)

    assert await _collect(model) == "ok"


def test_is_disconnect() -> None:
    assert _is_disconnect(_cut())
    assert _is_disconnect(openai.APITimeoutError(request=_request()))
    assert _is_disconnect(httpx.ReadError("reset"))
    assert not _is_disconnect(
        openai.InternalServerError(
            "boom", response=httpx.Response(500, request=_request()), body=None
        )
    )
    assert not _is_disconnect(ValueError("not a network error"))


def test_stream_reconnects_defaults_off() -> None:
    assert OpenAIModel(model="gpt-4o").config.stream_reconnects == 0
