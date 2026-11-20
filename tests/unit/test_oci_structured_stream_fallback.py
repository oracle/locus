"""``OCIChatCompletionsModel.stream`` answers a structured turn in one piece when the service streams nothing (#585).

On OCI Generative AI the Llama family answers a ``json_schema`` ``response_format`` without
streaming it: an empty content delta, then ``finish_reason: "tool_calls"`` with no tool-call
delta. The same request without ``stream`` returns the JSON.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from locus.core.messages import Message
from locus.models.providers.oci import OCIChatCompletionsModel


BASE = "https://fake-oci.test/openai/v1"
CHAT = f"{BASE}/chat/completions"
SCHEMA = {
    "type": "json_schema",
    "json_schema": {"name": "VendorList", "strict": True, "schema": {"type": "object"}},
}
ANSWER = '{"vendors": [{"name": "A", "score": 0.9}]}'


class _NoopSigner:
    def do_request_sign(self, prepared: Any) -> None:  # noqa: PLR6301
        return None


def _model(model: str = "meta.llama-3.3-70b-instruct") -> OCIChatCompletionsModel:
    m = OCIChatCompletionsModel(
        model=model,
        auth_type="instance_principal",
        compartment_id="ocid1.compartment.oc1..fake",
        base_url=BASE,
        max_retries=0,
    )
    object.__setattr__(m, "_build_signer", lambda: _NoopSigner())
    return m


def _sse(frames: list[dict[str, Any]]) -> bytes:
    body = "".join(f"data: {json.dumps(f)}\n\n" for f in frames)
    return (body + "data: [DONE]\n\n").encode()


def _chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    choice: dict[str, Any] = {"index": 0, "delta": delta}
    if finish is not None:
        choice["finish_reason"] = finish
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "meta.llama-3.3-70b-instruct",
        "choices": [choice],
    }


SILENT_STREAM = _sse(
    [_chunk({"role": "assistant", "content": ""}), _chunk({}, finish="tool_calls")]
)
COMPLETION = {
    "id": "chatcmpl-2",
    "object": "chat.completion",
    "created": 2,
    "model": "meta.llama-3.3-70b-instruct",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": ANSWER}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18},
}


def _route(stream_body: bytes, completion: dict[str, Any]) -> list[dict[str, Any]]:
    """Mock the endpoint; return the list of request bodies it saw."""
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if body.get("stream"):
            return httpx.Response(
                200, content=stream_body, headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(200, json=completion)

    respx.post(CHAT).mock(side_effect=handler)
    return seen


async def _collect(model: OCIChatCompletionsModel, **kwargs: Any) -> list[Any]:
    return [
        e async for e in model.stream([Message.user("three vendors as JSON")], tools=None, **kwargs)
    ]


@respx.mock
@pytest.mark.asyncio
async def test_a_silent_structured_stream_is_answered_without_streaming() -> None:
    seen = _route(SILENT_STREAM, COMPLETION)
    events = await _collect(_model(), response_format=SCHEMA)

    assert [b.get("stream") for b in seen] == [True, None], (
        "one streamed attempt, then one plain completion"
    )
    assert seen[1]["response_format"] == SCHEMA, "the completion asks for the same schema"
    contents = [e.content for e in events if e.content]
    assert contents == [ANSWER]
    terminal = events[-1]
    assert terminal.done
    assert terminal.stop_reason == "stop"
    assert terminal.usage == {"prompt_tokens": 7, "completion_tokens": 11}


@respx.mock
@pytest.mark.asyncio
async def test_a_stream_that_delivers_content_is_left_alone() -> None:
    talkative = _sse(
        [
            _chunk({"role": "assistant", "content": ""}),
            _chunk({"content": '{"vendors": '}),
            _chunk({"content": "[]}"}, finish="stop"),
        ]
    )
    seen = _route(talkative, COMPLETION)
    events = await _collect(_model("openai.gpt-4.1"), response_format=SCHEMA)

    assert [b.get("stream") for b in seen] == [True], "no second request"
    assert "".join(e.content for e in events if e.content) == '{"vendors": []}'
    assert events[-1].done
    assert events[-1].stop_reason == "stop"


@respx.mock
@pytest.mark.asyncio
async def test_a_silent_stream_without_a_response_format_is_not_retried() -> None:
    seen = _route(SILENT_STREAM, COMPLETION)
    events = await _collect(_model())

    assert [b.get("stream") for b in seen] == [True], "an empty plain turn is the model's answer"
    assert [e.content for e in events if e.content] == []
    assert events[-1].done
    assert events[-1].stop_reason == "tool_calls"
