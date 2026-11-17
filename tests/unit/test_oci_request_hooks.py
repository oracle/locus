# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The ``on_request`` / ``on_response`` inspection hooks of the HTTP-backed OCI models.

Both :class:`OCIResponsesModel` and :class:`OCIChatCompletionsModel` take a
pair of sync callables and run them as httpx event hooks on the client they
build, so a request-shape test (or a user debugging what goes on the wire)
no longer has to reach into ``model._http_client()``. The outbound POST is
intercepted with :mod:`respx`; the signer is a no-op stub.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from locus.core.messages import Message
from locus.models.providers.oci import OCIChatCompletionsModel, OCIResponsesModel
from locus.models.providers.oci.openai_compat import request_event_hooks


BASE = "https://fake-oci.test/openai/v1"
COMPARTMENT = "ocid1.compartment.oc1..fake"


class _NoopSigner:
    def do_request_sign(self, prepared: Any) -> None:  # noqa: PLR6301
        return None


def _responses_model(**overrides: Any) -> OCIResponsesModel:
    kwargs: dict[str, Any] = {
        "model": "openai.gpt-5.5-pro",
        "auth_type": "instance_principal",
        "compartment_id": COMPARTMENT,
        "base_url": BASE,
        "store": False,
    }
    kwargs.update(overrides)
    model = OCIResponsesModel(**kwargs)
    object.__setattr__(model, "_build_signer", lambda: _NoopSigner())
    return model


def _chat_model(**overrides: Any) -> OCIChatCompletionsModel:
    kwargs: dict[str, Any] = {
        "model": "openai.gpt-5.5",
        "auth_type": "instance_principal",
        "compartment_id": COMPARTMENT,
        "base_url": BASE,
        "max_retries": 0,
    }
    kwargs.update(overrides)
    model = OCIChatCompletionsModel(**kwargs)
    object.__setattr__(model, "_build_signer", lambda: _NoopSigner())
    return model


_RESPONSES_REPLY: dict[str, Any] = {
    "id": "resp_1",
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hi"}],
        }
    ],
    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
}

_CHAT_REPLY: dict[str, Any] = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "openai.gpt-5.5",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


class _Capture:
    """Collects what the two hooks see."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.statuses: list[int] = []

    def on_request(self, request: httpx.Request) -> None:
        self.requests.append(request)

    def on_response(self, response: httpx.Response) -> None:
        self.statuses.append(response.status_code)

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]


# =============================================================================
# The helper
# =============================================================================


def test_no_callables_install_no_hooks() -> None:
    assert request_event_hooks(None, None) == {}


async def test_sync_callables_are_wrapped_into_awaitable_hooks() -> None:
    seen: list[Any] = []
    hooks = request_event_hooks(seen.append, None)
    assert list(hooks) == ["request"]
    request = httpx.Request("POST", "https://x.test/")
    await hooks["request"][0](request)
    assert seen == [request]

    hooks = request_event_hooks(None, seen.append)
    assert list(hooks) == ["response"]
    response = httpx.Response(204)
    await hooks["response"][0](response)
    assert seen == [request, response]


# =============================================================================
# Absent by default
# =============================================================================


def test_hooks_are_optional_and_absent_by_default() -> None:
    model = _responses_model()
    assert model.config.on_request is None
    assert model.config.on_response is None
    assert model._http_client().event_hooks == {"request": [], "response": []}

    chat = _chat_model()
    assert chat.config.on_request is None
    assert chat.config.on_response is None


# =============================================================================
# Responses transport
# =============================================================================


@respx.mock
async def test_request_hook_captures_the_responses_body_and_response_hook_the_status() -> None:
    capture = _Capture()
    model = _responses_model(on_request=capture.on_request, on_response=capture.on_response)
    route = respx.post(f"{BASE}/responses").mock(
        return_value=httpx.Response(200, json=_RESPONSES_REPLY)
    )

    result = await model.complete([Message.system("be brief"), Message.user("hi")])

    assert result.message.content == "hi"
    assert route.called
    assert len(capture.bodies) == 1
    body = capture.bodies[0]
    assert body["model"] == "openai.gpt-5.5-pro"
    assert body["instructions"] == "be brief"
    assert body["store"] is False
    assert body["input"] == [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
    ]
    assert capture.statuses == [200]


@respx.mock
async def test_request_hook_sees_the_request_after_signing() -> None:
    """The hook runs inside the auth flow: the signer's headers are already on."""
    capture = _Capture()
    model = _responses_model(on_request=capture.on_request)
    respx.post(f"{BASE}/responses").mock(return_value=httpx.Response(200, json=_RESPONSES_REPLY))

    await model.complete([Message.user("hi")])

    [request] = capture.requests
    assert request.headers["opc-compartment-id"] == COMPARTMENT
    assert request.url.path.endswith("/responses")


@respx.mock
async def test_request_hook_captures_the_compaction_body() -> None:
    capture = _Capture()
    model = _responses_model(on_request=capture.on_request, on_response=capture.on_response)
    respx.post(f"{BASE}/responses/compact").mock(
        return_value=httpx.Response(
            200,
            json={
                "output": [{"type": "compaction", "id": "cmp_1", "encrypted_content": "Zm9v"}],
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
        )
    )

    compacted, _usage = await model.compact_history(
        [Message.system("rules"), Message.user("one"), Message.assistant("two")]
    )

    assert len(compacted) == 1
    [body] = capture.bodies
    assert set(body) == {"model", "input", "instructions"}
    assert body["instructions"] == "rules"
    assert capture.statuses == [200]


@respx.mock
async def test_response_hook_sees_an_error_status_before_the_model_raises() -> None:
    capture = _Capture()
    model = _responses_model(on_response=capture.on_response)
    respx.post(f"{BASE}/responses").mock(
        return_value=httpx.Response(400, json={"error": {"message": "bad request"}})
    )

    with pytest.raises(RuntimeError):
        await model.complete([Message.user("hi")])

    assert capture.statuses == [400]


# =============================================================================
# Chat-completions transport
# =============================================================================


@respx.mock
async def test_request_hook_captures_the_chat_completions_body() -> None:
    capture = _Capture()
    model = _chat_model(on_request=capture.on_request, on_response=capture.on_response)
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=_CHAT_REPLY)
    )

    result = await model.complete([Message.user("hi")])

    assert result.message.content == "hi"
    assert route.called
    [body] = capture.bodies
    assert body["model"] == "openai.gpt-5.5"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert capture.requests[0].headers["opc-compartment-id"] == COMPARTMENT
    assert capture.statuses == [200]


def test_chat_model_accepts_the_hooks_through_a_splatted_config() -> None:
    """``model_kwargs`` from the ``oci:`` factory arrive as one dict."""
    capture = _Capture()
    model_kwargs: dict[str, Any] = {
        "on_request": capture.on_request,
        "on_response": capture.on_response,
    }
    model = _chat_model(**model_kwargs)
    assert model.config.on_request == capture.on_request
    assert model.config.on_response == capture.on_response
