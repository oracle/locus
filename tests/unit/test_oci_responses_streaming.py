# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Streaming + edge-case tests for :class:`OCIResponsesModel`.

Covers the SSE stream path, the SSE line parser, ``aclose()``, and the
parser branches not exercised by the request/response shape tests.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from locus.core.events import ModelChunkEvent
from locus.core.messages import Message
from locus.models.providers.oci import OCIResponsesModel
from locus.models.providers.oci._responses_parse import (
    _translate_tool_schema,
    build_request_body,
    parse_response,
    parse_stream_event,
)


class _NoopSigner:
    def do_request_sign(self, prepared: Any) -> None:  # noqa: PLR6301
        return None


def _make_model(**overrides: Any) -> OCIResponsesModel:
    kwargs: dict[str, Any] = {
        "model": "openai.gpt-5.5-pro",
        "auth_type": "instance_principal",
        "compartment_id": "ocid1.compartment.oc1..fake",
        "base_url": "https://fake-oci.test/openai/v1",
    }
    kwargs.update(overrides)
    model = OCIResponsesModel(**kwargs)
    object.__setattr__(model, "_build_signer", lambda: _NoopSigner())
    return model


# =============================================================================
# Streaming end-to-end
# =============================================================================


def _sse(events: list[str]) -> bytes:
    """Build an SSE payload: ``data: <line>\\n\\n`` per event, ``data: [DONE]`` terminator."""
    body = "".join(f"data: {e}\n\n" for e in events)
    body += "data: [DONE]\n\n"
    return body.encode()


@respx.mock
@pytest.mark.asyncio
async def test_stream_yields_content_tool_calls_and_done() -> None:
    """End-to-end streaming: text deltas, tool-call args accumulated, then done."""
    model = _make_model()

    payload = _sse(
        [
            '{"type": "response.output_text.delta", "delta": "Hello "}',
            '{"type": "response.output_text.delta", "delta": "world."}',
            '{"type": "response.function_call_arguments.delta", '
            '"call_id": "call_1", "delta": "{\\"q\\":"}',
            '{"type": "response.function_call_arguments.delta", '
            '"call_id": "call_1", "delta": "\\"weather\\"}"}',
            '{"type": "response.completed", "response": {"id": "resp_stream_1"}}',
        ]
    )

    respx.post("https://fake-oci.test/openai/v1/responses").mock(
        return_value=httpx.Response(200, content=payload)
    )

    chunks: list[ModelChunkEvent] = []
    async for event in model.stream([Message.user("hi")]):
        chunks.append(event)

    # Two text deltas + one accumulated tool-call list + done.
    text_chunks = [c for c in chunks if c.content]
    assert [c.content for c in text_chunks] == ["Hello ", "world."]

    tool_call_chunks = [c for c in chunks if c.tool_calls]
    assert len(tool_call_chunks) == 1
    calls = tool_call_chunks[0].tool_calls
    assert len(calls) == 1
    assert calls[0].id == "call_1"
    # Args concatenated and JSON-parsed.
    assert calls[0].arguments == {"q": "weather"}

    done_chunks = [c for c in chunks if c.done]
    assert len(done_chunks) == 1

    await model.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_stream_error_event_raises() -> None:
    """An SSE error event aborts the stream with a RuntimeError."""
    model = _make_model()

    payload = _sse(
        [
            '{"type": "response.output_text.delta", "delta": "starting"}',
            '{"type": "response.error", "error": {"message": "rate limited"}}',
        ]
    )
    respx.post("https://fake-oci.test/openai/v1/responses").mock(
        return_value=httpx.Response(200, content=payload)
    )

    with pytest.raises(RuntimeError, match="rate limited"):
        async for _ in model.stream([Message.user("hi")]):
            pass
    await model.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_stream_with_previous_response_id() -> None:
    """Streaming threads the continuation id into the request body too."""
    model = _make_model()

    captured: dict[str, Any] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        captured["body"] = _json.loads(request.content)
        return httpx.Response(
            200,
            content=_sse(
                [
                    '{"type": "response.completed", "response": {"id": "resp_2"}}',
                ]
            ),
        )

    respx.post("https://fake-oci.test/openai/v1/responses").mock(side_effect=_handler)

    async for _ in model.stream(
        [Message.user("continue")],
        provider_state={"previous_response_id": "resp_1"},
    ):
        pass

    assert captured["body"]["previous_response_id"] == "resp_1"
    assert captured["body"]["stream"] is True
    await model.aclose()


# =============================================================================
# aclose() cleanup
# =============================================================================


@respx.mock
@pytest.mark.asyncio
async def test_aclose_idempotent() -> None:
    """aclose() can be called multiple times safely."""
    model = _make_model()

    respx.post("https://fake-oci.test/openai/v1/responses").mock(
        return_value=httpx.Response(
            200,
            json={"id": "resp_x", "output": [], "usage": {}},
        )
    )

    await model.complete([Message.user("hi")])
    await model.aclose()
    await model.aclose()  # no-op the second time
    # _client should be None after close.
    assert model._client is None


# =============================================================================
# Parser edge cases (push _responses_parse.py over the 90% line)
# =============================================================================


class TestBuildRequestBodyEdgeCases:
    def test_extra_field_merges_into_body(self) -> None:
        body = build_request_body(
            [Message.user("hi")],
            model="x",
            extra={"custom_knob": "value"},
        )
        assert body["custom_knob"] == "value"

    def test_response_format_passes_through(self) -> None:
        body = build_request_body(
            [Message.user("hi")],
            model="x",
            response_format={"type": "json_schema", "json_schema": {"name": "X", "schema": {}}},
        )
        assert body["response_format"]["type"] == "json_schema"

    def test_assistant_message_in_history_becomes_message_item(self) -> None:
        """If a caller hand-primes the history, prior assistant text passes through."""
        body = build_request_body(
            [Message.assistant("Earlier reply."), Message.user("follow-up")],
            model="x",
        )
        roles = [item.get("role") for item in body["input"] if item.get("type") == "message"]
        assert "assistant" in roles
        assert "user" in roles

    def test_empty_system_messages_are_dropped(self) -> None:
        body = build_request_body(
            [Message.system(""), Message.user("hi")],
            model="x",
        )
        assert "instructions" not in body

    def test_translate_tool_schema_passes_through_unknown_types(self) -> None:
        """Built-in OCI tools (e.g. file_search) come through unchanged."""
        builtin = {"type": "file_search", "vector_store_ids": ["vs_1"]}
        assert _translate_tool_schema(builtin) == builtin


class TestParseResponseEdgeCases:
    def test_empty_output_array(self) -> None:
        """No assistant content + no tool calls — message has content=None."""
        msg, _, _, _ = parse_response({"id": "resp_q", "output": []})
        assert msg.content is None
        assert msg.tool_calls == []

    def test_unknown_output_item_type_is_ignored(self) -> None:
        """Reasoning / refusal items don't crash the parser."""
        payload = {
            "id": "resp_r",
            "output": [
                {"type": "reasoning", "summary": "I thought about it."},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello."}],
                },
            ],
        }
        msg, _, _, _ = parse_response(payload)
        assert msg.content == "Hello."

    def test_tool_call_id_falls_back_to_id_field(self) -> None:
        """Some Responses revisions emit ``id`` instead of ``call_id``."""
        payload = {
            "id": "resp_t",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_alt",  # not call_id
                    "name": "x",
                    "arguments": "{}",
                }
            ],
        }
        msg, _, _, _ = parse_response(payload)
        assert msg.tool_calls[0].id == "fc_alt"


class TestParseStreamEventEdgeCases:
    def test_completed_without_response_id_emits_done_only(self) -> None:
        out = parse_stream_event({"type": "response.completed", "response": {}})
        assert out == {"done": True}

    def test_function_call_delta_with_item_id_fallback(self) -> None:
        """call_id missing — falls back to item_id."""
        out = parse_stream_event(
            {
                "type": "response.function_call.delta",
                "item_id": "item_42",
                "delta": "args",
            }
        )
        assert out == {"tool_calls": [{"id": "item_42", "arguments_delta": "args"}]}

    def test_nested_error_event(self) -> None:
        """Some servers emit ``foo.error`` instead of ``response.error``."""
        out = parse_stream_event({"type": "stream.error", "error": {"message": "transport closed"}})
        assert out == {"error": "transport closed"}


# =============================================================================
# Constructor edge cases (compartment resolution)
# =============================================================================


def test_init_requires_compartment_under_workload_auth() -> None:
    """Workload auth without compartment_id raises immediately.

    The model follows ``OCIChatCompletionsModel``'s precedence exactly: env-var
    fallback runs only after the compartment-required check, and only
    matters in profile mode where the check doesn't fire. Under
    workload auth (``auth_type=``), an explicit compartment_id kwarg
    is mandatory.
    """
    with pytest.raises(ValueError, match="compartment_id is required"):
        OCIResponsesModel(model="x", auth_type="instance_principal")


def test_init_accepts_profile_with_explicit_compartment() -> None:
    """Profile mode with explicit compartment_id constructs cleanly."""
    model = OCIResponsesModel(
        model="x",
        profile="DEFAULT",
        compartment_id="ocid1.compartment.oc1..explicit",
    )
    assert model.config.compartment_id == "ocid1.compartment.oc1..explicit"
    assert model.config.profile == "DEFAULT"


def test_init_profile_mode_resolves_compartment_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In profile mode without explicit compartment, env-var wins over profile tenancy."""
    monkeypatch.setenv("OCI_COMPARTMENT", "ocid1.compartment.oc1..fromenv")
    model = OCIResponsesModel(model="x", profile="DEFAULT")
    assert model.config.compartment_id == "ocid1.compartment.oc1..fromenv"


def test_init_profile_mode_falls_back_to_tenancy_when_profile_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unreadable profile file leaves compartment_id None — no exception."""
    monkeypatch.delenv("OCI_COMPARTMENT", raising=False)
    monkeypatch.delenv("OCI_COMPARTMENT_ID", raising=False)
    # Point at a path that definitely doesn't exist so _load_profile_config
    # raises and the constructor swallows it.
    model = OCIResponsesModel(
        model="x",
        profile="DEFAULT",
        config_file="/tmp/definitely-not-an-oci-config-xyz",  # noqa: S108
    )
    assert model.config.compartment_id is None


# =============================================================================
# _http_client + _extra_headers
# =============================================================================


@respx.mock
@pytest.mark.asyncio
async def test_http_client_is_cached_across_calls() -> None:
    """Repeated complete() reuses the same httpx client."""
    model = _make_model()
    respx.post("https://fake-oci.test/openai/v1/responses").mock(
        return_value=httpx.Response(200, json={"id": "r", "output": [], "usage": {}})
    )

    await model.complete([Message.user("hi")])
    first_client = model._client
    await model.complete([Message.user("hi again")])
    assert model._client is first_client
    await model.aclose()


def test_extra_headers_empty_when_no_project_ocid() -> None:
    """No project OCID → no opc-project-id header (the default path)."""
    model = _make_model()
    assert model._extra_headers() == {}


def test_extra_headers_includes_project_ocid_when_set() -> None:
    model = _make_model(project_ocid="ocid1.genaiagentproject.oc1..x")
    headers = model._extra_headers()
    assert headers == {"opc-project-id": "ocid1.genaiagentproject.oc1..x"}


# =============================================================================
# _iter_sse_events parser
# =============================================================================


@respx.mock
@pytest.mark.asyncio
async def test_complete_raises_runtime_error_on_httpx_transport_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transport-level httpx error becomes a Locus RuntimeError."""
    model = _make_model()
    respx.post("https://fake-oci.test/openai/v1/responses").mock(
        side_effect=httpx.ConnectError("network unreachable")
    )

    with pytest.raises(RuntimeError, match="OCI Responses request failed"):
        await model.complete([Message.user("hi")])
    await model.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_complete_raises_runtime_error_on_non_json_body() -> None:
    """A 200 response with a non-JSON body surfaces a clear RuntimeError."""
    model = _make_model()
    respx.post("https://fake-oci.test/openai/v1/responses").mock(
        return_value=httpx.Response(200, content=b"<html>oops</html>")
    )

    with pytest.raises(RuntimeError, match="non-JSON body"):
        await model.complete([Message.user("hi")])
    await model.aclose()


def test_build_signer_instance_principal(monkeypatch: pytest.MonkeyPatch) -> None:
    """_build_signer dispatches to the instance-principal helper."""
    sentinel = object()
    monkeypatch.setattr(
        "locus.models.providers.oci.responses._build_instance_principal_signer",
        lambda: sentinel,
    )
    model = OCIResponsesModel(
        model="x",
        auth_type="instance_principal",
        compartment_id="ocid1.compartment.oc1..fake",
    )
    assert model._build_signer() is sentinel


def test_build_signer_resource_principal(monkeypatch: pytest.MonkeyPatch) -> None:
    """_build_signer dispatches to the resource-principal helper."""
    sentinel = object()
    monkeypatch.setattr(
        "locus.models.providers.oci.responses._build_resource_principal_signer",
        lambda: sentinel,
    )
    model = OCIResponsesModel(
        model="x",
        auth_type="resource_principal",
        compartment_id="ocid1.compartment.oc1..fake",
    )
    assert model._build_signer() is sentinel


def test_build_signer_profile_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """_build_signer dispatches to the profile helper in profile mode."""
    sentinel = object()
    monkeypatch.setattr(
        "locus.models.providers.oci.responses._build_signer_from_profile",
        lambda profile, config_file: sentinel,
    )
    model = OCIResponsesModel(
        model="x",
        profile="DEFAULT",
        compartment_id="ocid1.compartment.oc1..fake",
    )
    assert model._build_signer() is sentinel


def test_profile_compartment_resolves_from_tenancy(monkeypatch: pytest.MonkeyPatch) -> None:
    """When no env var and no explicit kwarg, profile mode picks up tenancy from the profile."""
    monkeypatch.delenv("OCI_COMPARTMENT", raising=False)
    monkeypatch.delenv("OCI_COMPARTMENT_ID", raising=False)
    monkeypatch.setattr(
        "locus.models.providers.oci.responses._load_profile_config",
        lambda profile, config_file: {"tenancy": "ocid1.tenancy.oc1..fromprofile"},
    )
    model = OCIResponsesModel(model="x", profile="DEFAULT")
    assert model.config.compartment_id == "ocid1.tenancy.oc1..fromprofile"


@respx.mock
@pytest.mark.asyncio
async def test_stream_skips_non_data_lines_and_malformed_json() -> None:
    """SSE lines that aren't ``data:`` or that can't be parsed are dropped."""
    model = _make_model()
    # Mix of comments, blank lines, malformed JSON, then a valid completion.
    body = (
        b": this is a comment line\n\n"
        b"event: ignored\n\n"
        b"data: {not valid json\n\n"
        b'data: {"type": "response.completed", "response": {"id": "resp_end"}}\n\n'
        b"data: [DONE]\n\n"
    )
    respx.post("https://fake-oci.test/openai/v1/responses").mock(
        return_value=httpx.Response(200, content=body)
    )

    chunks: list[ModelChunkEvent] = []
    async for event in model.stream([Message.user("hi")]):
        chunks.append(event)

    # Only the completion event reaches us; the rest are silently dropped.
    assert any(c.done for c in chunks)
    await model.aclose()
