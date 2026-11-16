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
from locus.core.messages import Message, Role, ToolCall
from locus.core.warnings import LocusWarning
from locus.models.providers.oci import (
    OCI_RESPONSES_ITEMS_METADATA_KEY,
    OCIProjectRequiredError,
    OCIResponsesModel,
    OCIResponsesStateLostError,
    OCIResponsesZeroDataRetentionError,
)
from locus.models.providers.oci._responses_parse import (
    _translate_tool_schema,
    build_request_body,
    parse_compaction_response,
    parse_response,
    parse_stream_event,
    reasoning_text,
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
        assert out == {
            "tool_calls": [{"id": "item_42", "item_id": "item_42", "arguments_delta": "args"}]
        }

    def test_output_item_added_carries_name_and_call_id(self) -> None:
        out = parse_stream_event(
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "add_numbers",
                    "arguments": "",
                },
            }
        )
        assert out == {
            "tool_calls": [
                {"id": "call_1", "item_id": "fc_1", "name": "add_numbers", "arguments_delta": ""}
            ]
        }

    def test_output_item_done_carries_the_full_arguments(self) -> None:
        out = parse_stream_event(
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "add_numbers",
                    "arguments": '{"a": 2, "b": 3}',
                },
            }
        )
        assert out["tool_calls"][0]["final"] is True
        assert out["tool_calls"][0]["arguments"] == '{"a": 2, "b": 3}'

    def test_non_function_output_items_stay_structural(self) -> None:
        added = {"type": "response.output_item.added", "item": {"type": "message", "id": "msg_1"}}
        assert parse_stream_event(added) == {}

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
    assert headers == {
        "OpenAI-Project": "ocid1.genaiagentproject.oc1..x",
        "opc-project-id": "ocid1.genaiagentproject.oc1..x",
    }


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


# =============================================================================
# reasoning_effort on the streaming path
# =============================================================================


def _capture_stream_body(captured: dict[str, Any]) -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        captured["body"] = _json.loads(request.content)
        return httpx.Response(
            200,
            content=_sse(['{"type": "response.completed", "response": {"id": "resp_r"}}']),
        )

    respx.post("https://fake-oci.test/openai/v1/responses").mock(side_effect=_handler)


@respx.mock
@pytest.mark.asyncio
async def test_stream_sends_no_reasoning_by_default() -> None:
    model = _make_model()
    captured: dict[str, Any] = {}
    _capture_stream_body(captured)

    async for _ in model.stream([Message.user("hi")]):
        pass
    assert "reasoning" not in captured["body"]
    await model.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_stream_sends_configured_reasoning_effort() -> None:
    model = _make_model(reasoning_effort="low")
    captured: dict[str, Any] = {}
    _capture_stream_body(captured)

    async for _ in model.stream([Message.user("hi")]):
        pass
    assert captured["body"]["reasoning"] == {"effort": "low"}
    await model.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_stream_reasoning_effort_kwarg_overrides_config() -> None:
    model = _make_model(reasoning_effort="low")
    captured: dict[str, Any] = {}
    _capture_stream_body(captured)

    async for _ in model.stream([Message.user("hi")], reasoning_effort="high"):
        pass
    assert captured["body"]["reasoning"] == {"effort": "high"}
    await model.aclose()


# =============================================================================
# Error bodies on the streaming path
# =============================================================================


@respx.mock
@pytest.mark.asyncio
async def test_stream_4xx_surfaces_the_server_message() -> None:
    """A non-2xx stream is read before classification, so the server's
    message reaches the caller instead of ``httpx.ResponseNotRead``."""
    model = _make_model()
    respx.post("https://fake-oci.test/openai/v1/responses").mock(
        return_value=httpx.Response(
            400, json={"code": "400", "message": "Proxy model has been retired."}
        )
    )

    with pytest.raises(RuntimeError, match="Proxy model has been retired") as exc_info:
        async for _ in model.stream([Message.user("hi")]):
            pass
    assert not isinstance(exc_info.value, httpx.ResponseNotRead)
    await model.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_stream_400_project_header_maps_to_typed_exception() -> None:
    model = _make_model(model="google.gemini-2.5-flash")
    respx.post("https://fake-oci.test/openai/v1/responses").mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {
                    "code": "invalid_value",
                    "message": (
                        "Non-OpenAI models require 'OpenAI-Project' or "
                        "'opc-conversation-store-id' header for the Responses API."
                    ),
                }
            },
        )
    )

    with pytest.raises(OCIProjectRequiredError, match="project_ocid="):
        async for _ in model.stream([Message.user("hi")]):
            pass
    await model.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_stream_unknown_previous_response_id_maps_to_state_lost() -> None:
    model = _make_model()
    respx.post("https://fake-oci.test/openai/v1/responses").mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {
                    "code": "unsupported_parameter",
                    "message": "Previous response cannot be used for this organization.",
                    "param": "previous_response_id",
                }
            },
        )
    )

    with pytest.raises(OCIResponsesStateLostError):
        async for _ in model.stream(
            [Message.user("hi")], provider_state={"previous_response_id": "resp_old"}
        ):
            pass
    await model.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_stream_zdr_400_flips_an_auto_model_and_raises_the_typed_error() -> None:
    model = _make_model()
    respx.post("https://fake-oci.test/openai/v1/responses").mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {
                    "code": "unsupported_parameter",
                    "message": (
                        "Previous response cannot be used for this organization "
                        "due to Zero Data Retention."
                    ),
                    "param": "previous_response_id",
                }
            },
        )
    )

    async def _drain() -> None:
        async for _ in model.stream(
            [Message.user("hi")], provider_state={"previous_response_id": "resp_old"}
        ):
            pass

    with (
        pytest.warns(LocusWarning, match="Zero Data Retention") as record,
        pytest.raises(OCIResponsesZeroDataRetentionError),
    ):
        await _drain()
    zdr = [
        w
        for w in record
        if issubclass(w.category, LocusWarning) and "Zero Data Retention" in str(w.message)
    ]
    assert len(zdr) == 1
    assert model.zdr_detected is True
    assert model.server_stateful is False
    await model.aclose()


# =============================================================================
# Reasoning controls and verbatim replay (parse layer)
# =============================================================================

_REASONING_ITEM: dict[str, Any] = {
    "id": "rs_1",
    "type": "reasoning",
    "summary": [{"type": "summary_text", "text": "Add the two numbers."}],
    "encrypted_content": "gAAAAABencrypted",
}
_FUNCTION_CALL_ITEM: dict[str, Any] = {
    "id": "fc_1",
    "type": "function_call",
    "call_id": "call_1",
    "name": "add",
    "arguments": '{"a": 2, "b": 3}',
}
_COMPACTION_ITEM: dict[str, Any] = {
    "id": "cmp_1",
    "type": "compaction",
    "encrypted_content": "gAAAAABcompacted",
}


class TestReasoningControls:
    def test_nothing_is_sent_when_both_are_unset(self) -> None:
        for store in (True, False):
            body = build_request_body([Message.user("hi")], model="m", store=store)
            assert "reasoning" not in body
            assert "include" not in body

    def test_the_dict_is_sent_as_given(self) -> None:
        body = build_request_body(
            [Message.user("hi")], model="m", reasoning={"effort": "high", "summary": "auto"}
        )
        assert body["reasoning"] == {"effort": "high", "summary": "auto"}

    def test_the_effort_fills_a_dict_without_one(self) -> None:
        body = build_request_body(
            [Message.user("hi")], model="m", reasoning={"summary": "auto"}, reasoning_effort="low"
        )
        assert body["reasoning"] == {"summary": "auto", "effort": "low"}

    def test_the_dict_wins_on_conflict(self) -> None:
        body = build_request_body(
            [Message.user("hi")], model="m", reasoning={"effort": "high"}, reasoning_effort="low"
        )
        assert body["reasoning"] == {"effort": "high"}

    def test_the_callers_dict_is_not_mutated(self) -> None:
        given = {"summary": "auto"}
        build_request_body([Message.user("hi")], model="m", reasoning=given, reasoning_effort="low")
        assert given == {"summary": "auto"}

    def test_encrypted_reasoning_is_requested_only_when_stateless_and_set(self) -> None:
        stateful = build_request_body([Message.user("hi")], model="m", reasoning_effort="low")
        assert "include" not in stateful
        stateless = build_request_body(
            [Message.user("hi")], model="m", store=False, reasoning_effort="low"
        )
        assert stateless["include"] == ["reasoning.encrypted_content"]
        with_dict = build_request_body(
            [Message.user("hi")], model="m", store=False, reasoning={"summary": "auto"}
        )
        assert with_dict["include"] == ["reasoning.encrypted_content"]


class TestVerbatimReplay:
    def test_parse_response_keeps_the_raw_items_when_a_reasoning_item_is_present(self) -> None:
        message, _, _, _ = parse_response(
            {"id": "r", "output": [_REASONING_ITEM, _FUNCTION_CALL_ITEM]}
        )
        assert message.metadata[OCI_RESPONSES_ITEMS_METADATA_KEY] == [
            _REASONING_ITEM,
            _FUNCTION_CALL_ITEM,
        ]
        assert [tc.id for tc in message.tool_calls] == ["call_1"]
        assert reasoning_text(message) == "Add the two numbers."

    def test_parse_response_keeps_nothing_without_reasoning(self) -> None:
        message, _, _, _ = parse_response({"id": "r", "output": [_FUNCTION_CALL_ITEM]})
        assert message.metadata == {}
        assert reasoning_text(message) is None

    def test_a_turn_with_raw_items_is_replayed_verbatim(self) -> None:
        turn, _, _, _ = parse_response(
            {"id": "r", "output": [_REASONING_ITEM, _FUNCTION_CALL_ITEM]}
        )
        result = Message(role=Role.TOOL, content="5", tool_call_id="call_1", name="add")
        body = build_request_body([Message.user("add"), turn, result], model="m", store=False)
        assert body["input"][1:3] == [_REASONING_ITEM, _FUNCTION_CALL_ITEM]
        assert body["input"][3] == {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "5",
        }

    def test_a_hand_built_turn_is_reconstructed_without_item_ids(self) -> None:
        turn = Message.assistant(tool_calls=[ToolCall(id="call_1", name="add", arguments={"a": 2})])
        body = build_request_body([Message.user("add"), turn], model="m", store=False)
        assert body["input"][1] == {
            "type": "function_call",
            "call_id": "call_1",
            "name": "add",
            "arguments": '{"a": 2}',
        }

    def test_a_compaction_reply_becomes_messages_that_replay_the_item(self) -> None:
        kept = {
            "id": "msg_1",
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "My name is Ada."}],
        }
        messages, usage = parse_compaction_response(
            {
                "object": "response.compaction",
                "output": [kept, _COMPACTION_ITEM],
                "usage": {"input_tokens": 75, "output_tokens": 21, "total_tokens": 96},
            }
        )
        assert [m.role for m in messages] == [Role.USER, Role.ASSISTANT]
        assert messages[0].content == "My name is Ada."
        assert messages[1].metadata[OCI_RESPONSES_ITEMS_METADATA_KEY] == [_COMPACTION_ITEM]
        assert usage == {"prompt_tokens": 75, "completion_tokens": 21, "total_tokens": 96}

        body = build_request_body(
            [Message.system("sys"), *messages, Message.user("What is my name?")],
            model="m",
            store=False,
        )
        assert body["instructions"] == "sys"
        assert body["input"][1] == _COMPACTION_ITEM
        assert body["input"][2]["role"] == "user"


@respx.mock
@pytest.mark.asyncio
async def test_stream_sends_the_reasoning_dict_and_asks_for_encrypted_items_when_stateless() -> (
    None
):
    model = _make_model(store=False, reasoning={"effort": "low", "summary": "auto"})
    captured: dict[str, Any] = {}
    _capture_stream_body(captured)

    async for _ in model.stream([Message.user("hi")]):
        pass
    assert captured["body"]["reasoning"] == {"effort": "low", "summary": "auto"}
    assert captured["body"]["include"] == ["reasoning.encrypted_content"]
    await model.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_stream_reasoning_kwarg_replaces_the_config_dict() -> None:
    model = _make_model(reasoning={"effort": "low"})
    captured: dict[str, Any] = {}
    _capture_stream_body(captured)

    async for _ in model.stream([Message.user("hi")], reasoning={"summary": "concise"}):
        pass
    assert captured["body"]["reasoning"] == {"summary": "concise"}
    assert "include" not in captured["body"]
    await model.aclose()


# =============================================================================
# Edge paths of the parse layer and the stream loop
# =============================================================================


class TestParseLayerEdgePaths:
    def test_replay_metadata_without_dict_items_falls_back_to_reconstruction(self) -> None:
        turn = Message(
            role=Role.ASSISTANT,
            content="hi",
            tool_calls=[ToolCall(id="c1", name="add", arguments={})],
            metadata={OCI_RESPONSES_ITEMS_METADATA_KEY: ["not an item", 7]},
        )
        body = build_request_body([turn], model="m", store=False)
        assert [item["type"] for item in body["input"]] == ["message", "function_call"]
        assert body["input"][1]["arguments"] == "{}"

    def test_an_empty_user_message_sends_no_input_item(self) -> None:
        body = build_request_body([Message.user(""), Message.user("hi")], model="m")
        assert len(body["input"]) == 1

    def test_temperature_is_sent_when_given(self) -> None:
        body = build_request_body([Message.user("hi")], model="m", temperature=0.2)
        assert body["temperature"] == 0.2

    def test_translate_tool_schema_keeps_only_the_fields_present(self) -> None:
        assert _translate_tool_schema({"type": "function", "function": {"name": "ping"}}) == {
            "type": "function",
            "name": "ping",
        }

    def test_parse_response_skips_non_dict_output_items(self) -> None:
        message, _, _, _ = parse_response(
            {
                "id": "r",
                "output": [
                    "garbage",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "refusal", "refusal": "no"},
                            {"type": "output_text", "text": ""},
                            {"type": "output_text", "text": "ok"},
                        ],
                    },
                ],
            }
        )
        assert message.content == "ok"
        assert message.metadata == {}

    def test_parse_response_unparseable_tool_arguments_become_empty(self) -> None:
        message, _, _, _ = parse_response(
            {
                "id": "r",
                "output": [
                    {"type": "function_call", "call_id": "c1", "name": "add", "arguments": "{oops"}
                ],
            }
        )
        assert message.tool_calls[0].arguments == {}

    def test_reasoning_text_skips_blocks_without_text(self) -> None:
        message = Message(
            role=Role.ASSISTANT,
            content=None,
            metadata={
                OCI_RESPONSES_ITEMS_METADATA_KEY: [
                    {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text"}, "raw"],
                        "content": [],
                    },
                    {"type": "function_call", "call_id": "c1", "name": "add", "arguments": "{}"},
                    "not an item",
                ]
            },
        )
        assert reasoning_text(message) is None
        assert reasoning_text(Message.assistant(content="plain")) is None

    def test_compaction_response_edge_items(self) -> None:
        messages, usage = parse_compaction_response(
            {
                "output": [
                    "garbage",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Noted."}],
                    },
                    {"type": "function_call", "call_id": "c1", "name": "add", "arguments": "{}"},
                    {"type": "compaction", "id": "cmp_1", "encrypted_content": "x"},
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        )
        assert [m.role for m in messages] == [Role.ASSISTANT, Role.ASSISTANT, Role.ASSISTANT]
        assert messages[0].content == "Noted."
        assert messages[1].content is None
        assert messages[1].metadata[OCI_RESPONSES_ITEMS_METADATA_KEY][0]["type"] == "function_call"
        assert messages[2].metadata[OCI_RESPONSES_ITEMS_METADATA_KEY][0]["type"] == "compaction"
        assert usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    def test_parse_stream_event_ignores_structural_events(self) -> None:
        assert parse_stream_event({"type": "response.created"}) == {}
        assert parse_stream_event({"type": "response.output_item.added", "item": {}}) == {}


@respx.mock
@pytest.mark.asyncio
async def test_stream_without_done_sentinel_skips_structural_events_and_bad_arguments() -> None:
    """The SSE body ends without ``[DONE]``, carries lifecycle events the
    translator drops, and a tool call whose accumulated arguments are not
    JSON: the stream still yields the call (empty arguments) and ``done``."""
    model = _make_model()
    body = "".join(
        f"data: {event}\n\n"
        for event in [
            '{"type": "response.created", "response": {"id": "resp_x"}}',
            '{"type": "response.function_call_arguments.delta", "call_id": "call_1", "delta": "{oops"}',
            '{"type": "response.completed", "response": {"id": "resp_x"}}',
        ]
    ).encode()
    respx.post("https://fake-oci.test/openai/v1/responses").mock(
        return_value=httpx.Response(200, content=body)
    )

    chunks = [event async for event in model.stream([Message.user("hi")])]

    tool_chunks = [c for c in chunks if c.tool_calls]
    assert len(tool_chunks) == 1
    assert tool_chunks[0].tool_calls is not None
    assert tool_chunks[0].tool_calls[0].id == "call_1"
    assert tool_chunks[0].tool_calls[0].arguments == {}
    assert chunks[-1].done is True
    await model.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_error_envelope_that_is_not_an_object_falls_back_to_the_body_text() -> None:
    model = _make_model()
    respx.post("https://fake-oci.test/openai/v1/responses").mock(
        return_value=httpx.Response(503, json={"error": "upstream unavailable"})
    )

    with pytest.raises(RuntimeError, match="503") as exc_info:
        await model.complete([Message.user("hi")])
    assert "upstream unavailable" in str(exc_info.value)
    assert not isinstance(exc_info.value, OCIResponsesStateLostError)
    await model.aclose()


# =============================================================================
# Streamed tool calls carry the function name and the call id (#474)
# =============================================================================


def _fc_item(call_id: str, item_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": item_id,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def _events(*events: dict[str, Any]) -> bytes:
    import json as _json

    return _sse([_json.dumps(e) for e in events])


@respx.mock
@pytest.mark.asyncio
async def test_streamed_tool_call_has_the_name_and_call_id_complete_would_return() -> None:
    """The deltas only carry the item id; the call id and the name come from the
    ``output_item`` events, and the emitted ToolCall must match ``complete()``."""
    model = _make_model()
    respx.post("https://fake-oci.test/openai/v1/responses").mock(
        return_value=httpx.Response(
            200,
            content=_events(
                {"type": "response.created", "response": {"id": "resp_1"}},
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": _fc_item("call_1", "fc_1", "add_numbers", ""),
                },
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": "fc_1",
                    "delta": '{"a": 2, ',
                },
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": "fc_1",
                    "delta": '"b": 3}',
                },
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": "fc_1",
                    "arguments": '{"a": 2, "b": 3}',
                },
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": _fc_item("call_1", "fc_1", "add_numbers", '{"a": 2, "b": 3}'),
                },
                {"type": "response.completed", "response": {"id": "resp_1"}},
            ),
        )
    )

    calls = []
    async for chunk in model.stream([Message.user("add 2 and 3")]):
        if chunk.tool_calls:
            calls.extend(chunk.tool_calls)
    assert [(c.id, c.name, c.arguments) for c in calls] == [
        ("call_1", "add_numbers", {"a": 2, "b": 3})
    ]
    await model.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_streamed_deltas_before_the_added_event_still_land_on_the_call_id() -> None:
    """Out-of-order delivery: deltas keyed by the item id are moved under the
    call id once ``output_item.added`` names it; no ``done`` item is needed."""
    model = _make_model()
    respx.post("https://fake-oci.test/openai/v1/responses").mock(
        return_value=httpx.Response(
            200,
            content=_events(
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": "fc_9",
                    "delta": '{"x": 1}',
                },
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": _fc_item("call_9", "fc_9", "noop", ""),
                },
                {"type": "response.completed", "response": {"id": "resp_9"}},
            ),
        )
    )
    calls = []
    async for chunk in model.stream([Message.user("go")]):
        if chunk.tool_calls:
            calls.extend(chunk.tool_calls)
    assert [(c.id, c.name, c.arguments) for c in calls] == [("call_9", "noop", {"x": 1})]
    await model.aclose()
