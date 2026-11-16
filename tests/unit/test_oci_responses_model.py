# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for :class:`OCIResponsesModel` with mocked HTTP.

Uses :mod:`respx` to intercept the outbound POST to ``/responses`` so
these tests run without OCI credentials or network access. Covers:

- Request body shape (continuation id, tools, instructions).
- Successful response parsing through the model.
- Error mapping: project-required, state-lost, generic errors.

Live integration is gated separately under ``tests/integration/`` and
requires real OCI auth.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

import httpx
import pytest
import respx
from pydantic import ValidationError

from locus.core.messages import Message
from locus.core.warnings import LocusWarning
from locus.models.providers.oci import (
    OCIProjectRequiredError,
    OCIResponsesModel,
    OCIResponsesStateLostError,
    OCIResponsesZeroDataRetentionError,
)


# Every test in this file constructs the model with auth_type+compartment_id
# (no real OCI auth) and stubs `_build_signer` so the httpx client doesn't
# try to load real signers. The OCIRequestSigner expects a signer object
# that exposes sign(); since respx intercepts before signing matters,
# we replace _build_signer with a no-op.


class _NoopSigner:
    """OCI signer stub. The real signer's only entry point used by
    ``OCIRequestSigner._sign`` is ``do_request_sign(prepared)`` — a
    no-op here is enough because respx intercepts before the signed
    request would actually be transmitted.
    """

    def do_request_sign(self, prepared: Any) -> None:  # noqa: PLR6301
        return None


def _make_model(**overrides: Any) -> OCIResponsesModel:
    """Construct an OCIResponsesModel pointed at a fake OCI base URL."""
    kwargs = {
        "model": "openai.gpt-5.5-pro",
        "auth_type": "instance_principal",
        "compartment_id": "ocid1.compartment.oc1..fake",
        "base_url": "https://fake-oci.test/openai/v1",
    }
    kwargs.update(overrides)
    model = OCIResponsesModel(**kwargs)
    # Replace the signer builder so the httpx client init doesn't try to
    # actually authenticate. Signing happens after respx intercepts.
    object.__setattr__(model, "_build_signer", lambda: _NoopSigner())
    return model


@respx.mock
@pytest.mark.asyncio
async def test_complete_sends_expected_request_body() -> None:
    model = _make_model()

    captured: dict[str, Any] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "id": "resp_first",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hi."}],
                    }
                ],
                "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
            },
        )

    respx.post("https://fake-oci.test/openai/v1/responses").mock(side_effect=_handler)

    response = await model.complete(
        [Message.system("Be brief."), Message.user("Hi.")],
        provider_state=None,
    )

    assert response.message.content == "Hi."
    assert response.provider_state == {"previous_response_id": "resp_first"}
    assert response.usage["prompt_tokens"] == 5

    body = captured["body"]
    assert body["model"] == "openai.gpt-5.5-pro"
    assert body["instructions"] == "Be brief."
    assert body["input"][0]["role"] == "user"
    assert "previous_response_id" not in body
    await model.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_complete_threads_previous_response_id() -> None:
    model = _make_model()

    captured: dict[str, Any] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "resp_second",
                "status": "completed",
                "output": [],
                "usage": {"input_tokens": 1, "output_tokens": 0, "total_tokens": 1},
            },
        )

    respx.post("https://fake-oci.test/openai/v1/responses").mock(side_effect=_handler)

    await model.complete(
        [Message.user("continue")],
        provider_state={"previous_response_id": "resp_first"},
    )

    assert captured["body"]["previous_response_id"] == "resp_first"
    await model.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_complete_passes_project_ocid_header_when_set() -> None:
    model = _make_model(project_ocid="ocid1.genaiagentproject.oc1..fake")

    captured: dict[str, Any] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={"id": "resp_x", "output": [], "usage": {}},
        )

    respx.post("https://fake-oci.test/openai/v1/responses").mock(side_effect=_handler)

    await model.complete([Message.user("hi")])
    assert captured["headers"].get("opc-project-id") == "ocid1.genaiagentproject.oc1..fake"
    # The Responses endpoint reads the project from ``OpenAI-Project``.
    assert captured["headers"].get("openai-project") == "ocid1.genaiagentproject.oc1..fake"
    await model.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_403_project_required_maps_to_typed_exception() -> None:
    model = _make_model()

    respx.post("https://fake-oci.test/openai/v1/responses").mock(
        return_value=httpx.Response(
            403,
            json={
                "error": {
                    "code": "InvalidParameter",
                    "message": "GenAI Project OCID is required for this feature",
                }
            },
        )
    )

    with pytest.raises(OCIProjectRequiredError) as exc_info:
        await model.complete([Message.user("hi")])

    assert "project_ocid=" in str(exc_info.value)
    await model.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_404_unknown_previous_response_id_maps_to_state_lost() -> None:
    model = _make_model()

    respx.post("https://fake-oci.test/openai/v1/responses").mock(
        return_value=httpx.Response(
            404,
            json={
                "error": {
                    "code": "NotFound",
                    "message": "previous_response_id not found or expired",
                }
            },
        )
    )

    with pytest.raises(OCIResponsesStateLostError):
        await model.complete(
            [Message.user("hi")],
            provider_state={"previous_response_id": "resp_old"},
        )
    await model.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_500_yields_generic_runtime_error_with_body() -> None:
    model = _make_model()

    respx.post("https://fake-oci.test/openai/v1/responses").mock(
        return_value=httpx.Response(500, text="upstream timeout")
    )

    with pytest.raises(RuntimeError) as exc_info:
        await model.complete([Message.user("hi")])

    assert "500" in str(exc_info.value)
    assert "upstream timeout" in str(exc_info.value)
    await model.aclose()


def test_init_rejects_both_auth_modes_set() -> None:
    with pytest.raises(ValueError, match="exactly one of"):
        OCIResponsesModel(
            model="x",
            profile="some-profile",
            auth_type="instance_principal",
        )


def test_init_rejects_invalid_auth_type() -> None:
    with pytest.raises(ValueError, match="auth_type must be one of"):
        OCIResponsesModel(
            model="x",
            auth_type="garbage",
            compartment_id="ocid1.compartment.oc1..fake",
        )


def test_init_requires_compartment_for_workload_identity() -> None:
    with pytest.raises(ValueError, match="compartment_id is required"):
        OCIResponsesModel(model="x", auth_type="instance_principal")


class TestHttpClientWiresRefreshSigner:
    """``_http_client`` must pass a *rebuild-the-signer* refresh callable
    into ``OCIRequestSigner`` so instance-principal federation tokens
    auto-refresh both on 401 and on the periodic timer. Rebuilding (vs
    in-place ``refresh_security_token``) mirrors oracle-samples/
    oci-genai-auth-python; without it the captured token expires after
    ~15-30 min and every subsequent /v1/responses call 401s. Same fix as
    openai_compat.py."""

    def test_token_signer_gets_rebuild_refresh_and_short_interval(self) -> None:
        from unittest.mock import MagicMock, patch

        from locus.models.providers.oci.openai_compat import _build_instance_principal_signer

        model = OCIResponsesModel(
            model="openai.gpt-5.5-pro",
            auth_type="instance_principal",
            compartment_id="ocid1.compartment.oc1..fake",
            base_url="https://fake-oci.test/openai/v1",
        )
        signer = MagicMock(name="instance_signer")
        object.__setattr__(model, "_build_signer", lambda: signer)

        with (
            patch("locus.models.providers.oci.responses.OCIRequestSigner") as mock_sig_ctor,
            patch("httpx.AsyncClient"),
        ):
            _ = model._http_client()
            kwargs = mock_sig_ctor.call_args.kwargs
            assert kwargs["compartment_id"] == "ocid1.compartment.oc1..fake"
            # Rebuild-on-refresh: the callable re-mints a fresh signer from
            # IMDS each refresh so the auth_flow refresh-on-401 + periodic
            # paths fire with valid credentials.
            assert kwargs["refresh_signer"] is _build_instance_principal_signer
            # 600s beats the typical 15-30min federation-token TTL.
            assert kwargs["refresh_interval"] == 600.0

    def test_static_signer_gets_no_refresh_callback(self) -> None:
        """User-principal API-key signers don't expire — refresh callback
        should be None and the refresh branches stay dormant."""
        from unittest.mock import MagicMock, patch

        with patch(
            "locus.models.providers.oci.openai_compat._load_profile_config",
            return_value={"tenancy": "ocid1.compartment.oc1..fake"},
        ):
            model = OCIResponsesModel(
                model="openai.gpt-5.5-pro",
                profile="MY_PROFILE",
                base_url="https://fake-oci.test/openai/v1",
            )
        # API-key signer — auth_type is None and the profile has no
        # security_token_file, so the dispatch yields None.
        static_signer = MagicMock(spec=["do_request_sign"])
        object.__setattr__(model, "_build_signer", lambda: static_signer)

        with (
            patch("locus.models.providers.oci.responses.OCIRequestSigner") as mock_sig_ctor,
            patch("httpx.AsyncClient"),
        ):
            _ = model._http_client()
            assert mock_sig_ctor.call_args.kwargs["refresh_signer"] is None


def test_server_stateful_default_true() -> None:
    """Default instance advertises server_stateful=True (full Responses flow)."""
    model = _make_model()
    assert model.server_stateful is True


def test_server_stateful_false_when_store_disabled() -> None:
    """store=False (ZDR tenancies) flips the model to stateless mode."""
    model = _make_model(store=False)
    assert model.server_stateful is False


@respx.mock
@pytest.mark.asyncio
async def test_complete_does_not_send_temperature_by_default() -> None:
    """OpenAI reasoning families (gpt-5, o-series) reject temperature
    on the Responses endpoint. Verified live against OCI gpt-5: the
    server returns 400 ``Unsupported parameter: 'temperature'`` when
    we send it. Drop it unless the caller explicitly overrides.
    """
    model = _make_model()
    captured: dict[str, Any] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "r", "output": [], "usage": {}})

    respx.post("https://fake-oci.test/openai/v1/responses").mock(side_effect=_handler)

    await model.complete([Message.user("hi")])
    assert "temperature" not in captured["body"]
    await model.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_store_false_sends_store_field_and_drops_previous_response_id() -> None:
    """ZDR mode: body carries ``store: false`` and never sends previous_response_id
    even when the agent threads one in."""
    model = _make_model(store=False)
    captured: dict[str, Any] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "r", "output": [], "usage": {}})

    respx.post("https://fake-oci.test/openai/v1/responses").mock(side_effect=_handler)

    # Even with provider_state threaded in, store=False drops it.
    await model.complete(
        [Message.user("hi")],
        provider_state={"previous_response_id": "resp_old"},
    )
    body = captured["body"]
    assert body["store"] is False
    assert "previous_response_id" not in body
    await model.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_store_true_default_sends_no_store_field_and_uses_continuation() -> None:
    """Default mode: omit the store field (server default applies) and thread
    previous_response_id when provided."""
    model = _make_model()  # store default True
    captured: dict[str, Any] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "r2", "output": [], "usage": {}})

    respx.post("https://fake-oci.test/openai/v1/responses").mock(side_effect=_handler)

    await model.complete(
        [Message.user("hi")],
        provider_state={"previous_response_id": "resp_prev"},
    )
    body = captured["body"]
    assert "store" not in body  # default True → omitted, server default wins
    assert body["previous_response_id"] == "resp_prev"
    await model.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_complete_drops_temperature_even_when_kwarg_passed() -> None:
    """Reasoning models (gpt-5, o-series) reject temperature. The Agent
    loop always passes temperature as a kwarg from its config default,
    so we drop it unconditionally on the Responses path."""
    model = _make_model()
    captured: dict[str, Any] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "r", "output": [], "usage": {}})

    respx.post("https://fake-oci.test/openai/v1/responses").mock(side_effect=_handler)

    await model.complete([Message.user("hi")], temperature=0.3)
    assert "temperature" not in captured["body"]
    await model.aclose()


# =============================================================================
# reasoning_effort
# =============================================================================


def _capture_body(captured: dict[str, Any]) -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "r", "output": [], "usage": {}})

    respx.post("https://fake-oci.test/openai/v1/responses").mock(side_effect=_handler)


def test_reasoning_effort_is_kept_on_the_config() -> None:
    assert _make_model().config.reasoning_effort is None
    assert _make_model(reasoning_effort="high").config.reasoning_effort == "high"


@respx.mock
@pytest.mark.asyncio
async def test_complete_sends_no_reasoning_by_default() -> None:
    """Unset, the body is unchanged: the server's default effort applies."""
    model = _make_model()
    captured: dict[str, Any] = {}
    _capture_body(captured)

    await model.complete([Message.user("hi")])
    assert "reasoning" not in captured["body"]
    await model.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_complete_sends_configured_reasoning_effort() -> None:
    model = _make_model(reasoning_effort="low")
    captured: dict[str, Any] = {}
    _capture_body(captured)

    await model.complete([Message.user("hi")])
    assert captured["body"]["reasoning"] == {"effort": "low"}
    await model.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_complete_reasoning_effort_kwarg_overrides_config() -> None:
    """A per-call kwarg (an agent's ``model_kwargs``) beats the config."""
    model = _make_model(reasoning_effort="low")
    captured: dict[str, Any] = {}
    _capture_body(captured)

    await model.complete([Message.user("hi")], reasoning_effort="high")
    assert captured["body"]["reasoning"] == {"effort": "high"}
    await model.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_complete_reasoning_effort_kwarg_without_config() -> None:
    model = _make_model()
    captured: dict[str, Any] = {}
    _capture_body(captured)

    await model.complete([Message.user("hi")], reasoning_effort="medium")
    assert captured["body"]["reasoning"] == {"effort": "medium"}
    await model.aclose()


# =============================================================================
# Project header signalled with a 400
# =============================================================================


@respx.mock
@pytest.mark.asyncio
async def test_400_naming_the_project_header_maps_to_typed_exception() -> None:
    """Non-OpenAI models on the Responses endpoint ask for the project with a 400."""
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
                    "param": None,
                    "type": "invalid_request_error",
                }
            },
        )
    )

    with pytest.raises(OCIProjectRequiredError, match="project_ocid="):
        await model.complete([Message.user("hi")])
    await model.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_400_mentioning_a_project_elsewhere_stays_generic() -> None:
    """Only the header wording means "send a project"; other 400s stay RuntimeError."""
    model = _make_model()
    respx.post("https://fake-oci.test/openai/v1/responses").mock(
        return_value=httpx.Response(
            400,
            json={"error": {"code": "LimitExceeded", "message": "project request quota exceeded"}},
        )
    )

    with pytest.raises(RuntimeError, match="quota exceeded") as exc_info:
        await model.complete([Message.user("hi")])
    assert not isinstance(exc_info.value, OCIProjectRequiredError)
    await model.aclose()


# =============================================================================
# Zero Data Retention
# =============================================================================

_URL = "https://fake-oci.test/openai/v1/responses"

#: The 400 a ZDR tenancy returns for every request carrying previous_response_id.
_ZDR_BODY: dict[str, Any] = {
    "error": {
        "code": "unsupported_parameter",
        "message": (
            "Previous response cannot be used for this organization due to Zero Data Retention."
        ),
        "param": "previous_response_id",
        "type": "invalid_request_error",
    }
}


def _tool_call_response(response_id: str, call_id: str) -> dict[str, Any]:
    return {
        "id": response_id,
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "call_id": call_id,
                "name": "add",
                "arguments": json.dumps({"a": 2, "b": 3}),
            }
        ],
        "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
    }


def _text_response(response_id: str, text: str) -> dict[str, Any]:
    return {
        "id": response_id,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
    }


def _zdr_warnings(record: Any) -> list[Any]:
    """Only the Locus ZDR warning: the interpreter may record others alongside."""
    return [
        w
        for w in record
        if issubclass(w.category, LocusWarning) and "Zero Data Retention" in str(w.message)
    ]


def test_store_defaults_to_auto_and_validates_its_values() -> None:
    assert _make_model().config.store == "auto"
    assert _make_model(store=True).config.store is True
    assert _make_model(store=False).config.store is False
    assert _make_model().zdr_detected is False
    with pytest.raises(ValidationError):
        _make_model(store="never")


def test_zdr_error_is_a_state_lost_error() -> None:
    """Handlers written for the expired-thread case keep catching ZDR."""
    assert issubclass(OCIResponsesZeroDataRetentionError, OCIResponsesStateLostError)


@respx.mock
@pytest.mark.asyncio
async def test_store_true_raises_the_typed_zdr_error_and_stays_stateful() -> None:
    """store=True keeps today's behaviour: the error propagates, nothing switches."""
    model = _make_model(store=True)
    respx.post(_URL).mock(return_value=httpx.Response(400, json=_ZDR_BODY))

    with pytest.raises(OCIResponsesZeroDataRetentionError, match="Zero Data Retention"):
        await model.complete(
            [Message.user("hi")], provider_state={"previous_response_id": "resp_1"}
        )
    assert model.server_stateful is True
    assert model.zdr_detected is False

    # Existing handlers for the parent type still see it.
    with pytest.raises(OCIResponsesStateLostError):
        await model.complete(
            [Message.user("hi")], provider_state={"previous_response_id": "resp_1"}
        )
    await model.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_store_auto_switches_to_stateless_after_the_zdr_error() -> None:
    """Under store="auto" the first ZDR 400 flips the model for good and warns once."""
    model = _make_model()
    bodies: list[dict[str, Any]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "previous_response_id" in body:
            return httpx.Response(400, json=_ZDR_BODY)
        return httpx.Response(200, json=_text_response("resp_2", "ok"))

    respx.post(_URL).mock(side_effect=_handler)

    with (
        pytest.warns(LocusWarning, match="Zero Data Retention") as record,
        pytest.raises(OCIResponsesZeroDataRetentionError),
    ):
        await model.complete(
            [Message.user("hi")], provider_state={"previous_response_id": "resp_1"}
        )
    assert len(_zdr_warnings(record)) == 1
    assert model.zdr_detected is True
    assert model.server_stateful is False

    # The next call is stateless even with a continuation id threaded in,
    # and there is no second warning.
    with warnings.catch_warnings(record=True) as again:
        warnings.simplefilter("always")
        response = await model.complete(
            [Message.user("hi")], provider_state={"previous_response_id": "resp_1"}
        )
    assert _zdr_warnings(again) == []
    assert response.message.content == "ok"
    assert bodies[-1]["store"] is False
    assert "previous_response_id" not in bodies[-1]
    await model.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_store_false_is_unchanged_by_the_zdr_detection() -> None:
    model = _make_model(store=False)
    captured: dict[str, Any] = {}
    _capture_body(captured)

    await model.complete([Message.user("hi")], provider_state={"previous_response_id": "r"})
    assert captured["body"]["store"] is False
    assert model.server_stateful is False
    assert model.zdr_detected is False
    await model.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_agent_run_completes_statelessly_on_a_zdr_tenancy() -> None:
    """First call 200 with a tool call, the continuation 400 ZDR: the runtime
    replays the turn with the full history and store=false, the run completes
    with the tool's result, the warning fires once, and a second run on the
    same model never tries the stateful path again."""
    from locus.agent import Agent
    from locus.tools.decorator import tool

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    model = _make_model()
    bodies: list[dict[str, Any]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "previous_response_id" in body:
            return httpx.Response(400, json=_ZDR_BODY)
        if any(item.get("type") == "function_call_output" for item in body["input"]):
            return httpx.Response(200, json=_text_response("resp_2", "5"))
        return httpx.Response(200, json=_tool_call_response("resp_1", "call_add"))

    respx.post(_URL).mock(side_effect=_handler)
    agent = Agent(model=model, tools=[add], max_iterations=4)

    with pytest.warns(LocusWarning, match="Zero Data Retention") as record:
        result = await agent.arun("Add 2 and 3.")

    assert result.success
    assert result.message == "5"
    assert len(_zdr_warnings(record)) == 1
    assert len(bodies) == 3
    first, rejected, replay = bodies
    assert "previous_response_id" not in first
    assert "store" not in first
    assert rejected["previous_response_id"] == "resp_1"
    # The replay is the stateless request the agent would have sent from
    # the start: full history, store=false, no continuation id.
    assert replay["store"] is False
    assert "previous_response_id" not in replay
    assert [item["type"] for item in replay["input"]] == [
        "message",
        "function_call",
        "function_call_output",
    ]
    assert replay["input"][0]["role"] == "user"
    assert replay["input"][1]["call_id"] == "call_add"
    assert replay["input"][2] == {
        "type": "function_call_output",
        "call_id": "call_add",
        "output": "5",
    }
    assert model.server_stateful is False
    assert result.state.provider_state is None

    # Same model, second run: stateless from the first request, no new warning.
    bodies.clear()
    with warnings.catch_warnings(record=True) as again:
        warnings.simplefilter("always")
        second = await agent.arun("Add 2 and 3 again.")
    assert _zdr_warnings(again) == []
    assert second.success
    assert second.message == "5"
    assert len(bodies) == 2
    assert all(body["store"] is False for body in bodies)
    assert all("previous_response_id" not in body for body in bodies)
    await model.aclose()
