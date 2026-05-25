# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for the native LiteLLM model provider.

These tests mock ``litellm.acompletion`` so they run without network access,
without an OCI tenancy, and without the ``litellm`` package needing to make
real calls. The integration suite (``tests/integration/test_litellm_*.py``)
covers the live OCI Generative AI Infrastructure path.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from locus.core.messages import Message
from locus.models.base import ModelResponse
from locus.models.native.litellm import (
    LiteLLMConfig,
    LiteLLMModel,
    _model_fixes_temperature,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestLiteLLMConfig:
    def test_defaults(self):
        cfg = LiteLLMConfig(model="oci/xai.grok-4.20")
        assert cfg.model == "oci/xai.grok-4.20"
        assert cfg.max_tokens == 4096
        assert cfg.temperature == 0.7
        assert cfg.drop_params is True
        assert cfg.max_retries == 2
        assert cfg.oci_region is None
        assert cfg.api_key is None

    def test_oci_fields_settable(self):
        cfg = LiteLLMConfig(
            model="oci/cohere.command-latest",
            oci_region="us-chicago-1",
            oci_user="ocid1.user.oc1..xxx",
            oci_compartment_id="ocid1.tenancy.oc1..xxx",
        )
        assert cfg.oci_region == "us-chicago-1"
        assert cfg.oci_user == "ocid1.user.oc1..xxx"
        assert cfg.oci_compartment_id == "ocid1.tenancy.oc1..xxx"

    def test_extra_kwargs_allowed(self):
        # ``model_config = {"extra": "allow"}`` on ModelConfig — provider-specific
        # kwargs should round-trip without raising.
        cfg = LiteLLMConfig(model="oci/xai.grok-4.20", weird_passthrough_param=42)
        assert cfg.weird_passthrough_param == 42


# ---------------------------------------------------------------------------
# Model wiring / completion kwargs
# ---------------------------------------------------------------------------


class TestModelWiring:
    def test_construction_uses_shortcut_kwargs(self):
        m = LiteLLMModel(
            model="oci/cohere.command-latest",
            oci_region="us-chicago-1",
            oci_compartment_id="ocid1.tenancy.oc1..xxx",
        )
        assert m.config.model == "oci/cohere.command-latest"
        assert m.config.oci_region == "us-chicago-1"

    def test_completion_kwargs_only_includes_set_oci_fields(self):
        """OCI fields left as ``None`` must not be forwarded to litellm
        (otherwise LiteLLM's env-var resolution can't fill them in)."""
        m = LiteLLMModel(model="oci/cohere.command-latest", oci_region="us-chicago-1")
        kw = m._completion_kwargs
        assert kw["model"] == "oci/cohere.command-latest"
        assert kw["oci_region"] == "us-chicago-1"
        assert "oci_user" not in kw
        assert "oci_fingerprint" not in kw
        assert "oci_compartment_id" not in kw

    def test_completion_kwargs_includes_drop_params(self):
        m = LiteLLMModel(model="oci/xai.grok-4.20")
        assert m._completion_kwargs["drop_params"] is True
        assert m._completion_kwargs["num_retries"] == 2


# ---------------------------------------------------------------------------
# Fixed-temperature detection
# ---------------------------------------------------------------------------


class TestFixedTemperatureModels:
    @pytest.mark.parametrize(
        "model",
        [
            "oci/openai.gpt-5",
            "oci/openai.gpt-5-mini",
            "oci/openai.gpt-5-nano",
            "oci/openai.o1",
            "oci/openai.o3",
            "oci/openai.o4",
        ],
    )
    def test_known_fixed_temperature_models(self, model):
        assert _model_fixes_temperature(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "oci/xai.grok-4.20",
            "oci/cohere.command-latest",
            "oci/meta.llama-4-maverick-17b-128e-instruct-fp8",
            "oci/google.gemini-2.5-flash",
            "gpt-4o",
        ],
    )
    def test_non_fixed_temperature_models(self, model):
        assert _model_fixes_temperature(model) is False


# ---------------------------------------------------------------------------
# Tool-argument coercion (LiteLLM hands back JSON strings, Locus wants dict)
# ---------------------------------------------------------------------------


class TestToolArgumentCoercion:
    def test_json_string_to_dict(self):
        out = LiteLLMModel._coerce_tool_arguments('{"location": "Boston"}')
        assert out == {"location": "Boston"}

    def test_passthrough_dict(self):
        out = LiteLLMModel._coerce_tool_arguments({"already": "a dict"})
        assert out == {"already": "a dict"}

    def test_malformed_json_returns_empty(self):
        # Must not crash an agent loop on a malformed-arguments chunk.
        out = LiteLLMModel._coerce_tool_arguments("{not valid json")
        assert out == {}

    def test_json_scalar_wrapped(self):
        out = LiteLLMModel._coerce_tool_arguments("42")
        assert out == {"value": 42}

    def test_none_returns_empty(self):
        assert LiteLLMModel._coerce_tool_arguments(None) == {}


# ---------------------------------------------------------------------------
# complete() — mocked litellm.acompletion
# ---------------------------------------------------------------------------


def _mock_chat_response(
    content: str | None = "Hello back.",
    tool_calls: list[Any] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    reasoning_tokens: int | None = None,
    finish_reason: str = "stop",
) -> SimpleNamespace:
    """Build a SimpleNamespace shaped like a litellm ModelResponse."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    details = None
    if reasoning_tokens is not None:
        details = SimpleNamespace(reasoning_tokens=reasoning_tokens)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        completion_tokens_details=details,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


@pytest.mark.asyncio
class TestComplete:
    async def test_basic_completion(self):
        m = LiteLLMModel(model="oci/cohere.command-latest")
        mock_ac = AsyncMock(return_value=_mock_chat_response(content="Tokyo."))
        with patch("litellm.acompletion", mock_ac):
            resp = await m.complete(messages=[Message.user("Capital of Japan?")])
        assert resp.message.content == "Tokyo."
        assert resp.usage["prompt_tokens"] == 10
        assert resp.usage["completion_tokens"] == 5
        assert resp.stop_reason == "stop"

    async def test_messages_converted_to_openai_shape(self):
        m = LiteLLMModel(model="oci/cohere.command-latest")
        captured: dict[str, Any] = {}

        async def _capture(**kwargs):
            captured.update(kwargs)
            return _mock_chat_response()

        with patch("litellm.acompletion", _capture):
            await m.complete(
                messages=[
                    Message.system("Be terse."),
                    Message.user("Hello"),
                ]
            )

        assert captured["messages"] == [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Hello"},
        ]

    async def test_temperature_dropped_for_gpt5(self):
        m = LiteLLMModel(model="oci/openai.gpt-5-mini")
        captured: dict[str, Any] = {}

        async def _capture(**kwargs):
            captured.update(kwargs)
            return _mock_chat_response()

        with patch("litellm.acompletion", _capture):
            await m.complete(messages=[Message.user("hi")])

        assert "temperature" not in captured, (
            "gpt-5 family rejects temperature; LiteLLMModel must not send it."
        )

    async def test_temperature_sent_for_normal_models(self):
        m = LiteLLMModel(model="oci/xai.grok-4.20", temperature=0.3)
        captured: dict[str, Any] = {}

        async def _capture(**kwargs):
            captured.update(kwargs)
            return _mock_chat_response()

        with patch("litellm.acompletion", _capture):
            await m.complete(messages=[Message.user("hi")])

        assert captured["temperature"] == 0.3

    async def test_reasoning_effort_passthrough(self):
        m = LiteLLMModel(model="oci/openai.gpt-5-mini", reasoning_effort="high")
        captured: dict[str, Any] = {}

        async def _capture(**kwargs):
            captured.update(kwargs)
            return _mock_chat_response(reasoning_tokens=128)

        with patch("litellm.acompletion", _capture):
            resp = await m.complete(messages=[Message.user("solve")])

        assert captured["reasoning_effort"] == "high"
        assert resp.usage["reasoning_tokens"] == 128

    async def test_tools_forwarded_and_choice_optional(self):
        m = LiteLLMModel(model="oci/xai.grok-4.20")
        captured: dict[str, Any] = {}

        async def _capture(**kwargs):
            captured.update(kwargs)
            return _mock_chat_response()

        tools = [{"type": "function", "function": {"name": "f"}}]
        with patch("litellm.acompletion", _capture):
            await m.complete(messages=[Message.user("hi")], tools=tools)
        assert captured["tools"] == tools
        assert "tool_choice" not in captured

        captured.clear()
        with patch("litellm.acompletion", _capture):
            await m.complete(messages=[Message.user("hi")], tools=tools, tool_choice="auto")
        assert captured["tool_choice"] == "auto"

    async def test_tool_call_parsed_from_response(self):
        tool_call = SimpleNamespace(
            id="call_abc",
            function=SimpleNamespace(name="get_weather", arguments='{"location":"Boston"}'),
        )
        mock_ac = AsyncMock(return_value=_mock_chat_response(content=None, tool_calls=[tool_call]))
        m = LiteLLMModel(model="oci/xai.grok-4.20")
        with patch("litellm.acompletion", mock_ac):
            resp = await m.complete(messages=[Message.user("weather?")])

        assert resp.message.content is None
        assert len(resp.message.tool_calls) == 1
        tc = resp.message.tool_calls[0]
        assert tc.id == "call_abc"
        assert tc.name == "get_weather"
        assert tc.arguments == {"location": "Boston"}

    async def test_tool_call_without_id_gets_synthetic_id(self):
        tool_call = SimpleNamespace(
            id=None,
            function=SimpleNamespace(name="f", arguments="{}"),
        )
        mock_ac = AsyncMock(return_value=_mock_chat_response(content=None, tool_calls=[tool_call]))
        m = LiteLLMModel(model="oci/google.gemini-2.5-flash")
        with patch("litellm.acompletion", mock_ac):
            resp = await m.complete(messages=[Message.user("hi")])

        assert resp.message.tool_calls[0].id == "call_0"

    async def test_max_tokens_per_call_override(self):
        m = LiteLLMModel(model="oci/cohere.command-latest", max_tokens=100)
        captured: dict[str, Any] = {}

        async def _capture(**kwargs):
            captured.update(kwargs)
            return _mock_chat_response()

        with patch("litellm.acompletion", _capture):
            await m.complete(messages=[Message.user("hi")], max_tokens=42)

        assert captured["max_tokens"] == 42


# ---------------------------------------------------------------------------
# stream() — mocked async iterator
# ---------------------------------------------------------------------------


def _stream_chunk(content: str | None = None, tool_call_delta=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_call_delta)
    choice = SimpleNamespace(delta=delta, index=0)
    return SimpleNamespace(choices=[choice])


class _AsyncIter:
    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


@pytest.mark.asyncio
class TestStream:
    async def test_streams_text_then_terminal(self):
        chunks = [
            _stream_chunk(content="Hi"),
            _stream_chunk(content=" "),
            _stream_chunk(content="world"),
        ]
        mock_ac = AsyncMock(return_value=_AsyncIter(chunks))
        m = LiteLLMModel(model="oci/cohere.command-latest")

        events = []
        with patch("litellm.acompletion", mock_ac):
            async for ev in m.stream(messages=[Message.user("greet")]):
                events.append(ev)

        text = "".join(ev.content or "" for ev in events if not ev.done)
        terminal = [ev for ev in events if ev.done]
        assert text == "Hi world"
        assert len(terminal) == 1
        assert terminal[0].tool_calls is None

    async def test_stream_assembles_partial_tool_calls(self):
        # Two delta chunks for one tool call, then terminal.
        td1 = [
            SimpleNamespace(
                index=0,
                id="call_z",
                function=SimpleNamespace(name="get_weather", arguments='{"loc'),
            )
        ]
        td2 = [
            SimpleNamespace(
                index=0,
                id=None,
                function=SimpleNamespace(name=None, arguments='ation":"NYC"}'),
            )
        ]
        chunks = [
            _stream_chunk(tool_call_delta=td1),
            _stream_chunk(tool_call_delta=td2),
        ]
        mock_ac = AsyncMock(return_value=_AsyncIter(chunks))
        m = LiteLLMModel(model="oci/xai.grok-4.20")

        events = []
        with patch("litellm.acompletion", mock_ac):
            async for ev in m.stream(
                messages=[Message.user("weather?")],
                tools=[{"type": "function", "function": {"name": "f"}}],
            ):
                events.append(ev)

        terminal = next(ev for ev in events if ev.done)
        assert terminal.tool_calls is not None
        assert len(terminal.tool_calls) == 1
        tc = terminal.tool_calls[0]
        assert tc.id == "call_z"
        assert tc.name == "get_weather"
        assert tc.arguments == {"location": "NYC"}

    async def test_empty_stream_still_emits_terminal(self):
        mock_ac = AsyncMock(return_value=_AsyncIter([]))
        m = LiteLLMModel(model="oci/cohere.command-latest")

        events = []
        with patch("litellm.acompletion", mock_ac):
            async for ev in m.stream(messages=[Message.user("ping")]):
                events.append(ev)

        assert len(events) == 1
        assert events[0].done is True


# ---------------------------------------------------------------------------
# Context-manager / close lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aenter_aexit_noop_close():
    """LiteLLMModel.close() is a no-op (LiteLLM owns the HTTP client lifecycle).
    The context-manager protocol must still work so callers can use ``async with``
    interchangeably with OpenAIModel / AnthropicModel.
    """
    async with LiteLLMModel(model="oci/cohere.command-latest") as m:
        assert isinstance(m, LiteLLMModel)
    # No exception means lifecycle is sound.


# ---------------------------------------------------------------------------
# Transport switching: OCI native vs OCI OpenAI-V1
# ---------------------------------------------------------------------------


class TestTransportSelection:
    """``transport=`` selects between LiteLLM's native ``oci/*`` provider
    (default) and Locus's existing ``OCIChatCompletionsModel`` for OCI's
    V1 OpenAI-compat shim. The dispatch must happen at construction time so
    a misconfigured V1 transport surfaces immediately."""

    def test_default_transport_is_native(self):
        m = LiteLLMModel(model="oci/xai.grok-4.20")
        assert m.config.transport == "oci_native"
        assert m._oci_backend is None
        # Native path computes completion kwargs eagerly.
        assert m._completion_kwargs["model"] == "oci/xai.grok-4.20"

    def test_v1_transport_strips_oci_prefix_for_backend(self, monkeypatch):
        captured: dict[str, Any] = {}

        # Patch OCIChatCompletionsModel to capture init kwargs without
        # touching OCI SDK or network.
        class _FakeV1Backend:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            async def complete(self, **_):  # pragma: no cover - not invoked here
                raise NotImplementedError

            async def stream(self, **_):  # pragma: no cover
                raise NotImplementedError

            async def close(self):
                pass

        import locus.models.providers.oci as oci_mod

        monkeypatch.setattr(oci_mod, "OCIChatCompletionsModel", _FakeV1Backend)

        m = LiteLLMModel(
            model="oci/openai.gpt-5-mini",  # native-style prefix
            transport="oci_openai_v1",
            oci_profile="DEFAULT",
            oci_region="us-chicago-1",
        )
        assert m._oci_backend is not None
        # The ``oci/`` prefix must be stripped — V1 shim wants the bare model id.
        assert captured["model"] == "openai.gpt-5-mini"
        assert captured["profile"] == "DEFAULT"
        assert captured["region"] == "us-chicago-1"
        # Native completion-kwargs cache is intentionally empty for V1.
        assert m._completion_kwargs == {}

    def test_v1_transport_accepts_bare_model_id(self, monkeypatch):
        captured: dict[str, Any] = {}

        class _FakeV1Backend:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            async def close(self):
                pass

        import locus.models.providers.oci as oci_mod

        monkeypatch.setattr(oci_mod, "OCIChatCompletionsModel", _FakeV1Backend)

        # Bare id (no ``oci/`` prefix) should pass straight through.
        LiteLLMModel(
            model="cohere.command-latest",
            transport="oci_openai_v1",
            oci_profile="DEFAULT",
        )
        assert captured["model"] == "cohere.command-latest"

    @pytest.mark.asyncio
    async def test_v1_transport_delegates_complete(self, monkeypatch):
        delegated: dict[str, Any] = {}

        class _FakeV1Backend:
            def __init__(self, **kwargs):
                pass

            async def complete(self, messages, tools=None, **kwargs):
                delegated["messages"] = messages
                delegated["tools"] = tools
                delegated["kwargs"] = kwargs
                return ModelResponse(
                    message=Message.assistant(content="delegated-ok"),
                    usage={"prompt_tokens": 1, "completion_tokens": 1},
                    stop_reason="stop",
                )

            async def close(self):
                pass

        import locus.models.providers.oci as oci_mod

        monkeypatch.setattr(oci_mod, "OCIChatCompletionsModel", _FakeV1Backend)

        m = LiteLLMModel(
            model="openai.gpt-5-mini",
            transport="oci_openai_v1",
            oci_profile="DEFAULT",
        )
        resp = await m.complete(messages=[Message.user("hi")])

        assert resp.message.content == "delegated-ok"
        assert delegated["messages"][0].content == "hi"

    @pytest.mark.asyncio
    async def test_v1_transport_close_propagates(self, monkeypatch):
        closed = {"flag": False}

        class _FakeV1Backend:
            def __init__(self, **kwargs):
                pass

            async def close(self):
                closed["flag"] = True

        import locus.models.providers.oci as oci_mod

        monkeypatch.setattr(oci_mod, "OCIChatCompletionsModel", _FakeV1Backend)

        async with LiteLLMModel(
            model="openai.gpt-5-mini",
            transport="oci_openai_v1",
            oci_profile="DEFAULT",
        ):
            pass

        assert closed["flag"] is True, "V1 backend close() must be called on aexit"

    def test_responses_transport_builds_backend(self, monkeypatch):
        """``transport='oci_responses'`` must construct an
        ``OCIResponsesModel`` (NOT an OCIChatCompletionsModel) and forward
        the Responses-specific fields (``store``, ``project_ocid``,
        ``request_timeout``)."""
        captured: dict[str, Any] = {}

        class _FakeResponsesBackend:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            async def close(self):
                pass

        import locus.models.providers.oci as oci_mod

        monkeypatch.setattr(oci_mod, "OCIResponsesModel", _FakeResponsesBackend)

        m = LiteLLMModel(
            model="oci/openai.gpt-5-mini",  # ``oci/`` prefix must be stripped
            transport="oci_responses",
            oci_profile="DEFAULT",
            oci_region="us-chicago-1",
            oci_responses_store=False,
            oci_project_ocid="ocid1.generativeaiproject.oc1..xxx",
            oci_responses_request_timeout=120.0,
            max_tokens=512,
        )
        assert m._oci_backend is not None
        # Bare model id (no ``oci/`` prefix) is what OCIResponsesModel wants.
        assert captured["model"] == "openai.gpt-5-mini"
        # Responses uses ``max_output_tokens`` (not ``max_tokens``).
        assert captured["max_output_tokens"] == 512
        assert captured["store"] is False
        assert captured["project_ocid"] == "ocid1.generativeaiproject.oc1..xxx"
        assert captured["request_timeout"] == 120.0
        assert captured["profile"] == "DEFAULT"
        assert captured["region"] == "us-chicago-1"

    @pytest.mark.asyncio
    async def test_responses_transport_delegates_complete(self, monkeypatch):
        delegated: dict[str, Any] = {}

        class _FakeResponsesBackend:
            def __init__(self, **kwargs):
                pass

            async def complete(self, messages, tools=None, **kwargs):
                delegated["messages"] = messages
                delegated["tools"] = tools
                return ModelResponse(
                    message=Message.assistant(content="from-responses"),
                    usage={"prompt_tokens": 2, "completion_tokens": 2},
                    stop_reason="stop",
                )

            async def close(self):
                pass

        import locus.models.providers.oci as oci_mod

        monkeypatch.setattr(oci_mod, "OCIResponsesModel", _FakeResponsesBackend)

        m = LiteLLMModel(
            model="openai.gpt-5-mini",
            transport="oci_responses",
            oci_profile="DEFAULT",
        )
        resp = await m.complete(messages=[Message.user("ping")])
        assert resp.message.content == "from-responses"
        assert delegated["messages"][0].content == "ping"

    def test_unknown_transport_raises(self):
        """Defending callers from typos — an unrecognised transport must
        fail at construction time, not at first ``complete()``."""
        with pytest.raises(Exception):  # noqa: B017 — pydantic ValidationError is intentionally broad
            LiteLLMModel(model="oci/cohere.command-latest", transport="oci_bogus")
