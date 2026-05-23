# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Coverage tests for ``locus.models.providers.oci.OCIModel``.

Existing OCI tests cover the inner provider implementations (Cohere /
Generic) and the OpenAI-compat path. This file covers the ``OCIModel``
glue: provider selection, ``client`` lazy init, ``complete()`` retry
behavior, and the ``stream()`` SSE iteration + non-stream fallback.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


pytest.importorskip("oci")

from locus.core.messages import Message  # noqa: E402
from locus.models.providers.oci import (  # noqa: E402
    CohereProvider,
    GenericProvider,
    OCIAuthType,
    OCIClient,
    OCIClientConfig,
    OCIModel,
)


# ---------------------------------------------------------------------------
# Capability + constructor
# ---------------------------------------------------------------------------


class TestOCIModelBasics:
    def test_supports_structured_output_is_false(self) -> None:
        m = OCIModel(model_id="meta.llama-3.3-70b-instruct")
        assert m.supports_structured_output is False

    def test_string_auth_type_coerced(self) -> None:
        m = OCIModel(model_id="cohere.command-r-plus", auth_type="api_key")
        assert m.config.auth_type == OCIAuthType.API_KEY


class TestOCIModelRegionAndProfileAlias:
    """``region=`` builds the inference endpoint; ``profile=`` is an
    alias for ``profile_name``. Both exist so users don't have to type
    out the full ``https://inference.generativeai.<region>.oci.oraclecloud.com``
    URL just to target a cross-region model, and so ``profile=`` (the
    natural name) stops being silently swallowed into ``**kwargs``."""

    def test_region_builds_inference_endpoint(self) -> None:
        m = OCIModel(model_id="openai.gpt-oss-20b", region="us-chicago-1")
        assert m.config.service_endpoint == (
            "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
        )

    def test_explicit_endpoint_wins_over_region(self) -> None:
        explicit = "https://example.invalid/v1"
        m = OCIModel(
            model_id="openai.gpt-oss-20b",
            region="us-chicago-1",
            service_endpoint=explicit,
        )
        assert m.config.service_endpoint == explicit

    def test_no_region_no_endpoint_means_endpoint_stays_none(self) -> None:
        m = OCIModel(model_id="cohere.command-r-plus")
        assert m.config.service_endpoint is None

    def test_profile_alias_sets_profile_name(self) -> None:
        m = OCIModel(model_id="cohere.command-r-plus", profile="MY_SESSION_PROFILE")
        assert m.config.profile_name == "MY_SESSION_PROFILE"

    def test_profile_and_profile_name_same_value_is_fine(self) -> None:
        m = OCIModel(
            model_id="cohere.command-r-plus",
            profile="MY",
            profile_name="MY",
        )
        assert m.config.profile_name == "MY"

    def test_profile_and_profile_name_conflict_raises(self) -> None:
        with pytest.raises(ValueError, match=r"profile.*profile_name"):
            OCIModel(
                model_id="cohere.command-r-plus",
                profile="ONE",
                profile_name="TWO",
            )


# ---------------------------------------------------------------------------
# _get_provider dispatch
# ---------------------------------------------------------------------------


class TestProviderSelection:
    def test_cohere_r_picks_cohere_provider(self) -> None:
        m = OCIModel(model_id="cohere.command-r-plus")
        assert isinstance(m.provider, CohereProvider)

    def test_cohere_r_caps_picks_cohere(self) -> None:
        m = OCIModel(model_id="COHERE.COMMAND-R-PLUS")
        assert isinstance(m.provider, CohereProvider)

    def test_cohere_a_picks_generic_provider(self) -> None:
        m = OCIModel(model_id="cohere.command-a-03-2025")
        assert isinstance(m.provider, GenericProvider)

    def test_meta_picks_generic(self) -> None:
        m = OCIModel(model_id="meta.llama-3.3-70b-instruct")
        assert isinstance(m.provider, GenericProvider)

    def test_provider_is_cached(self) -> None:
        m = OCIModel(model_id="meta.llama-3.3-70b-instruct")
        first = m.provider
        second = m.provider
        assert first is second


# ---------------------------------------------------------------------------
# client lazy init
# ---------------------------------------------------------------------------


class TestClientLazyInit:
    def test_client_constructs_oci_client(self) -> None:
        m = OCIModel(
            model_id="meta.llama-3.3-70b-instruct",
            compartment_id="ocid1.compartment.oc1..x",
            profile_name="MY",
        )
        with patch.object(OCIClient, "__init__", return_value=None) as mock_init:
            c = m.client
            mock_init.assert_called_once()
            cfg_arg = mock_init.call_args.args[0]
            assert isinstance(cfg_arg, OCIClientConfig)
            assert cfg_arg.profile_name == "MY"
            # Returned object is an OCIClient (uninitialised, but isinstance check passes).
            assert isinstance(c, OCIClient)

    def test_client_cached(self) -> None:
        m = OCIModel(model_id="meta.llama-3.3-70b-instruct")
        with patch.object(OCIClient, "__init__", return_value=None):
            first = m.client
            second = m.client
        assert first is second


# ---------------------------------------------------------------------------
# complete() — full path with mocked SDK + retry
# ---------------------------------------------------------------------------


def _make_oci_response(
    *,
    content: str | None = "hello",
    tool_calls: list[Any] | None = None,
    stop_reason: str = "stop",
) -> SimpleNamespace:
    """Build an OCI-shaped response object (data.chat_response.usage etc.)."""
    return SimpleNamespace(
        data=SimpleNamespace(
            chat_response=SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
                content=content,
                tool_calls=tool_calls or [],
                finish_reason=stop_reason,
            )
        )
    )


def _patch_client_and_provider(model: OCIModel, response: Any) -> tuple[Any, Any]:
    """Install a MagicMock OCIClient + provider on ``model``."""
    client = MagicMock()
    client.compartment_id = "ocid1.compartment.oc1..x"
    client.get_serving_mode = MagicMock(return_value=SimpleNamespace(name="OnDemand"))
    client.chat = MagicMock(return_value=response)
    model._client = client

    provider = MagicMock()
    provider.convert_messages.return_value = [{"role": "user", "content": "hi"}]
    provider.convert_tools.return_value = None
    provider.build_request.return_value = SimpleNamespace()
    provider.parse_response.return_value = ("hello", [], "stop")
    provider.parse_usage.return_value = {"prompt_tokens": 10, "completion_tokens": 5}
    provider.parse_stream_chunk.return_value = ("", [], False)
    model._provider = provider

    return client, provider


class TestComplete:
    @pytest.mark.asyncio
    async def test_returns_first_attempt_when_content_present(self) -> None:
        m = OCIModel(model_id="meta.llama-3.3-70b-instruct")
        _patch_client_and_provider(m, _make_oci_response())
        result = await m.complete([Message.user("hi")])
        assert result.message.content == "hello"
        assert result.usage["prompt_tokens"] == 10

    @pytest.mark.asyncio
    async def test_retries_on_empty_response(self) -> None:
        m = OCIModel(model_id="meta.llama-3.3-70b-instruct")
        client, provider = _patch_client_and_provider(m, _make_oci_response())

        # First two parse_response calls return empty; third returns content.
        provider.parse_response.side_effect = [
            (None, [], "stop"),
            (None, [], "stop"),
            ("recovered", [], "stop"),
        ]
        result = await m.complete([Message.user("hi")])
        assert result.message.content == "recovered"
        assert client.chat.call_count == 3

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries(self) -> None:
        m = OCIModel(model_id="meta.llama-3.3-70b-instruct")
        client, provider = _patch_client_and_provider(m, _make_oci_response())
        provider.parse_response.return_value = (None, [], "stop")
        provider.parse_usage.return_value = {}
        # Speed up the test by removing the asyncio.sleep cost.
        with patch.object(asyncio, "sleep", AsyncMock(return_value=None)):
            result = await m.complete([Message.user("hi")])
        assert result.message.content is None
        assert client.chat.call_count == 3

    @pytest.mark.asyncio
    async def test_cohere_dict_messages_merged_into_request(self) -> None:
        m = OCIModel(model_id="cohere.command-r-plus")
        _patch_client_and_provider(m, _make_oci_response())
        # Override convert_messages to return a dict (Cohere shape).
        m._provider.convert_messages.return_value = {
            "message": "hi",
            "chat_history": [],
        }
        await m.complete([Message.user("hi")])
        # build_request should receive an empty message list (the dict
        # gets merged into kwargs instead).
        first_call = m._provider.build_request.call_args
        assert first_call.args[0] == []
        # And the dict's keys ended up as kwargs.
        assert "message" in first_call.kwargs
        assert "chat_history" in first_call.kwargs


# ---------------------------------------------------------------------------
# stream() — happy path + fallback
# ---------------------------------------------------------------------------


class _FakeEventsIter:
    """Iterator that yields fake SSE events."""

    def __init__(self, events: list[Any]) -> None:
        self._events = events
        self._idx = 0

    def __iter__(self) -> _FakeEventsIter:
        return self

    def __next__(self) -> Any:
        if self._idx >= len(self._events):
            raise StopIteration
        ev = self._events[self._idx]
        self._idx += 1
        return ev


class TestStream:
    @pytest.mark.asyncio
    async def test_streams_chunks_then_done(self) -> None:
        m = OCIModel(model_id="meta.llama-3.3-70b-instruct")
        client, provider = _patch_client_and_provider(m, _make_oci_response())

        # Build an SDK-shape streaming response: data.events() returns an
        # iterator of objects with ``data`` strings (JSON deltas).
        events = [
            SimpleNamespace(data='{"text":"foo"}'),
            SimpleNamespace(data='{"text":"bar"}'),
            SimpleNamespace(data="not-json"),  # malformed → skipped
        ]
        stream_resp = SimpleNamespace(data=SimpleNamespace(events=lambda: _FakeEventsIter(events)))
        client.chat = MagicMock(return_value=stream_resp)

        # Provider parses each delta into (content, tool_calls, is_done)
        provider.parse_stream_chunk.side_effect = [
            ("foo", [], False),
            ("bar", [], False),
        ]

        chunks = []
        async for ev in m.stream([Message.user("hi")]):
            chunks.append(ev)
        # foo + bar + done
        assert len(chunks) == 3
        assert chunks[0].content == "foo"
        assert chunks[1].content == "bar"
        assert chunks[2].done is True

    @pytest.mark.asyncio
    async def test_stream_sets_is_stream_on_request_object(self) -> None:
        m = OCIModel(model_id="meta.llama-3.3-70b-instruct")
        client, provider = _patch_client_and_provider(m, _make_oci_response())
        # build_request returns an object that already has is_stream=False
        # so we hit the ``hasattr → set is_stream=True`` branch.
        chat_request = SimpleNamespace(is_stream=False)
        provider.build_request.return_value = chat_request

        events: list[Any] = []
        stream_resp = SimpleNamespace(data=SimpleNamespace(events=lambda: _FakeEventsIter(events)))
        client.chat = MagicMock(return_value=stream_resp)
        provider.parse_stream_chunk.return_value = ("", [], False)

        async for _ in m.stream([Message.user("hi")]):
            pass
        # After the call, the request's is_stream flag must be True.
        assert chat_request.is_stream is True

    @pytest.mark.asyncio
    async def test_stream_falls_back_to_complete_on_chat_failure(self) -> None:
        m = OCIModel(model_id="meta.llama-3.3-70b-instruct")
        _patch_client_and_provider(m, _make_oci_response())
        # First chat raises (the stream attempt) — fallback to complete.
        m._client.chat = MagicMock(
            side_effect=[RuntimeError("DAC rejected is_stream"), _make_oci_response()]
        )
        # Provider's parse_response returns content + a fake tool call so we
        # exercise both yield branches in the fallback.
        from locus.core.messages import ToolCall

        m._provider.parse_response.return_value = (
            "fallback content",
            [ToolCall(id="t1", name="x", arguments={})],
            "stop",
        )
        events = []
        async for ev in m.stream([Message.user("hi")]):
            events.append(ev)
        # Expect: content, tool_calls, done
        assert len(events) == 3
        assert events[0].content == "fallback content"
        assert events[1].tool_calls
        assert events[1].tool_calls[0].name == "x"
        assert events[2].done is True

    @pytest.mark.asyncio
    async def test_stream_skips_event_without_data_attr(self) -> None:
        m = OCIModel(model_id="meta.llama-3.3-70b-instruct")
        client, provider = _patch_client_and_provider(m, _make_oci_response())
        events = [SimpleNamespace(data=None), SimpleNamespace(data='{"x":1}')]
        stream_resp = SimpleNamespace(data=SimpleNamespace(events=lambda: _FakeEventsIter(events)))
        client.chat = MagicMock(return_value=stream_resp)
        # Only the JSON event should reach parse_stream_chunk.
        provider.parse_stream_chunk.return_value = ("hi", [], False)

        out = []
        async for ev in m.stream([Message.user("hi")]):
            out.append(ev)
        assert out[0].content == "hi"
        assert out[-1].done is True
        # parse_stream_chunk only called for the valid JSON event.
        assert provider.parse_stream_chunk.call_count == 1
