# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Coverage tests for ``locus.models.providers.oci.base.OCIModelProvider``.

The base class is abstract, but the default property values plus
``parse_usage`` and ``parse_stream_chunk`` are concrete and need
exercising via a minimal subclass. The Cohere / Generic implementations
override ``parse_usage`` so the base default never executed under their
tests.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from locus.core.messages import Message, ToolCall
from locus.models.providers.oci.base import OCIModelProvider


class _MinimalProvider(OCIModelProvider):
    """Concrete provider that just uses every default in the base class."""

    @property
    def api_format(self) -> str:
        return "minimal"

    def build_request(self, messages: Any, tools: Any = None, **_: Any) -> Any:
        return messages

    def parse_response(self, response: Any) -> tuple[str | None, list[ToolCall], str | None]:
        return None, [], None

    def convert_messages(
        self, messages: list[Message], model_id: str | None = None
    ) -> list[dict[str, Any]]:
        return [{"role": m.role.value, "content": m.content} for m in messages]

    def convert_tools(self, tools: list[dict[str, Any]] | None) -> list[Any] | None:
        return tools


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_stop_sequence_key_default(self) -> None:
        assert _MinimalProvider().stop_sequence_key == "stop"

    def test_supports_tools_default(self) -> None:
        assert _MinimalProvider().supports_tools is True

    def test_supports_streaming_default(self) -> None:
        assert _MinimalProvider().supports_streaming is True


# ---------------------------------------------------------------------------
# parse_usage
# ---------------------------------------------------------------------------


class TestParseUsage:
    def test_extracts_token_counts(self) -> None:
        response = SimpleNamespace(
            data=SimpleNamespace(
                chat_response=SimpleNamespace(
                    usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
                )
            )
        )
        out = _MinimalProvider().parse_usage(response)
        assert out == {"prompt_tokens": 11, "completion_tokens": 7}

    def test_treats_none_token_counts_as_zero(self) -> None:
        response = SimpleNamespace(
            data=SimpleNamespace(
                chat_response=SimpleNamespace(
                    usage=SimpleNamespace(prompt_tokens=None, completion_tokens=None),
                )
            )
        )
        out = _MinimalProvider().parse_usage(response)
        assert out == {"prompt_tokens": 0, "completion_tokens": 0}

    def test_no_usage_returns_empty_dict(self) -> None:
        response = SimpleNamespace(data=SimpleNamespace(chat_response=SimpleNamespace(usage=None)))
        out = _MinimalProvider().parse_usage(response)
        assert out == {}

    def test_chat_response_without_usage_attr(self) -> None:
        response = SimpleNamespace(data=SimpleNamespace(chat_response=SimpleNamespace()))
        out = _MinimalProvider().parse_usage(response)
        assert out == {}


# ---------------------------------------------------------------------------
# parse_stream_chunk default
# ---------------------------------------------------------------------------


class TestParseStreamChunkDefault:
    def test_finish_reason_marks_done(self) -> None:
        content, calls, done = _MinimalProvider().parse_stream_chunk({"finishReason": "stop"})
        assert content == ""
        assert calls == []
        assert done is True

    def test_no_finish_reason_marks_not_done(self) -> None:
        content, calls, done = _MinimalProvider().parse_stream_chunk({"text": "hi"})
        assert content == ""
        assert calls == []
        assert done is False


# ---------------------------------------------------------------------------
# api_format must be implemented
# ---------------------------------------------------------------------------


class TestAbstractContract:
    def test_cannot_instantiate_base(self) -> None:
        with pytest.raises(TypeError):
            OCIModelProvider()  # type: ignore[abstract]
