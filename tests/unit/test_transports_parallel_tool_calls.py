# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Per-transport guard that N>1 wire-format tool_calls are normalized
into ``ModelResponse.message.tool_calls`` of length N.

Companion to ``tests/integration/test_concurrent_tools_models.py``: the
live matrix proves end-to-end parallelism on transports where the LLM
chose to fan out (and OCI's server accepted the fan-out). Where the LLM
declined — or OCI's server rejected — this file deterministically pins
the transport's normalization independently of live model behavior, so
a regression in any transport's ``parse_response`` is caught regardless
of model whims.

Covers:

* ``OpenAIModel._parse_response`` (also covers ``OCIChatCompletionsModel`` via
  inheritance — the OpenAI-compat wire is shared)
* ``GenericProvider.parse_response`` (OCI native SDK for
  OpenAI/Meta/xAI/Mistral/Google)
* ``CohereProvider.parse_response`` (OCI native SDK for Cohere R+)
* ``locus.models.providers.oci._responses_parse.parse_response``
  (``OCIResponsesModel`` wire)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# OpenAIModel / OCIChatCompletionsModel — OpenAI-compat wire
# =============================================================================


class _Func:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _ToolCallStub:
    def __init__(self, *, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.function = _Func(name=name, arguments=arguments)


class _MsgStub:
    def __init__(self, *, content: str | None, tool_calls: list[_ToolCallStub]) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, *, message: _MsgStub, finish_reason: str = "tool_calls") -> None:
        self.message = message
        self.finish_reason = finish_reason


class _Response:
    def __init__(self, *, choices: list[_Choice]) -> None:
        self.choices = choices
        self.usage = None


def test_openai_parse_response_preserves_n_parallel_tool_calls() -> None:
    """OpenAI wire returns 3 ``tool_calls``; normalised list must have length 3
    with ids/names/args intact."""
    from locus.models.native.openai import OpenAIModel

    m = OpenAIModel()
    resp = _Response(
        choices=[
            _Choice(
                message=_MsgStub(
                    content=None,
                    tool_calls=[
                        _ToolCallStub(call_id="c0", name="lookup", arguments='{"topic": "a"}'),
                        _ToolCallStub(call_id="c1", name="lookup", arguments='{"topic": "b"}'),
                        _ToolCallStub(call_id="c2", name="lookup", arguments='{"topic": "c"}'),
                    ],
                )
            )
        ]
    )

    out = m._parse_response(resp)

    assert len(out.message.tool_calls) == 3
    assert [tc.id for tc in out.message.tool_calls] == ["c0", "c1", "c2"]
    assert {tc.name for tc in out.message.tool_calls} == {"lookup"}
    assert [tc.arguments for tc in out.message.tool_calls] == [
        {"topic": "a"},
        {"topic": "b"},
        {"topic": "c"},
    ]


# =============================================================================
# GenericProvider (OCI native SDK for OpenAI/Meta/xAI/Mistral/Google)
# =============================================================================


@pytest.fixture
def generic_provider() -> Any:
    models = MagicMock()
    models.BaseChatRequest = MagicMock()
    models.BaseChatRequest.API_FORMAT_GENERIC = "GENERIC"
    with (
        patch.dict(
            "sys.modules",
            {"oci": MagicMock(), "oci.generative_ai_inference": MagicMock()},
        ),
        patch("oci.generative_ai_inference.models", models),
    ):
        from locus.models.providers.oci.models.generic import GenericProvider

        return GenericProvider()


def test_generic_parse_response_preserves_n_parallel_tool_calls(generic_provider: Any) -> None:
    """OCI native SDK (GenericProvider) parses N wire-format tool_calls back
    into a length-N ToolCall list."""
    raw_calls = []
    for i, topic in enumerate(["a", "b", "c"]):
        tc = MagicMock()
        tc.id = f"c{i}"
        tc.name = "lookup"
        tc.arguments = json.dumps({"topic": topic})
        raw_calls.append(tc)

    message = MagicMock()
    message.content = []  # no text parts
    message.tool_calls = raw_calls

    chat_response = MagicMock()
    chat_response.choices = [MagicMock(message=message, finish_reason="tool_calls")]

    response = MagicMock()
    response.data.chat_response = chat_response

    content, tool_calls, stop_reason = generic_provider.parse_response(response)

    assert content is None
    assert len(tool_calls) == 3
    assert [tc.id for tc in tool_calls] == ["c0", "c1", "c2"]
    assert {tc.name for tc in tool_calls} == {"lookup"}
    assert [tc.arguments for tc in tool_calls] == [
        {"topic": "a"},
        {"topic": "b"},
        {"topic": "c"},
    ]


# =============================================================================
# CohereProvider (OCI native SDK for Cohere R-series)
# =============================================================================


@pytest.fixture
def cohere_provider() -> Any:
    models = MagicMock()
    models.BaseChatRequest = MagicMock()
    models.BaseChatRequest.API_FORMAT_COHERE = "COHERE"
    with (
        patch.dict(
            "sys.modules",
            {"oci": MagicMock(), "oci.generative_ai_inference": MagicMock()},
        ),
        patch("oci.generative_ai_inference.models", models),
    ):
        from locus.models.providers.oci.models.cohere import CohereProvider

        return CohereProvider()


def test_cohere_parse_response_preserves_n_parallel_tool_calls(cohere_provider: Any) -> None:
    """CohereProvider parses N wire-format tool_calls into a length-N ToolCall list.

    Cohere R+ on OCI is the canonical native-SDK Cohere transport. Even though
    Cohere historically emits one call per turn more often than OpenAI-style
    models, the wire spec allows N>1 and the normalisation must support it.
    """
    raw_calls = [
        MagicMock(name="lookup", parameters={"topic": "a"}),
        MagicMock(name="lookup", parameters={"topic": "b"}),
        MagicMock(name="lookup", parameters={"topic": "c"}),
    ]
    # MagicMock(name=...) sets the *mock identity* name, not the .name
    # attribute Cohere's parser reads — set them explicitly.
    for i, tc in enumerate(raw_calls):
        tc.name = "lookup"
        tc.parameters = {"topic": ["a", "b", "c"][i]}

    response = MagicMock()
    response.data.chat_response.text = None
    response.data.chat_response.finish_reason = "TOOL_CALL"
    response.data.chat_response.tool_calls = raw_calls

    content, tool_calls, stop_reason = cohere_provider.parse_response(response)

    assert content is None
    assert len(tool_calls) == 3
    assert {tc.name for tc in tool_calls} == {"lookup"}
    assert [tc.arguments for tc in tool_calls] == [
        {"topic": "a"},
        {"topic": "b"},
        {"topic": "c"},
    ]


# =============================================================================
# OCIResponsesModel — Responses API wire (free-function parser)
# =============================================================================


def test_responses_parse_response_preserves_n_parallel_tool_calls() -> None:
    """Responses-API parse_response turns 3 ``function_call`` output items
    into a length-3 ToolCall list."""
    from locus.models.providers.oci._responses_parse import parse_response

    payload = {
        "id": "resp_abc",
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "call_id": "c0",
                "name": "lookup",
                "arguments": json.dumps({"topic": "a"}),
            },
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "lookup",
                "arguments": json.dumps({"topic": "b"}),
            },
            {
                "type": "function_call",
                "call_id": "c2",
                "name": "lookup",
                "arguments": json.dumps({"topic": "c"}),
            },
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }

    msg, _usage, stop_reason, provider_state = parse_response(payload)

    assert msg.content is None
    assert len(msg.tool_calls) == 3
    assert [tc.id for tc in msg.tool_calls] == ["c0", "c1", "c2"]
    assert {tc.name for tc in msg.tool_calls} == {"lookup"}
    assert [tc.arguments for tc in msg.tool_calls] == [
        {"topic": "a"},
        {"topic": "b"},
        {"topic": "c"},
    ]
    assert stop_reason == "completed"
    assert provider_state == {"previous_response_id": "resp_abc"}
