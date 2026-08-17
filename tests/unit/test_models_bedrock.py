# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Bedrock provider — no credentials, no network.

Every request is checked with ``botocore.stub.Stubber``, which validates the
call against botocore's own service model: a malformed ``toolConfig`` or a
misnamed parameter fails here exactly as it would against the real service,
without an AWS account.

The conversion tests carry the weight. Bedrock's Converse API is stricter than
the OpenAI wire format in ways that are invisible until a specific model
rejects a specific shape — and the strictness is not uniform across models on
the service, so "it worked against Nova" proves very little. The mixed-content
rule below was found exactly that way: it passed on Amazon's models and 400'd
on Meta's.
"""

from __future__ import annotations

import json

import pytest

from tulip.core.messages import Message, Role, ToolCall, ToolResult
from tulip.models.registry import get_model, list_providers


boto3 = pytest.importorskip("boto3", reason="bedrock extra not installed")
from botocore.stub import ANY, Stubber  # noqa: E402 — after importorskip

from tulip.models.native.bedrock import BedrockModel, _loads_or_empty  # noqa: E402


MODEL_ID = "us.amazon.nova-micro-v1:0"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up an order",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    }
]


def _model(**kw) -> BedrockModel:
    """A model with fake credentials — never contacts AWS; the Stubber intercepts."""
    kw.setdefault("region", "us-east-1")
    return BedrockModel(
        model=MODEL_ID,
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",  # noqa: S106 — fake creds; Stubber intercepts
        **kw,
    )


def _any_request() -> dict:
    """Stubber wants a dict of expected params; ANY is only valid per-key."""
    return {"modelId": ANY, "messages": ANY, "inferenceConfig": ANY}


def _text_response(text: str = "hello") -> dict:
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 10, "outputTokens": 4, "totalTokens": 14},
        "metrics": {"latencyMs": 42},
    }


# --------------------------------------------------------------- registration


def test_bedrock_is_registered() -> None:
    assert "bedrock" in list_providers()


def test_registry_builds_a_bedrock_model() -> None:
    model = get_model(f"bedrock:{MODEL_ID}", region="us-east-1")
    assert isinstance(model, BedrockModel)
    assert model.config.model == MODEL_ID


# ------------------------------------------------------------------ messages


def test_system_prompt_is_lifted_out_of_the_message_list() -> None:
    """Converse takes ``system`` as a separate argument, not as a message."""
    system, messages = _model()._convert_messages(
        [Message.system("You are terse."), Message.user("hi")]
    )
    assert system == [{"text": "You are terse."}]
    assert messages == [{"role": "user", "content": [{"text": "hi"}]}]


def test_tool_result_becomes_a_user_turn() -> None:
    _, messages = _model()._convert_messages(
        [
            Message.tool(
                ToolResult(
                    tool_call_id="tu-1", name="lookup_order", content="Order ord-1: a widget"
                )
            )
        ]
    )
    assert messages == [
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "tu-1",
                        "content": [{"text": "Order ord-1: a widget"}],
                    }
                }
            ],
        }
    ]


def test_assistant_text_and_tool_use_are_never_mixed() -> None:
    """The regression that motivated this file.

    Nova and Claude accept an assistant turn holding both a text block and a
    toolUse block. Llama and Mistral reject the whole request:

        ValidationException: messages.N.content: Conversation blocks and tool
        use blocks cannot be provided in the same turn.

    Found live against ``us.meta.llama3-3-70b-instruct-v1:0`` after the same
    conversation worked against Nova.
    """
    _, messages = _model()._convert_messages(
        [
            Message.assistant(
                content="Let me look that up.",
                tool_calls=[ToolCall(id="tu-1", name="lookup_order", arguments={"order_id": "o1"})],
            )
        ]
    )
    [turn] = messages
    kinds = {key for block in turn["content"] for key in block}
    assert kinds == {"toolUse"}, f"text leaked into a tool-use turn: {turn['content']}"


def test_assistant_text_survives_when_there_are_no_tool_calls() -> None:
    _, messages = _model()._convert_messages([Message.assistant(content="Just talking.")])
    assert messages == [{"role": "assistant", "content": [{"text": "Just talking."}]}]


def test_an_empty_assistant_turn_is_dropped() -> None:
    """Converse 400s on an empty content list; the OpenAI format tolerates it."""
    _, messages = _model()._convert_messages([Message.assistant(content=None)])
    assert messages == []


# --------------------------------------------------------------------- tools


def test_tools_become_a_tool_config() -> None:
    config = _model()._convert_tools(TOOLS)
    assert config == {
        "tools": [
            {
                "toolSpec": {
                    "name": "lookup_order",
                    "description": "Look up an order",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {"order_id": {"type": "string"}},
                            "required": ["order_id"],
                        }
                    },
                }
            }
        ]
    }


def test_no_tools_means_no_tool_config() -> None:
    assert _model()._convert_tools(None) is None
    assert _model()._convert_tools([]) is None


def test_guardrail_config_is_sent_when_configured() -> None:
    params = _model(guardrail_id="gr-1", guardrail_version="2")._params([Message.user("hi")], None)
    assert params["guardrailConfig"] == {"guardrailIdentifier": "gr-1", "guardrailVersion": "2"}


def test_guardrail_defaults_to_draft() -> None:
    params = _model(guardrail_id="gr-1")._params([Message.user("hi")], None)
    assert params["guardrailConfig"]["guardrailVersion"] == "DRAFT"


# ------------------------------------------------------------------ complete


async def test_complete_returns_text_and_usage() -> None:
    model = _model()
    with Stubber(model.client) as stub:
        stub.add_response(
            "converse",
            _text_response("hello there"),
            {
                "modelId": MODEL_ID,
                "messages": [{"role": "user", "content": [{"text": "hi"}]}],
                "inferenceConfig": ANY,
            },
        )
        response = await model.complete([Message.user("hi")])

    assert response.message.content == "hello there"
    assert response.message.role == Role.ASSISTANT
    assert response.usage == {"prompt_tokens": 10, "completion_tokens": 4}
    assert response.stop_reason == "end_turn"


async def test_complete_parses_tool_use() -> None:
    model = _model()
    with Stubber(model.client) as stub:
        stub.add_response(
            "converse",
            {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "tu-9",
                                    "name": "lookup_order",
                                    "input": {"order_id": "ord-4821"},
                                }
                            }
                        ],
                    }
                },
                "stopReason": "tool_use",
                "usage": {"inputTokens": 20, "outputTokens": 8, "totalTokens": 28},
                "metrics": {"latencyMs": 50},
            },
            {"modelId": MODEL_ID, "messages": ANY, "inferenceConfig": ANY, "toolConfig": ANY},
        )
        response = await model.complete([Message.user("look up ord-4821")], tools=TOOLS)

    [call] = response.message.tool_calls
    assert (call.id, call.name, call.arguments) == (
        "tu-9",
        "lookup_order",
        {"order_id": "ord-4821"},
    )
    assert response.stop_reason == "tool_use"


async def test_cache_token_counts_are_surfaced_when_present() -> None:
    model = _model()
    payload = _text_response()
    payload["usage"] |= {"cacheReadInputTokens": 7, "cacheWriteInputTokens": 3}
    with Stubber(model.client) as stub:
        stub.add_response("converse", payload, _any_request())
        response = await model.complete([Message.user("hi")])
    assert response.usage["cache_read_input_tokens"] == 7
    assert response.usage["cache_creation_input_tokens"] == 3


@pytest.mark.parametrize("reason", ["guardrail_intervened", "content_filtered", "max_tokens"])
async def test_stop_reasons_pass_through_unflattened(reason: str) -> None:
    """A guardrail cutting a run short is not the same event as a model finishing."""
    model = _model()
    payload = _text_response()
    payload["stopReason"] = reason
    with Stubber(model.client) as stub:
        stub.add_response("converse", payload, _any_request())
        response = await model.complete([Message.user("hi")])
    assert response.stop_reason == reason


# -------------------------------------------------------------------- stream


def _stream_events() -> list[dict]:
    """A tool call streamed the way Converse actually sends one."""
    return [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Look"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "ing"}, "contentBlockIndex": 0}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {
            "contentBlockStart": {
                "start": {"toolUse": {"toolUseId": "tu-1", "name": "lookup_order"}},
                "contentBlockIndex": 1,
            }
        },
        # Arguments arrive as partial JSON, split anywhere — including
        # mid-token, which is why they cannot be parsed until the block closes.
        {
            "contentBlockDelta": {
                "delta": {"toolUse": {"input": '{"order_'}},
                "contentBlockIndex": 1,
            }
        },
        {
            "contentBlockDelta": {
                "delta": {"toolUse": {"input": 'id": "ord'}},
                "contentBlockIndex": 1,
            }
        },
        {"contentBlockDelta": {"delta": {"toolUse": {"input": '-4821"}'}}, "contentBlockIndex": 1}},
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"messageStop": {"stopReason": "tool_use"}},
        {"metadata": {"usage": {"inputTokens": 12, "outputTokens": 6, "totalTokens": 18}}},
    ]


async def test_stream_yields_text_then_the_assembled_tool_call(monkeypatch) -> None:
    model = _model()
    monkeypatch.setattr(
        type(model.client),
        "converse_stream",
        lambda self, **kw: {"stream": iter(_stream_events())},
        raising=False,
    )

    text, tool_calls, final = "", [], None
    async for event in model.stream([Message.user("look up ord-4821")], tools=TOOLS):
        if event.content:
            text += event.content
        if event.tool_calls:
            tool_calls.extend(event.tool_calls)
        if event.done:
            final = event

    assert text == "Looking"
    [call] = tool_calls
    assert call.name == "lookup_order"
    # The whole point: three partial JSON fragments reassembled into arguments.
    assert call.arguments == {"order_id": "ord-4821"}
    assert final is not None
    assert final.stop_reason == "tool_use"
    assert final.usage == {"prompt_tokens": 12, "completion_tokens": 6}


async def test_stream_propagates_an_error_from_the_reader_thread(monkeypatch) -> None:
    """A failure inside the boto3 iterator must surface, not hang the consumer."""

    def explode(self, **kw):
        raise RuntimeError("bedrock exploded")

    model = _model()
    monkeypatch.setattr(type(model.client), "converse_stream", explode, raising=False)

    with pytest.raises(RuntimeError, match="bedrock exploded"):
        async for _ in model.stream([Message.user("hi")]):
            pass


# ------------------------------------------------------------------ plumbing


def test_client_is_built_once() -> None:
    model = _model()
    assert model.client is model.client


def test_explicit_region_reaches_the_client() -> None:
    assert _model(region="eu-west-1").client.meta.region_name == "eu-west-1"


def test_endpoint_override_is_honoured() -> None:
    """VPC endpoints and local mocks both go through endpoint_url."""
    model = _model(endpoint_url="https://bedrock.internal")
    assert model.client.meta.endpoint_url == "https://bedrock.internal"


@pytest.mark.parametrize(
    ("buffer", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ("", {}),  # a no-argument tool streams zero input deltas
        ("   ", {}),
        ("not json", {}),
        ('"a string"', {}),  # valid JSON, wrong shape
    ],
)
def test_tool_argument_parsing_is_total(buffer: str, expected: dict) -> None:
    assert _loads_or_empty(buffer) == expected


def test_retry_mode_is_adaptive() -> None:
    """Throttling is the first failure every Bedrock account meets."""
    config = _model().client.meta.config
    assert config.retries["mode"] == "adaptive"


def test_json_round_trip_of_streamed_arguments() -> None:
    """Guards the fragment-reassembly assumption itself."""
    payload = {"order_id": "ord-4821", "note": "split, anywhere"}
    blob = json.dumps(payload)
    fragments = [blob[i : i + 3] for i in range(0, len(blob), 3)]
    assert _loads_or_empty("".join(fragments)) == payload
