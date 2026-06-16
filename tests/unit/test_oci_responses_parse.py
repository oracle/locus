# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for the OCI Responses request/response translation layer.

Pure-function tests against the parsers in
:mod:`locus.models.providers.oci._responses_parse`. No network or
SDK imports — these run identically in any environment.
"""

from __future__ import annotations

import json

from locus.core.messages import Message, ToolResult
from locus.models.providers.oci._responses_parse import (
    build_request_body,
    parse_response,
    parse_stream_event,
)


# =============================================================================
# build_request_body
# =============================================================================


class TestBuildRequestBody:
    def test_first_turn_extracts_system_to_instructions_and_user_to_input(self) -> None:
        body = build_request_body(
            [Message.system("You are concise."), Message.user("Hello.")],
            model="openai.gpt-5",
        )
        assert body["model"] == "openai.gpt-5"
        assert body["instructions"] == "You are concise."
        assert body["input"] == [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello."}],
            }
        ]
        assert "previous_response_id" not in body
        assert "tools" not in body
        assert "stream" not in body

    def test_multiple_system_messages_join_with_blank_line(self) -> None:
        body = build_request_body(
            [Message.system("Rule 1."), Message.system("Rule 2."), Message.user("Go.")],
            model="openai.gpt-5",
        )
        assert body["instructions"] == "Rule 1.\n\nRule 2."

    def test_continuation_turn_carries_previous_response_id(self) -> None:
        body = build_request_body(
            [Message.user("Step 2.")],
            model="openai.gpt-5",
            previous_response_id="resp_abc",
        )
        assert body["previous_response_id"] == "resp_abc"
        assert body["input"][0]["role"] == "user"

    def test_tool_result_message_becomes_function_call_output(self) -> None:
        tool_msg = Message.tool(
            ToolResult(tool_call_id="call_xyz", name="search", content="found 5 results")
        )
        body = build_request_body([tool_msg], model="openai.gpt-5")
        assert body["input"] == [
            {
                "type": "function_call_output",
                "call_id": "call_xyz",
                "output": "found 5 results",
            }
        ]

    def test_tools_get_flattened_for_responses(self) -> None:
        chat_tool = {
            "type": "function",
            "function": {
                "name": "search",
                "description": "search the docs",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            },
        }
        body = build_request_body(
            [Message.user("hi")],
            model="openai.gpt-5",
            tools=[chat_tool],
        )
        assert body["tools"] == [
            {
                "type": "function",
                "name": "search",
                "description": "search the docs",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            }
        ]

    def test_streaming_flag_emits_stream_true(self) -> None:
        body = build_request_body([Message.user("hi")], model="openai.gpt-5", stream=True)
        assert body["stream"] is True

    def test_temperature_and_max_tokens_pass_through(self) -> None:
        body = build_request_body(
            [Message.user("hi")],
            model="openai.gpt-5",
            temperature=0.3,
            max_output_tokens=512,
        )
        assert body["temperature"] == 0.3
        assert body["max_output_tokens"] == 512


# =============================================================================
# parse_response
# =============================================================================


class TestParseResponse:
    def test_assistant_text_only(self) -> None:
        payload = {
            "id": "resp_1",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hi there."}],
                }
            ],
            "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        }
        msg, usage, stop_reason, provider_state = parse_response(payload)
        assert msg.content == "Hi there."
        assert msg.tool_calls == []
        assert usage == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
        assert stop_reason == "completed"
        assert provider_state == {"previous_response_id": "resp_1"}

    def test_tool_call_extraction(self) -> None:
        payload = {
            "id": "resp_2",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_42",
                    "name": "search",
                    "arguments": json.dumps({"q": "weather"}),
                }
            ],
        }
        msg, _, _, provider_state = parse_response(payload)
        assert msg.content is None
        assert len(msg.tool_calls) == 1
        call = msg.tool_calls[0]
        assert call.id == "call_42"
        assert call.name == "search"
        assert call.arguments == {"q": "weather"}
        assert provider_state == {"previous_response_id": "resp_2"}

    def test_interleaved_text_and_tool_calls(self) -> None:
        payload = {
            "id": "resp_3",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Let me check. "}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_x",
                    "name": "lookup",
                    "arguments": "{}",
                },
            ],
        }
        msg, _, _, _ = parse_response(payload)
        assert msg.content == "Let me check. "
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "lookup"

    def test_malformed_tool_arguments_falls_back_to_empty_dict(self) -> None:
        payload = {
            "id": "resp_4",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_q",
                    "name": "x",
                    "arguments": "not-json",
                }
            ],
        }
        msg, _, _, _ = parse_response(payload)
        assert msg.tool_calls[0].arguments == {}

    def test_usage_falls_back_to_chat_completion_keys(self) -> None:
        payload = {
            "id": "resp_5",
            "output": [],
            "usage": {"prompt_tokens": 7, "completion_tokens": 11},
        }
        _, usage, _, _ = parse_response(payload)
        assert usage["prompt_tokens"] == 7
        assert usage["completion_tokens"] == 11
        assert usage["total_tokens"] == 18  # derived

    def test_no_id_yields_empty_provider_state(self) -> None:
        _, _, _, provider_state = parse_response({"output": []})
        assert provider_state == {}


# =============================================================================
# parse_stream_event
# =============================================================================


class TestParseStreamEvent:
    def test_output_text_delta(self) -> None:
        out = parse_stream_event({"type": "response.output_text.delta", "delta": "abc"})
        assert out == {"content": "abc"}

    def test_function_call_arguments_delta(self) -> None:
        out = parse_stream_event(
            {
                "type": "response.function_call_arguments.delta",
                "call_id": "call_1",
                "delta": '{"q":',
            }
        )
        assert out == {"tool_calls": [{"id": "call_1", "arguments_delta": '{"q":'}]}

    def test_response_completed_emits_done_and_provider_state(self) -> None:
        out = parse_stream_event({"type": "response.completed", "response": {"id": "resp_final"}})
        assert out == {"done": True, "provider_state": {"previous_response_id": "resp_final"}}

    def test_response_error(self) -> None:
        out = parse_stream_event({"type": "response.error", "error": {"message": "rate limited"}})
        assert out == {"error": "rate limited"}

    def test_unknown_event_returns_empty_dict(self) -> None:
        out = parse_stream_event({"type": "response.output_item.added"})
        assert out == {}
