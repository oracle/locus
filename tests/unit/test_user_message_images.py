# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Images a person attaches to a prompt reach a model that can see them.

``encode_image`` already carried screenshots inside tool results. A harness
that lets a person write ``look at @diagram.png`` needs the same for the user
turn: real image parts on every transport that supports them, and a
placeholder — never a base64 blob read as text — on the one that does not.
"""

from __future__ import annotations

from tulip.core.media import IMAGE_OMITTED, encode_image
from tulip.core.messages import Message
from tulip.models.native.anthropic import AnthropicModel
from tulip.models.native.bedrock import BedrockModel
from tulip.models.native.openai import OpenAIModel


PNG = b"\x89PNG\r\n\x1a\nfake-diagram"
PROMPT = "what does this show?" + encode_image(PNG)


def test_anthropic_sends_text_and_image_blocks() -> None:
    m = AnthropicModel(api_key="sk-x")  # noqa: S106
    _, converted = m._convert_messages([Message.user(PROMPT)])
    blocks = converted[0]["content"]
    assert blocks[0] == {"type": "text", "text": "what does this show?"}
    assert blocks[1]["type"] == "image"
    assert blocks[1]["source"]["media_type"] == "image/png"


def test_a_plain_user_turn_stays_a_string() -> None:
    m = AnthropicModel(api_key="sk-x")  # noqa: S106
    _, converted = m._convert_messages([Message.user("hello")])
    assert converted[0]["content"] == "hello"


def test_chat_completions_sends_image_url_parts() -> None:
    m = OpenAIModel(model="gpt-5.5", api_key="sk-x")  # noqa: S106
    converted = m._convert_messages([Message.user(PROMPT)])
    parts = converted[0]["content"]
    assert parts[0] == {"type": "text", "text": "what does this show?"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_responses_sends_input_image_parts() -> None:
    m = OpenAIModel(model="gpt-5.6-sol", api_key="sk-x")  # noqa: S106
    items = m._convert_messages_responses([Message.user(PROMPT)])
    parts = items[0]["content"]
    assert parts[0] == {"type": "input_text", "text": "what does this show?"}
    assert parts[1]["type"] == "input_image"


def test_bedrock_sends_a_placeholder_not_base64() -> None:
    m = BedrockModel(model="anthropic.claude-sonnet-4-6", region="us-east-1")
    _, converted = m._convert_messages([Message.user(PROMPT)])
    text = converted[0]["content"][0]["text"]
    assert IMAGE_OMITTED in text
    assert "tulip-image" not in text


def test_tool_images_follow_the_whole_tool_batch_on_chat_completions() -> None:
    from tulip.core.messages import ToolCall, ToolResult

    m = OpenAIModel(model="gpt-5.5", api_key="sk-x")  # noqa: S106
    converted = m._convert_messages(
        [
            Message.user("read both"),
            Message.assistant(
                content=None,
                tool_calls=[
                    ToolCall(id="a", name="read", arguments={}),
                    ToolCall(id="b", name="read", arguments={}),
                ],
            ),
            Message.tool(
                ToolResult(tool_call_id="a", name="read", content="a.png" + encode_image(PNG))
            ),
            Message.tool(ToolResult(tool_call_id="b", name="read", content="b.txt")),
            Message.assistant("both read"),
        ]
    )
    roles = [entry["role"] for entry in converted]
    # The tool messages stay contiguous behind their assistant turn; the image
    # rides in a user message after the batch.
    assert roles == ["user", "assistant", "tool", "tool", "user", "assistant"]
    assert converted[4]["content"][1]["type"] == "image_url"
