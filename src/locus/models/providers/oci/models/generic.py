# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Generic provider for OpenAI, Meta, xAI, Mistral, Google models on OCI GenAI."""

from __future__ import annotations

import json
from typing import Any

from locus.core.messages import Message, Role, ToolCall
from locus.models.providers.oci.base import OCIModelProvider


def _flatten_parallel_tool_calls(messages: list[Message]) -> list[Message]:
    """Flatten parallel tool calls into sequential Assistant->Tool pairs.

    Gemini models require each function call turn to have exactly one
    matching function response. When the model makes N parallel tool
    calls (one Assistant message with N tool_calls followed by N Tool messages),
    this method splits them into N sequential (Assistant, Tool) pairs so each
    turn has a 1:1 call-to-response mapping.

    Non-Gemini models are unaffected — this is only called when needed.
    """
    result: list[Message] = []
    i = 0

    while i < len(messages):
        msg = messages[i]

        # Check if this is an assistant message with multiple tool calls
        if msg.role == Role.ASSISTANT and len(msg.tool_calls) > 1:
            tool_calls = msg.tool_calls

            # Collect consecutive Tool messages following this Assistant message
            j = i + 1
            while j < len(messages) and messages[j].role == Role.TOOL:
                j += 1
            tool_msgs = messages[i + 1 : j]

            # Map tool_call_id -> Tool message for correct pairing
            tool_msg_map = {tm.tool_call_id: tm for tm in tool_msgs if tm.tool_call_id}

            # Create sequential Assistant -> Tool pairs
            for idx, tc in enumerate(tool_calls):
                # First keeps original content; rest get placeholder
                content = msg.content if idx == 0 else "."
                if not content:
                    content = "."

                synthetic_assistant = Message(
                    role=Role.ASSISTANT,
                    content=content,
                    tool_calls=[tc],
                )
                result.append(synthetic_assistant)

                # Add matching Tool message
                matching = tool_msg_map.get(tc.id)
                if matching:
                    result.append(matching)

            i = j  # Skip past processed Tool messages
        else:
            result.append(msg)
            i += 1

    return result


class GenericProvider(OCIModelProvider):
    """Provider for models using the generic/OpenAI-style API format.

    Supports: Meta Llama, xAI Grok, OpenAI, Mistral, Google models on OCI.
    """

    def __init__(self) -> None:
        from oci.generative_ai_inference import models

        # Chat request class
        self.oci_chat_request = models.GenericChatRequest

        # Message classes
        self.oci_user_message = models.UserMessage
        self.oci_system_message = models.SystemMessage
        self.oci_assistant_message = models.AssistantMessage
        self.oci_tool_message = models.ToolMessage

        # Content classes
        self.oci_text_content = models.TextContent

        # Tool classes
        self.oci_function_definition = models.FunctionDefinition
        self.oci_function_call = models.FunctionCall

        # API format
        self._api_format = models.BaseChatRequest.API_FORMAT_GENERIC

    @property
    def api_format(self) -> str:
        return str(self._api_format)

    @property
    def stop_sequence_key(self) -> str:
        return "stop"

    def build_request(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Build a GenericChatRequest."""
        # OCI GenAI vendors disagree on the token-limit field name:
        #   * Meta Llama   — only accepts `max_tokens`; rejects max_completion_tokens
        #   * OpenAI (gpt-4, gpt-5, o-series) — only accepts `max_completion_tokens`
        #   * Cohere / xAI / Google — accept either
        # Pick the spelling based on the model vendor so both families work.
        token_value = kwargs.get("max_tokens", 4096)
        model_id = kwargs.get("model_id", "") or ""
        request_kwargs: dict[str, Any] = {
            "messages": messages,
            "api_format": self.api_format,
        }
        if model_id.startswith("openai."):
            request_kwargs["max_completion_tokens"] = token_value
        else:
            request_kwargs["max_tokens"] = token_value
        request = self.oci_chat_request(**request_kwargs)

        # Add tools if provided
        if tools:
            request.tools = tools

        # Add stop sequences if provided
        stop = kwargs.get("stop_sequences") or kwargs.get("stop")
        if stop:
            request.stop = stop

        return request

    def convert_messages(self, messages: list[Message], model_id: str | None = None) -> list[Any]:
        """Convert Locus messages to OCI GenericChatRequest format.

        Args:
            messages: List of Locus messages to convert
            model_id: Optional model ID to enable model-specific handling
                     (e.g., Gemini requires flattening parallel tool calls)
        """
        # Gemini requires 1:1 function_call to function_response per turn.
        # Flatten parallel tool calls into sequential pairs.
        if model_id and model_id.startswith("google."):
            messages = _flatten_parallel_tool_calls(messages)

        oci_messages = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                content = [self.oci_text_content(text=msg.content or "")]
                oci_messages.append(self.oci_system_message(content=content))

            elif msg.role == Role.USER:
                content = [self.oci_text_content(text=msg.content or "")]
                oci_messages.append(self.oci_user_message(content=content))

            elif msg.role == Role.ASSISTANT:
                content = []
                if msg.content:
                    content.append(self.oci_text_content(text=msg.content))
                # Add empty text if no content (required by some models)
                if not content:
                    content.append(self.oci_text_content(text="."))

                # Handle tool calls
                tool_calls = None
                if msg.tool_calls:
                    tool_calls = []
                    for tc in msg.tool_calls:
                        tool_calls.append(
                            self.oci_function_call(
                                id=tc.id,
                                name=tc.name,
                                arguments=json.dumps(tc.arguments)
                                if isinstance(tc.arguments, dict)
                                else tc.arguments,
                            )
                        )

                oci_messages.append(
                    self.oci_assistant_message(content=content, tool_calls=tool_calls)
                )

            elif msg.role == Role.TOOL:
                content = [self.oci_text_content(text=str(msg.content or ""))]
                oci_messages.append(
                    self.oci_tool_message(
                        content=content,
                        tool_call_id=msg.tool_call_id or "",
                    )
                )

        return oci_messages

    def convert_tools(self, tools: list[dict[str, Any]] | None) -> list[Any] | None:
        """Convert OpenAI-style tools to OCI FunctionDefinition format."""
        if not tools:
            return None

        oci_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                oci_tools.append(
                    self.oci_function_definition(
                        name=func["name"],
                        description=func.get("description", ""),
                        parameters=func.get("parameters", {"type": "object", "properties": {}}),
                    )
                )

        return oci_tools or None

    def parse_response(self, response: Any) -> tuple[str | None, list[ToolCall], str | None]:
        """Parse a GenericChatRequest response."""
        chat_response = response.data.chat_response
        choices = getattr(chat_response, "choices", None)

        if not choices:
            return None, [], None

        choice = choices[0]
        message = getattr(choice, "message", None)
        stop_reason = getattr(choice, "finish_reason", None)

        if not message:
            return None, [], stop_reason

        # Extract content
        content: str | None = None
        content_parts = getattr(message, "content", [])
        if content_parts:
            texts = []
            for part in content_parts:
                if hasattr(part, "type") and part.type == "TEXT":
                    # getattr returns None if attr exists but is None, so use "or" fallback
                    text = getattr(part, "text", None) or ""
                    if text:
                        texts.append(text)
            content = "".join(texts) if texts else None

        # Extract tool calls
        tool_calls: list[ToolCall] = []
        raw_tool_calls = getattr(message, "tool_calls", None)
        if raw_tool_calls:
            for tc in raw_tool_calls:
                args = getattr(tc, "arguments", None) or "{}"
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                # Handle None values with fallbacks - name is required
                tc_name = getattr(tc, "name", None) or "unknown_tool"
                tc_id = getattr(tc, "id", None) or f"call_{tc_name}"

                tool_calls.append(
                    ToolCall(
                        id=tc_id,
                        name=tc_name,
                        arguments=args if isinstance(args, dict) else {},
                    )
                )

        return content, tool_calls, stop_reason

    def parse_stream_chunk(self, event_data: dict) -> tuple[str, list[ToolCall], bool]:
        """Parse a streaming event."""
        is_done = "finishReason" in event_data
        content = ""
        tool_calls: list[ToolCall] = []

        # Extract content from message.content
        message = event_data.get("message", {})
        content_parts = message.get("content", [])
        if content_parts:
            for part in content_parts:
                if isinstance(part, dict) and part.get("type") == "TEXT":
                    content += part.get("text", "")

        # Extract tool calls
        raw_tool_calls = message.get("toolCalls", [])
        for tc in raw_tool_calls:
            args = tc.get("arguments") or "{}"
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            # Handle None values with fallbacks - name is required
            tc_name = tc.get("name") or "unknown_tool"
            tc_id = tc.get("id") or f"call_{tc_name}"

            tool_calls.append(
                ToolCall(
                    id=tc_id,
                    name=tc_name,
                    arguments=args if isinstance(args, dict) else {},
                )
            )

        return content, tool_calls, is_done
