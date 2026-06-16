# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Cohere provider for Command R/R+ models on OCI GenAI."""

from __future__ import annotations

import json
from typing import Any

from locus.core.messages import Message, Role, ToolCall
from locus.models.providers.oci.base import OCIModelProvider


class CohereProvider(OCIModelProvider):
    """Provider for Cohere Command R and R+ models on OCI.

    Supports: cohere.command-r-plus, cohere.command-r-16k, etc.
    Note: cohere.command-a-* (A series) models use the GenericProvider.
    """

    def __init__(self) -> None:
        from oci.generative_ai_inference import models

        # Chat request class
        self.oci_chat_request = models.CohereChatRequest

        # Message classes
        self.oci_user_message = models.CohereUserMessage
        self.oci_chatbot_message = models.CohereChatBotMessage
        self.oci_system_message = models.CohereSystemMessage
        self.oci_tool_message = models.CohereToolMessage

        # Tool classes
        self.oci_tool = models.CohereTool
        self.oci_tool_param = models.CohereParameterDefinition
        self.oci_tool_result = models.CohereToolResult
        self.oci_tool_call = models.CohereToolCall

        # API format
        self._api_format = models.BaseChatRequest.API_FORMAT_COHERE

    @property
    def api_format(self) -> str:
        return str(self._api_format)

    @property
    def stop_sequence_key(self) -> str:
        return "stop_sequences"

    def build_request(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Build a CohereChatRequest.

        Cohere uses a different format: message (current) + chat_history (previous).
        """
        # Extract current message and chat history from converted messages
        current_message = kwargs.pop("_current_message", "")
        chat_history = kwargs.pop("_chat_history", [])
        tool_results = kwargs.pop("_tool_results", None)

        request = self.oci_chat_request(
            message=current_message,
            chat_history=chat_history or None,
            api_format=self.api_format,
            max_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
        )

        # Add tools if provided
        if tools:
            request.tools = tools

        # Add tool results if provided
        if tool_results:
            request.tool_results = tool_results

        # Add stop sequences if provided
        stop = kwargs.get("stop_sequences") or kwargs.get("stop")
        if stop:
            request.stop_sequences = stop

        return request

    def convert_messages(
        self, messages: list[Message], model_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Convert Locus messages to Cohere format.

        Args:
            messages: List of Locus messages to convert
            model_id: Optional model ID (not used for Cohere, but required by base)

        Returns a dict with:
        - _current_message: The current user message
        - _chat_history: List of previous messages
        - _tool_results: Any tool results to send

        Only tool results that trail the final assistant message (i.e. still
        unresolved for the current request) are sent as top-level tool_results.
        Historical tool results are embedded in chat_history as CohereToolMessage
        entries — Cohere R rejects requests that combine a new ``message`` with
        stale ``tool_results``.
        """
        chat_history: list[Any] = []
        current_message = ""
        tool_results: list[Any] = []
        pending_tool_results: list[Any] = []

        last_assistant_idx = -1
        for i, msg in enumerate(messages):
            if msg.role == Role.ASSISTANT:
                last_assistant_idx = i

        def flush_pending() -> None:
            if pending_tool_results:
                chat_history.append(self.oci_tool_message(tool_results=list(pending_tool_results)))
                pending_tool_results.clear()

        for i, msg in enumerate(messages):
            is_last = i == len(messages) - 1

            if msg.role == Role.SYSTEM:
                flush_pending()
                chat_history.append(self.oci_system_message(message=msg.content or ""))

            elif msg.role == Role.USER:
                flush_pending()
                if is_last:
                    current_message = msg.content or ""
                else:
                    chat_history.append(self.oci_user_message(message=msg.content or ""))

            elif msg.role == Role.ASSISTANT:
                flush_pending()
                tool_calls = None
                if msg.tool_calls:
                    tool_calls = [
                        self.oci_tool_call(name=tc.name, parameters=tc.arguments)
                        for tc in msg.tool_calls
                    ]
                chat_history.append(
                    self.oci_chatbot_message(
                        message=msg.content or " ",
                        tool_calls=tool_calls,
                    )
                )

            elif msg.role == Role.TOOL:
                tool_result = self.oci_tool_result(
                    call=self.oci_tool_call(name=msg.name or "", parameters={}),
                    outputs=[{"output": msg.content or ""}],
                )
                if i > last_assistant_idx:
                    tool_results.append(tool_result)
                else:
                    pending_tool_results.append(tool_result)

        flush_pending()

        return {  # type: ignore[return-value]
            "_current_message": current_message,
            "_chat_history": chat_history,
            "_tool_results": tool_results or None,
        }

    def convert_tools(self, tools: list[dict[str, Any]] | None) -> list[Any] | None:
        """Convert OpenAI-style tools to Cohere CohereTool format."""
        if not tools:
            return None

        # JSON type to Python type mapping for Cohere
        type_map = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "array": "list",
            "object": "dict",
        }

        oci_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                params = func.get("parameters", {})
                properties = params.get("properties", {})
                required = params.get("required", [])

                # Convert parameters to CohereParameterDefinition
                param_defs = {}
                for p_name, p_def in properties.items():
                    param_defs[p_name] = self.oci_tool_param(
                        description=p_def.get("description", ""),
                        type=type_map.get(p_def.get("type", "string"), "str"),
                        is_required=p_name in required,
                    )

                oci_tools.append(
                    self.oci_tool(
                        name=func["name"],
                        description=func.get("description", ""),
                        parameter_definitions=param_defs,
                    )
                )

        return oci_tools or None

    def parse_response(self, response: Any) -> tuple[str | None, list[ToolCall], str | None]:
        """Parse a CohereChatRequest response."""
        chat_response = response.data.chat_response

        # Extract content
        content = getattr(chat_response, "text", None)

        # Extract stop reason
        stop_reason = getattr(chat_response, "finish_reason", None)

        # Extract tool calls
        tool_calls: list[ToolCall] = []
        raw_tool_calls = getattr(chat_response, "tool_calls", None)
        if raw_tool_calls:
            for tc in raw_tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=f"call_{tc.name}",  # Cohere doesn't provide IDs
                        name=tc.name,
                        arguments=tc.parameters or {},
                    )
                )

        return content, tool_calls, stop_reason

    def parse_stream_chunk(self, event_data: dict) -> tuple[str, list[ToolCall], bool]:
        """Parse a Cohere streaming event."""
        is_done = "finishReason" in event_data
        content = event_data.get("text", "")
        tool_calls: list[ToolCall] = []

        # Handle tool calls in stream
        raw_tool_calls = event_data.get("toolCalls", [])
        for tc in raw_tool_calls:
            params = tc.get("parameters", {})
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except json.JSONDecodeError:
                    params = {}
            tool_calls.append(
                ToolCall(
                    id=f"call_{tc.get('name', 'unknown')}",
                    name=tc.get("name", ""),
                    arguments=params,
                )
            )

        return content, tool_calls, is_done
