# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Base provider class for OCI GenAI models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from locus.core.messages import Message, ToolCall


class OCIModelProvider(ABC):
    """Abstract base class for OCI GenAI model providers.

    Each provider handles a specific model family (Cohere, Meta, OpenAI, etc.)
    with its own request/response format.
    """

    @property
    @abstractmethod
    def api_format(self) -> str:
        """Return the API format identifier for this provider."""
        ...

    @property
    def stop_sequence_key(self) -> str:
        """Return the parameter name for stop sequences."""
        return "stop"

    @property
    def supports_tools(self) -> bool:
        """Whether this provider supports tool/function calling."""
        return True

    @property
    def supports_streaming(self) -> bool:
        """Whether this provider supports streaming responses."""
        return True

    @abstractmethod
    def build_request(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Build a provider-specific chat request.

        Args:
            messages: Converted messages in OCI format
            tools: Converted tools in OCI format
            **kwargs: Additional parameters (max_tokens, temperature, etc.)

        Returns:
            Provider-specific request object (e.g., CohereChatRequest, GenericChatRequest)
        """
        ...

    @abstractmethod
    def parse_response(self, response: Any) -> tuple[str | None, list[ToolCall], str | None]:
        """Parse a provider-specific response.

        Args:
            response: Raw response from OCI API

        Returns:
            Tuple of (content, tool_calls, stop_reason)
        """
        ...

    @abstractmethod
    def convert_messages(
        self, messages: list[Message], model_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Convert Locus messages to provider-specific format.

        Args:
            messages: List of Locus Message objects
            model_id: Optional model identifier for model-specific handling

        Returns:
            List of message dicts in provider-specific format
        """
        ...

    @abstractmethod
    def convert_tools(self, tools: list[dict[str, Any]] | None) -> list[Any] | None:
        """Convert OpenAI-style tools to provider-specific format.

        Args:
            tools: Tools in OpenAI function calling format

        Returns:
            Tools in provider-specific format, or None
        """
        ...

    def parse_usage(self, response: Any) -> dict[str, int]:
        """Extract token usage from response.

        Args:
            response: Raw response from OCI API

        Returns:
            Dict with prompt_tokens, completion_tokens
        """
        chat_response = response.data.chat_response
        if hasattr(chat_response, "usage") and chat_response.usage:
            return {
                "prompt_tokens": getattr(chat_response.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(chat_response.usage, "completion_tokens", 0) or 0,
            }
        return {}

    def parse_stream_chunk(self, event_data: dict) -> tuple[str, list[ToolCall], bool]:
        """Parse a streaming event.

        Args:
            event_data: Streaming event data

        Returns:
            Tuple of (content, tool_calls, is_done)
        """
        # Default implementation - providers should override
        return "", [], "finishReason" in event_data
