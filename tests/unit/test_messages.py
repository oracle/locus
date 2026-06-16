# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Tests for message types."""

import pytest
from pydantic import ValidationError

from locus.core.messages import Message, Role, ToolCall, ToolResult


class TestRole:
    """Tests for Role enum."""

    def test_role_values(self):
        """Role values match expected strings."""
        assert Role.SYSTEM.value == "system"
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"
        assert Role.TOOL.value == "tool"


class TestToolCall:
    """Tests for ToolCall."""

    def test_create_tool_call(self):
        """Create a tool call with arguments."""
        tc = ToolCall(name="search", arguments={"query": "test"})

        assert tc.name == "search"
        assert tc.arguments == {"query": "test"}
        assert tc.id.startswith("call_")

    def test_tool_call_to_openai_format(self):
        """Convert tool call to OpenAI format."""
        tc = ToolCall(id="call_123", name="search", arguments={"query": "test"})
        result = tc.to_openai_format()

        assert result["id"] == "call_123"
        assert result["type"] == "function"
        assert result["function"]["name"] == "search"
        assert '"query": "test"' in result["function"]["arguments"]


class TestToolResult:
    """Tests for ToolResult."""

    def test_successful_result(self):
        """Create a successful tool result."""
        result = ToolResult(
            tool_call_id="call_123",
            name="search",
            content="Found 5 results",
        )

        assert result.success is True
        assert result.error is None
        assert result.content == "Found 5 results"

    def test_failed_result(self):
        """Create a failed tool result."""
        result = ToolResult(
            tool_call_id="call_123",
            name="search",
            content="",
            error="Connection timeout",
        )

        assert result.success is False
        assert result.error == "Connection timeout"


class TestMessage:
    """Tests for Message."""

    def test_create_system_message(self):
        """Create a system message."""
        msg = Message.system("You are a helpful assistant.")

        assert msg.role == Role.SYSTEM
        assert msg.content == "You are a helpful assistant."
        assert msg.tool_calls == []

    def test_create_user_message(self):
        """Create a user message."""
        msg = Message.user("Hello!")

        assert msg.role == Role.USER
        assert msg.content == "Hello!"

    def test_create_assistant_message_with_content(self):
        """Create an assistant message with content."""
        msg = Message.assistant(content="Hello! How can I help?")

        assert msg.role == Role.ASSISTANT
        assert msg.content == "Hello! How can I help?"
        assert msg.tool_calls == []

    def test_create_assistant_message_with_tool_calls(self):
        """Create an assistant message with tool calls."""
        tc = ToolCall(name="search", arguments={"query": "test"})
        msg = Message.assistant(tool_calls=[tc])

        assert msg.role == Role.ASSISTANT
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "search"

    def test_create_tool_message(self):
        """Create a tool result message."""
        result = ToolResult(
            tool_call_id="call_123",
            name="search",
            content="Found results",
        )
        msg = Message.tool(result)

        assert msg.role == Role.TOOL
        assert msg.content == "Found results"
        assert msg.tool_call_id == "call_123"
        assert msg.name == "search"

    def test_message_is_frozen(self):
        """Messages are immutable."""
        msg = Message.user("Hello!")

        with pytest.raises(ValidationError):
            msg.content = "Changed"  # type: ignore[misc]

    def test_to_openai_format(self):
        """Convert message to OpenAI format."""
        msg = Message.user("Hello!")
        result = msg.to_openai_format()

        assert result["role"] == "user"
        assert result["content"] == "Hello!"

    def test_to_openai_format_with_tool_calls(self):
        """Convert message with tool calls to OpenAI format."""
        tc = ToolCall(id="call_123", name="search", arguments={"q": "test"})
        msg = Message.assistant(content="Let me search", tool_calls=[tc])
        result = msg.to_openai_format()

        assert result["role"] == "assistant"
        assert result["content"] == "Let me search"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "call_123"

    def test_to_openai_format_tool_message(self):
        """Convert tool message to OpenAI format."""
        tool_result = ToolResult(
            tool_call_id="call_123",
            name="search",
            content="Found results",
        )
        msg = Message.tool(tool_result)
        result = msg.to_openai_format()

        assert result["role"] == "tool"
        assert result["content"] == "Found results"
        assert result["tool_call_id"] == "call_123"
        assert result["name"] == "search"
