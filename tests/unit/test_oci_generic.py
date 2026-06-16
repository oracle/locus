# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for OCI Generic model provider."""

from unittest.mock import MagicMock, patch

import pytest

from locus.core.messages import Message, Role, ToolCall


class TestFlattenParallelToolCalls:
    """Tests for _flatten_parallel_tool_calls helper."""

    def test_no_tool_calls_unchanged(self):
        """Test messages without tool calls remain unchanged."""
        from locus.models.providers.oci.models.generic import _flatten_parallel_tool_calls

        messages = [
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi there"),
        ]
        result = _flatten_parallel_tool_calls(messages)

        assert len(result) == 2
        assert result[0].content == "Hello"
        assert result[1].content == "Hi there"

    def test_single_tool_call_unchanged(self):
        """Test single tool call remains unchanged."""
        from locus.models.providers.oci.models.generic import _flatten_parallel_tool_calls

        tc = ToolCall(id="1", name="search", arguments={})
        messages = [
            Message(role=Role.ASSISTANT, content="Calling", tool_calls=[tc]),
            Message(role=Role.TOOL, tool_call_id="1", content="Result"),
        ]
        result = _flatten_parallel_tool_calls(messages)

        assert len(result) == 2

    def test_parallel_tool_calls_flattened(self):
        """Test parallel tool calls are flattened into sequential pairs."""
        from locus.models.providers.oci.models.generic import _flatten_parallel_tool_calls

        tc1 = ToolCall(id="1", name="search", arguments={"q": "a"})
        tc2 = ToolCall(id="2", name="lookup", arguments={"k": "b"})

        messages = [
            Message(role=Role.ASSISTANT, content="Calling tools", tool_calls=[tc1, tc2]),
            Message(role=Role.TOOL, tool_call_id="1", content="Result 1"),
            Message(role=Role.TOOL, tool_call_id="2", content="Result 2"),
        ]
        result = _flatten_parallel_tool_calls(messages)

        # Should be: Assistant(tc1), Tool(1), Assistant(tc2), Tool(2)
        assert len(result) == 4
        assert result[0].role == Role.ASSISTANT
        assert len(result[0].tool_calls) == 1
        assert result[0].tool_calls[0].id == "1"
        assert result[0].content == "Calling tools"

        assert result[1].role == Role.TOOL
        assert result[1].tool_call_id == "1"

        assert result[2].role == Role.ASSISTANT
        assert result[2].content == "."  # Placeholder for subsequent

        assert result[3].role == Role.TOOL
        assert result[3].tool_call_id == "2"

    def test_parallel_tool_calls_no_content(self):
        """Test parallel tool calls with no content uses placeholder."""
        from locus.models.providers.oci.models.generic import _flatten_parallel_tool_calls

        tc1 = ToolCall(id="1", name="a", arguments={})
        tc2 = ToolCall(id="2", name="b", arguments={})

        messages = [
            Message(role=Role.ASSISTANT, content=None, tool_calls=[tc1, tc2]),
            Message(role=Role.TOOL, tool_call_id="1", content="R1"),
            Message(role=Role.TOOL, tool_call_id="2", content="R2"),
        ]
        result = _flatten_parallel_tool_calls(messages)

        assert result[0].content == "."


class TestGenericProvider:
    """Tests for GenericProvider class."""

    @pytest.fixture
    def mock_oci_models(self):
        """Create mock OCI models module."""
        models = MagicMock()
        models.GenericChatRequest = MagicMock
        models.UserMessage = MagicMock
        models.SystemMessage = MagicMock
        models.AssistantMessage = MagicMock
        models.ToolMessage = MagicMock
        models.TextContent = MagicMock
        models.FunctionDefinition = MagicMock
        models.FunctionCall = MagicMock
        models.BaseChatRequest = MagicMock()
        models.BaseChatRequest.API_FORMAT_GENERIC = "GENERIC"
        return models

    @pytest.fixture
    def provider(self, mock_oci_models):
        """Create provider with mocked OCI models."""
        with (
            patch.dict(
                "sys.modules",
                {"oci": MagicMock(), "oci.generative_ai_inference": MagicMock()},
            ),
            patch("oci.generative_ai_inference.models", mock_oci_models),
        ):
            from locus.models.providers.oci.models.generic import GenericProvider

            return GenericProvider()

    def test_init_sets_oci_classes(self, provider):
        """Test that init sets all required OCI classes."""
        assert provider.oci_chat_request is not None
        assert provider.oci_user_message is not None
        assert provider.oci_system_message is not None
        assert provider.oci_assistant_message is not None
        assert provider.oci_tool_message is not None
        assert provider.oci_text_content is not None
        assert provider.oci_function_definition is not None
        assert provider.oci_function_call is not None

    def test_api_format(self, provider):
        """Test api_format property."""
        assert provider.api_format == "GENERIC"

    def test_stop_sequence_key(self, provider):
        """Test stop_sequence_key property."""
        assert provider.stop_sequence_key == "stop"

    def test_build_request_basic(self, provider):
        """Test building a basic request.

        Non-OpenAI vendors (Meta Llama, Cohere, xAI, Google) take the
        ``max_tokens`` spelling; the OpenAI-family spelling is exercised
        by ``test_build_request_openai_uses_max_completion_tokens``.
        """
        provider.oci_chat_request = MagicMock(return_value=MagicMock())

        messages = [MagicMock()]
        result = provider.build_request(messages, max_tokens=2000)

        provider.oci_chat_request.assert_called_once()
        call_kwargs = provider.oci_chat_request.call_args
        assert call_kwargs[1]["messages"] == messages
        assert call_kwargs[1]["max_tokens"] == 2000

    def test_build_request_openai_uses_max_completion_tokens(self, provider):
        """OpenAI-family models require ``max_completion_tokens``."""
        provider.oci_chat_request = MagicMock(return_value=MagicMock())

        messages = [MagicMock()]
        provider.build_request(messages, max_tokens=2000, model_id="openai.gpt-4.1")

        call_kwargs = provider.oci_chat_request.call_args
        assert call_kwargs[1]["max_completion_tokens"] == 2000
        assert "max_tokens" not in call_kwargs[1]

    def test_build_request_with_tools(self, provider):
        """Test building request with tools."""
        mock_request = MagicMock()
        provider.oci_chat_request = MagicMock(return_value=mock_request)

        tools = [MagicMock()]
        result = provider.build_request([], tools=tools)

        assert mock_request.tools == tools

    def test_build_request_with_stop(self, provider):
        """Test building request with stop sequences."""
        mock_request = MagicMock()
        provider.oci_chat_request = MagicMock(return_value=mock_request)

        result = provider.build_request([], stop=["STOP"])

        assert mock_request.stop == ["STOP"]

    def test_convert_messages_system(self, provider):
        """Test converting system message."""
        provider.oci_text_content = MagicMock(return_value="text")
        provider.oci_system_message = MagicMock(return_value="sys")

        messages = [Message(role=Role.SYSTEM, content="System prompt")]
        result = provider.convert_messages(messages)

        assert len(result) == 1
        provider.oci_system_message.assert_called_once()

    def test_convert_messages_user(self, provider):
        """Test converting user message."""
        provider.oci_text_content = MagicMock(return_value="text")
        provider.oci_user_message = MagicMock(return_value="user")

        messages = [Message(role=Role.USER, content="Hello")]
        result = provider.convert_messages(messages)

        assert len(result) == 1
        provider.oci_user_message.assert_called_once()

    def test_convert_messages_assistant(self, provider):
        """Test converting assistant message."""
        provider.oci_text_content = MagicMock(return_value="text")
        provider.oci_assistant_message = MagicMock(return_value="asst")

        messages = [Message(role=Role.ASSISTANT, content="Response")]
        result = provider.convert_messages(messages)

        assert len(result) == 1
        provider.oci_assistant_message.assert_called_once()

    def test_convert_messages_assistant_no_content(self, provider):
        """Test converting assistant message with no content uses placeholder."""
        provider.oci_text_content = MagicMock(return_value="text")
        provider.oci_assistant_message = MagicMock(return_value="asst")

        messages = [Message(role=Role.ASSISTANT, content=None)]
        result = provider.convert_messages(messages)

        # Should have called text_content with "."
        calls = provider.oci_text_content.call_args_list
        assert any(call[1].get("text") == "." for call in calls)

    def test_convert_messages_assistant_with_tool_calls(self, provider):
        """Test converting assistant message with tool calls."""
        provider.oci_text_content = MagicMock(return_value="text")
        provider.oci_assistant_message = MagicMock(return_value="asst")
        provider.oci_function_call = MagicMock(return_value="fc")

        tc = ToolCall(id="1", name="search", arguments={"q": "test"})
        messages = [Message(role=Role.ASSISTANT, content="Calling", tool_calls=[tc])]
        result = provider.convert_messages(messages)

        provider.oci_function_call.assert_called_once()

    def test_convert_messages_tool(self, provider):
        """Test converting tool message."""
        provider.oci_text_content = MagicMock(return_value="text")
        provider.oci_tool_message = MagicMock(return_value="tool")

        messages = [Message(role=Role.TOOL, tool_call_id="1", content="Result")]
        result = provider.convert_messages(messages)

        assert len(result) == 1
        provider.oci_tool_message.assert_called_once()

    def test_convert_messages_gemini_flattens(self, provider):
        """Test Gemini model flattens parallel tool calls."""
        provider.oci_text_content = MagicMock(return_value="text")
        provider.oci_assistant_message = MagicMock(return_value="asst")
        provider.oci_tool_message = MagicMock(return_value="tool")
        provider.oci_function_call = MagicMock(return_value="fc")

        tc1 = ToolCall(id="1", name="a", arguments={})
        tc2 = ToolCall(id="2", name="b", arguments={})

        messages = [
            Message(role=Role.ASSISTANT, content="Call", tool_calls=[tc1, tc2]),
            Message(role=Role.TOOL, tool_call_id="1", content="R1"),
            Message(role=Role.TOOL, tool_call_id="2", content="R2"),
        ]

        result = provider.convert_messages(messages, model_id="google.gemini-pro")

        # Should create 4 messages (2 assistant + 2 tool)
        assert len(result) == 4

    def test_convert_tools_none(self, provider):
        """Test converting None tools."""
        result = provider.convert_tools(None)
        assert result is None

    def test_convert_tools_empty(self, provider):
        """Test converting empty tools list."""
        result = provider.convert_tools([])
        assert result is None

    def test_convert_tools_function(self, provider):
        """Test converting function tools."""
        provider.oci_function_definition = MagicMock(return_value="fd")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search web",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        result = provider.convert_tools(tools)

        assert result is not None
        assert len(result) == 1
        provider.oci_function_definition.assert_called_once()

    def test_parse_response_no_choices(self, provider):
        """Test parsing response with no choices."""
        mock_response = MagicMock()
        mock_response.data.chat_response.choices = None

        content, tool_calls, stop_reason = provider.parse_response(mock_response)

        assert content is None
        assert tool_calls == []
        assert stop_reason is None

    def test_parse_response_no_message(self, provider):
        """Test parsing response with no message."""
        mock_choice = MagicMock()
        mock_choice.message = None
        mock_choice.finish_reason = "STOP"

        mock_response = MagicMock()
        mock_response.data.chat_response.choices = [mock_choice]

        content, tool_calls, stop_reason = provider.parse_response(mock_response)

        assert content is None
        assert tool_calls == []
        assert stop_reason == "STOP"

    def test_parse_response_text_content(self, provider):
        """Test parsing response with text content."""
        mock_text = MagicMock()
        mock_text.type = "TEXT"
        mock_text.text = "Hello world"

        mock_message = MagicMock()
        mock_message.content = [mock_text]
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "COMPLETE"

        mock_response = MagicMock()
        mock_response.data.chat_response.choices = [mock_choice]

        content, tool_calls, stop_reason = provider.parse_response(mock_response)

        assert content == "Hello world"
        assert tool_calls == []
        assert stop_reason == "COMPLETE"

    def test_parse_response_with_tool_calls(self, provider):
        """Test parsing response with tool calls."""
        mock_tc = MagicMock()
        mock_tc.id = "call_1"
        mock_tc.name = "search"
        mock_tc.arguments = '{"q": "test"}'

        mock_message = MagicMock()
        mock_message.content = []
        mock_message.tool_calls = [mock_tc]

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "TOOL_CALL"

        mock_response = MagicMock()
        mock_response.data.chat_response.choices = [mock_choice]

        content, tool_calls, stop_reason = provider.parse_response(mock_response)

        assert len(tool_calls) == 1
        assert tool_calls[0].name == "search"
        assert tool_calls[0].arguments == {"q": "test"}

    def test_parse_response_tool_calls_dict_args(self, provider):
        """Test parsing response with dict arguments."""
        mock_tc = MagicMock()
        mock_tc.id = "call_1"
        mock_tc.name = "search"
        mock_tc.arguments = {"q": "test"}

        mock_message = MagicMock()
        mock_message.content = []
        mock_message.tool_calls = [mock_tc]

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "TOOL_CALL"

        mock_response = MagicMock()
        mock_response.data.chat_response.choices = [mock_choice]

        content, tool_calls, stop_reason = provider.parse_response(mock_response)

        assert tool_calls[0].arguments == {"q": "test"}

    def test_parse_response_tool_calls_invalid_json(self, provider):
        """Test parsing response with invalid JSON arguments."""
        mock_tc = MagicMock()
        mock_tc.id = "call_1"
        mock_tc.name = "search"
        mock_tc.arguments = "not valid json"

        mock_message = MagicMock()
        mock_message.content = []
        mock_message.tool_calls = [mock_tc]

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = None

        mock_response = MagicMock()
        mock_response.data.chat_response.choices = [mock_choice]

        content, tool_calls, stop_reason = provider.parse_response(mock_response)

        assert tool_calls[0].arguments == {}

    def test_parse_response_tool_calls_no_id(self, provider):
        """Test parsing response with tool call missing id."""
        mock_tc = MagicMock()
        mock_tc.id = None
        mock_tc.name = "search"
        mock_tc.arguments = "{}"

        mock_message = MagicMock()
        mock_message.content = []
        mock_message.tool_calls = [mock_tc]

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = None

        mock_response = MagicMock()
        mock_response.data.chat_response.choices = [mock_choice]

        content, tool_calls, stop_reason = provider.parse_response(mock_response)

        assert tool_calls[0].id == "call_search"

    def test_parse_stream_chunk_text(self, provider):
        """Test parsing text stream chunk."""
        event_data = {
            "message": {
                "content": [{"type": "TEXT", "text": "Hello"}],
            }
        }

        content, tool_calls, is_done = provider.parse_stream_chunk(event_data)

        assert content == "Hello"
        assert tool_calls == []
        assert is_done is False

    def test_parse_stream_chunk_done(self, provider):
        """Test parsing done stream chunk."""
        event_data = {"finishReason": "COMPLETE", "message": {"content": []}}

        content, tool_calls, is_done = provider.parse_stream_chunk(event_data)

        assert is_done is True

    def test_parse_stream_chunk_with_tool_calls(self, provider):
        """Test parsing stream chunk with tool calls."""
        event_data = {
            "message": {
                "content": [],
                "toolCalls": [{"id": "1", "name": "search", "arguments": '{"q": "test"}'}],
            }
        }

        content, tool_calls, is_done = provider.parse_stream_chunk(event_data)

        assert len(tool_calls) == 1
        assert tool_calls[0].name == "search"
        assert tool_calls[0].arguments == {"q": "test"}

    def test_parse_stream_chunk_tool_calls_invalid_json(self, provider):
        """Test parsing stream chunk with invalid JSON args."""
        event_data = {
            "message": {
                "content": [],
                "toolCalls": [{"name": "search", "arguments": "bad json"}],
            }
        }

        content, tool_calls, is_done = provider.parse_stream_chunk(event_data)

        assert tool_calls[0].arguments == {}

    def test_parse_stream_chunk_tool_calls_no_name(self, provider):
        """Test parsing stream chunk with missing tool name."""
        event_data = {
            "message": {
                "content": [],
                "toolCalls": [{"arguments": "{}"}],
            }
        }

        content, tool_calls, is_done = provider.parse_stream_chunk(event_data)

        assert tool_calls[0].name == "unknown_tool"
