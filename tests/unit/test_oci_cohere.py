# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for OCI Cohere model provider."""

from unittest.mock import MagicMock, patch

import pytest

from locus.core.messages import Message, Role, ToolCall


class TestCohereProvider:
    """Tests for CohereProvider class."""

    @pytest.fixture
    def mock_oci_models(self):
        """Create mock OCI models module."""
        models = MagicMock()
        models.CohereChatRequest = MagicMock
        models.CohereUserMessage = MagicMock
        models.CohereChatBotMessage = MagicMock
        models.CohereSystemMessage = MagicMock
        models.CohereToolMessage = MagicMock
        models.CohereTool = MagicMock
        models.CohereParameterDefinition = MagicMock
        models.CohereToolResult = MagicMock
        models.CohereToolCall = MagicMock
        models.BaseChatRequest = MagicMock()
        models.BaseChatRequest.API_FORMAT_COHERE = "COHERE"
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
            from locus.models.providers.oci.models.cohere import CohereProvider

            return CohereProvider()

    def test_init_sets_oci_classes(self, provider):
        """Test that init sets all required OCI classes."""
        assert provider.oci_chat_request is not None
        assert provider.oci_user_message is not None
        assert provider.oci_chatbot_message is not None
        assert provider.oci_system_message is not None
        assert provider.oci_tool_message is not None
        assert provider.oci_tool is not None
        assert provider.oci_tool_param is not None
        assert provider.oci_tool_result is not None
        assert provider.oci_tool_call is not None

    def test_api_format(self, provider):
        """Test api_format property."""
        assert provider.api_format == "COHERE"

    def test_stop_sequence_key(self, provider):
        """Test stop_sequence_key property."""
        assert provider.stop_sequence_key == "stop_sequences"

    def test_build_request_basic(self, provider):
        """Test building a basic request."""
        provider.oci_chat_request = MagicMock(return_value=MagicMock())

        result = provider.build_request(
            messages=[],
            _current_message="Hello",
            _chat_history=[],
            max_tokens=1000,
            temperature=0.5,
        )

        provider.oci_chat_request.assert_called_once()
        call_kwargs = provider.oci_chat_request.call_args
        assert call_kwargs[1]["message"] == "Hello"
        assert call_kwargs[1]["max_tokens"] == 1000
        assert call_kwargs[1]["temperature"] == 0.5

    def test_build_request_with_tools(self, provider):
        """Test building request with tools."""
        mock_request = MagicMock()
        provider.oci_chat_request = MagicMock(return_value=mock_request)

        tools = [MagicMock()]
        result = provider.build_request(
            messages=[],
            tools=tools,
            _current_message="Use tool",
            _chat_history=[],
        )

        assert mock_request.tools == tools

    def test_build_request_with_tool_results(self, provider):
        """Test building request with tool results."""
        mock_request = MagicMock()
        provider.oci_chat_request = MagicMock(return_value=mock_request)

        tool_results = [MagicMock()]
        result = provider.build_request(
            messages=[],
            _current_message="Process results",
            _chat_history=[],
            _tool_results=tool_results,
        )

        assert mock_request.tool_results == tool_results

    def test_build_request_with_stop_sequences(self, provider):
        """Test building request with stop sequences."""
        mock_request = MagicMock()
        provider.oci_chat_request = MagicMock(return_value=mock_request)

        result = provider.build_request(
            messages=[],
            _current_message="Stop here",
            _chat_history=[],
            stop_sequences=["STOP"],
        )

        assert mock_request.stop_sequences == ["STOP"]

    def test_build_request_with_stop_alias(self, provider):
        """Test building request with 'stop' instead of 'stop_sequences'."""
        mock_request = MagicMock()
        provider.oci_chat_request = MagicMock(return_value=mock_request)

        result = provider.build_request(
            messages=[],
            _current_message="Stop here",
            _chat_history=[],
            stop=["END"],
        )

        assert mock_request.stop_sequences == ["END"]

    def test_convert_messages_system(self, provider):
        """Test converting system message."""
        provider.oci_system_message = MagicMock(return_value="sys_msg")

        messages = [Message(role=Role.SYSTEM, content="System prompt")]
        result = provider.convert_messages(messages)

        assert result["_current_message"] == ""
        assert len(result["_chat_history"]) == 1
        provider.oci_system_message.assert_called_with(message="System prompt")

    def test_convert_messages_user_last(self, provider):
        """Test converting user message as last message."""
        messages = [Message(role=Role.USER, content="Hello")]
        result = provider.convert_messages(messages)

        assert result["_current_message"] == "Hello"
        assert result["_chat_history"] == []

    def test_convert_messages_user_in_history(self, provider):
        """Test converting user message in history."""
        provider.oci_user_message = MagicMock(return_value="user_msg")

        messages = [
            Message(role=Role.USER, content="First"),
            Message(role=Role.USER, content="Second"),
        ]
        result = provider.convert_messages(messages)

        assert result["_current_message"] == "Second"
        assert len(result["_chat_history"]) == 1
        provider.oci_user_message.assert_called_with(message="First")

    def test_convert_messages_assistant(self, provider):
        """Test converting assistant message."""
        provider.oci_chatbot_message = MagicMock(return_value="bot_msg")

        messages = [
            Message(role=Role.ASSISTANT, content="Response"),
            Message(role=Role.USER, content="Next"),
        ]
        result = provider.convert_messages(messages)

        provider.oci_chatbot_message.assert_called_with(message="Response", tool_calls=None)

    def test_convert_messages_assistant_with_tool_calls(self, provider):
        """Test converting assistant message with tool calls."""
        provider.oci_chatbot_message = MagicMock(return_value="bot_msg")
        provider.oci_tool_call = MagicMock(return_value="tool_call")

        tool_call = ToolCall(id="1", name="search", arguments={"q": "test"})
        messages = [
            Message(role=Role.ASSISTANT, content="Calling tool", tool_calls=[tool_call]),
            Message(role=Role.USER, content="Next"),
        ]
        result = provider.convert_messages(messages)

        provider.oci_tool_call.assert_called_with(name="search", parameters={"q": "test"})

    def test_convert_messages_tool(self, provider):
        """Test converting tool result message."""
        provider.oci_tool_result = MagicMock(return_value="tool_result")
        provider.oci_tool_call = MagicMock(return_value="tool_call")

        messages = [
            Message(role=Role.TOOL, name="search", content="Result data"),
            Message(role=Role.USER, content="Next"),
        ]
        result = provider.convert_messages(messages)

        assert result["_tool_results"] is not None
        assert len(result["_tool_results"]) == 1

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
        provider.oci_tool = MagicMock(return_value="oci_tool")
        provider.oci_tool_param = MagicMock(return_value="param")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "limit": {"type": "integer", "description": "Max results"},
                        },
                        "required": ["query"],
                    },
                },
            }
        ]
        result = provider.convert_tools(tools)

        assert result is not None
        assert len(result) == 1
        provider.oci_tool.assert_called_once()

    def test_convert_tools_type_mapping(self, provider):
        """Test type mapping in convert_tools."""
        provider.oci_tool = MagicMock(return_value="oci_tool")
        provider.oci_tool_param = MagicMock(return_value="param")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test",
                    "parameters": {
                        "properties": {
                            "s": {"type": "string"},
                            "i": {"type": "integer"},
                            "n": {"type": "number"},
                            "b": {"type": "boolean"},
                            "a": {"type": "array"},
                            "o": {"type": "object"},
                        },
                    },
                },
            }
        ]
        result = provider.convert_tools(tools)

        # Check that oci_tool_param was called for each property
        assert provider.oci_tool_param.call_count == 6

    def test_parse_response_text_only(self, provider):
        """Test parsing response with text only."""
        mock_response = MagicMock()
        mock_response.data.chat_response.text = "Hello world"
        mock_response.data.chat_response.finish_reason = "COMPLETE"
        mock_response.data.chat_response.tool_calls = None

        content, tool_calls, stop_reason = provider.parse_response(mock_response)

        assert content == "Hello world"
        assert tool_calls == []
        assert stop_reason == "COMPLETE"

    def test_parse_response_with_tool_calls(self, provider):
        """Test parsing response with tool calls."""
        mock_tool_call = MagicMock()
        mock_tool_call.name = "search"
        mock_tool_call.parameters = {"q": "test"}

        mock_response = MagicMock()
        mock_response.data.chat_response.text = None
        mock_response.data.chat_response.finish_reason = "TOOL_CALL"
        mock_response.data.chat_response.tool_calls = [mock_tool_call]

        content, tool_calls, stop_reason = provider.parse_response(mock_response)

        assert content is None
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "search"
        assert tool_calls[0].arguments == {"q": "test"}

    def test_parse_stream_chunk_text(self, provider):
        """Test parsing text stream chunk."""
        event_data = {"text": "Hello"}

        content, tool_calls, is_done = provider.parse_stream_chunk(event_data)

        assert content == "Hello"
        assert tool_calls == []
        assert is_done is False

    def test_parse_stream_chunk_done(self, provider):
        """Test parsing done stream chunk."""
        event_data = {"text": "", "finishReason": "COMPLETE"}

        content, tool_calls, is_done = provider.parse_stream_chunk(event_data)

        assert is_done is True

    def test_parse_stream_chunk_with_tool_calls(self, provider):
        """Test parsing stream chunk with tool calls."""
        event_data = {
            "text": "",
            "toolCalls": [{"name": "search", "parameters": {"q": "test"}}],
        }

        content, tool_calls, is_done = provider.parse_stream_chunk(event_data)

        assert len(tool_calls) == 1
        assert tool_calls[0].name == "search"

    def test_parse_stream_chunk_tool_calls_json_string(self, provider):
        """Test parsing stream chunk with JSON string parameters."""
        event_data = {
            "text": "",
            "toolCalls": [{"name": "search", "parameters": '{"q": "test"}'}],
        }

        content, tool_calls, is_done = provider.parse_stream_chunk(event_data)

        assert tool_calls[0].arguments == {"q": "test"}

    def test_parse_stream_chunk_tool_calls_invalid_json(self, provider):
        """Test parsing stream chunk with invalid JSON parameters."""
        event_data = {
            "text": "",
            "toolCalls": [{"name": "search", "parameters": "not valid json"}],
        }

        content, tool_calls, is_done = provider.parse_stream_chunk(event_data)

        assert tool_calls[0].arguments == {}
