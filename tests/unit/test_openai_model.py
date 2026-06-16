# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for OpenAI model provider."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from locus.core.messages import Message
from locus.models.native.openai import OpenAIConfig, OpenAIModel


class TestOpenAIConfig:
    """Tests for OpenAIConfig."""

    def test_default_config(self):
        """Test creating config with defaults."""
        config = OpenAIConfig()
        assert config.model == "gpt-4o"
        assert config.max_tokens == 4096
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.api_key is None
        assert config.base_url is None

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = OpenAIConfig(
            model="gpt-4",
            max_tokens=2048,
            temperature=0.5,
            api_key="test-key",
            base_url="https://custom.api.com",
        )
        assert config.model == "gpt-4"
        assert config.max_tokens == 2048
        assert config.api_key == "test-key"
        assert config.base_url == "https://custom.api.com"

    def test_openai_specific_settings(self):
        """Test OpenAI-specific settings."""
        config = OpenAIConfig(
            frequency_penalty=0.5,
            presence_penalty=0.5,
            seed=42,
            stop_sequences=["STOP"],
        )
        assert config.frequency_penalty == 0.5
        assert config.presence_penalty == 0.5
        assert config.seed == 42
        assert config.stop_sequences == ["STOP"]


class TestOpenAIModelInit:
    """Tests for OpenAIModel initialization."""

    def test_create_model_default(self):
        """Test creating model with defaults."""
        model = OpenAIModel()
        assert model.config.model == "gpt-4o"
        assert model.config.max_tokens == 4096

    def test_create_model_custom(self):
        """Test creating model with custom settings."""
        model = OpenAIModel(
            model="gpt-4",
            api_key="test-key",
            max_tokens=2048,
            temperature=0.5,
        )
        assert model.config.model == "gpt-4"
        assert model.config.api_key == "test-key"
        assert model.config.max_tokens == 2048
        assert model.config.temperature == 0.5


class TestOpenAIModelConversions:
    """Tests for message and tool conversions."""

    @pytest.fixture
    def model(self):
        """Create a model for testing."""
        return OpenAIModel()

    def test_convert_messages(self, model):
        """Test converting messages to OpenAI format."""
        messages = [
            Message.user("Hello"),
            Message.assistant("Hi there!"),
        ]

        result = model._convert_messages(messages)

        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"
        assert result[1]["role"] == "assistant"

    def test_convert_tools_none(self, model):
        """Test converting None tools."""
        result = model._convert_tools(None)
        assert result is None

    def test_convert_tools_empty(self, model):
        """Test converting empty tools list."""
        result = model._convert_tools([])
        assert result is None

    def test_convert_tools_unwrapped(self, model):
        """Test converting unwrapped tool definitions."""
        tools = [{"name": "search", "description": "Search", "parameters": {}}]

        result = model._convert_tools(tools)

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "search"

    def test_convert_tools_already_wrapped(self, model):
        """Test converting already wrapped tool definitions."""
        tools = [{"type": "function", "function": {"name": "search"}}]

        result = model._convert_tools(tools)

        assert len(result) == 1
        assert result[0]["type"] == "function"


class TestOpenAIModelParseResponse:
    """Tests for parsing OpenAI responses."""

    @pytest.fixture
    def model(self):
        """Create a model for testing."""
        return OpenAIModel()

    def test_parse_simple_response(self, model):
        """Test parsing a simple text response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.model = "gpt-4o"

        result = model._parse_response(mock_response)

        assert result.message.content == "Hello!"
        assert result.message.tool_calls == []
        assert result.usage["prompt_tokens"] == 10

    def test_parse_response_with_tool_calls(self, model):
        """Test parsing response with tool calls."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_response.choices[0].finish_reason = "tool_calls"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.model = "gpt-4o"

        # Mock tool call
        mock_tc = MagicMock()
        mock_tc.id = "call_123"
        mock_tc.function.name = "search"
        mock_tc.function.arguments = '{"query": "test"}'
        mock_response.choices[0].message.tool_calls = [mock_tc]

        result = model._parse_response(mock_response)

        assert len(result.message.tool_calls) == 1
        assert result.message.tool_calls[0].name == "search"
        assert result.message.tool_calls[0].arguments == {"query": "test"}

    def test_parse_response_invalid_json_arguments(self, model):
        """Test parsing response with invalid JSON in tool arguments."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_response.choices[0].finish_reason = "tool_calls"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.model = "gpt-4o"

        mock_tc = MagicMock()
        mock_tc.id = "call_123"
        mock_tc.function.name = "search"
        mock_tc.function.arguments = "invalid json {"
        mock_response.choices[0].message.tool_calls = [mock_tc]

        result = model._parse_response(mock_response)

        # Should handle invalid JSON gracefully
        assert len(result.message.tool_calls) == 1
        assert result.message.tool_calls[0].arguments == {}


class TestOpenAIModelContextManager:
    """Tests for context manager functionality."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test using model as async context manager."""
        async with OpenAIModel() as model:
            assert model is not None

    @pytest.mark.asyncio
    async def test_close_without_client(self):
        """Test close when no client created."""
        model = OpenAIModel()
        await model.close()  # Should not raise


class TestOpenAIModelComplete:
    """Tests for complete method."""

    @pytest.fixture
    def model(self):
        """Create a model with mocked client."""
        model = OpenAIModel()
        return model

    @pytest.mark.asyncio
    async def test_complete_simple(self, model):
        """Test simple completion."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.model = "gpt-4o"

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Set _client directly to avoid triggering the property
        model._client = mock_client
        result = await model.complete([Message.user("Hi")])

        assert result.message.content == "Hello!"

    @pytest.mark.asyncio
    async def test_complete_with_tools(self, model):
        """Test completion with tools."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_response.choices[0].finish_reason = "tool_calls"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.model = "gpt-4o"

        mock_tc = MagicMock()
        mock_tc.id = "call_123"
        mock_tc.function.name = "search"
        mock_tc.function.arguments = '{"query": "test"}'
        mock_response.choices[0].message.tool_calls = [mock_tc]

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        model._client = mock_client
        tools = [{"name": "search", "description": "Search", "parameters": {}}]
        result = await model.complete([Message.user("Hi")], tools=tools)

        assert len(result.message.tool_calls) == 1
        assert result.message.tool_calls[0].name == "search"


class TestOpenAIModelStreaming:
    """Tests for streaming functionality."""

    @pytest.fixture
    def model(self):
        """Create a model for testing."""
        return OpenAIModel()

    @pytest.mark.asyncio
    async def test_stream_simple(self, model):
        """Test simple streaming completion."""
        # Create mock stream chunks
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = "Hello"
        mock_chunk1.choices[0].delta.tool_calls = None
        mock_chunk1.choices[0].finish_reason = None

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = " world!"
        mock_chunk2.choices[0].delta.tool_calls = None
        mock_chunk2.choices[0].finish_reason = None

        mock_chunk3 = MagicMock()
        mock_chunk3.choices = [MagicMock()]
        mock_chunk3.choices[0].delta.content = None
        mock_chunk3.choices[0].delta.tool_calls = None
        mock_chunk3.choices[0].finish_reason = "stop"

        async def mock_stream():
            for chunk in [mock_chunk1, mock_chunk2, mock_chunk3]:
                yield chunk

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        model._client = mock_client

        chunks = []
        async for chunk in model.stream([Message.user("Hi")]):
            chunks.append(chunk)

        assert len(chunks) >= 2
        assert chunks[0].content == "Hello"
        assert chunks[1].content == " world!"

    @pytest.mark.asyncio
    async def test_stream_with_tool_calls(self, model):
        """Test streaming with tool calls."""
        # Initial chunk
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = None
        mock_chunk1.choices[0].delta.tool_calls = None
        mock_chunk1.choices[0].finish_reason = None

        # Tool call chunk with ID and name
        mock_tc_delta1 = MagicMock()
        mock_tc_delta1.index = 0
        mock_tc_delta1.id = "call_123"
        mock_tc_delta1.function = MagicMock()
        mock_tc_delta1.function.name = "search"
        mock_tc_delta1.function.arguments = '{"query":'

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = None
        mock_chunk2.choices[0].delta.tool_calls = [mock_tc_delta1]
        mock_chunk2.choices[0].finish_reason = None

        # Tool call chunk with more arguments
        mock_tc_delta2 = MagicMock()
        mock_tc_delta2.index = 0
        mock_tc_delta2.id = None
        mock_tc_delta2.function = MagicMock()
        mock_tc_delta2.function.name = None
        mock_tc_delta2.function.arguments = ' "test"}'

        mock_chunk3 = MagicMock()
        mock_chunk3.choices = [MagicMock()]
        mock_chunk3.choices[0].delta.content = None
        mock_chunk3.choices[0].delta.tool_calls = [mock_tc_delta2]
        mock_chunk3.choices[0].finish_reason = None

        # Final chunk
        mock_chunk4 = MagicMock()
        mock_chunk4.choices = [MagicMock()]
        mock_chunk4.choices[0].delta.content = None
        mock_chunk4.choices[0].delta.tool_calls = None
        mock_chunk4.choices[0].finish_reason = "tool_calls"

        async def mock_stream():
            for chunk in [mock_chunk1, mock_chunk2, mock_chunk3, mock_chunk4]:
                yield chunk

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        model._client = mock_client

        chunks = []
        async for chunk in model.stream([Message.user("Hi")]):
            chunks.append(chunk)

        # Should have a chunk with tool calls at the end
        tool_chunk = next((c for c in chunks if c.tool_calls), None)
        assert tool_chunk is not None
        assert len(tool_chunk.tool_calls) == 1
        assert tool_chunk.tool_calls[0].name == "search"

    @pytest.mark.asyncio
    async def test_stream_empty_choices(self, model):
        """Test streaming handles empty choices gracefully."""
        mock_chunk_empty = MagicMock()
        mock_chunk_empty.choices = []

        mock_chunk_final = MagicMock()
        mock_chunk_final.choices = [MagicMock()]
        mock_chunk_final.choices[0].delta.content = "Done"
        mock_chunk_final.choices[0].delta.tool_calls = None
        mock_chunk_final.choices[0].finish_reason = "stop"

        async def mock_stream():
            for chunk in [mock_chunk_empty, mock_chunk_final]:
                yield chunk

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        model._client = mock_client

        chunks = []
        async for chunk in model.stream([Message.user("Hi")]):
            chunks.append(chunk)

        # Should have content chunk and done chunk (empty choices should be skipped)
        content_chunks = [c for c in chunks if c.content]
        assert len(content_chunks) == 1
        assert content_chunks[0].content == "Done"

    @pytest.mark.asyncio
    async def test_stream_o1_model(self, model):
        """Test streaming with o1 model uses max_completion_tokens."""
        model.config.model = "o1-preview"

        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = "Response"
        mock_chunk.choices[0].delta.tool_calls = None
        mock_chunk.choices[0].finish_reason = "stop"

        async def mock_stream():
            yield mock_chunk

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        model._client = mock_client

        chunks = []
        async for chunk in model.stream([Message.user("Hi")]):
            chunks.append(chunk)

        # Verify create was called with max_completion_tokens
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "max_completion_tokens" in call_kwargs
        assert "temperature" not in call_kwargs  # o1 models don't use temperature


class TestOpenAIModelToolMessage:
    """Tests for tool message conversion."""

    def test_convert_tool_message(self):
        """Test converting tool message to OpenAI format."""
        model = OpenAIModel()

        from locus.tools.executor import ToolResult

        tool_msg = Message.tool(
            ToolResult(
                tool_call_id="call_123",
                name="search",
                content="Search results here",
                error=None,
                duration_ms=100,
            )
        )

        messages = model._convert_messages([tool_msg])

        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert messages[0]["tool_call_id"] == "call_123"
        assert messages[0]["content"] == "Search results here"
