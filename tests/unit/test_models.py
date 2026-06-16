# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for model providers (mocked)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from locus.core.messages import Message


# =============================================================================
# OpenAI Model Tests
# =============================================================================


class TestOpenAIModel:
    """Tests for OpenAIModel."""

    @pytest.fixture
    def mock_openai(self):
        """Mock the openai module."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            yield

    def test_init_default_config(self, mock_openai):
        """Initialize with default configuration."""
        from locus.models.native.openai import OpenAIModel

        model = OpenAIModel()

        assert model.config.model == "gpt-4o"
        assert model.config.max_tokens == 4096
        assert model.config.temperature == 0.7

    def test_init_custom_config(self, mock_openai):
        """Initialize with custom configuration."""
        from locus.models.native.openai import OpenAIModel

        model = OpenAIModel(
            model="gpt-4",
            max_tokens=2048,
            temperature=0.5,
            api_key="test-key",
            base_url="https://custom.api",
        )

        assert model.config.model == "gpt-4"
        assert model.config.max_tokens == 2048
        assert model.config.api_key == "test-key"
        assert model.config.base_url == "https://custom.api"

    def test_convert_messages(self, mock_openai):
        """Convert messages to OpenAI format."""
        from locus.models.native.openai import OpenAIModel

        model = OpenAIModel()
        messages = [
            Message.system("You are helpful."),
            Message.user("Hello!"),
        ]

        openai_msgs = model._convert_messages(messages)

        assert len(openai_msgs) == 2
        assert openai_msgs[0]["role"] == "system"
        assert openai_msgs[0]["content"] == "You are helpful."
        assert openai_msgs[1]["role"] == "user"

    def test_convert_tools(self, mock_openai):
        """Convert tools to proper OpenAI format."""
        from locus.models.native.openai import OpenAIModel

        model = OpenAIModel()
        tools = [
            {
                "name": "search",
                "description": "Search the web",
                "parameters": {"type": "object"},
            }
        ]

        openai_tools = model._convert_tools(tools)

        assert len(openai_tools) == 1
        assert openai_tools[0]["type"] == "function"
        assert openai_tools[0]["function"]["name"] == "search"

    @pytest.mark.asyncio
    async def test_complete(self, mock_openai):
        """Test complete method."""
        from locus.models.native.openai import OpenAIModel

        model = OpenAIModel()

        # Mock response
        mock_message = MagicMock()
        mock_message.content = "Hello there!"
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        model._client = mock_client

        messages = [Message.user("Hello!")]
        response = await model.complete(messages)

        assert response.content == "Hello there!"
        assert response.usage["prompt_tokens"] == 10
        assert response.stop_reason == "stop"

    @pytest.mark.asyncio
    async def test_complete_with_tool_calls(self, mock_openai):
        """Test complete with tool call response."""
        from locus.models.native.openai import OpenAIModel

        model = OpenAIModel()

        # Mock tool call
        mock_function = MagicMock()
        mock_function.name = "search"
        mock_function.arguments = '{"query": "test"}'

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function = mock_function

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        model._client = mock_client

        messages = [Message.user("Search for test")]
        response = await model.complete(messages)

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "search"
        assert response.tool_calls[0].id == "call_123"


# =============================================================================
# OCI Model Tests
# =============================================================================


class TestOCIModel:
    """Tests for OCIModel."""

    @pytest.fixture
    def mock_oci(self):
        """Mock the oci module."""
        mock_oci = MagicMock()
        mock_config = MagicMock()
        mock_config.from_file = MagicMock(return_value={"tenancy": "test-tenancy"})
        mock_oci.config = mock_config
        mock_oci.auth = MagicMock()
        mock_oci.generative_ai_inference = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "oci": mock_oci,
                "oci.config": mock_config,
                "oci.auth": mock_oci.auth,
                "oci.auth.signers": MagicMock(),
                "oci.generative_ai_inference": mock_oci.generative_ai_inference,
                "oci.generative_ai_inference.models": MagicMock(),
            },
        ):
            yield mock_oci

    def test_init_default_config(self, mock_oci):
        """Initialize with default configuration."""
        from locus.models.providers.oci import OCIModel

        model = OCIModel()

        assert model.config.model_id == "cohere.command-r-plus"
        assert model.config.profile_name == "DEFAULT"
        assert model.config.auth_type.value == "api_key"

    def test_init_custom_config(self, mock_oci):
        """Initialize with custom configuration."""
        from locus.models.providers.oci import OCIAuthType, OCIModel

        model = OCIModel(
            model_id="meta.llama-3-70b-instruct",
            profile_name="DEFAULT",
            auth_type="security_token",
            compartment_id="test-compartment",
        )

        assert model.config.model_id == "meta.llama-3-70b-instruct"
        assert model.config.profile_name == "DEFAULT"
        assert model.config.auth_type == OCIAuthType.SECURITY_TOKEN
        assert model.config.compartment_id == "test-compartment"


# =============================================================================
# Registry Tests
# =============================================================================


class TestModelRegistry:
    """Tests for model registry."""

    def test_get_model_invalid_format(self):
        """Get model with invalid format raises error."""
        from locus.models.registry import get_model

        with pytest.raises(ValueError, match="must be 'provider:model'"):
            get_model("invalid_no_colon")

    def test_get_model_unknown_provider(self):
        """Get model with unknown provider raises error."""
        from locus.models.registry import get_model

        with pytest.raises(ValueError, match="Unknown provider"):
            get_model("unknown:model")

    def test_list_providers(self):
        """List available providers."""
        from locus.models.registry import list_providers

        providers = list_providers()

        # Should have registered providers (depending on installed packages)
        assert isinstance(providers, list)

    def test_register_and_get_custom_provider(self):
        """Test registering and getting a custom provider."""
        from locus.models.registry import _PROVIDERS, get_model, register_provider

        # Create a mock model
        mock_model = MagicMock()

        def test_factory(model_id, **kwargs):
            mock_model.model_id = model_id
            mock_model.kwargs = kwargs
            return mock_model

        # Register custom provider
        register_provider("test_provider", test_factory)

        try:
            # Get model from custom provider
            result = get_model("test_provider:my-model", custom_arg="value")

            assert result is mock_model
            assert mock_model.model_id == "my-model"
            assert mock_model.kwargs == {"custom_arg": "value"}
        finally:
            # Clean up
            del _PROVIDERS["test_provider"]

    def test_get_model_openai(self):
        """Test getting OpenAI model through registry."""
        from locus.models.registry import get_model, list_providers

        if "openai" in list_providers():
            model = get_model("openai:gpt-4o")
            assert model is not None
        else:
            pytest.skip("OpenAI provider not available")

    def test_get_model_oci(self):
        """Test getting OCI model through registry."""
        from locus.models.registry import get_model, list_providers

        if "oci" in list_providers():
            model = get_model("oci:cohere.command-r-plus")
            assert model is not None
        else:
            pytest.skip("OCI provider not available")


# =============================================================================
# Anthropic Provider Tests
# =============================================================================


class TestAnthropicModel:
    """Tests for Anthropic model provider."""

    def test_create_model(self):
        """Create an Anthropic model with default config."""
        pytest.importorskip("anthropic")
        from locus.models.native.anthropic import AnthropicModel

        model = AnthropicModel(model="claude-sonnet-4-20250514", api_key="test-key")
        assert model.config.model == "claude-sonnet-4-20250514"
        assert model.config.api_key == "test-key"

    def test_convert_messages_extracts_system(self):
        """System message extracted separately for Anthropic API."""
        pytest.importorskip("anthropic")
        from locus.models.native.anthropic import AnthropicModel

        model = AnthropicModel(api_key="test")
        system, messages = model._convert_messages(
            [
                Message.system("You are helpful"),
                Message.user("Hello"),
            ]
        )
        assert system == "You are helpful"
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_convert_tools(self):
        """OpenAI-format tools converted to Anthropic format."""
        pytest.importorskip("anthropic")
        from locus.models.native.anthropic import AnthropicModel

        model = AnthropicModel(api_key="test")
        tools = model._convert_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search for info",
                        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                    },
                }
            ]
        )
        assert tools is not None
        assert tools[0]["name"] == "search"
        assert "input_schema" in tools[0]

    def test_registry_has_anthropic(self):
        """Anthropic provider registered in model registry."""
        pytest.importorskip("anthropic")
        from locus.models.registry import list_providers

        assert "anthropic" in list_providers()

    def test_structured_output_tool_translates_response_format(self):
        """``response_format`` becomes a synthetic ``respond_with_schema`` tool.

        Anthropic does not accept OpenAI's ``response_format``; the idiomatic
        equivalent is to declare a tool whose ``input_schema`` is the desired
        JSON schema and pin ``tool_choice`` to it. This guards that translation.
        """
        pytest.importorskip("anthropic")
        from pydantic import BaseModel

        from locus.core.structured import build_response_format
        from locus.models.native.anthropic import AnthropicModel

        class Reply(BaseModel):
            answer: str
            confidence: float

        model = AnthropicModel(api_key="test")
        rf = build_response_format(Reply, strict=True)
        tool = model._structured_output_tool(rf)

        assert tool["name"] == "respond_with_schema"
        assert "input_schema" in tool
        # The tool's ``input_schema`` must carry the field names from the model.
        properties = tool["input_schema"].get("properties", {})
        assert "answer" in properties
        assert "confidence" in properties

    async def test_structured_output_extracts_tool_args_as_content(self):
        """In structured mode the tool's args are surfaced as message content.

        Downstream ``parse_structured`` should be able to validate them just
        as it would a native ``response_format`` provider response.
        """
        pytest.importorskip("anthropic")
        from unittest.mock import AsyncMock, MagicMock

        from pydantic import BaseModel

        from locus.core.messages import Message
        from locus.core.structured import build_response_format, parse_structured
        from locus.models.native.anthropic import AnthropicModel

        class Reply(BaseModel):
            answer: str
            confidence: float

        # Build a fake Anthropic SDK response with a single ``tool_use`` block.
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "tu_1"
        tool_block.name = "respond_with_schema"
        tool_block.input = {"answer": "42", "confidence": 0.99}
        fake_response = MagicMock()
        fake_response.content = [tool_block]
        fake_response.usage.input_tokens = 10
        fake_response.usage.output_tokens = 5
        fake_response.stop_reason = "tool_use"

        model = AnthropicModel(api_key="test")
        # Inject a stub client so we don't hit the network.
        stub_client = MagicMock()
        stub_client.messages.create = AsyncMock(return_value=fake_response)
        model._client = stub_client

        result = await model.complete(
            [Message.user("What is the answer?")],
            tools=None,
            response_format=build_response_format(Reply, strict=True),
        )

        # The tool_use block's input must be JSON-dumped onto ``message.content``.
        assert result.message.content is not None
        parsed = parse_structured(result.message.content, Reply)
        assert parsed.parsed is not None
        assert parsed.parsed.answer == "42"
        assert parsed.parsed.confidence == 0.99
        # The synthetic tool must NOT bubble up as a tool_call to the agent loop.
        assert result.message.tool_calls == []
        # The Anthropic call must have shipped tool_choice pinned to our tool.
        sent_kwargs = stub_client.messages.create.call_args.kwargs
        assert sent_kwargs["tool_choice"] == {
            "type": "tool",
            "name": "respond_with_schema",
        }
        assert any(t["name"] == "respond_with_schema" for t in sent_kwargs["tools"])


# =============================================================================
# Ollama Provider Tests
# =============================================================================


class TestOllamaModel:
    """Tests for Ollama model provider."""

    def test_create_model(self):
        """Create an Ollama model with default config."""
        pytest.importorskip("ollama")
        from locus.models.native.ollama import OllamaModel

        model = OllamaModel(model="llama3.3")
        assert model.config.model == "llama3.3"
        assert model.config.base_url == "http://localhost:11434"

    def test_convert_messages(self):
        """Messages converted to Ollama format."""
        pytest.importorskip("ollama")
        from locus.models.native.ollama import OllamaModel

        model = OllamaModel()
        messages = model._convert_messages(
            [
                Message.system("Be helpful"),
                Message.user("Hi"),
            ]
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_registry_has_ollama(self):
        """Ollama provider registered in model registry."""
        pytest.importorskip("ollama")
        from locus.models.registry import list_providers

        assert "ollama" in list_providers()
