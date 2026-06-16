# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for models module init (lazy imports)."""

import pytest


class TestModelsLazyImports:
    """Tests for lazy imports in models module."""

    def test_import_base_classes(self):
        """Test importing base classes."""
        from locus.models import (
            ModelConfig,
            ModelProtocol,
            ModelResponse,
            RequestBuilder,
            ResponseParser,
        )

        assert ModelConfig is not None
        assert ModelProtocol is not None
        assert ModelResponse is not None
        assert RequestBuilder is not None
        assert ResponseParser is not None

    def test_import_registry_functions(self):
        """Test importing registry functions."""
        from locus.models import get_model, list_providers, register_provider

        assert callable(get_model)
        assert callable(list_providers)
        assert callable(register_provider)

    def test_lazy_import_openai_model(self):
        """Test lazy import of OpenAIModel."""
        from locus.models import OpenAIModel

        assert OpenAIModel is not None

    def test_lazy_import_openai_config(self):
        """Test lazy import of OpenAIConfig."""
        from locus.models import OpenAIConfig

        assert OpenAIConfig is not None

    def test_lazy_import_oci_model(self):
        """Test lazy import of OCIModel."""
        from locus.models import OCIModel

        assert OCIModel is not None

    def test_lazy_import_oci_config(self):
        """Test lazy import of OCIConfig."""
        from locus.models import OCIConfig

        assert OCIConfig is not None

    def test_lazy_import_oci_auth_type(self):
        """Test lazy import of OCIAuthType."""
        from locus.models import OCIAuthType

        assert OCIAuthType is not None

    def test_lazy_import_unknown_raises(self):
        """Test that unknown attribute raises AttributeError."""
        from locus import models

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = models.NonExistentClass


from locus.core.messages import Message
from locus.models import ModelConfig, ModelResponse


class TestModelResponse:
    """Tests for ModelResponse."""

    def test_create_response(self):
        """Test creating a model response."""
        message = Message.assistant("Hello!")
        response = ModelResponse(
            message=message,
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            stop_reason="stop",
        )

        assert response.message is message
        assert response.stop_reason == "stop"

    def test_content_property(self):
        """Test content property returns message content."""
        response = ModelResponse(
            message=Message.assistant("Test content"),
        )
        assert response.content == "Test content"

    def test_content_property_none(self):
        """Test content property with no content."""
        response = ModelResponse(
            message=Message.assistant(None),
        )
        assert response.content is None

    def test_tool_calls_property(self):
        """Test tool_calls property returns message tool calls."""
        from locus.core.messages import ToolCall

        tool_call = ToolCall(id="tc1", name="search", arguments={"q": "test"})
        message = Message.assistant("", tool_calls=[tool_call])
        response = ModelResponse(message=message)

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "search"

    def test_prompt_tokens(self):
        """Test prompt_tokens property."""
        response = ModelResponse(
            message=Message.assistant("Hi"),
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )
        assert response.prompt_tokens == 100

    def test_prompt_tokens_default(self):
        """Test prompt_tokens default when not in usage."""
        response = ModelResponse(
            message=Message.assistant("Hi"),
            usage={},
        )
        assert response.prompt_tokens == 0

    def test_completion_tokens(self):
        """Test completion_tokens property."""
        response = ModelResponse(
            message=Message.assistant("Hi"),
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )
        assert response.completion_tokens == 50

    def test_completion_tokens_default(self):
        """Test completion_tokens default when not in usage."""
        response = ModelResponse(
            message=Message.assistant("Hi"),
            usage={},
        )
        assert response.completion_tokens == 0

    def test_total_tokens(self):
        """Test total_tokens property."""
        response = ModelResponse(
            message=Message.assistant("Hi"),
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )
        assert response.total_tokens == 150

    def test_total_tokens_empty_usage(self):
        """Test total_tokens with empty usage."""
        response = ModelResponse(
            message=Message.assistant("Hi"),
        )
        assert response.total_tokens == 0


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_config(self):
        """Test creating config with defaults."""
        config = ModelConfig(model="gpt-4o")
        assert config.model == "gpt-4o"
        assert config.max_tokens == 4096
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.stop_sequences == []

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = ModelConfig(
            model="gpt-4",
            max_tokens=2048,
            temperature=0.5,
            top_p=0.95,
            stop_sequences=["END", "STOP"],
        )
        assert config.max_tokens == 2048
        assert config.temperature == 0.5
        assert config.top_p == 0.95
        assert config.stop_sequences == ["END", "STOP"]

    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed."""
        config = ModelConfig(
            model="gpt-4o",
            custom_field="value",
        )
        assert config.custom_field == "value"
