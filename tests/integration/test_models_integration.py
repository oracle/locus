# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Integration tests for model providers - requires API keys."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from locus.core.messages import Message, ToolResult


# Skip all integration tests if not explicitly enabled
pytestmark = pytest.mark.integration


def load_local_config() -> dict:
    """Load local config if available."""
    config_path = Path(__file__).parent.parent.parent / "config.local.yaml"
    if config_path.exists():
        with config_path.open() as f:
            return yaml.safe_load(f) or {}
    return {}


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_tools():
    """Sample tools for testing tool calling."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name",
                        },
                    },
                    "required": ["location"],
                },
            },
        },
    ]


# =============================================================================
# OpenAI Integration Tests
# =============================================================================


@pytest.mark.requires_openai
class TestOpenAIIntegration:
    """Integration tests for OpenAI."""

    @pytest.fixture
    async def model(self):
        """Create OpenAI model with proper cleanup."""
        from locus.models.native.openai import OpenAIModel

        model = OpenAIModel(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            max_tokens=256,
        )
        yield model
        await model.close()

    @pytest.mark.asyncio
    async def test_simple_completion(self, model):
        """Test simple completion."""
        messages = [
            Message.system("You are a helpful assistant. Be brief."),
            Message.user("What is 2 + 2?"),
        ]

        response = await model.complete(messages)

        assert response.content is not None
        assert "4" in response.content
        assert response.usage["prompt_tokens"] > 0

    @pytest.mark.asyncio
    async def test_tool_calling(self, model, sample_tools):
        """Test tool calling."""
        messages = [
            Message.user("What's the weather in San Francisco?"),
        ]

        response = await model.complete(messages, tools=sample_tools)

        assert len(response.tool_calls) > 0
        assert response.tool_calls[0].name == "get_weather"

    @pytest.mark.asyncio
    async def test_tool_call_conversation(self, model, sample_tools):
        """Test multi-turn conversation with tool results."""
        # First turn: get tool call
        messages = [
            Message.user("What's the weather in Tokyo?"),
        ]

        response = await model.complete(messages, tools=sample_tools)
        assert len(response.tool_calls) > 0

        # Second turn: provide tool result
        tool_result = ToolResult(
            tool_call_id=response.tool_calls[0].id,
            name="get_weather",
            content="Sunny, 72°F",
        )

        messages.append(response.message)
        messages.append(Message.tool(tool_result))

        response2 = await model.complete(messages, tools=sample_tools)

        assert response2.content is not None
        assert "72" in response2.content or "sunny" in response2.content.lower()

    @pytest.mark.asyncio
    async def test_streaming(self, model):
        """Test streaming response."""
        messages = [
            Message.user("Say hello in 3 languages."),
        ]

        chunks = []
        async for chunk in model.stream(messages):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert any(c.done for c in chunks)


# =============================================================================
# OCI GenAI Integration Tests
# =============================================================================


def _get_oci_config() -> dict:
    """Get OCI config from environment variables only."""
    return {
        "profile_name": os.getenv("OCI_PROFILE"),
        "auth_type": os.getenv("OCI_AUTH_TYPE"),
        "endpoint": os.getenv("OCI_ENDPOINT"),
        "compartment": os.getenv("OCI_COMPARTMENT"),
        "gpt_model": os.getenv("OCI_GPT_MODEL") or os.getenv("OCI_MODEL_ID"),
    }


@pytest.mark.requires_oci
class TestOCIIntegration:
    """Integration tests for OCI GenAI."""

    @pytest.fixture
    def oci_config(self):
        """Get OCI config."""
        config = _get_oci_config()
        # Check required env vars
        if not config["profile_name"]:
            pytest.skip("OCI_PROFILE not set")
        if not config["endpoint"]:
            pytest.skip("OCI_ENDPOINT not set")
        return config

    @pytest.fixture
    def gpt_model(self, oci_config):
        """Create OCI model with GPT."""
        if not oci_config["gpt_model"]:
            pytest.skip("OCI_GPT_MODEL not set")

        from locus.models.providers.oci import OCIModel

        return OCIModel(
            model_id=oci_config["gpt_model"],
            profile_name=oci_config["profile_name"],
            auth_type=oci_config["auth_type"],
            service_endpoint=oci_config["endpoint"],
            compartment_id=oci_config["compartment"],
            max_tokens=256,
        )

    @pytest.mark.asyncio
    async def test_gpt_completion(self, gpt_model):
        """Test GPT completion."""
        messages = [
            Message.user("What is 2 + 2? Just the number."),
        ]

        response = await gpt_model.complete(messages)

        assert response.content is not None
        assert "4" in response.content

    @pytest.mark.asyncio
    async def test_gpt_streaming(self, gpt_model):
        """Test GPT streaming response."""
        messages = [
            Message.user("Say hello."),
        ]

        chunks = []
        async for chunk in gpt_model.stream(messages):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert any(c.done for c in chunks)


# =============================================================================
# Cross-Provider Tests
# =============================================================================


class TestModelRegistry:
    """Test model registry with real providers."""

    @pytest.mark.requires_openai
    @pytest.mark.asyncio
    async def test_get_openai_model(self):
        """Get OpenAI model from registry."""
        from locus.models import get_model

        model = get_model(f"openai:{os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}", max_tokens=256)
        try:
            response = await model.complete([Message.user("Hi!")])
            assert response.content is not None
        finally:
            await model.close()
