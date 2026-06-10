# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for model providers - requires API keys."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from tulip.core.messages import Message, ToolResult


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
        from tulip.models.native.openai import OpenAIModel

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
# Cross-Provider Tests
# =============================================================================


class TestModelRegistry:
    """Test model registry with real providers."""

    @pytest.mark.requires_openai
    @pytest.mark.asyncio
    async def test_get_openai_model(self):
        """Get OpenAI model from registry."""
        from tulip.models import get_model

        model = get_model(f"openai:{os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}", max_tokens=256)
        try:
            response = await model.complete([Message.user("Hi!")])
            assert response.content is not None
        finally:
            await model.close()
