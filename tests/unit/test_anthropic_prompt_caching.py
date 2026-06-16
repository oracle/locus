"""Unit tests for Anthropic prompt-caching wiring.

Verifies that with ``prompt_cache=True`` on ``AnthropicModel``:
1. The system prompt is sent as a block list with ``cache_control: ephemeral``.
2. The last tool entry carries ``cache_control: ephemeral`` (caches the catalog).
3. Cache token counts on the response (``cache_creation_input_tokens`` /
   ``cache_read_input_tokens``) flow into ``usage`` and propagate up to
   ``ExecutionMetrics``.

The Anthropic SDK is mocked so we can inspect the request params we pass
to ``client.messages.create()`` without a real API call.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


pytest.importorskip("anthropic")

from locus.core.messages import Message
from locus.models.native.anthropic import AnthropicModel


def _make_response_with_usage(
    *,
    input_tokens: int = 100,
    output_tokens: int = 20,
    cache_creation: int | None = None,
    cache_read: int | None = None,
):
    """Build a fake Anthropic response with the requested usage shape."""
    from types import SimpleNamespace

    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    if cache_creation is not None:
        usage.cache_creation_input_tokens = cache_creation
    if cache_read is not None:
        usage.cache_read_input_tokens = cache_read

    text_block = SimpleNamespace(type="text", text="hi")
    return SimpleNamespace(
        content=[text_block],
        usage=usage,
        stop_reason="end_turn",
    )


def _build_model_with_mocked_client(*, prompt_cache: bool):
    model = AnthropicModel(
        model="claude-sonnet-4-20250514",
        api_key="sk-test",
        prompt_cache=prompt_cache,
    )
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=_make_response_with_usage())
    model._client = mock_client
    return model, mock_client


def test_system_prompt_uses_cache_control_when_prompt_cache_enabled():
    import asyncio

    model, mock_client = _build_model_with_mocked_client(prompt_cache=True)

    asyncio.run(model.complete([Message.system("You are helpful."), Message.user("hi")]))

    call = mock_client.messages.create.call_args
    system_param = call.kwargs["system"]
    assert isinstance(system_param, list), "system should be a block list when caching"
    assert system_param[0]["type"] == "text"
    assert system_param[0]["text"] == "You are helpful."
    assert system_param[0]["cache_control"] == {"type": "ephemeral"}


def test_system_prompt_is_plain_string_when_caching_disabled():
    """Backward-compat: prompt_cache=False keeps system as a bare string."""
    import asyncio

    model, mock_client = _build_model_with_mocked_client(prompt_cache=False)

    asyncio.run(model.complete([Message.system("You are helpful."), Message.user("hi")]))

    call = mock_client.messages.create.call_args
    assert call.kwargs["system"] == "You are helpful."


def test_tool_catalog_gets_cache_control_when_caching_enabled():
    import asyncio

    model, mock_client = _build_model_with_mocked_client(prompt_cache=True)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "summarise",
                "description": "Summarise text.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    asyncio.run(model.complete([Message.user("hi")], tools=tools))

    call = mock_client.messages.create.call_args
    anthropic_tools = call.kwargs["tools"]
    assert len(anthropic_tools) == 2
    # Last tool carries cache_control; first does not.
    assert "cache_control" not in anthropic_tools[0]
    assert anthropic_tools[1]["cache_control"] == {"type": "ephemeral"}


def test_cache_tokens_surfaced_on_usage_dict():
    """When the response includes cache stats, they flow into ModelResponse.usage."""
    import asyncio

    model = AnthropicModel(model="claude-sonnet-4-20250514", api_key="sk-test", prompt_cache=True)
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(
        return_value=_make_response_with_usage(
            input_tokens=50,
            output_tokens=10,
            cache_creation=1000,
            cache_read=4500,
        )
    )
    model._client = mock_client

    response = asyncio.run(model.complete([Message.user("hi")]))

    assert response.usage["prompt_tokens"] == 50
    assert response.usage["completion_tokens"] == 10
    assert response.usage["cache_creation_input_tokens"] == 1000
    assert response.usage["cache_read_input_tokens"] == 4500


def test_no_cache_fields_when_anthropic_omits_them():
    """Old SDK versions / non-cache responses don't include cache fields."""
    import asyncio

    model = AnthropicModel(model="claude-sonnet-4-20250514", api_key="sk-test", prompt_cache=False)
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(
        return_value=_make_response_with_usage(input_tokens=50, output_tokens=10)
    )
    model._client = mock_client

    response = asyncio.run(model.complete([Message.user("hi")]))

    assert "cache_creation_input_tokens" not in response.usage
    assert "cache_read_input_tokens" not in response.usage


def test_execution_metrics_have_cache_fields():
    """ExecutionMetrics carries cache_creation_input_tokens / cache_read_input_tokens."""
    from locus.agent.result import ExecutionMetrics

    metrics = ExecutionMetrics(
        iterations=2,
        prompt_tokens=100,
        completion_tokens=20,
        cache_creation_input_tokens=500,
        cache_read_input_tokens=2000,
    )
    assert metrics.cache_creation_input_tokens == 500
    assert metrics.cache_read_input_tokens == 2000


def test_state_with_token_usage_accepts_cache_counts():
    """AgentState.with_token_usage accepts and accumulates cache counts."""
    from locus.core.state import AgentState

    state = AgentState()
    state = state.with_token_usage(
        prompt_tokens=100,
        completion_tokens=20,
        cache_creation_tokens=500,
        cache_read_tokens=2000,
    )
    state = state.with_token_usage(
        prompt_tokens=50,
        completion_tokens=10,
        cache_read_tokens=1500,
    )

    assert state.prompt_tokens_used == 150
    assert state.completion_tokens_used == 30
    assert state.cache_creation_tokens_used == 500
    assert state.cache_read_tokens_used == 3500
