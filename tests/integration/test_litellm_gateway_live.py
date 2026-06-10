# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Integration tests: drive a live LiteLLM AI Gateway end-to-end.

Mirrors what ``examples/notebook_71_litellm_gateway.py`` does, but
asserts on the responses so a CI run catches regressions in the
gateway integration before they hit users. Auto-skips when the
``LITELLM_GATEWAY_URL`` / ``LITELLM_GATEWAY_KEY`` env vars aren't set,
so ``pytest tests/integration`` is safe on developer laptops and on
forks.

Run locally::

    # 1. Bring up the sample gateway in another shell with whatever
    #    upstream provider credentials it is configured for.
    cd examples/litellm-gateway/
    export LITELLM_MASTER_KEY="$(openssl rand -hex 32)"
    docker compose up -d

    # 2. Run the test against the gateway.
    export LITELLM_GATEWAY_URL="http://localhost:4000"
    export LITELLM_GATEWAY_KEY="$LITELLM_MASTER_KEY"
    export LITELLM_GATEWAY_MODEL="gpt-4o-mini"   # alias from config.yaml
    pytest tests/integration/test_litellm_gateway_live.py -v

The CI workflow ``.github/workflows/_litellm_integration.yml`` (added
alongside this file) wires the same three env vars from GitHub
Secrets when present.
"""

from __future__ import annotations

import os

import pytest


_GATEWAY_URL = os.environ.get("LITELLM_GATEWAY_URL", "").rstrip("/")
_GATEWAY_KEY = os.environ.get("LITELLM_GATEWAY_KEY", "")
_GATEWAY_MODEL = os.environ.get("LITELLM_GATEWAY_MODEL", "gpt-4o-mini")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_GATEWAY_URL and _GATEWAY_KEY),
        reason=(
            "LITELLM_GATEWAY_URL / LITELLM_GATEWAY_KEY not set — bring up the "
            "sample gateway under examples/litellm-gateway/ and export the URL "
            "+ a virtual key. See docs/how-to/litellm-gateway.md."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Gateway health (no Tulip involvement — just HTTP)
# ---------------------------------------------------------------------------


def test_gateway_models_endpoint_lists_documented_alias():
    """The gateway's /v1/models must include the alias the test points
    at; otherwise the docs claim a model exists that doesn't."""
    import httpx

    resp = httpx.get(
        f"{_GATEWAY_URL}/v1/models",
        headers={"Authorization": f"Bearer {_GATEWAY_KEY}"},
        timeout=10.0,
    )
    resp.raise_for_status()
    aliases = {m["id"] for m in resp.json().get("data", [])}
    assert _GATEWAY_MODEL in aliases, (
        f"gateway at {_GATEWAY_URL} doesn't expose {_GATEWAY_MODEL!r}; "
        f"aliases present: {sorted(aliases)}"
    )


def test_gateway_rejects_unauthenticated_call():
    """Negative-path: without a Bearer token the gateway must NOT
    forward upstream. This catches "I forgot to set master_key" and
    open-gateway-by-accident regressions."""
    import httpx

    resp = httpx.post(
        f"{_GATEWAY_URL}/v1/chat/completions",
        json={"model": _GATEWAY_MODEL, "messages": [{"role": "user", "content": "hi"}]},
        timeout=10.0,
    )
    assert resp.status_code in (401, 403), (
        f"gateway accepted unauthenticated POST: status={resp.status_code} body={resp.text[:200]!r}"
    )


# ---------------------------------------------------------------------------
# End-to-end through Tulip's existing OpenAIModel
# ---------------------------------------------------------------------------


@pytest.fixture
def model():
    """Build the exact OpenAIModel the docs / notebook 71 instruct users
    to build. If this fixture stops working, the docs are wrong."""
    from tulip.models.native.openai import OpenAIModel

    return OpenAIModel(
        model=_GATEWAY_MODEL,
        api_key=_GATEWAY_KEY,
        base_url=_GATEWAY_URL,
        max_tokens=60,
        temperature=0.2,
    )


@pytest.mark.asyncio
async def test_basic_completion_via_gateway(model):
    """Tulip → gateway → upstream → response. The minimal happy path."""
    from tulip.core.messages import Message

    resp = await model.complete(
        messages=[Message.user("What is the capital of Japan? Reply with one word only.")],
    )
    assert resp.message.content
    assert "Tokyo" in resp.message.content
    # Usage must arrive populated — the gateway propagates upstream usage
    # back through to Tulip.
    assert resp.usage.get("prompt_tokens", 0) > 0
    assert resp.usage.get("completion_tokens", 0) > 0


@pytest.mark.asyncio
async def test_multi_turn_with_system_message(model):
    """System + user multi-turn must round-trip through the gateway
    unchanged."""
    from tulip.core.messages import Message

    resp = await model.complete(
        messages=[
            Message.system("Answer with a single integer and nothing else."),
            Message.user("What is 7 times 8?"),
        ],
    )
    assert resp.message.content
    assert "56" in resp.message.content


@pytest.mark.asyncio
async def test_streaming_via_gateway(model):
    """SSE end-to-end. The gateway re-serialises upstream events, so
    this is the regression test that catches a broken passthrough."""
    from tulip.core.events import ModelChunkEvent
    from tulip.core.messages import Message

    chunks: list[str] = []
    terminal = 0
    async for ev in model.stream(
        messages=[Message.user("List 3 primary colors, comma-separated, one line.")],
    ):
        if isinstance(ev, ModelChunkEvent):
            if ev.content:
                chunks.append(ev.content)
            if ev.done:
                terminal += 1

    assert chunks, "gateway returned no streamed content chunks"
    # Tulip's OpenAIModel may emit more than one ``done=True`` event on a
    # successful stream (final content delta + a trailing finish-reason
    # event); the contract is that at least one done event fires before
    # iteration ends.
    assert terminal >= 1, "expected at least one terminal ModelChunkEvent"


@pytest.mark.asyncio
async def test_tool_call_via_gateway(model):
    """Tool calling must work through the gateway. OpenAI-shape tool
    schema in, OpenAI-shape tool call out."""
    from tulip.core.messages import Message

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a location.",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]
    resp = await model.complete(
        messages=[Message.user("What's the weather in Tokyo right now?")],
        tools=tools,
        tool_choice="auto",
    )
    # Most chat models will issue the call given an explicit prompt.
    # If the configured alias points at a non-tool-capable model and
    # this assertion ever flakes, change LITELLM_GATEWAY_MODEL to a
    # tool-capable alias.
    assert resp.message.tool_calls, (
        "model issued no tool call — set LITELLM_GATEWAY_MODEL to a tool-capable alias"
    )
    tc = resp.message.tool_calls[0]
    assert tc.name == "get_weather"
    assert isinstance(tc.arguments, dict)
    assert "location" in tc.arguments


# ---------------------------------------------------------------------------
# Agent-loop end-to-end — mirrors the notebook
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_loop_via_gateway(model):
    """Build an Agent, run a prompt, assert the result — this is what
    notebook_71 does in production form. Confirms the entire wiring
    (Agent + OpenAIModel + gateway + upstream) holds together."""
    from tulip.agent import Agent

    agent = Agent(model=model, system_prompt="Reply with a single sentence.")
    result = agent.run_sync("Name one programming language.")

    assert result.message
    assert result.message.strip()
    # The agent succeeded — no error_type, success flag is True.
    assert getattr(result, "success", True)
