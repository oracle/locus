# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Integration tests: ``LiteLLMModel`` against the real OCI Generative AI
Infrastructure inference endpoint.

These tests are auto-skipped when the OCI credential set is incomplete, so
``pytest tests/integration`` is safe to run on developer laptops without OCI
access and inside CI environments where these secrets aren't provisioned.

To run locally:

    export OCI_REGION="us-chicago-1"
    export OCI_USER="ocid1.user.oc1..xxx"
    export OCI_FINGERPRINT="aa:bb:cc:..."
    export OCI_TENANCY="ocid1.tenancy.oc1..xxx"
    export OCI_KEY_FILE="$HOME/.oci/keys/your_api_key.pem"
    export OCI_COMPARTMENT_ID="ocid1.compartment.oc1..xxx"
    pytest tests/integration/test_litellm_oci_integration.py -v

The tests intentionally cover one model per vendor family that LiteLLM
exposes on OCI, plus the streaming and tool-calling paths. Vendor- and model-
specific quirks (fixed-temperature for gpt-5, ``reasoning_tokens`` only for
reasoning models, etc.) are asserted only where they're deterministic.
"""

from __future__ import annotations

import os

import pytest


pytestmark = pytest.mark.integration

# All six OCI env vars must be present and non-empty.
_REQUIRED_ENV = (
    "OCI_REGION",
    "OCI_USER",
    "OCI_FINGERPRINT",
    "OCI_TENANCY",
    "OCI_KEY_FILE",
    "OCI_COMPARTMENT_ID",
)
_missing_env = [v for v in _REQUIRED_ENV if not os.environ.get(v)]
_skip_reason = (
    f"OCI integration env not configured (missing: {', '.join(_missing_env)})"
    if _missing_env
    else ""
)
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(bool(_missing_env), reason=_skip_reason),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def litellm_installed():
    """Skip the whole module if litellm isn't installed.

    LiteLLMModel imports ``litellm`` lazily; in environments where it's
    intentionally absent, skip rather than error.
    """
    pytest.importorskip("litellm")


@pytest.fixture
def cohere_chat(litellm_installed):
    from locus.models.native.litellm import LiteLLMModel

    return LiteLLMModel(model="oci/cohere.command-latest", max_tokens=60, temperature=0.2)


@pytest.fixture
def grok_chat(litellm_installed):
    from locus.models.native.litellm import LiteLLMModel

    return LiteLLMModel(model="oci/xai.grok-4.20", max_tokens=200, temperature=0.2)


@pytest.fixture
def gpt5_reasoning(litellm_installed):
    from locus.models.native.litellm import LiteLLMModel

    # gpt-5 family rejects temperature; LiteLLMModel handles that automatically.
    return LiteLLMModel(
        model="oci/openai.gpt-5-mini",
        max_tokens=400,
        reasoning_effort="medium",
    )


@pytest.fixture
def gemini_chat(litellm_installed):
    from locus.models.native.litellm import LiteLLMModel

    return LiteLLMModel(model="oci/google.gemini-2.5-flash", max_tokens=80, temperature=0.2)


# ---------------------------------------------------------------------------
# Per-vendor chat smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cohere_chat(cohere_chat):
    from locus.core.messages import Message

    resp = await cohere_chat.complete(
        messages=[Message.user("What is the capital of Japan? Answer with one word only.")],
    )
    assert resp.message.content
    assert "Tokyo" in resp.message.content
    assert resp.usage.get("prompt_tokens", 0) > 0
    assert resp.usage.get("completion_tokens", 0) > 0


@pytest.mark.asyncio
async def test_grok_chat(grok_chat):
    from locus.core.messages import Message

    resp = await grok_chat.complete(
        messages=[Message.user("What is the capital of Japan? Answer with one word only.")],
    )
    assert resp.message.content
    assert "Tokyo" in resp.message.content


@pytest.mark.asyncio
async def test_gemini_chat(gemini_chat):
    from locus.core.messages import Message

    resp = await gemini_chat.complete(
        messages=[Message.user("What is 2 + 2? Reply with just the number.")],
    )
    assert resp.message.content
    assert "4" in resp.message.content


# ---------------------------------------------------------------------------
# Tool calling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grok_tool_call(grok_chat):
    from locus.core.messages import Message

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
    resp = await grok_chat.complete(
        messages=[Message.user("What's the weather in Tokyo right now?")],
        tools=tools,
        tool_choice="auto",
    )
    assert resp.message.tool_calls, "Grok should issue a tool call for a weather query"
    tc = resp.message.tool_calls[0]
    assert tc.name == "get_weather"
    assert isinstance(tc.arguments, dict)
    assert "location" in tc.arguments


# ---------------------------------------------------------------------------
# Reasoning model + reasoning_tokens surfacing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gpt5_reasoning(gpt5_reasoning):
    from locus.core.messages import Message

    resp = await gpt5_reasoning.complete(
        messages=[Message.user("If 3x + 7 = 22, what is x? Brief answer.")],
    )
    # The wrapper must successfully call gpt-5-mini without sending
    # ``temperature`` (OCI's hosted deployment 400s otherwise).
    assert resp.message.content
    assert "5" in resp.message.content
    # reasoning_tokens isn't strictly guaranteed for every short question,
    # but the usage dict must at least exist and contain the standard fields.
    assert "completion_tokens" in resp.usage


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cohere_streaming(cohere_chat):
    from locus.core.messages import Message

    chunks = []
    terminal_count = 0
    async for ev in cohere_chat.stream(
        messages=[Message.user("List 3 primary colors, comma-separated, one line.")],
    ):
        if ev.content:
            chunks.append(ev.content)
        if ev.done:
            terminal_count += 1

    assert chunks, "Stream produced no content chunks"
    assert terminal_count == 1, "Expected exactly one terminal ModelChunkEvent"


# ---------------------------------------------------------------------------
# Multi-turn + system message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cohere_multi_turn(cohere_chat):
    from locus.core.messages import Message

    resp = await cohere_chat.complete(
        messages=[
            Message.system("Answer with a single integer and nothing else."),
            Message.user("What is 7 times 8?"),
        ],
    )
    assert resp.message.content
    assert "56" in resp.message.content


# ---------------------------------------------------------------------------
# Env-var auth: explicit assertion that NO oci_* params are passed,
# proving env-var resolution works end-to-end.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_env_var_auth_path(litellm_installed):
    """Construct LiteLLMModel with zero OCI auth kwargs and confirm the call
    succeeds purely from ``OCI_*`` environment variables."""
    from locus.core.messages import Message
    from locus.models.native.litellm import LiteLLMModel

    model = LiteLLMModel(model="oci/cohere.command-latest", max_tokens=20)
    # Sanity-check: completion_kwargs must NOT contain any oci_* keys.
    assert not any(k.startswith("oci_") for k in model._completion_kwargs), (
        "No oci_* fields should be forwarded when constructing without them — "
        "LiteLLM's own resolve_oci_credentials() must read from the env."
    )
    resp = await model.complete(messages=[Message.user("Say 'hi' and nothing else.")])
    assert resp.message.content


# ---------------------------------------------------------------------------
# OCI OpenAI-V1 transport (delegates to OCIChatCompletionsModel)
# ---------------------------------------------------------------------------

# The V1 transport requires an OCI config-file profile rather than env vars.
# Auto-skip when no usable profile is available.
_V1_PROFILE = os.environ.get("OCI_INTEGRATION_PROFILE")
_v1_skip = pytest.mark.skipif(
    not _V1_PROFILE,
    reason=(
        "OCI_INTEGRATION_PROFILE not set — the V1 OpenAI-compat transport "
        "needs an OCI config-file profile (e.g. OCI_INTEGRATION_PROFILE=DEFAULT)."
    ),
)


@_v1_skip
@pytest.mark.asyncio
async def test_v1_transport_chat():
    """``transport='oci_openai_v1'`` routes through OCI's
    ``/openai/v1/chat/completions`` shim via OCIChatCompletionsModel,
    not through LiteLLM's native provider."""
    from locus.core.messages import Message
    from locus.models.native.litellm import LiteLLMModel

    async with LiteLLMModel(
        model="openai.gpt-5-mini",  # bare id — V1 shim doesn't want "oci/" prefix
        transport="oci_openai_v1",
        oci_profile=_V1_PROFILE,
        oci_region=os.environ.get("OCI_REGION", "us-chicago-1"),
        max_tokens=200,
    ) as model:
        resp = await model.complete(
            messages=[Message.user("What is the capital of France? One word.")],
        )

    assert resp.message.content
    assert "Paris" in resp.message.content


@_v1_skip
@pytest.mark.asyncio
async def test_v1_transport_streaming():
    """Streaming through the V1 shim must yield content chunks and a
    terminal ``done=True`` event, exactly like the native path."""
    from locus.core.messages import Message
    from locus.models.native.litellm import LiteLLMModel

    chunks: list[str] = []
    terminal_count = 0
    async with LiteLLMModel(
        model="openai.gpt-5-mini",
        transport="oci_openai_v1",
        oci_profile=_V1_PROFILE,
        oci_region=os.environ.get("OCI_REGION", "us-chicago-1"),
        max_tokens=200,
    ) as model:
        async for ev in model.stream(
            messages=[Message.user("List 3 primary colors, comma-separated.")],
        ):
            if ev.content:
                chunks.append(ev.content)
            if ev.done:
                terminal_count += 1

    assert chunks, "V1 streaming produced no content chunks"
    assert terminal_count == 1


@_v1_skip
@pytest.mark.asyncio
async def test_v1_transport_tool_call():
    """Tool calling under the V1 shim — model schema is OpenAI-shape,
    so the same tool definition the native path uses must work here too."""
    from locus.core.messages import Message
    from locus.models.native.litellm import LiteLLMModel

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

    async with LiteLLMModel(
        model="openai.gpt-5-mini",
        transport="oci_openai_v1",
        oci_profile=_V1_PROFILE,
        oci_region=os.environ.get("OCI_REGION", "us-chicago-1"),
        max_tokens=400,
    ) as model:
        resp = await model.complete(
            messages=[Message.user("What's the weather in Tokyo right now?")],
            tools=tools,
            tool_choice="auto",
        )

    assert resp.message.tool_calls, "V1 shim should issue a tool call for a weather query"
    tc = resp.message.tool_calls[0]
    assert tc.name == "get_weather"
    assert isinstance(tc.arguments, dict)
    assert "location" in tc.arguments


# ---------------------------------------------------------------------------
# OCI OpenAI Responses API transport (delegates to OCIResponsesModel)
# ---------------------------------------------------------------------------

# The Responses transport requires (a) the same OCI profile gate, AND (b) a
# tenancy that's actually enabled for the Responses API. We additionally gate
# on ``OCI_RESPONSES_PROFILE`` so a tenancy missing Responses entitlement
# doesn't fail this test — OCI returns a 404 "Authorization failed or
# requested resource not found" rather than a clear 403.
_RESPONSES_PROFILE = os.environ.get("OCI_RESPONSES_PROFILE")
_responses_skip = pytest.mark.skipif(
    not _RESPONSES_PROFILE,
    reason=(
        "OCI_RESPONSES_PROFILE not set — the Responses transport needs a "
        "tenancy that's entitled for the OCI Responses API."
    ),
)


@_responses_skip
@pytest.mark.asyncio
async def test_responses_transport_chat():
    """``transport='oci_responses'`` routes through OCIResponsesModel
    against OCI's OpenAI Responses-compat shim at
    ``POST /openai/v1/responses``. Server-stateful by default; pass
    ``oci_responses_store=False`` for ZDR tenancies."""
    from locus.core.messages import Message
    from locus.models.native.litellm import LiteLLMModel

    async with LiteLLMModel(
        model="openai.gpt-5-mini",
        transport="oci_responses",
        oci_profile=_RESPONSES_PROFILE,
        oci_region=os.environ.get("OCI_REGION", "us-chicago-1"),
        max_tokens=120,
    ) as model:
        resp = await model.complete(
            messages=[Message.user("Capital of Germany? One word.")],
        )

    assert resp.message.content
    assert "Berlin" in resp.message.content


@_responses_skip
@pytest.mark.asyncio
async def test_responses_transport_zdr_mode():
    """Responses transport with ``store=False`` for ZDR tenancies — the
    server rejects ``previous_response_id`` so the agent sends the full
    history each turn but still targets the Responses endpoint."""
    from locus.core.messages import Message
    from locus.models.native.litellm import LiteLLMModel

    async with LiteLLMModel(
        model="openai.gpt-5-mini",
        transport="oci_responses",
        oci_profile=_RESPONSES_PROFILE,
        oci_region=os.environ.get("OCI_REGION", "us-chicago-1"),
        oci_responses_store=False,
        max_tokens=80,
    ) as model:
        resp = await model.complete(
            messages=[Message.user("What is 2 + 2? Reply with just the number.")],
        )

    assert resp.message.content
    assert "4" in resp.message.content
