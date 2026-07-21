# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Notebook 56: Model providers — one cloud-ops agent, swap models with a string.

The Gateway is the seam every AI request in the org flows through: one
cloud-operations agent codebase, any backing model. Tulip supports OpenAI
and Anthropic as first-class providers, and the same agent code runs
against either — only the model object changes, so a platform team can
re-platform a provider (for cost, latency, data residency, or an outage
failover) without rewriting a single runbook. The provider abstraction is
also a control point: routing, rate limits, and audit live at the Gateway,
not in each agent.

Provider matrix:

  OpenAI    — GPT-4o, o1, o3, gpt-5.x against the direct API.
  Anthropic — Claude models (opus / sonnet / haiku).
  Registry  — get_model("provider:model_name") returns the right client.

Run it
    # Default: the bundled mock model (set TULIP_MODEL_PROVIDER for a live provider)
    python examples/notebook_56_model_providers.py

    # Offline / no credentials:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_56_model_providers.py

    # Pin a specific model:
    TULIP_MODEL_PROVIDER=openai TULIP_MODEL_ID=gpt-4o python examples/notebook_56_model_providers.py
"""

import asyncio
import time

from config import get_model as get_configured_model

from tulip.agent import Agent
from tulip.models.registry import get_model, list_providers


async def _llm_call(
    prompt: str, *, system: str = "Reply in one short sentence.", max_tokens: int = 80
) -> str:
    """One model call with a timing/token banner — used by every part."""
    agent = Agent(model=get_configured_model(max_tokens=max_tokens), system_prompt=system)
    t0 = time.perf_counter()
    res = await agent.arun(prompt)
    dt = time.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · {res.metrics.prompt_tokens}→{res.metrics.completion_tokens} tokens]"
    )
    return res.message.strip()


# Part 1: list every provider the registry knows about.


async def example_providers():
    """List available model providers."""
    print("=== Available providers ===\n")

    providers = list_providers()
    print(f"Registered providers: {providers}")
    print(
        f"AI rationale: {await _llm_call('In one sentence, why does a cloud platform team want a model registry instead of hard-coding one provider?')}"
    )

    print("\nUsage:")
    print('  model = get_model("openai:gpt-4o")')
    print('  model = get_model("anthropic:claude-sonnet-4-6")')
    print()
    print("The provider prefix before the colon selects the client; the rest")
    print("is the model id that provider expects. See docs/concepts/models.md.")


# Part 2: instantiate each provider directly, without the registry.


async def example_direct():
    """Use providers directly without the registry."""
    print("\n=== Direct provider usage ===\n")
    print(
        f"AI rationale: {await _llm_call('In one sentence, when would a cloud operations team instantiate a model class directly instead of via the registry?')}"
    )

    print("OpenAI (direct API, requires OPENAI_API_KEY):")
    print("  from tulip.models import OpenAIModel")
    print('  model = OpenAIModel(model="gpt-4o")')

    print("\nAnthropic (requires ANTHROPIC_API_KEY):")
    print("  from tulip.models.native.anthropic import AnthropicModel")
    print('  model = AnthropicModel(model="claude-sonnet-4-6")')


async def example_live_call() -> None:
    """Run the cloud-ops agent on whichever provider the environment configures."""
    print("\n=== Live provider call ===\n")
    model = get_configured_model(max_tokens=80)
    agent = Agent(
        model=model,
        system_prompt="You are a concise cloud operations assistant. Reply with one short sentence.",
    )
    import time as _t

    t0 = _t.perf_counter()
    result = await agent.arun(
        "Name two reasons a cloud platform team routes every model through one Gateway "
        "rather than letting each agent call a provider directly."
    )
    dt = _t.perf_counter() - t0
    print(f"  Model class: {type(model).__name__}")
    print(f"  Reply:       {result.message.strip()}")
    print(
        f"  [model call:   {dt:.2f}s · {result.metrics.prompt_tokens}→{result.metrics.completion_tokens} tokens]"
    )


async def main() -> None:
    await example_providers()
    await example_direct()
    await example_live_call()


if __name__ == "__main__":
    asyncio.run(main())
