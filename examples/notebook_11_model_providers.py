# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Notebook 11: Model providers — one cloud-ops agent, swap models with a string.

The Gateway is the seam every AI request in the org flows through: one
cloud-operations agent codebase, any backing model. Locus supports OCI
Generative AI, OpenAI and Anthropic as first-class providers, and the same
agent code runs against any of them — only the model object changes, so a platform team can
re-platform a provider (for cost, latency, data residency, or an outage
failover) without rewriting a single runbook. The provider abstraction is
also a control point: routing, rate limits, and audit live at the Gateway,
not in each agent.

Provider matrix:

  OCI       — oci:<vendor>.<model>: the OCI Generative AI catalog, auto-routed.
  OpenAI    — GPT-4o, o3, gpt-5.x against the direct API.
  Anthropic — Claude models (fable / opus / sonnet / haiku).
  Registry  — get_model("provider:model_name") returns the right client.

Run it:
    # Default: OCI GenAI if ~/.oci/config has a usable profile, else the bundled mock model
    python examples/notebook_11_model_providers.py

    # Offline / no credentials:
    LOCUS_MODEL_PROVIDER=mock python examples/notebook_11_model_providers.py

    # Pin a specific model:
    LOCUS_MODEL_PROVIDER=openai LOCUS_MODEL_ID=gpt-4o python examples/notebook_11_model_providers.py
"""

import asyncio
import functools
import os

from config import get_model as get_configured_model
from config import print_config

from locus.agent import Agent
from locus.models.registry import list_providers


@functools.lru_cache(maxsize=1)
def _model():
    """One configured model client, shared by every part.

    Each part wants its own system prompt, so each builds a cheap Agent;
    the provider client underneath (credentials, HTTP pool) is built once.
    """
    return get_configured_model(max_tokens=80)


def _banner(result) -> None:
    m = result.metrics
    print(
        f"  [model call: {m.duration_ms / 1000.0:.2f}s · "
        f"{m.prompt_tokens}→{m.completion_tokens} tokens]"
    )


async def _llm_call(
    prompt: str,
    *,
    system: str = "You are a concise cloud operations assistant. Reply with one short sentence.",
    model_kwargs: dict | None = None,
) -> str:
    """One model call on the shared client, with a timing/token banner."""
    agent = Agent(model=_model(), system_prompt=system)
    res = await agent.arun(prompt, model_kwargs=model_kwargs)
    _banner(res)
    return res.message.strip()


# =============================================================================
# Part 1: the registry — every provider Locus knows about
# =============================================================================


async def example_providers():
    """List available model providers."""
    print("=== Part 1: Available providers ===\n")

    providers = list_providers()
    print(f"Registered providers: {providers}")

    print("\nUsage:")
    print('  model = get_model("oci:openai.gpt-5.5")')
    print('  model = get_model("openai:gpt-4o")')
    print('  model = get_model("anthropic:claude-sonnet-5-5")')
    print()
    print("The provider prefix before the colon selects the client; the rest")
    print("is the model id that provider expects. See docs/concepts/models.md.")
    print()


# =============================================================================
# Part 2: instantiate each provider class directly, without the registry
# =============================================================================


def _build_oci():
    from locus.models.providers.oci import OCIChatCompletionsModel

    return OCIChatCompletionsModel(
        model="openai.gpt-5.5", profile=os.environ.get("OCI_PROFILE", "DEFAULT")
    )


def _build_openai():
    from locus.models import OpenAIModel

    return OpenAIModel(model="gpt-4o")


def _build_anthropic():
    from locus.models.native.anthropic import AnthropicModel

    return AnthropicModel(model="claude-sonnet-5-5")


async def example_direct():
    """Use providers directly without the registry."""
    print("=== Part 2: Direct provider usage ===\n")

    # Constructing a client does not call the provider, so this runs
    # offline. A missing SDK extra or an unusable OCI profile surfaces here
    # as the class's own error; a missing API key surfaces on the first
    # request, so "ready" below means "built", not "authenticated".
    attempts = [
        ("OCI GenAI — OCIChatCompletionsModel (profile from OCI_PROFILE)", _build_oci),
        ("OpenAI — OpenAIModel (OPENAI_API_KEY read at call time)", _build_openai),
        ("Anthropic — AnthropicModel (ANTHROPIC_API_KEY read at call time)", _build_anthropic),
    ]
    for label, build in attempts:
        print(f"  {label}")
        try:
            model = build()
            print(f"    -> {type(model).__name__} ready")
        except Exception as e:  # noqa: BLE001 — the point is to show what each class needs
            print(f"    -> {type(e).__name__}: {str(e)[:70]}")
    print()


# =============================================================================
# Part 3: run the cloud-ops agent on whichever provider the environment picks
# =============================================================================


async def example_live_call() -> None:
    """Run the cloud-ops agent on whichever provider the environment configures."""
    print("=== Part 3: Live provider call ===\n")

    print(f"  Model class: {type(_model()).__name__}")
    reply = await _llm_call(
        "Name two reasons a cloud platform team routes every model through one Gateway "
        "rather than letting each agent call a provider directly."
    )
    print(f"  Reply:       {reply}")
    print()


# =============================================================================
# Part 4: request parameters — model config vs per-run model_kwargs
# =============================================================================


async def example_request_parameters() -> None:
    """Two places to set request parameters, and the difference matters.

    model config  — fixed for the model's lifetime (temperature, top_p,
                    penalties, extra_body), set when the class is built.
    model_kwargs  — per run, passed to ``agent.arun(...)``; wins over the
                    above for that run only, and is the only workable home
                    for anything that changes between runs, ``tool_choice``
                    above all. Any Chat Completions parameter is forwarded.
    """
    print("=== Part 4: Request parameters ===\n")

    prompt = "In one sentence, why pin temperature to 0 for runbook automation?"

    print("Model's own temperature:")
    default = await _llm_call(prompt)
    print(f"  {default}")

    print("\nSame agent, model_kwargs={'temperature': 0} for this run only:")
    pinned = await _llm_call(prompt, model_kwargs={"temperature": 0})
    print(f"  {pinned}")

    print("\nThe mock answers identically either way; on a live model the second")
    print("reply is the deterministic one. Nothing about the agent or its model changed.")
    print()


async def main() -> None:
    print("=" * 60)
    print("Notebook 11: Model Providers — Swap Models with a String")
    print("=" * 60)
    print()

    print_config()
    print()

    await example_providers()
    await example_direct()
    await example_live_call()
    await example_request_parameters()

    print("=" * 60)
    print("Next: Notebook 12 — Data-subject-request triage pipeline as a graph")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
