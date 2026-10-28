# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Notebook 33: call any non-R-series OCI GenAI model with ``OCIChatCompletionsModel``.

This walks through the default Oracle Cloud Infrastructure (OCI)
Generative AI client class. ``OCIChatCompletionsModel`` targets the
OpenAI-compatible endpoint at ``/openai/v1/chat/completions`` and is
the right choice for almost every agent you'll build on OCI.

Key concepts:

- One class covers OpenAI, Meta, Mistral, xAI, Gemini, and non-R-series
  Cohere — change ``model="..."`` and nothing else.
- Stateless on the wire: Locus owns the conversation history, hooks,
  tool calls, and memory. No Project OCID, no continuation tokens.
- Streaming arrives as ``ModelChunkEvent``\\s through ``model.stream(...)``.
- Structured output is native: pass a Pydantic class as
  ``output_schema=`` and ``result.parsed`` is a typed instance.
- New models OCI ships on this endpoint work without a Locus update.

For Responses-only models or server-side conversation state, see
``notebook_34_oci_responses``. For Cohere R-series, see
``notebook_01_oci_transports``.

Run it:
    export OCI_PROFILE=<your-profile>             # ~/.oci/config profile
    export OCI_REGION=us-chicago-1                # GenAI inference region
    export OCI_COMPARTMENT=ocid1.compartment.…    # GenAI access
    python examples/notebook_33_oci_openai_chat.py

The walkthrough uses ``meta.llama-3.3-70b-instruct`` and
``openai.gpt-5``; both are real OCI GenAI model ids.

Difficulty: Beginner / Intermediate
"""

from __future__ import annotations

import asyncio
import os
import sys

from pydantic import BaseModel

from locus.agent import Agent
from locus.core.events import ModelChunkEvent
from locus.core.messages import Message
from locus.models.providers.oci import OCIChatCompletionsModel
from locus.tools.decorator import tool


def _env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        sys.stderr.write(
            f"skipped: missing env var {name} — see prerequisites in the notebook docstring\n"
        )
        # 77 = skipped: scripts/run_notebooks.py reports SKIP, not FAIL.
        sys.exit(77)
    return val


REGION = os.environ.get("OCI_REGION", "us-chicago-1")
COMPARTMENT = os.environ.get("OCI_COMPARTMENT")


def _make_model(model_id: str) -> OCIChatCompletionsModel:
    """Same class, different model id per part."""
    return OCIChatCompletionsModel(
        model=model_id,
        profile=_env("OCI_PROFILE"),
        region=REGION,
        compartment_id=COMPARTMENT,
    )


# =============================================================================
# Part 1 — Basic completion
# =============================================================================


def part1_basic() -> None:
    """One-shot completion against an OCI-hosted model."""
    print("=== Part 1: basic completion ===\n")
    model = _make_model("meta.llama-3.3-70b-instruct")
    agent = Agent(model=model, system_prompt="Answer in one sentence.")
    result = agent.run_sync("What is the capital of Australia?")
    print(f"  reply: {result.message.strip()}\n")


# =============================================================================
# Part 2 — Streaming
# =============================================================================


async def part2_streaming() -> None:
    """Stream tokens from the model as they arrive."""
    print("=== Part 2: streaming ===\n")
    model = _make_model("openai.gpt-5")
    print("  streamed: ", end="", flush=True)
    chunks = 0
    async for event in model.stream(
        [Message.system("Be brief."), Message.user("List 3 fruits, comma-separated.")]
    ):
        if isinstance(event, ModelChunkEvent) and event.content:
            print(event.content, end="", flush=True)
            chunks += 1
    print(f"\n  ({chunks} chunks)\n")


# =============================================================================
# Part 3 — Tool calling end-to-end
# =============================================================================


@tool
def get_time_in_city(city: str) -> str:
    """Return the current local time for a city (fake — always 14:30)."""
    return f"It is 14:30 in {city}."


def part3_tool_calling() -> None:
    """Agent picks the tool, the runtime executes it, model summarizes."""
    print("=== Part 3: tool calling ===\n")
    model = _make_model("openai.gpt-5")
    agent = Agent(
        model=model,
        tools=[get_time_in_city],
        max_iterations=4,
        system_prompt="Use get_time_in_city to answer time questions.",
    )
    result = agent.run_sync("What time is it in Tokyo?")
    print(f"  reply:      {result.message.strip()}")
    print(f"  iterations: {result.metrics.iterations}")
    print(f"  tool_calls: {result.metrics.tool_calls}\n")


# =============================================================================
# Part 4 — Structured output
# =============================================================================


class Weather(BaseModel):
    city: str
    temperature_c: int
    condition: str


def part4_structured_output() -> None:
    """Bind the model output to a Pydantic schema. OCI forwards the JSON
    schema to the model's constrained-decoding mode and ``result.parsed``
    is a typed ``Weather`` instance — no manual JSON parsing."""
    from locus.agent import AgentConfig

    print("=== Part 4: structured output (Weather schema) ===\n")
    model = _make_model("openai.gpt-5")
    agent = Agent(
        config=AgentConfig(
            model=model,
            system_prompt="Return a weather report as JSON.",
            output_schema=Weather,
        )
    )
    result = agent.run_sync("Make up a sunny weather report for Lisbon.")
    parsed: Weather | None = result.parsed
    print(f"  parsed: {parsed!r}")
    print(f"  type:   {type(parsed).__name__}\n")


# =============================================================================
# Part 5 — Same class, different families
# =============================================================================


def part5_model_swap() -> None:
    """Same class against four different OCI model families — change one string."""
    print("=== Part 5: same class, swap families ===\n")
    families = [
        "openai.gpt-5",
        "meta.llama-3.3-70b-instruct",
        "xai.grok-4-fast-non-reasoning",
        "cohere.command-a-03-2025",
    ]
    for fam in families:
        try:
            model = _make_model(fam)
            agent = Agent(model=model, system_prompt="Answer in three words or fewer.")
            result = agent.run_sync("Largest planet?")
            print(f"  {fam:42s} → {result.message.strip()}")
        except Exception as e:  # noqa: BLE001 — model availability varies per tenancy
            print(f"  {fam:42s} → [unavailable: {type(e).__name__}]")
    print()


def main() -> None:
    print("=" * 70)
    print("Notebook 33 — OCIChatCompletionsModel: the default OCI GenAI client")
    print("=" * 70 + "\n")
    print(f"OCI_PROFILE     = {os.environ.get('OCI_PROFILE', '(unset)')}")
    print(f"OCI_REGION      = {REGION}")
    print(f"OCI_COMPARTMENT = {(COMPARTMENT or '(unset)')[:60]}...\n")

    part1_basic()
    asyncio.run(part2_streaming())
    part3_tool_calling()
    part4_structured_output()
    part5_model_swap()

    print("=" * 70)
    print("Next: notebook 34 — OCIResponsesModel and the Responses endpoint")
    print("=" * 70)


if __name__ == "__main__":
    main()
