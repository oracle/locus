# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 21: Composing data-privacy agents into pipelines.

When privacy work decomposes cleanly into agent-shaped pieces — a
PII-discovery summary feeding a privacy-impact write-up, two reviewers
working the same processing activity in parallel — you don't need a full
StateGraph. The three pipeline classes here are batteries-included
composition primitives that take a list of Agent instances and
orchestrate them for you. Scenario: a Data Protection Impact Assessment
(DPIA) over a new customer-analytics dataset.

- SequentialPipeline — each agent's output becomes the next agent's input.
- ParallelPipeline — run agents concurrently, then merge their results.
- LoopAgent — run one agent repeatedly until a stop condition fires.
- Helpers sequential() / parallel() / loop() exist for one-liners.

Run it:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_21_composition.py

The default provider is the bundled mock model; set TULIP_MODEL_PROVIDER for a live provider.
Set TULIP_MODEL_PROVIDER=mock for offline runs. Pick a live provider with
TULIP_MODEL_ID=openai.gpt-4.1 (or meta.llama-3.3-70b-instruct, etc.).
"""

import asyncio

from config import get_model

from tulip.agent import (
    Agent,
    AgentConfig,
    LoopAgent,
    ParallelPipeline,
    SequentialPipeline,
)


# =============================================================================
# Part 1: Sequential — PII discovery summary then privacy-impact write-up
# =============================================================================


async def example_sequential():
    """PII analyst summarizes personal-data findings; report writer writes the DPIA prose."""
    print("=== Part 1: Sequential — PII summary then privacy-impact write-up ===\n")

    model = get_model()

    pii_analyst = Agent(
        config=AgentConfig(
            system_prompt=(
                "You are a PII-discovery analyst. From the data-inventory notes, list "
                "the 3 most significant personal-data exposures."
            ),
            max_iterations=3,
            model=model,
        )
    )
    report_writer = Agent(
        config=AgentConfig(
            system_prompt=(
                "You are a privacy-impact report writer. Take the personal-data "
                "summary and write a short DPIA paragraph for the data protection officer."
            ),
            max_iterations=3,
            model=model,
        )
    )

    pipeline = SequentialPipeline(agents=[pii_analyst, report_writer])
    result = await pipeline.run(
        "Inventory of the customer_analytics table: stores full names, email "
        "addresses, and precise geolocation; rows retained indefinitely with no "
        "documented retention policy; exported nightly to a third-party vendor."
    )

    print(f"Stage 1 (PII summary): {result.outputs[0][:100]}...")
    print(f"Stage 2 (DPIA write-up): {result.outputs[1][:100]}...")
    print(f"Duration: {result.duration_ms:.0f}ms")


# =============================================================================
# Part 2: Parallel — privacy risks vs safeguards in one call
# =============================================================================


async def example_parallel():
    """Two agents assess the same processing activity independently; the pipeline merges them."""
    print("\n=== Part 2: Parallel — privacy risks vs safeguards in one call ===\n")

    model = get_model()

    risk_assessor = Agent(
        config=AgentConfig(
            system_prompt="List 2 privacy risks this processing creates for data subjects. Be concise.",
            max_iterations=3,
            model=model,
        )
    )
    safeguard_planner = Agent(
        config=AgentConfig(
            system_prompt="List 2 safeguards that would reduce this privacy risk. Be concise.",
            max_iterations=3,
            model=model,
        )
    )

    pipeline = ParallelPipeline(agents=[risk_assessor, safeguard_planner])
    result = await pipeline.run(
        "Marketing team plans to enrich profiles with purchased third-party data"
    )

    print(f"Risks: {result.outputs[0][:100]}...")
    print(f"Safeguards: {result.outputs[1][:100]}...")
    print(f"Merged: {result.final_output[:150]}...")


# =============================================================================
# Part 3: Loop — iterate until APPROVED or max_loops
# =============================================================================


async def example_loop():
    """LoopAgent re-runs the same agent, feeding back the previous draft."""
    print("\n=== Part 3: Loop — iterate until APPROVED or max_loops ===\n")

    model = get_model()

    response_hardener = Agent(
        config=AgentConfig(
            system_prompt=(
                "You tighten data-subject access request (DSAR) response drafts. When "
                "the draft is ready for the data protection officer, include the word "
                "APPROVED at the end."
            ),
            max_iterations=3,
            model=model,
        )
    )

    loop = LoopAgent(
        agent=response_hardener,
        condition=lambda output: "APPROVED" in output.upper(),
        max_loops=3,
        loop_prompt="Tighten this DSAR response. Say APPROVED when ready:\n{previous_output}",
    )

    result = await loop.run(
        "Subject jane.doe@example.com requests all data we hold; confirm the records "
        "exported and the retention window applied."
    )
    print(f"Iterations: {len(result.outputs)}")
    print(f"Final: {result.final_output[:100]}...")


if __name__ == "__main__":
    asyncio.run(example_sequential())
    asyncio.run(example_parallel())
    asyncio.run(example_loop())
