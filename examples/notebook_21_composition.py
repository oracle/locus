# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 21: Composing security agents into pipelines.

When security work decomposes cleanly into agent-shaped pieces — a
recon summary feeding an exposure-validation report, two assessors
working the same finding in parallel — you don't need a full
StateGraph. The three pipeline classes here are batteries-included
composition primitives that take a list of Agent instances and
orchestrate them for you. Scenario: external attack-surface review
(MITRE ATT&CK T1595, Active Scanning).

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
# Part 1: Sequential — recon summary then exposure-validation report
# =============================================================================


async def example_sequential():
    """Recon analyst summarizes exposures; report writer validates them in prose."""
    print("=== Part 1: Sequential — recon summary then validation report ===\n")

    model = get_model()

    recon_analyst = Agent(
        config=AgentConfig(
            system_prompt=(
                "You are a recon analyst. From the scan notes, list the 3 most "
                "significant external exposures."
            ),
            max_iterations=3,
            model=model,
        )
    )
    report_writer = Agent(
        config=AgentConfig(
            system_prompt=(
                "You are an exposure-validation report writer. Take the exposure "
                "summary and write a short validation paragraph for the asset owner."
            ),
            max_iterations=3,
            model=model,
        )
    )

    pipeline = SequentialPipeline(agents=[recon_analyst, report_writer])
    result = await pipeline.run(
        "External scan of example.com: ports 22/80/443 open, stale subdomain "
        "old.example.com still resolving, TLS cert expires in 9 days."
    )

    print(f"Stage 1 (Recon summary): {result.outputs[0][:100]}...")
    print(f"Stage 2 (Validation report): {result.outputs[1][:100]}...")
    print(f"Duration: {result.duration_ms:.0f}ms")


# =============================================================================
# Part 2: Parallel — risks vs mitigations in one call
# =============================================================================


async def example_parallel():
    """Two agents assess the same exposure independently; the pipeline merges them."""
    print("\n=== Part 2: Parallel — risks vs mitigations in one call ===\n")

    model = get_model()

    risk_assessor = Agent(
        config=AgentConfig(
            system_prompt="List 2 risks this exposure creates for the business. Be concise.",
            max_iterations=3,
            model=model,
        )
    )
    mitigation_planner = Agent(
        config=AgentConfig(
            system_prompt="List 2 mitigations that would close this exposure. Be concise.",
            max_iterations=3,
            model=model,
        )
    )

    pipeline = ParallelPipeline(agents=[risk_assessor, mitigation_planner])
    result = await pipeline.run("Publicly exposed admin panel at admin.example.com")

    print(f"Risks: {result.outputs[0][:100]}...")
    print(f"Mitigations: {result.outputs[1][:100]}...")
    print(f"Merged: {result.final_output[:150]}...")


# =============================================================================
# Part 3: Loop — iterate until APPROVED or max_loops
# =============================================================================


async def example_loop():
    """LoopAgent re-runs the same agent, feeding back the previous draft."""
    print("\n=== Part 3: Loop — iterate until APPROVED or max_loops ===\n")

    model = get_model()

    report_hardener = Agent(
        config=AgentConfig(
            system_prompt=(
                "You tighten incident-report drafts. When the draft is ready for "
                "the incident commander, include the word APPROVED at the end."
            ),
            max_iterations=3,
            model=model,
        )
    )

    loop = LoopAgent(
        agent=report_hardener,
        condition=lambda output: "APPROVED" in output.upper(),
        max_loops=3,
        loop_prompt="Tighten this incident report. Say APPROVED when ready:\n{previous_output}",
    )

    result = await loop.run(
        "Suspicious login on prod-web-01 from 198.51.100.23; password reset issued."
    )
    print(f"Iterations: {len(result.outputs)}")
    print(f"Final: {result.final_output[:100]}...")


if __name__ == "__main__":
    asyncio.run(example_sequential())
    asyncio.run(example_parallel())
    asyncio.run(example_loop())
