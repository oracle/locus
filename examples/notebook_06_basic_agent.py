# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 06: payments triage assistant — your first Tulip agent.

Build LEDGER, the payments team's tier-1 transaction-triage agent, ask
it the question every fraud-ops analyst asks all day — "is this
transaction worth holding?" — two different ways (blocking and
streaming), and inspect what comes back. This is the smallest possible
end-to-end Tulip example, and the first appearance of an agent that
recurs across the later notebooks.

Key ideas:
- An ``Agent`` pairs a model with a system prompt and optional tools.
- ``agent.run_sync(prompt)`` returns a single ``AgentResult``.
- ``agent.run(prompt)`` is an async generator that yields events — the
  agent shows its work instead of handing you an opaque verdict.
- ``AgentResult`` carries the final message, success flag, stop reason,
  and per-run metrics.
- The same agent can triage many transactions in a row.

The sample transactions map to common payments patterns — card testing
(many small declines then a charge), an authorized recurring
subscription, and a friendly-fraud chargeback.

Run it:
    .venv/bin/python examples/notebook_06_basic_agent.py

The default provider is the bundled mock model. Set TULIP_MODEL_PROVIDER=openai (or anthropic)
and the agent talks to a live model (e.g.
``openai:gpt-4o`` or ``anthropic:claude-sonnet-4-6``). Without a
config — or for offline runs — set ``TULIP_MODEL_PROVIDER=mock`` to use
the bundled deterministic model. OpenAI, Anthropic are also
supported provider strings.
"""

import asyncio

# Shared helper that builds a model from env vars (TULIP_MODEL_PROVIDER,
# TULIP_MODEL, etc.). See examples/config.py.
from config import get_model, print_config

from tulip.agent import Agent


# =============================================================================
# Part 1: build LEDGER and call it once
# =============================================================================


def example_create_agent():
    """Build LEDGER and run one tiny prompt to confirm the provider works."""
    print("=== Part 1: Creating an Agent ===\n")

    model = get_model(max_tokens=40)

    agent = Agent(
        model=model,
        system_prompt="You are LEDGER, the payments team's tier-1 transaction-triage agent. Be concise.",
    )

    print(f"Agent created with model: {type(model).__name__}")
    print(f"System prompt: {agent.system_prompt[:50]}...")

    import time as _t

    t0 = _t.perf_counter()
    smoke = agent.run_sync("Say 'ready' in one word.")
    dt = _t.perf_counter() - t0
    print(
        f"  [provider call: {dt:.2f}s · "
        f"{smoke.metrics.prompt_tokens}→{smoke.metrics.completion_tokens} tokens]"
    )
    print(f"  Smoke reply: {smoke.message.strip()}")
    print()

    return agent


# =============================================================================
# Part 2: blocking call with run_sync
# =============================================================================


def example_sync_run():
    """Block until the agent finishes — simplest possible triage call."""
    print("=== Part 2: Synchronous Execution ===\n")

    model = get_model(max_tokens=100)

    agent = Agent(
        model=model,
        system_prompt="You are LEDGER, a payments triage agent. Keep responses under 20 words.",
    )

    # Several small declines then a charge — classic card-testing pattern.
    txn = "Alert: 5 declined $1.00 auths then a $420 capture on card ****7788. Hold?"
    result = agent.run_sync(txn)

    print(f"Prompt: {txn}")
    print(f"Response: {result.message}")
    print(f"Success: {result.success}")
    print(f"Stop reason: {result.stop_reason}")
    print()


# =============================================================================
# Part 3: async call with streaming events
# =============================================================================


async def example_async_run():
    """Stream the agent's lifecycle events as it works the transaction."""
    print("=== Part 3: Async Execution with Events ===\n")

    model = get_model(max_tokens=100)

    agent = Agent(
        model=model,
        system_prompt="You are LEDGER, a payments triage agent. Be brief.",
    )

    print("Prompt: Name 3 signs that a declined-card alert is a false positive.")
    print("Events:")

    # agent.run(...) yields ThinkEvent, ToolStartEvent, ToolCompleteEvent,
    # TerminateEvent, etc., in order. Notebook 11 covers the full event set.
    async for event in agent.run("Name 3 signs that a declined-card alert is a false positive."):
        print(f"  {event.event_type}: ", end="")
        if hasattr(event, "reasoning") and event.reasoning:
            print(f"{event.reasoning[:60]}...")
        elif hasattr(event, "final_message") and event.final_message:
            print(f"Final: {event.final_message[:60]}...")
        else:
            print(f"{event}")

    print()


# =============================================================================
# Part 4: what's inside AgentResult
# =============================================================================


def example_agent_result():
    """Print every notable field on AgentResult so you know what's available."""
    print("=== Part 4: Understanding Results ===\n")

    model = get_model(max_tokens=50)

    agent = Agent(
        model=model,
        system_prompt="You are LEDGER, a payments analyst. One sentence answers only.",
    )

    result = agent.run_sync("Is a recurring monthly charge from a known subscription suspicious?")

    print("AgentResult fields:")
    print(f"  .message     = {result.message}")
    print(f"  .success     = {result.success}")
    print(f"  .stop_reason = {result.stop_reason}")
    print(f"  .confidence  = {result.confidence}")

    print("\nMetrics:")
    print(f"  .metrics.iterations  = {result.metrics.iterations}")
    print(f"  .metrics.tool_calls  = {result.metrics.tool_calls}")
    print(f"  .metrics.duration_ms = {result.metrics.duration_ms:.0f}")
    print()


# =============================================================================
# Part 5: reuse the same agent across transactions
# =============================================================================


def example_multiple_prompts():
    """One agent, many transactions. Each call is independent unless you opt in to memory."""
    print("=== Part 5: Multiple Transactions ===\n")

    model = get_model(max_tokens=50)

    agent = Agent(
        model=model,
        system_prompt="You are LEDGER, a payments triage agent. Reply in one line: hold or release.",
    )

    # A card-testing burst, an authorized recurring subscription, and a
    # friendly-fraud chargeback — the bread-and-butter of a tier-1 queue.
    prompts = [
        "Triage: 5 declined $1.00 auths then a $420 capture on card ****7788.",  # card testing
        "Triage: $9.99 recurring charge from a customer's active subscription.",
        "Triage: chargeback filed on a $180 order the customer already received.",
    ]

    for prompt in prompts:
        result = agent.run_sync(prompt)
        print(f"Q: {prompt}")
        print(f"A: {result.message}")
        print()


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all notebook parts."""
    print("=" * 60)
    print("Notebook 06: LEDGER — Payments Triage Assistant")
    print("=" * 60)
    print()

    print_config()
    print()

    example_create_agent()
    example_sync_run()
    asyncio.run(example_async_run())
    example_agent_result()
    example_multiple_prompts()

    print("=" * 60)
    print("Next: Notebook 07 — Transaction Enrichment with Tools")
    print("=" * 60)


if __name__ == "__main__":
    main()
