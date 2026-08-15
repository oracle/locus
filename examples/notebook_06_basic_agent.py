# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 06: your first agent — build LEDGER, send it a transaction, read the result.

LEDGER is a tier-1 transaction-triage agent for a payments team. A
transaction arrives, LEDGER says what it looks like and what a human
should do about it. That is the whole job, and it is enough to show every
part of an ``Agent`` without a single tool, memory, or multi-agent hop.

Triage is the example rather than trivia because the shape is what
matters: a short structured input, a judgement, and a recommended action.
Swap payments for alerts, tickets, or log lines and nothing about the
code changes.

Key ideas:
- An ``Agent`` pairs a model with a system prompt and optional tools.
- ``agent.run_sync(prompt)`` returns one ``AgentResult`` — the blocking
  call, for scripts and notebooks.
- ``agent.run(prompt)`` is an async generator yielding events as they
  happen — the agent shows its work instead of handing back an opaque
  answer.
- ``AgentResult`` carries the final message, success flag, stop reason,
  and per-run metrics.
- The same agent triages many transactions in a row.

The sample transactions map to patterns a payments team actually sees:
card testing (a run of small declines, then one that authorizes), an
authorized recurring subscription, and a friendly-fraud chargeback. The
data is fictional throughout — invented merchant names, placeholder card
BINs, and made-up transaction ids.

Run it:
    .venv/bin/python examples/notebook_06_basic_agent.py

The default provider is the bundled deterministic mock model, so this
runs offline with no credentials. Set ``TULIP_MODEL_PROVIDER=openai`` (or
``anthropic``) and the matching key to send the same prompts to a live
model.
"""

import asyncio
import textwrap

# Shared helper that builds a model from env vars (TULIP_MODEL_PROVIDER,
# TULIP_MODEL_ID). See examples/config.py.
from config import get_model, print_config

from tulip.agent import Agent


# The one instruction every part of this notebook shares. Kept in one place
# because changing how LEDGER is briefed is the single highest-leverage edit
# in the file, and it should not have to be made five times.
LEDGER_BRIEF = (
    "You are LEDGER, a tier-1 transaction triage agent for a payments team. "
    "For each transaction, say in one or two sentences what pattern it looks "
    "like and what a human should do next. Be specific and be brief. Never "
    "state a certainty you do not have — say 'possible' or 'consistent with' "
    "when the evidence is suggestive rather than conclusive."
)

# --- The fictional transactions ------------------------------------------
# Written as short field dumps rather than prose so the model sees the same
# shape a real triage queue would hand it.

CARD_TESTING = textwrap.dedent("""\
    txn_id: txn_8841
    merchant: Northwind Coffee (online)
    amount: 1.00 USD (authorized)
    card_bin: 411111
    preceding_24h: 47 declines from the same BIN across 12 merchants,
                   amounts 0.50-2.00 USD
    """)

RECURRING_SUBSCRIPTION = textwrap.dedent("""\
    txn_id: txn_8842
    merchant: Cirrus Backup Co
    amount: 12.00 USD (authorized)
    card_bin: 552200
    history: same amount, same merchant, on the 14th of each of the last
             9 months, never disputed
    """)

FRIENDLY_FRAUD = textwrap.dedent("""\
    txn_id: txn_8843
    merchant: Harbourline Outfitters
    amount: 289.40 USD (settled 62 days ago)
    card_bin: 401288
    event: chargeback filed, reason code 10.4 "other fraud - card absent"
    delivery: signed for at the billing address; account has 3 prior
              undisputed orders to the same address
    """)


# =============================================================================
# Part 1: build LEDGER
# =============================================================================


def example_create_agent() -> Agent:
    """Build LEDGER and run one tiny prompt to confirm the provider works."""
    print("=== Part 1: Creating an Agent ===\n")

    model = get_model(max_tokens=40)
    agent = Agent(model=model, system_prompt=LEDGER_BRIEF)

    print(f"Agent created with model: {type(model).__name__}")
    print(f"System prompt: {agent.system_prompt[:60]}...")

    import time as _t

    started = _t.perf_counter()
    smoke = agent.run_sync("Reply with the single word: ready.")
    elapsed = _t.perf_counter() - started
    print(
        f"  [provider call: {elapsed:.2f}s · "
        f"{smoke.metrics.prompt_tokens}→{smoke.metrics.completion_tokens} tokens]"
    )
    print(f"  Smoke reply: {(smoke.message or '').strip()}")
    print()

    return agent


# =============================================================================
# Part 2: the blocking call — run_sync
# =============================================================================


def example_run_sync() -> None:
    """``run_sync`` waits for the whole run and hands back one result.

    This is the call to reach for in a script, a notebook, or a test. It is
    also the one that makes an agent feel like a function, which is most of
    why it is worth showing first.
    """
    print("=== Part 2: Blocking Execution (run_sync) ===\n")

    agent = Agent(model=get_model(max_tokens=120), system_prompt=LEDGER_BRIEF)

    result = agent.run_sync(f"Triage this transaction:\n\n{CARD_TESTING}")

    print("Transaction: txn_8841 (Northwind Coffee, $1.00)")
    print(f"LEDGER: {result.message}")
    print(f"Success: {result.success}")
    print(f"Stop reason: {result.stop_reason}")
    print()


# =============================================================================
# Part 3: the streaming call — run
# =============================================================================


async def example_run_streaming() -> None:
    """``run`` yields events as the agent works.

    Same agent, same kind of input, different shape of answer: instead of
    one result at the end, you see each step as it happens. That is what a
    UI needs, and what you want when a run is long enough that silence is
    indistinguishable from a hang.
    """
    print("=== Part 3: Streaming Execution (run) ===\n")

    agent = Agent(model=get_model(max_tokens=120), system_prompt=LEDGER_BRIEF)

    print("Transaction: txn_8843 (Harbourline Outfitters, $289.40 chargeback)")
    print("Events:")

    # run(...) yields ThinkEvent, ToolStartEvent, ToolCompleteEvent,
    # TerminateEvent and friends, in order. Notebook 11 covers the full set.
    async for event in agent.run(f"Triage this transaction:\n\n{FRIENDLY_FRAUD}"):
        detail = getattr(event, "reasoning", None) or getattr(event, "final_message", None)
        if detail:
            print(f"  {event.event_type}: {detail.strip()[:70]}")
        else:
            print(f"  {event.event_type}")

    print()


# =============================================================================
# Part 4: what's inside AgentResult
# =============================================================================


def example_agent_result() -> None:
    """Print every notable field on AgentResult so you know what's available."""
    print("=== Part 4: Understanding Results ===\n")

    agent = Agent(model=get_model(max_tokens=80), system_prompt=LEDGER_BRIEF)

    result = agent.run_sync(f"Triage this transaction:\n\n{RECURRING_SUBSCRIPTION}")

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
# Part 5: one agent, a queue of transactions
# =============================================================================


def example_a_triage_queue() -> None:
    """One agent, three transactions.

    Each call is independent: LEDGER does not remember txn_8841 while
    looking at txn_8842. That is the default on purpose — carrying context
    between unrelated transactions is how one flagged card starts colouring
    the judgement on the next. Notebook 08 adds memory when you want it.
    """
    print("=== Part 5: A Triage Queue ===\n")

    agent = Agent(model=get_model(max_tokens=80), system_prompt=LEDGER_BRIEF)

    queue = [
        ("txn_8841 — card testing", CARD_TESTING),
        ("txn_8842 — recurring subscription", RECURRING_SUBSCRIPTION),
        ("txn_8843 — chargeback", FRIENDLY_FRAUD),
    ]

    for label, transaction in queue:
        result = agent.run_sync(f"Triage this transaction:\n\n{transaction}")
        print(f"▸ {label}")
        print(f"  {result.message}")
        print()


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    """Run all notebook parts."""
    print("=" * 60)
    print("Notebook 06: LEDGER — Your First Agent")
    print("=" * 60)
    print()

    print_config()
    print()

    example_create_agent()
    example_run_sync()
    await example_run_streaming()
    example_agent_result()
    example_a_triage_queue()

    print("=" * 60)
    print("Next: Notebook 07 — Giving an Agent Tools")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
