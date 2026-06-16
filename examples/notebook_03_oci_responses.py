#!/usr/bin/env python3
# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Notebook 03: reach Responses-only OCI GenAI models with ``OCIResponsesModel``.

``OCIResponsesModel`` is Locus's client for the Oracle Cloud
Infrastructure (OCI) Generative AI Responses endpoint
(``/openai/v1/responses``). Use it for two reasons:

1. To call Responses-only OCI models such as ``openai.gpt-5.5-pro`` —
   the OpenAI-compatible chat endpoint rejects these.
2. To let OCI hold the conversation thread. Locus then sends only the
   latest turn and references the prior response by id, keeping
   per-request payloads small.

Key concepts:

- ``store=False`` is the Zero Data Retention (ZDR) mode. OCI keeps
  nothing server-side; the agent sends the full history each turn. Use
  this in enterprise tenancies with ZDR enabled — you still unlock
  Responses-only models.
- ``store=True`` is the server-stateful mode. OCI persists each
  response and Locus passes the continuation id via
  ``provider_state["previous_response_id"]`` on the next turn.
- Tool calls round-trip as ``function_call`` items in the input and
  ``function_call_output`` items in the response. The agent loop's
  hooks (``on_before_tool_call``, idempotency, etc.) behave the same
  as with any other model.
- Streaming arrives as ``ModelChunkEvent`` from the SSE
  ``output_text.delta`` stream — agent-layer code doesn't change.
- ``ConversationManager`` is the one Locus primitive that bypasses on
  this path (no client-side history to trim). Memory, Reflexion, GSAR,
  grounding, idempotency, checkpointing, output schema, streaming, and
  termination conditions all apply.

Run it::

    export OCI_PROFILE=<your-profile>
    export OCI_REGION=us-chicago-1
    export OCI_COMPARTMENT=ocid1.compartment.…
    # Optional: also exercise store=True (skip if your tenant has ZDR).
    export OCI_RESPONSES_STORE=0
    python examples/notebook_03_oci_responses.py

Difficulty: Intermediate
"""

from __future__ import annotations

import asyncio
import os
import sys

from locus.agent import Agent
from locus.core.events import ModelChunkEvent, TerminateEvent
from locus.core.messages import Message
from locus.models.providers.oci import OCIResponsesModel
from locus.tools.decorator import tool


def _env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        sys.stderr.write(f"missing env var {name} — see prerequisites in the notebook docstring\n")
        sys.exit(2)
    return val


REGION = os.environ.get("OCI_REGION", "us-chicago-1")
COMPARTMENT = os.environ.get("OCI_COMPARTMENT")
STORE_ENABLED = os.environ.get("OCI_RESPONSES_STORE", "0") == "1"


def _make_model(*, store: bool) -> OCIResponsesModel:
    return OCIResponsesModel(
        # Any Responses-supported model id works; use gpt-5.5-pro for the Pro tier.
        model="openai.gpt-5",
        profile=_env("OCI_PROFILE"),
        region=REGION,
        compartment_id=COMPARTMENT,
        store=store,
    )


# =============================================================================
# Part 1 — Stateless / ZDR-friendly (store=False)
# =============================================================================


async def part1_stateless_multiturn() -> None:
    """Multi-turn against Responses with ``store=False``: the agent sends
    the full history each turn and OCI keeps nothing. Works in every
    tenancy, including those with Zero Data Retention enforced."""
    print("=== Part 1: stateless multi-turn (store=False, ZDR-safe) ===\n")
    model = _make_model(store=False)
    print(f"  server_stateful (instance): {model.server_stateful}  (False = stateless)\n")

    msgs = [
        Message.system("You remember user-shared facts and recall them on demand."),
        Message.user("My favorite color is purple. Just acknowledge."),
    ]
    r1 = await model.complete(msgs)
    print(f"  turn 1 reply: {(r1.message.content or '').strip()[:80]!r}")
    print(f"  turn 1 provider_state: {r1.provider_state}  (empty in stateless mode)\n")

    # Hand the model the full prior history including its own reply.
    msgs2 = [*msgs, r1.message, Message.user("What is my favorite color?")]
    r2 = await model.complete(msgs2)
    print(f"  turn 2 reply: {(r2.message.content or '').strip()[:120]!r}")
    print(f"  recalled 'purple' from history: {'purple' in (r2.message.content or '').lower()}\n")

    await model.aclose()


# =============================================================================
# Part 2 — Server-side state (store=True, default; non-ZDR only)
# =============================================================================


async def part2_server_state_multiturn() -> None:
    """Multi-turn with ``store=True``: OCI holds the thread and Locus
    threads ``previous_response_id`` between turns. Skipped unless
    ``OCI_RESPONSES_STORE=1`` (does not work in ZDR tenancies)."""
    print("=== Part 2: server-side state (store=True, non-ZDR) ===\n")
    if not STORE_ENABLED:
        print("  skipped — set OCI_RESPONSES_STORE=1 to run (non-ZDR tenancies only)\n")
        return

    model = _make_model(store=True)
    print(f"  server_stateful (instance): {model.server_stateful}  (True = stateful)\n")

    r1 = await model.complete(
        [
            Message.system("You remember user-shared facts and recall them on demand."),
            Message.user("My favorite color is purple. Just acknowledge."),
        ]
    )
    print(f"  turn 1 reply: {(r1.message.content or '').strip()[:80]!r}")
    print(f"  turn 1 provider_state: {r1.provider_state}\n")

    # Only the new user message goes on the wire; OCI joins it to the
    # prior thread via the continuation id in provider_state.
    r2 = await model.complete(
        [Message.user("What is my favorite color?")],
        provider_state=r1.provider_state,
    )
    print(f"  turn 2 reply: {(r2.message.content or '').strip()[:120]!r}")
    print(f"  turn 2 provider_state: {r2.provider_state}  (new continuation id)\n")

    await model.aclose()


# =============================================================================
# Part 3 — Tool round-trip
# =============================================================================


@tool
def get_temperature(city: str) -> str:
    """Return the current temperature for a city in Celsius (fake)."""
    return f"The temperature in {city} is 18 C."


async def part3_tool_roundtrip() -> None:
    """Full Agent + @tool flow through the Responses endpoint. The model
    emits a ``function_call`` item, Locus runs the tool client-side and
    posts back a ``function_call_output``, the model produces the
    final answer. Agent-layer code is identical to any other model."""
    print("=== Part 3: tool round-trip (store=False) ===\n")
    model = _make_model(store=False)
    agent = Agent(
        model=model,
        tools=[get_temperature],
        max_iterations=4,
        system_prompt="Use get_temperature to answer temperature questions.",
    )

    final: TerminateEvent | None = None
    async for ev in agent.run("What's the temperature in Paris?"):
        if isinstance(ev, TerminateEvent):
            final = ev
            break

    assert final is not None
    print(f"  reply:      {(final.final_message or '').strip()}")
    print(f"  iterations: {final.iterations_used}")
    print(f"  tool_calls: {final.total_tool_calls}\n")
    await model.aclose()


# =============================================================================
# Part 4 — Streaming via SSE
# =============================================================================


async def part4_streaming() -> None:
    """Stream tokens from the Responses endpoint. The SSE
    ``output_text.delta`` events arrive as ``ModelChunkEvent``\\s —
    same surface every other Locus model presents."""
    print("=== Part 4: streaming ===\n")
    model = _make_model(store=False)
    print("  streamed: ", end="", flush=True)
    chunks = 0
    async for event in model.stream(
        [Message.system("Be brief."), Message.user("Count to five, one number per line.")]
    ):
        if isinstance(event, ModelChunkEvent) and event.content:
            print(event.content, end="", flush=True)
            chunks += 1
    print(f"\n  ({chunks} chunks)\n")
    await model.aclose()


# =============================================================================
# Main
# =============================================================================


async def _amain() -> None:
    await part1_stateless_multiturn()
    await part2_server_state_multiturn()
    await part3_tool_roundtrip()
    await part4_streaming()


def main() -> None:
    print("=" * 70)
    print("Notebook 03 — OCIResponsesModel: the OCI GenAI Responses endpoint")
    print("=" * 70 + "\n")
    print(f"OCI_PROFILE              = {os.environ.get('OCI_PROFILE', '(unset)')}")
    print(f"OCI_REGION               = {REGION}")
    print(f"OCI_COMPARTMENT          = {(COMPARTMENT or '(unset)')[:60]}...")
    print(
        f"OCI_RESPONSES_STORE      = {'1 (try store=True too)' if STORE_ENABLED else '0 (ZDR-safe only)'}\n"
    )

    asyncio.run(_amain())

    print("=" * 70)
    print("See also: docs/concepts/oci-responses.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
