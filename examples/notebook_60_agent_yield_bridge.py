#!/usr/bin/env python3
# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Notebook 55: Agent yield bridge and token usage.

Every Agent.run is decorated with @_bus_bridge so the nine typed events
it yields (ThinkEvent, ToolStartEvent, ToolCompleteEvent, ReflectEvent,
GroundingEvent, ModelChunkEvent, ModelCompleteEvent, InterruptEvent,
TerminateEvent) get republished on the bus as agent.* events when a
run_context is open. ModelCompleteEvent additionally fires
agent.tokens.used so cost dashboards subscribe without parsing the
completion payload.

- How nine yielded LocusEvent types map to agent.* bus events.
- Tool-call telemetry with span_id pairing (agent.tool.started and
  agent.tool.completed share an id).
- Token usage from result.metrics — the recommended source for cost
  meters and budget enforcers.

Run it
    # Default: OCI Generative AI auto-detected from ~/.oci/config
    python examples/notebook_60_agent_yield_bridge.py

    # Offline / no credentials:
    LOCUS_MODEL_PROVIDER=mock python examples/notebook_60_agent_yield_bridge.py
"""

from __future__ import annotations

import asyncio

from config import get_model

from locus.agent import Agent
from locus.observability import get_event_bus, run_context
from locus.tools import tool


@tool
def add_numbers(a: int, b: int) -> int:
    """Return the sum of two integers."""
    return a + b


@tool
def multiply_numbers(a: int, b: int) -> int:
    """Return the product of two integers."""
    return a * b


# Part 1: full agent.* lifecycle on one tool-using run. Print every
# event with its span_id so you can see start/complete pairing.


async def part1_full_lifecycle() -> None:
    print("\n--- Part 1: full agent.* lifecycle ---")

    agent = Agent(
        model=get_model(),
        tools=[add_numbers, multiply_numbers],
        max_iterations=4,
        system_prompt=(
            "You answer with one tool call at a time. After all tool calls, give the final answer."
        ),
    )

    async with run_context() as rid:
        bus = get_event_bus()

        async def consumer() -> None:
            async for ev in bus.subscribe(rid):
                span = ev.data.get("span_id", "")
                tag = f" span={span[:8]}" if span else ""
                if ev.event_type.startswith("agent."):
                    print(f"  {ev.event_type}{tag}")
                if ev.event_type == "agent.terminate":
                    return

        consumer_task = asyncio.create_task(consumer())
        await asyncio.sleep(0)

        result = None
        async for event in agent.run("Compute (3 + 4) and then (5 * 7), and tell me both."):
            from locus.core.events import TerminateEvent

            if isinstance(event, TerminateEvent):
                result = event
        print(f"agent reply: {result.final_message[:160] if result else '(no reply)'}")
        await asyncio.wait_for(consumer_task, timeout=20.0)
        await bus.close_stream(rid)


# Part 2: token usage as a cost meter. result.metrics is authoritative;
# agent.tokens.used SSE events are for streaming consumers that want
# per-call deltas instead of the final total.


async def part2_token_meter() -> None:
    print("\n--- Part 2: token meter via result.metrics ---")

    running_prompt = running_completion = running_total = 0

    # Multi-run session: accumulate token totals across calls.
    prompts = [
        "In one sentence: what is JSON?",
        "In one sentence: what is a REST API?",
    ]

    for prompt in prompts:
        agent = Agent(model=get_model(), max_iterations=2)
        result = agent.run_sync(prompt)
        m = result.metrics
        running_prompt += m.prompt_tokens
        running_completion += m.completion_tokens
        running_total += m.total_tokens
        print(
            f"  run: prompt={m.prompt_tokens:4d}  "
            f"completion={m.completion_tokens:3d}  "
            f"total={m.total_tokens:4d}  | '{prompt[:40]}'"
        )

    print(
        f"  ─── session total: prompt={running_prompt}  "
        f"completion={running_completion}  total={running_total}"
    )


async def main() -> None:
    await part1_full_lifecycle()
    await part2_token_meter()


if __name__ == "__main__":
    asyncio.run(main())
