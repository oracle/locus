#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 60: Scanner telemetry — the agent yield bridge.

A scanning agent's tool calls are the part auditors care about most.
Every Agent.run is decorated with @_bus_bridge so the nine typed events
it yields (ThinkEvent, ToolStartEvent, ToolCompleteEvent, ReflectEvent,
GroundingEvent, ModelChunkEvent, ModelCompleteEvent, InterruptEvent,
TerminateEvent) get republished on the bus as agent.* events when a
run_context is open — every lookup the scanner makes is visible in the
run stream. ModelCompleteEvent additionally fires agent.tokens.used so
cost dashboards subscribe without parsing the completion payload.

- How nine yielded TulipEvent types map to agent.* bus events.
- Tool-call telemetry with span_id pairing (agent.tool.started and
  agent.tool.completed share an id) — start/finish of each scan step.
- Token usage from result.metrics — the canonical source for cost
  meters and budget enforcers on always-on SOC automation.

Run it
    # Default: the bundled mock model (set TULIP_MODEL_PROVIDER for a live provider)
    python examples/notebook_60_agent_yield_bridge.py

    # Offline / no credentials:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_60_agent_yield_bridge.py
"""

from __future__ import annotations

import asyncio

from config import get_model

from tulip.agent import Agent
from tulip.observability import get_event_bus, run_context
from tulip.tools import tool


@tool
def lookup_ip_reputation(ip: str) -> str:
    """Return a canned reputation verdict for an IP address (offline mock data)."""
    # RFC 5737 documentation range stands in for a known-bad block.
    return "malicious" if ip.startswith("192.0.2.") else "clean"


@tool
def count_failed_logins(host: str) -> int:
    """Return a canned count of failed SSH logins for a host (offline mock data)."""
    return 42 if host == "web-01" else 3


# Part 1: full agent.* lifecycle on one scanning run. Print every
# event with its span_id so you can see start/complete pairing — each
# tool span is one auditable scan step.


async def part1_full_lifecycle() -> None:
    print("\n--- Part 1: full agent.* lifecycle ---")

    agent = Agent(
        model=get_model(),
        tools=[lookup_ip_reputation, count_failed_logins],
        max_iterations=4,
        system_prompt=(
            "You are a SOC scanning agent. Make one tool call at a time. After all "
            "tool calls, report your findings."
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
        async for event in agent.run(
            "Check the reputation of 192.0.2.44, count failed logins on web-01, and report both."
        ):
            from tulip.core.events import TerminateEvent

            if isinstance(event, TerminateEvent):
                result = event
        print(f"agent reply: {result.final_message[:160] if result else '(no reply)'}")
        await asyncio.wait_for(consumer_task, timeout=20.0)
        await bus.close_stream(rid)


# Part 2: token usage as a cost meter. result.metrics is authoritative;
# agent.tokens.used SSE events are for streaming consumers that want
# per-call deltas instead of the final total. Always-on triage automation
# needs both: budget enforcement and live burn-rate dashboards.


async def part2_token_meter() -> None:
    print("\n--- Part 2: token meter via result.metrics ---")

    running_prompt = running_completion = running_total = 0

    # Multi-run shift: accumulate token totals across triage calls.
    prompts = [
        "In one sentence: what is an indicator of compromise?",
        "In one sentence: what is lateral movement?",
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
        f"  ─── shift total: prompt={running_prompt}  "
        f"completion={running_completion}  total={running_total}"
    )


async def main() -> None:
    await part1_full_lifecycle()
    await part2_token_meter()


if __name__ == "__main__":
    asyncio.run(main())
