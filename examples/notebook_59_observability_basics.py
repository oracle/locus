#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 59: Support timeline — every step a ticket bot takes, on the record.

When a support agent handles a customer ticket, you want to see — later,
on a QA review or when a customer disputes an outcome — exactly what it
did. Tulip ships an in-process pub/sub EventBus that publishes typed
StreamEvents for every meaningful step of execution: agent thinking, tool
calls, model completions, token usage, multi-agent fan-outs, checkpoints
— all under one canonical event_type per component. That stream is the
Timeline: the support desk's typed, ordered, replayable record of every
action an agent took on a ticket.

A complete action timeline is what lets a supervisor reconstruct, after
the fact, whether the bot followed policy before it refunded an order or
closed a case.

Telemetry is opt-in. Code that never enters a run_context pays one
ContextVar.get() per emission site — no bus, no events, no
allocations.

- Run an Agent with no telemetry: the SDK-default path.
- Wrap the same ticket reply in run_context() and subscribe to the bus.
- The canonical events: agent.think, agent.tool.started/completed,
  agent.tokens.used, agent.terminate.
- Read the per-run history buffer after the fact (replay semantics for
  late subscribers — i.e. reconstruct the ticket timeline after the run).

Run it
    # Default: the bundled mock model (set TULIP_MODEL_PROVIDER for a live provider)
    python examples/notebook_59_observability_basics.py

    # Offline / no credentials:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_59_observability_basics.py
"""

from __future__ import annotations

import asyncio

from config import get_model

from tulip.agent import Agent
from tulip.observability import get_event_bus, run_context


# Part 1: no telemetry. The bus singleton never gets built.


async def part1_no_telemetry() -> None:
    """No run_context: every emit short-circuits before allocating anything."""
    print("\n--- Part 1: no run_context — bus stays uninstantiated ---")

    agent = Agent(model=get_model(), max_iterations=2)
    result = agent.run_sync(
        "In one sentence, summarize ticket CS-1042: customer says their order arrived damaged."
    )
    print("agent reply:", result.message[:120])

    # Probe the module-level singleton to prove it was never built.
    from tulip.observability import event_bus as _bus_mod

    assert _bus_mod._event_bus is None, (
        "running an Agent without run_context must not construct the bus"
    )
    print("bus singleton is None — zero allocations spent on telemetry")


# Part 2: opt in by wrapping the ticket reply in run_context(); subscribe
# to the bus and watch every action the agent takes land on the timeline.


async def part2_subscribe() -> None:
    """Same Agent.run, this time inside a run_context — the timeline is live."""
    print("\n--- Part 2: run_context active — subscribe to the ticket timeline ---")

    agent = Agent(model=get_model(), max_iterations=2)
    seen: list[str] = []

    async with run_context() as rid:
        bus = get_event_bus()

        async def consumer() -> None:
            async for ev in bus.subscribe(rid):
                seen.append(ev.event_type)
                if ev.event_type == "agent.terminate":
                    return

        consumer_task = asyncio.create_task(consumer())
        await asyncio.sleep(0)  # let the subscriber register

        # The contextvar set by run_context() means the @_bus_bridge
        # decorator on Agent.run forwards every yielded TulipEvent to
        # the bus automatically — nothing the agent does goes unrecorded.
        result = await asyncio.to_thread(agent.run_sync, "Reply with the single word: resolved")
        print("agent reply:", result.message[:120])

        await asyncio.wait_for(consumer_task, timeout=10.0)
        # Closing the stream ends any other subscribers cleanly.
        await bus.close_stream(rid)

    print(f"timeline events recorded ({len(seen)}):")
    for e in seen:
        print(f"  - {e}")


# Part 3: late subscribers see the whole run replayed from history
# (capped at 500 events x 200 retained runs) — a QA review the next day
# reads the same timeline the live agent console saw.


async def part3_history_replay() -> None:
    """Subscribe after the run finished; the history deque replays the ticket."""
    print("\n--- Part 3: late subscriber — after-the-fact QA replay ---")

    agent = Agent(model=get_model(), max_iterations=2)

    async with run_context() as rid:
        bus = get_event_bus()
        # Run first; close the stream; subscribe second.
        await asyncio.to_thread(agent.run_sync, "Reply: acknowledged")
        await bus.close_stream(rid)

        replayed: list[str] = []
        async for ev in bus.subscribe(rid):
            replayed.append(ev.event_type)

    print(f"replayed {len(replayed)} events from history (after the run finished)")
    for e in replayed[:10]:
        print(f"  - {e}")
    if len(replayed) > 10:
        print(f"  ... +{len(replayed) - 10} more")


async def main() -> None:
    await part1_no_telemetry()
    await part2_subscribe()
    await part3_history_replay()


if __name__ == "__main__":
    asyncio.run(main())
