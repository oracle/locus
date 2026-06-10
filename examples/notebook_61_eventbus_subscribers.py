#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 61: Shipping agent events to a SIEM.

The bus has three subscribe shapes, each suited to a different consumer
in a SOC pipeline:

- bus.subscribe(run_id): events for one investigation, with history
  replay on connect then live events, terminated by a sentinel on
  stream close. The shape an analyst console uses for a single case.
- bus.subscribe_global(): every event from every run, no history
  replay. The shape a SIEM forwarder uses — one tap covering all
  concurrent investigations.
- bus._history.get(run_id, ()): direct read of the per-run history
  deque (test helper; capped at 500 events x 200 runs LRU).

- Per-case subscriber alongside a SIEM-style global subscriber on two
  concurrent investigations.
- bus.stats() snapshot — queue sizes, history depth, drop counter,
  retained-run count: the health metrics of your event pipeline.

Run it
    # Default: the bundled mock model (set TULIP_MODEL_PROVIDER for a live provider)
    python examples/notebook_61_eventbus_subscribers.py

    # Offline / no credentials:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_61_eventbus_subscribers.py
"""

from __future__ import annotations

import asyncio

from config import get_model

from tulip.agent import Agent
from tulip.observability import get_event_bus, run_context


# Part 1: the SIEM forwarder (global subscriber) sees both
# investigations; a per-case subscriber sees only its own.


async def part1_global_vs_per_run() -> None:
    print("\n--- Part 1: SIEM forwarder vs per-case subscriber ---")

    bus = get_event_bus()
    siem_kinds: list[str] = []
    case_a_kinds: list[str] = []

    async def siem_forwarder() -> None:
        # In production this loop would POST each event to your SIEM's
        # HTTP event collector; here it just records what it would ship.
        async for ev in bus.subscribe_global():
            siem_kinds.append(f"{ev.run_id[:6]}/{ev.event_type}")
            if ev.event_type == "agent.terminate" and len(siem_kinds) >= 4:
                return

    async def case_a_sub(rid: str) -> None:
        async for ev in bus.subscribe(rid):
            case_a_kinds.append(ev.event_type)
            if ev.event_type == "agent.terminate":
                return

    async def dispatch(rid: str, prompt: str) -> None:
        async with run_context(rid):
            agent = Agent(model=get_model(), max_iterations=2)
            await asyncio.to_thread(agent.run_sync, prompt)
            await bus.close_stream(rid)

    g_task = asyncio.create_task(siem_forwarder())
    a_task = asyncio.create_task(case_a_sub("case-A"))
    await asyncio.sleep(0)

    await asyncio.gather(
        dispatch("case-A", "Reply: triage note filed for case A"),
        dispatch("case-B", "Reply: triage note filed for case B"),
    )
    # Both subscribers exit on their close-stream sentinels.
    await asyncio.wait_for(asyncio.gather(g_task, a_task), timeout=15.0)

    print(f"SIEM forwarder saw {len(siem_kinds)} events across both investigations:")
    for k in siem_kinds[:6]:
        print(f"  - {k}")
    if len(siem_kinds) > 6:
        print(f"  ... +{len(siem_kinds) - 6} more")

    print(f"case-A subscriber saw {len(case_a_kinds)} events (only its own case):")
    for k in case_a_kinds[:6]:
        print(f"  - {k}")


# Part 2: bus.stats() snapshot. Pipe these into the same dashboard that
# watches your log-shipping pipeline — dropped events are missing evidence.


async def part2_stats() -> None:
    print("\n--- Part 2: bus.stats() ---")

    bus = get_event_bus()
    snapshot = bus.stats()
    for k, v in snapshot.items():
        print(f"  {k}: {v}")


async def main() -> None:
    await part1_global_vs_per_run()
    await part2_stats()


if __name__ == "__main__":
    asyncio.run(main())
