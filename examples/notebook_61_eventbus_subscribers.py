#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 56: EventBus subscriber patterns.

The bus has three subscribe shapes, each suited to a different consumer:

- bus.subscribe(run_id): events for one dispatch, with history replay
  on connect then live events, terminated by a sentinel on stream close.
- bus.subscribe_global(): every event from every run, no history
  replay. Good fit for a monitoring dashboard that spans concurrent runs.
- bus._history.get(run_id, ()): direct read of the per-run history
  deque (test helper; capped at 500 events x 200 runs LRU).

- Per-run subscriber alongside a global subscriber on two concurrent
  dispatches.
- bus.stats() snapshot — queue sizes, history depth, drop counter,
  retained-run count.

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


# Part 1: a global subscriber sees both dispatches; a per-run
# subscriber sees only its own.


async def part1_global_vs_per_run() -> None:
    print("\n--- Part 1: global vs per-run subscribers ---")

    bus = get_event_bus()
    global_kinds: list[str] = []
    run_a_kinds: list[str] = []

    async def global_sub() -> None:
        async for ev in bus.subscribe_global():
            global_kinds.append(f"{ev.run_id[:6]}/{ev.event_type}")
            if ev.event_type == "agent.terminate" and len(global_kinds) >= 4:
                return

    async def run_a_sub(rid: str) -> None:
        async for ev in bus.subscribe(rid):
            run_a_kinds.append(ev.event_type)
            if ev.event_type == "agent.terminate":
                return

    async def dispatch(rid: str, prompt: str) -> None:
        async with run_context(rid):
            agent = Agent(model=get_model(), max_iterations=2)
            await asyncio.to_thread(agent.run_sync, prompt)
            await bus.close_stream(rid)

    g_task = asyncio.create_task(global_sub())
    a_task = asyncio.create_task(run_a_sub("run-A"))
    await asyncio.sleep(0)

    await asyncio.gather(
        dispatch("run-A", "Reply: hi from A"),
        dispatch("run-B", "Reply: hi from B"),
    )
    # Both subscribers exit on their close-stream sentinels.
    await asyncio.wait_for(asyncio.gather(g_task, a_task), timeout=15.0)

    print(f"global saw {len(global_kinds)} events across both runs:")
    for k in global_kinds[:6]:
        print(f"  - {k}")
    if len(global_kinds) > 6:
        print(f"  ... +{len(global_kinds) - 6} more")

    print(f"run-A subscriber saw {len(run_a_kinds)} events (only its own run):")
    for k in run_a_kinds[:6]:
        print(f"  - {k}")


# Part 2: bus.stats() snapshot. Pipe these into a monitoring dashboard.


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
