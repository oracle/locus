#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 61: Shipping agent events to a cloud telemetry platform.

The bus has three subscribe shapes, each suited to a different consumer
in a cloud-ops pipeline:

- bus.subscribe(run_id): events for one deployment, with history
  replay on connect then live events, terminated by a sentinel on
  stream close. The shape a rollout console uses for a single release.
- bus.subscribe_global(): every event from every run, no history
  replay. The shape a telemetry forwarder uses — one tap covering all
  concurrent deployments.
- bus._history.get(run_id, ()): direct read of the per-run history
  deque (test helper; capped at 500 events x 200 runs LRU).

- Per-deployment subscriber alongside an observability-style global
  subscriber on two concurrent rollouts.
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


# Part 1: the telemetry forwarder (global subscriber) sees both
# deployments; a per-deployment subscriber sees only its own.


async def part1_global_vs_per_run() -> None:
    print("\n--- Part 1: telemetry forwarder vs per-deployment subscriber ---")

    bus = get_event_bus()
    telemetry_kinds: list[str] = []
    deploy_a_kinds: list[str] = []

    async def telemetry_forwarder() -> None:
        # In production this loop would POST each event to your
        # observability platform's intake API; here it just records
        # what it would ship.
        async for ev in bus.subscribe_global():
            telemetry_kinds.append(f"{ev.run_id[:6]}/{ev.event_type}")
            if ev.event_type == "agent.terminate" and len(telemetry_kinds) >= 4:
                return

    async def deploy_a_sub(rid: str) -> None:
        async for ev in bus.subscribe(rid):
            deploy_a_kinds.append(ev.event_type)
            if ev.event_type == "agent.terminate":
                return

    async def dispatch(rid: str, prompt: str) -> None:
        async with run_context(rid):
            agent = Agent(model=get_model(), max_iterations=2)
            await asyncio.to_thread(agent.run_sync, prompt)
            await bus.close_stream(rid)

    g_task = asyncio.create_task(telemetry_forwarder())
    a_task = asyncio.create_task(deploy_a_sub("deploy-A"))
    await asyncio.sleep(0)

    await asyncio.gather(
        dispatch("deploy-A", "Reply: rollout recorded for deployment A"),
        dispatch("deploy-B", "Reply: rollout recorded for deployment B"),
    )
    # Both subscribers exit on their close-stream sentinels.
    await asyncio.wait_for(asyncio.gather(g_task, a_task), timeout=15.0)

    print(f"telemetry forwarder saw {len(telemetry_kinds)} events across both deployments:")
    for k in telemetry_kinds[:6]:
        print(f"  - {k}")
    if len(telemetry_kinds) > 6:
        print(f"  ... +{len(telemetry_kinds) - 6} more")

    print(f"deploy-A subscriber saw {len(deploy_a_kinds)} events (only its own rollout):")
    for k in deploy_a_kinds[:6]:
        print(f"  - {k}")


# Part 2: bus.stats() snapshot. Pipe these into the same dashboard that
# watches your telemetry-shipping pipeline — dropped events are missing
# metrics.


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
