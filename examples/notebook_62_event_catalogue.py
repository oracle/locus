#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 57: Full event catalogue tour.

Every component in Tulip emits typed events under one stable prefix:
agent.*, multiagent.*, composition.*, router.*, rag.*, memory.*, a2a.*,
skills.*, deepagent.*. The EV_* constants in tulip.observability.emit
are the canonical registry — change one place, propagates everywhere.

- List every EV_* constant and its category prefix (always in sync with
  the codebase because it's read at import time).
- Drive a SequentialPipeline + LoopAgent that surfaces composition.*
  events end-to-end.

Run it
    # Default: the bundled mock model (set TULIP_MODEL_PROVIDER for a live provider)
    python examples/notebook_62_event_catalogue.py

    # Offline / no credentials:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_62_event_catalogue.py
"""

from __future__ import annotations

import asyncio
import sys
from collections import defaultdict

from config import get_model

from tulip.agent import Agent
from tulip.agent.composition import LoopAgent, SequentialPipeline
from tulip.observability import get_event_bus, run_context


# Part 1: enumerate the EV_* constants at runtime so the catalogue
# never drifts from the codebase.


def part1_catalogue_tour() -> None:
    print("\n--- Part 1: canonical event_type catalogue ---")

    emit_mod = sys.modules["tulip.observability.emit"]
    by_prefix: dict[str, list[str]] = defaultdict(list)
    for name in dir(emit_mod):
        if not name.startswith("EV_"):
            continue
        value = getattr(emit_mod, name)
        if not isinstance(value, str):
            continue
        prefix = value.split(".", 1)[0]
        by_prefix[prefix].append(value)

    for prefix in sorted(by_prefix):
        print(f"  {prefix}.*  ({len(by_prefix[prefix])} events)")
        for ev in sorted(by_prefix[prefix]):
            print(f"    - {ev}")


# Part 2: SequentialPipeline and LoopAgent emit composition.* events
# at every stage / iteration boundary.


async def part2_composition() -> None:
    print("\n--- Part 2: composition.* events ---")

    a = Agent(model=get_model(), max_iterations=1)
    b = Agent(model=get_model(), max_iterations=1)

    pipeline = SequentialPipeline(agents=[a, b])

    async with run_context() as rid:
        bus = get_event_bus()

        async def consumer() -> None:
            seen: list[str] = []
            async for ev in bus.subscribe(rid):
                if ev.event_type.startswith("composition."):
                    seen.append(ev.event_type)
                if ev.event_type == "composition.fanout.completed":
                    break
                if ev.event_type == "composition.stage.completed" and ev.data.get("stage") == 1:
                    print("composition events seen so far:", seen)
                    return

        consumer_task = asyncio.create_task(consumer())
        await asyncio.sleep(0)

        await pipeline.run("Tell me a one-line haiku about JSON.")
        await bus.close_stream(rid)
        await consumer_task


async def main() -> None:
    part1_catalogue_tour()
    await part2_composition()


if __name__ == "__main__":
    asyncio.run(main())
