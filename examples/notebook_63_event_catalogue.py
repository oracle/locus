# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Notebook 63: The event taxonomy as a compliance artifact.

When a PCI DSS assessor asks "what can your payment agents do, and how
would you know they did it?" — the question behind a payments-operations
monitoring control — the answer is the event catalogue. Every component
in Locus emits typed events under one stable prefix: agent.*,
multiagent.*, composition.*, router.*, rag.*, memory.*, a2a.*, skills.*,
deepagent.*. The EV_* constants in locus.observability.emit are the
canonical registry — change one place, propagates everywhere — so the
artifact you hand an assessor is generated from the code itself, not a
spreadsheet that drifts out of date the day after it's signed.

- List every EV_* constant and its category prefix (always in sync with
  the codebase because it's read at import time — no stale spreadsheet).
- Drive a two-stage dispute-triage SequentialPipeline that surfaces
  composition.* events end-to-end.

Run it:
    # Default: OCI GenAI if ~/.oci/config has a usable profile, else the bundled mock model
    python examples/notebook_63_event_catalogue.py

    # Offline / no credentials:
    LOCUS_MODEL_PROVIDER=mock python examples/notebook_63_event_catalogue.py
"""

from __future__ import annotations

import asyncio
import sys
from collections import defaultdict

from config import get_model

from locus.agent import Agent
from locus.agent.composition import SequentialPipeline
from locus.observability import get_event_bus, run_context


# Part 1: enumerate the EV_* constants at runtime so the audit artifact
# never drifts from the codebase.


def part1_catalogue_tour() -> None:
    print("\n--- Part 1: canonical event_type catalogue ---")

    emit_mod = sys.modules["locus.observability.emit"]
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


# Part 2: SequentialPipeline emits composition.* events at every stage
# boundary — each stage of the dispute-triage pipeline leaves its own
# entry in the audit trail.


async def part2_composition() -> None:
    print("\n--- Part 2: composition.* events ---")

    enricher = Agent(model=get_model(), max_iterations=1)
    verdict_writer = Agent(model=get_model(), max_iterations=1)

    pipeline = SequentialPipeline(agents=[enricher, verdict_writer])

    async with run_context() as rid:
        bus = get_event_bus()

        async def consumer() -> None:
            seen: list[str] = []
            async for ev in bus.subscribe(rid):
                if ev.event_type.startswith("composition."):
                    seen.append(ev.event_type)
                # Stages are 0-indexed, so stage 1 is the last of our two.
                if ev.event_type == "composition.stage.completed" and ev.data.get("stage") == 1:
                    print("composition events seen so far:", seen)
                    return

        consumer_task = asyncio.create_task(consumer())
        # Yield once so the subscriber attaches before the first event is
        # published; otherwise the first stage's events are missed.
        await asyncio.sleep(0)

        await pipeline.run(
            "Summarize this disputed charge in one line: cardholder reports an "
            "unrecognized $420.00 payment to merchant ACME-STORE on card ending 4242."
        )
        await bus.close_stream(rid)
        await consumer_task


async def main() -> None:
    part1_catalogue_tour()
    await part2_composition()


if __name__ == "__main__":
    asyncio.run(main())
