# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Notebook 60: Support timeline — every step a ticket bot takes, on the record.

When a support agent handles a customer ticket, you want to see — later,
on a QA review or when a customer disputes an outcome — exactly what it
did. Locus ships an in-process pub/sub EventBus that publishes typed
StreamEvents for the meaningful steps of execution: agent thinking, tool
calls, multi-agent fan-outs, checkpoints — each under a canonical
event_type whose prefix names the component (agent.*, multiagent.*,
composition.*, memory.*). That stream is the timeline: the support desk's
typed, ordered, replayable record of every action an agent took on a
ticket.

A complete action timeline is what lets a supervisor reconstruct, after
the fact, whether the bot followed policy before it refunded an order or
closed a case.

Telemetry is opt-in. Code that never enters a run_context pays one
ContextVar.get() per emission site — no bus, no events, no
allocations.

- Run an Agent with no telemetry: the SDK-default path.
- Reply on the same ticket inside run_context(), this time with a lookup
  tool attached, and subscribe to the bus.
- The canonical events on the agent path: agent.think,
  agent.tool.started/completed, agent.terminate, then agent.tokens.used
  with the run's cumulative counts when the provider metered the run.
- Read the per-run history buffer after the fact (replay semantics for
  late subscribers — i.e. reconstruct the ticket timeline after the run).
- The same ticket reply as OpenTelemetry spans: TelemetryHook names them per
  the GenAI semantic conventions (invoke_agent, chat, execute_tool) with the
  gen_ai.* attributes, caught here by the in-memory exporter — no backend,
  no keys — and checked against the conventions.

Run it:
    # Default: OCI GenAI if ~/.oci/config has a usable profile, else the bundled mock model
    python examples/notebook_60_observability_basics.py

    # Offline / no credentials:
    LOCUS_MODEL_PROVIDER=mock python examples/notebook_60_observability_basics.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from config import get_model

from locus.agent import Agent
from locus.observability import get_event_bus, run_context
from locus.tools import tool


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order's shipping and claim status (mock data)."""
    return f"{order_id}: delivered 2026-10-20, damage claim open, replacement eligible"


# Part 1: no telemetry. The bus singleton never gets built.


async def part1_no_telemetry() -> None:
    """No run_context: every emit short-circuits before allocating anything."""
    print("\n--- Part 1: no run_context — bus stays uninstantiated ---")

    agent = Agent(model=get_model(), max_iterations=2)
    result = await agent.arun(
        "In one sentence, summarize ticket CS-1042: customer says their order arrived damaged."
    )
    print("agent reply:", result.message[:120])

    # get_event_bus() builds the singleton on first call, so the only way to
    # observe "never built" is to read the module attribute directly.
    from locus.observability import event_bus as _bus_mod

    assert _bus_mod._event_bus is None, (
        "running an Agent without run_context must not construct the bus"
    )
    print("bus singleton is None — zero allocations spent on telemetry")


# Part 2: opt in by wrapping the ticket reply in run_context(); subscribe
# to the bus and watch every action the agent takes land on the timeline.


async def part2_subscribe() -> None:
    """Same Agent.run, this time inside a run_context — the timeline is live."""
    print("\n--- Part 2: run_context active — subscribe to the ticket timeline ---")

    # A tool on the agent is what puts agent.tool.started/completed on the
    # timeline — the events a QA reviewer actually wants to see.
    agent = Agent(model=get_model(), tools=[lookup_order], max_iterations=3)
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
        # decorator on Agent.run republishes each yielded LocusEvent the
        # bridge maps — every think, tool and terminate step of this run.
        result = await agent.arun("Look up order CS-1042 and reply with its status in one line.")
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
        await agent.arun("Reply: acknowledged")
        await bus.close_stream(rid)

        replayed: list[str] = []
        async for ev in bus.subscribe(rid):
            replayed.append(ev.event_type)

    print(f"replayed {len(replayed)} events from history (after the run finished)")
    for e in replayed[:10]:
        print(f"  - {e}")
    if len(replayed) > 10:
        print(f"  ... +{len(replayed) - 10} more")


# Part 4: the same reply as OpenTelemetry spans. TelemetryHook files the run
# under the GenAI semantic conventions — one invoke_agent span, a chat span per
# model call, an execute_tool span per tool call, gen_ai.* attributes — and an
# in-memory exporter stands in for the backend. Swapping it for an OTLP
# exporter (or ``oci_apm(...).span_processor()``) is the only change a
# deployment makes.


def _span_tree(spans: list[Any]) -> list[str]:
    """Render finished spans as an indented tree, parents before children."""
    ids = {s.context.span_id for s in spans}
    children: dict[int | None, list[Any]] = {}
    for s in spans:
        parent = s.parent.span_id if s.parent is not None and s.parent.span_id in ids else None
        children.setdefault(parent, []).append(s)
    shown = (
        "gen_ai.operation.name",
        "gen_ai.tool.name",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
    )
    lines: list[str] = []

    def walk(parent: int | None, depth: int) -> None:
        for s in sorted(children.get(parent, []), key=lambda x: x.start_time):
            attrs = s.attributes or {}
            detail = ", ".join(f"{k}={attrs[k]!r}" for k in shown if k in attrs)
            lines.append(f"{'  ' * depth}{s.name}  [{s.kind.name}]  {detail}".rstrip())
            walk(s.context.span_id, depth + 1)

    walk(None, 0)
    return lines


async def part4_semconv_spans() -> None:
    """TelemetryHook + in-memory exporter: the run as GenAI-convention spans."""
    print("\n--- Part 4: OpenTelemetry spans under the GenAI semantic conventions ---")
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    except ImportError:
        print("OpenTelemetry SDK not installed: pip install 'locus-sdk[telemetry]' for this part")
        return

    from locus.hooks.builtin import TelemetryHook
    from locus.observability.semconv import validate_spans

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # The agent lends the hook its model and tools, so gen_ai.request.model,
    # gen_ai.provider.name and gen_ai.tool.description need no configuration.
    agent = Agent(
        model=get_model(),
        tools=[lookup_order],
        hooks=[TelemetryHook()],
        name="support-desk",
        max_iterations=3,
    )
    result = await agent.arun("Look up order CS-1042 and reply with its status in one line.")
    print("agent reply:", result.message[:120])

    spans = list(exporter.get_finished_spans())
    print(f"spans recorded ({len(spans)}):")
    for line in _span_tree(spans):
        print(f"  {line}")
    problems = validate_spans(spans)
    print("GenAI conventions check:", "ok" if not problems else "; ".join(problems))


async def main() -> None:
    await part1_no_telemetry()
    await part2_subscribe()
    await part3_history_replay()
    await part4_semconv_spans()


if __name__ == "__main__":
    asyncio.run(main())
