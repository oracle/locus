#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 64: Research workflow — execute, causal, summarise, ground, replan.

create_research_workflow composes six node primitives into a StateGraph
that mirrors the production pattern used in research-shaped specialist
agents: a ReAct loop that gathers evidence, causal inference before
summary, LLM-as-judge grounding evaluation, and a two-level recovery
loop.

Recovery strategy::

    grounding score < threshold (first failure) → regenerate_summary (cheap)
    grounding score < threshold (subsequent)    → replan + full execute retry

Every node emits research.* SSE events so you can stream the workflow
end-to-end the same way you would stream an Agent run.

- Run create_research_workflow with the convenience factory.
- Subscribe to research.* SSE events in real time.
- Compose custom graphs from individual node primitives.
- Read the final state: summary, structured output, grounding score.

Run it
    # Default: the bundled mock model (set TULIP_MODEL_PROVIDER for a live provider)
    python examples/notebook_69_research_workflow.py

    # Offline / no credentials:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_69_research_workflow.py
"""

from __future__ import annotations

import asyncio

from config import get_model
from pydantic import BaseModel, Field

from tulip.deepagent.workflow import (
    KEY_CAUSAL_CONFIDENCE,
    KEY_CAUSAL_HYPOTHESIS,
    KEY_GROUNDING_SCORE,
    KEY_PROMPT,
    KEY_REGENERATION_COUNT,
    KEY_REPLAN_COUNT,
    KEY_STRUCTURED_OUTPUT,
    KEY_SUMMARY,
    create_research_workflow,
)
from tulip.observability import get_event_bus, run_context
from tulip.tools import tool


# Tiny module catalogue the agent can investigate via its three tools.


_CATALOGUE = {
    "tulip.router": {
        "purpose": "Cognitive router — compiles NL to orchestration shape.",
        "key_classes": ["Router", "GoalFrame", "ProtocolRegistry", "CognitiveCompiler"],
        "since": "0.2.0",
        "stability": "stable",
    },
    "tulip.deepagent": {
        "purpose": "Research-shaped agent factory + research workflow primitives.",
        "key_classes": ["create_deepagent", "create_research_workflow", "make_execute_node"],
        "since": "0.2.0",
        "stability": "stable",
    },
    "tulip.observability": {
        "purpose": "In-process SSE bus — run_context, EventBus, canonical EV_* constants.",
        "key_classes": ["EventBus", "run_context", "get_event_bus"],
        "since": "0.2.0",
        "stability": "stable",
    },
}


@tool
def list_modules() -> list[str]:
    """Return the list of available modules."""
    return list(_CATALOGUE.keys())


@tool
def describe_module(name: str) -> dict:
    """Return purpose, key classes, and stability for a module.

    Args:
        name: Dotted module name (e.g. ``tulip.router``).
    """
    return _CATALOGUE.get(name, {"error": f"module '{name}' not found"})


@tool
def count_classes(name: str) -> int:
    """Return the number of key public classes in a module.

    Args:
        name: Dotted module name.
    """
    entry = _CATALOGUE.get(name, {})
    return len(entry.get("key_classes", []))


# Structured output schema — the workflow emits this typed instance.


class ModuleSurvey(BaseModel):
    modules_covered: list[str] = Field(description="Modules that were researched.")
    summary: str = Field(description="2-3 sentence survey of what was found.")
    stability: str = Field(description="Overall stability assessment.")
    confidence: float = Field(ge=0.0, le=1.0)


# Part 1: convenience factory + live SSE stream of research.* events.


async def part1_factory_with_sse() -> None:
    print("\n--- Part 1: research workflow with SSE ---")

    workflow = create_research_workflow(
        model=get_model(),
        tools=[list_modules, describe_module, count_classes],
        system_prompt=(
            "You are a tulip SDK analyst. Use tools to survey the available modules. "
            "List all modules, then describe each one in detail."
        ),
        output_schema=ModuleSurvey,
        grounding_threshold=0.60,
        max_replans=1,
        max_regenerations=1,
        max_iterations=8,
    )

    events_seen: list[str] = []

    async def stream_research_events(rid: str) -> None:
        async for ev in get_event_bus().subscribe(rid):
            if ev.event_type.startswith("research.") or ev.event_type.startswith("agent."):
                events_seen.append(ev.event_type)
                if ev.event_type.startswith("research."):
                    print(f"  📡 {ev.event_type} {ev.data}")

    async with run_context() as rid:
        streamer = asyncio.create_task(stream_research_events(rid))
        result = await workflow.execute({KEY_PROMPT: "Survey all tulip modules."})
        await get_event_bus().close_stream(rid)
        await asyncio.wait_for(streamer, timeout=5.0)

    final = result.final_state
    print(f"\n  grounding score : {final.get(KEY_GROUNDING_SCORE, 0):.0%}")
    print(f"  causal hypothesis: {final.get(KEY_CAUSAL_HYPOTHESIS, '')[:80]}")
    print(f"  causal confidence: {final.get(KEY_CAUSAL_CONFIDENCE, 0):.0%}")
    print(f"  replans used     : {final.get(KEY_REPLAN_COUNT, 0)}")
    print(f"  regenerations    : {final.get(KEY_REGENERATION_COUNT, 0)}")

    survey: ModuleSurvey | None = final.get(KEY_STRUCTURED_OUTPUT)
    if survey:
        print(f"\n  modules covered: {survey.modules_covered}")
        print(f"  summary: {survey.summary[:200]}")
        print(f"  stability: {survey.stability} | confidence: {survey.confidence:.0%}")
    else:
        summary = final.get(KEY_SUMMARY, "")
        print(f"\n  summary: {summary[:300]}")

    research_events = [e for e in events_seen if e.startswith("research.")]
    print(f"\n  research.* events fired: {research_events}")


# Part 2: build a minimal graph manually — execute + summarise, no causal.


async def part2_custom_graph() -> None:
    print("\n--- Part 2: custom graph (no causal inference) ---")

    from tulip.deepagent.workflow import (  # noqa: PLC0415
        KEY_EVIDENCE,
        make_execute_node,
        make_grounding_eval_node,
        make_summarize_node,
        route_after_grounding,
    )
    from tulip.multiagent.graph import END, START, StateGraph  # noqa: PLC0415

    graph = StateGraph()
    graph.add_node("execute", make_execute_node(get_model(), [list_modules, describe_module]))
    graph.add_node("summarize", make_summarize_node(get_model()))
    graph.add_node("grounding_eval", make_grounding_eval_node(get_model()))

    router = route_after_grounding(threshold=0.5, max_replans=0, max_regenerations=0)

    graph.add_edge(START, "execute")
    graph.add_edge("execute", "summarize")
    graph.add_edge("summarize", "grounding_eval")
    graph.add_conditional_edges(
        "grounding_eval",
        router,
        {"regenerate": END, "replan": END, END: END},
    )

    workflow = graph.compile()

    async with run_context() as rid:
        result = await workflow.execute({KEY_PROMPT: "What does tulip.router do?"})
        await get_event_bus().close_stream(rid)

    final = result.final_state
    evidence_count = len(final.get(KEY_EVIDENCE, []))
    print(f"  evidence pieces: {evidence_count}")
    print(f"  grounding score: {final.get(KEY_GROUNDING_SCORE, 0):.0%}")
    print(f"  summary: {final.get(KEY_SUMMARY, '')[:200]}")


async def main() -> None:
    await part1_factory_with_sse()
    await part2_custom_graph()


if __name__ == "__main__":
    asyncio.run(main())
