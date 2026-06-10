#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 69: Vulnerability research workflow — recon, hypothesize, validate, report.

create_research_workflow composes six node primitives into a StateGraph
that mirrors how a vulnerability-research agent should work: a ReAct
loop that gathers evidence from advisory tools (recon), causal
inference that proposes a root-cause hypothesis before the summary,
LLM-as-judge grounding evaluation that scores whether the report is
actually backed by that evidence, and a two-level recovery loop — an
ungrounded vulnerability claim is a false positive (OWASP LLM09,
Misinformation), so it gets regenerated or re-planned instead of
shipped. No claim reaches the report without evidence behind it.

Recovery strategy::

    grounding score < threshold (first failure) → regenerate_summary (cheap)
    grounding score < threshold (subsequent)    → replan + full execute retry

Every node emits research.* SSE events so you can stream the engagement
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


# Tiny advisory catalogue the agent can investigate via its three tools.
# All advisory IDs and data are invented (CVE-2024-999xx is a fake range).


_ADVISORIES = {
    "CVE-2024-99901": {
        "summary": "SQL injection in the ticket-search endpoint of HelpDeskPro.",
        "affected_components": ["helpdeskpro-api", "ticket-search", "legacy-reports"],
        "cvss": 9.1,
        "patch_available": True,
    },
    "CVE-2024-99902": {
        "summary": "Hard-coded credential shipped in the AcmeBackup agent installer.",
        "affected_components": ["acmebackup-agent", "installer"],
        "cvss": 7.4,
        "patch_available": True,
    },
    "CVE-2024-99903": {
        "summary": "Path traversal in StaticServe file downloads (intranet only).",
        "affected_components": ["staticserve"],
        "cvss": 5.8,
        "patch_available": False,
    },
}


@tool
def list_advisories() -> list[str]:
    """Return the list of tracked advisory IDs."""
    return list(_ADVISORIES.keys())


@tool
def describe_advisory(cve_id: str) -> dict:
    """Return summary, affected components, CVSS, and patch status for an advisory.

    Args:
        cve_id: Advisory identifier (e.g. ``CVE-2024-99901``).
    """
    return _ADVISORIES.get(cve_id, {"error": f"advisory '{cve_id}' not found"})


@tool
def count_affected_components(cve_id: str) -> int:
    """Return the number of components affected by an advisory.

    Args:
        cve_id: Advisory identifier.
    """
    entry = _ADVISORIES.get(cve_id, {})
    return len(entry.get("affected_components", []))


# Structured output schema — the workflow emits this typed instance.


class ExposureAssessment(BaseModel):
    advisories_covered: list[str] = Field(description="Advisory IDs that were researched.")
    summary: str = Field(description="2-3 sentence exposure assessment of what was found.")
    overall_risk: str = Field(description="Overall risk rating: low, medium, high, or critical.")
    confidence: float = Field(ge=0.0, le=1.0)


# Part 1: convenience factory + live SSE stream of research.* events.


async def part1_factory_with_sse() -> None:
    print("\n--- Part 1: vulnerability research workflow with SSE ---")

    workflow = create_research_workflow(
        model=get_model(),
        tools=[list_advisories, describe_advisory, count_affected_components],
        system_prompt=(
            "You are a vulnerability research analyst. Use tools to survey the tracked "
            "advisories. List all advisories, then describe each one in detail. "
            "Every claim must trace to tool evidence."
        ),
        output_schema=ExposureAssessment,
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
        result = await workflow.execute(
            {KEY_PROMPT: "Assess our exposure across all tracked advisories."}
        )
        await get_event_bus().close_stream(rid)
        await asyncio.wait_for(streamer, timeout=5.0)

    final = result.final_state
    print(f"\n  grounding score : {final.get(KEY_GROUNDING_SCORE, 0):.0%}")
    print(f"  causal hypothesis: {final.get(KEY_CAUSAL_HYPOTHESIS, '')[:80]}")
    print(f"  causal confidence: {final.get(KEY_CAUSAL_CONFIDENCE, 0):.0%}")
    print(f"  replans used     : {final.get(KEY_REPLAN_COUNT, 0)}")
    print(f"  regenerations    : {final.get(KEY_REGENERATION_COUNT, 0)}")

    assessment: ExposureAssessment | None = final.get(KEY_STRUCTURED_OUTPUT)
    if assessment:
        print(f"\n  advisories covered: {assessment.advisories_covered}")
        print(f"  summary: {assessment.summary[:200]}")
        print(f"  risk: {assessment.overall_risk} | confidence: {assessment.confidence:.0%}")
    else:
        summary = final.get(KEY_SUMMARY, "")
        print(f"\n  summary: {summary[:300]}")

    research_events = [e for e in events_seen if e.startswith("research.")]
    print(f"\n  research.* events fired: {research_events}")


# Part 2: build a minimal graph manually — recon + summarise, no causal step.


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
    graph.add_node("execute", make_execute_node(get_model(), [list_advisories, describe_advisory]))
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
        result = await workflow.execute({KEY_PROMPT: "What does CVE-2024-99901 affect?"})
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
