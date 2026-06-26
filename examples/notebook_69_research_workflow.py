#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 69: Customer-support research workflow — recon, hypothesize, validate, report.

create_research_workflow composes six node primitives into a StateGraph
that mirrors how a customer-support research agent should work: a ReAct
loop that gathers evidence from knowledge-base tools (recon), causal
inference that proposes a root-cause hypothesis before the summary,
LLM-as-judge grounding evaluation that scores whether the reply is
actually backed by that evidence, and a two-level recovery loop — an
ungrounded support claim is a hallucinated answer, so it gets
regenerated or re-planned instead of sent to the customer. No claim
reaches the reply without evidence behind it.

Recovery strategy::

    grounding score < threshold (first failure) → regenerate_summary (cheap)
    grounding score < threshold (subsequent)    → replan + full execute retry

Every node emits research.* SSE events so you can stream the
investigation end-to-end the same way you would stream an Agent run.

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


# Tiny knowledge base the agent can investigate via its three tools.
# All article IDs and data are invented for this offline demo.


_KB_ARTICLES = {
    "KB-2024-00101": {
        "summary": "Checkout hangs when a saved card has an expired billing address.",
        "affected_features": ["checkout", "saved-payments", "address-book"],
        "impact": 9.1,
        "fix_shipped": True,
    },
    "KB-2024-00102": {
        "summary": "Password-reset emails land in spam for self-hosted mail domains.",
        "affected_features": ["password-reset", "email-delivery"],
        "impact": 7.4,
        "fix_shipped": True,
    },
    "KB-2024-00103": {
        "summary": "Mobile app shows stale order status until a manual refresh.",
        "affected_features": ["mobile-app"],
        "impact": 5.8,
        "fix_shipped": False,
    },
}


@tool
def list_articles() -> list[str]:
    """Return the list of tracked knowledge-base article IDs."""
    return list(_KB_ARTICLES.keys())


@tool
def describe_article(article_id: str) -> dict:
    """Return summary, affected features, impact, and fix status for an article.

    Args:
        article_id: Knowledge-base identifier (e.g. ``KB-2024-00101``).
    """
    return _KB_ARTICLES.get(article_id, {"error": f"article '{article_id}' not found"})


@tool
def count_affected_features(article_id: str) -> int:
    """Return the number of product features affected by an article.

    Args:
        article_id: Knowledge-base identifier.
    """
    entry = _KB_ARTICLES.get(article_id, {})
    return len(entry.get("affected_features", []))


# Structured output schema — the workflow emits this typed instance.


class SupportAssessment(BaseModel):
    articles_covered: list[str] = Field(description="Article IDs that were researched.")
    summary: str = Field(description="2-3 sentence assessment of the customer impact found.")
    overall_severity: str = Field(
        description="Overall severity rating: low, medium, high, or critical."
    )
    confidence: float = Field(ge=0.0, le=1.0)


# Part 1: convenience factory + live SSE stream of research.* events.


async def part1_factory_with_sse() -> None:
    print("\n--- Part 1: customer-support research workflow with SSE ---")

    workflow = create_research_workflow(
        model=get_model(),
        tools=[list_articles, describe_article, count_affected_features],
        system_prompt=(
            "You are a customer-support research analyst. Use tools to survey the tracked "
            "knowledge-base articles. List all articles, then describe each one in detail. "
            "Every claim must trace to tool evidence."
        ),
        output_schema=SupportAssessment,
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
            {KEY_PROMPT: "Assess the customer impact across all tracked articles."}
        )
        await get_event_bus().close_stream(rid)
        await asyncio.wait_for(streamer, timeout=5.0)

    final = result.final_state
    print(f"\n  grounding score : {final.get(KEY_GROUNDING_SCORE, 0):.0%}")
    print(f"  causal hypothesis: {final.get(KEY_CAUSAL_HYPOTHESIS, '')[:80]}")
    print(f"  causal confidence: {final.get(KEY_CAUSAL_CONFIDENCE, 0):.0%}")
    print(f"  replans used     : {final.get(KEY_REPLAN_COUNT, 0)}")
    print(f"  regenerations    : {final.get(KEY_REGENERATION_COUNT, 0)}")

    assessment: SupportAssessment | None = final.get(KEY_STRUCTURED_OUTPUT)
    if assessment:
        print(f"\n  articles covered: {assessment.articles_covered}")
        print(f"  summary: {assessment.summary[:200]}")
        print(
            f"  severity: {assessment.overall_severity} | confidence: {assessment.confidence:.0%}"
        )
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
    graph.add_node("execute", make_execute_node(get_model(), [list_articles, describe_article]))
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
        result = await workflow.execute({KEY_PROMPT: "What does KB-2024-00101 affect?"})
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
