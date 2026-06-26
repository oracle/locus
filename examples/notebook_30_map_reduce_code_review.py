#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 30: customer-support triage at scale — scatter-gather with Send.

Three inbound tickets, three triage lenses (sentiment, routing, resolution)
= nine analyst agents running in parallel, then one synthesizer collapses
everything into a single triage summary. Each lens emits a structured tag —
SENTIMENT, QUEUE, PRIORITY — so the output lands in vocabulary a helpdesk
or routing pipeline already speaks.

- ``Send(node, payload, metadata)`` is a first-class graph primitive.
  The splitter node returns a list of Sends; the executor fans them out
  and runs them concurrently. No queues, no manual ``asyncio.gather``.
- Each analyst is a distinct ``Agent`` with its own lens-specific
  system prompt — the graph orchestrates them, not a hand-rolled loop.
- The synthesizer reads each Send's output back from merged state and
  emits a single Markdown triage summary.
- The whole pipeline is one ``StateGraph.execute`` call. Streaming,
  cancellation, checkpointing, and GSAR judgment attach for free.

Run it:
    .venv/bin/python examples/notebook_30_map_reduce_code_review.py

The default provider is the bundled mock model. Set TULIP_MODEL_PROVIDER=openai
(or anthropic) and the matching credentials to use a live model. Set
``TULIP_MODEL_PROVIDER=mock`` for offline runs.

Prerequisites:
- Notebook 16 (basic graph).
- Notebook 24 (Swarm) for the dynamic-claim counterpoint.
"""

from __future__ import annotations

import asyncio
from typing import Any

from config import get_model

from tulip.agent import Agent, AgentConfig
from tulip.core.send import Send
from tulip.multiagent.graph import END, START, StateGraph


# ----------------------------------------------------------------------------
# Three independent tickets to triage in parallel. Each message carries a
# concrete, quotable issue so the analysts have something specific to cite.
#   T-4821 — duplicate billing charge, refund request (billing)
#   T-4822 — login failure / app hang, time-sensitive (technical)
#   T-4823 — cancellation / churn risk, open to a downgrade (account)
# ----------------------------------------------------------------------------

SAMPLE_TICKETS = {
    "T-4821": (
        "I was charged twice for my June subscription — $29.99 on the 3rd and "
        "again on the 5th. I only have one account. Please refund the duplicate "
        "charge today; this is the second time this has happened."
    ),
    "T-4822": (
        "I've been trying to log in for the past hour and the app just spins on a "
        "blank screen. I'm on the latest iOS build. I have a client demo in 30 "
        "minutes and I'm locked out — terrible timing."
    ),
    "T-4823": (
        "I'd like to cancel my plan. The product is fine but I'm not using it "
        "enough to justify the cost, and the renewal hit before I could downgrade. "
        "If there's a cheaper tier I might stay; otherwise please close the account."
    ),
}


# ----------------------------------------------------------------------------
# Triage lenses — each runs as its own Agent with a lens-specific prompt
# ----------------------------------------------------------------------------

ANALYST_ROLES = {
    "sentiment": (
        "You are a customer-sentiment analyst. Read the support ticket and "
        "gauge the customer's emotional state and frustration level. Quote the "
        "specific phrases that signal how the customer feels. Do not invent "
        "emotions the text does not show. End with a single "
        "line: SENTIMENT=<calm|frustrated|angry|at-risk>."
    ),
    "routing": (
        "You are a support-routing analyst. Classify the ticket into the right "
        "queue (billing, technical, account, or shipping) and name the single "
        "most likely root cause you can support from the message. Quote the line "
        "that decides the category. Do not invent details. End "
        "with: QUEUE=<billing|technical|account|shipping>."
    ),
    "resolution": (
        "You are a resolution analyst. Propose the single best next action to "
        "resolve the ticket — a concrete step the agent can take. Quote the part "
        "of the ticket that drives your recommendation. Be terse. End "
        "with: PRIORITY=<low|medium|high|urgent>."
    ),
}


def _make_analyst(role: str, model: Any) -> Agent:
    return Agent(
        config=AgentConfig(
            agent_id=f"analyst-{role}",
            model=model,
            system_prompt=ANALYST_ROLES[role],
            # One model call is enough to triage a short ticket.
            max_iterations=2,
            max_tokens=400,
        )
    )


# ----------------------------------------------------------------------------
# Graph nodes
# ----------------------------------------------------------------------------


async def split_tickets(state: dict[str, Any]) -> list[Send]:
    """Fan out: one Send per (ticket, lens) — 3 × 3 = 9 analysts, all concurrent."""
    tickets: dict[str, str] = state["tickets"]
    roles = list(ANALYST_ROLES)
    return [
        Send(
            node="analyze_one",
            payload={"ticket_id": ticket_id, "message": message, "role": role},
            metadata={"ticket": ticket_id, "role": role},
        )
        for ticket_id, message in tickets.items()
        for role in roles
    ]


async def analyze_one(state: dict[str, Any]) -> dict[str, Any]:
    """One analyst Agent against one (ticket, lens).

    Uses ``async for event in agent.run(...)`` instead of ``run_sync()``
    so the 9 instances run truly in parallel inside the graph's
    ``asyncio.gather`` — ``run_sync`` would serialise them on a shared
    thread-pool worker.
    """
    from tulip.core.events import TerminateEvent

    role: str = state["role"]
    ticket_id: str = state["ticket_id"]
    message: str = state["message"]
    model = state["__model__"]
    agent = _make_analyst(role, model)
    prompt = (
        f"Ticket: {ticket_id}\n"
        f"Lens: {role}\n\n"
        f'"""\n{message}\n"""\n\n'
        f"Analyze the ticket through the {role} lens."
    )
    final_msg: str = ""
    iterations = 0
    async for event in agent.run(prompt):
        if isinstance(event, TerminateEvent):
            final_msg = event.final_message or ""
            iterations = event.iterations_used
    return {
        "analysis": {
            "ticket": ticket_id,
            "role": role,
            "comments": final_msg.strip(),
            "iterations": iterations,
        }
    }


async def synthesize(state: dict[str, Any]) -> dict[str, Any]:
    """Reduce: walk the merged state, collect every ``analysis`` payload, render."""
    analyses = [v["analysis"] for v in state.values() if isinstance(v, dict) and "analysis" in v]
    by_ticket: dict[str, list[dict[str, Any]]] = {}
    for a in analyses:
        by_ticket.setdefault(a["ticket"], []).append(a)

    lines = ["# Customer-support triage — summary report", ""]
    for ticket_id in sorted(by_ticket):
        lines.append(f"## {ticket_id}")
        for a in sorted(by_ticket[ticket_id], key=lambda x: x["role"]):
            lines.append(f"### {a['role']}")
            lines.append(a["comments"])
            lines.append("")
    return {"report": "\n".join(lines), "analysis_count": len(analyses)}


# ----------------------------------------------------------------------------
# Build the graph
# ----------------------------------------------------------------------------


def build_triage_graph(model: Any) -> StateGraph:
    """Wire the three nodes: split → analyze_one (parallel) → synthesize → END.

    The model is threaded through state under ``__model__`` rather than
    captured by closure so the graph stays picklable for checkpointing.
    """
    graph = StateGraph(name="support-triage-crew")
    graph.add_node("split", split_tickets)
    graph.add_node("analyze_one", analyze_one)
    graph.add_node("synthesize", synthesize)

    graph.add_edge(START, "split")
    # No explicit edge "split → analyze_one" — the Sends from ``split``
    # carry their own routing. Once every Send finishes, control returns
    # to ``split``'s adjacency, which points at ``synthesize``.
    graph.add_edge("split", "synthesize")
    graph.add_edge("synthesize", END)
    return graph


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------


async def main() -> None:
    print("Notebook 30: customer-support triage at scale — scatter-gather with Send")
    print("=" * 60)

    model = get_model()
    graph = build_triage_graph(model)

    initial = {"tickets": SAMPLE_TICKETS, "__model__": model}

    print(
        f"\nFanning out {len(SAMPLE_TICKETS)} tickets × {len(ANALYST_ROLES)} lenses "
        f"= {len(SAMPLE_TICKETS) * len(ANALYST_ROLES)} analyst agents in parallel...\n"
    )

    result = await graph.execute(initial)

    print(
        f"Graph completed in {result.duration_ms:.0f} ms across "
        f"{result.iterations} graph iteration(s)"
    )
    print(f"Analyses collected: {result.final_state.get('analysis_count', 0)}")
    print()
    print(result.final_state.get("report", "(no report)"))


if __name__ == "__main__":
    asyncio.run(main())
