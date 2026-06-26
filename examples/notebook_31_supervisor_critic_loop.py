#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 31: PII-exposure report vs skeptical reviewer — kill unproven claims.

An analyst gathers evidence, an author drafts a data-exposure report, and
a skeptical reviewer (MIRROR — the privacy team's adversarial review role)
either signs off or sends it back because a claim isn't backed by evidence.
The revise loop caps at two passes to bound runtime.

The point of the notebook is the *last* step, not the loop: a report that
reads well is not the same as a report that is grounded. Before anything
ships, the reviewer runs the drafted finding through ``ground_finding`` —
the GSAR grounding gate from ``tulip.security``. A finding is emitted only
when its evidence partition clears the proceed threshold; otherwise the
call returns an ``Abstention`` and nothing reaches the privacy queue. An
unproven exposure claim is a false positive *by construction* and never ships.

- The control flow is a ``StateGraph`` with conditional edges — no
  hand-rolled ``while True`` plus message passing.
- Each role is its own ``Agent`` with a role-specific system prompt.
  No agent can see the others' internal state; they communicate only
  through graph state keys (``notes``, ``draft``, ``revision_request``).
- The reviewer node is where prose review meets mechanical grounding:
  ``ground_finding(...)`` scores the evidence partition and returns a
  ``Evidence`` or an ``Abstention``. ``is_finding(...)`` narrows the union.
- ``stream(mode=StreamMode.NODES)`` emits one event per node completion,
  so a UI can show "Evidence gathered / Author drafting / Reviewer
  adjudicating…" with no extra code.
- ``execute(...)`` returns the authoritative final state plus a
  ``GraphResult`` with timing and iteration metrics.

```text
START → gather → draft → review → END (ship grounded Evidence | abstain)
                   ↑        │
                   └── revise (cap: 2)
```

Run it:
    .venv/bin/python examples/notebook_31_supervisor_critic_loop.py

The default provider is the bundled mock model. Set TULIP_MODEL_PROVIDER=openai
(or anthropic) and the matching credentials to use a live model. Set
``TULIP_MODEL_PROVIDER=mock`` for offline runs — the grounding gate is
deterministic and exercises the same admit/abstain path either way.

Prerequisites:
- Notebook 16 (basic graph).
- Notebook 25 (agent handoff) for an alternative shape.
- Notebook 37 (GSAR grounded findings) for the grounding primitive in depth.
"""

from __future__ import annotations

import asyncio
from typing import Any

from config import get_model

from tulip.agent import Agent, AgentConfig
from tulip.core.events import TerminateEvent
from tulip.multiagent.graph import END, START, StateGraph
from tulip.reasoning.gsar import Claim, EvidenceType, Partition
from tulip.security import (
    AtlasTechnique,
    Indicator,
    IndicatorType,
    OwaspLLM,
    Severity,
    ground_finding,
    is_finding,
)


# ---------------------------------------------------------------------------
# Each role is a real Agent with a role-specific system prompt
# ---------------------------------------------------------------------------


def _make_agent(role: str, system_prompt: str, model: Any, max_iterations: int = 2) -> Agent:
    return Agent(
        config=AgentConfig(
            agent_id=f"agent-{role}",
            model=model,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            max_tokens=400,
        )
    )


SUPERVISOR_PROMPT = (
    "You are the privacy-review team lead. Given the suspected PII exposure "
    "and the current state, decide whether the Analyst, Author, or Reviewer "
    "should run next. Respond with ONE word: gather, draft, or review."
)

ANALYST_PROMPT = (
    "You are a data-privacy analyst. Given a suspected PII exposure, return "
    "3-5 concise evidence notes drawn from DLP scan output and data-lineage "
    "review. No speculation. Bullet points only."
)

AUTHOR_PROMPT = (
    "You are a privacy-incident report author. Given evidence notes (and "
    "optionally a reviewer's revision request), produce a concise 1-2 "
    "paragraph report. State only what the evidence supports. Plain prose."
)

# MIRROR is the privacy team's skeptical / adversarial review role: it
# reflects the author's claims back against the evidence and refuses
# anything that can't be traced to it.
REVIEWER_PROMPT = (
    "You are a skeptical privacy reviewer. Read the draft report and kill "
    "unproven claims: every stated impact must trace to the evidence notes. "
    "If the report is defensible, respond with exactly: APPROVE. "
    "If not, respond with: REVISE: <one-line specific instruction>."
)


# ---------------------------------------------------------------------------
# The evidence partition — what the reviewer's grounding gate scores.
#
# In a live run the analyst's notes would be parsed into typed claims; here
# the partition is built deterministically from the seeded DLP scan facts so
# the grounding gate exercises the same admit/abstain path under the mock
# model. Each claim carries the provenance label GSAR weights against:
# a scanner/tool row outranks inference, which outranks domain priors.
# ---------------------------------------------------------------------------


def _evidence_partition(state: dict[str, Any]) -> Partition:
    """Build the GSAR partition for the drafted finding.

    The grounded claims trace to DLP scan rows and the data-lineage graph;
    the lone unproven claim ("records were accessed by a third party") sits
    in ``ungrounded`` until access logs confirm it.
    """
    return Partition(
        grounded=[
            Claim(
                text="Unmasked 'email' column is selected in the customer_export view.",
                type=EvidenceType.TOOL_MATCH,
                evidence_refs=["dlp-scan:DLP-4471:analytics/customer_export.sql:23:class=email"],
            ),
            Claim(
                text="The column is exported with no masking or hashing applied.",
                type=EvidenceType.SPECIFIC_DATA,
                evidence_refs=["schema-review:analytics/customer_export.sql:23"],
            ),
            Claim(
                text="Data lineage shows customer_export feeds the public BI dashboard.",
                type=EvidenceType.COMPLEMENTARY_FINDING,
                evidence_refs=["lineage:DLP-4475:customer_export->bi_public_dashboard"],
            ),
        ],
        # The author wanted to assert that the records were actually exfiltrated,
        # but nothing in evidence proves any unauthorized access occurred — so
        # MIRROR's gate keeps that claim out of the grounded set.
        ungrounded=[
            Claim(
                text="The exposed records were accessed by an unauthorized third party.",
                type=EvidenceType.INFERENCE,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Drive an Agent inside a graph node and return the final text
# ---------------------------------------------------------------------------


async def _run_agent(agent: Agent, prompt: str) -> str:
    final = ""
    async for event in agent.run(prompt):
        if isinstance(event, TerminateEvent):
            final = event.final_message or ""
    return final.strip()


# ---------------------------------------------------------------------------
# Graph nodes — one per role
# ---------------------------------------------------------------------------


async def gather_node(state: dict[str, Any]) -> dict[str, Any]:
    agent = _make_agent("analyst", ANALYST_PROMPT, state["__model__"])
    notes = await _run_agent(agent, f"Suspected PII exposure: {state['finding']}")
    return {"notes": notes}


async def draft_node(state: dict[str, Any]) -> dict[str, Any]:
    agent = _make_agent("author", AUTHOR_PROMPT, state["__model__"])
    revision = state.get("revision_request", "")
    prompt = f"Evidence: {state['finding']}\nEvidence notes:\n{state.get('notes', '')}\n"
    if revision:
        prompt += f"\nReviewer feedback (apply this): {revision}\n"
    prompt += "\nWrite the report now."

    draft = await _run_agent(agent, prompt)
    revisions_done = state.get("revisions_done", 0) + (1 if revision else 0)
    return {"draft": draft, "revisions_done": revisions_done}


async def review_node(state: dict[str, Any]) -> dict[str, Any]:
    """Prose review, then the mechanical grounding gate.

    First the reviewer Agent gives a prose verdict (APPROVE / REVISE). Then
    — regardless of how persuasive the prose is — the drafted finding is
    run through ``ground_finding``. The function returns a ``Evidence`` only
    when the evidence partition clears the GSAR proceed threshold; below it
    the caller gets an ``Abstention`` and nothing ships. That is what makes
    "kill unproven claims" a guarantee rather than a hope.
    """
    agent = _make_agent("reviewer", REVIEWER_PROMPT, state["__model__"])
    verdict = await _run_agent(agent, f"Draft report to review:\n{state.get('draft', '')}")
    approved = verdict.strip().upper().startswith("APPROVE")
    revision_request = "" if approved else verdict

    # The grounding gate. ground_finding scores the evidence partition and
    # returns Evidence | Abstention — an ungrounded claim cannot become a
    # Evidence. Tag with the relevant taxonomy IDs so the artifact is
    # portable into a privacy register or DPIA (unmasked-PII exposure is
    # CWE-359; in the AI-stack threat model it maps to OWASP LLM02 sensitive
    # information disclosure and MITRE ATLAS exfiltration via an agent tool).
    result = ground_finding(
        title="Unmasked PII (email) exposed via analytics customer_export view",
        description=(
            "The 'email' column is exported without masking in "
            "analytics/customer_export.sql:23 and flows to a public BI "
            "dashboard; confirmed by DLP scan DLP-4471 and a lineage trace."
        ),
        severity=Severity.HIGH,
        asset="analytics-export:customer_export",
        remediation="Mask or hash the 'email' column, or drop it from the dashboard feed.",
        partition=_evidence_partition(state),
        indicators=[Indicator(type=IndicatorType.HOST, value="analytics.corp.example")],
        taxonomy=[
            OwaspLLM.SENSITIVE_INFORMATION_DISCLOSURE,
            AtlasTechnique.EXFILTRATION_VIA_AGENT_TOOL,
        ],
    )

    return {
        "approved": approved,
        "revision_request": revision_request,
        "reviewer_verdict": verdict,
        "grounded_result": result,
    }


# ---------------------------------------------------------------------------
# Conditional routing — approve, or send back to the author (capped)
# ---------------------------------------------------------------------------


def route_after_review(state: dict[str, Any]) -> str:
    if state.get("approved"):
        return "done"
    if state.get("revisions_done", 0) >= 2:
        return "done"
    return "revise"


# ---------------------------------------------------------------------------
# Wire it: gather → draft → review → (revise → draft | done → END)
# ---------------------------------------------------------------------------


def build_supervisor_graph() -> StateGraph:
    graph = StateGraph(name="pii-exposure-review-loop")
    graph.add_node("gather", gather_node)
    graph.add_node("draft", draft_node)
    graph.add_node("review", review_node)

    graph.add_edge(START, "gather")
    graph.add_edge("gather", "draft")
    graph.add_edge("draft", "review")
    graph.add_conditional_edges(
        "review",
        route_after_review,
        targets={"revise": "draft", "done": END},
    )
    return graph


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def main() -> None:
    print("Notebook 31: PII-exposure report vs skeptical reviewer")
    print("=" * 60)

    model = get_model()
    graph = build_supervisor_graph()

    initial = {
        "finding": (
            "Suspected PII exposure in the analytics-export pipeline — DLP "
            "scan DLP-4471 flagged an unmasked 'email' column in the "
            "customer_export view"
        ),
        "__model__": model,
    }

    print(f"\nFinding: {initial['finding']!r}\n")

    # Stream node-completion events for live UI feedback, then call
    # execute() for the authoritative final state with metrics.
    from tulip.multiagent.graph import StreamMode

    async for event in graph.stream(initial, mode=StreamMode.NODES):
        if event.node_id:
            print(f"  ✓ {event.node_id}", flush=True)

    final = await graph.execute(initial)
    final_state = final.final_state

    print()
    print(f"Revisions:    {final_state.get('revisions_done', 0)}")
    verdict = final_state.get("reviewer_verdict") or "(unknown)"
    print(f"Reviewer:     {verdict[:80]}")
    print(f"Total time:   ~{final.duration_ms:.0f} ms across {final.iterations} graph iterations")

    # The grounding gate decides what actually ships.
    print()
    print("Grounding gate:")
    print("-" * 60)
    result = final_state.get("grounded_result")
    if result is not None and is_finding(result):
        print(f"  SHIPPED — {result.title}")
        print(f"  severity={result.severity}  gsar_score={result.gsar_score:.2f}")
        print(f"  taxonomy={[str(t) for t in result.taxonomy]}")
        print(f"  evidence_refs={result.evidence_refs}")
    elif result is not None:
        print(f"  WITHHELD ({result.decision}) — {result.candidate_title}")
        print(f"  reason: {result.reason}")
    else:
        print("  (no result)")

    print()
    print("Draft report (prose):")
    print("-" * 60)
    print(final_state.get("draft", "(no draft)"))


if __name__ == "__main__":
    asyncio.run(main())
