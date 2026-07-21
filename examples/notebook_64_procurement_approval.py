#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 64: Customer-support concession approval with risk-tiered gates.

Before a support agent can grant a costly concession — a refund, a
service credit, a goodwill gesture, a contract-level make-good — the
request has to clear a tier-based escalation chain. The more it costs
and the more precedent it sets, the more approvals it takes: a blanket
"yes" to every angry customer drains margin and trains customers to
escalate, so larger concessions climb the ladder::

    Ticket intake (case history on file)
       │
       ▼
    Ticket analyst  (summarises what the customer is asking for and why)
       │
       ▼
    Impact analyst  (concession cost + precedent risk + churn downside)
       │
       ▼
    Risk-tier router ── score < 25 ──> auto-approve (small credit, low cost)
                     ── 25–49     ──> support-manager approval (interrupt)
                     ── 50–74     ──> manager + billing approval (two interrupts)
                     ── >= 75     ──> manager + billing + director approval (three interrupts)
       │
       ▼
    Decision recorder  (emits structured ConcessionDecision)

Each approval gate is a separate interrupt() so a reviewer can come back
to it later. The terminal node is SCRIBE, the support org's case
recorder: it emits a typed ConcessionDecision Pydantic model that files
into the concessions ledger without parsing. A large concession spends
real money and sets a precedent other customers will cite, so the impact
step is where you weigh the customer's standing and lifetime value
against the cost of the make-good and the downside of a denial.

- Risk-tier router is a plain conditional edge — no DSL, no policy file.
- Each gate is its own node — easy to add a tier, easy to re-order,
  easy to swap a human gate for an automated rule.
- output_schema=ConcessionDecision keeps SCRIBE's terminal artifact typed.

Run it
    # Default: the bundled mock model (set TULIP_MODEL_PROVIDER for a live provider)
    python examples/notebook_64_procurement_approval.py

    # Offline / no credentials:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_64_procurement_approval.py

    # Pin a strong-enough model for the structured ConcessionDecision schema:
    TULIP_MODEL_ID=openai.gpt-4.1 python examples/notebook_64_procurement_approval.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from config import get_model
from pydantic import BaseModel, Field

from tulip.agent import Agent, AgentConfig
from tulip.core import Command, interrupt
from tulip.core.events import TerminateEvent
from tulip.multiagent.graph import END, START, StateGraph


# Data shape for the terminal artifact.


class ConcessionDecision(BaseModel):
    """Final structured artifact filed into the customer-concessions ledger."""

    request_id: str
    customer: str
    concession: str
    risk_score: float
    ticket_summary: str
    impact_assessment: str
    approvals: list[str] = Field(description="ordered list of approver titles")
    approved_at: str
    status: str = Field(description="approved | denied")


# Specialist prompts.


PROMPTS = {
    "ticket": (
        "You are a customer-support triage analyst. Given a support ticket "
        "excerpt, write a two-sentence summary of what the customer is asking "
        "for and the reason they give."
    ),
    "impact": (
        "You are a customer-support concession assessor. Given a customer and "
        "the remedy they are requesting, write a one-paragraph assessment "
        "covering: the cost of the concession to the business, the precedent "
        "risk (will granting it set an expectation other customers will cite), "
        "the customer's standing and lifetime value, and the downside if the "
        "concession is denied (churn, escalation, public complaint)."
    ),
}


def _make_agent(role: str, model: Any) -> Agent:
    return Agent(
        config=AgentConfig(
            agent_id=f"concession-{role}",
            model=model,
            system_prompt=PROMPTS[role],
            max_iterations=2,
            max_tokens=300,
        )
    )


async def _run(agent: Agent, prompt: str) -> str:
    final = ""
    async for event in agent.run(prompt):
        if isinstance(event, TerminateEvent):
            final = event.final_message or ""
    return final.strip()


# Graph nodes.


async def summarize_ticket(state: dict[str, Any]) -> dict[str, Any]:
    agent = _make_agent("ticket", state["__model__"])
    text = await _run(
        agent,
        f"Customer: {state['customer']}\nTicket excerpt: {state['ticket']}",
    )
    return {"ticket_summary": text}


async def assess_impact(state: dict[str, Any]) -> dict[str, Any]:
    agent = _make_agent("impact", state["__model__"])
    text = await _run(
        agent, f"Customer: {state['customer']}\nRequested remedy: {state['concession']}"
    )
    return {"impact_assessment": text}


def risk_tier_router(state: dict[str, Any]) -> str:
    """Route by risk score; each tier picks up the prior tier's approvals."""
    score = float(state.get("risk_score", 0.0))
    if score < 25:
        return "auto"
    if score < 50:
        return "manager"
    if score < 75:
        return "billing"
    return "director"


async def approve_manager(state: dict[str, Any]) -> dict[str, Any]:
    decision = interrupt(
        {
            "type": "approval",
            "tier": "manager",
            "question": (
                f"Support-manager approval needed at risk score "
                f"{state['risk_score']:.0f}/100 "
                f"({state['customer']} — {state['concession']}). Approve?"
            ),
            "options": ["yes", "no"],
        }
    )
    return _record_decision(state, "Support Manager", decision)


async def approve_billing(state: dict[str, Any]) -> dict[str, Any]:
    decision = interrupt(
        {
            "type": "approval",
            "tier": "billing",
            "question": (
                f"Billing approval needed at risk score {state['risk_score']:.0f}/100 "
                f"({state['customer']}). Approve?"
            ),
            "options": ["yes", "no"],
        }
    )
    return _record_decision(state, "Billing Lead", decision)


async def approve_director(state: dict[str, Any]) -> dict[str, Any]:
    decision = interrupt(
        {
            "type": "approval",
            "tier": "director",
            "question": (
                f"Support-director approval needed at risk score "
                f"{state['risk_score']:.0f}/100 "
                f"({state['customer']}). Approve?"
            ),
            "options": ["yes", "no"],
        }
    )
    return _record_decision(state, "Support Director", decision)


def _record_decision(state: dict[str, Any], role: str, decision: str) -> dict[str, Any]:
    approvals: list[str] = list(state.get("approvals", []))
    if decision == "yes":
        approvals.append(role)
        return {"approvals": approvals, "status": "pending"}
    return {"approvals": approvals, "status": "denied"}


def gate_after_manager(state: dict[str, Any]) -> str:
    """A denial at any tier jumps straight to the decision node (status=denied)."""
    if state.get("status") == "denied":
        return "record_decision"
    score = float(state.get("risk_score", 0.0))
    if score >= 50:
        return "approve_billing"
    return "record_decision"


def gate_after_billing(state: dict[str, Any]) -> str:
    if state.get("status") == "denied":
        return "record_decision"
    score = float(state.get("risk_score", 0.0))
    if score >= 75:
        return "approve_director"
    return "record_decision"


async def auto_approve(state: dict[str, Any]) -> dict[str, Any]:
    return {"approvals": ["AUTO (low cost tier)"], "status": "pending"}


async def record_decision(state: dict[str, Any]) -> dict[str, Any]:
    """SCRIBE writes the record via Agent.output_schema=ConcessionDecision.

    Routing through an Agent with output_schema means the artifact is a
    typed Pydantic instance — the workflow can POST it directly to the
    concessions ledger without parsing.
    """
    import asyncio as _asyncio

    final_status = "approved" if state.get("status") != "denied" else "denied"
    agent = Agent(
        config=AgentConfig(
            agent_id="scribe-decision-recorder",
            model=state["__model__"],
            system_prompt=(
                "You are a customer-support case officer producing a "
                "ConcessionDecision. Use the supplied fields verbatim. Don't "
                "invent customers or scores."
            ),
            output_schema=ConcessionDecision,
            max_iterations=2,
            max_tokens=300,
        )
    )
    prompt = (
        f"Request: {state.get('request_id')}\n"
        f"Customer: {state['customer']}\n"
        f"Concession: {state['concession']}\n"
        f"Risk score: {float(state['risk_score'])}\n"
        f"Status: {final_status}\n"
        f"Approvals: {state.get('approvals', [])}\n"
        f"Ticket summary: {state.get('ticket_summary', '')[:200]}\n"
        f"Impact assessment: {state.get('impact_assessment', '')[:200]}\n\n"
        "Emit the ConcessionDecision."
    )
    last_exc: BaseException | None = None
    result = None
    for attempt in range(3):
        try:
            result = await agent.arun(prompt)
            break
        except Exception as exc:  # noqa: BLE001 — retry transient provider flakiness
            last_exc = exc
            await _asyncio.sleep(0.5 * (attempt + 1))
    if result is None:
        raise RuntimeError(
            f"Decision recorder failed after 3 attempts. Last error: {last_exc!r}"
        ) from last_exc
    decision = result.parsed
    if decision is None:
        raise RuntimeError(
            "Decision recorder returned no parsed ConcessionDecision. The "
            "configured model could not honor the JSON schema. Use a stronger "
            "model (e.g. openai.gpt-4o, openai.gpt-5, anthropic.claude-3-5-sonnet) "
            f"for notebook 64. Raw output: {result.message!r}"
        )
    return {"concession_decision": decision}


# Build the concession-review graph.


def build_review_graph() -> StateGraph:
    g = StateGraph(name="concession-approval-review")
    g.add_node("summarize_ticket", summarize_ticket)
    g.add_node("assess_impact", assess_impact)
    g.add_node("auto_approve", auto_approve)
    g.add_node("approve_manager", approve_manager)
    g.add_node("approve_billing", approve_billing)
    g.add_node("approve_director", approve_director)
    g.add_node("record_decision", record_decision)

    g.add_edge(START, "summarize_ticket")
    g.add_edge("summarize_ticket", "assess_impact")
    g.add_conditional_edges(
        "assess_impact",
        risk_tier_router,
        targets={
            "auto": "auto_approve",
            "manager": "approve_manager",
            "billing": "approve_manager",
            "director": "approve_manager",
        },
    )
    g.add_edge("auto_approve", "record_decision")
    g.add_conditional_edges(
        "approve_manager",
        gate_after_manager,
        targets={
            "approve_billing": "approve_billing",
            "record_decision": "record_decision",
        },
    )
    g.add_conditional_edges(
        "approve_billing",
        gate_after_billing,
        targets={
            "approve_director": "approve_director",
            "record_decision": "record_decision",
        },
    )
    g.add_edge("approve_director", "record_decision")
    g.add_edge("record_decision", END)
    return g


# Driver.


def _print_decision(d: ConcessionDecision | None) -> None:
    print("\nConcession decision:")
    print("-" * 60)
    if d is None:
        print("(missing)")
        return
    print(f"  Request:        {d.request_id}")
    print(f"  Status:         {d.status}")
    print(f"  Customer:       {d.customer}")
    print(f"  Concession:     {d.concession}")
    print(f"  Risk score:     {d.risk_score:.0f}/100")
    print(f"  Approvals:      " + (" → ".join(d.approvals) if d.approvals else "(none)"))
    print(f"  Ticket:         {d.ticket_summary[:120]}")


async def _drive(graph: StateGraph, initial: dict[str, Any], answers: list[str]) -> Any:
    """Run the graph, auto-resuming interrupts with the supplied answers."""
    result = await graph.execute(initial)
    answer_idx = 0
    while result.interrupt:
        answer = answers[answer_idx] if answer_idx < len(answers) else "yes"
        answer_idx += 1
        payload = result.interrupt.interrupt.payload
        print(f"  ⏸  [{payload.get('tier', '?')}] {payload.get('question')}")
        print(f"  ▶  Reviewer responds: {answer!r}")
        result = await graph.execute(
            Command(
                resume=answer,
                update={**result.final_state, "__model__": initial["__model__"]},
            )
        )
    return result


async def main() -> None:
    print("Notebook 64: Customer-support concession approval with risk-tiered gates")
    print("=" * 60)

    model = get_model()
    graph = build_review_graph()

    # (request, customer, concession, risk score, ticket excerpt)
    scenarios = [
        (
            "CS-1001",
            "Dana R. (Starter plan)",
            "$15 account credit for a late shipping notification",
            12.0,
            "Polite first-time complaint; order arrived two days late; asks for a small credit.",
        ),
        (
            "CS-1002",
            "Marco T. (Pro plan)",
            "one-month subscription refund after a billing double-charge",
            38.0,
            "Charged twice this cycle; wants the duplicate month refunded; otherwise happy customer.",
        ),
        (
            "CS-1003",
            "Acme Studio (Team plan, 40 seats)",
            "full annual refund plus a goodwill credit after repeated outages",
            62.0,
            "Three outages this quarter; threatening to switch; asking to unwind the annual contract.",
        ),
        (
            "CS-1004",
            "Globex Corp (Enterprise, $480k ARR)",
            "SLA-breach service credit and a contract make-good after a data-export failure",
            88.0,
            "Missed an SLA-bound export during their fiscal close; legal CC'd; demanding a contract concession.",
        ),
    ]

    for req_id, customer, concession, score, ticket in scenarios:
        print(f"\n--- {req_id}: risk {score:.0f}/100 — {customer} — {concession} ---")
        initial = {
            "request_id": req_id,
            "customer": customer,
            "concession": concession,
            "risk_score": score,
            "ticket": ticket,
            "__model__": model,
        }
        result = await _drive(graph, initial, ["yes", "yes", "yes"])
        _print_decision(result.final_state.get("concession_decision"))


if __name__ == "__main__":
    asyncio.run(main())
