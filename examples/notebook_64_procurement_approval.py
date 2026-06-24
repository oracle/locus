#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 64: Vendor security review with risk-tiered approval gates.

Real third-party-risk programs have a tier-based escalation chain::

    Vendor intake (questionnaire on file)
       │
       ▼
    Questionnaire analyst  (summarises the vendor's security questionnaire)
       │
       ▼
    Posture analyst  (assesses data exposure + control posture)
       │
       ▼
    Risk-tier router ── score < 25 ──> auto-approve (low risk)
                     ── 25–49     ──> security-manager approval (interrupt)
                     ── 50–74     ──> manager + GRC approval (two interrupts)
                     ── >= 75     ──> manager + GRC + CISO approval (three interrupts)
       │
       ▼
    Decision recorder  (emits structured VendorDecision)

Each approval gate is a separate interrupt() so a reviewer can come back
to it later. The terminal node is SCRIBE, the SOC's compliance reporter:
it emits a typed VendorDecision Pydantic model that files into the
vendor-risk register without parsing. Third-party AI services widen the
agentic supply chain (OWASP ASI04), so the posture step is where you
weigh attestations (SOC 2, ISO 27001) against what data the vendor would
actually touch.

- Risk-tier router is a plain conditional edge — no DSL, no policy file.
- Each gate is its own node — easy to add a tier, easy to re-order,
  easy to swap a human gate for an automated rule.
- output_schema=VendorDecision keeps SCRIBE's terminal artifact typed.

Run it
    # Default: the bundled mock model (set TULIP_MODEL_PROVIDER for a live provider)
    python examples/notebook_64_procurement_approval.py

    # Offline / no credentials:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_64_procurement_approval.py

    # Pin a strong-enough model for the structured VendorDecision schema:
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


class VendorDecision(BaseModel):
    """Final structured artifact filed into the vendor-risk register."""

    request_id: str
    vendor: str
    service: str
    risk_score: float
    questionnaire_summary: str
    posture_assessment: str
    approvals: list[str] = Field(description="ordered list of approver titles")
    approved_at: str
    status: str = Field(description="approved | denied")


# Specialist prompts.


PROMPTS = {
    "questionnaire": (
        "You are a third-party-risk analyst. Given a vendor's security-"
        "questionnaire excerpt, write a two-sentence summary of the security "
        "posture the vendor claims."
    ),
    "posture": (
        "You are a vendor security assessor. Given a vendor and the service it "
        "provides, write a one-paragraph assessment covering: the data the "
        "vendor would touch, the attestations you would expect (SOC 2, ISO "
        "27001), and any obvious exposure concerns for this category."
    ),
}


def _make_agent(role: str, model: Any) -> Agent:
    return Agent(
        config=AgentConfig(
            agent_id=f"vsr-{role}",
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


async def summarize_questionnaire(state: dict[str, Any]) -> dict[str, Any]:
    agent = _make_agent("questionnaire", state["__model__"])
    text = await _run(
        agent,
        f"Vendor: {state['vendor']}\nQuestionnaire excerpt: {state['questionnaire']}",
    )
    return {"questionnaire_summary": text}


async def assess_posture(state: dict[str, Any]) -> dict[str, Any]:
    agent = _make_agent("posture", state["__model__"])
    text = await _run(agent, f"Vendor: {state['vendor']}\nService: {state['service']}")
    return {"posture_assessment": text}


def risk_tier_router(state: dict[str, Any]) -> str:
    """Route by risk score; each tier picks up the prior tier's approvals."""
    score = float(state.get("risk_score", 0.0))
    if score < 25:
        return "auto"
    if score < 50:
        return "manager"
    if score < 75:
        return "grc"
    return "ciso"


async def approve_manager(state: dict[str, Any]) -> dict[str, Any]:
    decision = interrupt(
        {
            "type": "approval",
            "tier": "manager",
            "question": (
                f"Security-manager approval needed at risk score "
                f"{state['risk_score']:.0f}/100 "
                f"({state['vendor']} — {state['service']}). Approve?"
            ),
            "options": ["yes", "no"],
        }
    )
    return _record_decision(state, "Security Manager", decision)


async def approve_grc(state: dict[str, Any]) -> dict[str, Any]:
    decision = interrupt(
        {
            "type": "approval",
            "tier": "grc",
            "question": (
                f"GRC approval needed at risk score {state['risk_score']:.0f}/100 "
                f"({state['vendor']}). Approve?"
            ),
            "options": ["yes", "no"],
        }
    )
    return _record_decision(state, "GRC Lead", decision)


async def approve_ciso(state: dict[str, Any]) -> dict[str, Any]:
    decision = interrupt(
        {
            "type": "approval",
            "tier": "ciso",
            "question": (
                f"CISO approval needed at risk score {state['risk_score']:.0f}/100 "
                f"({state['vendor']}). Approve?"
            ),
            "options": ["yes", "no"],
        }
    )
    return _record_decision(state, "CISO", decision)


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
        return "approve_grc"
    return "record_decision"


def gate_after_grc(state: dict[str, Any]) -> str:
    if state.get("status") == "denied":
        return "record_decision"
    score = float(state.get("risk_score", 0.0))
    if score >= 75:
        return "approve_ciso"
    return "record_decision"


async def auto_approve(state: dict[str, Any]) -> dict[str, Any]:
    return {"approvals": ["AUTO (low risk tier)"], "status": "pending"}


async def record_decision(state: dict[str, Any]) -> dict[str, Any]:
    """SCRIBE writes the record via Agent.output_schema=VendorDecision.

    Routing through an Agent with output_schema means the artifact is a
    typed Pydantic instance — the workflow can POST it directly to the
    vendor-risk register without parsing.
    """
    import asyncio as _asyncio

    final_status = "approved" if state.get("status") != "denied" else "denied"
    agent = Agent(
        config=AgentConfig(
            agent_id="scribe-decision-recorder",
            model=state["__model__"],
            system_prompt=(
                "You are a vendor-risk officer producing a VendorDecision. "
                "Use the supplied fields verbatim. Don't invent vendors or scores."
            ),
            output_schema=VendorDecision,
            max_iterations=2,
            max_tokens=300,
        )
    )
    prompt = (
        f"Request: {state.get('request_id')}\n"
        f"Vendor: {state['vendor']}\n"
        f"Service: {state['service']}\n"
        f"Risk score: {float(state['risk_score'])}\n"
        f"Status: {final_status}\n"
        f"Approvals: {state.get('approvals', [])}\n"
        f"Questionnaire summary: {state.get('questionnaire_summary', '')[:200]}\n"
        f"Posture assessment: {state.get('posture_assessment', '')[:200]}\n\n"
        "Emit the VendorDecision."
    )
    last_exc: BaseException | None = None
    result = None
    for attempt in range(3):
        try:
            result = await _asyncio.to_thread(agent.run_sync, prompt)
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
            "Decision recorder returned no parsed VendorDecision. The configured "
            "model could not honor the JSON schema. Use a stronger model "
            "(e.g. openai.gpt-4o, openai.gpt-5, anthropic.claude-3-5-sonnet) "
            f"for notebook 64. Raw output: {result.message!r}"
        )
    return {"vendor_decision": decision}


# Build the vendor-review graph.


def build_review_graph() -> StateGraph:
    g = StateGraph(name="vendor-security-review")
    g.add_node("summarize_questionnaire", summarize_questionnaire)
    g.add_node("assess_posture", assess_posture)
    g.add_node("auto_approve", auto_approve)
    g.add_node("approve_manager", approve_manager)
    g.add_node("approve_grc", approve_grc)
    g.add_node("approve_ciso", approve_ciso)
    g.add_node("record_decision", record_decision)

    g.add_edge(START, "summarize_questionnaire")
    g.add_edge("summarize_questionnaire", "assess_posture")
    g.add_conditional_edges(
        "assess_posture",
        risk_tier_router,
        targets={
            "auto": "auto_approve",
            "manager": "approve_manager",
            "grc": "approve_manager",
            "ciso": "approve_manager",
        },
    )
    g.add_edge("auto_approve", "record_decision")
    g.add_conditional_edges(
        "approve_manager",
        gate_after_manager,
        targets={"approve_grc": "approve_grc", "record_decision": "record_decision"},
    )
    g.add_conditional_edges(
        "approve_grc",
        gate_after_grc,
        targets={"approve_ciso": "approve_ciso", "record_decision": "record_decision"},
    )
    g.add_edge("approve_ciso", "record_decision")
    g.add_edge("record_decision", END)
    return g


# Driver.


def _print_decision(d: VendorDecision | None) -> None:
    print("\nVendor decision:")
    print("-" * 60)
    if d is None:
        print("(missing)")
        return
    print(f"  Request:        {d.request_id}")
    print(f"  Status:         {d.status}")
    print(f"  Vendor:         {d.vendor}")
    print(f"  Service:        {d.service}")
    print(f"  Risk score:     {d.risk_score:.0f}/100")
    print(f"  Approvals:      " + (" → ".join(d.approvals) if d.approvals else "(none)"))
    print(f"  Questionnaire:  {d.questionnaire_summary[:120]}")


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
            Command(resume=answer, update={**result.final_state, "__model__": initial["__model__"]})
        )
    return result


async def main() -> None:
    print("Notebook 64: Vendor security review with risk-tiered approval gates")
    print("=" * 60)

    model = get_model()
    graph = build_review_graph()

    # (request, vendor, service, risk score, questionnaire excerpt)
    scenarios = [
        (
            "VSR-1001",
            "PrintFleet Co",
            "office print-management SaaS (no customer data)",
            12.0,
            "SOC 2 Type II on file; SSO enforced; no PII processed.",
        ),
        (
            "VSR-1002",
            "MailMetrics",
            "marketing email analytics (contact emails only)",
            38.0,
            "SOC 2 Type I; MFA optional for admins; data encrypted at rest.",
        ),
        (
            "VSR-1003",
            "WarehouseQL",
            "hosted analytics over a copy of the production database",
            62.0,
            "ISO 27001 claimed, certificate expired; sub-processors undisclosed.",
        ),
        (
            "VSR-1004",
            "CloudPay",
            "payroll processing (employee PII + bank details)",
            88.0,
            "No current attestation; breach disclosed in 2025; remediation in progress.",
        ),
    ]

    for req_id, vendor, service, score, questionnaire in scenarios:
        print(f"\n--- {req_id}: risk {score:.0f}/100 — {vendor} — {service} ---")
        initial = {
            "request_id": req_id,
            "vendor": vendor,
            "service": service,
            "risk_score": score,
            "questionnaire": questionnaire,
            "__model__": model,
        }
        result = await _drive(graph, initial, ["yes", "yes", "yes"])
        _print_decision(result.final_state.get("vendor_decision"))


if __name__ == "__main__":
    asyncio.run(main())
