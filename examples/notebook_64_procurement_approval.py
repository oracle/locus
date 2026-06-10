#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 59: Procurement approval with tiered human gates.

Real procurement workflows have a threshold-based escalation chain::

    Request submitted
       │
       ▼
    Justifier  (drafts business justification)
       │
       ▼
    Vendor analyst  (validates vendor + pricing)
       │
       ▼
    Tier router   ── < $1k     ──> auto-approve
                  ── $1k-$10k  ──> manager approval (interrupt)
                  ── $10k-$100k──> manager + finance approval (two interrupts)
                  ── > $100k   ──> manager + finance + CFO approval (three interrupts)
       │
       ▼
    PO generator  (emits structured PurchaseOrder)

Each approval gate is a separate interrupt() so a reviewer can come back
to it later. The workflow ends with a typed PurchaseOrder Pydantic model
that can be filed into an ERP without parsing.

- Tier router is a plain conditional edge — no DSL, no policy file.
- Each gate is its own node — easy to add a tier, easy to re-order,
  easy to swap a human gate for an automated rule.
- output_schema=PurchaseOrder keeps the terminal artifact typed.

Run it
    # Default: the bundled mock model (set TULIP_MODEL_PROVIDER for a live provider)
    python examples/notebook_64_procurement_approval.py

    # Offline / no credentials:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_64_procurement_approval.py

    # Pin a strong-enough model for the structured PurchaseOrder schema:
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


class PurchaseOrder(BaseModel):
    """Final structured artifact filed into ERP."""

    request_id: str
    vendor: str
    item: str
    amount_usd: float
    business_justification: str
    vendor_assessment: str
    approvals: list[str] = Field(description="ordered list of approver titles")
    approved_at: str
    status: str = Field(description="approved | denied")


# Specialist prompts.


PROMPTS = {
    "justify": (
        "You are a procurement analyst. Given an item and use-case, write a "
        "two-sentence business justification."
    ),
    "vendor": (
        "You are a vendor-risk analyst. Given a vendor name and an item, write "
        "a one-paragraph assessment covering: financial stability, data-handling "
        "posture (if applicable), and pricing reasonableness for this category."
    ),
}


def _make_agent(role: str, model: Any) -> Agent:
    return Agent(
        config=AgentConfig(
            agent_id=f"proc-{role}",
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


async def justify(state: dict[str, Any]) -> dict[str, Any]:
    agent = _make_agent("justify", state["__model__"])
    text = await _run(agent, f"Item: {state['item']}\nUse-case: {state['use_case']}")
    return {"justification": text}


async def assess_vendor(state: dict[str, Any]) -> dict[str, Any]:
    agent = _make_agent("vendor", state["__model__"])
    text = await _run(agent, f"Vendor: {state['vendor']}\nItem: {state['item']}")
    return {"vendor_assessment": text}


def tier_router(state: dict[str, Any]) -> str:
    """Route by amount; each tier picks up the prior tier's approvals."""
    amt = float(state.get("amount_usd", 0.0))
    if amt < 1_000:
        return "auto"
    if amt < 10_000:
        return "manager"
    if amt < 100_000:
        return "finance"
    return "cfo"


async def approve_manager(state: dict[str, Any]) -> dict[str, Any]:
    decision = interrupt(
        {
            "type": "approval",
            "tier": "manager",
            "question": (
                f"Manager approval needed for ${state['amount_usd']:,.2f} "
                f"({state['vendor']} — {state['item']}). Approve?"
            ),
            "options": ["yes", "no"],
        }
    )
    return _record_decision(state, "Manager", decision)


async def approve_finance(state: dict[str, Any]) -> dict[str, Any]:
    decision = interrupt(
        {
            "type": "approval",
            "tier": "finance",
            "question": (
                f"Finance approval needed for ${state['amount_usd']:,.2f} "
                f"({state['vendor']}). Approve?"
            ),
            "options": ["yes", "no"],
        }
    )
    return _record_decision(state, "Finance Director", decision)


async def approve_cfo(state: dict[str, Any]) -> dict[str, Any]:
    decision = interrupt(
        {
            "type": "approval",
            "tier": "cfo",
            "question": (
                f"CFO approval needed for ${state['amount_usd']:,.2f} ({state['vendor']}). Approve?"
            ),
            "options": ["yes", "no"],
        }
    )
    return _record_decision(state, "CFO", decision)


def _record_decision(state: dict[str, Any], role: str, decision: str) -> dict[str, Any]:
    approvals: list[str] = list(state.get("approvals", []))
    if decision == "yes":
        approvals.append(role)
        return {"approvals": approvals, "status": "pending"}
    return {"approvals": approvals, "status": "denied"}


def gate_after_manager(state: dict[str, Any]) -> str:
    """A denial at any tier jumps straight to the PO node (status=denied)."""
    if state.get("status") == "denied":
        return "emit_po"
    amt = float(state.get("amount_usd", 0.0))
    if amt >= 10_000:
        return "approve_finance"
    return "emit_po"


def gate_after_finance(state: dict[str, Any]) -> str:
    if state.get("status") == "denied":
        return "emit_po"
    amt = float(state.get("amount_usd", 0.0))
    if amt >= 100_000:
        return "approve_cfo"
    return "emit_po"


async def auto_approve(state: dict[str, Any]) -> dict[str, Any]:
    return {"approvals": ["AUTO (under threshold)"], "status": "pending"}


async def emit_po(state: dict[str, Any]) -> dict[str, Any]:
    """Build the PO via Agent.output_schema=PurchaseOrder.

    Routing through an Agent with output_schema means the artifact is a
    typed Pydantic instance — the workflow can POST it directly to the
    ERP without parsing.
    """
    import asyncio as _asyncio

    final_status = "approved" if state.get("status") != "denied" else "denied"
    agent = Agent(
        config=AgentConfig(
            agent_id="po-emitter",
            model=state["__model__"],
            system_prompt=(
                "You are a procurement-ops officer producing a PurchaseOrder. "
                "Use the supplied fields verbatim. Don't invent vendors or amounts."
            ),
            output_schema=PurchaseOrder,
            max_iterations=2,
            max_tokens=300,
        )
    )
    prompt = (
        f"Request: {state.get('request_id')}\n"
        f"Vendor: {state['vendor']}\n"
        f"Item: {state['item']}\n"
        f"Amount: {float(state['amount_usd'])}\n"
        f"Status: {final_status}\n"
        f"Approvals: {state.get('approvals', [])}\n"
        f"Justification: {state.get('justification', '')[:200]}\n"
        f"Vendor assessment: {state.get('vendor_assessment', '')[:200]}\n\n"
        "Emit the PurchaseOrder."
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
            f"PO emitter failed after 3 attempts. Last error: {last_exc!r}"
        ) from last_exc
    po = result.parsed
    if po is None:
        raise RuntimeError(
            "PO emitter returned no parsed PurchaseOrder. The configured model "
            "could not honor the JSON schema. Use a stronger model "
            "(e.g. openai.gpt-4o, openai.gpt-5, anthropic.claude-3-5-sonnet) "
            f"for notebook 58. Raw output: {result.message!r}"
        )
    return {"purchase_order": po}


# Build the procurement graph.


def build_procurement_graph() -> StateGraph:
    g = StateGraph(name="procurement-approval")
    g.add_node("justify", justify)
    g.add_node("assess_vendor", assess_vendor)
    g.add_node("auto_approve", auto_approve)
    g.add_node("approve_manager", approve_manager)
    g.add_node("approve_finance", approve_finance)
    g.add_node("approve_cfo", approve_cfo)
    g.add_node("emit_po", emit_po)

    g.add_edge(START, "justify")
    g.add_edge("justify", "assess_vendor")
    g.add_conditional_edges(
        "assess_vendor",
        tier_router,
        targets={
            "auto": "auto_approve",
            "manager": "approve_manager",
            "finance": "approve_manager",
            "cfo": "approve_manager",
        },
    )
    g.add_edge("auto_approve", "emit_po")
    g.add_conditional_edges(
        "approve_manager",
        gate_after_manager,
        targets={"approve_finance": "approve_finance", "emit_po": "emit_po"},
    )
    g.add_conditional_edges(
        "approve_finance",
        gate_after_finance,
        targets={"approve_cfo": "approve_cfo", "emit_po": "emit_po"},
    )
    g.add_edge("approve_cfo", "emit_po")
    g.add_edge("emit_po", END)
    return g


# Driver.


def _print_po(po: PurchaseOrder | None) -> None:
    print("\nPurchase order:")
    print("-" * 60)
    if po is None:
        print("(missing)")
        return
    print(f"  Request:        {po.request_id}")
    print(f"  Status:         {po.status}")
    print(f"  Vendor:         {po.vendor}")
    print(f"  Item:           {po.item}")
    print(f"  Amount:         ${po.amount_usd:,.2f}")
    print(f"  Approvals:      " + (" → ".join(po.approvals) if po.approvals else "(none)"))
    print(f"  Justification:  {po.business_justification[:120]}")


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
    print("Notebook 59: Procurement approval with tiered human gates")
    print("=" * 60)

    model = get_model()
    graph = build_procurement_graph()

    scenarios = [
        ("REQ-1001", "Acme Corp", "USB hubs (10x)", 280.00, "office equipment refresh"),
        ("REQ-1002", "DataDog", "APM annual subscription", 9_500.00, "production observability"),
        ("REQ-1003", "Salesforce", "Sales Cloud (50 seats, 1 yr)", 75_000.00, "GTM ramp"),
        ("REQ-1004", "Acme DB", "Managed database — 12-month commit", 480_000.00, "DB platform"),
    ]

    for req_id, vendor, item, amt, use_case in scenarios:
        print(f"\n--- {req_id}: ${amt:,.2f} — {vendor} — {item} ---")
        initial = {
            "request_id": req_id,
            "vendor": vendor,
            "item": item,
            "amount_usd": amt,
            "use_case": use_case,
            "__model__": model,
        }
        result = await _drive(graph, initial, ["yes", "yes", "yes"])
        _print_po(result.final_state.get("purchase_order"))


if __name__ == "__main__":
    asyncio.run(main())
