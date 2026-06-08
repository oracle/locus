#!/usr/bin/env python3
# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Notebook 34: multi-agent workflows with a human in the loop.

Notebook 20 covered HITL for a single agent. Production systems
typically have a triage agent, several specialists, and a human gate
for irreversible actions. This notebook walks three combinations:

- Pattern A — approval gate: triage classifies a refund, a specialist
  drafts the response, a human approves before it ships.
- Pattern B — human-as-tool: when triage isn't confident, it asks the
  human for the category instead of guessing. The answer becomes part
  of state for downstream specialists.
- Pattern C — long-pause workflow: state survives across an interrupt
  boundary so the human can come back hours later (different process,
  different caller) and the workflow resumes where it left off.

- ``interrupt(payload)`` is a function-level primitive. Any node can
  call it; the graph catches the InterruptException, snapshots state,
  and returns an ``InterruptState`` to the caller.
- Resume with ``graph.execute(Command(resume=<answer>, update=state))``
  — the ``interrupt()`` call returns the resume value.
- Pair with a checkpointer for multi-process / multi-day pauses that
  preserve every specialist's context.

Run it:
    .venv/bin/python examples/notebook_39_multiagent_human_in_loop.py

The default provider is the bundled mock model. Set TULIP_MODEL_PROVIDER=openai
(or anthropic) and the matching credentials to use a live model. Set
``TULIP_MODEL_PROVIDER=mock`` for offline runs.

Prerequisites:
- Notebook 17 (basic graph).
- Notebook 20 (single-agent HITL).
"""

from __future__ import annotations

import asyncio
from typing import Any

from config import get_model

from tulip.agent import Agent, AgentConfig
from tulip.core import Command, interrupt
from tulip.core.events import TerminateEvent
from tulip.multiagent.graph import END, START, StateGraph


# ---------------------------------------------------------------------------
# Specialists — triage classifier and a refund drafter
# ---------------------------------------------------------------------------


def _make_agent(role: str, system_prompt: str, model: Any) -> Agent:
    return Agent(
        config=AgentConfig(
            agent_id=f"agent-{role}",
            model=model,
            system_prompt=system_prompt,
            max_iterations=2,
            max_tokens=300,
        )
    )


TRIAGE_PROMPT = (
    "You are a customer-support triage agent. Read the request and "
    "respond with EXACTLY ONE of: refund, billing, technical, escalate. "
    "Use 'escalate' only when the request is ambiguous or requires "
    "manager judgment."
)
REFUND_PROMPT = (
    "You are a refund specialist. Draft a polite, concise reply confirming "
    "the refund will be processed. Two sentences max."
)


async def _run_agent(agent: Agent, prompt: str) -> str:
    final = ""
    async for event in agent.run(prompt):
        if isinstance(event, TerminateEvent):
            final = event.final_message or ""
    return final.strip()


# ---------------------------------------------------------------------------
# Pattern A — Approval gate
# ---------------------------------------------------------------------------


async def triage_node(state: dict[str, Any]) -> dict[str, Any]:
    agent = _make_agent("triage", TRIAGE_PROMPT, state["__model__"])
    category = await _run_agent(agent, f"Customer request: {state['request']!r}")
    return {"category": category.strip().lower().split()[0] if category else "escalate"}


async def draft_refund_node(state: dict[str, Any]) -> dict[str, Any]:
    agent = _make_agent("refund", REFUND_PROMPT, state["__model__"])
    draft = await _run_agent(
        agent, f"Customer request: {state['request']!r}\nDraft the refund response."
    )
    return {"draft": draft}


async def human_approval_node(state: dict[str, Any]) -> dict[str, Any]:
    """Pause the graph until the human approves or rejects the draft.

    ``interrupt()`` raises an InterruptException; the graph catches it,
    snapshots state, and hands an ``InterruptState`` back to the caller.
    On resume, ``interrupt()`` returns the resume value.
    """
    response = interrupt(
        {
            "type": "approval",
            "question": "Approve this refund response?",
            "draft": state.get("draft", ""),
            "options": ["yes", "no"],
        }
    )
    return {"approved": response == "yes", "human_response": response}


async def send_or_cancel_node(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("approved"):
        return {"result": "✓ Sent to customer", "outcome": "sent"}
    return {"result": "✗ Cancelled by human reviewer", "outcome": "cancelled"}


def build_approval_graph() -> StateGraph:
    g = StateGraph(name="hitl-approval-gate")
    g.add_node("triage", triage_node)
    g.add_node("draft", draft_refund_node)
    g.add_node("approve", human_approval_node)
    g.add_node("send", send_or_cancel_node)
    g.add_edge(START, "triage")
    g.add_edge("triage", "draft")
    g.add_edge("draft", "approve")
    g.add_edge("approve", "send")
    g.add_edge("send", END)
    return g


async def demo_pattern_a(model: Any) -> None:
    print("\n=== Pattern A: Approval gate ===\n")
    graph = build_approval_graph()
    initial = {"request": "I want a refund for order #42 — it never shipped.", "__model__": model}

    # Runs triage, drafts the reply, then pauses at the approval node.
    result = await graph.execute(initial)
    if not result.interrupt:
        print(f"  ✗ unexpected: graph completed without interrupt: {result.final_state}")
        return
    payload = result.interrupt.interrupt.payload
    print(f"  ⏸  Paused at: {result.interrupt.node_id}")
    print(f"     Question:  {payload.get('question')}")
    print(f"     Draft:     {payload.get('draft')}")

    print("  ▶  Human responds: 'yes'")
    final = await graph.execute(
        Command(resume="yes", update=result.final_state),
    )
    print(f"  ✓ Final outcome: {final.final_state.get('result')}")


# ---------------------------------------------------------------------------
# Pattern B — Human-as-tool (escalation when triage isn't confident)
# ---------------------------------------------------------------------------


async def smart_triage_node(state: dict[str, Any]) -> dict[str, Any]:
    """Triage with an explicit escalation fallback to the human."""
    valid = {"refund", "billing", "technical"}
    agent = _make_agent("triage", TRIAGE_PROMPT, state["__model__"])
    raw = await _run_agent(agent, f"Customer request: {state['request']!r}")
    first = (raw.lower().split() or ["escalate"])[0]
    # Anything outside the explicit category set — including the mock
    # model's filler text — falls through to escalation. In production
    # you never want a specialist running with a bogus category.
    category = first if first in valid else "escalate"

    if category == "escalate":
        category = interrupt(
            {
                "type": "escalation",
                "question": (
                    f"Triage agent is not confident. Pick a category for: {state['request']!r}"
                ),
                "options": ["refund", "billing", "technical", "drop"],
            }
        )
    return {"category": category}


async def route_node(state: dict[str, Any]) -> dict[str, Any]:
    return {"final_category": state.get("category", "drop")}


def build_escalation_graph() -> StateGraph:
    g = StateGraph(name="hitl-escalation")
    g.add_node("triage", smart_triage_node)
    g.add_node("route", route_node)
    g.add_edge(START, "triage")
    g.add_edge("triage", "route")
    g.add_edge("route", END)
    return g


async def demo_pattern_b(model: Any) -> None:
    print("\n=== Pattern B: Human-as-tool (escalation) ===\n")
    graph = build_escalation_graph()
    initial = {
        "request": "weird flickering on the dashboard but only on Tuesdays?",
        "__model__": model,
    }

    result = await graph.execute(initial)
    if result.interrupt:
        payload = result.interrupt.interrupt.payload
        print(f"  ⏸  Triage escalated. Asking human:")
        print(f"     {payload.get('question')}")
        print("  ▶  Human responds: 'technical'")
        final = await graph.execute(Command(resume="technical", update=result.final_state))
        print(f"  ✓ Routed to: {final.final_state.get('final_category')}")
    else:
        print(f"  ✓ Triage confident ({result.final_state.get('category')}) — no escalation")


# ---------------------------------------------------------------------------
# Pattern C — Long-pause workflow with checkpointing
# ---------------------------------------------------------------------------


async def demo_pattern_c(model: Any) -> None:
    """Long-pause workflow: persist the snapshot, resume later.

    The simple in-memory shape is: hold the InterruptState from the
    first ``execute`` call somewhere durable, then call ``execute``
    again with ``Command(resume=...)`` when the human responds.

    For multi-process / multi-day workflows, swap the in-memory snapshot
    for a checkpointer (Redis / Postgres / MySQL / S3 object storage).
    The graph's built-in checkpointer hook expects an AgentState; for
    pure-graph flows like this one, persisting the InterruptState
    yourself is the simpler path.
    """
    print("\n=== Pattern C: Long-pause workflow (snapshot + resume) ===\n")

    graph = build_approval_graph()
    initial = {"request": "Refund for order #42 — never arrived.", "__model__": model}

    paused = await graph.execute(initial)
    if not paused.interrupt:
        print("  ✗ unexpected: workflow completed without pause")
        return
    snapshot_state = paused.final_state
    print(f"  ⏸  Paused at {paused.interrupt.node_id}")
    print(
        f"     Snapshot has {len(snapshot_state)} state keys — persist these "
        "to Redis / Postgres / a queue / etc."
    )

    # ... time passes; reviewer comes back ...
    print("  ▶  Hours later: load snapshot, resume with the human's answer")
    # The snapshot only carries JSON-friendly state. Re-attach the model
    # object explicitly; production code would also rebuild it from
    # config rather than holding a reference in memory.
    resumed = await graph.execute(
        Command(resume="yes", update={**snapshot_state, "__model__": model}),
    )
    print(f"  ✓ Resumed and finished: {resumed.final_state.get('result')}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def main() -> None:
    print("Notebook 34: multi-agent workflows with a human in the loop")
    print("=" * 60)

    model = get_model()
    await demo_pattern_a(model)
    await demo_pattern_b(model)
    await demo_pattern_c(model)


if __name__ == "__main__":
    asyncio.run(main())
