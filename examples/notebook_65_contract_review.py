#!/usr/bin/env python3
# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Notebook 60: Contract-review workflow (parallel review and negotiation loop).

Real contract review involves multiple stakeholders working in parallel,
then a back-and-forth negotiation phase, then sign-off::

    Contract intake
       │
       ▼
    Parser  (extracts clauses)
       │
       ▼
    Scatter to 3 parallel reviewers
       ├── Legal    (regulatory risk, indemnity, termination)
       ├── Risk     (financial exposure, liability cap)
       └── Commercial (price, terms, SLAs)
       ▼
    Synthesizer  (consolidated review report)
       │
       ▼
    Negotiation gate ── any blockers? ── yes ──> Negotiate (interrupt; loop)
                                       │            │
                                       │            └── revised terms ──┐
                                       │                                │
                                       └── no ──┐                       │
                                                ▼                       │
                                          Sign-off  <───────────────────┘
                                                ▼
                                          ContractDecision (typed)

- Send: three reviewers run concurrently.
- add_conditional_edges with cycles enabled: negotiation can loop back
  to re-review when terms change. Hard cap of 3 rounds.
- interrupt(): negotiation step pauses for human counsel to edit terms.
- output_schema=ContractDecision: typed terminal artifact.

Run it
    # Default: the bundled mock model (set TULIP_MODEL_PROVIDER for a live provider)
    python examples/notebook_65_contract_review.py

    # Offline / no credentials:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_65_contract_review.py

    # Pin a strong-enough model for the structured ContractDecision schema:
    TULIP_MODEL_ID=openai.gpt-4.1 python examples/notebook_65_contract_review.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from config import get_model
from pydantic import BaseModel, Field

from tulip.agent import Agent, AgentConfig
from tulip.core import Command, interrupt
from tulip.core.events import TerminateEvent
from tulip.core.send import Send
from tulip.multiagent.graph import END, START, GraphConfig, StateGraph


# Data shapes.


class ReviewerFinding(BaseModel):
    perspective: str  # "legal" | "risk" | "commercial"
    blockers: list[str]
    recommended_changes: list[str]
    risk_score: float = Field(ge=0.0, le=1.0)


class ContractDecision(BaseModel):
    contract_id: str
    counterparty: str
    rounds: int
    blockers_resolved: list[str]
    open_blockers: list[str]
    final_terms_summary: str
    decision: str = Field(description="signed | rejected | abandoned")


# Specialist prompts.


PROMPTS = {
    "legal": (
        "You are an in-house counsel. Read the contract excerpt and identify "
        "concrete legal blockers (indemnity, jurisdiction, termination, IP, "
        "liability cap). Bullets. End with: BLOCKERS=<count>."
    ),
    "risk": (
        "You are an enterprise-risk analyst. Identify concrete financial "
        "or operational risks. Bullets. End with: BLOCKERS=<count>."
    ),
    "commercial": (
        "You are a commercial-terms reviewer. Identify pricing or SLA "
        "concerns. Bullets. End with: BLOCKERS=<count>."
    ),
}


def _make_agent(role: str, model: Any) -> Agent:
    return Agent(
        config=AgentConfig(
            agent_id=f"contract-{role}",
            model=model,
            system_prompt=PROMPTS[role],
            max_iterations=2,
            max_tokens=400,
        )
    )


async def _run(agent: Agent, prompt: str) -> str:
    final = ""
    async for event in agent.run(prompt):
        if isinstance(event, TerminateEvent):
            final = event.final_message or ""
    return final.strip()


# Graph nodes.


async def parse_contract(state: dict[str, Any]) -> dict[str, Any]:
    """In production this would chunk the PDF; here we normalise text."""
    return {"clauses": state.get("contract_text", "").strip()}


async def scatter_reviewers(state: dict[str, Any]) -> list[Send]:
    perspectives = ("legal", "risk", "commercial")
    return [
        Send(node="review_one", payload={"perspective": p}, metadata={"perspective": p})
        for p in perspectives
    ]


async def review_one(state: dict[str, Any]) -> dict[str, Any]:
    perspective = state["perspective"]
    agent = _make_agent(perspective, state["__model__"])
    text = await _run(
        agent,
        f"Contract clauses:\n{state.get('clauses', '')}\n\nGive your {perspective} review.",
    )
    # Heuristic: any bulleted line is a finding; first half are blockers,
    # the rest are recommended changes.
    bullets = [
        b.lstrip("- *•").strip() for b in text.splitlines() if b.strip().startswith(("-", "*", "•"))
    ]
    half = max(1, len(bullets) // 2)
    return {
        "finding": ReviewerFinding(
            perspective=perspective,
            blockers=bullets[:half] if bullets else [text or "(no findings)"],
            recommended_changes=bullets[half:],
            risk_score=0.5,
        )
    }


async def synthesize(state: dict[str, Any]) -> dict[str, Any]:
    findings = [v["finding"] for v in state.values() if isinstance(v, dict) and "finding" in v]
    blockers = [b for f in findings for b in f.blockers]
    return {
        "findings": findings,
        "open_blockers": blockers,
        "rounds": state.get("rounds", 0) + 1,
    }


def negotiation_gate(state: dict[str, Any]) -> str:
    """Loop back to re-review if blockers exist and we're under the cap."""
    if not state.get("open_blockers"):
        return "sign_off"
    if state.get("rounds", 0) >= 3:
        # Cap at 3 rounds; sign-off decides reject vs sign.
        return "sign_off"
    return "negotiate"


async def negotiate(state: dict[str, Any]) -> Command:
    """Pause for counsel to redline a clause; always return a Command.

    Three outcomes:

    - ``RESOLVED``: counterparty accepted our terms. Route to sign-off.
    - ``WALK``: counterparty refused; route to sign-off as 'abandoned'.
    - Custom redline text: route back to ``parse`` with the new clauses
      so the parallel reviewers re-evaluate.
    """
    open_blockers = state.get("open_blockers", [])
    response = interrupt(
        {
            "type": "negotiation",
            "round": state.get("rounds"),
            "question": "Counterparty redline the contract — what's the new clause language?",
            "open_blockers": open_blockers,
            "options": [
                "RESOLVED: counterparty agreed to our terms",
                "WALK: counterparty refused; abandon",
                "<custom redline text>",
            ],
        }
    )
    if response.startswith("WALK"):
        return Command(
            goto="sign_off",
            update={"walk_away": True, "open_blockers": open_blockers},
        )
    if response.startswith("RESOLVED"):
        return Command(
            goto="sign_off",
            update={
                "blockers_resolved": list(state.get("blockers_resolved", [])) + open_blockers,
                "open_blockers": [],
                "clauses": state.get("clauses", "") + "\n[All blockers resolved per redline.]",
            },
        )
    # Counterparty redlined — re-parse the new text and re-run reviewers.
    return Command(
        goto="parse",
        update={
            "contract_text": response,
            "blockers_resolved": list(state.get("blockers_resolved", [])) + open_blockers,
            "open_blockers": [],
        },
    )


async def sign_off(state: dict[str, Any]) -> dict[str, Any]:
    """Emit ContractDecision via Agent.output_schema=ContractDecision.

    The Agent reads accumulated state and produces the typed Pydantic
    instance. If the model can't honour the JSON schema we surface a
    hard error rather than fabricating a record.
    """
    import asyncio as _asyncio

    if state.get("walk_away"):
        outcome = "abandoned"
    elif state.get("open_blockers"):
        outcome = "rejected"
    else:
        outcome = "signed"

    # Trim the prompt so the model focuses on the structured fields
    # rather than long reviewer prose.
    open_blockers = state.get("open_blockers", []) or []
    resolved = state.get("blockers_resolved", []) or []

    def _trim(items: list[str], n: int = 5, w: int = 80) -> list[str]:
        out = [s[:w] for s in items[:n]]
        if len(items) > n:
            out.append(f"... and {len(items) - n} more")
        return out

    agent = Agent(
        config=AgentConfig(
            agent_id="contract-signoff",
            model=state["__model__"],
            system_prompt=(
                "You are a contract-ops officer writing the final ContractDecision. "
                "Use the supplied fields. Summarise the final terms in one sentence."
            ),
            output_schema=ContractDecision,
            max_iterations=2,
            max_tokens=400,
        )
    )
    prompt = (
        f"Contract: {state.get('contract_id')}\n"
        f"Counterparty: {state.get('counterparty')}\n"
        f"Decision: {outcome}\n"
        f"Rounds: {state.get('rounds', 0)}\n"
        f"Resolved blockers ({len(resolved)}): {_trim(resolved)}\n"
        f"Open blockers ({len(open_blockers)}): {_trim(open_blockers)}\n\n"
        "Emit the ContractDecision."
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
            f"Sign-off agent failed after 3 attempts. Last error: {last_exc!r}"
        ) from last_exc
    decision = result.parsed
    if decision is None:
        raise RuntimeError(
            "Sign-off agent returned no parsed ContractDecision. The configured "
            "model could not honor the JSON schema. Use a stronger model "
            "(e.g. openai.gpt-4o, openai.gpt-5, anthropic.claude-3-5-sonnet) "
            f"for notebook 59. Raw output: {result.message!r}"
        )
    return {"decision": decision}


# Build the review graph.


def build_review_graph() -> StateGraph:
    g = StateGraph(
        name="contract-review",
        # parse → scatter → synthesize → negotiate → parse is a real cycle,
        # so we have to opt in.
        config=GraphConfig(allow_cycles=True, max_iterations=20),
    )
    g.add_node("parse", parse_contract)
    g.add_node("scatter", scatter_reviewers)
    g.add_node("review_one", review_one)
    g.add_node("synthesize", synthesize)
    g.add_node("negotiate", negotiate)
    g.add_node("sign_off", sign_off)

    g.add_edge(START, "parse")
    g.add_edge("parse", "scatter")
    g.add_edge("scatter", "synthesize")
    g.add_conditional_edges(
        "synthesize",
        negotiation_gate,
        targets={"negotiate": "negotiate", "sign_off": "sign_off"},
    )
    g.add_edge("negotiate", "parse")  # loop back: re-review the new terms
    g.add_edge("sign_off", END)
    return g


# Driver and sample contract text.


SAMPLE_CONTRACT = """\
MASTER SERVICES AGREEMENT — EXCERPT

This Master Services Agreement ("Agreement") is entered into by and between
MegaCorp Cloud Solutions, Inc. ("Vendor") and the customer entity named on
the order form ("Customer"), and is effective as of the date of last signature
below.

1. SERVICES AND ORDER FORMS

1.1 Vendor will provide the cloud-platform services described in one or more
order forms executed by the parties. Each order form is incorporated into
this Agreement by reference. In the event of a conflict between an order form
and this Agreement, the order form controls.

1.2 Vendor reserves the right to modify the technical implementation of the
Services at any time provided that the functional description on the most
recent order form is materially preserved.

2. TERM AND RENEWAL

2.1 The initial term is thirty-six (36) months from the effective date.

2.2 The Agreement auto-renews for successive twelve (12) month terms unless
Customer provides written notice of non-renewal at least ninety (90) days
prior to the end of the then-current term. Notice given any later than that
window will not be effective until the following renewal cycle.

3. FEES, PAYMENT, AND PRICE ESCALATION

3.1 Customer shall pay all fees within Net-30 of invoice date. Fees not paid
when due bear a late charge of five percent (5%) per month, compounding, with
no cap on the total late charge accumulation.

3.2 At each renewal, Vendor may increase fees in its sole discretion and is
not obligated to provide advance notice of the increase prior to invoicing
the renewal term.

4. INTELLECTUAL PROPERTY

4.1 Customer retains all right, title and interest in Customer Data.

4.2 Any feedback, suggestions, or requests submitted by Customer or its users
regarding the Services (including bug reports, feature requests, and any
configuration patterns or workflows developed using the Services) shall be
deemed assigned to Vendor on a worldwide, royalty-free, perpetual,
sublicensable basis without further consideration.

5. INDEMNITY

5.1 Vendor will defend and indemnify Customer against third-party claims
alleging that the Services infringe a U.S. patent or copyright, subject to
the liability cap in Section 7.

5.2 Customer will defend, indemnify, and hold harmless Vendor and its
affiliates, officers, employees and contractors from and against any and all
claims, damages, losses, fines, judgments, and reasonable attorneys' fees
arising out of or relating to (a) Customer's use of the Services, (b) any
content uploaded to the Services by Customer or its end users, and (c) any
breach of this Agreement by Customer. Customer's obligations under this
Section 5.2 are not subject to the liability cap in Section 7.

6. DATA, PROCESSING, AND DELETION

6.1 Customer Data may be processed and stored in any region in which Vendor
or its sub-processors operate. Vendor will use commercially reasonable
efforts to comply with applicable data-protection law.

6.2 Upon termination, Vendor will retain Customer Data for at least thirty
(30) days for operational continuity. After thirty days Vendor may, but is
not obligated to, delete Customer Data; deletion certifications are not
provided.

7. LIABILITY CAP

7.1 Each party's aggregate liability under this Agreement is capped at one
times (1×) the fees paid by Customer in the twelve months preceding the claim,
except as set out in Sections 5.2 and 8.

8. TERMINATION

8.1 Either party may terminate this Agreement for the other party's
uncured material breach upon thirty (30) days written notice.

8.2 Vendor may additionally terminate this Agreement, or any order form, for
convenience upon thirty (30) days written notice. Customer may not terminate
for convenience.

8.3 Service-level credits, if any, are Customer's sole and exclusive remedy
for unavailability or performance failures. The SLA schedule, attached as
Exhibit A, may be revised by Vendor unilaterally with thirty (30) days
notice.
"""

REDLINE_ROUND_1 = """\
MASTER SERVICES AGREEMENT — EXCERPT (counterparty redline, round 1)

The contract above stands except for the following counterparty edits:

3.1 Late charge reduced to one percent (1.0%) per month, capped at fifteen
percent (15%) of the unpaid invoice. Payment terms remain Net-30.

5.2 Customer indemnity is now subject to the same liability cap in Section 7
as Vendor's indemnity. The carve-outs for content-based and breach claims
remain.

8.2 Vendor's right to terminate for convenience is removed. Either party may
terminate for material uncured breach on 30 days notice; otherwise the
Agreement runs to end of term.

All other clauses (term length, renewal notice window, IP feedback
assignment, data residency, liability cap, SLA unilateral revision) are
unchanged from the prior draft.
"""

REDLINE_ROUND_2 = """\
MASTER SERVICES AGREEMENT — EXCERPT (counterparty redline, round 2)

Building on round-1 edits, counterparty has further accepted:

2.2 Renewal-notice window shortened from 90 days to 30 days, and Customer
may opt out of renewal at any time after the initial 36-month term with 30
days written notice (no auto-roll into a new 12-month commit).

3.2 Renewal price escalation capped at the lesser of CPI or 5% per renewal,
with at least 60 days advance written notice of any increase.

4.2 IP-feedback assignment removed. Customer feedback and suggestions remain
Customer's property; Vendor receives only a non-exclusive licence to act on
the feedback.

6.2 Vendor will delete Customer Data within 60 days of termination and
provide a written deletion certification at no charge.

8.3 SLA schedule may not be revised unilaterally by Vendor; revisions
require Customer's written consent. Service-level credits remain a remedy
but are no longer the *sole* remedy.

Open items (counterparty's position): liability cap remains 1× annual fees;
data-residency clause remains "any region"; sub-processor list will be made
available on request.
"""


def _print_decision(d: ContractDecision | None) -> None:
    print("\nContract decision:")
    print("-" * 60)
    if d is None:
        print("(missing)")
        return
    print(f"  Contract:           {d.contract_id}")
    print(f"  Counterparty:       {d.counterparty}")
    print(f"  Decision:           {d.decision.upper()}")
    print(f"  Negotiation rounds: {d.rounds}")
    print(f"  Resolved blockers:  {len(d.blockers_resolved)}")
    print(f"  Open blockers:      {len(d.open_blockers)}")


async def main() -> None:
    print("Notebook 60: Contract review workflow")
    print("=" * 60)

    model = get_model()
    graph = build_review_graph()
    initial = {
        "contract_id": "C-2026-0815",
        "counterparty": "MegaCorp Cloud Solutions",
        "contract_text": SAMPLE_CONTRACT,
        "__model__": model,
    }

    print(f"\nReviewing: {initial['counterparty']} ({initial['contract_id']})")

    # Drive two negotiation rounds: counterparty redlines, redlines again,
    # then accepts our terms on the third pass.
    answers = [
        REDLINE_ROUND_1,
        REDLINE_ROUND_2,
        "RESOLVED: counterparty agreed to our terms",
    ]
    result = await graph.execute(initial)
    answer_idx = 0
    while result.interrupt:
        answer = answers[answer_idx] if answer_idx < len(answers) else "RESOLVED"
        answer_idx += 1
        payload = result.interrupt.interrupt.payload
        print(
            f"\n  ⏸  Round {payload.get('round')}: {len(payload.get('open_blockers', []))} blocker(s)"
        )
        for b in payload.get("open_blockers", [])[:3]:
            print(f"      - {b[:80]}")
        preview = answer if len(answer) <= 80 else answer[:77] + "..."
        print(f"  ▶  Counsel responds: {preview!r}")
        result = await graph.execute(
            Command(resume=answer, update={**result.final_state, "__model__": model})
        )

    print(f"\nWorkflow finished in {result.duration_ms:.0f} ms")
    _print_decision(result.final_state.get("decision"))


if __name__ == "__main__":
    asyncio.run(main())
