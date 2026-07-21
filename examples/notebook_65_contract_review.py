#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 65: Vendor DPA review — data-subject rights, retention, transfer gaps.

Reviewing a data processor's Data Processing Agreement involves multiple
stakeholders working in parallel, then a back-and-forth negotiation
phase, then sign-off::

    DPA intake
       │
       ▼
    Parser  (extracts clauses)
       │
       ▼
    Scatter to 3 parallel reviewers
       ├── Legal       (lawful basis, data-subject rights, consent, purpose)
       ├── Governance  (minimisation, retention/deletion, residency, sub-processors)
       └── Compliance  (GDPR/CCPA, DPIA, transfer mechanisms, liability for fines)
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

The processor here is a marketing-analytics SaaS that ingests your
customers' personal data (email addresses, device identifiers,
behavioural events) — a data processor in the controller's vendor chain.
A weak erasure SLA or a vague international-transfer mechanism is a real
compliance gap, not a paperwork nit: the vendor's breach-notification
window should track the GDPR Art. 33 72-hour timeline you owe your own
supervisory authority. The DPO — the privacy office's data-protection
officer — writes the typed sign-off.

- Send: three reviewers run concurrently.
- add_conditional_edges with cycles enabled: negotiation can loop back
  to re-review when terms change. Hard cap of 3 rounds.
- interrupt(): negotiation step pauses for human counsel to edit terms.
- output_schema=ContractDecision: the DPO's typed terminal artifact.

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
    perspective: str  # "legal" | "governance" | "compliance"
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
        "You are privacy legal counsel. Read the DPA excerpt and identify "
        "concrete data-protection blockers (lawful basis, data-subject rights "
        "— access, erasure, portability — consent, and purpose limitation). "
        "Bullets. End with: BLOCKERS=<count>."
    ),
    "governance": (
        "You are a data-governance reviewer assessing a vendor that will "
        "process the company's customer personal data. Identify concrete "
        "governance gaps (data minimisation, retention and deletion SLAs, "
        "data residency, and sub-processor controls). Bullets. End with: "
        "BLOCKERS=<count>."
    ),
    "compliance": (
        "You are a privacy-compliance analyst. Identify regulatory gaps "
        "(GDPR / CCPA obligations, DPIA requirements, cross-border transfer "
        "mechanisms such as SCCs, GDPR Art. 28 terms, and liability for "
        "regulatory fines). Bullets. End with: BLOCKERS=<count>."
    ),
}


def _make_agent(role: str, model: Any) -> Agent:
    return Agent(
        config=AgentConfig(
            agent_id=f"dpa-{role}",
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
    """In production this would chunk the signed PDF; here we normalise text."""
    return {"clauses": state.get("contract_text", "").strip()}


async def scatter_reviewers(state: dict[str, Any]) -> list[Send]:
    perspectives = ("legal", "governance", "compliance")
    return [
        Send(node="review_one", payload={"perspective": p}, metadata={"perspective": p})
        for p in perspectives
    ]


async def review_one(state: dict[str, Any]) -> dict[str, Any]:
    perspective = state["perspective"]
    agent = _make_agent(perspective, state["__model__"])
    text = await _run(
        agent,
        f"DPA clauses:\n{state.get('clauses', '')}\n\nGive your {perspective} review.",
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
    """Loop back to re-review if compliance gaps exist and we're under the cap."""
    if not state.get("open_blockers"):
        return "sign_off"
    if state.get("rounds", 0) >= 3:
        # Cap at 3 rounds; sign-off decides reject vs sign.
        return "sign_off"
    return "negotiate"


async def negotiate(state: dict[str, Any]) -> Command:
    """Pause for counsel to redline a clause; always return a Command.

    Three outcomes:

    - ``RESOLVED``: vendor accepted our terms. Route to sign-off.
    - ``WALK``: vendor refused; route to sign-off as 'abandoned'.
    - Custom redline text: route back to ``parse`` with the new clauses
      so the parallel reviewers re-evaluate.
    """
    open_blockers = state.get("open_blockers", [])
    response = interrupt(
        {
            "type": "negotiation",
            "round": state.get("rounds"),
            "question": "Vendor redlined the DPA — what's the new clause language?",
            "open_blockers": open_blockers,
            "options": [
                "RESOLVED: vendor agreed to our terms",
                "WALK: vendor refused; abandon",
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
    # Vendor redlined — re-parse the new text and re-run reviewers.
    return Command(
        goto="parse",
        update={
            "contract_text": response,
            "blockers_resolved": list(state.get("blockers_resolved", [])) + open_blockers,
            "open_blockers": [],
        },
    )


async def sign_off(state: dict[str, Any]) -> dict[str, Any]:
    """SCRIBE emits ContractDecision via Agent.output_schema=ContractDecision.

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
            agent_id="dpo-dpa-signoff",
            model=state["__model__"],
            system_prompt=(
                "You are a data-protection officer writing the final ContractDecision. "
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
            result = await agent.arun(prompt)
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
            f"for notebook 65. Raw output: {result.message!r}"
        )
    return {"decision": decision}


# Build the review graph.


def build_review_graph() -> StateGraph:
    g = StateGraph(
        name="dpa-review",
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


# Driver and sample DPA text (invented for this notebook — deliberately
# full of compliance gaps for the reviewers to catch).


SAMPLE_CONTRACT = """\
DATA PROCESSING ADDENDUM — EXCERPT

This Data Processing Addendum ("DPA") is entered into by and between
PixelPulse Analytics, Inc. ("Vendor", the processor) and the customer
entity named on the order form ("Customer", the controller), and forms part
of the Master Services Agreement between the parties.

1. PROCESSING AND SUB-PROCESSORS

1.1 Vendor processes Customer Personal Data (end-user email addresses,
device identifiers, and page-view and behavioural events) to provide the
marketing-analytics Services as described in the applicable order form.
Vendor may also use Customer Personal Data to improve its own products.

1.2 Vendor may appoint sub-processors at its sole discretion and without
prior notice to Customer. A list of current sub-processors is available on
written request, within thirty (30) days of the request.

2. SECURITY MEASURES

2.1 Vendor will maintain commercially reasonable technical and
organisational measures. Customer Personal Data is encrypted at rest where
practicable. Encryption in transit applies to public-network hops only;
traffic between Vendor's internal services may be unencrypted.

2.2 Vendor's online Security Policy, as amended by Vendor from time to time
in its sole discretion, sets out the current measures and controls, and in
the event of conflict the online Security Policy controls over this DPA.

3. PERSONAL DATA BREACH NOTIFICATION

3.1 Vendor will notify Customer of a Personal Data Breach without undue
delay, and in any event within thirty (30) days of Vendor confirming the
breach. Whether an incident constitutes a notifiable breach is determined
by Vendor in its sole discretion.

3.2 Vendor's notification obligations are satisfied by posting to the
Vendor status page; direct notice to Customer's privacy contact is not
required.

4. AUDIT RIGHTS

4.1 On-site audits are not permitted. On written request no more than once
per year, Vendor will provide a summary letter describing its most recent
SOC 2 examination. Full reports are not shared. Customer bears Vendor's
reasonable costs of responding to any audit request.

5. DATA RESIDENCY AND INTERNATIONAL TRANSFERS

5.1 Customer Personal Data may be processed and stored in any region in
which Vendor or its sub-processors operate. Transfers out of the original
region rely on appropriate safeguards as determined by Vendor.

6. RETENTION AND DELETION

6.1 Upon termination, Vendor will retain Customer Personal Data for at
least thirty (30) days for billing reconciliation. Backup copies may be
retained indefinitely. Deletion certifications are not provided.

7. DATA SUBJECT REQUESTS

7.1 Vendor will forward any data-subject access, erasure, or portability
request it receives to Customer within thirty (30) days. Vendor does not
otherwise assist Customer in responding, and charges professional-services
fees for any data export performed on Customer's behalf.

8. LIABILITY FOR DATA INCIDENTS

8.1 Vendor's aggregate liability for any breach of this DPA, including
regulatory fines and penalties arising from Vendor's negligence, is capped
at one times (1×) the fees paid by Customer in the twelve months preceding
the claim.
"""

REDLINE_ROUND_1 = """\
DATA PROCESSING ADDENDUM — EXCERPT (vendor redline, round 1)

The DPA above stands except for the following vendor edits:

2.1 Vendor commits to encryption of Customer Personal Data in transit and
at rest in all cases (TLS 1.2+ and AES-256 or equivalent), including
traffic between Vendor's internal services.

2.2 The unilateral-precedence clause is removed. The security measures in
this DPA control; the online Security Policy may only add protections, not
reduce them.

3.1 Breach notification window shortened to seventy-two (72) hours from
confirmation, with direct notice to Customer's designated privacy contact.
The status-page-only provision in 3.2 is deleted.

All other clauses (sub-processor discretion, audit rights, data residency,
retention and deletion, data-subject requests, liability cap including
fines) are unchanged from the prior draft.
"""

REDLINE_ROUND_2 = """\
DATA PROCESSING ADDENDUM — EXCERPT (vendor redline, round 2)

Building on round-1 edits, vendor has further accepted:

1.2 Vendor will give thirty (30) days advance written notice of any new
sub-processor, with a right for Customer to object on reasonable
data-protection grounds. The current sub-processor list is published and
kept up to date.

4.1 Vendor will provide its full SOC 2 Type II report annually under NDA at
no charge, plus an annual penetration-test summary. One audit (remote or
on-site) per year is permitted at Customer's own cost following a confirmed
Personal Data Breach.

6.1 Vendor will delete Customer Personal Data, including backups, within
sixty (60) days of termination and provide a written deletion certification
at no charge.

7.1 Vendor will assist Customer in fulfilling data-subject access, erasure,
and portability requests within ten (10) business days at no charge, and
provides a self-service export endpoint. The product-improvement use in 1.1
is removed; Vendor processes Customer Personal Data only on documented
instructions.

Open items (vendor's position): liability cap remains 1× annual fees and
continues to include regulatory fines; data-residency clause remains "any
region", with transfers governed by standard contractual clauses on
request.
"""


def _print_decision(d: ContractDecision | None) -> None:
    print("\nDPA decision:")
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
    print("Notebook 65: Vendor DPA / data-processing-agreement review")
    print("=" * 60)

    model = get_model()
    graph = build_review_graph()
    initial = {
        "contract_id": "DPA-2026-0815",
        "counterparty": "PixelPulse Analytics",
        "contract_text": SAMPLE_CONTRACT,
        "__model__": model,
    }

    print(f"\nReviewing: {initial['counterparty']} ({initial['contract_id']})")

    # Drive two negotiation rounds: vendor redlines, redlines again,
    # then accepts our terms on the third pass.
    answers = [
        REDLINE_ROUND_1,
        REDLINE_ROUND_2,
        "RESOLVED: vendor agreed to our terms",
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
