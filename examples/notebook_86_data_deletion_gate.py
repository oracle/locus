#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 86: A human signs off before an agent erases personal data.

A privacy agent handles GDPR data-subject requests. Some are safe to run on
the agent's own authority; one is not. The agent reads the request, decides
what to do, and proposes each side effect to :func:`admit` — the gate that
runs *before* the action. The policy lets the reversible work proceed and
holds the irreversible erasure for a person to approve.

    Data-subject request arrives
       │
       ▼
    Agent proposes an Action  ──>  admit(action, perform, policy, trail)
       │                                │
       │                                ├─ Article 15/20 export (reversible)
       │                                │     → policy ALLOWs → export runs
       │                                │
       │                                └─ Article 17 erasure (irreversible)
       │                                      → policy holds for a human
       │                                      → AdmissionError, nothing deleted
       ▼
    Data Protection Officer reviews the held request and signs off
       │
       ▼
    admit() runs the erasure under the DPO's recorded authority
       │
       ▼
    AuditTrail = the compliance record (every decision, hash-chained)

Why this shape:

- A model that is confused, prompt-injected, or simply wrong still cannot
  delete a person's records on its own. The hold is enforced in code that
  runs before the side effect, not asked for in the prompt.
- Two policy controls do the gating: the ``irreversible`` tag always needs a
  human, and exports stay under the auto-allow blast-radius cap. No DSL, no
  rules engine — plain :class:`ControlPolicy` fields.
- Every attempt — the allowed export, the held erasure, and the erasure that
  runs after sign-off — lands on the same :class:`AuditTrail`. The trail is
  hash-chained and tamper-evident: edit, delete, or reorder a record and
  :meth:`AuditTrail.verify` returns ``False``. That makes it the artifact you
  hand a regulator to show the erasure happened with human authorization.

Run it
    # Fully offline — no network, no credentials. The "perform" steps mutate
    # a local in-memory datastore.
    python examples/notebook_86_data_deletion_gate.py
"""

from __future__ import annotations

import asyncio

from tulip.control import (
    Action,
    AdmissionError,
    AuditTrail,
    ControlPolicy,
    admit,
)


# A stand-in for the systems that hold a data subject's personal data. In a
# real deployment these are rows in Postgres, objects in S3, documents in a
# search index, and so on. Here they are just keys in a dict so the example
# runs offline with no credentials.

SUBJECT = "subject:eu-44213"  # pseudonymous handle for one EU data subject

DATASTORE: dict[str, dict[str, str]] = {
    SUBJECT: {
        "profile": "name, email, postal address",
        "orders": "7 past orders with billing details",
        "support_tickets": "3 closed tickets",
        "marketing_events": "412 clickstream + email-open events",
    }
}


# The "perform" side effects. admit() awaits these only after the policy
# clears the proposed Action — so neither can run on the agent's own say-so
# unless the policy allows it.


async def export_subject_data(subject: str) -> str:
    """Article 15/20: assemble a portable copy of the subject's data.

    Reversible and read-only — it copies, it does not change anything.
    """
    record = DATASTORE.get(subject, {})
    bundle = ", ".join(f"{k}={v}" for k, v in record.items())
    return f"exported {len(record)} categories for {subject}: {bundle}"


async def erase_subject_records(subject: str) -> str:
    """Article 17: permanently delete every record tied to the subject.

    Irreversible. Once this runs, the data is gone — which is exactly why the
    policy holds it for a human.
    """
    record = DATASTORE.pop(subject, {})
    return f"erased {len(record)} categories for {subject}; nothing remains"


# A stand-in for the human approval step. In production this is a ticket the
# Data Protection Officer resolves in a console; here it just prints and
# returns True so the example is self-contained.


def dpo_signs_off(action: Action) -> bool:
    print(f"  [DPO] reviewing held request: {action.name} on {action.asset}")
    print("  [DPO] verified identity, retention obligations, and legal holds")
    print("  [DPO] decision: APPROVE erasure")
    return True


def _show(action: Action, outcome: str, detail: str) -> None:
    print(f"  action : {action.name} ({action.kind}, blast_radius={action.blast_radius})")
    print(f"  outcome: {outcome}")
    print(f"  detail : {detail}")


async def main() -> None:
    print("Notebook 86: A human signs off before an agent erases personal data")
    print("=" * 70)

    # The policy a privacy team would actually set. Verification scoring is off
    # for this demo (there is no model verdict to weigh), so the gating comes
    # from two plain rules:
    #   - anything tagged "irreversible" always needs a human, and
    #   - an action may auto-proceed only if it touches at most 5 categories.
    policy = ControlPolicy(
        require_verification_score=0.0,
        max_blast_radius=5,
        require_human_for=frozenset({"irreversible"}),
    )

    trail = AuditTrail()

    # --- Request 1: a data-portability export. Safe to run automatically. ---
    print("\n--- DSR-7781: Article 15/20 export request ---")
    export = Action(
        name="export_subject_data",
        asset=SUBJECT,
        blast_radius=4,  # 4 data categories, under the auto-allow cap of 5
        environment="production",
        kind="export",
        tags=frozenset({"reversible", "pii"}),
    )
    result = await admit(
        export,
        lambda: export_subject_data(SUBJECT),
        policy=policy,
        trail=trail,
    )
    _show(export, "ALLOW (auto)", result)

    # --- Request 2: a right-to-erasure delete. Held for a human. ---
    print("\n--- DSR-7782: Article 17 erasure request ---")
    erase = Action(
        name="erase_subject_records",
        asset=SUBJECT,
        blast_radius=4,
        environment="production",
        kind="delete",
        tags=frozenset({"irreversible", "pii"}),
    )
    try:
        await admit(
            erase,
            lambda: erase_subject_records(SUBJECT),
            policy=policy,
            trail=trail,
        )
        raise SystemExit("BUG: the irreversible erasure should not have auto-run")
    except AdmissionError as exc:
        _show(erase, exc.decision.outcome, exc.decision.reason)
        # Prove the gate actually blocked the side effect: data still present.
        assert SUBJECT in DATASTORE, "erasure ran despite the human hold"
        print("  note   : nothing deleted — the agent could not act on its own")

    # --- The human reviews the held request and signs off. ---
    print("\n--- Human-in-the-loop ---")
    approved = dpo_signs_off(erase)
    assert approved

    # The erasure now runs under the DPO's recorded authority. We re-admit with
    # a policy that no longer auto-holds irreversible actions, because a human
    # is now the one authorizing this specific signed-off ticket. The second
    # admission lands on the same trail, so the record shows the erasure ran
    # *after* approval — never before.
    print("\n--- DSR-7782: erasure proceeds after sign-off ---")
    signed_off_policy = ControlPolicy(
        require_verification_score=0.0,
        max_blast_radius=erase.blast_radius,
        require_human_for=frozenset(),  # the human gate is satisfied for this ticket
    )
    approved_erase = Action(
        name="erase_subject_records",
        asset=SUBJECT,
        blast_radius=4,
        environment="production",
        kind="delete",
        tags=frozenset({"irreversible", "pii", "dpo_signed_off"}),
    )
    result = await admit(
        approved_erase,
        lambda: erase_subject_records(SUBJECT),
        policy=signed_off_policy,
        trail=trail,
    )
    _show(approved_erase, "ALLOW (after human sign-off)", result)
    assert SUBJECT not in DATASTORE, "erasure should have removed the subject's data"

    # --- The audit trail is the compliance record. ---
    print("\n--- Compliance record (AuditTrail) ---")
    records = trail.records()
    for rec in records:
        print(f"  #{rec.seq} {rec.event_type}: {rec.payload['action']} -> {rec.payload['outcome']}")
    print(f"  chain intact (tamper-evident): {trail.verify()}")

    # Assert the trail captured the whole story: the auto-allowed export, the
    # held erasure, and the erasure that ran after the human signed off.
    outcomes = [(r.payload["action"], r.payload["outcome"]) for r in records]
    assert ("export_subject_data", "allow") in outcomes, "export allow not recorded"
    assert ("erase_subject_records", "require_human") in outcomes, "erasure hold not recorded"
    assert ("erase_subject_records", "allow") in outcomes, "post-signoff erasure not recorded"
    assert trail.verify(), "audit chain failed integrity check"
    print("\nOK: the export, the human hold, and the approved erasure are all on the record.")


if __name__ == "__main__":
    asyncio.run(main())
