#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 83: A refund gate that pays out small refunds and holds big ones.

A support agent talks to an upset customer and decides a refund is owed. The
agent should be free to settle the small stuff on its own — a $12 shipping
credit, a duplicate-charge reversal — without paging a human. But a $4,000
refund, or one that reverses a whole batch of charges, should stop and wait
for a person.

The decision of *whether the money moves* does not live in the agent's prompt.
It lives in :func:`admit`, a gate that runs before the payout. The agent can be
confused, jailbroken, or wrong about the dollar amount; the refund still only
goes through if the :class:`ControlPolicy` admits it. Everything that happens —
the paid refund and the held one — lands on a tamper-evident
:class:`AuditTrail`.

    Support agent proposes a refund
       │
       ▼
    Build an Action  (amount → blast_radius, large → "high_value" tag)
       │
       ▼
    admit(action, pay_out, policy, trail)
       │
       ├─ small refund  ── policy allows ──> pay_out() runs, money moves
       │
       └─ large refund  ── blast_radius over the cap, or "high_value" tag ──>
                           AdmissionError(require_human); pay_out() never runs,
                           the hold is queued for a human approver
       │
       ▼
    AuditTrail records every decision — paid or held

The policy here is deliberately small: auto-allow a single-ledger-entry refund
that carries no ``high_value`` tag; hold anything bigger. ``require_verification_score``
is set to 0 because a refund is a business action, not a security finding — there
is no threat to verify, so the gate reasons purely over amount and scope.

Run it (fully offline — no network, no credentials; the payout is a local stub):
    python examples/notebook_83_payment_refund_gate.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from tulip.control import (
    Action,
    AdmissionError,
    AuditTrail,
    ControlPolicy,
    admit,
)


# The "side effect" the agent wants to take. Offline, it just mutates a local
# ledger — but this is the one place money would actually move in production.


@dataclass
class Ledger:
    """A stand-in payment ledger. In production this would call the PSP."""

    paid: list[tuple[str, float]] = field(default_factory=list)
    held: list[tuple[str, float]] = field(default_factory=list)

    def pay_out(self, customer: str, amount: float) -> str:
        """Actually move the money. Only ever reached after admit() allows it."""
        self.paid.append((customer, amount))
        return f"refunded ${amount:,.2f} to {customer}"


# How a proposed refund becomes an Action the policy can reason over. The dollar
# amount drives two policy-visible attributes: how many ledger entries the refund
# touches (blast_radius) and whether it crosses the human-review line (a tag).

LARGE_REFUND_THRESHOLD = 200.0


def refund_action(customer: str, amount: float, ledger_entries: int, reason: str) -> Action:
    tags = {"refund"}
    if amount >= LARGE_REFUND_THRESHOLD:
        tags.add("high_value")
    return Action(
        name="issue_refund",
        asset=customer,
        blast_radius=ledger_entries,
        environment="production",
        kind="payment",
        tags=frozenset(tags),
    )


async def propose_refund(
    ledger: Ledger,
    policy: ControlPolicy,
    trail: AuditTrail,
    *,
    customer: str,
    amount: float,
    ledger_entries: int,
    reason: str,
) -> None:
    """Run one refund through the gate and narrate what happened."""
    action = refund_action(customer, amount, ledger_entries, reason)
    print(f"\n--- Support agent proposes: refund ${amount:,.2f} to {customer} ---")
    print(f"    reason: {reason}")
    print(
        f"    scope:  {ledger_entries} ledger entr{'y' if ledger_entries == 1 else 'ies'}, "
        f"tags={sorted(action.tags)}"
    )

    try:
        result = await admit(
            action,
            lambda: _pay_out(ledger, customer, amount),
            policy=policy,
            trail=trail,
        )
        print(f"  PAID    {result}")
    except AdmissionError as exc:
        ledger.held.append((customer, amount))
        print(f"  HELD    not admitted ({exc.decision.outcome}) — queued for a human")
        for check in exc.decision.checks:
            print(f"            · {check}")


async def _pay_out(ledger: Ledger, customer: str, amount: float) -> str:
    # admit() awaits this; keep it async to match the perform() contract.
    return ledger.pay_out(customer, amount)


async def main() -> None:
    print("Notebook 83: A refund gate that pays out small refunds and holds big ones")
    print("=" * 60)

    # The CISO/finance knobs. require_verification_score=0 because a refund is a
    # business action, not a security finding — there is nothing to "verify".
    # max_blast_radius=1: a refund may touch one ledger entry to auto-pay.
    # require_human_for={"high_value"}: any refund tagged high_value waits for a person.
    policy = ControlPolicy(
        require_verification_score=0.0,
        max_blast_radius=1,
        require_human_for=frozenset({"high_value"}),
    )
    trail = AuditTrail()
    ledger = Ledger()

    # 1) Small refund: one ledger entry, well under the threshold -> auto-paid.
    await propose_refund(
        ledger,
        policy,
        trail,
        customer="cust:4821",
        amount=12.50,
        ledger_entries=1,
        reason="duplicate shipping charge on order #A-9920",
    )

    # 2) Large refund: high dollar value (so it's tagged high_value) AND it would
    #    reverse a 6-charge subscription batch (blast_radius over the cap). Either
    #    one alone would hold it; here both fire, so the held decision lists both.
    await propose_refund(
        ledger,
        policy,
        trail,
        customer="cust:7763",
        amount=4_000.00,
        ledger_entries=6,
        reason="full-year subscription reversal after billing dispute",
    )

    # The narrative outcome.
    print("\nLedger:")
    print("-" * 60)
    print(f"  paid: {ledger.paid or '(none)'}")
    print(f"  held: {ledger.held or '(none)'}")

    print("\nAudit trail:")
    print("-" * 60)
    for rec in trail.records():
        print(
            f"  #{rec.seq} {rec.event_type}: {rec.payload['action']} "
            f"{rec.payload['asset']} -> {rec.payload['outcome']}"
        )
    print(f"  chain intact (tamper-evident): {trail.verify()}")

    # Assert the gate did exactly what we claimed: the small refund moved money,
    # the large one did not, and BOTH decisions were recorded on the trail.
    assert ledger.paid == [("cust:4821", 12.50)], ledger.paid
    assert ledger.held == [("cust:7763", 4_000.00)], ledger.held
    assert len(trail) == 2, len(trail)
    outcomes = [rec.payload["outcome"] for rec in trail.records()]
    assert outcomes == ["allow", "require_human"], outcomes
    assert trail.verify()
    print("\nOK: small refund paid, large refund held, both on the audit trail.")


if __name__ == "__main__":
    asyncio.run(main())
