#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 85: A support agent changes a customer account — admit() holds the big ones.

A customer-support agent resolves tickets by changing the customer's
account: applying a goodwill credit, bumping a plan, waiving a fee. Most
of these are routine and should just happen. A few move real money or
change what the customer is entitled to, and those should pause for a
human before anything is written.

The agent does not decide that for itself. Every account change runs
through ``admit()`` first. ``admit()`` weighs the change against a
:class:`ControlPolicy` you wrote, runs the side-effecting function only
if the policy allows it, and records the decision either way::

    Support agent proposes an account change (an Action)
       │
       ▼
    admit(action, perform, policy=…, trail=…)
       │
       ├─ small credit, single account, routine ─────────> ALLOW   → perform() runs
       │
       └─ plan upgrade / large credit (tagged high_value) > HOLD    → AdmissionError
                                                                       perform() never runs
       │
       ▼
    AuditTrail records BOTH decisions (allowed and held)

The gate is code that runs *before* the write, not a line in the agent's
prompt. A model that is talked into issuing a $500 credit still cannot
issue it: the high-value change holds for a human, and the attempt lands
on a tamper-evident audit trail. Fooling the model does not move money.

The policy here gates on the action's labels, not on a human reading the
ticket: a change tagged ``high_value`` always holds, and any change that
touches more than ``max_blast_radius`` accounts holds too. Everything
else auto-allows.

Run it
    # Fully offline — no network, no credentials. The "perform" is a
    # local stub that mutates an in-memory account dict.
    python examples/notebook_85_support_account_gate.py
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


# The "side effect" — a local stub standing in for the billing/account API.
#
# In production this would call the real account service. Here it just
# mutates an in-memory dict so the example runs offline with no creds.
# The point is that this function only ever runs once admit() has allowed
# it: there is no path to a write that the policy didn't clear.

ACCOUNTS: dict[str, dict[str, object]] = {
    "cust:4821": {"plan": "starter", "credit_balance_usd": 0.0},
}


async def apply_account_change(account: str, field: str, value: object) -> str:
    """Mutate the customer account. Stands in for a real billing API call."""
    ACCOUNTS.setdefault(account, {})[field] = value
    return f"{account}: set {field} = {value!r}"


# Driver.


async def main() -> None:
    print("Notebook 85: A support agent changes a customer account — admit() holds the big ones")
    print("=" * 60)

    # The support-ops knobs. Routine changes auto-allow; anything tagged
    # high_value, or touching more than one account, holds for a human.
    #
    # require_verification_score=0.0 turns off the security-grounding bar:
    # that check belongs to the security domain (a finding that survived
    # verification), not to a support agent crediting an account. Here the
    # gate is purely blast radius + labels.
    policy = ControlPolicy(
        require_verification_score=0.0,
        max_blast_radius=1,
        require_human_for=frozenset({"high_value"}),
    )
    trail = AuditTrail()

    # --- Change 1: a small goodwill credit. Routine — should auto-allow. ---
    print("\n--- Ticket SUP-7781: $5 goodwill credit for a late reply ---")
    small_credit = Action(
        name="apply_credit",
        asset="cust:4821",
        blast_radius=1,
        environment="live",
        kind="account_credit",
        tags=frozenset(),  # not high_value
    )
    print(f"  Agent proposes: {small_credit.name} (+$5) on {small_credit.asset}")
    try:
        result = await admit(
            small_credit,
            lambda: apply_account_change("cust:4821", "credit_balance_usd", 5.0),
            policy=policy,
            trail=trail,
        )
        print(f"  ✓ ALLOWED — the change ran: {result}")
    except AdmissionError as exc:
        print(f"  ✗ HELD — {exc.decision.outcome}: {exc.decision.reason}")

    # --- Change 2: a plan upgrade + $500 credit. High value — should hold. ---
    print("\n--- Ticket SUP-7782: upgrade to enterprise + $500 retention credit ---")
    big_change = Action(
        name="upgrade_plan_and_credit",
        asset="cust:4821",
        blast_radius=2,  # touches the plan entitlement *and* the billing balance
        environment="live",
        kind="account_credit",
        tags=frozenset({"high_value"}),
    )
    print(f"  Agent proposes: {big_change.name} (enterprise + $500) on {big_change.asset}")
    held = False
    try:
        await admit(
            big_change,
            lambda: apply_account_change("cust:4821", "plan", "enterprise"),
            policy=policy,
            trail=trail,
        )
        print("  ✓ ALLOWED — the change ran (unexpected)")
    except AdmissionError as exc:
        held = True
        print(f"  ⏸  HELD for a human — {exc.decision.outcome}: {exc.decision.reason}")
        print("     The upgrade did NOT run. It waits for a support lead to approve.")

    # The high-value change must not have touched the account.
    assert held, "high-value change should have been held for a human"
    assert ACCOUNTS["cust:4821"]["plan"] == "starter", "held change must not mutate the account"
    assert ACCOUNTS["cust:4821"]["credit_balance_usd"] == 5.0, "only the allowed credit applied"

    # --- The audit trail recorded BOTH decisions. ---
    print("\nAudit trail")
    print("-" * 60)
    records = [r for r in trail.records() if r.event_type == "action-admission"]
    for rec in records:
        p = rec.payload
        print(f"  {p['outcome']:<14} {p['action']:<26} {p['asset']}  —  {p['reason']}")

    outcomes = [r.payload["outcome"] for r in records]
    assert len(records) == 2, f"expected 2 admission records, got {len(records)}"
    assert "allow" in outcomes, "the small credit should be recorded as allowed"
    assert "require_human" in outcomes, "the plan upgrade should be recorded as held"
    assert trail.verify(), "the audit trail must verify (tamper-evident)"

    print("\nBoth decisions are on the trail: one allowed, one held. No write went unrecorded.")


if __name__ == "__main__":
    asyncio.run(main())
