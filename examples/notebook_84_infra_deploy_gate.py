#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 84: An ops agent that ships to staging on its own and waits for a human in production.

A deploy agent watches CI, and when a build is green it proposes the next
action: roll the new image onto staging, or promote / roll back production.
Letting it act freely is how a 3 a.m. agent takes the checkout API down.
Blocking everything for a human is how you stop shipping. The middle path
is a gate: cheap, reversible staging changes go through automatically; any
change to the production environment stops and waits for a named human.

    Build is green
       │
       ▼
    Ops agent proposes an Action  (deploy / rollback, with environment + blast radius)
       │
       ▼
    admit(action, perform, policy, trail)
       │
       ├─ environment == "staging"     ──> ALLOW: perform() runs, kubectl applies
       │
       └─ environment == "production"  ──> require_human: AdmissionError, nothing runs

The rule lives in one `ControlPolicy`. `require_human_for={"production"}`
means any action carrying the `production` label always stops for a person —
no matter how small the blast radius looks. Staging is left to clear on
blast radius alone. `admit()` is the enforcement point: it asks the policy,
records the decision to the `AuditTrail` whether or not it allows, and only
then awaits `perform`. There is no path to the side effect that skips the
log — the held production rollback is on the trail next to the staging
deploy that actually ran.

`perform` here is a local stub: it appends to an in-memory list instead of
calling kubectl, so the script runs offline with no cluster, no creds, no
network. Swap that one function for a real `kubectl rollout` call and the
gate is unchanged.

Run it
    python examples/notebook_84_infra_deploy_gate.py

    # Fully offline — this example needs no model, no provider, no network.
"""

from __future__ import annotations

import asyncio

from tulip.control import Action, AdmissionError, AuditTrail, ControlPolicy, admit


# The side effect — a local stub standing in for `kubectl rollout`.
#
# In production this would shell out to the cluster. Here it just appends to
# a list so the example is deterministic and offline. `perform` is the only
# thing that touches the world; everything above it is the decision.


APPLIED: list[str] = []


def make_rollout(action: Action, image: str):
    """Build the zero-arg async callable `admit` will run only on ALLOW."""

    async def perform() -> str:
        # The real version: `kubectl set image deploy/<asset> app=<image>`.
        record = f"{action.name} {action.asset} -> {image} ({action.environment})"
        APPLIED.append(record)
        return record

    return perform


# The policy — the one place the deploy rule is written down.
#
# - require_verification_score=0.0 turns off the security-verification bar:
#   this is a pure ops gate, not a finding-response gate, so there is no
#   verdict to weigh.
# - max_blast_radius=4 lets a staging rollout across a few replicas clear on
#   its own.
# - require_human_for={"production"} is the spine: anything labelled
#   production always waits for a human, regardless of blast radius.


POLICY = ControlPolicy(
    require_verification_score=0.0,
    max_blast_radius=4,
    require_human_for=frozenset({"production"}),
)


async def propose(
    trail: AuditTrail,
    action: Action,
    image: str,
) -> None:
    """Hand one proposed action to the gate and narrate what happened."""
    print(
        f"\n--- {action.name} {action.asset} in {action.environment} "
        f"(blast radius {action.blast_radius}) ---"
    )
    try:
        result = await admit(
            action,
            make_rollout(action, image),
            policy=POLICY,
            trail=trail,
        )
        print(f"  ALLOWED — the agent acted on its own authority.")
        print(f"  Applied: {result}")
    except AdmissionError as exc:
        decision = exc.decision
        print(f"  HELD ({decision.outcome}) — nothing was applied.")
        print(f"  Reason: {decision.reason}")
        print(f"  Next: route to the on-call human for sign-off.")


async def main() -> None:
    print(
        "Notebook 84: An ops agent that ships to staging on its own and waits for a human in production"
    )
    print("=" * 60)

    trail = AuditTrail()

    # 1) Cheap, reversible staging deploy — the agent should just do it.
    staging_deploy = Action(
        name="deploy",
        asset="checkout-api",
        blast_radius=3,
        environment="staging",
        kind="deploy",
        tags=frozenset({"rollout"}),
    )
    await propose(trail, staging_deploy, image="checkout-api:1.8.2")

    # 2) Production rollback — small blast radius, but production. The gate
    #    holds it for a human no matter how safe it looks.
    prod_rollback = Action(
        name="rollback",
        asset="checkout-api",
        blast_radius=2,
        environment="production",
        kind="deploy",
        tags=frozenset({"rollback"}),
    )
    await propose(trail, prod_rollback, image="checkout-api:1.8.1")

    # The audit trail — both decisions are on it, allowed or not.
    print("\nAudit trail")
    print("-" * 60)
    records = trail.records()
    for rec in records:
        payload = rec.payload
        print(f"  {rec.event_type}: {payload['action']} {payload['asset']} -> {payload['outcome']}")
    print(f"  chain intact: {trail.verify()}")

    # What actually touched the cluster: only the staging deploy.
    print(f"\nSide effects applied: {APPLIED}")

    # Assertions — the contract this example demonstrates.
    #
    # Both the allowed staging deploy and the held production rollback were
    # recorded. Only the staging deploy ran. The hash chain still verifies.
    outcomes = [rec.payload["outcome"] for rec in records]
    assert len(records) == 2, f"expected 2 audit records, got {len(records)}"
    assert outcomes == ["allow", "require_human"], f"unexpected outcomes: {outcomes}"
    assert APPLIED == ["deploy checkout-api -> checkout-api:1.8.2 (staging)"], (
        f"only the staging deploy should have run, got: {APPLIED}"
    )
    assert trail.verify(), "audit trail hash chain failed to verify"
    print("\nOK — staging shipped, production held, both on the trail.")


if __name__ == "__main__":
    asyncio.run(main())
