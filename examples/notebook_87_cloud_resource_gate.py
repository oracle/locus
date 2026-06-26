#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 87: A cloud-ops agent that can only touch production with a human's say-so.

A cloud-ops agent is handy and dangerous in the same breath. It can free up a
forgotten dev box in seconds — and it can just as easily terminate the database
that the whole product runs on, or hand an outside principal admin over your
account. The model deciding *what* to do is not the place to enforce *whether*
it is allowed to. That belongs in code that runs before the API call.

This example puts :func:`tulip.control.admit` in front of every cloud mutation.
The agent proposes an :class:`Action`; the gate weighs it against a
:class:`ControlPolicy` you wrote; the actual cloud call (here a local stub —
no network, no credentials) fires *only* if the policy admits it. Small,
contained changes proceed on their own. Anything that touches a production
resource, or reaches across many resources at once, is held for a human.
Either way the decision lands on a tamper-evident :class:`AuditTrail`, so there
is no un-recorded path to a side effect.

    Agent proposes a cloud action
       │
       ▼
    admit(action, perform, policy, trail)
       │
       ├─ blast radius small AND not production ──> perform() runs (resize dev box)
       │
       └─ production OR wide blast radius ────────> AdmissionError, held for a human
                                                    (terminate prod DB, open IAM)
       │
       ▼
    AuditTrail records every attempt — admitted or not

The policy here gates on two attributes of the action:

- ``max_blast_radius`` — how many resources one action may touch and still
  auto-proceed. Resizing one dev instance is blast radius 1. Deleting an
  auto-scaling group of 12 nodes is blast radius 12.
- ``require_human_for`` — labels that always need a person. ``production`` is
  the default; you can add more (``iam``, ``billing``, a specific tag).

A model that is jailbroken, prompt-injected, or just wrong still cannot push a
production change past this gate. The check is not advice in the system prompt —
it is the function that owns the cloud call.

Run it (fully offline — no cloud account, no credentials needed):
    python examples/notebook_87_cloud_resource_gate.py
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


# A tiny in-memory stand-in for the cloud account. No SDK, no network — every
# "API call" below just mutates this dict so the example runs anywhere.

INVENTORY: dict[str, dict[str, str]] = {
    "i-dev-7a3": {"role": "ci-runner", "env": "staging", "state": "running", "size": "t3.small"},
    "i-prod-db1": {
        "role": "primary-db",
        "env": "production",
        "state": "running",
        "size": "r6i.4xlarge",
    },
}

IAM: dict[str, list[str]] = {
    "svc-deploy": ["ecr:Pull", "ecs:UpdateService"],
}


# The side effects. Each is a zero-arg async callable — exactly what admit()
# expects for `perform`. In a real agent these would call boto3 / the OCI SDK /
# gcloud; here they just edit the dicts above and return a short receipt.


def resize_instance(instance_id: str, new_size: str):
    async def _perform() -> str:
        INVENTORY[instance_id]["size"] = new_size
        return f"resized {instance_id} -> {new_size}"

    return _perform


def terminate_instance(instance_id: str):
    async def _perform() -> str:
        INVENTORY[instance_id]["state"] = "terminated"
        return f"terminated {instance_id}"

    return _perform


def grant_iam(principal: str, permission: str):
    async def _perform() -> str:
        IAM.setdefault(principal, []).append(permission)
        return f"granted {permission} to {principal}"

    return _perform


# Driver.


async def attempt(action: Action, perform, *, policy: ControlPolicy, trail: AuditTrail) -> None:
    """Run one gated action and narrate what the runtime decided."""
    print(f"\n--- agent proposes: {action.name} ---")
    print(
        f"    asset={action.asset!r}  env={action.environment!r}  blast_radius={action.blast_radius}"
    )
    try:
        receipt = await admit(action, perform, policy=policy, trail=trail)
        print(f"  ✅ ADMITTED — the cloud call ran: {receipt}")
    except AdmissionError as exc:
        print(f"  ⏸  HELD FOR A HUMAN — outcome={exc.decision.outcome}")
        print(f"     why: {exc.decision.reason}")
        print("     the cloud call did NOT run; the resource is untouched.")


async def main() -> None:
    print("Notebook 87: gating a cloud-ops agent's actions by blast radius and environment")
    print("=" * 60)

    # The policy. Verification scoring is off here (require_verification_score=0)
    # so the example stays focused on the two cloud knobs: blast radius and the
    # production label. One resource at a time may auto-proceed; production
    # always needs a human; opening IAM always needs a human.
    policy = ControlPolicy(
        require_verification_score=0.0,
        max_blast_radius=1,
        require_human_for=frozenset({"production", "iam"}),
    )
    trail = AuditTrail()

    # 1) Auto-allowed: resize one staging CI runner. Blast radius 1, not
    #    production — nothing the agent could be tricked into here is dangerous,
    #    so the gate lets it through without a human.
    await attempt(
        Action(
            name="resize_instance",
            asset="i-dev-7a3",
            blast_radius=1,
            environment="staging",
            kind="compute",
            tags=frozenset({"ci"}),
        ),
        resize_instance("i-dev-7a3", "t3.medium"),
        policy=policy,
        trail=trail,
    )

    # 2) Human-held: terminate the production primary database. Same shape of
    #    action, very different consequence — the `production` label trips the
    #    gate and the terminate call never fires.
    await attempt(
        Action(
            name="terminate_instance",
            asset="i-prod-db1",
            blast_radius=1,
            environment="production",
            kind="compute",
            tags=frozenset({"database"}),
        ),
        terminate_instance("i-prod-db1"),
        policy=policy,
        trail=trail,
    )

    # Show the account state. The dev box changed; the prod DB is exactly as it
    # was, because its terminate call was blocked before it could run.
    print("\nCloud account after both attempts:")
    print("-" * 60)
    for iid, row in INVENTORY.items():
        print(f"  {iid:12}  env={row['env']:10}  state={row['state']:10}  size={row['size']}")

    # The audit trail recorded BOTH attempts — the one that ran and the one that
    # was held. The hash chain is intact, so the record is tamper-evident.
    print("\nAudit trail:")
    print("-" * 60)
    for rec in trail.records():
        print(
            f"  #{rec.seq}  {rec.payload['action']:20}  -> {rec.payload['outcome']:14}  ({rec.payload['asset']})"
        )
    print(f"  chain intact (tamper-evident): {trail.verify()}")

    # Hard assertions: both decisions are on the trail, the chain verifies, the
    # staging box was resized, and the production DB was left untouched.
    assert len(trail) == 2, f"expected 2 audit records, got {len(trail)}"
    assert trail.records()[0].payload["outcome"] == "allow"
    assert trail.records()[1].payload["outcome"] == "require_human"
    assert trail.verify(), "audit chain failed integrity check"
    assert INVENTORY["i-dev-7a3"]["size"] == "t3.medium", "staging resize should have run"
    assert INVENTORY["i-prod-db1"]["state"] == "running", "prod DB must be untouched"

    print("\nAll assertions passed: low-risk action ran, production change held, both audited.")


if __name__ == "__main__":
    asyncio.run(main())
