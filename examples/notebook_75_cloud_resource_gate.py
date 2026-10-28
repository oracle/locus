# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Notebook 75: A cloud-ops agent that can only touch production with a human's say-so.

A cloud-ops agent is handy and dangerous in the same breath. It can free up a
forgotten dev box in seconds — and it can just as easily terminate the database
that the whole product runs on, or hand an outside principal admin over your
account. The model deciding *what* to do is not the place to enforce *whether*
it is allowed to. That belongs in code that runs before the API call.

This example puts :func:`locus.control.admit` in front of every cloud mutation.
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
    python examples/notebook_75_cloud_resource_gate.py
"""

from __future__ import annotations

import asyncio

from locus.control import (
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
# expects for `perform`. In a real agent these would call the OCI SDK, boto3, or
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


# One gated attempt: run the Action through admit() and narrate the outcome.


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
    print("Notebook 75: gating a cloud-ops agent's actions by blast radius and environment")
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

    # 3) Human-held: widen an IAM grant. Not production and blast radius 1,
    #    but the policy lists `iam` as a label that always needs a person —
    #    handing out permissions is how a confused agent escalates itself.
    await attempt(
        Action(
            name="grant_iam",
            asset="svc-deploy",
            blast_radius=1,
            environment="staging",
            kind="iam",
            tags=frozenset({"iam"}),
        ),
        grant_iam("svc-deploy", "iam:PassRole"),
        policy=policy,
        trail=trail,
    )

    # Show the account state. The dev box changed; the prod DB and the IAM
    # policy are exactly as they were, because those calls were blocked before
    # they could run.
    print("\nCloud account after all three attempts:")
    print("-" * 60)
    for iid, row in INVENTORY.items():
        print(f"  {iid:12}  env={row['env']:10}  state={row['state']:10}  size={row['size']}")
    for principal, perms in IAM.items():
        print(f"  {principal:12}  permissions={perms}")

    # The audit trail recorded ALL three attempts — the one that ran and the two
    # that were held. The hash chain is intact, so the record is tamper-evident.
    print("\nAudit trail:")
    print("-" * 60)
    for rec in trail.records():
        print(
            f"  #{rec.seq}  {rec.payload['action']:20}  -> {rec.payload['outcome']:14}  ({rec.payload['asset']})"
        )
    print(f"  chain intact (tamper-evident): {trail.verify()}")

    # Hard assertions: all three decisions are on the trail, the chain verifies,
    # the staging box was resized, and the production DB and IAM were untouched.
    assert len(trail) == 3, f"expected 3 audit records, got {len(trail)}"
    assert trail.records()[0].payload["outcome"] == "allow"
    assert trail.records()[1].payload["outcome"] == "require_human"
    assert trail.records()[2].payload["outcome"] == "require_human"
    assert trail.verify(), "audit chain failed integrity check"
    assert INVENTORY["i-dev-7a3"]["size"] == "t3.medium", "staging resize should have run"
    assert INVENTORY["i-prod-db1"]["state"] == "running", "prod DB must be untouched"
    assert "iam:PassRole" not in IAM["svc-deploy"], "IAM grant must not have run"

    print(
        "\nAll assertions passed: low-risk action ran, production and IAM changes held, all audited."
    )


if __name__ == "__main__":
    asyncio.run(main())
