# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Notebook 72: An ops agent that ships to staging on its own and waits for a human in production.

A deploy agent watches CI, and when a build is green it proposes the next
action: roll the new image onto staging, or promote / roll back production.
Left to act on its own, the agent can take the checkout API down overnight
with nobody watching; requiring a human for every change stops releases
altogether. The middle path is a gate: cheap, reversible staging changes go
through automatically; any change to the production environment stops and
waits for a human.

    Build is green
       │
       ▼
    Ops agent proposes an Action  (deploy / rollback, with environment + blast radius)
       │
       ▼
    admit(action, perform, policy, trail)
       │
       ├─ staging, blast_radius <= max_blast_radius  ──> ALLOW: perform() runs, kubectl applies
       │
       └─ environment == "production"                ──> require_human: AdmissionError, nothing runs

The rule lives in one `ControlPolicy`. `require_human_for={"production"}`
means any action carrying the `production` label always stops for a person —
no matter how small the blast radius looks. Staging is left to clear on
blast radius alone. `admit()` is the enforcement point: it asks the policy,
records the decision to the `AuditTrail` whether or not it allows, and only
then awaits `perform`. There is no path to the side effect that skips the
log — the held production rollback is on the trail next to the staging
deploy that actually ran.

An optional third part adds a `ToolCallJudge`: a model that reads the
proposed call itself — tool name and arguments — against the intent in plain
words, and can turn the policy's allow into a hold. It can never loosen a
policy decision, and its verdict lands on the same audit entry.

`perform` here is a local stub: it appends to an in-memory list instead of
calling kubectl, so the script runs offline with no cluster, no credentials,
no network. Swap that one function for a real `kubectl rollout` call and the
gate is unchanged. The judge follows the series convention: with no provider
configured it runs on the SDK's scripted test double, so that part is
offline too; with ``LOCUS_MODEL_PROVIDER`` set it is the configured model
reading the call.

Run it:
    python examples/notebook_72_infra_deploy_gate.py

    # Offline — the gate needs no model and the judge step uses the bundled
    # scripted double. No provider, no network.

    export LOCUS_MODEL_PROVIDER=oci LOCUS_MODEL_ID=openai.gpt-4.1
    python examples/notebook_72_infra_deploy_gate.py
    # The judge is the configured model; its id is on the audit entries.
"""

from __future__ import annotations

import asyncio
import json

from config import get_model, resolved_provider

from locus.control import (
    Action,
    AdmissionError,
    AuditTrail,
    ControlPolicy,
    ToolCallJudge,
    admit,
    gate_tool,
)
from locus.testing import ScriptedModel, text
from locus.tools.decorator import tool


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
        print("  ALLOWED — the agent acted on its own authority.")
        print(f"  Applied: {result}")
    except AdmissionError as exc:
        decision = exc.decision
        print(f"  HELD ({decision.outcome}) — nothing was applied.")
        print(f"  Reason: {decision.reason}")
        print("  Next: route to the on-call human for sign-off.")


# Optional: a model that reads the call.
#
# Everything above is deterministic. The policy weighs the labels an Action
# carries, and labels are all it can see: a staging rollout pinned to 1.8.2
# and one to the floating `latest` tag carry the same labels, so the gate
# allows both. A `ToolCallJudge` reads the tool call itself — name and
# arguments — against the intent written in words plus a line of guidance for
# this tool, and returns allow / hold / deny with a reason. It is folded into
# the admission decision so that it can only tighten it: the policy's allow may
# become a hold or a deny; a policy hold or deny stands whatever the judge
# says. The verdict rides on the same audit entry.
#
# The judge model follows `examples/config.py`: the SDK's scripted test
# double when no provider is configured, so this step is offline and its
# two verdicts are written down; the configured model when
# LOCUS_MODEL_PROVIDER is set, so the verdicts below are the model's own.
# On an agent, leave `model=` unset and the judge borrows the agent's.


@tool
def rollout(asset: str, image: str, environment: str = "staging") -> str:
    """Roll `image` onto the `asset` deployment in `environment`."""
    record = f"deploy {asset} -> {image} ({environment})"
    APPLIED.append(record)
    return record


SCRIPTED_JUDGE = ScriptedModel(
    [
        text(json.dumps({"verdict": "allow", "reason": "staging rollout to a pinned image"})),
        text(
            json.dumps(
                {
                    "verdict": "hold",
                    "reason": "image tag 'latest' is not pinned; a person should confirm",
                }
            )
        ),
    ]
)

OFFLINE = resolved_provider() == "mock"
JUDGE_MODEL = SCRIPTED_JUDGE if OFFLINE else get_model()

JUDGE = ToolCallJudge(
    JUDGE_MODEL,
    intent=(
        "Ship CI-green builds to staging without waiting. Never change production without a person."
    ),
    guidance={"rollout": "Hold any image that is not pinned to a version, such as `latest`."},
    budget=10,
)


def deploy_action(name: str, args: dict) -> Action:
    """The Action the policy weighs, derived from the call's arguments."""
    return Action(
        name="deploy",
        asset=str(args["asset"]),
        blast_radius=3,
        environment=str(args.get("environment", "staging")),
        kind="deploy",
        tags=frozenset({"rollout"}),
    )


async def propose_judged(trail: AuditTrail, asset: str, image: str, environment: str) -> None:
    """Send one rollout through the gate with the judge reading the call."""
    gated = gate_tool(
        rollout,
        policy=POLICY,
        action=deploy_action,
        trail=trail,
        judge=JUDGE,
        on_refusal="raise",
    )
    print(f"\n--- judged: deploy {asset} -> {image} ({environment}) ---")
    try:
        result = await gated.execute(asset=asset, image=image, environment=environment)
        print("  ALLOWED — policy allow, judge allow.")
        print(f"  Applied: {result}")
    except AdmissionError as exc:
        decision = exc.decision
        print(
            f"  HELD ({decision.outcome}) — the policy said {decision.policy_outcome}; "
            "the judge tightened it. Nothing was applied."
        )
        print(f"  Reason: {decision.reason}")


async def main() -> None:
    print(
        "Notebook 72: An ops agent that ships to staging on its own and waits for a human in production"
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

    # 3) Optional: the same staging rollout twice over, with the judge reading
    #    the call. Both carry the labels the policy allows; only the pinned
    #    image gets through.
    await propose_judged(trail, "checkout-api", "checkout-api:1.8.2", "staging")
    await propose_judged(trail, "checkout-api", "checkout-api:latest", "staging")

    # The audit trail — every decision is on it, allowed or not, and the two
    # judged entries carry the verdict next to the policy outcome.
    print("\nAudit trail")
    print("-" * 60)
    records = trail.records()
    for rec in records:
        payload = rec.payload
        line = f"  {rec.event_type}: {payload['action']} {payload['asset']} -> {payload['outcome']}"
        judge = payload.get("judge")
        if judge is not None:
            line += (
                f"  [judge {judge['model']}: {judge['verdict']}, cache_hit={judge['cache_hit']}]"
            )
        print(line)
    print(f"  chain intact: {trail.verify()}")

    # What actually touched the cluster: the staging deploy, and the pinned
    # staging rollout the judge let through.
    print(f"\nSide effects applied: {APPLIED}")

    # Assertions — the contract this example demonstrates.
    #
    # Both the allowed staging deploy and the held production rollback were
    # recorded. The judge let the pinned image through and held `latest`,
    # which the policy alone had allowed. Only the two allowed rollouts ran.
    # The hash chain still verifies.
    outcomes = [rec.payload["outcome"] for rec in records]
    assert len(records) == 4, f"expected 4 audit records, got {len(records)}"
    assert outcomes[:3] == ["allow", "require_human", "allow"], f"unexpected outcomes: {outcomes}"
    # The unpinned image: the scripted double holds it; a live judge may hold
    # or deny it, and either is a tightening the policy alone did not make.
    assert outcomes[3] in ("require_human", "deny"), f"unexpected outcomes: {outcomes}"
    assert records[3].payload["judge"]["verdict"] in ("hold", "deny")
    assert records[3].payload["judge"]["source"] == "model"
    assert APPLIED == [
        "deploy checkout-api -> checkout-api:1.8.2 (staging)",
        "deploy checkout-api -> checkout-api:1.8.2 (staging)",
    ], f"only the two allowed rollouts should have run, got: {APPLIED}"
    if OFFLINE:
        assert SCRIPTED_JUDGE.call_count == 2, "the judge was asked once per judged call"
    assert trail.verify(), "audit trail hash chain failed to verify"
    print(
        "\nOK — staging shipped, production held, the unpinned image held by the judge; all on the trail."
    )


if __name__ == "__main__":
    asyncio.run(main())
