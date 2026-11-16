---
title: First controlled action — allow, hold, deny, and inspect
description: Run a credential-free refund example that demonstrates all three admission outcomes and its audit record.
---

# First controlled action

This tutorial runs entirely offline. It uses a local ledger stub so you can
observe policy behavior without credentials or real money moving.

## What this demonstrates

| Field | Value |
|---|---|
| Status | Supported core API |
| Execution mode | Offline simulation |
| Requirements | Python 3.11+, build-tested SDK {{ locus_sdk_version }} |
| External credentials | None |
| Limitation | The ledger is in memory; no payment provider is contacted |

Three actions enter the same gate:

- a $12.50 refund is allowed and reaches the ledger;
- a $4,000 refund is held and does not reach the ledger;
- an action tagged `prohibited` is denied and does not reach the ledger.

## Save the example

Create `controlled_refund.py`:

```python
import asyncio

from locus.control import (
    Action,
    AdmissionError,
    AuditTrail,
    ControlPolicy,
    admit,
)


policy = ControlPolicy(
    require_verification_score=0.0,
    max_blast_radius=1,
    require_human_for=frozenset({"high_value"}),
    deny_for=frozenset({"prohibited"}),
)
trail = AuditTrail()
paid = []


async def try_refund(customer, amount, *, entries=1, tags=frozenset()):
    action = Action(
        name="issue_refund",
        asset=customer,
        kind="payment",
        environment="production",
        blast_radius=entries,
        tags=frozenset({"refund", *tags}),
    )

    async def pay():
        paid.append((customer, amount))
        return f"paid ${amount:,.2f}"

    try:
        result = await admit(action, pay, policy=policy, trail=trail)
        print("ALLOW", result)
    except AdmissionError as exc:
        print(exc.decision.outcome.upper(), "not run")


async def main():
    await try_refund("cust:small", 12.50)
    await try_refund("cust:large", 4_000, entries=6, tags={"high_value"})
    await try_refund("cust:blocked", 10, tags={"prohibited"})

    print("paid:", paid)
    print("outcomes:", [r.payload["outcome"] for r in trail.records()])
    print("chain valid:", trail.verify())


asyncio.run(main())
```

Run it:

```bash
python controlled_refund.py
```

Expected output:

```text
ALLOW paid $12.50
REQUIRE_HUMAN not run
DENY not run
paid: [('cust:small', 12.5)]
outcomes: ['allow', 'require_human', 'deny']
chain valid: True
```

## What the result proves

For these three calls, `admit()` invoked `pay()` only after the configured
policy returned `allow`. It wrote each decision to the supplied in-memory
trail, and `trail.verify()` confirmed that trail's hash chain was intact.

It does **not** prove that every refund path in an application is gated. You
must route each consequential path through `admit()` or `gate_tool()`, classify
the action accurately, store approvals and audit records durably, and use a
payment-provider idempotency key for crash-safe external effects.

## Deciding a hold from the command line

The `REQUIRE_HUMAN` above is where a person comes in. With
`gate_tool(..., on_refusal="interrupt", approval=store)` the run pauses on the
hold and the store keeps it; an operator with no UI decides with the `locus`
command, against the same store the agent uses:

```bash
locus approvals --store file://approvals.json list
locus approvals --store file://approvals.json approve appr-… --by alice@example.com --reason "ticket 4821"
locus approvals --store postgresql://locus@db.internal:5432/locus reject appr-… --by bob
```

`--store` takes `memory://`, `file://`, `postgresql://`, `redis://` or
`oracle://` (or `LOCUS_APPROVALS_STORE`), and `--by` falls back to
`LOCUS_APPROVER`. The [support account gate](../notebooks/notebook_73_support_account_gate.md)
runs this end to end offline: a held upgrade, approved from another process,
performed by a fresh agent.

## Next steps

- [Control-layer architecture](../concepts/control-layer.md) explains the
  enforcement boundary, the approval stores and the restart story.
- [Writing a policy that holds](../concepts/policy-authoring.md) covers missing
  risk families and second paths.
- [Payment refund example](../notebooks/notebook_71_payment_refund_gate.md)
  provides a more fully narrated runnable file.
- [Idempotency](../concepts/idempotency.md) explains retries and the external
  operation/recording crash window.
