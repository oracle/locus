# A refund gate that pays small refunds and holds big ones

A support agent talks to an upset customer and decides a refund is owed. It
should be free to settle the small stuff on its own — a $12 shipping credit, a
duplicate-charge reversal — without paging a human. But a $4,000 refund, or one
that reverses a whole batch of charges, should stop and wait for a person.

The decision of *whether the money moves* does not live in the agent's prompt.
It lives in ``admit()``, a gate that runs before the payout. The agent can be
confused, jailbroken, or wrong about the dollar amount; the refund still only
goes through if the ``ControlPolicy`` admits it. Each proposed refund becomes an
``Action`` whose dollar amount drives two policy-visible attributes: how many
ledger entries it touches (``blast_radius``) and whether it crosses the
human-review line (a ``high_value`` tag). The policy auto-allows a
single-ledger-entry refund that carries no ``high_value`` tag and holds anything
bigger.

``require_verification_score`` is set to ``0`` on purpose: a refund is a business
action, not a security finding — there is no threat to verify, so the gate
reasons purely over amount and scope. Everything that happens — the paid refund
and the held one — lands on a tamper-evident ``AuditTrail``, and
``trail.verify()`` confirms the SHA-256 chain is internally consistent: an edit,
a reorder, or a deletion from the middle fails it. On its own it cannot catch
records dropped off the end, or a chain rewritten and re-hashed from an edited
record onward; the bare call this script makes returns ``True`` for both. Anchor
``trail.head`` where the agent cannot reach it and pass it back as
``verify(expected_head=…)``, and sign the trail, before treating it as evidence
— see
[what `verify()` catches](../api/control.md#what-verify-catches-and-the-one-thing-it-cannot).

In this run the small refund (one ledger entry, $12.50) is paid automatically,
while the large one ($4,000 reversing a six-charge subscription batch) trips
*both* rules — the blast radius exceeds ``max_blast_radius`` and the
``high_value`` tag is in ``require_human_for`` — so it raises an
``AdmissionError(require_human)``, the payout never runs, and the hold is queued
for a human approver. Both decisions are recorded on the trail.

The payout is a local ledger stub, so the script needs no credentials.

Run it (fully offline — no model, no provider, no network):

    python examples/notebook_71_payment_refund_gate.py

## Output

Running it offline — no model and no credentials — prints a small refund paid, a large one held:

```text
Notebook 71: A refund gate that pays out small refunds and holds big ones
============================================================

--- Support agent proposes: refund $12.50 to cust:4821 ---
    reason: duplicate shipping charge on order #A-9920
    scope:  1 ledger entry, tags=['refund']
  PAID    refunded $12.50 to cust:4821

--- Support agent proposes: refund $4,000.00 to cust:7763 ---
    reason: full-year subscription reversal after billing dispute
    scope:  6 ledger entries, tags=['high_value', 'refund']
  HELD    not admitted (require_human) — queued for a human
            · blast radius 6 exceeds the maximum 1
            · labels ['high_value'] require human approval

Ledger:
------------------------------------------------------------
  paid: [('cust:4821', 12.5)]
  held: [('cust:7763', 4000.0)]

Audit trail:
------------------------------------------------------------
  #0 action-admission: issue_refund cust:4821 -> allow
  #1 action-admission: issue_refund cust:7763 -> require_human
  chain intact (tamper-evident): True

OK: small refund paid, large refund held, both on the audit trail.
```
<!-- notebook-output:end -->

## Source

````python
--8<-- "examples/notebook_71_payment_refund_gate.py"
````
