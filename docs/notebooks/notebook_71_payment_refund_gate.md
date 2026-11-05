# A refund gate that pays small refunds and holds big ones

A support agent talks to an upset customer and decides a refund is owed. It
should be free to settle small amounts on its own — a $12 shipping credit, a
duplicate-charge reversal — without paging a human. But a $4,000 refund, or one
that reverses a whole batch of charges, should stop and wait for a person.

The decision of *whether the money moves* does not live in the agent's prompt.
It lives in `admit()`, a gate that runs before the payout. The agent can be
confused, manipulated by a prompt injection, or wrong about the dollar amount;
the refund still only goes through if the `ControlPolicy` admits it, and every
decision — the paid refund and the held one — lands on a tamper-evident
`AuditTrail`.

Each proposed refund becomes an `Action` with two policy-visible attributes:
how many ledger entries it touches (`blast_radius`) and whether its dollar
amount crosses the human-review line (a `high_value` tag). The policy
auto-allows a single-ledger-entry refund that carries no `high_value` tag and
holds anything bigger. In the script, the small refund (one ledger entry,
$12.50) is paid automatically, while the large one ($4,000 reversing a
six-charge subscription batch) trips *both* rules — the blast radius exceeds
`max_blast_radius` and the `high_value` tag is in `require_human_for` — so it
raises an `AdmissionError(require_human)`, the payout never runs, and the
script records the hold for a human approver.

What this notebook covers:

- Turning a proposed refund into an `Action` the policy reasons over: the
  ledger entries it touches become `blast_radius`, and an amount at or above
  $200 adds the `high_value` tag.
- A `ControlPolicy` with `max_blast_radius=1`,
  `require_human_for={"high_value"}` and `require_verification_score=0.0`. A
  refund is a business action, not a security finding: there is no threat to
  verify, so the gate reasons purely over amount and scope.
- `admit()` as the single enforcement point: the payout runs only on allow, a
  hold raises `AdmissionError` carrying the checks that fired, and both
  decisions are recorded on the `AuditTrail` before anything moves.
- What `trail.verify()` confirms — the SHA-256 chain is internally consistent,
  so an edit, a reorder or a deletion from the middle fails it — and what it
  cannot catch on its own: records dropped off the end, or a chain rewritten
  and re-hashed from an edited record onward. The bare call this script makes
  returns `True` for both; anchor `trail.head` where the agent cannot reach it,
  pass it back as `verify(expected_head=…)` and sign the trail before treating
  it as evidence — see
  [what `verify()` catches](../api/control.md#what-verify-catches-and-the-one-thing-it-cannot).

The drawing is the [control layer](../concepts/control-layer.md) as a gated
tool call traverses it; this notebook walks its central path by hand.
`refund_action()` builds the `Action` the gate would otherwise derive from the
agent's call, and `admit()` weighs it against the `ControlPolicy` and records
the decision on the `AuditTrail` before anything moves. On allow the
`pay_out()` side effect runs; a hold surfaces as an `AdmissionError` rather
than landing in an approval store, because the script sets up no interrupting
gate and no store. The agent loop, checkpointer and grounding in the drawing
play no part here:

{{ locus_diagram control-layer-flow }}

## Key lines

`refund_action()` turns a proposed refund into the `Action` the policy reasons
over: the number of ledger entries it touches becomes `blast_radius`, and an
amount at or above `LARGE_REFUND_THRESHOLD` ($200) adds the `high_value` tag:

{{ locus_source examples/notebook_71_payment_refund_gate.py::refund_action }}

`propose_refund()` is the gate call: `admit()` takes the `Action`, the payout
as a callable it awaits only on allow, and the `ControlPolicy` that `main()`
builds with `max_blast_radius=1` and `require_human_for={"high_value"}`. It
records the decision on the `AuditTrail` either way. A hold arrives as
`AdmissionError`; its `decision.checks` name the rules that stopped the
refund, and the script then records the hold for a person:

{{ locus_source examples/notebook_71_payment_refund_gate.py::propose_refund }}

`Ledger.pay_out()` is the one place money would move, reached only after
`admit()` allows the action. It is written as `async` because `admit()` awaits
the callable it is handed; `admit_sync()` is the counterpart for a synchronous
side effect:

{{ locus_source examples/notebook_71_payment_refund_gate.py::Ledger.pay_out }}

## Run it

```bash
python examples/notebook_71_payment_refund_gate.py
```

The payout is a local ledger stub and the script builds no model — its only
import is `locus.control` — so it runs fully offline with no provider, no
credentials and no network. `LOCUS_MODEL_PROVIDER` has no effect here.

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

??? note "Full source — `examples/notebook_71_payment_refund_gate.py`"

    ````python
    --8<-- "examples/notebook_71_payment_refund_gate.py"
    ````
