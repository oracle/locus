# An account gate that applies small credits and holds the big changes

A customer-support agent works tickets by changing the customer's account:
applying a goodwill credit, bumping a plan, waiving a fee. Most of those are
routine and should just happen — a $5 credit for a late reply does not need a
human. But a plan upgrade, or a $500 retention credit, moves real money and
changes what the customer is entitled to. Those should pause and wait for a
support lead before anything is written.

The decision of *whether the change lands* does not live in the agent's prompt.
It lives in ``admit()``, a gate that runs before the account is touched. The
agent can be confused, jailbroken, or talked into a $500 credit; the change
still only goes through if the ``ControlPolicy`` admits it. Each proposed change
becomes an ``Action`` whose scope drives two policy-visible attributes: how many
account entitlements it touches (``blast_radius``) and whether it crosses the
human-review line (a ``high_value`` tag). The policy auto-allows a
single-field change that carries no ``high_value`` tag and holds anything bigger.

{{ locus_diagram gate-support-account }}

The rule lives in one ``ControlPolicy``. ``require_human_for={"high_value"}``
means any change carrying the ``high_value`` label always stops for a person —
no matter how small the blast radius looks. Routine changes are left to clear on
blast radius alone, capped by ``max_blast_radius=1``.
``require_verification_score`` is set to ``0.0`` on purpose: crediting an account
is a business action, not a security finding — there is no threat to verify, so
the gate reasons purely over scope and labels.

``admit()`` is the single enforcement point. It asks the policy (``approve()``),
records the decision to the ``AuditTrail`` whether or not it allows, and only
then awaits ``perform``. There is no path to the write that skips the log — the
held plan upgrade lands on the trail next to the small credit that actually ran,
and ``trail.verify()`` confirms the SHA-256 chain is internally consistent: an
edit, a reorder, or a deletion from the middle fails it. On its own it cannot
catch records dropped off the end, or a chain rewritten and re-hashed from an
edited record onward; the bare call this script makes returns ``True`` for both.
Anchor ``trail.head`` where the agent cannot reach it and pass it back as
``verify(expected_head=…)``, and sign the trail, before treating it as evidence
— see
[what `verify()` catches](../api/control.md#what-verify-catches-and-the-one-thing-it-cannot).

In this run the small change (one field, a $5 credit on ticket SUP-7781) is
applied automatically, while the big one (ticket SUP-7782: upgrade to enterprise
plus a $500 retention credit) trips *both* rules — the blast radius exceeds
``max_blast_radius`` and the ``high_value`` tag is in ``require_human_for`` — so
it raises an ``AdmissionError(require_human)``, the upgrade never runs, and the
account is left untouched. Both decisions are recorded on the trail.

``apply_account_change`` here is a local stub: it mutates an in-memory account
dict instead of calling the billing/account API, so no account service is needed
to run the script. Swap that one function for a real account-API call and the
gate is unchanged.

Run it (fully offline — no model, no provider, no network):

    python examples/notebook_73_support_account_gate.py

## Output

Running it offline — no model and no credentials — prints a small credit applied, a plan upgrade held, and the trail that records both:

```text
Notebook 73: A support agent changes a customer account — admit() holds the big ones
============================================================

--- Ticket SUP-7781: $5 goodwill credit for a late reply ---
  Agent proposes: apply_credit (+$5) on cust:4821
  ✓ ALLOWED — the change ran: cust:4821: set credit_balance_usd = 5.0

--- Ticket SUP-7782: upgrade to enterprise + $500 retention credit ---
  Agent proposes: upgrade_plan_and_credit (enterprise + $500) on cust:4821
  ⏸  HELD for a human — require_human: blast radius 2 exceeds the maximum 1; labels ['high_value'] require human approval
     The upgrade did NOT run. It waits for a support lead to approve.

Audit trail
------------------------------------------------------------
  allow          apply_credit               cust:4821  —  all policy checks passed
  require_human  upgrade_plan_and_credit    cust:4821  —  blast radius 2 exceeds the maximum 1; labels ['high_value'] require human approval

Both decisions are on the trail: one allowed, one held. No write went unrecorded.
```
<!-- notebook-output:end -->

## Source

````python
--8<-- "examples/notebook_73_support_account_gate.py"
````
