# Support account gate

A customer-support agent works tickets by changing the customer's account:
applying a goodwill credit, upgrading a plan, waiving a fee. Most of those
changes are routine and should proceed without review — a $5 credit for a late
reply does not need a human. A plan upgrade, or a $500 retention credit, moves
real money and changes what the customer is entitled to; those should wait for
a support lead before anything is written — and the wait has to outlive the
process that started it.

This example places that decision in `admit()`, a gate that runs before the
account is touched, rather than in the agent's prompt. The agent can be
confused, jailbroken or persuaded to issue a $500 credit; the change still goes
through only if the `ControlPolicy` admits it, and the applied credit and the
held upgrade both land on a hash-chained `AuditTrail`. The account write itself
is a local stub that mutates an in-memory dict, so the script runs offline;
replacing it with a call to the billing API leaves the gate unchanged.

What this notebook covers:

- Describing each proposed change as an `Action` whose scope sets the two
  attributes this policy reads: `blast_radius`, the number of account
  entitlements the change touches, and a `high_value` tag on changes that cross
  the human-review line.
- A `ControlPolicy` with `max_blast_radius=1` and
  `require_human_for={"high_value"}`: a change to a single entitlement without
  the tag clears on its own, while a change that touches more than one
  entitlement, or carries the tag, holds for a person.
  `require_verification_score` is deliberately `0.0`: crediting an account is a
  business action, not a security finding, so the gate requires no verification
  verdict and reasons over scope and labels alone.
- `admit()` as the single enforcement point. It asks `approve()` for a decision,
  records that decision on the `AuditTrail` whether or not it allows, and only
  then awaits `perform`. The $5 credit (ticket SUP-7781) is applied; the
  enterprise upgrade with a $500 retention credit (ticket SUP-7782) trips both
  rules — blast radius 2 exceeds the maximum of 1, and the `high_value` label
  requires human approval — so `admit()` raises an `AdmissionError` whose
  `decision.outcome` is `require_human` and the account is left untouched.
- `trail.verify()` on the resulting chain. It confirms the SHA-256 chain is
  internally consistent — an edit, a reorder or a deletion from the middle
  fails it — but on its own it cannot catch records dropped off the end, or a
  chain re-hashed from an edited record onward; the bare call this script makes
  returns `True` for both. Anchor `trail.head` where the agent cannot reach it,
  pass it back as `verify(expected_head=…)` and sign the trail before treating
  it as evidence; see
  [what `verify()` catches](../api/control.md#what-verify-catches-and-the-one-thing-it-cannot).

- The held upgrade, taken to the person who decides it. Part 2 puts the same
  policy behind `gate_tool(..., on_refusal="interrupt")` on an agent with a
  `FileCheckpointer` and a `FileApprovals` store, so the hold pauses the run
  instead of refusing it. The process that paused then goes away; what is left
  on disk is enough. `pending_holds(checkpointer, thread_id)` reads the hold
  back from the thread's checkpoint — the approval id, the action and asset,
  the spec of the store it is in — and `hold.state(store)` re-checks it. A
  support lead approves from another process with
  `locus approvals --store file://… approve <id> --by lead@example.com --reason …`
  (run here as a subprocess), and a fresh agent built from the same policy,
  store file and checkpoint directory resumes the thread with
  `perform_dangling=True`: the upgrade runs exactly once, the approval is
  `consumed`, and the thread carries no hold. The store is a file so the
  example needs no service; `PostgresApprovals`, `RedisApprovals` and
  `OracleApprovals` take the same place when replicas share the store.

The drawing follows both changes of Part 1 through the gate: `admit()` hands
each `Action` to `approve()` with the policy, the $5 credit clears and
`perform()` updates the account, the upgrade is held and nothing is written,
and each decision is recorded on the `AuditTrail` before `perform()` runs or
the error is raised.

{{ locus_diagram gate-support-account }}

## Key lines

`apply_account_change` is the `perform` callable that the two `admit()` calls in
`main()` guard: a stand-in for the billing API that mutates an in-memory account
dict, and the only code that writes to the account, so it runs for the allowed
$5 credit and never for the held upgrade:

{{ locus_source examples/notebook_73_support_account_gate.py::apply_account_change }}

`_support_agent` is what each "process" of Part 2 builds: the gated upgrade
tool over the file-backed approval store and the file checkpointer. The first
agent runs the ticket and is parked on the hold; the second, built the same
way after the approval, resumes the thread and performs the call:

{{ locus_source examples/notebook_73_support_account_gate.py::_support_agent }}

## Run it

```bash
python examples/notebook_73_support_account_gate.py
```

Part 1 stands in for the agent by building the two proposed `Action` values
itself and needs no model. Part 2 drives an agent, and follows the series
convention in `examples/config.py`: with no provider configured the model is a
scripted test double, so the whole script runs offline with no credentials or
network; set `LOCUS_MODEL_PROVIDER` (and `LOCUS_MODEL_ID`) and the agent that
proposes the upgrade, and the fresh one that resumes the thread, run on the
configured model, which the output names:

```bash
LOCUS_MODEL_PROVIDER=oci LOCUS_MODEL_ID=openai.gpt-4.1 python examples/notebook_73_support_account_gate.py
```

The only side effect is a write to an in-memory dict. The approval store and
the checkpoints live in a temporary directory, and the "other process" is the
`locus` command run as a subprocess.

## Output

Running it offline — no credentials, bundled mock model — prints a small credit applied, a plan upgrade held, and the trail that records both; then the same upgrade held by a gated tool with the run parked, the hold read back from the thread's checkpoint, approved from another process with `locus approvals`, and performed once by a fresh agent resuming the thread:

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

Part 2: the held upgrade waits for a support lead, across a restart
------------------------------------------------------------
  ⏸  run parked on upgrade_plan_and_credit for cust:4821
  checkpoint carries the hold: upgrade_plan_and_credit on cust:4821
  approval appr-585c1b9d9878d6a6-1 is pending
  store: file://…/approvals.json
  locus approvals: appr-585c1b9d9878d6a6-1: approved by lead@example.com
  agent: Upgrade applied and credit added.
  approval appr-585c1b9d9878d6a6-1 is now consumed (decided by lead@example.com)
  holds left on the thread: []
  account: {'plan': 'enterprise', 'credit_balance_usd': 505.0}

The upgrade ran once, after a named person approved it, in a process that never saw the hold placed.
```
<!-- notebook-output:end -->

## Source

??? note "Full source — `examples/notebook_73_support_account_gate.py`"

    ````python
    --8<-- "examples/notebook_73_support_account_gate.py"
    ````
