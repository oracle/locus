# Customer Support Concession Approval

Support organizations gate costly concessions behind a tier-based
escalation chain: the more a concession costs and the more precedent it
sets, the more approvals it takes.

This example runs four support tickets of rising risk through a
`StateGraph`. A ticket analyst summarizes what the customer is asking
for, an impact analyst weighs the cost of the concession and its
precedent risk against the customer's standing and the downside of a
denial (churn, escalation, public complaint), and a risk-tier router
sends the case to automatic approval or to up to three human gates —
support manager, billing lead, support director — each a separate
`interrupt()`.

The terminal node, the decision recorder (the
`scribe-decision-recorder` agent in the source), emits a typed
`ConcessionDecision` Pydantic model, ready to file into a concessions
ledger without parsing. A large refund or contract make-good commits
real money and sets a precedent other customers will cite, which is why
the impact assessment sits in front of the gates.

What this notebook covers:

- Risk-tier router is a plain conditional edge — no domain-specific
  language (DSL), no policy file.
- Each gate is its own node — easy to add a tier, easy to re-order,
  easy to swap a human gate for an automated rule.
- `output_schema=ConcessionDecision` keeps the terminal artifact typed.

The drawing is the graph the example builds: the risk score arrives with
the ticket, a score under 25 is approved automatically, 25–49 stops at
the support manager, 50–74 adds the billing lead, 75 and above adds the
support director, and a no at any gate goes straight to the decision
recorder with status `denied`:

{{ locus_diagram concession-approval }}

## Key lines

One factory builds the three human gates. The `interrupt()` payload
carries the tier, the question and the yes/no options — all the run loop
needs to print the gate — and `_record_decision` turns the answer into
state: a yes appends the approver's title to `approvals`, a no sets
`status` to `denied`, which is what the gate functions check before
climbing a tier:

{{ locus_source examples/notebook_78_concession_approval.py::_approval_node }}

The wiring is ordinary graph edges. `risk_tier_router` is a plain
conditional edge off the impact assessment: a score under 25 goes to
`auto_approve`, every other tier enters at the manager gate, and
`gate_after_manager` and `gate_after_billing` climb to the next tier only
while the score requires it and no gate has said no:

{{ locus_source examples/notebook_78_concession_approval.py::build_review_graph }}

The run loop answers each gate from a scripted list. `execute()` returns
with `result.interrupt` set at every gate and the reviewer's answer goes
back inside `Command(resume=...)`. The graph has no checkpointer, so the
state travels back in the same `Command`: the returned `final_state`
carries a `__resume_node__` marker that `execute()` pops to find the gate
to re-run, and the loop keeps the model object in that state so every
later node can still build its agent:

{{ locus_source examples/notebook_78_concession_approval.py::_drive }}

## Run it

```bash
python examples/notebook_78_concession_approval.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

The decision recorder needs a model that can honor the `ConcessionDecision`
JSON schema. The mock model returns plain text, so under it `output_schema`
yields no parsed `ConcessionDecision` and the `record_decision` node raises.
`execute()` records the failed node and returns `success=False` rather than
propagating the error, and `main()` does not check `result.success`, so the
script prints the gates and the reviewer answers, reports the final decision
as `(missing)` and still exits with status 0:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_78_concession_approval.py
```

Pin a model that supports structured output; the direct `openai` form needs
`OPENAI_API_KEY` in the environment:

```bash
export LOCUS_MODEL_PROVIDER=openai
export LOCUS_MODEL_ID=gpt-4o
python examples/notebook_78_concession_approval.py
```

## Output

Running it offline — no credentials, bundled mock model — prints each ticket's risk tier and the gates it climbs: the 12/100 credit approved with no gate, the manager gate alone at 38, the manager then the billing lead at 62 (where the scripted reviewer says no, so the director gate is never reached) and all three gates at 88; the decision itself prints as `(missing)` because the mock model cannot fill the `ConcessionDecision` schema, as the Run it section explains:

```text
Notebook 78: Customer-support concession approval with risk-tiered gates
============================================================

--- CS-1001: risk 12/100 — Dana R. (Starter plan) — $15 account credit for a late shipping notification ---

Concession decision:
------------------------------------------------------------
(missing)

--- CS-1002: risk 38/100 — Marco T. (Pro plan) — one-month subscription refund after a billing double-charge ---
  ⏸  [manager] Support Manager approval needed at risk score 38/100 (Marco T. (Pro plan) — one-month subscription refund after a billing double-charge). Approve?
  ▶  Reviewer responds: 'yes'

Concession decision:
------------------------------------------------------------
(missing)

--- CS-1003: risk 62/100 — Acme Studio (Team plan, 40 seats) — full annual refund plus a goodwill credit after repeated outages ---
  ⏸  [manager] Support Manager approval needed at risk score 62/100 (Acme Studio (Team plan, 40 seats) — full annual refund plus a goodwill credit after repeated outages). Approve?
  ▶  Reviewer responds: 'yes'
  ⏸  [billing] Billing Lead approval needed at risk score 62/100 (Acme Studio (Team plan, 40 seats) — full annual refund plus a goodwill credit after repeated outages). Approve?
  ▶  Reviewer responds: 'no'

Concession decision:
------------------------------------------------------------
(missing)

--- CS-1004: risk 88/100 — Globex Corp (Enterprise, $480k ARR) — SLA-breach service credit and a contract make-good after a data-export failure ---
  ⏸  [manager] Support Manager approval needed at risk score 88/100 (Globex Corp (Enterprise, $480k ARR) — SLA-breach service credit and a contract make-good after a data-export failure). Approve?
  ▶  Reviewer responds: 'yes'
  ⏸  [billing] Billing Lead approval needed at risk score 88/100 (Globex Corp (Enterprise, $480k ARR) — SLA-breach service credit and a contract make-good after a data-export failure). Approve?
  ▶  Reviewer responds: 'yes'
  ⏸  [director] Support Director approval needed at risk score 88/100 (Globex Corp (Enterprise, $480k ARR) — SLA-breach service credit and a contract make-good after a data-export failure). Approve?
  ▶  Reviewer responds: 'yes'

Concession decision:
------------------------------------------------------------
(missing)
```
<!-- notebook-output:end -->

## See also

- [Notebook 15 — Human in the loop](notebook_15_human_in_the_loop.md) —
  `interrupt(payload)`, `Command(update=..., resume=...)` and
  `interrupt_before`
- [Concepts: Interrupts](../concepts/interrupts.md)
- [Concepts: Structured output](../concepts/structured-output.md) —
  `output_schema` and `AgentResult.parsed`

## Source

??? note "Full source — `examples/notebook_78_concession_approval.py`"

    ````python
    --8<-- "examples/notebook_78_concession_approval.py"
    ````
