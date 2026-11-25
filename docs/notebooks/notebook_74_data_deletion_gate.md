# Data deletion gate

A privacy agent works a queue of General Data Protection Regulation (GDPR)
data-subject requests. An Article 15/20 export assembles a portable copy of a
person's data and changes nothing; an Article 17 erasure permanently destroys
every record tied to that person. This example puts `admit()` between the
agent and the datastore: the reversible export clears the gate on its own, the
irreversible erasure stops and waits for the Data Protection Officer (DPO),
and every decision lands on one hash-chained `AuditTrail`.

The decision of whether the data is deleted does not live in the agent's prompt. Two fields on one `ControlPolicy` carry it:

- `require_human_for={"irreversible"}` holds any action whose labels include the `irreversible` tag, however few records it touches.
- `max_blast_radius=5` holds any action whose `blast_radius` is above five — here the number of data categories a request touches.

`require_verification_score=0.0` switches the verification check off, because a data-subject request is a compliance action with no threat verdict to weigh.

`admit()` asks the policy, records the decision on the trail whether or not it allows, and only then awaits `perform`, so an agent that is confused, prompt-injected or wrong about which request it is processing still cannot delete a record on its own authority.

After the hold a stub stands in for the DPO resolving the ticket, and the erasure is re-admitted under a second policy whose `require_human_for` is empty. It lands on the same trail, so the record shows the allowed export, the held erasure and the erasure that ran only after sign-off.

That third record is a plain `allow` and does not name the approver; re-admitting the held action under the original policy with `admit(..., approved_by="dpo:alice@corp")` runs it instead and writes a record that keeps the `require_human` outcome and gains an `approved_by` field.

`trail.verify()` confirms the SHA-256 chain is internally consistent: an edited or reordered record, or one deleted from the middle, returns `False`. A bare call cannot catch records dropped off the end or a chain re-hashed from an edited record onward, so keep `trail.head` where the agent cannot reach it and pass it back as `verify(expected_head=...)` before treating the trail as evidence — see [what `verify()` catches](../api/control.md#what-verify-catches-and-the-one-thing-it-cannot).

What this notebook covers:

- A `ControlPolicy` that gates on the `irreversible` tag and on `blast_radius`,
  with verification scoring switched off.
- `admit()` allowing the export, raising `AdmissionError` for the erasure and
  recording both decisions on the trail before any side effect runs.
- The DPO sign-off and the re-admission under a second policy, and
  `approved_by` as the way to name the approver on the record.
- `trail.verify()`: what a bare call proves about the SHA-256 chain and what
  `expected_head` adds.

The drawing follows both requests through the gate:

- `admit()` hands each `Action` to `approve()` with the policy.
- The Article 15/20 export carries no label in `require_human_for` and its blast radius of 4 is within the cap of 5, so it is allowed and `perform()` runs it.
- The Article 17 erasure is held because its `irreversible` tag is in `require_human_for`, and nothing is deleted.
- Once the DPO signs off, the erasure is re-admitted under a policy whose `require_human_for` is empty.
- Each decision is recorded on the same `AuditTrail` before its side effect runs.

{{ locus_diagram gate-data-deletion }}

## Key lines

The gate call sits in `main()`, collapsed under Source:
`admit(export, lambda: export_subject_data(SUBJECT), policy=policy, trail=trail)`
returns what this function produces, because none of the export's labels is in
`require_human_for` and its blast radius of 4 is within the cap of 5; the
function copies the subject's record out of the in-memory datastore where a
real version would read from a database, an object store or a search index:

{{ locus_source examples/notebook_74_data_deletion_gate.py::export_subject_data }}

The same call with the erasure raises `AdmissionError` before this function is
awaited — the exception's `decision.outcome` is `require_human` and its
`decision.reason` names the `irreversible` label — and the function runs only
when the erasure is re-admitted under `signed_off_policy`; it pops the record
from the datastore, which is why the script can assert that the subject is
still present after the hold and gone after the approved run:

{{ locus_source examples/notebook_74_data_deletion_gate.py::erase_subject_records }}

## Run it

```bash
python examples/notebook_74_data_deletion_gate.py
```

The example calls `admit()` directly and builds no model, so it runs fully
offline: no provider, no credentials, no network. `examples/config.py` and
`LOCUS_MODEL_PROVIDER` play no part in this notebook. The two side effects read
from and pop from an in-memory dict in place of a datastore; replace them with
the real export and erasure and the gate is unchanged.

## Output

Running it offline — no model and no credentials — prints the export allowed on its own, the erasure held with nothing deleted until the Data Protection Officer signs off and it is re-admitted, and the three-record trail that verifies intact:

```text
Notebook 74: A human signs off before an agent erases personal data
======================================================================

--- DSR-7781: Article 15/20 export request ---
  action : export_subject_data (export, blast_radius=4)
  outcome: ALLOW (auto)
  detail : exported 4 categories for subject:eu-44213: profile=name, email, postal address, orders=7 past orders with billing details, support_tickets=3 closed tickets, marketing_events=412 clickstream + email-open events

--- DSR-7782: Article 17 erasure request ---
  action : erase_subject_records (delete, blast_radius=4)
  outcome: require_human
  detail : labels ['irreversible'] require human approval
  note   : nothing deleted — the agent could not act on its own

--- Human-in-the-loop ---
  [DPO] reviewing held request: erase_subject_records on subject:eu-44213
  [DPO] verified identity, retention obligations, and legal holds
  [DPO] decision: APPROVE erasure

--- DSR-7782: erasure proceeds after sign-off ---
  action : erase_subject_records (delete, blast_radius=4)
  outcome: ALLOW (after human sign-off)
  detail : erased 4 categories for subject:eu-44213; nothing remains

--- Compliance record (AuditTrail) ---
  #0 action-admission: export_subject_data -> allow
  #1 action-admission: erase_subject_records -> require_human
  #2 action-admission: erase_subject_records -> allow
  chain intact (tamper-evident): True

OK: the export, the human hold, and the approved erasure are all on the record.
```
<!-- notebook-output:end -->

## See also

- [Control-layer architecture](../concepts/control-layer.md) — where `admit()`
  sits among the agent loop, approvals, persistence and audit.
- [Admitting an action](../api/control.md#admitting-an-action) — the `admit()`
  and `AdmissionError` reference, including `approved_by`.
- [What `verify()` catches, and the one thing it cannot](../api/control.md#what-verify-catches-and-the-one-thing-it-cannot)
  — anchoring `trail.head` and signing the trail.
- [Notebook 71 — Payment refund gate](notebook_71_payment_refund_gate.md) — the
  same gate over a refund's amount and ledger scope.
- [Notebook 75 — Cloud resource gate](notebook_75_cloud_resource_gate.md) — the
  same gate over `production` and `iam` labels.

## Source

??? note "Full source — `examples/notebook_74_data_deletion_gate.py`"

    ````python
    --8<-- "examples/notebook_74_data_deletion_gate.py"
    ````
