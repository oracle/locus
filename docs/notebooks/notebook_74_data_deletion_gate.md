# A privacy gate that exports on its own and holds erasure for a human

A privacy agent works a queue of GDPR data-subject requests. Some are safe to
run on the agent's own authority — assembling a portable copy of someone's data
under Article 15/20 only copies; it does not change anything. One is not: a
right-to-erasure delete under Article 17 permanently destroys every record tied
to a person. The middle path is a gate: the reversible export goes through
automatically; the irreversible erasure stops and waits for a named human.

The decision of *whether the data is deleted* does not live in the agent's
prompt. It lives in ``admit``, a gate that runs before the side effect touches
the datastore. The agent can be confused, prompt-injected, or simply wrong about
which request it is processing; the erasure still only happens if the
``ControlPolicy`` admits it.

{{ locus_diagram gate-data-deletion }}

Two plain fields on one ``ControlPolicy`` do the gating — no DSL, no rules
engine. ``require_human_for={"irreversible"}`` means any action carrying the
``irreversible`` tag always stops for a person, no matter how few records it
touches. The export is left to clear on scope alone, capped by
``max_blast_radius`` (an action may auto-proceed only if it touches at most 5
data categories). ``require_verification_score`` is set to ``0.0`` because a
data-subject request is a compliance action, not a security finding: there is no
threat verdict to weigh, so the gate reasons purely over tags and scope.

``admit()`` is the single enforcement point. It asks the policy, records the
decision to the ``AuditTrail`` whether or not it allows, and only then awaits
``perform``. There is no path to the side effect that skips the log. When the
erasure is held, the script asserts the subject is still present in the
datastore — proof the gate blocked the delete in code that runs *before* the
side effect, not in a prompt instruction the model could circumvent.

After the hold, a human-in-the-loop step stands in for the Data Protection
Officer resolving the ticket. The erasure is then re-admitted under a policy that
no longer auto-holds irreversible actions — because a person is now the one
authorizing this specific signed-off ticket — and lands on the *same* trail. So
the record shows three events in order: the auto-allowed export, the held
erasure, and the erasure that ran only after approval. That third record is a
plain ``allow`` and does not name the DPO. To put the approver on the trail,
re-admit the held action under the original policy with
``admit(..., approved_by="dpo:alice@corp")`` instead: the action runs, and its
record keeps the hold outcome (``require_human``) and gains an ``approved_by``
field.

``trail.verify()`` confirms the SHA-256 chain is internally consistent: edit or
reorder a record, or delete one from the middle, and it returns ``False``. On
its own it cannot catch records dropped off the end, or a chain rewritten and
re-hashed from an edited record onward; the bare call this script makes returns
``True`` for both. Before you hand the exported trail to a regulator, keep
``trail.head`` where the agent cannot reach it and pass it back as
``verify(expected_head=...)``, and sign the trail — see
[what `verify()` catches](../api/control.md#what-verify-catches-and-the-one-thing-it-cannot).

``perform`` here is a local stub: ``export`` reads from and ``erase`` pops from an
in-memory dict instead of mutating Postgres, S3, or a search index, so the script
runs offline with no datastore, no credentials, no network. Swap those two
functions for real deletes and the gate is unchanged.

Run it (fully offline — no model, no provider, no network):

    python examples/notebook_74_data_deletion_gate.py

## Output

Running it offline — no model and no credentials — prints a GDPR erasure, and the chain that proves it:

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

## Source

````python
--8<-- "examples/notebook_74_data_deletion_gate.py"
````
