# Incident response with a tamper-evident audit chain

A production outage pages the on-call at 2 AM. A ``GovernedAgent`` runs a
runbook defined in the notebook, ``sre_incident_runbook`` (detect → triage →
mitigate → recover). Every tool call — metrics query, log tail, deploy
rollback — is logged to a tamper-evident ``AuditTrail`` via ``AuditHook``.

After the incident concludes, ``trail.verify()`` recomputes the SHA-256 hash
chain and confirms each record still matches its hash and links to the one before, and
``trail.export_jsonl()`` produces a portable record of the mitigation actions.
The chain is tamper-*evident*, not tamper-proof. The notebook's trail is
unsigned, and an unsigned trail is a hash chain held in memory: ``verify()``
*detects* an edit, reorder or deletion in the middle, but not records dropped
from the end, and not a chain rebuilt around an edit by someone who can write
the log. Pin ``trail.head`` externally and pass it to
``verify(expected_head=...)`` to catch truncation.
``AuditTrail(signer=Ed25519Signer.from_pem(...))`` (needs the ``audit`` extra)
signs each record's hash, so a chain rebuilt without the private key fails
``verify_jsonl(exported, keys={key_id: public_pem}, expected_head=...)``, which
an auditor can run with only the export, the public keys and the pinned head
hash: no access to the running agent or its in-memory state, just the
locus-sdk package with the ``audit`` extra. The signature is made when the
record is written, so it cannot show the payload was right.
Persist the JSONL and pin the head hash externally before relying on it for
SOC 2 / ISO 27001 change-management evidence, postmortem records, or audit hold.

AI agents making remediation decisions without any audit log are a
liability — if the agent is later questioned, *"the AI decided"* is not a
defensible answer; a tamper-evident trail at least shows what was done and flags
after-the-fact edits. The notebook's own ``ops_toolset(allow_mitigation=True)``
helper shows the pattern: write-capable tools (``rollback_deploy``) are opt-in,
and without the flag the agent has only read tools. ``GovernedAgent``,
``AuditHook`` and ``AuditTrail`` are the SDK pieces (``locus.control``); the
runbook and toolset are yours to write.

Run it:

    python examples/notebook_76_ir_audit_trail.py

## Source

````python
--8<-- "examples/notebook_76_ir_audit_trail.py"
````
