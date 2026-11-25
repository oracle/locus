# Incident response with a tamper-evident audit chain

A production outage pages the on-call engineer at 2 AM. A `GovernedAgent` runs the four-phase Site Reliability Engineering (SRE) runbook the notebook defines, `sre_incident_runbook` (detect, triage, mitigate, recover).

Every tool call — metrics query, log tail, deploy rollback — is recorded on a tamper-evident `AuditTrail` by the `AuditHook` that `governed_agent` attaches when its profile has `audit=True`, and the runner writes its own `incident-start`, `phase-<id>` and `incident-end` markers to the same trail. When the incident closes, `trail.verify()` recomputes the SHA-256 hash chain and `trail.export_jsonl()` produces a portable JSON Lines (JSONL) record of the mitigation.

Without an audit log there is no record of why a remediation was taken; a tamper-evident trail shows what was done and flags after-the-fact edits. The chain is tamper-*evident*, not tamper-proof.

The notebook's trail is unsigned and held in memory, so `verify()` called with no arguments catches an edit, a reorder or a deletion from the middle of the chain, but not records dropped from the end, and not a chain rebuilt around an edit by someone who can write the log. Pin `trail.head` somewhere the agent cannot reach and pass it back as `verify(expected_head=...)` to catch truncation.

Signing closes the rebuild gap. Give the trail a signer with `AuditTrail(signer=Ed25519Signer.from_pem(...))`, which needs the `audit` extra (`pip install "locus-sdk[audit]"`), and every record's hash is signed with Ed25519.

A chain rebuilt without the private key then fails `verify_jsonl(exported, keys={key_id: public_pem}, expected_head=...)`, a check an auditor can run with only the export, the public keys and the pinned head: no access to the running agent, just the `locus-sdk` package with the `audit` extra. The signature attests that a record was written as shown, not that its contents were correct.

Persist the JSONL and pin the head hash externally before relying on the trail as SOC 2 (System and Organization Controls) or ISO 27001 change-management evidence, as a postmortem record, or under an audit hold.

What this notebook covers:

- Building the agent with `governed_agent(..., profile=..., audit_trail=trail)`
  and a `GovernanceProfile` that keeps grounding on, turns guardrails off for a
  brief that arrives from an internal alerting system, and sets `audit=True` so
  an `AuditHook` writes to the caller's trail.
- What the hook records: `agent.invocation.start` and `agent.invocation.end`
  around each `arun()`, `agent.tool.before` and `agent.tool.after` around each
  tool call.
- Adding the runner's own records with `trail.record(event_type, payload)`:
  `incident-start`, one `phase-<id>` per runbook phase and `incident-end`.
- Opting in to the write-capable `rollback_deploy` with
  `ops_toolset(allow_mitigation=True)`; without the flag the agent has only
  the read-only telemetry tools.
- Checking the chain with `trail.verify()`, reading `trail.head`, and
  exporting one record per line with `trail.export_jsonl()`.

In the trust chain below, a result passes through
[GSAR (typed grounding)](../concepts/gsar.md), verification and the admission
gate `admit()`, and the `AuditTrail` at the end records each decision, allowed
or not; this notebook stays on that last link, attaching one trail to a
governed agent so every tool call made during the incident is recorded next to
the incident markers the runner writes itself:

{{ locus_diagram trust-chain }}

## Key lines

`ops_toolset` hands the agent the read-only telemetry tools by default and
adds the write-capable `rollback_deploy` only when `allow_mitigation=True` is
passed:

{{ locus_source examples/notebook_76_ir_audit_trail.py::ops_toolset }}

`run_incident_response` creates the trail and builds the governed agent with
`GovernanceProfile(grounding=True, guardrails=False, audit=True)`, where
`audit=True` makes `governed_agent` attach an `AuditHook` to that trail; it
then records `incident-start`, runs the four runbook phases with a `phase-<id>`
record after each, records `incident-end`, and closes with `trail.verify()`
and `trail.export_jsonl()`:

{{ locus_source examples/notebook_76_ir_audit_trail.py::run_incident_response }}

## Run it

```bash
python examples/notebook_76_ir_audit_trail.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

Under the mock model each phase gets a fixed sample answer after one mock tool
call, so the trail still fills with invocation, tool and phase records, and the
integrity check and the export run as they would against a live model.

## See also

- [Control API — The record](../api/control.md#the-record) — `AuditTrail`,
  what `verify()` catches, `expected_head`, signing the trail and
  `verify_jsonl()`.
- [Control API — Governed agents](../api/control.md#governed-agents) —
  `governed_agent`, `GovernedAgent` and `GovernanceProfile`.
- [Control-layer architecture](../concepts/control-layer.md) — what the audit
  layer guarantees and where that guarantee stops.
- [Notebook 72 — Infra deploy gate](notebook_72_infra_deploy_gate.md) — the
  admission gate drawn in the diagram, recording each decision on a trail.

## Source

??? note "Full source — `examples/notebook_76_ir_audit_trail.py`"

    ````python
    --8<-- "examples/notebook_76_ir_audit_trail.py"
    ````
