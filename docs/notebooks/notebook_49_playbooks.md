# Playbooks

A playbook is a typed, ordered sequence of steps with declared
`expected_tools`. Wire it into an agent and the agent is constrained
to walk the steps in order, calling only the tools each step allows.
Useful for GDPR data-subject requests, right-to-erasure workflows, and
any privacy procedure where you want auditability over agent freedom.

- `PlaybookStep` — id, description, expected tools, hints, validation
  rules.
- `Playbook` — a collection of steps with ordering, max-iteration, and
  tagging.
- `PlaybookPlan` and `StepExecution` — runtime tracking, progress, and
  status (`PENDING` / `IN_PROGRESS` / `COMPLETED` / `FAILED` /
  `SKIPPED`).
- `Agent(playbook=...)` — bind a playbook to an agent and watch it
  execute against real tools.

Parts 1–8 are deterministic Python over the playbook objects; Part 9
binds the playbook to a live agent and prints
`[model call: X.XXs · prompt→completion tokens]` for the round-trip.

## Run it

The bundled mock model is the default; set `LOCUS_MODEL_PROVIDER` for a live provider:

```bash
LOCUS_MODEL_PROVIDER=openai LOCUS_MODEL_ID=gpt-4o python examples/notebook_49_playbooks.py
```

Offline:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_49_playbooks.py
```

## Prerequisites

- An OCI profile in `~/.oci/config` (picked up automatically), or
  `LOCUS_MODEL_PROVIDER` set to `openai` / `anthropic` / `mock` with the
  matching key.

## Source

````python
--8<-- "examples/notebook_49_playbooks.py"
````
