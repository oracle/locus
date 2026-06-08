# Playbooks

A playbook is a typed, ordered sequence of steps with declared
`expected_tools`. Wire it into an agent and the agent is constrained
to walk the steps in order, calling only the tools each step allows.
Useful for incident response, deployments, and any procedure where
you want auditability over agent freedom.

- `PlaybookStep` — id, description, expected tools, hints, validation
  rules.
- `Playbook` — a collection of steps with ordering, max-iteration, and
  tagging.
- `PlaybookPlan` and `StepExecution` — runtime tracking, progress, and
  status (`PENDING` / `IN_PROGRESS` / `COMPLETED` / `FAILED` /
  `SKIPPED`).
- `Agent(playbook=...)` — bind a playbook to an agent and watch it
  execute against real tools.

Each part fires a real model call so you can see live behaviour next
to the structured execution mechanics — every section prints
`[model call: X.XXs · prompt→completion tokens]`.

## Run it

The bundled mock model is the default; set `TULIP_MODEL_PROVIDER` for a live provider:

```bash
TULIP_MODEL_ID=openai.gpt-4.1 python examples/notebook_46_playbooks.py
```

Offline:

```bash
TULIP_MODEL_PROVIDER=mock python examples/notebook_46_playbooks.py
```

## Prerequisites

- An OpenAI or Anthropic API key, or `TULIP_MODEL_PROVIDER` set to
  `openai` / `anthropic` / `mock`.

## Source

```python
--8<-- "examples/notebook_46_playbooks.py"
```
