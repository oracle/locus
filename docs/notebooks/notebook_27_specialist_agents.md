# Specialist Agents

Notebook 27 introduced the Specialist as the worker an orchestrator
hands tasks to. This notebook dives into the Specialist itself: how to
narrow a model's failure surface with a focused system prompt, a
hand-picked tool set, optional playbooks, and a confidence threshold.

This notebook covers:

- `Specialist` — a Tulip `Agent` with role metadata (`specialist_type`,
  `description`), a tool list, and a `confidence_threshold`.
- `Playbook` + `PlaybookStep` — encode a procedure: preconditions,
  ordered steps with required tools and expected outputs, plus failure
  handling.
- `specialist.select_playbook(task)` — picks one playbook from a pool
  by matching the task description.
- Pre-built helpers (`create_log_analyst`, `create_metrics_analyst`,
  `create_trace_analyst`, `create_code_analyst`) for common
  observability domains.

## Prerequisites

- Notebook 08 (Agent basics).
- Notebook 27 (Orchestrator) — Specialists are the workers it routes to.

## Run

```bash
python examples/notebook_27_specialist_agents.py
```

The default provider is the bundled mock model. Set `TULIP_MODEL_PROVIDER`
(openai / anthropic) and credentials to use a live model. Set
`TULIP_MODEL_PROVIDER=mock` for offline runs.

## Source

```python
--8<-- "examples/notebook_27_specialist_agents.py"
```
