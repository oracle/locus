# Specialist Agents

The orchestrator-pattern notebook introduced the Specialist as the worker an orchestrator
hands tasks to. This notebook dives into the Specialist itself, using RIGHTSIZER — a
cloud cost-and-capacity specialist — as the running case: how to
narrow a model's failure surface with a focused system prompt, a
hand-picked tool set, optional playbooks, and a confidence threshold.

RIGHTSIZER fingerprints a compute workload for rightsizing: it maps observable
utilization telemetry (CPU p50, memory mean, IOPS, network throughput) to a
`(family, size, action)` verdict. Notebooks 21 and 55 show how such a
verdict is gated by GSAR grounding before it ships; here the subject is the
Specialist API itself.

This notebook covers:

- `Specialist` — a Locus `Agent` with role metadata (`specialist_type`,
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

- Agent basics.
- The orchestrator-pattern notebook — Specialists are the workers it routes to.

## Run

```bash
python examples/notebook_23_specialist_agents.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

## Source

````python
--8<-- "examples/notebook_23_specialist_agents.py"
````
