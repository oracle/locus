# Specialist agents

The orchestrator-pattern notebook introduced the Specialist as the worker an
orchestrator hands tasks to. This notebook turns to the Specialist itself, using
RIGHTSIZER — a cloud cost-and-capacity specialist — as the running case: how to
narrow a model's failure surface with a focused system prompt, a hand-picked
tool set, optional playbooks, and a confidence threshold.

RIGHTSIZER fingerprints a compute workload for rightsizing: it maps observable
utilization telemetry (CPU p50, memory mean, IOPS, network throughput) to a
`(shape, action)` verdict. [Notebook 21](notebook_21_supervisor_critic_loop.md)
and [Notebook 55](notebook_55_gsar_typed_grounding.md) show how such a finding
is gated by [GSAR (typed grounding)](../concepts/gsar.md) before it ships; here
the subject is the Specialist API itself.

What this notebook covers:

- `Specialist` — an agent role: a Pydantic model that pairs role metadata
  (`specialist_type`, `description`) with a system prompt, a tool list, an
  iteration budget and a `confidence_threshold`.
- `Playbook` + `PlaybookStep` — encode a procedure as typed data: preconditions,
  ordered steps with required tools and expected outputs, optional per-step
  failure handling, and a success criterion.
- `specialist.select_playbook(task)` — picks the playbook whose name and
  description share the most words with the task, or `None` when none match.
- Pre-built helpers (`create_log_analyst`, `create_metrics_analyst`,
  `create_trace_analyst`, `create_code_analyst`) for common observability
  domains.

## Prerequisites

- [Agent basics](notebook_02_basic_agent.md).
- [The orchestrator pattern](notebook_22_orchestrator_pattern.md) — Specialists
  are the workers it routes to.

The drawing shows where a Specialist sits in the orchestrator run from the
previous notebook: the orchestrator's model picks which registered specialists
run and gives each a subtask, the picked ones run in parallel under their own
system prompts and never talk to each other, and their results go back to be
correlated and summarized into one answer; this notebook is about the inside of
one of those parallel boxes, which for RIGHTSIZER means its system prompt, tool
list, playbooks and confidence threshold:

{{ locus_diagram orchestrator }}

## Key lines

`Specialist` is a Pydantic model, not an `Agent` subclass: read the role
metadata — `specialist_type`, and the `description` an orchestrator's model sees
when deciding whether this specialist runs — the focused system prompt and
three-tool list that limit what RIGHTSIZER can do, and the two numbers,
`max_iterations` and `confidence_threshold`, that `execute` records on the
result's `AgentState` but neither loops on nor gates with, since it makes a
single `model.complete` call; the closing `execute` is a one-call smoke test
of the assembled specialist:

{{ locus_source examples/notebook_23_specialist_agents.py::example_anatomy }}

A `Playbook` is the team's rightsizing runbook as typed data: preconditions,
ordered `PlaybookStep`s that each name the tool they require and the output they
should produce (the classify step also says what to do on failure), and a
success criterion; `to_prompt()` renders it as the text block `execute` appends
to the system prompt when `select_playbook` picks this playbook for the task:

{{ locus_source examples/notebook_23_specialist_agents.py::example_playbooks }}

The call that runs it: `execute` takes the task and a `context` dictionary
(formatted as a user message ahead of the task), selects a playbook itself,
makes one model call with the tool schemas (a tool call the model returns is
recorded on the state, not run), and returns a `SpecialistResult` whose
`confidence` comes from the built-in estimator — a 0.5 baseline moved by 0.1
for each hedging or certainty marker found in the reply, a rough proxy to
replace with a domain scorer in production — alongside `success`,
`duration_ms`, `output` and any `error`; the comparison against
`confidence_threshold` is the example's own print, since the SDK records the
threshold on the state but does not gate on it:

{{ locus_source examples/notebook_23_specialist_agents.py::example_execute }}

## Run it

```bash
python examples/notebook_23_specialist_agents.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

Parts 2 to 4 (the tool list, the playbook and its selection) are deterministic
and make no model call; Parts 1 and 5 to 8 make one each. To stay offline
regardless of the OCI profile on the machine, force the mock model. With tools
bound, the mock answers with a synthesized tool call and no text, and
`Specialist.execute` is single-turn, so Parts 1, 5 and 7 print no output line;
the estimator finds no hedging or certainty markers in the empty reply, so the
Part 5 confidence sits at its 50% baseline against RIGHTSIZER's 85% threshold.
Parts 6 and 8, whose specialists carry no tools, print the mock's fixed prose
reply:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_23_specialist_agents.py
```

## Source

??? note "Full source — `examples/notebook_23_specialist_agents.py`"

    ````python
    --8<-- "examples/notebook_23_specialist_agents.py"
    ````
