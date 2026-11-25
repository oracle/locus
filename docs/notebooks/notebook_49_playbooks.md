# Playbooks

A playbook is a typed, ordered sequence of steps, each declaring the tools it expects. Bound to an agent, it holds the agent to that order: while a step is current, only the tools it expects may run (with the default `allow_extra_tools=False`), and the plan moves to the next step once they have all been called.

That is what a General Data Protection Regulation (GDPR) data subject access request (DSAR) procedure, a right-to-erasure workflow, or any privacy procedure that must be auditable afterwards asks of an agent: no step skipped, every tool call attributable to a step.

Parts 1–8 are plain Python over the playbook objects (Part 3 asks the
model for one next-step suggestion from the plan's state); Part 9 binds
a three-step triage playbook to a live agent and prints
`[model call: X.XXs · prompt→completion tokens · iters=N]` for the
round-trip.

What this notebook covers:

- `PlaybookStep` — id, description, expected tools, hints, validation
  rules
- `Playbook` — the ordered steps, with version, tags,
  `allow_extra_tools` and the authoring-only `strict_sequence` and
  `max_iterations`
- `PlaybookPlan` and `StepExecution` — runtime tracking, progress and
  status (`PENDING` / `IN_PROGRESS` / `COMPLETED` / `SKIPPED` /
  `FAILED`)
- `Agent(playbook=...)` — a playbook bound to an agent that calls real
  tools

## Prerequisites

- An OCI profile in `~/.oci/config` (picked up automatically), or
  `LOCUS_MODEL_PROVIDER` set to `openai` or `anthropic` with the
  matching key, or `mock` for an offline run.

Part 9 is where the enforcement drawn below happens. The triage playbook has three steps, `gather`, `classify` and `notify`, each expecting one tool. When the model proposes a tool call, the `PlaybookEnforcerHook` checks it against the plan's current step:

- an in-sequence call runs, is recorded in that step's `StepExecution`, and completes the step once its expected tools have all been seen;
- an out-of-sequence call is cancelled before the tool body runs, and the model reads the `PlaybookEnforcer blocked:` message, which names the current step's expected tools and hints, as the tool's result.

The drawing labels the steps with the refund playbook from [the playbooks concept page](../concepts/playbooks.md); the path is the same:

{{ locus_diagram playbook-enforcement }}

## Key lines

Each `PlaybookStep` declares the tools it expects, hints for the model,
whether it is required, and an optional `max_tool_calls` cap:

{{ locus_source examples/notebook_49_playbooks.py::part1_steps }}

The `Playbook` orders the steps; the enforcer walks them through the plan's current-step pointer, which advances once the current step's expected tools have all been called or its `max_tool_calls` is reached. `strict_sequence` and `max_iterations` are carried for authoring and serialisation and are not read by the enforcer.

`allow_extra_tools=True` relaxes enforcement: a call goes through even when its tool is not in the current step's `expected_tools`, so it is not the setting for a compliance-grade procedure. This playbook is never bound to an agent here; the Part 9 triage playbook keeps the default:

{{ locus_source examples/notebook_49_playbooks.py::part2_playbook }}

`Agent(playbook=...)` installs the `PlaybookEnforcerHook`. The triage
agent is given three tools and a three-step playbook; `arun` runs it
under that enforcement, and `tool_executions` lists every call the model
made, in order, a cancelled call included with the blocked message as
its result:

{{ locus_source examples/notebook_49_playbooks.py::part9_live_agent }}

## Run it

```bash
python examples/notebook_49_playbooks.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

To run against the mock model regardless of the local OCI configuration:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_49_playbooks.py
```

## See also

- [Concepts: Playbooks](../concepts/playbooks.md) — loading from YAML or
  JSON, strict versus lenient enforcement, steps that use skills,
  evidence probes
- [Notebook 48 — Skills](notebook_48_skills.md) — the capability bundles
  a step can name through `PlaybookStep.uses`

## Source

??? note "Full source — `examples/notebook_49_playbooks.py`"

    ````python
    --8<-- "examples/notebook_49_playbooks.py"
    ````
