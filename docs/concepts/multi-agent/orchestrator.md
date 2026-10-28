# Orchestrator + Specialists

One coordinator picks which specialist handles each sub-task. The
specialists never talk to each other — only to the orchestrator.
Think *project manager + team*.

{{ locus_diagram orchestrator }}

## What it is

The coordinator uses its `model` to pick which specialists handle a
task and to write each one a subtask, then runs the selected
specialists, correlates their results and summarizes them into one
answer. The selected specialists run in parallel (bounded by
`max_parallel_specialists`). It is a single pass: routing happens once
and nothing loops back to it.

Each specialist answers in a single model turn under its own system
prompt. Its `tools` are sent to the model as tool schemas, but the
specialist does not execute the tool calls that come back.

Each `Specialist` is configured on its own. Its fields:

- a `name` — how the coordinator's prompts label it (the router picks
  specialists by their generated `id`)
- a `specialist_type` — a short type tag (e.g. `"research"`)
- a `description` — what the specialist is good at (the coordinator reads this)
- a `system_prompt` — the specialist's own instructions
- its own `tools` and `model`
- an optional `confidence_threshold` (default `0.85`) — a bar you can
  compare the result's `confidence` against; the orchestrator does not
  check it

## When to use it

- ✅ The work splits cleanly into **expert domains** (Research, Data,
  Writing; or Triage, Forensics, Containment).
- ✅ You want **one place to attribute decisions to** — the coordinator.
- ✅ Specialists need their **own playbooks, tools, or models** (a
  cheap model for triage, a strong one for compliance).
- ✅ **Auditability** matters — the result's `decisions` record which
  specialists the router picked and its reasoning. A specialist does
  not execute the tool calls it returns, so run consequential actions
  in your own code through [`admit()`](../security-context.md) so every
  decision lands on the hash-chained `AuditTrail`.

## When NOT to use it

- ❌ The flow is **linear, not delegated** — use [Composition](composition.md).
- ❌ No central coordinator should exist; agents should
  **self-organise** — use [Swarm](swarm.md).
- ❌ The conversation itself moves between roles — use [Handoff](handoff.md).

## Code

```python
from locus.models import get_model
from locus.multiagent import Specialist, create_orchestrator

model = get_model("{{ locus_example_model }}")

research = Specialist(
    name="research",
    specialist_type="research",
    description="Searches sources. Collects evidence. Summarises what it found.",
    system_prompt="You are the Research specialist.",
    tools=[web_search, fetch_page],
)

data = Specialist(
    name="data",
    specialist_type="data",
    description="Queries the warehouse. Reconstructs timelines. Quantifies impact.",
    system_prompt="You are the Data specialist.",
    tools=[query_metrics, query_warehouse],
)

orchestrator = create_orchestrator(
    name="coordinator",
    specialists=[research, data],
).with_model(model)                        # coordinator's model, copied into every specialist
orchestrator.system_prompt = (
    "You are the coordinator. Delegate source-gathering to research "
    "and metrics to data."
)

result = await orchestrator.execute(
    "Why did checkout conversion drop last week? Investigate and draft a report.",
)
```

`create_orchestrator` registers the specialists but does not hand its
`model` to them, and a `Specialist` with no model returns an error
result instead of running. `.with_model(model)` sets the coordinator's
model and copies it into every registered specialist, replacing
any model a specialist already carried. To give specialists different
models, set `model=` on each `Specialist` and pass the coordinator's own
to `create_orchestrator(model=...)` instead. The coordinator's
`system_prompt` is its instructions for routing, correlating and
summarizing.
`execute()` is async — `await` it (or wrap in `asyncio.run`). It returns
an `OrchestratorResult`: `summary` is the coordinator's final answer,
and `specialist_results` holds each picked specialist's
`SpecialistResult`, keyed by specialist id. The same shape runs an
incident commander with triage, forensics, and containment specialists.
Routing happens once and can pick any registered specialist, so to hold
containment until the other two report, register only triage and
forensics on the first orchestrator, then run a second one holding
containment with their findings in its task.

## What runs in parallel

Every specialist the router picks runs **concurrently**, at most
`max_parallel_specialists` at a time. So when the router gives research
"pull the release notes for deploy-77" and data "pull last week's funnel
metrics", both specialists run at the same time. Neither sees the
other's work: each gets its own subtask, plus the original task when
the two differ. Their results come back to the coordinator together,
and its model correlates them and then summarizes. A specialist whose
reply comes back empty is retried once (an error result counts as
empty), and one that still fails returns an error result while the rest
carry on.

## Confidence thresholds

Each specialist carries a `confidence_threshold` (default `0.85`).
Every successful `SpecialistResult` reports a `confidence` score read
from the wording of the reply (words like "confirmed" raise it, hedges like
"might" lower it), so you can compare it against the threshold and
decide whether to trust the output or route the sub-task to another
expert. The orchestrator leaves that check to you:

```python
Specialist(
    name="data-quality",
    specialist_type="data_quality",
    description="Audits metric definitions and flags unreliable numbers.",
    system_prompt="You audit metrics before they are cited.",
    tools=[query_warehouse, profile_table],
    confidence_threshold=0.7,    # compare result.confidence against this
)
```

## Notebooks

- [`notebook_22_orchestrator_pattern.py`](https://github.com/oracle/locus/blob/main/examples/notebook_22_orchestrator_pattern.py)
  — router over three specialists run in parallel, results correlated and summarized.
- [`notebook_23_specialist_agents.py`](https://github.com/oracle/locus/blob/main/examples/notebook_23_specialist_agents.py)
  — confidence thresholds and per-specialist playbooks.
- [`notebook_78_concession_approval.py`](https://github.com/oracle/locus/blob/main/examples/notebook_78_concession_approval.py)
  — customer-support concession approval with risk-tiered approval gates
  and a typed `ConcessionDecision` artifact (built on a `StateGraph`, not
  the orchestrator).

## Source

[`multiagent/orchestrator.py`](https://github.com/oracle/locus/blob/main/src/locus/multiagent/orchestrator.py)
— `Orchestrator`, `create_orchestrator`;
[`multiagent/specialist.py`](https://github.com/oracle/locus/blob/main/src/locus/multiagent/specialist.py)
— `Specialist`, `SpecialistResult`.

## See also

- [Multi-agent overview](../multi-agent.md) — pick a shape.
- [Handoff](handoff.md) — when the conversation itself transfers, not just sub-tasks.
- [Playbooks](../playbooks.md) — declarative step plans an agent's run is checked against. A
  `Specialist` takes its own, simpler `Playbook` from `locus.multiagent`,
  picked by keyword match against the task and added to its system prompt.
