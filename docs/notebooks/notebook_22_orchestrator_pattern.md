# Orchestrator Pattern

A privacy officer routes a data-subject request to a chosen set of
specialist agents, runs them in parallel under a semaphore, then
correlates their outputs into a single privacy assessment. Compared with
a [swarm](notebook_24_swarm_multiagent.md), the decision of who
investigates what is centralized here instead of emerging from
capability tags.

That makes the routing step one named place where you can see, log and
review which specialist was asked what: `OrchestratorResult.decisions`
carries the `RoutingDecision` trail, rather than leaving you to
reconstruct it from a self-organizing pool.

Routing does not itself restrict what a specialist can reach; the tools
attached to each `Specialist` set what its model may ask for. Gating a
side-effecting call is a separate step you add by wrapping the call in
[`admit()`](../concepts/security-context.md#admission-control-the-enforcement-point);
neither the orchestrator nor the specialist calls it for you.

What this notebook covers:

- `Specialist` — a domain-focused agent with tools, a system prompt and a
  confidence threshold. The SDK ships pre-built ones for logs, metrics,
  traces and code (`create_log_analyst`, `create_metrics_analyst`,
  `create_trace_analyst`, `create_code_analyst`), useful evidence
  sources when tracing where personal data flows.
- `Orchestrator` — registers specialists, records one `RoutingDecision`
  per step (invoke, correlate, summarize) on the result's `decisions`
  trail, and runs the chosen specialists concurrently behind
  `max_parallel_specialists` (an `asyncio.Semaphore`).
- `RoutingDecision` — the typed object the routing step produces: which
  specialists, which sub-task per specialist (under
  `context["subtasks"]`) and the reasoning.
- `OrchestratorResult` — each specialist's `SpecialistResult`, the
  decisions trail and the correlated summary.

## Prerequisites

- [Basic agent](notebook_02_basic_agent.md).
- [Swarm multi-agent](notebook_24_swarm_multiagent.md), for the
  unsupervised counterpoint.

The drawing shows the generic shape of one `orchestrator.execute(...)`
call; its Research, Data and not-picked boxes stand for whichever
specialists are registered, here the log analyst, the metrics analyst
and the data-inventory specialist. The data-subject request reaches the
privacy officer, whose model picks which of the registered specialists
run and gives each a sub-task; the picked specialists run in parallel,
at most `max_parallel_specialists` at a time, without seeing one
another's work; and their results return to the officer's model, which
correlates and summarizes them into one `OrchestratorResult`, with a
failed specialist recorded as an error while the rest carry on:

{{ locus_diagram orchestrator }}

## Key lines

What the data-inventory `Specialist` can ask for is set entirely by the
two tools attached to it (`Specialist.execute` passes their schemas to
the model with the task, and routing neither widens nor narrows the
list); the first reports which stores in the data catalog hold a
subject's records:

{{ locus_source examples/notebook_22_orchestrator_pattern.py::main.query_data_catalog }}

The second reports the subject's current consent status, which the
specialist's system prompt tells it to read against where the data is
stored before recommending minimization, restriction or erasure:

{{ locus_source examples/notebook_22_orchestrator_pattern.py::main.check_consent_ledger }}

In this release `Specialist.execute` is a single model turn: a tool call
the model returns is kept on the result's `state` but is not run, so
these two functions mark the boundary of what the specialist may request
rather than code that executes during the run (see
[Orchestrator + Specialists](../concepts/multi-agent/orchestrator.md)).
The `Specialist` that carries them is built in Part 2 of `main()` in the
full source; Part 4 registers it beside the pre-built log and metrics
analysts with `create_orchestrator(...)`, and Part 7 makes the
`orchestrator.execute(...)` call that routes the request.

## Run it

```bash
python examples/notebook_22_orchestrator_pattern.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

Routing reads a fenced JSON block out of the model's reply; when there is
none, or it does not parse, routing falls back to the specialist IDs the
reply names, or to every registered specialist if it names none. The mock
model's reply carries no JSON block, so offline all three registered
specialists run in Part 7. The mock answers a tool-bearing request with a
tool call and no text, and `Specialist.execute` does not run it, so the
inventory specialist's reply is empty: Part 3 prints no output line, and in
Part 7 it shows no output preview even after the orchestrator's single
retry on an empty reply. To stay offline regardless of what `~/.oci/config`
holds:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_22_orchestrator_pattern.py
```

## See also

- [Orchestrator + Specialists](../concepts/multi-agent/orchestrator.md)
  — the pattern in full: single-pass routing, what runs in parallel, the
  empty-reply retry and confidence thresholds.
- [Specialist agents](notebook_23_specialist_agents.md) — notebook 23,
  the inside of one specialist: system prompt, tools, playbooks and
  confidence threshold.

## Source

??? note "Full source — `examples/notebook_22_orchestrator_pattern.py`"

    ````python
    --8<-- "examples/notebook_22_orchestrator_pattern.py"
    ````
