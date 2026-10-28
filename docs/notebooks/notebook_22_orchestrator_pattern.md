# Orchestrator Pattern

A privacy officer routes a data-subject request to a chosen set of
specialist agents, runs them in parallel under a semaphore, then
correlates their outputs into a single privacy assessment. Compared with
a swarm, the decision of who investigates what is centralised here
instead of emerging from capability tags. The routing step is one named
place where you can see, log and review which specialist was asked what
— `OrchestratorResult.decisions` carries the `RoutingDecision` trail —
rather than reconstructing it from a self-organising pool. It does not
itself restrict what a specialist can reach: the tools you attach to
each `Specialist` set that. Gating a side-effecting call is a separate
step you add by wrapping it in `admit()`; neither the orchestrator nor
the specialist calls it for you. If the planner does not return a
parseable fenced JSON block,
routing falls back to the specialist IDs named in its reply, or to every
registered specialist if it names none.

This notebook covers:

- `Specialist` — domain-focused agent with tools, system prompt, and a
  confidence threshold. Locus ships pre-built ones for logs, metrics,
  traces, and code — useful evidence sources when tracing where personal
  data flows.
- `Orchestrator` — registers specialists, emits `RoutingDecision`
  objects, and runs the chosen ones concurrently behind
  `max_parallel_specialists` (an `asyncio.Semaphore`).
- `RoutingDecision` — the typed object the planner returns: which
  specialists, which sub-task per specialist, and the reasoning.
- `OrchestratorResult` — each specialist's output, the decisions
  trail, and a correlated summary.

## Prerequisites

- Agent basics.
- The swarm multi-agent notebook for the unsupervised counterpoint.

## Run

```bash
python examples/notebook_22_orchestrator_pattern.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

## Source

````python
--8<-- "examples/notebook_22_orchestrator_pattern.py:47:"
````
