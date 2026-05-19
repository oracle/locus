# Orchestrator Pattern

An orchestrator routes a task to a chosen set of specialist agents, runs
them in parallel under a semaphore, then correlates their outputs into a
single summary. Compared with a swarm (Notebook 25), the decision of who
does what is centralised here instead of emerging from capability tags.

This notebook covers:

- `Specialist` — domain-focused agent with tools, system prompt, and a
  confidence threshold. Locus ships pre-built ones for logs, metrics,
  traces, and code.
- `Orchestrator` — registers specialists, emits `RoutingDecision`
  objects, and runs the chosen ones concurrently behind
  `max_parallel_specialists` (an `asyncio.Semaphore`).
- `RoutingDecision` — the typed object the planner returns: which
  specialists, which sub-task per specialist, and the reasoning.
- `OrchestrationResult` — each specialist's output, the decisions
  trail, and a correlated summary.

## Prerequisites

- Notebook 08 (Agent basics).
- Notebook 25 (Swarm) for the unsupervised counterpoint.

## Run

```bash
python examples/notebook_32_orchestrator_pattern.py
```

The default provider is OCI Generative AI. With `~/.oci/config`
present the agents talk to a live OCI model; canonical picks are
`openai.gpt-4.1` or `meta.llama-3.3-70b-instruct`. Set
`LOCUS_MODEL_PROVIDER=mock` for offline runs.

## Source

```python
--8<-- "examples/notebook_32_orchestrator_pattern.py"
```
