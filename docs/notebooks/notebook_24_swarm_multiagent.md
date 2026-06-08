# Swarm Multi-Agent

A swarm is a pool of agents pulling tasks from a shared queue. No
supervisor decides who does what — each task finds the worker whose
declared capabilities fit best, and the swarm runs them in parallel
where it can.

This notebook covers:

- `create_swarm_agent` — agents advertise free-form capability tags.
- `SwarmTask` — `required_tags` are a hard filter; `preferred_tags`
  boost score. Tagless tasks fall back to substring matching against
  the description.
- `SharedContext` — the blackboard agents use to leave findings,
  messages, and recorded results for each other.
- `Swarm.execute(initial_task, decompose_tasks=True)` — break a
  high-level brief into capability-matched subtasks and run them.
- Three common shapes: specialist team, redundant team, pipeline.

## Prerequisites

- Notebook 08 (Agent basics).
- Notebook 27 (Orchestrator pattern) if you want the supervised
  counterpoint to a swarm.

## Run

```bash
python examples/notebook_24_swarm_multiagent.py
```

The default provider is the bundled mock model. Set
`TULIP_MODEL_PROVIDER` (openai / anthropic) and credentials to use a
live model. Set
`TULIP_MODEL_PROVIDER=mock` for offline runs.

## Source

```python
--8<-- "examples/notebook_24_swarm_multiagent.py"
```
