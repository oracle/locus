# Swarm Multi-Agent

A swarm is a pool of agents pulling tasks from a shared queue. No
supervisor decides who does what — each agent claims the next queued
task its declared capabilities make it eligible for. `Swarm.execute`
runs the agents one after another, not concurrently, so each one sees
the results earlier agents recorded on the shared context. Here the
swarm is an outage war room: mitigation, diagnostics, and comms
responders claiming work off the same board during a production
incident.

This notebook covers:

- `create_swarm_agent` — agents advertise free-form capability tags.
- `SwarmTask` — `required_tags` are a hard filter; `preferred_tags`
  feed `SwarmAgent.priority_for_task`, an advisory 0-1 fit score you
  can read to rank candidates yourself — claiming is first-eligible, so
  preferred tags do not change who gets the task. Tasks with no
  `required_tags` (including ones that set only `preferred_tags`) fall
  back to substring matching of the agent's capabilities against the
  description; an agent with no capabilities (the `create_swarm_agent`
  default) can claim any such task.
- `SharedContext` — the blackboard agents use to leave findings,
  messages, and recorded results for each other.
- `Swarm.execute(initial_task, decompose_tasks=True)` — break a
  high-level brief into untagged subtasks that any eligible agent can
  pick up, and run them.
- Three common shapes: specialist team, redundant team, pipeline.

## Prerequisites

- Agent basics.
- The orchestrator-pattern notebook if you want the supervised
  counterpoint to a swarm.

## Run

```bash
python examples/notebook_24_swarm_multiagent.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

## Source

The example's docstring below predates this page: claiming is
first-eligible, `preferred_tags` only affect `priority_for_task`, and
agents run one after another rather than in parallel.

````python
--8<-- "examples/notebook_24_swarm_multiagent.py"
````
