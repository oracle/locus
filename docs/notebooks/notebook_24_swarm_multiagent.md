# Swarm Multi-Agent

A swarm is a pool of agents pulling tasks from a shared queue. No
supervisor decides who does what — each agent claims the next queued
task its declared capabilities make it eligible for. `Swarm.execute`
runs the agents one after another, not concurrently, so each one sees
the results earlier agents recorded on the shared context. Here the
swarm is an outage war room: mitigation, diagnostics and communications
responders claiming work from a shared board during a production
incident.

What this notebook covers:

- `create_swarm_agent` — agents advertise free-form capability tags.
- `SwarmTask` — `required_tags` as a hard filter, `preferred_tags` as an
  advisory fit score.
- `SharedContext` — the blackboard agents use to leave findings,
  messages and recorded results for each other.
- `Swarm.execute(initial_task, decompose_tasks=True)` — break a
  high-level brief into untagged subtasks that any eligible agent can
  pick up, and run them.
- Three common shapes: specialist team, redundant team, pipeline.

## Prerequisites

- [Agent basics](notebook_02_basic_agent.md).
- [The orchestrator pattern](notebook_22_orchestrator_pattern.md) if you
  want the supervised counterpoint to a swarm.

The drawing follows one swarm run of the kind `example_full_swarm` performs; its Scout, Analyst and Writer are generic peers, and this notebook's Diagnostics, Mitigation and Comms responders take their place.

Tasks enter one priority-ordered queue, seeded with `add_task` or added by `execute(initial_task)`, which also asks the swarm's model to split the brief into subtasks. Each peer claims the first pending task its capabilities match, reads a summary of the `SharedContext` in its prompt and writes its findings back to it.

Once no task is pending — or, when a task no peer can claim remains, after `max_iterations` rounds — the run returns a `SwarmResult` with the completed and failed tasks and a summary the swarm's model writes from the shared context:

{{ locus_diagram swarm }}

## Key lines

`SharedContext` is the blackboard: findings and blackboard messages are
posted by key with the posting agent's `agent_id`, task results by
`task_id`, and `get_summary` renders the view every agent reads in its
prompt before it works a task:

{{ locus_source examples/notebook_24_swarm_multiagent.py::example_shared_context }}

Eligibility and fit are separate questions. `can_handle` treats
`required_tags` as set membership against the agent's capabilities, and
it alone decides who may claim a task: `Swarm.execute` gives each pending
task to the first agent in the swarm's list that is eligible for it, so
`priority_for_task` — the 0-1 score that `preferred_tags` raise — is
advisory, there for you to rank candidates yourself. The last task
carries no tags at all, so eligibility falls back to matching capability
names against the description:

{{ locus_source examples/notebook_24_swarm_multiagent.py::example_capability_matching }}

`create_swarm` assembles the three tagged responders around one model, and a single `Swarm.execute` call with `decompose_tasks=True` queues the incident brief, asks the model to split it into untagged subtasks and lets the responders claim and work them in turn; it returns a `SwarmResult` with the completed and failed tasks and a summary.

Because the brief and its subtasks carry no tags, a responder is eligible only when one of its capability names, such as `investigate` or `write`, appears in the brief or in a subtask the model wrote. A task no responder is eligible for stays pending, appears in neither list and does not affect `success`:

{{ locus_source examples/notebook_24_swarm_multiagent.py::example_full_swarm }}

## Run it

```bash
python examples/notebook_24_swarm_multiagent.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

## See also

- [Swarm](../concepts/multi-agent/swarm.md) — the shape, when to use it
  and its [termination rules](../concepts/multi-agent/swarm.md#termination).
- [Multi-agent overview](../concepts/multi-agent.md#pick-a-shape) — pick a
  shape.
- [Notebook 22 — Orchestrator](notebook_22_orchestrator_pattern.md) — the
  supervised counterpoint, where one coordinator picks which specialists
  run.

## Source

??? note "Full source — `examples/notebook_24_swarm_multiagent.py`"

    ````python
    --8<-- "examples/notebook_24_swarm_multiagent.py"
    ````
