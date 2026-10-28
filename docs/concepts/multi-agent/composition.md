# Composition — pipelines

The composition primitives are for flows you can write as a regular
function: do A, then B, then C — with optional fan-out and merge.

{{ locus_diagram composition }}

## What it is

Three `BaseModel`-shaped pipeline classes, each wrapping one or more agents:

| Class | Shape |
|---|---|
| `SequentialPipeline(agents=[...])` | output of agent N feeds agent N+1 |
| `ParallelPipeline(agents=[...])` | one input fans out to all N agents; results merge |
| `LoopAgent(agent=..., max_loops=N)` | run one agent repeatedly until a condition holds or N is hit |

Each has an async `.run(task)` that calls each of its agents' `arun`
(or `run_sync`, for agents without `arun`) and returns one
`PipelineResult`. A pipeline has no `arun` or `run_sync` of its own, and
its result carries `final_output` rather than `.message`. So a pipeline is
not itself agent-shaped, and pipelines do not nest: a pipeline placed
inside another never runs, and the outer pipeline returns
`success=False`.

Inside an active `run_context`, the pipeline's `composition.*` stage,
fan-out and loop events go to the same event bus as its agents' events.

## When to use it

- ✅ The flow is **describable as a function** — "do A, then B, then C".
- ✅ Fan-out is **symmetric** — all branches do similar work on the same
  input (e.g., answer one question across web search, a data query, and
  docs retrieval).
- ✅ The flow is a **report chain** — research → draft → review, each
  step feeding the next.
- ✅ You need **revise-until-confidence** — wrap the report writer in a
  `LoopAgent` until the confidence bar clears.
- ✅ You don't need cycles, conditional branches, or per-node retry policies.

## When NOT to use it

- ❌ You need **cycles** that depend on state — use [StateGraph](graph.md).
- ❌ A central agent should **decide which expert runs** — use [Orchestrator](orchestrator.md).
- ❌ The branches need to **talk to each other** — use [Swarm](swarm.md).

## Code

```python
from locus.agent.composition import (
    SequentialPipeline, ParallelPipeline, LoopAgent,
)

# Sequential: research → draft → review → publish
pipeline = SequentialPipeline(agents=[
    research,
    draft,
    review,
    publish,
])

result = await pipeline.run("Why did checkout conversion drop last week?")
```

```python
# Parallel: answer one question across web search, a data query, and docs
parallel = ParallelPipeline(agents=[
    web_search_agent,
    data_query_agent,
    docs_retrieval_agent,
])
answers = await parallel.run(
    "Checkout conversion last week: what changed? which deploys? known issues?"
)

# Loop: revise the draft until the review clears, max 5 loops
revise = LoopAgent(agent=reviser_agent, condition=lambda out: "APPROVED" in out, max_loops=5)
final = await revise.run(initial_draft)
```

## Notebooks

- [`notebook_19_composition.py`](https://github.com/oracle/locus/blob/main/examples/notebook_19_composition.py)
  — `SequentialPipeline`, `ParallelPipeline`, `LoopAgent`.
- [`notebook_20_map_reduce_support_triage.py`](https://github.com/oracle/locus/blob/main/examples/notebook_20_map_reduce_support_triage.py)
  — same fan-out shape with `Send` inside a graph (use this when you
  need state-aware fan-out beyond what `ParallelPipeline` gives you).
- [`notebook_21_supervisor_critic_loop.py`](https://github.com/oracle/locus/blob/main/examples/notebook_21_supervisor_critic_loop.py)
  — `LoopAgent`-style refine-until-confidence written as a graph
  (the cycle version when you also need conditional edges).

## Source

[`agent/composition.py`](https://github.com/oracle/locus/blob/main/src/locus/agent/composition.py)
— `SequentialPipeline`, `ParallelPipeline`, `LoopAgent`.

## See also

- [Multi-agent overview](../multi-agent.md) — all eight coordination shapes, A2A included.
- [StateGraph](graph.md) — when you need cycles or conditional branches.
- [Functional](functional.md) — when you'd rather use plain asyncio.gather.
