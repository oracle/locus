# StateGraph

`StateGraph` is the **explicit-control-flow** shape: nodes do work,
edges decide what runs next, and state flows through. Cycles
(retry-until-confidence), conditional branches, and subgraphs are
all first-class.

{{ locus_diagram graph }}

## What it is

- **`StateGraph(state_schema=...)`** — a builder that takes an optional
  Pydantic `BaseModel` declaring the state fields and their reducers.
- **`add_node(name, fn)`** — a unit of work: a function, sync or
  async, that takes the state dict and returns the fields it updates
  as a dict. A `StateGraph` passed here runs as a subgraph. An `Agent`
  is not callable, so call `await agent.arun(...)` inside the function.
- **`add_edge(src, dst)`** — unconditional transition.
- **`add_conditional_edges(src, fn)`** — `fn(state) → next_node_name`
  picks the next node dynamically.
- **`compile()`** — returns the runnable graph. `run_sync(...)` /
  `invoke(...)` give you the final state dict; `await execute(...)`
  gives you the full `GraphResult` (`final_state`, `node_results`,
  `interrupt`).

State is carried and returned as a plain dict. `state_schema`
supplies the custom **reducers** that control how each node's output
merges into shared fields; it does not coerce state into the model.

## When to use it

- ✅ The flow has **cycles** — review-loop, retry, refine-until-confidence.
- ✅ You want **explicit, inspectable control flow** with a Mermaid
  diagram you can paste into a design doc.
- ✅ You need **per-node retry / cache policies** (different policies
  per node).
- ✅ **Subgraphs** — encapsulate a sub-workflow as a single node.

## When NOT to use it

- ❌ The flow is a **straight pipe** with maybe a fan-out → use
  [Composition](composition.md).
- ❌ You don't know the flow at design time; agents should
  **self-organise** → use [Swarm](swarm.md).
- ❌ A central agent should **decide which expert runs next** → use
  [Orchestrator](orchestrator.md).

## Code

```python
from locus.multiagent import StateGraph, START, END
from pydantic import BaseModel, Field

class ResearchState(BaseModel):
    prompt: str
    findings: list[str] = Field(default_factory=list)
    report: str = ""
    confidence: float = 0.0

graph = StateGraph(state_schema=ResearchState)

# Each node is a function that takes the state dict and returns the fields it updates.
# An Agent is not callable; await it inside the function.
async def plan(state):
    return {"findings": [(await planner.arun(state["prompt"])).message]}

graph.add_node("plan", plan)
graph.add_node("investigate", investigate)   # same shape, wrapping its own agent
graph.add_node("report", report)
graph.add_node("review", review)

graph.add_edge(START, "plan")
graph.add_edge("plan", "investigate")
graph.add_edge("investigate", "report")
graph.add_edge("report", "review")

graph.add_conditional_edges(
    "review",
    lambda state: END if state.get("confidence", 0.0) >= 0.85 else "investigate",
)

result = graph.compile().run_sync({"prompt": "Research why checkout conversion dropped last week."})
print(result["report"])
```

## Per-node policies

```python
from locus.multiagent.graph import RetryPolicy, CachePolicy

graph.add_node(
    "investigate",
    investigate,
    retry_policy=RetryPolicy(max_attempts=3, backoff_factor=2.0),
    cache_policy=CachePolicy(ttl_seconds=3600),
)
```

Retries handle transient failures inside the node. Cache
short-circuits when an identical input has been seen recently
(useful for expensive pure-function nodes).

## Send / SendBatch — fan-out inside a graph

For map/reduce inside a graph use the `Send` primitive from
`locus.core.send`:

```python
from locus.core.send import Send

def fan_out_review(state):
    return [Send(node="review_one", payload={"vendor": v}) for v in state["vendors"]]

graph.add_node("fan_out", fan_out_review)
graph.add_node("review_one", review_one_vendor_agent)
graph.add_node("merge", merge_reviews_agent)
graph.add_edge("plan", "fan_out")
graph.add_edge("fan_out", "merge")
```

The fan-out happens when a **node** returns a `Send` or a list of
them; the executor does not fan out a returned `SendBatch`. Whatever a
router passed to `add_conditional_edges` returns is read as node
names. Each `Send` names its own target, so `review_one` needs no
incoming edge: the targets run in parallel, each with the current
state plus that Send's payload. Once every Send has finished, each
successful result is stored in state under that Send's `send_id`,
and control follows `fan_out`'s own edge to `merge`.

## Mermaid visualisation

```python
graph.compile().get_mermaid()
```

…returns a Mermaid flowchart string you can paste into the docs or
a design review. It draws a conditional edge only from the `targets`
map and `default` you passed to `add_conditional_edges`. A router
passed on its own, like the one in the example above, draws no edge,
so that graph's review loop is missing from the output. Pass a
`targets` map to have the branches drawn.

## Notebooks

- [`notebook_12_basic_graph.py`](https://github.com/oracle/locus/blob/main/examples/notebook_12_basic_graph.py)
  — your first StateGraph.
- [`notebook_13_conditional_routing.py`](https://github.com/oracle/locus/blob/main/examples/notebook_13_conditional_routing.py)
  — `add_conditional_edges`.
- [`notebook_14_state_reducers.py`](https://github.com/oracle/locus/blob/main/examples/notebook_14_state_reducers.py)
  — custom state reducers.
- [`notebook_17_graph_advanced.py`](https://github.com/oracle/locus/blob/main/examples/notebook_17_graph_advanced.py)
  — `RetryPolicy`, `CachePolicy`, Mermaid output, streaming with `emit_custom`.
- [`notebook_20_map_reduce_support_triage.py`](https://github.com/oracle/locus/blob/main/examples/notebook_20_map_reduce_support_triage.py)
  — `Send` fan-out / reduce in a graph.
- [`notebook_21_supervisor_critic_loop.py`](https://github.com/oracle/locus/blob/main/examples/notebook_21_supervisor_critic_loop.py)
  — a draft → review loop on `add_conditional_edges`, capped at two
  revisions by a counter in state.
- [`notebook_77_incident_response.py`](https://github.com/oracle/locus/blob/main/examples/notebook_77_incident_response.py)
  — triage → parallel investigators → severity gate → page-the-human.
- [`notebook_78_concession_approval.py`](https://github.com/oracle/locus/blob/main/examples/notebook_78_concession_approval.py)
  — stacked `interrupt()` gates with tier routing.
- [`notebook_79_contract_review.py`](https://github.com/oracle/locus/blob/main/examples/notebook_79_contract_review.py)
  — parallel reviewers + `Command(goto=...)` to short-circuit a loop.

## Source

[`multiagent/graph.py`](https://github.com/oracle/locus/blob/main/src/locus/multiagent/graph.py)
— `StateGraph`, `Node`, `Edge`, `ConditionalEdge`, `RetryPolicy`,
`CachePolicy`. `Send` lives in
[`core/send.py`](https://github.com/oracle/locus/blob/main/src/locus/core/send.py).

## See also

- [Multi-agent overview](../multi-agent.md) — pick a shape.
- [Functional](functional.md) — when you'd rather use plain asyncio.
- [Termination](../termination.md) — composable stop conditions.
