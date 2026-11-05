---
title: Multi-agent workflows
---

# Multi-agent workflows

Multi-agent workflows are first-class in the SDK: eight shapes you compose
in one process or scale across a mesh, plus the Functional API for
hand-written asyncio glue, every one backed by the same `Agent` class, the
same event stream, and the same primitives. Compose the shape that fits
the work.

{{ locus_diagram multi-agent-patterns }}

!!! tip "Don't know which shape to use?"
    See [Pick a shape](#pick-a-shape) — three questions get you there. For
    a straight chain or fan-out, start with
    [Composition](multi-agent/composition.md).

## What the examples cover

Every example below is a real `examples/notebook_NN_*.py` file in the
repository. Each runs on the bundled `MockModel` with no credentials, uses
OCI GenAI automatically when `~/.oci/config` holds a usable profile, and
switches provider with `LOCUS_MODEL_PROVIDER`. The typed terminal
artifacts need a live model: on the mock they come back empty, and
notebook 26 stops with setup instructions instead.

| Notebook | Workflow |
|---|---|
| [`notebook_31`](https://github.com/oracle/locus/blob/main/examples/notebook_31_deepagent.py) | **DeepAgent — research factory.** `create_deepagent` with reflexion + grounding + subagent dispatch + `deepagent.*` SSE events. |
| [`notebook_20`](https://github.com/oracle/locus/blob/main/examples/notebook_20_map_reduce_support_triage.py) | **Customer-support triage at scale.** Scatter 3 tickets × 3 lenses (sentiment, routing, resolution) to 9 analyst agents via `Send`, reduce into one triage summary. |
| [`notebook_21`](https://github.com/oracle/locus/blob/main/examples/notebook_21_supervisor_critic_loop.py) | **Supervisor + critic loop.** Privacy analyst → Report author → Skeptical reviewer, loop back to the author until the reviewer approves, with a cap of two revisions. |
| [`notebook_26`](https://github.com/oracle/locus/blob/main/examples/notebook_26_debate_with_judge.py) | **Adversarial debate + judge.** One agent argues the finding is a true positive, another argues benign, across N rounds; Judge emits a typed `Verdict` via `output_schema`. |
| [`notebook_27`](https://github.com/oracle/locus/blob/main/examples/notebook_27_multiagent_human_in_loop.py) | **Multi-agent + human-in-the-loop.** Three patterns in one file: approval gate, human-as-tool, long-pause snapshot/resume. |
| [`notebook_77`](https://github.com/oracle/locus/blob/main/examples/notebook_77_incident_response.py) | **SRE on-call incident runbook.** Triage → 3 parallel investigators (logs, metrics, traces) → Synthesizer → severity gate → `interrupt()` pages a human on critical → Mitigator → typed `Postmortem`. |
| [`notebook_78`](https://github.com/oracle/locus/blob/main/examples/notebook_78_concession_approval.py) | **Customer-support concession approval.** Ticket analyst → Impact analyst → risk-tier router (auto / support manager / +billing / +director) → typed `ConcessionDecision`. Stacked `interrupt()` gates on the top tiers. |
| [`notebook_79`](https://github.com/oracle/locus/blob/main/examples/notebook_79_contract_review.py) | **Vendor DPA review.** Parser → 3 parallel reviewers (legal / governance / compliance) → negotiation gate → counsel redlines via `interrupt()`, and `Command(goto="sign_off")` short-circuits to the typed `ContractDecision` once resolved. Cycles enabled. |

## Pick a shape

Three questions get you to the right shape almost every time:

1. **Do agents need to talk across processes or runtimes?** If yes, you
   want **A2A**. If no, everything else lives in one Python process.
2. **Does the flow have cycles or conditional routing?** If yes, you
   want **StateGraph**. If it is a straight chain or fan-out, you want
   **Composition**. If the agents decide the flow at run time, go to
   question 3.
3. **Do you want one coordinator picking the next agent, or peers
   collaborating without a central router?** Coordinator →
   **Orchestrator + Specialists**. Peers → **Swarm**. A single agent
   passes the conversation onward → **Handoff**.

The decision tree below is the same questions in diagram form.

{{ locus_diagram choose-a-shape }}

Writing your own glue (asyncio fan-out, retries, schedulers)? Use the
**Functional API** (`@task`, `@entrypoint`) — a thin wrapper that brings
agent runs into ordinary asyncio code.

## The eight shapes

Eight shapes across seven pattern pages —
[Composition](multi-agent/composition.md) covers three of them
(Sequential, Parallel, Loop) in one page, and the Functional API rides
alongside as the asyncio-native escape hatch.

| Pattern | Best for | Key class | Source |
|---|---|---|---|
| **[Composition](multi-agent/composition.md)** | fixed pipelines: sequential, parallel, loop | `SequentialPipeline`, `ParallelPipeline`, `LoopAgent` | [`composition.py`](https://github.com/oracle/locus/blob/main/src/locus/agent/composition.py) |
| **[Orchestrator + Specialists](multi-agent/orchestrator.md)** | one router assigns sub-tasks to experts | `Orchestrator`, `Specialist` | [`orchestrator.py`](https://github.com/oracle/locus/blob/main/src/locus/multiagent/orchestrator.py) |
| **[Swarm](multi-agent/swarm.md)** | peers claim tasks over a shared context | `Swarm`, `SharedContext` | [`swarm.py`](https://github.com/oracle/locus/blob/main/src/locus/multiagent/swarm.py) |
| **[Handoff](multi-agent/handoff.md)** | escalation with a findings summary | `Handoff`, `HandoffAgent`, `create_handoff_manager` | [`handoff.py`](https://github.com/oracle/locus/blob/main/src/locus/multiagent/handoff.py) |
| **[StateGraph](multi-agent/graph.md)** | explicit graphs: cycles, conditional edges, subgraphs | `StateGraph`, `Node`, `Edge` | [`graph.py`](https://github.com/oracle/locus/blob/main/src/locus/multiagent/graph.py) |
| **[Functional](multi-agent/functional.md)** | asyncio-native glue over agent runs | `@task`, `@entrypoint` | [`functional.py`](https://github.com/oracle/locus/blob/main/src/locus/multiagent/functional.py) |
| **[A2A](multi-agent/a2a.md)** | agents in other processes or runtimes | `A2AServer`, `A2AClient`, `AgentCard` | [`protocol.py`](https://github.com/oracle/locus/blob/main/src/locus/a2a/protocol.py) |

## Workflow primitives

The pieces every shape is built from. Drop them into any graph node.

### `Send` — scatter / map-reduce

```python
from locus.core.send import Send

async def split(state):
    return [Send(node="worker", payload={"task": t}) for t in state["tasks"]]

async def reduce(state):
    results = [v for k, v in state.items() if k.startswith("send_")]
    return {"summary": "\n".join(r["output"] for r in results)}
```

Returning a list of `Send` from a node spawns parallel executions —
no `asyncio.gather`, no shared mutable state. Each worker's return value
lands in `state[send.send_id]` (ids start with `send_`), which is what the
reduce node above collects.

### `interrupt()` — pause for a human

```python
from locus.core import interrupt

async def approval_node(state):
    response = interrupt({"question": "Ship it?", "options": ["yes", "no"]})
    return {"approved": response == "yes"}
```

`interrupt()` raises `InterruptException`; the graph catches it,
snapshots state, and returns control to the caller. Resume by calling
`graph.execute(Command(resume="yes"))`.

### `Command(goto=...)` — explicit routing

```python
from locus.core import goto

async def smart_router(state):
    if state["urgent"]:
        return goto("emergency", escalated=True)   # skip ahead; kwargs are state updates
    return {"score": compute_score(state)}      # normal flow
```

Return a `Command` from a node to override the default edge — useful
for short-circuiting refinement loops or skipping straight to sign-off.
Notebook 79 uses it to leave the negotiation loop and go straight to
sign-off once counsel answers RESOLVED.

### `Agent(output_schema=...)` — typed terminal artifacts

{{ locus_model_tabs }}

```python
from pydantic import BaseModel
from locus.agent import Agent, AgentConfig


class Verdict(BaseModel):
    disposition: str   # "true_positive" | "benign"
    confidence: float
    reasoning: str


agent = Agent(config=AgentConfig(model="{{ locus_model }}", output_schema=Verdict))
result = agent.run_sync("...")
verdict: Verdict = result.parsed   # validated Pydantic instance, not free text
```

When you need a typed artifact at the workflow boundary — `Verdict`,
`Postmortem`, `ConcessionDecision`, `ContractDecision` — `output_schema`
gives you a validated Pydantic instance.

### `GraphConfig(allow_cycles=True)` — refinement loops

```python
from locus.multiagent.graph import GraphConfig, StateGraph

graph = StateGraph(config=GraphConfig(allow_cycles=True, max_iterations=20))
graph.add_edge("critic", "writer")   # loop edge — only legal with allow_cycles
```

Cycles are off by default, so a graph cannot loop forever by accident.
Opt in with `allow_cycles=True` plus an iteration cap.

## Why these workflows are production-ready

The production concerns that turn a demo into a product — typed terminal
artifacts, idempotent tools, checkpointing, reflexion, grounding and
streaming — work in any of the eight shapes, so choosing a shape does not
trade away production readiness. And a side-effecting tool in any
shape can be put behind the admission gate — wrap it with
[`gate_tool(...)`](control-layer.md) or call
[`admit()`](security-context.md) in its body, passing your
`ControlPolicy` and an `AuditTrail`. Every decision on it, whether allow,
hold or deny, is then appended to that hash-chained trail: each entry
commits to the one before it, so an edit, reorder, or deletion in the
middle breaks verification. Sign the trail and verify it against the
public key (`verify(keys=...)`), and a chain rebuilt around an edit by
anyone without your signing key fails too.

→ [Production-readiness](multi-agent/production.md)

## One event stream across all of them

All eight shapes — A2A included — share the same typed event
taxonomy. Consume directly from the generator, or use the opt-in
`EventBus` to get per-component SSE events (`agent.think`,
`agent.tool.started`, `multiagent.orchestrator.routing`, etc.)
from every layer simultaneously:

```python
from locus.observability import run_context, get_event_bus

async with run_context() as rid:
    result = await orchestrator.execute("Resolve the disputed orders in this morning's queue.")
    # The run is done: close its stream so the subscriber below replays
    # the recorded history and then stops, instead of waiting for more.
    await get_event_bus().close_stream(rid)

    async for ev in get_event_bus().subscribe(rid):
        match ev.event_type:
            case "multiagent.orchestrator.decision" if "specialists_selected" in ev.data:
                print("orchestrator →", ev.data["specialists_selected"])
            case "multiagent.specialist.completed":
                preview = ev.data.get("output_preview", ev.data.get("error"))
                print("  done", ev.data["specialist_type"], preview)
```

Orchestrator events carry `orchestrator_id` and specialist events carry
`specialist_id` and `specialist_type`, so you can attribute output to the
specialist that produced it. SSE streams from `AgentServer` carry the
same shape — your front-end consumer is unchanged whether the back-end
is a single agent, an orchestrator, a swarm, or an A2A mesh.

→ [Observability — EventBus & SSE](observability.md) ·
[SSE event catalogue](sse-events.md)

## Mixing shapes

A `StateGraph` node runs any async callable, so one node can await
`orchestrator.execute(...)`, another `swarm.execute(...)`, and a third
can be a whole `StateGraph` added as a subgraph. Give a node a
`retry_policy` and the graph re-runs it with backoff when it raises, up
to `max_attempts` times. A `TimeoutError`, including a node's own
`timeout_ms`, fails the node without a retry.
`Orchestrator.execute` and `Swarm.execute` catch their own errors and
return a result with `success=False`, so a node that wraps one should
check `result.success` and raise if you want the retry to fire. An
orchestrator still reports `success=True` when a single specialist
fails: that failure is on `result.specialist_results[<id>].error`, so
check it as well if a failed specialist should trigger the retry. Pick
the shape that fits each layer of the problem.

## See also

- [Agent Loop](agent-loop.md) — the loop every agent in every shape runs.
- [Hooks](hooks.md) — observe and steer across all of them.
- [Streaming](streaming.md) — the typed event taxonomy.
