# Production-readiness

The boring stuff that turns a multi-agent demo into something you'd
deploy. Every primitive on this page works inside any of the eight
[workflow shapes](../multi-agent.md) — you don't pick "shape" or
"production-ready", you get both.

## Typed terminal artifacts

```python
from pydantic import BaseModel
from locus.agent import Agent, AgentConfig
class Postmortem(BaseModel):
    severity: str
    root_cause: str
    timeline: list[str]
    action_items: list[str]

writer = Agent(config=AgentConfig(
    model="{{ locus_example_model }}",
    output_schema=Postmortem,
))
result = writer.run_sync("Write a postmortem for incident #4421")
postmortem: Postmortem = result.parsed   # validated, not free text
```

`output_schema` validates the model's final answer against a Pydantic
schema. The workflow's terminal node returns a typed object — `Verdict`,
`Postmortem`, `ConcessionDecision`, `ContractDecision` — that the rest of
your system can consume without a brittle JSON re-parse.

Used by notebooks [26 (debate)][t44], [77 (incident)][t46],
[78 (concession approval)][t47], [79 (DPA review)][t48].

→ See [Structured output](../structured-output.md).

## Idempotent tools — deduplicate identical calls

```python
from locus.tools import tool
@tool(idempotent=True)
def isolate_host(host_id: str, case_id: str) -> dict:
    return edr.quarantine(host_id, case_id)
```

The ReAct loop reuses the recorded result for identical `(name, kwargs)` calls
in its documented run/checkpoint scope. A downstream idempotency key is still
required to cover the window where an external action succeeds before its
receipt is durably recorded.

→ See [Idempotency](../idempotency.md).

## Durable memory — survive every restart

```python
from locus.agent import Agent, AgentConfig
from locus.memory.backends import S3Backend

agent = Agent(config=AgentConfig(
    model="{{ locus_example_model }}",
    checkpointer=S3Backend(bucket="locus-state", prefix="..."),
))
```

One `BaseCheckpointer` contract across every backend — S3, file, HTTP
and in-memory natively; Postgres, MySQL, Redis, OpenSearch and Oracle
through their `*_checkpointer()` factories (see the
[Checkpointers](../checkpointers.md) table). The graph snapshots state at every `interrupt()` boundary; you can pause
for a human Friday afternoon and resume Monday morning from a
different process, region, or runtime.

→ See [Checkpointers](../checkpointers.md).

## Reflexion — catch a bad turn before the next one

```python
agent = Agent(config=AgentConfig(model="{{ locus_example_model }}", reflexion=True))
```

`reflexion=True` self-evaluates every turn and feeds the next Think a
sharper plan. When a critic loop is overkill or you want intra-agent
self-correction, flip the flag.

→ See [Reasoning](../reasoning.md).

## Grounding — verify claims against their source

```python
agent = Agent(config=AgentConfig(model="{{ locus_example_model }}", grounding=True))
```

The configured grounding judge scores the final answer against the tool
evidence collected during the run; if the score is below the threshold
(0.65 by default), the loop replans, up to `max_replans` times (2 by
default). For typed grounding (entity-level evidence and
attribution), use [GSAR](../gsar.md).

→ See [Reasoning](../reasoning.md) · [GSAR](../gsar.md).

## Streaming events — every node visible

```python
from locus.multiagent import StreamMode

async for event in graph.stream(initial, mode=StreamMode.UPDATES):
    print(event.node_id, event.data)   # which node ran, what it wrote to state
```

`graph.stream()` yields `StreamEvent(mode, node_id, data)`; pick the
mode for the granularity you need (`UPDATES` for each node's output
dict, `NODES` for the full `NodeResult` with status and duration). The
agents inside the nodes keep their own typed event stream, and the
opt-in `EventBus` carries both layers as SSE events with the same shape
whether the back-end is a single agent, an orchestrator, a swarm, or an
A2A mesh.

→ See [Streaming](../streaming.md) · [Events](../events.md) ·
[Graph streaming](../graph-streaming.md).

## Observability — traces, metrics, hooks

```python
from locus.agent import Agent, AgentConfig
from locus.hooks.builtin import TelemetryHook

agent = Agent(config=AgentConfig(
    model="{{ locus_example_model }}",
    hooks=[TelemetryHook(service_name="locus-incident-bot")],
))
```

OpenTelemetry wired through every event. Hooks let you observe and
steer each model and tool call (`BeforeModelCallEvent`,
`AfterModelCallEvent`, `BeforeToolCallEvent`, `AfterToolCallEvent`)
and wrap the whole run with the `on_before_invocation` /
`on_after_invocation` methods, without touching the graph.

→ See [Observability](../observability.md) · [Hooks](../hooks.md).

## Safety & guardrails

```python
from locus.agent import Agent, AgentConfig
from locus.hooks.builtin.guardrails import GuardrailsHook, GuardrailConfig

agent = Agent(config=AgentConfig(
    model="{{ locus_example_model }}",
    hooks=[GuardrailsHook(config=GuardrailConfig())],
))
```

Input validation, PII redaction, topic policies, and tool restrictions
ride on the same hook system, so nothing is bolted-on. Stack them
freely. One caveat: hooks attach per-agent, so in orchestrated shapes
gate the side-effecting tool itself with
[`admit()`](../security-context.md) — the gate travels with the tool,
and no orchestration path bypasses it.

→ See [Safety & Guardrails](../safety.md).

## Evaluation

```python
from locus.evaluation import EvalCase, EvalRunner, GraphEvalTarget

cases = [
    EvalCase(name="terminates", prompt="...", expected_output_contains=["resolved"]),
    EvalCase(name="uses_siem", prompt="...", expected_tools=["query_siem"]),
]
report = EvalRunner(agent=GraphEvalTarget(graph, input_key="prompt")).run(cases)
```

Run regression suites against any agent; a `StateGraph` is wrapped in
`GraphEvalTarget`, which maps each case prompt onto the named input
field. Failures point at the specific node and event that diverged.

→ See [Evaluation](../evaluation.md).

## Putting it together

A notebook-77-style incident-response graph in production looks like:

```python
from locus.agent import Agent, AgentConfig
from locus.multiagent.graph import StateGraph, GraphConfig
from locus.memory.backends import S3Backend

graph = StateGraph(config=GraphConfig(
    allow_cycles=True,
    max_iterations=20,
    checkpointer=S3Backend(bucket="incidents", prefix="..."),
))
# ... nodes use Send for parallel investigation, interrupt() for the
# severity gate, output_schema=Postmortem for the terminal artifact,
# idempotent tools for paging, hooks for OTel spans.
```

Pick a [shape](../multi-agent.md) directly, or let
the loop reach for a shape tool and build the
right one from a typed intent. Then wire the primitives above through
it and ship it.

[t44]: https://github.com/oracle/locus/blob/main/examples/notebook_26_debate_with_judge.py
[t46]: https://github.com/oracle/locus/blob/main/examples/notebook_77_incident_response.py
[t47]: https://github.com/oracle/locus/blob/main/examples/notebook_78_concession_approval.py
[t48]: https://github.com/oracle/locus/blob/main/examples/notebook_79_contract_review.py
