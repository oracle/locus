# Production-readiness

The boring stuff that turns a multi-agent demo into something you'd
deploy. Every primitive on this page works inside any of the seven
[workflow shapes](../multi-agent.md) — you don't pick "shape" or
"production-ready", you get both.

## Typed terminal artifacts

```python
from pydantic import BaseModel
from tulip.agent import Agent, AgentConfig
class Postmortem(BaseModel):
    severity: str
    root_cause: str
    timeline: list[str]
    action_items: list[str]

writer = Agent(config=AgentConfig(
    model="anthropic:claude-sonnet-4-6",
    output_schema=Postmortem,
))
result = writer.run_sync("Write a postmortem for incident #4421")
postmortem: Postmortem = result.parsed   # validated, not free text
```

`output_schema` validates the model's final answer against a Pydantic
schema. The workflow's terminal node returns a typed object — `Verdict`,
`Postmortem`, `PurchaseOrder`, `ContractDecision` — that the rest of
your system can consume without a brittle JSON re-parse.

Used by notebooks [44 (debate)][t44], [46 (incident)][t46],
[47 (procurement)][t47], [48 (contract)][t48].

→ See [Structured output](../structured-output.md).

## Idempotent tools — side effects fire once

```python
from tulip.tools import tool
@tool(idempotent=True)
def book_flight(flight_id: str, customer_id: str) -> dict:
    return billing.charge_and_book(flight_id, customer_id)
```

The ReAct loop dedupes repeat calls on the `(name, kwargs)` hash. The
model can't double-charge, double-book, or double-page even if the
graph retries a node or a checkpointed run resumes mid-tool. This is
the difference between a reliable agent and a horror story.

→ See [Idempotency](../idempotency.md).

## Durable memory — survive every restart

```python
from tulip.agent import Agent, AgentConfig
from tulip.memory.backends.oci_bucket import S3Backend

agent = Agent(config=AgentConfig(
    model="anthropic:claude-sonnet-4-6",
    checkpointer=S3Backend(bucket="tulip-state", namespace="..."),
))
```

Nine backend implementations, one `Checkpointer` Protocol — S3,
S3-compatible object storage, Postgres, MySQL, Redis, OpenSearch, HTTP, file, and
in-memory. The graph snapshots state at every `interrupt()`
boundary; you can pause for a human Friday afternoon and resume Monday
morning from a different process, region, or runtime.

→ See [Checkpointers](../checkpointers.md).

## Reflexion — catch a bad turn before the next one

```python
agent = Agent(config=AgentConfig(model="anthropic:claude-sonnet-4-6", reflexion=True))
```

`reflexion=True` self-evaluates every turn and feeds the next Think a
sharper plan. When a critic loop is overkill or you want intra-agent
self-correction, flip the flag.

→ See [Reasoning](../reasoning.md).

## Grounding — verify claims against their source

```python
agent = Agent(config=AgentConfig(model="anthropic:claude-sonnet-4-6", grounding=True))
```

Each claim in the model's output is scored against the tool result it
supposedly came from. Below-threshold claims get dropped or sent back
for revision. For typed grounding (entity-level evidence and
attribution), use [GSAR](../gsar.md).

→ See [Reasoning](../reasoning.md) · [GSAR](../gsar.md).

## Streaming events — every node visible

```python
from tulip.core.events import ToolStartEvent, TerminateEvent
from tulip.streaming import StreamMode

async for event in graph.stream(initial, mode=StreamMode.NODES):
    match event:
        case ToolStartEvent(tool_name=n, agent_name=a):
            print(f"{a} → {n}")
        case TerminateEvent(final_message=m, agent_name=a):
            print(f"{a} done: {m}")
```

Every shape in the framework emits the same typed event taxonomy.
`agent_name` is set on every event, so you can attribute output back to
the specialist that produced it. SSE-ready, match-statement friendly,
identical shape whether the back-end is a single agent, an
orchestrator, a swarm, or an A2A mesh.

→ See [Streaming](../streaming.md) · [Events](../events.md) ·
[Graph streaming](../graph-streaming.md).

## Observability — traces, metrics, hooks

```python
from tulip.agent import Agent, AgentConfig
from tulip.hooks.builtin import TelemetryHook

agent = Agent(config=AgentConfig(
    model="anthropic:claude-sonnet-4-6",
    hooks=[TelemetryHook(service_name="tulip-incident-bot")],
))
```

OpenTelemetry wired through every event. Hooks let you observe and
steer per-turn (`BeforeToolCallEvent`, `AfterToolCallEvent`,
`BeforeInvocationEvent`, etc.) without touching the graph.

→ See [Observability](../observability.md) · [Hooks](../hooks.md).

## Safety & guardrails

```python
from tulip.agent import Agent, AgentConfig
from tulip.hooks.builtin.guardrails import GuardrailsHook, GuardrailConfig

agent = Agent(config=AgentConfig(
    model="anthropic:claude-sonnet-4-6",
    hooks=[GuardrailsHook(config=GuardrailConfig())],
))
```

Input validation, PII redaction, topic policies, and tool restrictions
ride on the same hook system, so nothing is bolted-on. Stack them
freely.

→ See [Safety & Guardrails](../safety.md).

## Evaluation

```python
from tulip.evaluation import EvalCase, EvalRunner

cases = [
    EvalCase(input="...", expected_terminate=True),
    EvalCase(input="...", expected_tools=["search_logs"]),
]
report = EvalRunner(agent=graph).run(cases)
```

Run regression suites against any agent or graph. Failures point at the
specific node and event that diverged.

→ See [Evaluation](../evaluation.md).

## Putting it together

A notebook-46-style incident-response graph in production looks like:

```python
from tulip.agent import Agent, AgentConfig
from tulip.multiagent.graph import StateGraph, GraphConfig
from tulip.memory.backends.oci_bucket import S3Backend

graph = StateGraph(config=GraphConfig(
    allow_cycles=True,
    max_iterations=20,
    checkpointer=S3Backend(bucket="incidents", namespace="..."),
))
# ... nodes use Send for parallel investigation, interrupt() for the
# severity gate, output_schema=Postmortem for the terminal artifact,
# idempotent tools for paging, hooks for OTel spans.
```

That's the moat. Pick a [shape](../multi-agent.md) directly, or let
[PRISM — the cognitive router](../router.md) select and compile the
right one from a typed intent. Then wire the primitives above through
it and ship it.

[t44]: https://github.com/tuliplabs-ai/sdk-python/blob/main/examples/notebook_32_debate_with_judge.py
[t46]: https://github.com/tuliplabs-ai/sdk-python/blob/main/examples/notebook_63_incident_response.py
[t47]: https://github.com/tuliplabs-ai/sdk-python/blob/main/examples/notebook_64_procurement_approval.py
[t48]: https://github.com/tuliplabs-ai/sdk-python/blob/main/examples/notebook_65_contract_review.py
