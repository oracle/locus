# Agent

`Agent` is the unit you build everything else from in
Locus. Hand it a model, a
list of tools, a system prompt, and any optional features (reflexion,
grounding, checkpointing) — the SDK drives the
[Think → Execute → Reflect → Terminate](agent-loop.md) loop, streams
typed events as it runs, and returns a typed `AgentResult` when it
stops.

The same class is what multi-agent shapes (orchestrators, swarms,
handoff desks) instantiate: Locus has a single agent abstraction.

```python
from locus.agent import Agent
from locus.tools import tool

@tool
def lookup_order(order_id: str) -> str:
    """Look up an order's status, items, and payment history."""
    return "results"

agent = Agent(
    model="{{ locus_example_model }}",
    tools=[lookup_order],
    system_prompt="You are a customer-support agent.",
    max_iterations=20,
)
```

## Running the agent

There are three ways to drive the agent:

```python
# 1. Streaming events (async, fine-grained)
async for event in agent.run("Handle the refund request", thread_id="t1"):
    print(event)

# 2. Sync execution (blocks until done)
result = agent.run_sync("Handle the refund request", thread_id="t1")
print(result.message)

# 3. Alias for sync
result = agent.invoke("Handle the refund request", thread_id="t1")
```

All three drive the same underlying [ReAct loop](#the-react-loop). The
only difference is the surface: `run` yields `LocusEvent` values as the
loop progresses, `run_sync` / `invoke` return an `AgentResult` after
termination.

## The ReAct loop

Each iteration has three phases:

| Phase | What happens |
|---|---|
| **Think** | The model generates reasoning + optional tool calls. A `ThinkEvent` is emitted. |
| **Execute** | Tool calls run concurrently or sequentially depending on `tool_execution` (`"concurrent"` is the default). `ToolStartEvent` / `ToolCompleteEvent` fire per tool. |
| **Reflect** | Optional: reflexion re-checks the result; grounding verifies factual claims against evidence. |

The loop terminates with one of these `stop_reason` literals on
`AgentResult`: `complete`, `terminal_tool`, `confidence_met`,
`max_iterations`, `tool_loop`, `no_tools`, `grounding_failed`,
`token_budget`, `cost_budget`, `time_budget`, `context_exhausted`, `interrupted`,
`error`, `cancelled`.
Triggers:

- The model produces a response with no tool calls (`complete` / `no_tools`).
- A composable termination condition on `Agent(termination=...)` fires
  (see [`locus.core.termination`](https://github.com/oracle/locus/blob/main/src/locus/core/termination.py)
  for the eight built-in conditions).
- `max_iterations`, `token_budget`, `max_cost_usd`, or `time_budget_seconds` is reached.
- A terminal tool name (in `terminal_tools`, default
  `{submit, done, finish, complete, task_complete}`) is invoked.
- `agent.cancel()` is called from another thread.

## Configuration

Everything is held in an `AgentConfig`. You can construct the config
explicitly and pass it, or let the `Agent` constructor build one from
keyword arguments.

```python
from locus.agent import Agent
from locus.agent import AgentConfig

cfg = AgentConfig(
    model="{{ locus_example_model }}",   # see concepts/models.md
    tools=[...],
    system_prompt="...",
    max_iterations=50,
    completion_mode="explicit",
    tool_execution="concurrent",
    max_concurrency=8,
    checkpointer=...,
    hooks=[...],
)

agent = Agent(config=cfg)
```

See the [API reference](../api/agent.md) for every field.

## Headline kwargs

Six keyword arguments cover most agent configurations. Each is accepted as a
keyword on the `Agent(...)` constructor or as a field on `AgentConfig` when you
build the config explicitly.

| Kwarg | What it does |
|---|---|
| `output_schema=Foo` | Pydantic schema. Final assistant message is parsed into an instance of `Foo` and surfaced on `result.parsed` / `result.parsed_as(Foo)`. Provider-strict `response_format` on OpenAI; tool-use translation on Anthropic; prompted fallback elsewhere. See [structured-output](structured-output.md). |
| `termination=cond` | Composable stop algebra: conditions combine through the `__or__` and `__and__` operators (either condition, or both), so a rule such as "ten iterations, or the text DONE together with a confidence threshold" is plain Python. Eight built-in conditions; the example below composes two. |
| `playbook=plan` | A `locus.playbooks.Playbook`. Auto-installs `PlaybookEnforcerHook` so each tool call is validated against the current step's `expected_tools` and the plan auto-advances. Out-of-sequence calls are cancelled with a hint. |
| `auxiliary_model=...` | Lower-cost model for secondary calls (max-iterations summary, grounding evaluation, conversation compaction). String or `ModelProtocol` instance. Falls back to `model=` when unset. |
| `reflexion=True` / `ReflexionConfig(...)` | Reflexion self-evaluation node in the loop. |
| `grounding=True` / `GroundingConfig(...)` | LLM-as-judge grounding evaluation against retrieved evidence. |

```python
from pydantic import BaseModel
from locus.agent import Agent
from locus.core.termination import MaxIterations, ToolCalled

class OrderList(BaseModel):
    order_ids: list[str]

agent = Agent(
    model="{{ locus_example_model }}",
    tools=[lookup_order, issue_refund],
    output_schema=OrderList,
    termination=MaxIterations(8) | ToolCalled("issue_refund"),
    auxiliary_model="{{ locus_example_model }}",
    reflexion=True,
)
result = agent.run_sync("Look up the 3 orders and refund the eligible one.")
print(result.parsed_as(OrderList))
```
