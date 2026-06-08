# Agent

`Agent` is the unit you build everything else from in
Tulip. Hand it a model, a
list of tools, a system prompt, and any optional features (reflexion,
grounding, checkpointing) — the SDK drives the
[Think → Execute → Reflect → Terminate](agent-loop.md) loop, streams
typed events as it runs, and returns a typed `AgentResult` when it
stops.

The same class is what multi-agent shapes (orchestrators, swarms,
handoff desks) and the [cognitive router](router.md) instantiate under
the hood — there's one agent abstraction, not five.

```python
from tulip.agent import Agent
from tulip.tools import tool
@tool
def search(query: str) -> str:
    """Search the knowledge base."""
    return "results"

agent = Agent(
    model="openai:gpt-4o",
    tools=[search],
    system_prompt="You are a helpful assistant.",
    max_iterations=20,
)
```

## Running the agent

There are three ways to drive the agent:

```python
# 1. Streaming events (async, fine-grained)
async for event in agent.run("Do the task", thread_id="t1"):
    print(event)

# 2. Sync execution (blocks until done)
result = agent.run_sync("Do the task", thread_id="t1")
print(result.message)

# 3. Alias for sync
result = agent.invoke("Do the task", thread_id="t1")
```

All three drive the same underlying [ReAct loop](#the-react-loop). The
only difference is the surface: `run` yields `TulipEvent` values as the
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
`token_budget`, `time_budget`, `interrupted`, `error`, `cancelled`.
Triggers:

- The model produces a response with no tool calls (`complete` / `no_tools`).
- A composable termination condition on `Agent(termination=...)` fires
  (see [`tulip.core.termination`](https://github.com/tuliplabs-ai/sdk-python/-/blob/main/src/tulip/core/termination.py)
  for the eight built-in conditions).
- `max_iterations`, `token_budget`, or `time_budget_seconds` is reached.
- A terminal tool name (in `terminal_tools`, default
  `{submit, done, finish, complete, task_complete}`) is invoked.
- `agent.cancel()` is called from another thread.

## Configuration

Everything is held in an `AgentConfig`. You can construct the config
explicitly and pass it, or let the `Agent` constructor build one from
keyword arguments.

```python
from tulip.agent import Agent
from tulip.agent import AgentConfig

cfg = AgentConfig(
    model="anthropic:claude-sonnet-4-6",   # see concepts/models.md
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

Six knobs cover ~95% of agent configurations. All accept either a
keyword on the `Agent(...)` constructor (sugar) or a field on
`AgentConfig` (when you build the config explicitly).

| Kwarg | What it does |
|---|---|
| `output_schema=Foo` | Pydantic schema. Final assistant message is parsed into an instance of `Foo` and surfaced on `result.parsed` / `result.parsed_as(Foo)`. Provider-strict `response_format` on OpenAI; tool-use translation on Anthropic; prompted fallback elsewhere. See [structured-output](structured-output.md). |
| `termination=cond` | Composable stop algebra: `MaxIterations(10) \| TextMention("DONE") & ConfidenceMet(0.9)` is real Python. Eight built-in conditions; `\|` and `&` operator overloads. |
| `playbook=plan` | A `tulip.playbooks.Playbook`. Auto-installs `PlaybookEnforcerHook` so each tool call is validated against the current step's `expected_tools` and the plan auto-advances. Out-of-sequence calls are cancelled with a hint. |
| `auxiliary_model="anthropic:claude-sonnet-4-6"` | Cheap-tier model for non-primary calls (max-iterations summary, grounding eval, conversation compactor). String or `ModelProtocol` instance. Falls back to `model=` when unset. |
| `reflexion=True` / `ReflexionConfig(...)` | Reflexion self-evaluation node in the loop. |
| `grounding=True` / `GroundingConfig(...)` | LLM-as-judge grounding evaluation against retrieved evidence. |

```python
from pydantic import BaseModel
from tulip.agent import Agent
from tulip.core.termination import MaxIterations, ToolCalled

class VendorList(BaseModel):
    vendors: list[str]

agent = Agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search, book_flight],
    output_schema=VendorList,
    termination=MaxIterations(8) | ToolCalled("book_flight"),
    auxiliary_model="anthropic:claude-sonnet-4-6",
    reflexion=True,
)
result = agent.run_sync("Find 3 vendors and book one.")
print(result.parsed_as(VendorList))
```
