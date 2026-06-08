# State

`AgentState` is everything one agent run has seen so far — every
message in the conversation, every tool call and its result, the
running confidence score, the iteration counter, and a free-form
metadata dict for application code.

It's an **immutable Pydantic model**: every mutation returns a new
instance, every collection is a `tuple` or `frozenset`, and the whole
thing round-trips through JSON without loss. That immutability is
load-bearing — it's why checkpoints are reproducible, why two
parallel branches in a graph can each "modify" the state without
stepping on each other, and why a hook reading `state.tool_executions`
can't accidentally corrupt the run.

```python
from tulip.core.state import AgentState
from tulip.core.messages import Message, Role

state = AgentState(agent_id="my-agent", max_iterations=20)
state = state.with_message(Message(role=Role.USER, content="hi"))
state = state.with_confidence(0.85)

# The original is untouched.
assert state.confidence == 0.85
```

## When you'll touch state directly

Most of the time you don't — `Agent.run(...)` builds and threads it
for you. Reach for it when:

| Situation | What to do |
|---|---|
| You're writing a custom hook and want to inspect the conversation so far | Read `state.messages`, `state.tool_executions`, `state.confidence` |
| You're persisting a run and rehydrating later | `state.to_checkpoint()` / `AgentState.from_checkpoint(d)` — every checkpointer does this internally |
| You're writing a custom termination predicate | `CustomCondition(lambda s: ...)` — `s` is `AgentState` |
| You're building a multi-agent graph | Reducers compose new `AgentState` from parallel branches (see below) |
| You want to seed a run from a previous transcript | Construct `AgentState(messages=(...))` and pass to `agent.run(...)` |

## Fields

| Field | Type | Meaning |
|---|---|---|
| `run_id` | `str` (UUID) | Unique to this run. |
| `agent_id` | `str \| None` | Stable identifier carried across runs of the same agent. |
| `messages` | `tuple[Message, ...]` | Full conversation, in order. |
| `iteration` | `int` | Current ReAct iteration index. |
| `max_iterations` | `int` | Upper bound before termination. |
| `tool_executions` | `tuple[ToolExecution, ...]` | Every tool call: name, args, result/error, duration, idempotent-cache hit flag. |
| `reasoning_steps` | `tuple[ReasoningStep, ...]` | Per-iteration think → execute → reflect record. |
| `confidence` | `float` | Reflexion signal, 0.0–1.0. |
| `confidence_threshold` | `float` | Threshold used by `ConfidenceMet`. |
| `confidence_history` | `tuple[float, ...]` | Confidence at each iteration — useful for plotting. |
| `tool_history` | `tuple[str, ...]` | Just the tool names, in order. Powers loop detection. |
| `tool_loop_threshold` | `int` | How many identical consecutive calls qualify as a loop. |
| `terminal_tools` | `frozenset[str]` | Tool names that auto-end the run when called. |
| `total_tokens_used` | `int` | Running total. `prompt_tokens_used` + `completion_tokens_used`. |
| `token_budget` | `int \| None` | Optional cap; `TokenLimit` reads this. |
| `errors` | `tuple[str, ...]` | Tool/model error messages encountered this run. |
| `metadata` | `dict[str, Any]` | Free-form context you can attach. |
| `started_at`, `updated_at` | `datetime` | UTC timestamps. |

## Builder methods

Every "mutation" returns a new state. Helpers exist for the common
cases — you rarely need to construct an `AgentState` from scratch:

```python
state = (
    state
    .with_message(Message(role=Role.ASSISTANT, content="..."))
    .with_tool_execution(execution)
    .with_iteration(state.iteration + 1)
    .with_confidence(0.78)
    .with_error("rate-limited")
    .with_metadata("user_tz", "America/New_York")
    .with_token_usage(prompt_tokens=312, completion_tokens=87)
)
```

The full set: `with_message`, `with_messages`, `with_iteration`,
`with_tool_execution`, `with_reasoning_step`, `with_confidence`,
`with_error`, `with_metadata`, `with_token_usage`.

## Round-trip through JSON

```python
data: dict = state.to_checkpoint()           # plain dict, JSON-safe
restored = AgentState.from_checkpoint(data)
assert restored == state
```

Every checkpointer in `tulip.memory.backends` uses this pair under the
hood. If you're writing a custom backend, all you need to do is
serialize whatever `to_checkpoint()` returns and rehydrate with
`from_checkpoint()` on resume.

## Reducers (for graphs only)

When two branches of a [StateGraph](multi-agent/graph.md) modify the
state in parallel, Tulip needs
to know how to merge them. That's what reducers do:

| Reducer | Combines two values by… |
|---|---|
| `add_messages` | extending the message tuple |
| `merge_dict` / `deep_merge_dict` | shallow / recursive dict merge |
| `append_list` / `unique_append_list` | concatenating, optionally deduping |
| `add_numbers`, `max_value`, `min_value` | arithmetic / extremum |
| `first_value`, `last_value` | take one branch's value |
| `set_union` | union the two sets |

Reducers are **opt-in at the graph level** — a plain `agent.run(...)`
doesn't use them. See `tulip.core.reducers` for the source.

## Common gotchas

| Symptom | Likely cause |
|---|---|
| `state.messages.append(...)` raises | Tuples are immutable. Use `state.with_message(m)`. |
| `to_checkpoint()` round-trip drops a field | The field's value isn't JSON-serialisable (e.g., a custom class in `metadata`). Stash a serialisable form, or extend the checkpointer. |
| Two branches in a graph clobber each other's messages | You forgot to declare the reducer for `messages`. Use `add_messages`. |
| `confidence_history` has fewer entries than iterations | Reflexion isn't running (`reflexion=True` not set), or the run terminated before the first reflect step. |

## Source

- [`tulip.core.state`](https://github.com/tuliplabs-ai/sdk-python/blob/main/src/tulip/core/state.py) — `AgentState`, `ToolExecution`, `ReasoningStep`.
- [`tulip.core.reducers`](https://github.com/tuliplabs-ai/sdk-python/blob/main/src/tulip/core/reducers.py) — graph-level merge helpers.

## See also

- [Checkpointers](checkpointers.md) — durable persistence of `AgentState`.
- [Events](events.md) — what gets emitted as state changes.
- [Termination](termination.md) — `CustomCondition(fn)` is `(state) -> bool`.
- [Multi-agent: StateGraph](multi-agent/graph.md) — where reducers earn their keep.
