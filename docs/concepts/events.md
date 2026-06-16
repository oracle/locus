# Events

Every observable step of an agent run is a typed Pydantic event. Not
a dict, not a callback, not a string — a frozen class with named
fields you can `match` on.

This is the reference page. For the *how* (consuming the stream,
SSE, hooks), see [Streaming](streaming.md). For the *why* (frozen,
typed, write-protected), see [Agent loop](agent-loop.md).

```python
from locus.core.events import (
    ThinkEvent, ToolStartEvent, ToolCompleteEvent, TerminateEvent,
)

async for event in agent.run("Plan a trip"):
    match event:
        case ThinkEvent(reasoning=r) if r:
            print("💭", r)
        case ToolStartEvent(tool_name=n, arguments=a):
            print(f"🔧 {n}({a})")
        case ToolCompleteEvent(tool_name=n, result=r, error=e):
            print(f"   ↳ {e or r}")
        case TerminateEvent(reason=r, final_message=m):
            print(f"[{r}] {m}")
```

## Common fields

Every event inherits from `LocusEvent` and carries:

| Field | Type | Meaning |
|---|---|---|
| `event_type` | `Literal[...]` | Discriminator string — `"think"`, `"tool_start"`, etc. |
| `timestamp` | `datetime` | UTC, populated at emit time. |

Events are **frozen** Pydantic models. A hook can read every field;
it cannot mutate one. To steer a run, use the explicit method on the
event (`event.cancel()`, `event.replace_arguments(...)`) — the intent
is visible in code review.

## Core events

### `ThinkEvent`

The model emitted reasoning, optionally with tool calls.

| Field | Meaning |
|---|---|
| `iteration` | ReAct turn index (0-based) |
| `reasoning` | The model's chain-of-thought, if the provider exposed it |
| `tool_calls` | Tool calls the model decided to make this turn |

Render this as a "thinking…" bubble. Most providers return `None`
unless extended thinking is enabled (Claude 4 / o-series).

### `ToolStartEvent`

The agent is about to invoke a tool.

| Field | Meaning |
|---|---|
| `tool_name` | Tool registered with `@tool` |
| `tool_call_id` | Provider-issued id, used to correlate with the matching `ToolCompleteEvent` |
| `arguments` | The validated arguments dict |

Show a "calling X" indicator.

### `ToolCompleteEvent`

A tool returned, errored, or was cancelled.

| Field | Meaning |
|---|---|
| `tool_name` | Same name as the matching start event |
| `tool_call_id` | Pairs with `ToolStartEvent.tool_call_id` |
| `result` | The serialised return value, or `None` on error |
| `error` | Exception message, or `None` on success |
| `duration_ms` | How long the body actually ran |

Always check `error` first — a non-`None` `error` means `result` is
`None`.

### `ModelChunkEvent`

One streamed chunk from the LLM provider — the granularity that
drives token-by-token rendering.

| Field | Meaning |
|---|---|
| `content` | Text delta (may be `None` for tool-call-only chunks) |
| `tool_calls` | Tool-call deltas, if the provider streams those |
| `done` | `True` on the final chunk of a turn |

`None`-guard before printing: `if e.content: print(e.content, end="")`.

### `ModelCompleteEvent`

A full model response was received (paired with the chunks above).

| Field | Meaning |
|---|---|
| `content` | The complete text |
| `tool_calls` | All tool calls in this turn |
| `usage` | `{"input_tokens": ..., "output_tokens": ...}` |
| `stop_reason` | Provider-specific stop reason |

Telemetry hooks key off `usage` for cost tracking.

### `ReflectEvent`

[Reflexion](reasoning.md#reflexion) emitted a self-evaluation.

| Field | Meaning |
|---|---|
| `iteration` | Which turn this reflection concerns |
| `assessment` | `"on_track"`, `"stuck"`, `"new_findings"`, or `"loop_detected"` |
| `confidence_delta` | Change vs the previous turn |
| `new_confidence` | Current value, 0.0–1.0 |
| `guidance` | Free-text steering for the next turn |

Pair `new_confidence` with [`ConfidenceMet`](termination.md) for early
stopping.

### `GroundingEvent`

[Grounding](reasoning.md#grounding) finished evaluating claims.

| Field | Meaning |
|---|---|
| `score` | 0.0–1.0, fraction of claims supported |
| `claims_evaluated` | How many claims the judge looked at |
| `ungrounded_claims` | The text of every unsupported claim |
| `requires_replan` | `True` if the run should re-research |

### `InterruptEvent`

A tool requested human-in-the-loop input. The run pauses; resume by
calling the agent with the user's reply.

| Field | Meaning |
|---|---|
| `question` | What to ask the human |
| `options` | If multiple-choice, the allowed answers |
| `interrupt_id` | Pass back to resume |
| `metadata` | Free-form context for the UI |

See [Interrupts](interrupts.md).

### `TerminateEvent`

The run finished.

| Field | Meaning |
|---|---|
| `reason` | Which termination condition fired (its `repr`) |
| `iterations_used` | How many ReAct turns ran |
| `final_confidence` | Reflexion confidence at end of run |
| `total_tool_calls` | Distinct tool invocations |
| `final_message` | The assistant's last text, if any |

Always emitted exactly once per run.

## Multi-agent events

These appear when an `Orchestrator`, `Swarm`, or `StateGraph` is
running.

| Event | Fired when |
|---|---|
| `SpecialistStartEvent` | Orchestrator dispatched to a specialist |
| `SpecialistCompleteEvent` | Specialist returned a result |
| `OrchestratorDecisionEvent` | Orchestrator picked its next step (`invoke_specialist`, `correlate`, `summarize`, `finalize`) |

See [Multi-agent](multi-agent.md).

## Causal-reasoning events

When `causal=True`, the agent emits node and edge events as the graph
grows.

| Event | Fired when |
|---|---|
| `CausalNodeEvent` | A new entity entered the cause-effect graph (root cause / symptom / intermediate) |
| `CausalEdgeEvent` | A causal link was added between two nodes |

## Hook events

`BeforeInvocationEvent`, `AfterInvocationEvent`, `BeforeToolCallEvent`,
`AfterToolCallEvent` — emitted *to hooks* around the same lifecycle
points the user-visible events come from. See [Hooks](hooks.md).

## Common gotchas

| Symptom | Likely cause |
|---|---|
| `match` is non-exhaustive at the type checker | Add a `case _: pass` fallthrough or handle the missing variant. |
| `ModelChunkEvent.content` is `None` | Tool-call-only chunk. Guard with `if event.content:`. |
| `TerminateEvent` never arrives | Generator was cancelled mid-stream. Check the consumer for exceptions. |
| Hook tried to mutate `event.tool_name` and got `ValidationError` | Frozen by design — use `event.replace_arguments(...)` or `event.cancel()` instead. |

## Source

- [`locus.core.events`](https://github.com/oracle-samples/locus/blob/main/src/locus/core/events.py) — every event class.

## See also

- [Streaming](streaming.md) — how to consume the event stream.
- [Hooks](hooks.md) — observe the same events from inside the loop.
- [Agent server](server.md) — re-emit events over Server-Sent Events.
