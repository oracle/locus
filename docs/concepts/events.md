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

The streaming events on this page are **frozen** Pydantic models. A
consumer can read every field; mutating one raises `ValidationError`.
Steering is a separate surface: the **hook** events (`locus.hooks`,
a `ProtectedEvent` family) let a hook steer by **assigning** to a
writable field — `event.cancel = True` (or a reason string),
`event.retry = True`, `event.arguments = {...}`. There are no
`cancel()` / `retry()` / `replace_arguments()` methods; assigning to a
read-only field raises `AttributeError`. See [Hooks](hooks.md).

## Core events

### `ThinkEvent`

The model emitted reasoning, optionally with tool calls.

| Field | Meaning |
|---|---|
| `iteration` | ReAct turn index (0-based) |
| `reasoning` | The assistant's text for this turn (also readable as `content`). If the provider returns reasoning in a separate channel, that text is used instead. `None` on a turn with tool calls and no text |
| `tool_calls` | Tool calls the model decided to make this turn |

One fires per ReAct turn, on every provider. This is where interim
assistant text arrives on the streaming path; the final answer
arrives on `TerminateEvent.final_message`, and token-by-token deltas
on `ModelChunkEvent` (opt-in via `stream_tokens=True`).

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
drives token-by-token rendering. It fires only when you pass
`stream_tokens=True`, as in `agent.run(prompt, stream_tokens=True)`;
without the flag no chunk events arrive, and the turn's text comes
through on `ThinkEvent` and `TerminateEvent.final_message` instead.

| Field | Meaning |
|---|---|
| `content` | Text delta (may be `None` for tool-call-only chunks) |
| `tool_calls` | Tool-call deltas, if the provider streams those |
| `done` | `True` on the final chunk of a turn |

`None`-guard before printing: `if e.content: print(e.content, end="")`.

### `ModelCompleteEvent`

Defined in `locus.core.events` and listed in its `AllEvents` union,
but `agent.run()` does not emit it — with or without
`stream_tokens=True`. Each turn's text arrives on `ThinkEvent`, the
final answer on `TerminateEvent.final_message`, and the run's
cumulative token usage on `TerminateEvent.usage` (`None` when the
provider reported no usage).

| Field | Meaning |
|---|---|
| `content` | The complete text |
| `tool_calls` | All tool calls in this turn |
| `usage` | `{"input_tokens": ..., "output_tokens": ...}` |
| `stop_reason` | Provider-specific stop reason |

For cost tracking from the event stream, read `TerminateEvent.usage` instead.

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
| `usage` | Cumulative token usage for the run: `{"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}`, or `None` when the provider reported no usage (read that as unmetered, not zero) |

Always emitted exactly once per run.

## Multi-agent events

These appear when an `Orchestrator`, `Swarm`, or `StateGraph` is
running.

| Event | Fired when |
|---|---|
| `SpecialistStartEvent` | Orchestrator dispatched to a specialist |
| `SpecialistCompleteEvent` | Specialist returned a result |
| `OrchestratorDecisionEvent` | Orchestrator picked its next step (`invoke_specialist`, `correlate`, `summarize`, `finalize`) |

`Orchestrator.execute(task)` returns a result rather than yielding
events, so these arrive on the opt-in `EventBus` as
`multiagent.specialist.started` / `.completed`. Both carry
`specialist_id`, so you can attribute each phase to the agent that ran
it:

```python
from locus.observability import run_context, get_event_bus

phase = {}  # specialist id → wall-clock ms

async with run_context() as rid:
    result = await orchestrator.execute("Order dispute on ord-4821: verify the order, reprice, refund if owed.")
    await get_event_bus().close_stream(rid)  # replay the recorded history, then stop

    async for ev in get_event_bus().subscribe(rid):
        match ev.event_type:
            case "multiagent.specialist.started":
                print("→", ev.data["specialist_id"], ev.data["task_preview"])
            case "multiagent.specialist.completed":
                phase[ev.data["specialist_id"]] = round(ev.data["duration_ms"])

print(phase)
# {'verify': 1840, 'pricing': 6120, 'refund': 410}
```

See [Multi-agent](multi-agent.md).

## Causal-reasoning events

The cause-effect graph (`build_causal_chain()`) has a pair of typed
events for surfacing graph growth to a streaming consumer. They're part
of the `AllEvents` union in `locus.core.events` and rendered by the console handler; emit them
from your own wiring as you add nodes and edges.

| Event | Represents |
|---|---|
| `CausalNodeEvent` | A new entity in the cause-effect graph (root cause / symptom / intermediate) |
| `CausalEdgeEvent` | A causal link between two nodes |

## Hook events

Hooks do not receive the frozen classes on this page. A
`HookProvider` gets its own write-protected event family from
`locus.hooks`: `BeforeModelCallEvent` (writable `messages`),
`AfterModelCallEvent` (writable `retry`, `response`),
`BeforeToolCallEvent` (writable `arguments`, `cancel`) and
`AfterToolCallEvent` (writable `retry`, `result`).

Import them from the `locus.hooks` package — the same two tool-call
names also exist in `locus.core.events` (and are re-exported by
`locus.hooks.events`), but only the `locus.hooks.provider` versions
are what the loop passes to a hook. The invocation boundary has no
event object: `on_before_invocation(prompt, state)` and
`on_after_invocation(state, success)` receive plain arguments, and the
`BeforeInvocationEvent` / `AfterInvocationEvent` classes in
`locus.core.events` are never emitted. See [Hooks](hooks.md).

## Common gotchas

| Symptom | Likely cause |
|---|---|
| `match` is non-exhaustive at the type checker | Add a `case _: pass` fallthrough or handle the missing variant. |
| No `ModelChunkEvent`s arrive | `stream_tokens=True` was not passed to `agent.run()`. |
| `ModelChunkEvent.content` is `None` | Tool-call-only chunk. Guard with `if event.content:`. |
| `TerminateEvent` never arrives | Generator was cancelled mid-stream. Check the consumer for exceptions. |
| Tried to mutate a streaming event field and got `ValidationError` | Streaming events are frozen by design. To steer, do it on the hook event by assigning a writable field — `event.cancel = True`, `event.retry = True`, or `event.arguments = {...}`. |

## Source

- [`locus.core.events`](https://github.com/oracle/locus/blob/main/src/locus/core/events.py) — every event class.
- [`locus.hooks.provider`](https://github.com/oracle/locus/blob/main/src/locus/hooks/provider.py) — the write-protected hook events.

## See also

- [Streaming](streaming.md) — how to consume the event stream.
- [Hooks](hooks.md) — observe and steer the same lifecycle points from inside the loop.
- [Agent server](server.md) — re-emit events over Server-Sent Events.
