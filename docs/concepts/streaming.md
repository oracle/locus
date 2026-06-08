# Streaming

Every Tulip agent emits a typed
event stream as it runs. The events are frozen Pydantic classes — not
strings, not `dict[str, Any]` blobs — designed to drop into a `match`
statement that your type checker can verify exhaustively:

```python
async for event in agent.run("Plan a trip to Paris."):
    match event:
        case ThinkEvent(reasoning=r) if r:
            print(f"💭 {r}")
        case ToolStartEvent(tool_name=n, arguments=a):
            print(f"🔧 {n}({a})")
        case TerminateEvent(final_message=m):
            print(f"\n✅ {m}")
```

This is the surface a UI consumes (live token rendering, tool-call
indicators, reasoning bubbles), the surface telemetry hooks observe,
and the surface `AgentServer` re-emits over Server-Sent Events for
browsers.

## When to consume the event stream

| You want… | Use… |
|---|---|
| Live token-by-token rendering in a UI | `async for event in agent.run(...)` |
| The final answer as a single value (tests, scripts, REPL) | `agent.run_sync(prompt).message` — no event handling |
| Spans / metrics on every model + tool call | install [`TelemetryHook`](hooks.md#telemetryhook) |
| To stream over HTTP to a browser | [`AgentServer`](server.md) re-emits as SSE |

## Getting started

### 1. Use `agent.run(prompt)` instead of `run_sync`

```python
async for event in agent.run("Plan a trip to Paris."):
    print(event)
```

`agent.run(...)` returns an async iterator. Each iteration yields one
event in the order it occurred.

### 2. Pattern-match on the event types

```python
from tulip.core.events import (
    ThinkEvent,
    ToolStartEvent,
    ToolCompleteEvent,
    ModelChunkEvent,
    ReflectEvent,
    TerminateEvent,
)

async for event in agent.run("Plan a trip to Paris."):
    match event:
        case ThinkEvent(reasoning=r) if r:
            print(f"💭 {r}")
        case ToolStartEvent(tool_name=n, arguments=a):
            print(f"🔧 {n}({a})")
        case ToolCompleteEvent(tool_name=n, result=r):
            print(f"   ↳ {r}")
        case ModelChunkEvent(content=c) if c:
            print(c, end="", flush=True)            # token-level streaming
        case ReflectEvent(assessment=a, new_confidence=c):
            print(f"🪞 {a} ({c:.2f})")
        case TerminateEvent(final_message=m):
            print(f"\n✅ {m}")
```

`match` checks every branch against the event class. If you forget a
branch your IDE underlines it; if you mistype a field name (e.g.
`reasonng` instead of `reasoning`) you get a static error.

## The event taxonomy

| Event | When it fires | Useful for |
|---|---|---|
| `ThinkEvent` | The model emits reasoning (extended-thinking models like Claude 4 / o-series) | Render "thinking…" bubbles in a UI |
| `ModelChunkEvent` | Each streamed text chunk from the model | Token-level live rendering |
| `ToolStartEvent` | The agent decided to call a tool | Show a "calling X" indicator |
| `ToolCompleteEvent` | A tool returned (or raised — check `error`) | Show the result inline |
| `ReflectEvent` | Reflexion emitted a self-evaluation | Show "I'm checking my work" |
| `GroundingEvent` | Grounding evaluation finished | Show "verifying claims" |
| `InterruptEvent` | A tool requested human-in-the-loop input | Block on user approval |
| `TerminateEvent` | The run finished — terminal condition met | Show the final answer |

Every event carries an `event_type` discriminator and a UTC
`timestamp`, so persisted streams replay in their original order.

## Write-protected — by design

Events are **frozen** Pydantic models. A hook can read every field;
it **cannot** mutate one. Try and you get a `ValidationError`. If a
hook wants to steer the agent (cancel a tool, retry a model call),
it uses an explicit method on the event (`event.cancel()`,
`event.retry()`, `event.replace_arguments(...)`) — the intent is
visible in code review.

Why this is important: in callback-based event systems any code can
silently mutate a field and you find out three hops downstream when
the value's wrong. The SDK's frozen events make that impossible.

## Sync wrapper — when you don't need the stream

```python
result = agent.run_sync("What is 2+2?")
print(result.message)        # 'Four.'
print(result.metrics.iterations)
```

`agent.run_sync(prompt)` consumes the event stream internally and
returns the final `AgentResult`. The events still emit (hooks still
fire), but you get a single value back. Use this in tests, REPLs,
and scripts where the trace doesn't matter.

## Practical recipe — render to a terminal UI

```python
async for event in agent.run("Find Q3 revenue and email it to me."):
    match event:
        case ToolStartEvent(tool_name=n):
            print(f"\n🔧 {n}", end="", flush=True)
        case ToolCompleteEvent(error=e) if e:
            print(f" ✗ {e}")
        case ToolCompleteEvent():
            print(" ✓")
        case ModelChunkEvent(content=c) if c:
            print(c, end="", flush=True)
        case TerminateEvent():
            print()
```

Every event class is a small Pydantic record — there's no hidden
state. What you see is what gets serialised over SSE, what your
checkpointer persists, what your structured logger records.

## SSE over HTTP — for browser UIs

The reference [`AgentServer`](server.md) maps the same event stream
onto Server-Sent Events. Same `event_type`, same fields, just
`Content-Type: text/event-stream` over HTTP.

```python
from tulip.server import AgentServer
import uvicorn

server = AgentServer(agent=agent)
uvicorn.run(server.app, port=8000)
```

```javascript
// Browser-side
const es = new EventSource('/stream?prompt=...');
es.addEventListener('ModelChunkEvent', (e) => {
    const { content } = JSON.parse(e.data);
    document.getElementById('out').innerText += content;
});
```

## Common gotchas

| Symptom | Likely cause |
|---|---|
| `async for` exhausts immediately | You're calling `agent.run_sync()` (sync) instead of `agent.run()` (async). |
| `ModelChunkEvent`s but no `TerminateEvent` | Generator was cancelled mid-stream. Check for exceptions in the consumer. |
| Same event fires twice | A hook re-yielded an event it received. Hooks observe, they don't re-emit. |
| Browser SSE drops every 30s | Default proxy timeout. Set `proxy_read_timeout` higher or have the agent send heartbeats. |

## Notebooks

- [`notebook_11_agent_streaming.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/examples/notebook_11_agent_streaming.py) — your first event consumer.
- [`notebook_13_sse_streaming.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/examples/notebook_13_sse_streaming.py) — full SSE wiring against `AgentServer`.

## Source

- [`tulip.core.events`](https://github.com/tuliplabs-ai/sdk-python/blob/main/src/tulip/core/events.py) — every event class.
- [`Agent.run`](https://github.com/tuliplabs-ai/sdk-python/blob/main/src/tulip/agent/agent.py) — the iterator that emits them.
- [`AgentServer`](https://github.com/tuliplabs-ai/sdk-python/tree/main/src/tulip/server) — the SSE wrapper.

## See also

- [Events](events.md) — full taxonomy in reference form.
- [Hooks](hooks.md) — observe the same stream from inside the loop.
- [Agent Server](server.md) — re-emit over HTTP/SSE.
- [Graph streaming](graph-streaming.md) — multi-agent state-graph event streams.
