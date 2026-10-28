# Streaming

Every Locus agent emits a typed event stream as it runs — a live,
ordered trace of what the agent is doing. Each step (every model call,
every tool call such as `issue_refund` or `deploy_service`) lands as a
timestamped event the instant it happens, in the order it happened. The
stream is the surface you render in a UI, forward to telemetry, and
replay in original order when reviewing a run. (It is a faithful trace,
not a tamper-evident ledger — for a hash-chained record of decisions,
where each entry commits to the one before it,
route them through the [`AuditTrail`](agentic-ai-security.md).)

The events are frozen Pydantic classes — not strings, not
`dict[str, Any]` blobs — designed to drop into a `match` statement that
your type checker can verify exhaustively (see
[step 2](#2-pattern-match-on-the-event-types) below).

This is the surface an operations console consumes — a support desk, a
deploy dashboard, a SOC console — with live token rendering,
tool-call indicators, and reasoning bubbles; the surface telemetry hooks
observe; and the surface `AgentServer` re-emits over Server-Sent Events
for browsers.

## When to consume the event stream

| You want… | Use… |
|---|---|
| Live token-by-token rendering in a UI | `async for event in agent.run(..., stream_tokens=True)` |
| The final answer as a single value (tests, scripts, REPL) | `agent.run_sync(prompt).message` — no event handling |
| Spans / metrics on every model + tool call | install [`TelemetryHook`](hooks.md#telemetryhook) |
| To stream over HTTP to a browser | [`AgentServer`](server.md) re-emits as SSE |

## Getting started

### 1. Use `agent.run(prompt)` instead of `run_sync`

```python
async for event in agent.run("Resolve the duplicate charge on ord-4821."):
    print(event)
```

`agent.run(...)` returns an async iterator. Each iteration yields one
event in the order it occurred.

### 2. Pattern-match on the event types

```python
from locus.core.events import (
    ThinkEvent,
    ToolStartEvent,
    ToolCompleteEvent,
    ModelChunkEvent,
    ReflectEvent,
    TerminateEvent,
)

async for event in agent.run("Resolve the duplicate charge on ord-4821.", stream_tokens=True):
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
| `ThinkEvent` | Once per loop turn, on every provider, after the model responds — `reasoning` carries the assistant's text for that turn (on the last turn that is the final answer, which `TerminateEvent.final_message` repeats), or the provider's separate reasoning channel when there is one | Render the assistant's text as it arrives, or a "thinking…" bubble in a UI |
| `ModelChunkEvent` | Each streamed text chunk from the model — only with `agent.run(..., stream_tokens=True)` | Token-level live rendering |
| `ToolStartEvent` | The agent decided to call a tool | Show a "calling X" indicator |
| `ToolCompleteEvent` | A tool returned (or raised — check `error`) | Show the result inline |
| `ReflectEvent` | Reflexion emitted a self-evaluation | Show "I'm checking my work" |
| `GroundingEvent` | Grounding evaluation finished | Show "verifying claims" |
| `InterruptEvent` | A tool requested human-in-the-loop input | Block on user approval |
| `TerminateEvent` | The run finished — terminal condition met | Show the final answer |

Every event carries an `event_type` discriminator and a UTC
`timestamp`, so persisted streams replay in their original order.

## Write-protected in memory — by design

The streaming events emitted by `agent.run()` are **frozen** Pydantic
models. A consumer can read every field; it **cannot** mutate one. Try
and you get a `ValidationError`. Streaming events are observation-only —
they are not the steering surface.

Steering is a separate, explicit surface: the **hook** events passed to
`HookProvider` callbacks (a distinct `ProtectedEvent` family in
`locus.hooks`). A hook steers by **assigning** to the event's writable
fields — `event.cancel = True` (or a string reason) to skip a tool,
`event.retry = True` to re-run a model/tool call, `event.arguments =
{...}` to rewrite tool arguments before execution. Assigning to a
read-only field raises `AttributeError`. (There is no `cancel()` /
`retry()` / `replace_arguments()` method — these are attribute
assignments, not method calls.)

What `frozen=True` buys you: a hook, downstream consumer, or logging
shim cannot silently rewrite a streaming event in process — what the
agent did is what the in-memory stream says it did. Note the scope: this
is **write-protection in memory only**. Once an event is serialised to
SSE, a log, or a logging backend there is no integrity guarantee on the
bytes; for a tamper-evident record route decisions through the
[`AuditTrail`](agentic-ai-security.md) hash chain — each entry commits
to the one before it, so editing any record breaks verification.

## Sync wrapper — when you don't need the stream

```python
result = agent.run_sync("What's the refund status of ord-4821?")
print(result.message)        # 'Refunded.'
print(result.metrics.iterations)
```

`agent.run_sync(prompt)` consumes the event stream internally and
returns the final `AgentResult`. The events still emit (hooks still
fire), but you get a single value back. Use this in tests, REPLs,
and scripts where the trace doesn't matter.

## Practical recipe — render to a terminal UI

```python
async for event in agent.run(
    "Check ord-4821 for a duplicate charge and refund it if confirmed.",
    stream_tokens=True,
):
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

Every event class is a small Pydantic record — no hidden state. The
console render, the SSE serialisation, and the events your hooks forward
to telemetry all derive from the same event objects, so the live view
and the forwarded trace stay consistent.

## SSE over HTTP — for browser UIs

The reference [`AgentServer`](server.md) maps the event stream onto
Server-Sent Events. Each event is written as a single `data: {json}\n\n`
frame with `Content-Type: text/event-stream`. The JSON carries a `type`
field (`think`, `tool_start`, `tool_complete`, `done`, `error`, or the
raw `event_type` for everything else); the stream ends with a literal
`data: [DONE]` frame. The route does **not** emit named SSE events
(`event:` lines), so every frame arrives as the default `message`.

```python
from locus.server import AgentServer
import uvicorn

server = AgentServer(agent=agent)
uvicorn.run(server.app, port=8000)
```

`/stream` is a **POST** endpoint that reads the prompt from the JSON
body (and an `Authorization: Bearer` header when an `api_key` is set), so
the browser `EventSource` API (which only issues GET) cannot drive it.
Use `fetch()` + a `ReadableStream` reader and parse the `data:` frames
yourself:

```javascript
// Browser-side — POST + manual SSE-frame parsing
const res = await fetch('/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt: 'Resolve the duplicate charge on ord-4821.' }),
});
const reader = res.body.getReader();
const decoder = new TextDecoder();
let buf = '';
for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const frames = buf.split('\n\n');
    buf = frames.pop();                         // keep the partial frame
    for (const frame of frames) {
        const line = frame.replace(/^data: /, '');
        if (line === '[DONE]') continue;
        const evt = JSON.parse(line);           // { type, ... }
        if (evt.type === 'think') {
            document.getElementById('out').innerText += evt.content;
        }
    }
}
```

## Common gotchas

| Symptom | Likely cause |
|---|---|
| `async for` exhausts immediately | You're calling `agent.run_sync()` (sync) instead of `agent.run()` (async). |
| `ModelChunkEvent`s but no `TerminateEvent` | Generator was cancelled mid-stream. Check for exceptions in the consumer. |
| Same event fires twice | A hook re-yielded an event it received. Hooks observe, they don't re-emit. |
| Browser SSE drops every 30s | Default proxy timeout. Set `proxy_read_timeout` higher or have the agent send heartbeats. |

## Notebooks

- [`notebook_06_agent_streaming.py`](https://github.com/oracle/locus/blob/main/examples/notebook_06_agent_streaming.py) — your first event consumer.
- [`notebook_64_sse_streaming.py`](https://github.com/oracle/locus/blob/main/examples/notebook_64_sse_streaming.py) — full SSE wiring against `AgentServer`.

## Source

- [`locus.core.events`](https://github.com/oracle/locus/blob/main/src/locus/core/events.py) — every event class.
- [`Agent.run`](https://github.com/oracle/locus/blob/main/src/locus/agent/agent.py) — the iterator that emits them.
- [`AgentServer`](https://github.com/oracle/locus/tree/main/src/locus/server) — the SSE wrapper.

## See also

- [Events](events.md) — full taxonomy in reference form.
- [Hooks](hooks.md) — observe the same stream from inside the loop.
- [Agent Server](server.md) — re-emit over HTTP/SSE.
- [Graph streaming](graph-streaming.md) — multi-agent state-graph event streams.
