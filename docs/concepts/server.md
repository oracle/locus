# Agent Server

`AgentServer` is the reference way to run any `Agent` as a
service — drop in an `Agent` (a refund bot, a deploy operator, an
access-grant approver, or a SOC triage agent), and get a FastAPI app
with `/invoke`, `/stream`, [threads and runs](#threads-and-runs) with a
resumable event stream, a [live WebSocket](#live-websocket) and the two
[UI wire protocols](#ui-wire-protocols) web agent UIs speak, out of the
box. It re-emits the same event stream the Python API exposes as
Server-Sent Events, gated by optional bearer-token auth. Most requests
take the path below — the bearer check, one of the routes, the agent
with its thread store, and a JSON or SSE reply; `/health` skips the
check and the `/threads` routes read the store directly.

{{ locus_diagram server-request-flow }}

!!! note "Single shared key — single principal"
    `AgentServer` takes **one** `api_key`. Every authenticated request
    presents that same token, so all callers hash to the **same**
    principal and share one thread namespace. The bearer check (a 401 on
    `/invoke`, `/stream`, `/resume` and `/threads/{tid}`) is what stops a caller
    without the key; the principal prefix on thread IDs only keeps
    thread IDs from colliding between deployments with different keys
    that share a checkpointer — it does **not** give you per-analyst or
    per-tenant isolation behind a shared key. For real multi-tenant
    isolation, run one keyed `AgentServer` instance per tenant.

```python
from locus.server import AgentServer

server = AgentServer(
    agent=refund_agent,
    title="Refund desk",
    api_key="…",                       # bearer-token auth
)

if __name__ == "__main__":
    server.run(host="0.0.0.0", port=8080)
```

## When to use it

| Situation | Use AgentServer? |
|---|---|
| Exposing an agent to a web/mobile UI (a support console, an ops dashboard, a SOC console) | **yes — SSE plus per-thread persistence is what you want** |
| Internal one-off task, single Python script | no — call `agent.run_sync(...)` directly |
| Embedding an agent in your own FastAPI service | possible, but consider importing `AgentServer.app` and mounting it under your existing app |
| Scaling out across many workers, one thread resumable on any of them | yes, **with** an `S3Backend` (or another shared checkpointer) so every worker sees the same thread state |

## Getting started

### 1. Wrap the refund bot

{{ locus_model_tabs }}

```python
from locus.agent import Agent
from locus.core.termination import ToolCalled, ConfidenceMet, MaxIterations
from locus.memory.backends.file import FileCheckpointer
from locus.server import AgentServer

agent = Agent(
    model="{{ locus_model }}",
    tools=[lookup_order, check_refund_policy, issue_refund],
    system_prompt="You are a refund agent. Cite order evidence; escalate without it.",
    reflexion=True,
    checkpointer=FileCheckpointer(base_dir="./cases"),
    termination=(ToolCalled("issue_refund") & ConfidenceMet(0.9)) | MaxIterations(8),
)

server = AgentServer(agent=agent, api_key="…")
server.run(host="0.0.0.0", port=8080)
```

### 2. Call `/invoke` (one-shot)

```bash
curl -sS -X POST http://localhost:8080/invoke \
  -H "Authorization: Bearer $LOCUS_SERVER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Refund the duplicate charge on ord-4821.", "thread_id": "ord-4821"}'
```

Returns an `InvokeResponse` JSON (`message`, `success`, `stop_reason`,
`iterations`, `tool_calls`, `duration_ms`) in one response — the run
reduced to its result, not the event stream. Use this for scheduled
jobs, workflow-engine steps, and anything that doesn't render
incrementally.

### 3. Call `/stream` (Server-Sent Events)

`/stream` is a **POST** endpoint: the prompt comes from the JSON body
and the token from the `Authorization` header. The browser `EventSource`
API only issues GET requests, so it can't drive this route — use
`fetch()` + a `ReadableStream` reader and parse the `data:` frames
yourself. Each frame is `data: {json}\n\n`; the JSON carries a `type`
field — `think`, `tool_start`, `tool_complete`, `done` and `error` keep
their historical shapes, `model_chunk` is a token delta (on by default;
`stream_tokens=False` on the server turns it off), `interrupt` is a
pause, and every other event is its `model_dump` under its `type` — and
the stream ends with a literal `data: [DONE]` frame. The route does
**not** emit named SSE events.

```javascript
const res = await fetch("/stream", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + token,
  },
  body: JSON.stringify({
    prompt: "Refund the duplicate charge on ord-4821.",
    thread_id: "ord-4821",
  }),
});

const reader = res.body.getReader();
const decoder = new TextDecoder();
let buf = "";
for (;;) {
  const { value, done } = await reader.read();
  if (done) break;
  buf += decoder.decode(value, { stream: true });
  const frames = buf.split("\n\n");
  buf = frames.pop();                               // keep the partial frame
  for (const frame of frames) {
    const line = frame.replace(/^data: /, "");
    if (line === "[DONE]") continue;
    const evt = JSON.parse(line);
    if (evt.type === "tool_start") {
      status.innerText = `running ${evt.tool}`;     // lookup_order, issue_refund…
    } else if (evt.type === "done") {
      output.innerText = evt.message;
    }
  }
}
```

The `data:` payload is the JSON projection of each agent event (`type`
plus its fields), derived from the same `async for event in
agent.run(...)` stream the Python API exposes.

## Serving a graph

`AgentServer` only needs the agent contract — an async
`run(prompt, *, thread_id=None, metadata=None)` that yields events. A
compiled `StateGraph` doesn't speak that directly, so wrap it in
`GraphRunnable`:

```python
from locus.multiagent.graph import StateGraph, START, END
from locus.server import AgentServer, GraphRunnable


async def draft(state):
    return {"answer": f"Draft reply for: {state['prompt']}"}


graph = StateGraph()
graph.add_node("draft", draft)
graph.add_edge(START, "draft")
graph.add_edge("draft", END)

server = AgentServer(
    agent=GraphRunnable(graph.compile(), input_key="prompt", output_key="answer"),
    api_key="…",
)
server.run(host="0.0.0.0", port=8080)
```

The request's `prompt` lands in the graph's initial state under
`input_key`. Each streamed graph event goes out on `/stream` as a
`think` frame, and the final state's `output_key` value becomes the
`message` of the `done` frame and of the `/invoke` response (with no
`output_key`, the whole final state is stringified). `thread_id` and
`metadata` are accepted but not passed to the graph, so the
[case persistence](#case-persistence) flow does not apply: `GraphRunnable`
has no `config`, so `GET /threads/{tid}` answers from the server's
in-memory [run registry](#threads-and-runs) only (the thread's runs, no
transcript). `/resume` answers 404 — `GraphRunnable` has no `resume()`,
so the [human-in-the-loop](#human-in-the-loop) flow below applies to
`Agent` only; a cancel on a graph run stops its task outright, since
there is no `cancel()` to ask first. The same adapter slots into
`locus.a2a.A2AServer`.

## Endpoints

| Path | Method | Body | Returns |
|---|---|---|---|
| `/invoke` | POST | `{"prompt": "...", "thread_id": "..."}` | `InvokeResponse` JSON (`message`, `success`, `stop_reason`, `iterations`, `tool_calls`, `duration_ms`) |
| `/stream` | POST | same | `text/event-stream` SSE of `data: {json}` frames |
| `/resume` | POST | `{"thread_id": "...", "response": "..."}` and an optional `decision` | `text/event-stream` SSE of the continued thread; 409 and 404 cases below |
| `/health` | GET | — | liveness probe (200 OK) |
| `/threads` | POST | `{"thread_id": "...", "metadata": {...}}`, both optional | 201 with the thread; 409 if the id exists |
| `/threads/{tid}` | GET | — | thread state: the checkpointed transcript when there is one, plus the thread's runs |
| `/threads/{tid}` | DELETE | — | drop a thread and forget its finished runs |
| `/threads/{tid}/runs` | POST | `{"prompt": "..."}` or `{"resume": {...}}`, optional `metadata` | 202 with `run_id`; the run executes in the background |
| `/runs/{rid}` | GET | — | the run's `status`, `event_count`, `message`, `stop_reason`, `interrupt` |
| `/runs/{rid}/events` | GET | header `Last-Event-ID`, query `protocol` | `text/event-stream`, resumable; see below |
| `/runs/{rid}/cancel` | POST | — | `{"accepted": true}` for a run in flight, `false` once finished |
| `/runs/live` | WebSocket | JSON messages | turns, approval answers, cancel, subscribe and the events of each run |
| `/ag-ui` | POST | an AG-UI `RunAgentInput` | AG-UI events as SSE |
| `/vercel-ai` | POST | a UI message request (`id`, `messages`) | a Vercel AI SDK UI message stream |

`/resume` continues one thread, strictly by `thread_id`: the run this
process paused, or otherwise the thread rehydrated from its checkpoint.
It answers 409 while a held approval is still undecided, and 404 for an
unknown thread — no paused run in this process and no checkpoint under
that id — see [Human-in-the-loop](#human-in-the-loop) below.

## Threads and runs

`/invoke` and `/stream` hold the connection for the length of the run.
The thread and run resources separate the two: a turn is *started* with
one request and *followed* with another, so a browser tab that reloads,
a phone that changes network or a proxy that times out can pick the
stream back up without losing or repeating an event.

```bash
# Start a turn; the response comes back at once.
curl -sS -X POST http://localhost:8080/threads/ord-4821/runs \
  -H "Authorization: Bearer $LOCUS_SERVER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Refund the duplicate charge on ord-4821."}'
# {"run_id": "3f1c…", "thread_id": "ord-4821", "status": "queued"}

# Follow it. Each frame carries an id: line, the sequence number of the event.
curl -sN http://localhost:8080/runs/3f1c…/events \
  -H "Authorization: Bearer $LOCUS_SERVER_API_KEY"
# id: 1
# data: {"type": "think", "content": "..."}
#
# id: 2
# data: {"type": "tool_start", "tool": "lookup_order", ...}
# ...
# data: [DONE]

# Dropped after event 2? Reconnect from there: events 3… follow, once.
curl -sN http://localhost:8080/runs/3f1c…/events \
  -H "Authorization: Bearer $LOCUS_SERVER_API_KEY" \
  -H "Last-Event-ID: 2"
```

The frames are the `/stream` shapes (`think`, `tool_start`,
`tool_complete`, `model_chunk`, `interrupt`, `done`, `error`, …), each
preceded by an `id:` line that numbers the event from 1. A client keeps
the last id it saw and sends it back as `Last-Event-ID` when it
reconnects; the server replays every retained event after it and then
follows the run live. A stream always ends with `data: [DONE]`, including
a reconnect that has nothing left to deliver. The browser `EventSource`
API does this bookkeeping on its own, but cannot set the `Authorization`
header, so the reference page under
[`examples/server_ui/`](https://github.com/oracle/locus/tree/main/examples/server_ui)
uses `fetch()` and keeps the id itself.

**Threads.** `POST /threads` creates one (a client-chosen `thread_id` or
a minted one) and is optional: a run on a thread id the server has not
seen creates the thread. `GET /threads/{tid}` answers from the agent's
checkpointer when one is configured — the transcript, `iteration`,
`tool_executions` — and adds the thread's `runs` from the server's
registry; `source` says which answered (`checkpointer` or `registry`).
Without a checkpointer the server keeps no transcript: the thread is
known for its runs only, and only for as long as the registry keeps it.
`DELETE /threads/{tid}` drops the checkpoint and forgets the thread's
finished runs.

**Runs.** One run at a time per thread (a second `POST` answers 409
`run_in_progress`). `GET /runs/{rid}` reports `status` — `queued`,
`running`, then `completed`, `interrupted` (paused on an approval or an
`ask_user` question; the `interrupt` field holds the pause), `cancelled`
or `failed` (`error_correlation_id` names the log entry) — along with
`event_count`, `message` and `stop_reason`. Continue a paused thread
with `{"resume": {"response": "...", "decision": {...},
"perform_dangling": true}}` in the body of `POST /threads/{tid}/runs`:
the same semantics as [`/resume`](#human-in-the-loop) (409
`approval_pending` while the decision is missing, 404 when nothing is
paused), as a new run with its own id. `POST /runs/{rid}/cancel` asks an
`Agent` to stop through `agent.cancel(thread_id=...)`, so the loop
finishes its current step, writes its checkpoint and ends with
`stop_reason` `cancelled`; if it has not stopped after
`cancel_grace_seconds` (5 by default), or the runnable has no `cancel()`,
the run's task is cancelled outright and a synthetic
`{"type": "done", "reason": "cancelled"}` closes the stream.

**Bounds.** The registry is in memory and bounded; the limits are the
arguments of `RunRegistry`, passed as `AgentServer(run_registry=...)`:

```python
from locus.server import AgentServer, RunRegistry

server = AgentServer(
    agent=agent,
    api_key="…",
    run_registry=RunRegistry(
        history_per_run=500,   # events kept per run for Last-Event-ID replay
        max_runs=200,          # finished runs kept; the oldest is forgotten first
        ttl_seconds=3600,      # a finished run and an idle thread are kept this long
    ),
)
```

A reconnect that reaches past the retained window gets one
`{"type": "replay_gap", "from": N, "to": M}` frame naming the sequence
numbers it missed, then whatever is still held; `GET /runs/{rid}` for a
forgotten run is a 404. The registry is per process: a run started on
one worker is known to that worker only, so put sticky routing in front
of a multi-worker deployment, or run one worker per replica. Thread
*state* is a different matter — that lives in the checkpointer and is
shared the way the [case persistence](#case-persistence) section
describes.

## Live WebSocket

`WS /runs/live` carries the same runs over one socket in both directions:
turns and approval answers in, events out. Authenticate with the
`Authorization: Bearer` header when the client can set one; a browser
cannot, so the server also accepts `?token=<key>` on the socket URL (the
URL may be logged by proxies — prefer the header where possible).

Messages the client sends, each a JSON object with a `type` and an
optional `ref` the server echoes:

| `type` | Fields | Effect |
|---|---|---|
| `run` | `prompt`, optional `thread_id` (minted when absent), `metadata` | start a turn |
| `resume` | `thread_id`, `response`, `decision`, `perform_dangling`, `metadata` | continue a paused thread — the `/resume` semantics |
| `cancel` | `run_id` | cancel a run |
| `subscribe` | `run_id`, optional `after` | follow a run started elsewhere, from a sequence number |
| `ping` | — | `pong` |

The server answers `run_started` (`run_id`, `thread_id`), then one
`event` frame per run event (`run_id`, `seq`, `event` — the same payload
the SSE route sends), then `run_finished` with the run's `status`; a
refused request is an `error` frame with the `code` and `detail` the HTTP
route would have returned (`approval_pending` carries the interrupt
details too), and `cancelled` reports whether a cancel was accepted.
Several runs can be in flight on one socket; the `run_id` on every frame
tells them apart. Voice frames are not carried: the realtime sessions in
`locus.voice` are provider-socket clients with their own event
vocabulary and no run binding, so a voice application drives a
`RealtimeSession` directly for now. Tool results from client-executed
tools are not carried either — the server runs its tools itself.

## UI wire protocols

Two web-UI protocols are served from the same runs and the same history,
so a front end can speak the protocol its components already understand
and still cancel or resume through the run resources.

**AG-UI.** `POST /ag-ui` takes a `RunAgentInput` (`threadId`, `runId`,
`messages`, `tools`, `context`, `state`, `forwardedProps`) and streams
AG-UI events: `RUN_STARTED`, a `STATE_SNAPSHOT` (`status`, `iteration`,
`toolCalls`) with `STATE_DELTA` JSON Patch updates as the run moves,
`TEXT_MESSAGE_START` / `TEXT_MESSAGE_CONTENT` / `TEXT_MESSAGE_END` for
assistant text, `TOOL_CALL_START` / `TOOL_CALL_ARGS` / `TOOL_CALL_END`
and `TOOL_CALL_RESULT` for tool calls, `CUSTOM` for everything else
(`locus.interrupt`, `locus.reasoning`, `locus.<event type>`, and a Locus
`CustomEvent` under its own name), and `RUN_FINISHED` (with `result`, or
`outcome: {"type": "interrupt", ...}` when the run paused) or
`RUN_ERROR`. The prompt is the last `user` message; `forwardedProps`
becomes the run's client metadata; the input's `tools` and `state` are
not consumed (the server's tools run server-side). The same frames are
available for any run as `GET /runs/{rid}/events?protocol=ag-ui`,
`Last-Event-ID` included.

**Vercel AI SDK UI message stream.** `POST /vercel-ai` takes the request
a `useChat`-style client sends (`id`, `messages` with `parts`; the chat
`id` is the thread id unless the body names `thread_id`) and answers with
the `x-vercel-ai-ui-message-stream: v1` header and the stream parts
`start`, `start-step`, `text-start` / `text-delta` / `text-end`,
`reasoning-*`, `tool-input-start` / `tool-input-available`,
`tool-output-available`, `data-*` (`data-locus-interrupt`,
`data-locus-event`, a Locus `CustomEvent` as `data-<name>`), `error`,
`finish-step`, `finish` and the `[DONE]` terminator. Also
`GET /runs/{rid}/events?protocol=vercel-ai`.

Where the published protocol pages leave a choice, the conservative
reading was taken: assistant text comes from token deltas when the
server streams them (`stream_tokens=True`, the default) and from the
`think` content otherwise, never both, and the final `done` message is
not repeated when it equals the text already sent; tool arguments are
sent whole (one `TOOL_CALL_ARGS` delta, a `tool-input-available` right
after `tool-input-start`); a tool failure is a result whose content is an
`{"error": "..."}` object rather than a stream-level error, so the UI
keeps rendering; an interrupt is a `CUSTOM` / `data-locus-interrupt`
frame followed by the protocol's run-finished frames, and the answer goes
through `POST /threads/{tid}/runs` with `resume` (or the socket's
`resume` message); one step per run (`start-step` / `finish-step`) rather
than one per model call. On a reconnect with `Last-Event-ID`, the
adapter rebuilds its state from the retained history, so message ids
match the ones the client already holds. The `id:` line sits on the last
frame of each event's group: a drop in the middle of a group resumes
from the event before it and shows the whole group again.

`/docs`, `/redoc`, and `/openapi.json` are only mounted when
`debug=True` in your settings — production deployments don't expose
schema by default.

## Auth and thread scoping

- **Bearer token (opt-in).** Auth is **off by default**. Pass
  `api_key="..."` to the constructor or set `LOCUS_SERVER_API_KEY` to
  turn it on; then every request must carry `Authorization: Bearer
  <token>`, constant-time compared with `hmac.compare_digest`.
- **Refuses to start off-loopback.** With no `api_key` (argument or
  `LOCUS_SERVER_API_KEY`) and no `allow_unauthenticated=True`,
  `server.run(host=...)` raises `RuntimeError` for any non-loopback
  host — it does not fall back to `127.0.0.1`. `run()` itself defaults
  to `host="127.0.0.1"`, so a bare `server.run()` still starts for local
  work. The check lives in `AgentServer.run()` only: serve `server.app`
  under your own uvicorn or gunicorn and it is bypassed (the app just
  logs a warning), so set the key yourself or terminate auth upstream.
- **Single-principal thread namespacing.** The principal is derived
  server-side from the presented key — `sha256(token)[:12]`, or `anon`
  when unauthenticated. Because the server holds **one** shared
  `api_key`, every authenticated caller resolves to the **same**
  principal, and thread IDs are prefixed with it (`<principal>:<tid>`).
  A caller without the key never reaches a thread — the bearer check
  rejects it with a 401 first. The prefix keeps thread IDs from
  colliding between deployments with different keys that share one
  checkpointer — it
  does **not** separate two analysts sharing the same key. For
  per-tenant isolation, run one keyed `AgentServer` per tenant.

```python
server = AgentServer(
    agent=agent,
    api_key=os.environ["LOCUS_SERVER_API_KEY"],
)
```

For unauthenticated dev:

```python
server = AgentServer(agent=agent, allow_unauthenticated=True)
server.run(host="127.0.0.1", port=8080)   # never 0.0.0.0
```

## Human-in-the-loop

Reads (`lookup_order`, `check_refund_policy`, `get_deploy_status`)
can auto-run. Writes that change the world — `issue_refund`,
`deploy_service`, isolating a host — should pause
for a human first. The SDK primitive for this is the in-process
`interrupt()` / `ask_user()` call: a tool calls it, the agent yields an
`InterruptEvent` and pauses, and the caller continues with
`agent.resume(response)` (see [Interrupts](interrupts.md)).

```python
from locus.core.interrupt import interrupt

@tool
def request_human_approval(reason: str, action: str) -> str:
    """Pause the run until a human approves the pending action."""
    return interrupt({"reason": reason, "action": action})
```

!!! note "Over HTTP: `/stream` pauses, `/resume` continues"
    Over `/stream`, the pause arrives as a `{"type": "interrupt", ...}`
    frame; with a checkpointer and a `thread_id`, the agent saves the
    thread before yielding it. `POST /resume` with `{"thread_id": ...,
    "response": ...}` (plus an optional `decision` payload, recorded by
    the server's `decision_handler`) continues that one thread as a new
    SSE stream: 409 `approval_pending` while a held approval is still
    undecided, 404 for an unknown thread — no run paused in this process
    and no checkpoint under that id (so, without a checkpointer, any
    thread that is not paused). A thread whose run already finished but
    still has a checkpoint is **not** refused: the agent reloads it,
    appends the response as a `[User Response]` system message when no
    tool call is left unanswered, and runs the loop again as a new
    segment. An unexpected exception on either stream is returned as a
    sanitized `{"type": "error", "error": "internal error",
    "correlation_id": ...}` frame, with details in the server log under
    that id.

## Case persistence

If the underlying `Agent` has a checkpointer, the server honours
`thread_id` in the request body for cross-request continuity. The same
key and the same `thread_id` address the same thread and its accumulated
state.

```bash
# Day 1
curl -sS -X POST http://localhost:8080/invoke \
  -H "Authorization: Bearer $LOCUS_SERVER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Open a case for the duplicate charge on ord-4821", "thread_id": "ord-4821"}'
# Day 2 — same thread_id, the case continues
curl -sS -X POST http://localhost:8080/invoke \
  -H "Authorization: Bearer $LOCUS_SERVER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What did we establish so far?", "thread_id": "ord-4821"}'
```

For multi-worker deployments, swap the checkpointer to one the workers
share so a thread written on one worker can be loaded on any other —
`S3Backend(bucket=..., prefix=...)` is the simplest path (it
implements `BaseCheckpointer` directly, so pass it straight to the
agent); the `redis_checkpointer(...)` and `postgresql_checkpointer(...)`
factories work too. See [Checkpointers](checkpointers.md).

## Deployment

The server is plain FastAPI — deploy it however you deploy FastAPI.

| Target | Path |
|---|---|
| **Kubernetes / container services** | `docker build` and ship; gunicorn-uvicorn workers in front |
| **serverless functions** | Mangum-style adapter wrapping `server.app` (the FastAPI app is built on first access) |
| **Compute / VM** | `uvicorn myapp:app --workers 4 --port 8080`, where `myapp.py` sets `app = AgentServer(agent=agent, api_key=...).app` at module scope |
| **Anywhere else FastAPI runs** | serve `server.app` the same way |

Each of these serves `server.app` directly rather than calling
`server.run()`, so the loopback check in `AgentServer.run()` never runs.
Set `api_key` (or `LOCUS_SERVER_API_KEY`) explicitly, or put an
auth-terminating proxy in front and pass `allow_unauthenticated=True`.

Auth, rate-limiting, and request logging are FastAPI middleware
concerns — the SDK does not own
them. Add `slowapi`, `prometheus-fastapi-instrumentator`, or whatever
your platform expects.

## Common gotchas

| Symptom | Likely cause |
|---|---|
| `RuntimeError: Refusing to bind AgentServer to '0.0.0.0'…` | No `api_key` and no `allow_unauthenticated=True`. Pick one. |
| Console SSE connection drops mid-run | Reverse-proxy idle timeout. The stream is silent while a long tool call runs, and a proxy or load balancer closes a connection it considers idle. Raise `proxy_read_timeout` in nginx or the idle timeout on the load balancer so it exceeds your longest tool call; the server does not send heartbeat frames of its own. On `/runs/{rid}/events` a drop costs nothing: reconnect with `Last-Event-ID`. |
| `GET /runs/{rid}` is a 404 for a run you started | The registry forgot it: finished runs expire after `ttl_seconds` (1 hour by default) or once `max_runs` newer ones have finished — or the request reached another worker. The registry is per process; see the bounds under [Threads and runs](#threads-and-runs). |
| Threads don't persist across restarts | `FileCheckpointer` writes to disk in the working directory — ephemeral container filesystems lose state on restart. Mount a volume or move to `S3Backend`. |
| `/threads/{tid}` returns 404 for a tid you know exists | Thread IDs are prefixed with the principal — `<principal>:<tid>` is what is stored. The path you pass is *your* tid; the server prefixes. A thread written under a different key (the key was rotated, or another deployment sharing the checkpointer wrote it), or written in-process by `agent.run(..., thread_id=...)` without the prefix, will not be found. A request with a wrong or missing key gets a 401, not a 404. |

## Source and notebook

- [`notebook_65_agent_server.py`](https://github.com/oracle/locus/blob/main/examples/notebook_65_agent_server.py) — runnable wrapper plus a curl client, and a run followed, dropped and resumed with `Last-Event-ID`.
- [`examples/server_ui/`](https://github.com/oracle/locus/tree/main/examples/server_ui) — two plain HTML pages, no build step: `index.html` starts a run and follows `GET /runs/{rid}/events` with resume, `ag-ui.html` speaks AG-UI to `POST /ag-ui`.
- [`locus.server`](https://github.com/oracle/locus/tree/main/src/locus/server) — `AgentServer`, `GraphRunnable`, `RunRegistry`, `AGUIAdapter`, `VercelDataStreamAdapter`, `InvokeRequest`, `InvokeResponse`.

## See also

- [Interrupts](interrupts.md) — the in-process `interrupt()` / `agent.resume(...)` human-in-the-loop flow.
- [Streaming](streaming.md) — the Python iterator the SSE stream is built on.
- [Events](events.md) — every event type the server re-emits.
- [Checkpointers](checkpointers.md) — picking a backend that survives restarts, keeps parked cases, and scales out.
