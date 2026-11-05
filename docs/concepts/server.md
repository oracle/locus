# Agent Server

`AgentServer` is the reference way to run any `Agent` as a
service — drop in an `Agent` (a refund bot, a deploy operator, an
access-grant approver, or a SOC triage agent), and get a FastAPI app
with `/invoke`, `/stream`, and per-thread management out of the box. It
re-emits the same event stream the Python API exposes as Server-Sent
Events, gated by optional bearer-token auth. Most requests take the path
below — the bearer check, one of the routes, the agent with its thread
store, and a JSON or SSE reply; `/health` skips the check and the
`/threads` routes read the store directly.

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

The SOC variant is one swap:
`tools=security_toolset(siem=True, edr=True, threat_intel=True)` (from
`locus_security`, in the `locus-sdk-security` package) and a `ToolCalled("isolate_host")` terminal.

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
[case persistence](#case-persistence) flow and the `/threads/{tid}`
routes do not apply: those routes read `agent.config.checkpointer`, and
`GraphRunnable` has no `config`, so they fail with a 500 rather than the
no-checkpointer 404. `/resume` answers 404 as well — `GraphRunnable` has
no `resume()`, so the [human-in-the-loop](#human-in-the-loop) flow below
applies to `Agent` only. The same adapter slots into
`locus.a2a.A2AServer`.

## Endpoints

| Path | Method | Body | Returns |
|---|---|---|---|
| `/invoke` | POST | `{"prompt": "...", "thread_id": "..."}` | `InvokeResponse` JSON (`message`, `success`, `stop_reason`, `iterations`, `tool_calls`, `duration_ms`) |
| `/stream` | POST | same | `text/event-stream` SSE of `data: {json}` frames |
| `/resume` | POST | `{"thread_id": "...", "response": "..."}` and an optional `decision` | `text/event-stream` SSE of the continued thread; 409 and 404 cases below |
| `/health` | GET | — | liveness probe (200 OK) |
| `/threads/{tid}` | GET | — | thread state (requires checkpointer) |
| `/threads/{tid}` | DELETE | — | drop a thread |

`/resume` continues one thread, strictly by `thread_id`: the run this
process paused, or otherwise the thread rehydrated from its checkpoint.
It answers 409 while a held approval is still undecided, and 404 for an
unknown thread — no paused run in this process and no checkpoint under
that id — see [Human-in-the-loop](#human-in-the-loop) below.

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
| Console SSE connection drops mid-run | Reverse-proxy idle timeout. The stream is silent while a long tool call runs, and a proxy or load balancer closes a connection it considers idle. Raise `proxy_read_timeout` in nginx or the idle timeout on the load balancer so it exceeds your longest tool call; the server does not send heartbeat frames of its own. |
| Threads don't persist across restarts | `FileCheckpointer` writes to disk in the working directory — ephemeral container filesystems lose state on restart. Mount a volume or move to `S3Backend`. |
| `/threads/{tid}` returns 404 for a tid you know exists | Thread IDs are prefixed with the principal — `<principal>:<tid>` is what is stored. The path you pass is *your* tid; the server prefixes. A thread written under a different key (the key was rotated, or another deployment sharing the checkpointer wrote it), or written in-process by `agent.run(..., thread_id=...)` without the prefix, will not be found. A request with a wrong or missing key gets a 401, not a 404. |

## Source and notebook

- [`notebook_65_agent_server.py`](https://github.com/oracle/locus/blob/main/examples/notebook_65_agent_server.py) — runnable wrapper plus a curl client.
- [`locus.server`](https://github.com/oracle/locus/tree/main/src/locus/server) — `AgentServer`, `GraphRunnable`, `InvokeRequest`, `InvokeResponse`.

## See also

- [Interrupts](interrupts.md) — the in-process `interrupt()` / `agent.resume(...)` human-in-the-loop flow.
- [Streaming](streaming.md) — the Python iterator the SSE stream is built on.
- [Events](events.md) — every event type the server re-emits.
- [Checkpointers](checkpointers.md) — picking a backend that survives restarts, keeps parked cases, and scales out.
