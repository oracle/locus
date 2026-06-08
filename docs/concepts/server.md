# Agent Server

`AgentServer` is the reference HTTP wrapper — drop in an `Agent`,
get a FastAPI app with `/invoke`, `/stream`, and thread management
out of the box. It's the same event stream the Python API exposes,
re-emitted as Server-Sent Events with bearer-token auth and
per-principal thread isolation by default.

```python
from tulip.server import AgentServer

server = AgentServer(
    agent=my_agent,
    title="Booking concierge",
    api_key="…",                       # bearer-token auth
)

if __name__ == "__main__":
    server.run(host="0.0.0.0", port=8080)
```

## When to use it

| Situation | Use AgentServer? |
|---|---|
| Putting an agent behind a browser UI / mobile app | **yes — SSE plus thread persistence is what you want** |
| Internal tool, single Python script | no — call `agent.run_sync(...)` directly |
| Microservice in your own FastAPI app | possible, but consider importing `AgentServer.app` and mounting it under your existing app |
| Scaling out across many workers with shared threads | yes, **with** an `S3Backend` (or another shared checkpointer) so workers see the same conversation history |

## Getting started

### 1. Wrap an agent

```python
from tulip.agent import Agent
from tulip.memory.backends.file import FileCheckpointer
from tulip.server import AgentServer

agent = Agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search, summarise],
    checkpointer=FileCheckpointer(directory="./threads"),
)

server = AgentServer(agent=agent, api_key="…")
server.run(host="0.0.0.0", port=8080)
```

### 2. Call `/invoke` (one-shot)

```bash
curl -sS -X POST http://localhost:8080/invoke \
  -H "Authorization: Bearer $TULIP_SERVER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Find Q3 revenue.", "thread_id": "user-c42"}'
```

Returns the full `AgentResult` JSON in one response. Use this for
batch jobs, scripts, and anything that doesn't render incrementally.

### 3. Call `/stream` (Server-Sent Events)

```javascript
const es = new EventSource(
  "/stream?token=" + encodeURIComponent(token),
);

es.addEventListener("model_chunk", (e) => {
  const { content } = JSON.parse(e.data);
  output.innerText += content;
});

es.addEventListener("tool_start", (e) => {
  const { tool_name } = JSON.parse(e.data);
  status.innerText = `🔧 ${tool_name}`;
});

es.addEventListener("terminate", () => es.close());
```

Every typed event becomes its own SSE event-name; the `data:` payload
is the JSON-serialised event. Same shape as the Python API's
`async for event in agent.run(...)`.

## Endpoints

| Path | Method | Body | Returns |
|---|---|---|---|
| `/invoke` | POST | `{"prompt": "...", "thread_id": "..."}` | full `AgentResult` JSON |
| `/stream` | POST | same | `text/event-stream` SSE of typed events |
| `/health` | GET | — | liveness probe (200 OK) |
| `/threads/{tid}` | GET | — | conversation history (requires checkpointer) |
| `/threads/{tid}` | DELETE | — | drop a thread |

`/docs`, `/redoc`, and `/openapi.json` are only mounted when
`debug=True` in your settings — production deployments don't expose
schema by default.

## Auth and thread scoping

- **Bearer token.** Pass `api_key="..."` to the constructor or set
  `TULIP_SERVER_API_KEY`. Every request must carry
  `Authorization: Bearer <token>`. Constant-time compared with
  `hmac.compare_digest`.
- **Loopback-only fallback.** If you don't configure auth and don't
  pass `allow_unauthenticated=True`, the server warns and binds to
  loopback only — no accidental open agent endpoints on `0.0.0.0`.
- **Per-principal thread namespacing.** The principal is derived from
  the bearer token; thread IDs are prefixed with it server-side. One
  authenticated client can't resume another's conversation by
  guessing the `thread_id` (CWE-639).

```python
server = AgentServer(
    agent=agent,
    api_key=os.environ["TULIP_SERVER_API_KEY"],
)
```

For unauthenticated dev:

```python
server = AgentServer(agent=agent, allow_unauthenticated=True)
server.run(host="127.0.0.1", port=8080)   # never 0.0.0.0
```

## Thread persistence

If the underlying `Agent` has a checkpointer, the server honours
`thread_id` in the request body for cross-request continuity. Same
client + same `thread_id` → same conversation, same memory.

```bash
# Day 1
curl -X POST .../invoke -d '{"prompt":"Plan Tokyo", "thread_id":"user-c42"}'
# Day 2 — same thread_id, conversation continues
curl -X POST .../invoke -d '{"prompt":"What were we discussing?", "thread_id":"user-c42"}'
```

For multi-worker deployments, swap the checkpointer to one workers
share — `S3Backend(bucket=..., namespace=...)` is the
zero-friction path; `RedisCheckpointer` and
`PostgresCheckpointer` work too.

## Deployment

The server is plain FastAPI — deploy it however you deploy FastAPI.

| Target | Path |
|---|---|
| **Kubernetes / container services** | `docker build` and ship; gunicorn-uvicorn workers in front |
| **serverless functions** | Mangum-style adapter; cold-start friendly because `Agent` is constructed lazily |
| **Compute / VM** | `uvicorn tulip.server:app --workers 4 --port 8080` once you've defined `app` at module scope |
| **Anywhere else FastAPI runs** | …yes |

Auth, rate-limiting, and request logging are FastAPI middleware
concerns — Tulip does not own
them. Add `slowapi`, `prometheus-fastapi-instrumentator`, or whatever
your platform expects.

## Common gotchas

| Symptom | Likely cause |
|---|---|
| Server starts but binds to loopback only | No `api_key` and no `allow_unauthenticated=True`. Pick one. |
| Browser SSE drops every 30 seconds | Reverse-proxy idle timeout. Bump `proxy_read_timeout` in nginx / `idle_timeout` on the LB, or have the agent send heartbeats every ~25s. |
| Threads don't persist across restarts | `FileCheckpointer` writes to disk in the working directory — ephemeral container filesystems lose it. Mount a volume or move to `S3Backend`. |
| `/threads/{tid}` 404s for the right tid | Thread IDs are scoped to the principal — `<principal>:<tid>` is what's stored. The path you pass is *your* tid; the server prefixes. |

## Source and notebook

- [`notebook_68_agent_server.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/examples/notebook_68_agent_server.py) — runnable wrapper plus a curl client.
- [`tulip.server`](https://github.com/tuliplabs-ai/sdk-python/tree/main/src/tulip/server) — `AgentServer`, `InvokeRequest`, `InvokeResponse`.

## See also

- [Streaming](streaming.md) — the Python iterator the SSE stream is built on.
- [Events](events.md) — every event type the server re-emits.
- [Checkpointers](checkpointers.md) — picking a backend that survives restarts and scales out.
