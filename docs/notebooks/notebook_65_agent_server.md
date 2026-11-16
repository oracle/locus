# Agent server

This notebook deploys an on-call incident triage copilot as an HTTP API.
`AgentServer` wraps an `Agent` in a FastAPI app with a synchronous
`/invoke` route, a `/stream` route that returns Server-Sent Events (SSE),
and `/threads` routes that read the agent's checkpointer directly. Here the
checkpointer is Redis-backed, so an incident thread outlives any one server
process.

Authentication is a single shared bearer API key (the `api_key`
constructor argument or `LOCUS_SERVER_API_KEY`), and the checkpoint
namespace is derived from the presented key: a thread is stored as
`principal:thread_id`. That separates deployments with different keys that
share one checkpointer; it is not per-engineer isolation, because one
server holds one key and every caller who presents it shares one
principal. For per-tenant isolation, run one keyed `AgentServer` instance
per tenant. The derivation is described on the
[Agent Server concept page](../concepts/server.md#auth-and-thread-scoping).

What this notebook covers:

- Wrapping the triage `Agent`, with a Redis checkpointer, in `AgentServer`.
- Driving the routes through FastAPI's `TestClient` without binding a
  port: an `/invoke` with an explicit per-incident `thread_id`, the
  read-back on `GET /threads/{thread_id}`, a 404 for a thread never opened
  and an idempotent `DELETE`.
- A turn started in the background with `POST /threads/{thread_id}/runs`,
  followed on `GET /runs/{run_id}/events`, dropped after two frames and
  resumed with `Last-Event-ID`, so the rest of the run arrives once.
- Booting the same server under uvicorn with `LOCUS_NOTEBOOK_BOOT=1`.
- When to use `AgentServer` and when `A2AServer`.

Endpoints:

- `POST /invoke` — synchronous invocation.
- `POST /stream` — SSE streaming.
- `POST /threads/{thread_id}/runs` — start a turn in the background.
- `GET /runs/{run_id}/events` — follow it as SSE, resumable with
  `Last-Event-ID`.
- `GET /runs/{run_id}` — the run's status and outcome.
- `GET /threads/{thread_id}` — load a persisted incident thread and its
  runs.
- `DELETE /threads/{thread_id}` — drop a persisted incident thread.
- `GET /health` — health check.

The server also mounts `POST /resume`, which continues a paused thread,
`POST /runs/{run_id}/cancel`, the `WS /runs/live` socket and the `/ag-ui`
and `/vercel-ai` protocol endpoints; this notebook does not use them (see
[Human-in-the-loop](../concepts/server.md#human-in-the-loop),
[Live WebSocket](../concepts/server.md#live-websocket) and
[UI wire protocols](../concepts/server.md#ui-wire-protocols) on the
concept page).

When to use `AgentServer` vs `A2AServer`:

- **AgentServer**: first-party HTTP API with bearer-token authentication
  and persisted threads. Use it when the SDK is the system of record and the
  clients are yours (a PagerDuty webhook or a deploy dashboard).
- **A2AServer**: cross-framework interoperability over the
  [A2A (Agent-to-Agent)](../concepts/multi-agent/a2a.md) message
  specification. Use it when an agent runtime that speaks A2A needs to
  call your agent.

## Prerequisites

- `pip install fastapi uvicorn` and the Redis extra
  (`pip install "locus-sdk[redis]"`): `RedisBackend` imports its Redis
  client lazily, so without the extra the first checkpoint read or write
  raises `ImportError`.
- A Redis instance reachable at `REDIS_URL`. Both run modes require it:
  the script checks the variable before it chooses between the
  `TestClient` smoke test and the live uvicorn boot, prints the missing
  name and exits with status 0. The Redis-backed checkpointer is what
  makes `GET /threads/{thread_id}` survive a restart.

Every request the smoke test sends follows the path in the drawing:
`GET /health` answers without a token; every other route sits behind the
bearer check (the example passes no `api_key`, so unless
`LOCUS_SERVER_API_KEY` is set there is no token check and every caller is
the principal `anon`); `POST /invoke` and `POST /stream` run the triage
agent, which loads and saves the incident thread in the Redis checkpointer
under `principal:thread_id` when the request names a `thread_id`; and
`GET` and `DELETE /threads/{thread_id}` read or drop that stored state
directly, without running the agent:

{{ locus_diagram server-request-flow }}

## Key lines

A `RedisBackend` is raw key/value storage under the `locus_notebook_65:`
prefix; `StorageBackendAdapter` is what gives it the `Checkpointer`
contract the agent loop expects:

{{ locus_source examples/notebook_65_agent_server.py::_build_checkpointer }}

`_build_server` gives the triage `Agent` that checkpointer, so an incident
thread outlives any one server process, and wraps it in `AgentServer`;
the smoke test and the live boot share it:

{{ locus_source examples/notebook_65_agent_server.py::_build_server }}

`example_server` drives the routes through FastAPI's `TestClient` without
binding a port: `/health`, an `/invoke` with an explicit per-incident
`thread_id`, a `/stream`, the read-back on `GET /threads/inc-4711`, a 404
for a thread never opened and an idempotent `DELETE`. The client is opened
with `with`, so the event loop it runs the app on outlives each request —
which is what a run started in the background needs. The hint it prints
last applies to keyed deployments that share one checkpointer: on one
server, a token other than the configured key is answered with a 401, so
a second engineer with a second key needs a second keyed instance:

{{ locus_source examples/notebook_65_agent_server.py::example_server }}

`_show_resumable_run` is the part `/stream` cannot do. It starts a turn
with `POST /threads/inc-4712/runs`, which answers at once with a run id,
follows the run on `GET /runs/{run_id}/events` — every frame carries an
`id:` line with the event's sequence number — keeps the first two frames
as a client whose connection dropped would, and reconnects with
`Last-Event-ID` set to the last id it saw: the server replays exactly the
events after it, once, and ends with `[DONE]`. `GET /runs/{run_id}` then
shows the run's status and `GET /threads/inc-4712` lists the run under the
incident. The history behind the replay is bounded; the
[concept page](../concepts/server.md#threads-and-runs) gives the limits:

{{ locus_source examples/notebook_65_agent_server.py::_show_resumable_run }}

## Run it

```bash
python examples/notebook_65_agent_server.py
```

Without `LOCUS_NOTEBOOK_BOOT` the script runs the smoke test against
FastAPI's `TestClient`, so no port is bound. Model choice comes from
`examples/config.py`: OCI Generative AI when `~/.oci/config` has a usable
profile, otherwise the bundled mock model, so the example also runs offline
with no credentials. Set `LOCUS_MODEL_PROVIDER` (`oci`, `openai`,
`anthropic` or `mock`) to force one. Offline still needs the Redis
instance at `REDIS_URL` from Prerequisites.

`LOCUS_NOTEBOOK_BOOT=1` binds a live uvicorn server on
`http://127.0.0.1:8000` instead:

```bash
LOCUS_NOTEBOOK_BOOT=1 python examples/notebook_65_agent_server.py
```

## See also

- [Agent Server](../concepts/server.md) — endpoints, authentication and
  thread scoping, human-in-the-loop over `/resume`, deployment.
- [Checkpointers](../concepts/checkpointers.md) — the backends an
  incident thread can be persisted in, Redis among them.
- [SSE streaming](notebook_64_sse_streaming.md) — the SSE wire format and
  the handlers for it.
- [A2A protocol](notebook_30_a2a_protocol.md) — `A2AServer`, the
  cross-framework alternative.

## Source

??? note "Full source — `examples/notebook_65_agent_server.py`"

    ````python
    --8<-- "examples/notebook_65_agent_server.py"
    ````
