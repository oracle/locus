# Agent Server

`AgentServer` wraps any Tulip `Agent` in a FastAPI app: synchronous
invoke, streaming SSE, persisted threads scoped to the bearer principal
so two API keys sharing one server can't read each other's
conversations.

Endpoints:

- `POST /invoke` — synchronous invocation.
- `POST /stream` — SSE streaming.
- `GET /threads/{tid}` — load a persisted thread.
- `DELETE /threads/{tid}` — drop a persisted thread.
- `GET /health` — health check.

When to use `AgentServer` vs `A2AServer`:

- **AgentServer**: first-party HTTP API. Persisted threads, principal
  scoping, bearer auth. Use when Tulip is the system of record and
  clients are yours.
- **A2AServer**: cross-framework interop with the A2A message spec.
  Use when another framework (Strands, ADK) needs to call your Tulip
  agent.

Run it:

    # Smoke test against a TestClient (no live server, no live model):
    TULIP_MODEL_PROVIDER=mock python examples/notebook_68_agent_server.py

    # Boot a real uvicorn server on http://127.0.0.1:8000:
    TULIP_NOTEBOOK_BOOT=1 python examples/notebook_68_agent_server.py

Prerequisites:

- `pip install fastapi uvicorn`
- For the persisted thread paths: a Redis instance with `REDIS_URL`
  set. Without that env var the notebook prints what's missing and
  exits.

## Source

```python
--8<-- "examples/notebook_68_agent_server.py"
```
