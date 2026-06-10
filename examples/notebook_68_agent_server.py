# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Notebook 68: Agent server — deploy an agent as an HTTP API.

AgentServer wraps any Tulip Agent in a FastAPI app: synchronous invoke,
streaming SSE, persisted threads scoped to the bearer principal so two
API keys sharing one server can't read each other's conversations.

Endpoints:

- POST /invoke         — synchronous invocation.
- POST /stream         — SSE streaming.
- GET  /threads/{tid}  — load a persisted thread.
- DELETE /threads/{tid}— drop a persisted thread.
- GET  /health         — health check.

When to use AgentServer vs A2AServer:

- AgentServer: first-party HTTP API. Persisted threads, principal
  scoping, bearer auth. Use when Tulip is the system of record and
  clients are yours.
- A2AServer: cross-framework interop with the A2A message spec. Use
  when another framework (Strands, ADK) needs to call your Tulip agent.

Run it
    # Smoke test against a TestClient (no live server, no live model):
    TULIP_MODEL_PROVIDER=mock python examples/notebook_68_agent_server.py

    # Boot a real uvicorn server on http://127.0.0.1:8000:
    TULIP_NOTEBOOK_BOOT=1 python examples/notebook_68_agent_server.py

Prerequisites:

- pip install fastapi uvicorn
- For the persisted thread paths: a Redis instance with REDIS_URL set.
  Without that env var the notebook prints what's missing and exits.
"""

import os
import sys

from config import get_model

from tulip.agent import Agent, AgentConfig
from tulip.memory.backends import RedisBackend
from tulip.server import AgentServer


_REQUIRED_ENV = ("REDIS_URL",)


def _missing_env() -> list[str]:
    return [name for name in _REQUIRED_ENV if not os.environ.get(name)]


def _build_checkpointer():
    backend = RedisBackend(
        url=os.environ["REDIS_URL"],
        namespace="tulip_notebook_68",
    )
    return backend.as_checkpointer()


# Smoke test the server with FastAPI's TestClient. No port is bound.


def example_server():
    """Create an agent server with health, invoke, and stream endpoints."""
    print("=== Agent Server ===\n")

    model = get_model()

    agent = Agent(
        config=AgentConfig(
            system_prompt="You are a helpful assistant. Answer concisely.",
            max_iterations=5,
            model=model,
            # Redis checkpointer so /threads/{id} survives restarts.
            checkpointer=_build_checkpointer(),
        )
    )

    server = AgentServer(
        agent=agent,
        title="My Agent API",
        description="A helpful AI assistant exposed as HTTP API",
    )

    from fastapi.testclient import TestClient

    client = TestClient(server.app)

    r = client.get("/health")
    print(f"GET /health: {r.json()}")

    # Explicit thread_id so we can read it back through GET /threads.
    r = client.post(
        "/invoke",
        json={"prompt": "What is 2+2?", "thread_id": "demo-thread"},
    )
    data = r.json()
    print(f"POST /invoke: {data['message']} (success={data['success']})")

    r = client.post("/stream", json={"prompt": "Name 3 colors."})
    print(f"POST /stream: status={r.status_code}")

    r = client.get("/threads/demo-thread")
    if r.status_code == 200:
        thread = r.json()
        print(
            f"GET /threads/demo-thread: iteration={thread['iteration']}, "
            f"messages={len(thread['messages'])}"
        )
    else:
        print(f"GET /threads/demo-thread: status={r.status_code}")

    r = client.get("/threads/never-existed")
    print(f"GET /threads/never-existed: status={r.status_code}")

    # DELETE is idempotent — a second call returns deleted=False.
    r = client.delete("/threads/demo-thread")
    print(f"DELETE /threads/demo-thread: {r.json()}")

    print("\nTo run as a real server, set TULIP_NOTEBOOK_BOOT=1 and run this")
    print("file directly. Example session:")
    print("  TULIP_NOTEBOOK_BOOT=1 TULIP_MODEL_PROVIDER=openai \\")
    print("      python examples/notebook_68_agent_server.py")
    print("  curl -s -X POST http://127.0.0.1:8000/invoke \\")
    print("       -H 'Content-Type: application/json' \\")
    print('       -d \'{"prompt":"What is 2+2?"}\'')
    print("\nWith api_key= set, every /threads call is principal-scoped:")
    print("  AgentServer(agent=agent, api_key='secret')")
    print("  # Two clients with different bearer tokens see different threads")
    print("  # for the same client-supplied thread_id.")
    return server


def boot_live_server() -> None:
    """Bind a live uvicorn instance.

    Gated behind TULIP_NOTEBOOK_BOOT=1 so the integration runner that
    imports every notebook doesn't hang on a blocking server.
    """
    model = get_model()
    agent = Agent(
        config=AgentConfig(
            system_prompt="You are a helpful assistant. Answer concisely.",
            max_iterations=5,
            model=model,
            checkpointer=_build_checkpointer(),
        )
    )
    server = AgentServer(
        agent=agent,
        title="My Agent API",
        description="A helpful AI assistant exposed as HTTP API",
    )
    print("Booting AgentServer on http://127.0.0.1:8000 — Ctrl-C to stop.")
    print("Try: curl -X POST http://127.0.0.1:8000/invoke \\")
    print("          -H 'Content-Type: application/json' \\")
    print('          -d \'{"prompt":"What is 2+2?"}\'')
    server.run(host="127.0.0.1", port=8000)


if __name__ == "__main__":
    missing = _missing_env()
    if missing:
        print("\n--- Notebook 68: Agent Server (Redis-backed threads) ---")
        print(
            "Required environment variables not set; skipping the live "
            "demo so this file still runs cleanly in CI.\n"
        )
        for name in missing:
            print(f"  - {name}")
        print("\nStart a Redis instance, set REDIS_URL, and re-run.")
        sys.exit(0)
    if os.getenv("TULIP_NOTEBOOK_BOOT") == "1":
        boot_live_server()
    else:
        example_server()
