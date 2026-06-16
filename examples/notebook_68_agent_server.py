# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Notebook 63: Agent server — deploy an agent as an HTTP API.

AgentServer wraps any Locus Agent in a FastAPI app: synchronous invoke,
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
  scoping, bearer auth. Use when Locus is the system of record and
  clients are yours.
- A2AServer: cross-framework interop with the A2A message spec. Use
  when another framework (Strands, ADK) needs to call your Locus agent.

Run it
    # Smoke test against a TestClient (no live server, no live model):
    LOCUS_MODEL_PROVIDER=mock python examples/notebook_68_agent_server.py

    # Boot a real uvicorn server on http://127.0.0.1:8000:
    LOCUS_NOTEBOOK_BOOT=1 python examples/notebook_68_agent_server.py

Prerequisites:

- pip install fastapi uvicorn
- For the persisted thread paths: an Oracle Autonomous Database with
  ORACLE_DSN / ORACLE_USER / ORACLE_PASSWORD / ORACLE_WALLET set.
  Without those env vars the notebook prints what's missing and exits.
"""

import os
import sys

from config import get_model

from locus.agent import Agent, AgentConfig
from locus.memory.backends import oracle_checkpointer
from locus.server import AgentServer


_REQUIRED_ENV = (
    "ORACLE_DSN",
    "ORACLE_USER",
    "ORACLE_PASSWORD",
    "ORACLE_WALLET",
)


def _missing_env() -> list[str]:
    return [name for name in _REQUIRED_ENV if not os.environ.get(name)]


def _build_checkpointer():
    return oracle_checkpointer(
        dsn=os.environ["ORACLE_DSN"],
        user=os.environ["ORACLE_USER"],
        password=os.environ["ORACLE_PASSWORD"],
        wallet_location=os.path.expanduser(os.environ["ORACLE_WALLET"]),
        wallet_password=os.environ.get("ORACLE_WALLET_PASSWORD", ""),
        table_name="locus_notebook_68",
    )


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
            # Oracle 26ai checkpointer so /threads/{id} survives restarts.
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

    print("\nTo run as a real server, set LOCUS_NOTEBOOK_BOOT=1 and run this")
    print("file directly. Example session:")
    print("  LOCUS_NOTEBOOK_BOOT=1 LOCUS_MODEL_PROVIDER=oci \\")
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

    Gated behind LOCUS_NOTEBOOK_BOOT=1 so the integration runner that
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
        print("\n--- Notebook 63: Agent Server on Oracle 26ai ---")
        print(
            "Required environment variables not set; skipping the live "
            "demo so this file still runs cleanly in CI.\n"
        )
        for name in missing:
            print(f"  - {name}")
        print(
            "\nProvision an Autonomous Database 26ai, then set "
            "ORACLE_DSN / ORACLE_USER / ORACLE_PASSWORD / ORACLE_WALLET "
            "and re-run."
        )
        sys.exit(0)
    if os.getenv("LOCUS_NOTEBOOK_BOOT") == "1":
        boot_live_server()
    else:
        example_server()
