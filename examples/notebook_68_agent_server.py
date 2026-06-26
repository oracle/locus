# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Notebook 68: Agent server — deploy an on-call incident triage copilot as an HTTP API.

AgentServer wraps any Tulip Agent in a FastAPI app: synchronous invoke,
streaming SSE, persisted incident threads scoped to the bearer principal
so two on-call engineers' API keys sharing one server can't read each
other's incidents. Principal-scoped threads are an identity-and-privilege
control: the bearer token decides which incidents a caller can read, and
the persisted thread is the auditable record left for each one.

Endpoints:

- POST /invoke         — synchronous invocation.
- POST /stream         — SSE streaming.
- GET  /threads/{tid}  — load a persisted incident thread.
- DELETE /threads/{tid}— drop a persisted incident thread.
- GET  /health         — health check.

When to use AgentServer vs A2AServer:

- AgentServer: first-party HTTP API. Persisted threads, principal
  scoping, bearer auth. Use when Tulip is the system of record and
  clients are yours (PagerDuty webhook, deploy dashboard).
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


_TRIAGE_PROMPT = (
    "You are an on-call incident triage copilot for an SRE team. Given an "
    "alert summary, say whether it warrants paging the on-call engineer and "
    "why. Be concise."
)


# Smoke test the server with FastAPI's TestClient. No port is bound.


def example_server():
    """Create an incident-triage copilot server with health, invoke, and stream endpoints."""
    print("=== On-Call Incident Triage Copilot Server ===\n")

    model = get_model()

    agent = Agent(
        config=AgentConfig(
            system_prompt=_TRIAGE_PROMPT,
            max_iterations=5,
            model=model,
            # Redis checkpointer so /threads/{id} survives restarts —
            # an incident outlives any one server process.
            checkpointer=_build_checkpointer(),
        )
    )

    server = AgentServer(
        agent=agent,
        title="On-Call Incident Triage Copilot API",
        description="An incident-triage copilot exposed as an HTTP API",
    )

    from fastapi.testclient import TestClient

    client = TestClient(server.app)

    r = client.get("/health")
    print(f"GET /health: {r.json()}")

    # Explicit thread_id (one per incident) so we can read it back through GET /threads.
    r = client.post(
        "/invoke",
        json={
            "prompt": (
                "Alert: checkout-api p99 latency at 4.2s (SLO 800ms) for 6 "
                "minutes after the v3.8.1 rollout. Page on-call?"
            ),
            "thread_id": "inc-4711",
        },
    )
    data = r.json()
    print(f"POST /invoke: {data['message']} (success={data['success']})")

    r = client.post("/stream", json={"prompt": "Name 3 common signs of a bad deploy."})
    print(f"POST /stream: status={r.status_code}")

    r = client.get("/threads/inc-4711")
    if r.status_code == 200:
        thread = r.json()
        print(
            f"GET /threads/inc-4711: iteration={thread['iteration']}, "
            f"messages={len(thread['messages'])}"
        )
    else:
        print(f"GET /threads/inc-4711: status={r.status_code}")

    r = client.get("/threads/inc-never-opened")
    print(f"GET /threads/inc-never-opened: status={r.status_code}")

    # DELETE is idempotent — a second call returns deleted=False.
    r = client.delete("/threads/inc-4711")
    print(f"DELETE /threads/inc-4711: {r.json()}")

    print("\nTo run as a real server, set TULIP_NOTEBOOK_BOOT=1 and run this")
    print("file directly. Example session:")
    print("  TULIP_NOTEBOOK_BOOT=1 TULIP_MODEL_PROVIDER=openai \\")
    print("      python examples/notebook_68_agent_server.py")
    print("  curl -s -X POST http://127.0.0.1:8000/invoke \\")
    print("       -H 'Content-Type: application/json' \\")
    print(
        '       -d \'{"prompt":"Page on-call? CrashLoopBackOff on payments-worker, 12 restarts."}\''
    )
    print("\nWith api_key= set, every /threads call is principal-scoped:")
    print("  AgentServer(agent=agent, api_key='secret')")
    print("  # Two on-call engineers with different bearer tokens see different")
    print("  # incidents for the same client-supplied thread_id.")
    return server


def boot_live_server() -> None:
    """Bind a live uvicorn instance.

    Gated behind TULIP_NOTEBOOK_BOOT=1 so the integration runner that
    imports every notebook doesn't hang on a blocking server.
    """
    model = get_model()
    agent = Agent(
        config=AgentConfig(
            system_prompt=_TRIAGE_PROMPT,
            max_iterations=5,
            model=model,
            checkpointer=_build_checkpointer(),
        )
    )
    server = AgentServer(
        agent=agent,
        title="On-Call Incident Triage Copilot API",
        description="An incident-triage copilot exposed as an HTTP API",
    )
    print("Booting AgentServer on http://127.0.0.1:8000 — Ctrl-C to stop.")
    print("Try: curl -X POST http://127.0.0.1:8000/invoke \\")
    print("          -H 'Content-Type: application/json' \\")
    print(
        '          -d \'{"prompt":"Page on-call? CrashLoopBackOff on payments-worker, 12 restarts."}\''
    )
    server.run(host="127.0.0.1", port=8000)


if __name__ == "__main__":
    missing = _missing_env()
    if missing:
        print("\n--- Notebook 68: Agent Server (Redis-backed incident threads) ---")
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
