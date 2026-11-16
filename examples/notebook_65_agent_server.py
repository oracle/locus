# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Notebook 65: Agent server — deploy an on-call incident triage copilot as an HTTP API.

AgentServer wraps any Locus Agent in a FastAPI app: synchronous invoke,
streaming SSE, and persisted incident threads. Authentication is one
shared bearer API key (api_key= or LOCUS_SERVER_API_KEY); the checkpoint
namespace is derived from the presented key, so two keyed deployments
sharing one checkpointer store the same client-supplied thread_id under
different principals. One server holds one key: any other token is
answered with a 401, and every caller who presents the key shares one
principal. The persisted thread is the auditable record left for each
incident.

Endpoints:

- POST   /invoke                    — synchronous invocation.
- POST   /stream                    — SSE streaming.
- POST   /threads/{thread_id}/runs  — start a turn in the background.
- GET    /runs/{run_id}/events      — follow it as SSE; resumable with Last-Event-ID.
- GET    /runs/{run_id}             — the run's status and outcome.
- GET    /threads/{thread_id}       — load a persisted incident thread and its runs.
- DELETE /threads/{thread_id}       — drop a persisted incident thread.
- GET    /health                    — health check.

When to use AgentServer vs A2AServer:

- AgentServer: first-party HTTP API. Persisted threads, bearer auth,
  key-namespaced checkpoints. Use when Locus is the system of record and
  clients are yours (PagerDuty webhook, deploy dashboard).
- A2AServer: cross-framework interop with the A2A message spec. Use
  when an agent built on another framework needs to call your Locus agent.

Run it:
    # Smoke test against a TestClient (no live server, no live model):
    LOCUS_MODEL_PROVIDER=mock python examples/notebook_65_agent_server.py

    # Boot a real uvicorn server on http://127.0.0.1:8000:
    LOCUS_NOTEBOOK_BOOT=1 python examples/notebook_65_agent_server.py

Prerequisites:

- pip install fastapi uvicorn
- For the persisted thread paths: a Redis instance with REDIS_URL set.
  Without that env var the notebook prints what's missing and exits.
"""

import json
import os
import sys

from config import get_model

from locus.agent import Agent, AgentConfig
from locus.memory.backends import RedisBackend, StorageBackendAdapter
from locus.server import AgentServer


_REQUIRED_ENV = ("REDIS_URL",)


def _missing_env() -> list[str]:
    return [name for name in _REQUIRED_ENV if not os.environ.get(name)]


def _build_checkpointer():
    # ``prefix`` is the key namespace; RedisConfig forbids unknown kwargs, so
    # a typo fails fast instead of silently sharing one keyspace.
    backend = RedisBackend(
        url=os.environ["REDIS_URL"],
        prefix="locus_notebook_65:",
    )
    # A backend is raw key/value storage; StorageBackendAdapter is what gives it
    # the Checkpointer contract the agent loop expects.
    return StorageBackendAdapter(backend)


_TRIAGE_PROMPT = (
    "You are an on-call incident triage copilot for an SRE team. Given an "
    "alert summary, say whether it warrants paging the on-call engineer and "
    "why. Be concise."
)


def _build_server() -> AgentServer:
    """Wrap the triage agent in an AgentServer (shared by the smoke test and the live boot)."""
    agent = Agent(
        config=AgentConfig(
            system_prompt=_TRIAGE_PROMPT,
            max_iterations=5,
            model=get_model(),
            # Redis checkpointer so /threads/{thread_id} survives restarts —
            # an incident outlives any one server process.
            checkpointer=_build_checkpointer(),
        )
    )
    return AgentServer(
        agent=agent,
        title="On-Call Incident Triage Copilot API",
        description="An incident-triage copilot exposed as an HTTP API",
    )


def _sse_frames(body: str) -> list[tuple[str | None, str]]:
    """``(id, type)`` per SSE frame of a /runs/{run_id}/events response."""
    frames = []
    for chunk in body.split("\n\n"):
        event_id, data = None, None
        for line in chunk.split("\n"):
            if line.startswith("id: "):
                event_id = line[4:]
            elif line.startswith("data: "):
                data = line[6:]
        if data is None:
            continue
        frames.append((event_id, data if data == "[DONE]" else json.loads(data)["type"]))
    return frames


def _show_resumable_run(client) -> None:
    """Start a turn in the background, follow it, drop the stream, resume from the last id.

    /stream holds the connection for the whole run; a turn started with
    POST /threads/{thread_id}/runs is followed with GET /runs/{run_id}/events
    instead, and a client that reconnects with Last-Event-ID gets exactly the
    events it missed. The event history behind the replay is bounded
    (RunRegistry: 500 events per run, 200 finished runs, one hour).
    """
    r = client.post(
        "/threads/inc-4712/runs",
        json={"prompt": "Alert: error rate 12% on checkout-api since 09:40. Page on-call?"},
    )
    run_id = r.json()["run_id"]
    print(f"POST /threads/inc-4712/runs: status={r.status_code} run status={r.json()['status']}")

    # Follow the run. This smoke test reads the stream to its end and keeps the
    # first two frames, as a client whose connection dropped after them would.
    frames = _sse_frames(client.get(f"/runs/{run_id}/events").text)
    seen = frames[:2]
    print(f"GET /runs/{{run_id}}/events: {len(frames) - 1} events, first two: {seen}")

    # Reconnect from the last id seen: the rest of the run, once, then [DONE].
    last_id = seen[-1][0]
    resumed = _sse_frames(
        client.get(f"/runs/{run_id}/events", headers={"Last-Event-ID": last_id}).text
    )
    print(f"GET /runs/{{run_id}}/events with Last-Event-ID: {last_id}: {resumed}")

    status = client.get(f"/runs/{run_id}").json()
    print(
        f"GET /runs/{{run_id}}: status={status['status']}, events={status['event_count']}, "
        f"stop_reason={status['stop_reason']}"
    )
    thread = client.get("/threads/inc-4712").json()
    print(f"GET /threads/inc-4712: source={thread['source']}, runs={len(thread['runs'])}")
    client.delete("/threads/inc-4712")


# Smoke test the server with FastAPI's TestClient. No port is bound.


def example_server():
    """Create an incident-triage copilot server with health, invoke, and stream endpoints."""
    print("=== On-Call Incident Triage Copilot Server ===\n")

    server = _build_server()

    from fastapi.testclient import TestClient

    # ``with``: background runs (POST /threads/{thread_id}/runs) need the
    # client's event loop to outlive the request that started them.
    with TestClient(server.app) as client:
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

        print("\n--- A turn in the background, followed, dropped and resumed ---")
        _show_resumable_run(client)

    print("\nTo run as a real server, set LOCUS_NOTEBOOK_BOOT=1 and run this")
    print("file directly. Example session:")
    print("  LOCUS_NOTEBOOK_BOOT=1 LOCUS_MODEL_PROVIDER=openai \\")
    print("      python examples/notebook_65_agent_server.py")
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

    Gated behind LOCUS_NOTEBOOK_BOOT=1 so the integration runner that
    imports every notebook doesn't hang on a blocking server.
    """
    server = _build_server()
    print("Booting AgentServer on http://127.0.0.1:8000 — Ctrl-C to stop.")
    print("Try: curl -X POST http://127.0.0.1:8000/invoke \\")
    print("          -H 'Content-Type: application/json' \\")
    print(
        '          -d \'{"prompt":"Page on-call? CrashLoopBackOff on payments-worker, '
        "12 restarts.\"}'"
    )
    server.run(host="127.0.0.1", port=8000)


if __name__ == "__main__":
    missing = _missing_env()
    if missing:
        print("\n--- Notebook 65: Agent Server (Redis-backed incident threads) ---")
        print(
            "Required environment variables not set; skipping the live "
            "demo so this file still runs cleanly in CI.\n"
        )
        for name in missing:
            print(f"  - {name}")
        print("\nStart a Redis instance, set REDIS_URL, and re-run.")
        sys.exit(0)
    if os.getenv("LOCUS_NOTEBOOK_BOOT") == "1":
        boot_live_server()
    else:
        example_server()
