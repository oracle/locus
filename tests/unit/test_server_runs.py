# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Threads, background runs and resumable SSE on ``AgentServer``.

``with TestClient(app) as client`` keeps one event loop alive across
requests, which is what lets a run started by ``POST /threads/{id}/runs``
keep going while the test polls ``GET /runs/{id}`` and streams
``GET /runs/{id}/events``.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest


pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from locus.agent import Agent
from locus.control import Action, ControlPolicy, InMemoryApprovals, gate_tool
from locus.core.events import TerminateEvent, ThinkEvent
from locus.memory.backends.memory import MemoryCheckpointer
from locus.server import AgentServer, RunRegistry
from locus.server.runs import FINISHED_STATUSES
from locus.testing import ScriptedModel, text, tool_call
from locus.tools.decorator import tool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _frames(body: str) -> list[tuple[str | None, Any]]:
    """``(id, data)`` per SSE message; ``data`` is parsed unless it is ``[DONE]``."""
    out: list[tuple[str | None, Any]] = []
    for chunk in body.split("\n\n"):
        if not chunk.strip():
            continue
        event_id = None
        data = None
        for line in chunk.split("\n"):
            if line.startswith("id: "):
                event_id = line[4:]
            elif line.startswith("data: "):
                data = line[6:]
        assert data is not None, chunk
        out.append((event_id, data if data == "[DONE]" else json.loads(data)))
    return out


def _wait(client: TestClient, run_id: str, timeout: float = 5.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get(f"/runs/{run_id}").json()
        if status["status"] in FINISHED_STATUSES:
            return status
        time.sleep(0.01)
    raise AssertionError(f"run {run_id} did not finish")


class _Slow:
    """Yields ``count`` think events ``delay`` apart, then a terminate event."""

    config = None

    def __init__(self, count: int = 5, delay: float = 0.03) -> None:
        self.count = count
        self.delay = delay
        self.seen: list[dict[str, Any]] = []

    async def run(self, prompt: str, **kwargs: Any) -> Any:
        self.seen.append({"prompt": prompt, **kwargs})
        for i in range(self.count):
            yield ThinkEvent(iteration=i, reasoning=f"step {i}", tool_calls=[])
            await asyncio.sleep(self.delay)
        yield TerminateEvent(
            reason="complete",
            iterations_used=self.count,
            final_confidence=1.0,
            total_tool_calls=0,
            final_message="all done",
        )


calls: list[str] = []
gate = asyncio.Event()


@tool
def book_hotel(hotel_id: str, nights: int) -> str:
    """Book a hotel."""
    calls.append(hotel_id)
    return f"booked {hotel_id} x{nights}"


@tool
async def slow_lookup(query: str) -> str:
    """Wait for the test to release the gate."""
    await gate.wait()
    return f"looked up {query}"


@pytest.fixture(autouse=True)
def _reset() -> None:
    calls.clear()
    gate.clear()


def _gated_agent(store: InMemoryApprovals, model: ScriptedModel) -> Agent:
    gated = gate_tool(
        book_hotel,
        policy=ControlPolicy(
            require_verification_score=0.0, require_human_for=frozenset({"production"})
        ),
        action=lambda n, kw: Action(name=n, asset=kw["hotel_id"], environment="production"),
        approval=store,
        on_refusal="interrupt",
        principal="u1",
    )
    return Agent(
        model=model,
        tools=[gated],
        checkpointer=MemoryCheckpointer(),
        reflexion=False,
        grounding=False,
    )


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------


def test_threads_are_created_listed_and_dropped_without_a_checkpointer() -> None:
    server = AgentServer(agent=_Slow(count=1, delay=0), allow_unauthenticated=True)
    with TestClient(server.app) as client:
        created = client.post("/threads", json={"thread_id": "t1", "metadata": {"case": 7}})
        assert created.status_code == 201
        assert created.json()["thread_id"] == "t1"
        assert created.json()["metadata"] == {"case": 7}
        assert client.post("/threads", json={"thread_id": "t1"}).status_code == 409

        minted = client.post("/threads", json={}).json()
        assert len(minted["thread_id"]) == 32

        shown = client.get("/threads/t1")
        assert shown.status_code == 200
        assert shown.json()["source"] == "registry"
        assert shown.json()["messages"] == []
        assert shown.json()["metadata"] == {"case": 7}
        assert shown.json()["runs"] == []

        missing = client.get("/threads/nope")
        assert missing.status_code == 404
        assert "checkpointer" in missing.json()["detail"].lower()

        assert client.delete("/threads/t1").json() == {"thread_id": "t1", "deleted": True}
        assert client.get("/threads/t1").status_code == 404
        assert client.delete("/threads/t1").status_code == 404


# ---------------------------------------------------------------------------
# Runs and the resumable event stream
# ---------------------------------------------------------------------------


def test_run_lifecycle_and_live_resume_with_last_event_id() -> None:
    agent = _Slow(count=5, delay=0.03)
    server = AgentServer(agent=agent, allow_unauthenticated=True)
    with TestClient(server.app) as client:
        started = client.post("/threads/t1/runs", json={"prompt": "go", "metadata": {"a": 1}})
        assert started.status_code == 202
        run_id = started.json()["run_id"]
        assert started.json() == {"run_id": run_id, "thread_id": "t1", "status": "queued"}

        # Follow the stream live, then drop it after two events.
        received: list[tuple[str | None, Any]] = []
        with client.stream("GET", f"/runs/{run_id}/events") as response:
            assert response.headers["content-type"].startswith("text/event-stream")
            buffer = ""
            for chunk in response.iter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    frame, buffer = buffer.split("\n\n", 1)
                    received.extend(_frames(frame + "\n\n"))
                if len(received) >= 2:
                    break
        assert [(i, d["type"]) for i, d in received[:2]] == [("1", "think"), ("2", "think")]

        final = _wait(client, run_id)
        assert final["status"] == "completed"
        assert final["event_count"] == 6
        assert final["message"] == "all done"
        assert final["stop_reason"] == "complete"

        # Reconnect: exactly the missed events, once, then [DONE].
        resumed = _frames(client.get(f"/runs/{run_id}/events", headers={"Last-Event-ID": "2"}).text)
        assert [(i, d if d == "[DONE]" else d["type"]) for i, d in resumed] == [
            ("3", "think"),
            ("4", "think"),
            ("5", "think"),
            ("6", "done"),
            (None, "[DONE]"),
        ]
        assert resumed[3][1] == {"type": "done", "message": "all done", "reason": "complete"}

        # Everything was delivered already: only the terminator remains.
        assert _frames(
            client.get(f"/runs/{run_id}/events", headers={"Last-Event-ID": "6"}).text
        ) == [(None, "[DONE]")]
        # A full replay from the start still works after the run finished.
        assert len(_frames(client.get(f"/runs/{run_id}/events").text)) == 7

        bad = client.get(f"/runs/{run_id}/events", headers={"Last-Event-ID": "two"})
        assert bad.status_code == 400
        assert client.get(f"/runs/{run_id}/events", params={"protocol": "nope"}).status_code == 400
        assert client.get("/runs/missing/events").status_code == 404
        assert client.get("/runs/missing").status_code == 404

        # The agent saw the scoped thread id, the metadata and the token flag
        # (its ``run`` takes ``**kwargs``, so the flag is offered as on ``/stream``).
        assert agent.seen == [
            {"prompt": "go", "thread_id": "anon:t1", "metadata": {"a": 1}, "stream_tokens": True}
        ]
        shown = client.get("/threads/t1").json()
        assert [r["run_id"] for r in shown["runs"]] == [run_id]
        assert shown["runs"][0]["status"] == "completed"


def test_replay_gap_when_history_is_shorter_than_the_run() -> None:
    server = AgentServer(
        agent=_Slow(count=5, delay=0),
        allow_unauthenticated=True,
        run_registry=RunRegistry(history_per_run=3),
    )
    with TestClient(server.app) as client:
        run_id = client.post("/threads/t/runs", json={"prompt": "go"}).json()["run_id"]
        _wait(client, run_id)
        frames = _frames(client.get(f"/runs/{run_id}/events").text)
        assert frames[0] == ("3", {"type": "replay_gap", "from": 1, "to": 3})
        assert [i for i, _ in frames[1:]] == ["4", "5", "6", None]
        # A client that already had event 4 is not told about the gap before it.
        later = _frames(client.get(f"/runs/{run_id}/events", headers={"Last-Event-ID": "4"}).text)
        assert [i for i, _ in later] == ["5", "6", None]


def test_one_run_at_a_time_per_thread_and_cancel_of_a_stub() -> None:
    server = AgentServer(agent=_Slow(count=50, delay=0.02), allow_unauthenticated=True)
    with TestClient(server.app) as client:
        run_id = client.post("/threads/t/runs", json={"prompt": "go"}).json()["run_id"]
        busy = client.post("/threads/t/runs", json={"prompt": "again"})
        assert busy.status_code == 409
        assert busy.json()["code"] == "run_in_progress"
        # Another thread is free to run.
        assert client.post("/threads/other/runs", json={"prompt": "go"}).status_code == 202

        cancelled = client.post(f"/runs/{run_id}/cancel").json()
        assert cancelled["accepted"] is True
        final = _wait(client, run_id)
        assert final["status"] == "cancelled"
        frames = _frames(client.get(f"/runs/{run_id}/events").text)
        assert frames[-2][1] == {"type": "done", "message": "", "reason": "cancelled"}
        assert frames[-1] == (None, "[DONE]")
        # Cancelling a finished run is a no-op, not an error.
        assert client.post(f"/runs/{run_id}/cancel").json()["accepted"] is False
        # The thread is free again.
        assert client.post("/threads/t/runs", json={"prompt": "again"}).status_code == 202


def test_cancel_asks_an_agent_to_stop_cooperatively() -> None:
    agent = Agent(
        model=ScriptedModel(
            [tool_call("slow_lookup", call_id="c1", query="x"), text("never reached")]
        ),
        tools=[slow_lookup],
        reflexion=False,
        grounding=False,
    )
    server = AgentServer(agent=agent, allow_unauthenticated=True, stream_tokens=False)
    with TestClient(server.app) as client:
        run_id = client.post("/threads/t/runs", json={"prompt": "look"}).json()["run_id"]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if client.get(f"/runs/{run_id}").json()["event_count"] >= 2:
                break
            time.sleep(0.01)
        assert client.post(f"/runs/{run_id}/cancel").json()["accepted"] is True
        # Release the tool; the loop notices the cancel on its next check.
        client.portal.call(gate.set)  # type: ignore[union-attr]
        final = _wait(client, run_id)
        assert final["status"] == "cancelled"
        assert final["stop_reason"] == "cancelled"
        types = [d["type"] for _, d in _frames(client.get(f"/runs/{run_id}/events").text)[:-1]]
        assert types[:2] == ["think", "tool_start"]
        assert types[-1] == "done"


def test_a_run_that_raises_ends_failed_with_a_sanitized_error() -> None:
    class _Broken:
        config = None

        async def run(self, prompt: str, **kwargs: Any) -> Any:
            yield ThinkEvent(iteration=0, reasoning="about to fail", tool_calls=[])
            raise RuntimeError("DSN=postgres://leak/secret")

    server = AgentServer(agent=_Broken(), allow_unauthenticated=True)
    with TestClient(server.app) as client:
        run_id = client.post("/threads/t/runs", json={"prompt": "go"}).json()["run_id"]
        final = _wait(client, run_id)
        assert final["status"] == "failed"
        assert final["error_correlation_id"]
        body = client.get(f"/runs/{run_id}/events").text
        assert "DSN=postgres" not in body
        frames = _frames(body)
        assert frames[1][1]["type"] == "error"
        assert frames[1][1]["error"] == "internal error"


def test_runs_are_scoped_to_the_principal() -> None:
    server = AgentServer(agent=_Slow(count=1, delay=0), api_key="alice")
    with TestClient(server.app) as client:
        headers = {"Authorization": "Bearer alice"}
        assert client.post("/threads", json={}).status_code == 401
        run_id = client.post("/threads/t/runs", json={"prompt": "go"}, headers=headers).json()[
            "run_id"
        ]
        assert client.get(f"/runs/{run_id}", headers=headers).status_code == 200
        assert client.get(f"/runs/{run_id}").status_code == 401
        # A different key would be a different principal on another server;
        # here, make sure the registry itself refuses a foreign principal.
        assert server.runs.get_run("someone-else", run_id) is None


# ---------------------------------------------------------------------------
# Continuing a paused thread through the runs resource
# ---------------------------------------------------------------------------


def test_resume_payload_continues_a_paused_thread() -> None:
    store = InMemoryApprovals()
    agent = _gated_agent(
        store,
        ScriptedModel(
            [tool_call("book_hotel", call_id="c1", hotel_id="h1", nights=2), text("Confirmed.")]
        ),
    )

    def decision_handler(request: Any, principal: str, thread_id: str, decision: dict) -> None:
        store.decide(decision["approval_id"], decision["verdict"], by=principal)

    server = AgentServer(agent=agent, allow_unauthenticated=True, decision_handler=decision_handler)
    with TestClient(server.app) as client:
        first = client.post("/threads/A/runs", json={"prompt": "book h1"}).json()["run_id"]
        paused = _wait(client, first)
        assert paused["status"] == "interrupted"
        assert paused["interrupt"]["interrupt_id"] == "c1"
        assert calls == []

        # Undecided: 409 with the approval details, nothing performed.
        pending = client.post("/threads/A/runs", json={"resume": {}})
        assert pending.status_code == 409
        assert pending.json()["code"] == "approval_pending"
        assert pending.json()["interrupt_id"] == "c1"

        approval_id = paused["interrupt"]["metadata"]["approval_id"]
        second = client.post(
            "/threads/A/runs",
            json={"resume": {"decision": {"approval_id": approval_id, "verdict": "approved"}}},
        )
        assert second.status_code == 202
        done = _wait(client, second.json()["run_id"])
        assert done["status"] == "completed"
        assert done["kind"] == "resume"
        assert done["message"] == "Confirmed."
        assert calls == ["h1"]
        frames = _frames(client.get(f"/runs/{done['run_id']}/events").text)
        assert [d["type"] for _, d in frames[:-1]][:2] == ["tool_start", "tool_complete"]

        # Nothing paused on an unknown thread; both at once is a 422.
        assert client.post("/threads/C/runs", json={"resume": {"response": "x"}}).status_code == 404
        both = client.post("/threads/A/runs", json={"prompt": "p", "resume": {"response": "x"}})
        assert both.status_code == 422
        assert client.post("/threads/A/runs", json={}).status_code == 422

        shown = client.get("/threads/A").json()
        assert shown["source"] == "checkpointer"
        assert [r["kind"] for r in shown["runs"]] == ["run", "resume"]


def test_resume_without_a_handler_rejects_decisions() -> None:
    server = AgentServer(
        agent=_gated_agent(InMemoryApprovals(), ScriptedModel([text("x")])),
        allow_unauthenticated=True,
    )
    with TestClient(server.app) as client:
        refused = client.post(
            "/threads/A/runs", json={"resume": {"decision": {"verdict": "approved"}}}
        )
        assert refused.status_code == 400
        assert refused.json()["code"] == "decisions_not_accepted"


# ---------------------------------------------------------------------------
# UI wire protocols over the same runs
# ---------------------------------------------------------------------------


def test_protocol_streams_resume_from_the_same_history() -> None:
    server = AgentServer(agent=_Slow(count=2, delay=0), allow_unauthenticated=True)
    with TestClient(server.app) as client:
        run_id = client.post("/threads/t/runs", json={"prompt": "go"}).json()["run_id"]
        _wait(client, run_id)
        agui = _frames(client.get(f"/runs/{run_id}/events", params={"protocol": "ag-ui"}).text)
        assert [(i, d if d == "[DONE]" else d["type"]) for i, d in agui] == [
            (None, "RUN_STARTED"),
            ("0", "STATE_SNAPSHOT"),
            (None, "TEXT_MESSAGE_START"),
            (None, "TEXT_MESSAGE_CONTENT"),
            (None, "TEXT_MESSAGE_END"),
            ("1", "STATE_DELTA"),
            (None, "TEXT_MESSAGE_START"),
            (None, "TEXT_MESSAGE_CONTENT"),
            (None, "TEXT_MESSAGE_END"),
            ("2", "STATE_DELTA"),
            (None, "TEXT_MESSAGE_START"),
            (None, "TEXT_MESSAGE_CONTENT"),
            (None, "TEXT_MESSAGE_END"),
            (None, "STATE_DELTA"),
            ("3", "RUN_FINISHED"),
            (None, "[DONE]"),
        ]
        assert agui[0][1] == {"type": "RUN_STARTED", "threadId": "t", "runId": run_id}
        # Resuming after event 1 skips the opening frames and the first message,
        # but the second message keeps the id it had on the full stream.
        later = _frames(
            client.get(
                f"/runs/{run_id}/events",
                params={"protocol": "ag-ui"},
                headers={"Last-Event-ID": "1"},
            ).text
        )
        assert later[0][1] == {
            "type": "TEXT_MESSAGE_START",
            "messageId": f"{run_id}-msg-2",
            "role": "assistant",
        }
        assert [i for i, _ in later] == [None, None, None, "2", None, None, None, None, "3", None]

        vercel = client.get(f"/runs/{run_id}/events", params={"protocol": "vercel-ai"})
        assert vercel.headers["x-vercel-ai-ui-message-stream"] == "v1"
        assert [d if d == "[DONE]" else d["type"] for _, d in _frames(vercel.text)] == [
            "start",
            "start-step",
            "text-start",
            "text-delta",
            "text-end",
            "text-start",
            "text-delta",
            "text-end",
            "text-start",
            "text-delta",
            "text-end",
            "finish-step",
            "finish",
            "[DONE]",
        ]


def test_ag_ui_and_vercel_ai_request_endpoints() -> None:
    server = AgentServer(agent=_Slow(count=1, delay=0), allow_unauthenticated=True)
    with TestClient(server.app) as client:
        response = client.post(
            "/ag-ui",
            json={
                "threadId": "t-agui",
                "runId": "run-agui",
                "messages": [{"id": "m1", "role": "user", "content": "hello"}],
                "forwardedProps": {"user": "alice"},
            },
        )
        assert response.status_code == 200
        frames = _frames(response.text)
        assert frames[0][1] == {"type": "RUN_STARTED", "threadId": "t-agui", "runId": "run-agui"}
        assert frames[-2][1]["type"] == "RUN_FINISHED"
        assert client.get("/runs/run-agui").json()["status"] == "completed"
        assert server.agent.seen[-1]["metadata"] == {"user": "alice"}  # type: ignore[attr-defined]
        # The same run id again is a conflict; no user message is a 422.
        again = client.post(
            "/ag-ui",
            json={
                "threadId": "t2",
                "runId": "run-agui",
                "messages": [{"role": "user", "content": "x"}],
            },
        )
        assert again.status_code == 409
        assert again.json()["code"] == "run_exists"
        assert client.post("/ag-ui", json={"threadId": "t3", "messages": []}).status_code == 422
        assert client.post("/ag-ui", content=b"[]").status_code == 400

        response = client.post(
            "/vercel-ai",
            json={
                "id": "chat-1",
                "messages": [
                    {"id": "u1", "role": "user", "parts": [{"type": "text", "text": "hello"}]}
                ],
            },
        )
        assert response.status_code == 200
        assert response.headers["x-vercel-ai-ui-message-stream"] == "v1"
        kinds = [d if d == "[DONE]" else d["type"] for _, d in _frames(response.text)]
        assert kinds[:2] == ["start", "start-step"]
        assert kinds[-3:] == ["finish-step", "finish", "[DONE]"]
        assert client.get("/threads/chat-1").json()["runs"][0]["status"] == "completed"


# ---------------------------------------------------------------------------
# Registry bounds
# ---------------------------------------------------------------------------


def test_registry_ttl_and_run_cap() -> None:
    now = [1000.0]
    registry = RunRegistry(ttl_seconds=60, max_runs=2, clock=lambda: now[0])
    server = AgentServer(
        agent=_Slow(count=1, delay=0), allow_unauthenticated=True, run_registry=registry
    )
    with TestClient(server.app) as client:
        ids = []
        for i in range(3):
            ids.append(client.post(f"/threads/t{i}/runs", json={"prompt": "go"}).json()["run_id"])
            _wait(client, ids[-1])
        # Three finished runs, cap of two: the oldest is gone.
        assert client.get(f"/runs/{ids[0]}").status_code == 404
        assert client.get(f"/runs/{ids[1]}").status_code == 200
        assert client.get(f"/runs/{ids[2]}").status_code == 200
        now[0] += 61
        assert client.get(f"/runs/{ids[2]}").status_code == 404
        # Idle threads expire with the TTL too (no checkpointer: nothing else knows them).
        assert client.get("/threads/t2").status_code == 404


def test_registry_rejects_nonsense_bounds() -> None:
    with pytest.raises(ValueError, match="ttl_seconds"):
        RunRegistry(ttl_seconds=0)
    with pytest.raises(ValueError, match="max_runs"):
        RunRegistry(max_runs=0)
    with pytest.raises(ValueError, match="history_per_run"):
        RunRegistry(history_per_run=0)
