# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``WS /runs/live``: a user turn, an approval answer, cancel, subscribe, auth."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest


pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from locus.agent import Agent
from locus.control import Action, ControlPolicy, InMemoryApprovals, gate_tool
from locus.core.events import TerminateEvent, ThinkEvent
from locus.memory.backends.memory import MemoryCheckpointer
from locus.server import AgentServer
from locus.testing import ScriptedModel, text, tool_call
from locus.tools.decorator import tool


calls: list[str] = []


@tool
def book_hotel(hotel_id: str, nights: int) -> str:
    """Book a hotel."""
    calls.append(hotel_id)
    return f"booked {hotel_id} x{nights}"


@pytest.fixture(autouse=True)
def _reset() -> None:
    calls.clear()


def _gated_server(model: ScriptedModel, **kwargs: Any) -> tuple[AgentServer, InMemoryApprovals]:
    store = InMemoryApprovals()
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
    agent = Agent(
        model=model,
        tools=[gated],
        checkpointer=MemoryCheckpointer(),
        reflexion=False,
        grounding=False,
    )

    def decision_handler(connection: Any, principal: str, thread_id: str, decision: dict) -> None:
        store.decide(decision["approval_id"], decision["verdict"], by=principal)

    return AgentServer(agent=agent, decision_handler=decision_handler, **kwargs), store


def _until(ws: Any, kind: str) -> list[dict[str, Any]]:
    """Frames up to and including the first of type ``kind``."""
    out: list[dict[str, Any]] = []
    for _ in range(200):
        frame = ws.receive_json()
        out.append(frame)
        if frame["type"] == kind:
            return out
    raise AssertionError(f"no {kind} frame in {out}")


def test_user_turn_then_approval_answer_on_one_socket() -> None:
    server, _store = _gated_server(
        ScriptedModel(
            [tool_call("book_hotel", call_id="c1", hotel_id="h1", nights=2), text("Confirmed.")]
        ),
        allow_unauthenticated=True,
        stream_tokens=False,
    )
    with TestClient(server.app) as client, client.websocket_connect("/runs/live") as ws:
        ws.send_json({"type": "ping", "ref": 0})
        assert ws.receive_json() == {"type": "pong", "ref": 0}

        ws.send_json({"type": "run", "thread_id": "A", "prompt": "book h1", "ref": 1})
        frames = _until(ws, "run_finished")
        assert frames[0]["type"] == "run_started"
        assert frames[0]["thread_id"] == "A"
        assert frames[0]["ref"] == 1
        run_id = frames[0]["run_id"]
        events = [f for f in frames if f["type"] == "event"]
        assert [f["seq"] for f in events] == [1, 2, 3]
        assert [f["event"]["type"] for f in events] == ["think", "tool_start", "interrupt"]
        assert all(f["run_id"] == run_id for f in events)
        assert frames[-1] == {
            "type": "run_finished",
            "run_id": run_id,
            "thread_id": "A",
            "status": "interrupted",
            "ref": 1,
        }
        assert calls == []
        approval_id = events[-1]["event"]["metadata"]["approval_id"]

        # Still undecided: an error frame, the thread stays paused.
        ws.send_json({"type": "resume", "thread_id": "A", "ref": 2})
        pending = ws.receive_json()
        assert pending["type"] == "error"
        assert pending["code"] == "approval_pending"
        assert pending["interrupt_id"] == "c1"

        ws.send_json(
            {
                "type": "resume",
                "thread_id": "A",
                "decision": {"approval_id": approval_id, "verdict": "approved"},
                "ref": 3,
            }
        )
        frames = _until(ws, "run_finished")
        assert frames[0]["type"] == "run_started"
        kinds = [f["event"]["type"] for f in frames if f["type"] == "event"]
        assert kinds[:2] == ["tool_start", "tool_complete"]
        assert kinds[-1] == "done"
        assert frames[-1]["status"] == "completed"
        assert calls == ["h1"]

        # Re-subscribe to the first run from its second event.
        ws.send_json({"type": "subscribe", "run_id": run_id, "after": 2, "ref": 4})
        frames = _until(ws, "run_finished")
        assert [f["seq"] for f in frames if f["type"] == "event"] == [3]
        assert frames[-1]["status"] == "interrupted"

        ws.send_json({"type": "subscribe", "run_id": "nope"})
        assert ws.receive_json()["code"] == "not_found"
        ws.send_json({"type": "dance"})
        assert ws.receive_json()["code"] == "unknown_type"
        ws.send_json({"type": "run"})
        assert ws.receive_json()["code"] == "bad_request"
        ws.send_text("not json")
        assert ws.receive_json()["code"] == "bad_request"
        ws.send_text("[1, 2]")
        assert ws.receive_json()["code"] == "bad_request"


def test_cancel_over_the_socket_and_a_minted_thread() -> None:
    class _Slow:
        config = None

        async def run(self, prompt: str, **kwargs: Any) -> Any:
            for i in range(50):
                yield ThinkEvent(iteration=i, reasoning=str(i), tool_calls=[])
                await asyncio.sleep(0.02)
            yield TerminateEvent(
                reason="complete", iterations_used=50, final_confidence=1.0, total_tool_calls=0
            )

    server = AgentServer(agent=_Slow(), allow_unauthenticated=True)
    with TestClient(server.app) as client, client.websocket_connect("/runs/live") as ws:
        ws.send_json({"type": "run", "prompt": "go"})
        started = ws.receive_json()
        assert started["type"] == "run_started"
        assert len(started["thread_id"]) == 32  # minted
        first = ws.receive_json()
        assert first["type"] == "event"
        assert first["seq"] == 1
        ws.send_json({"type": "cancel", "run_id": started["run_id"], "ref": 9})
        frames = _until(ws, "run_finished")
        cancelled = next(f for f in frames if f["type"] == "cancelled")
        assert cancelled["accepted"] is True
        assert frames[-1]["status"] == "cancelled"
        events = [f["event"] for f in frames if f["type"] == "event"]
        assert events[-1] == {"type": "done", "message": "", "reason": "cancelled"}
        assert client.get(f"/runs/{started['run_id']}").json()["status"] == "cancelled"


def test_socket_auth_header_or_token_query() -> None:
    server, _ = _gated_server(ScriptedModel([text("hi")]), api_key="s3cret", stream_tokens=False)
    with TestClient(server.app) as client:
        for path in ("/runs/live", "/runs/live?token=wrong"):
            with pytest.raises(WebSocketDisconnect), client.websocket_connect(path) as ws:
                ws.receive_json()
        with client.websocket_connect(
            "/runs/live", headers={"Authorization": "Bearer s3cret"}
        ) as ws:
            ws.send_json({"type": "ping"})
            assert ws.receive_json()["type"] == "pong"
        with client.websocket_connect("/runs/live?token=s3cret") as ws:
            ws.send_json({"type": "run", "thread_id": "A", "prompt": "hello"})
            frames = _until(ws, "run_finished")
            assert frames[-1]["status"] == "completed"
        # The turn ran under the key's principal, not ``anon``.
        assert server.runs.get_run("anon", frames[0]["run_id"]) is None
        assert (
            client.get(
                f"/runs/{frames[0]['run_id']}", headers={"Authorization": "Bearer s3cret"}
            ).status_code
            == 200
        )
