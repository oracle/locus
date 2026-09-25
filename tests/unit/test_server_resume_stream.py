# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""AgentServer: token streaming, JSON events, /resume by thread, trusted metadata."""

from __future__ import annotations

import json
from typing import Any

import pytest


pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from tulip.agent import Agent
from tulip.control import Action, ControlPolicy, InMemoryApprovals, gate_tool
from tulip.hooks.provider import HookProvider
from tulip.memory.backends.memory import MemoryCheckpointer
from tulip.server import AgentServer
from tulip.testing import ScriptedModel, text, tool_call
from tulip.tools.decorator import tool


calls: list[dict[str, Any]] = []


@tool
def book_hotel(hotel_id: str, nights: int) -> str:
    """Book a hotel."""
    calls.append({"hotel_id": hotel_id, "nights": nights})
    return f"booked {hotel_id} x{nights}"


@pytest.fixture(autouse=True)
def _reset() -> None:
    calls.clear()


def _parse(body: str) -> list[dict[str, Any]]:
    out = []
    for chunk in body.split("\n\n"):
        if chunk.startswith("data: ") and chunk.strip() != "data: [DONE]":
            out.append(json.loads(chunk.removeprefix("data: ")))
    return out


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


def test_stream_sends_token_deltas_and_json_interrupts() -> None:
    store = InMemoryApprovals()
    agent = _gated_agent(
        store,
        ScriptedModel(
            [
                tool_call("book_hotel", call_id="c1", hotel_id="h1", nights=2),
                text("Your booking is confirmed."),
            ]
        ),
    )
    client = TestClient(AgentServer(agent=agent, allow_unauthenticated=True).app)
    events = _parse(client.post("/stream", json={"prompt": "book", "thread_id": "t1"}).text)

    assert [e["type"] for e in events][-1] == "interrupt"
    interrupt = events[-1]
    # A JSON object, not a Python repr string.
    assert interrupt["interrupt_id"] == "c1"
    assert interrupt["metadata"]["arguments"] == {"hotel_id": "h1", "nights": 2}
    assert interrupt["question"].startswith("Approve book_hotel")

    streamed = Agent(model=ScriptedModel([text("Hello there, streamed.")]), reflexion=False)
    body = TestClient(AgentServer(agent=streamed, allow_unauthenticated=True).app).post(
        "/stream", json={"prompt": "hi"}
    )
    events = _parse(body.text)
    chunks = [e["content"] for e in events if e["type"] == "model_chunk" and e["content"]]
    assert "".join(chunks) == "Hello there, streamed."
    assert events[-1] == {"type": "done", "message": "Hello there, streamed.", "reason": "complete"}


def test_stream_tokens_can_be_turned_off() -> None:
    agent = Agent(model=ScriptedModel([text("plain")]), reflexion=False)
    server = AgentServer(agent=agent, allow_unauthenticated=True, stream_tokens=False)
    events = _parse(TestClient(server.app).post("/stream", json={"prompt": "hi"}).text)
    assert [e["type"] for e in events] == ["think", "done"]


def test_resume_endpoint_is_strictly_per_thread() -> None:
    store = InMemoryApprovals()
    model = ScriptedModel(
        [tool_call("book_hotel", call_id="cA", hotel_id="hA", nights=1)], repeat_last=True
    )
    agent = _gated_agent(store, model)
    decided: list[tuple[str, str, dict[str, Any]]] = []

    def decision_handler(request: Any, principal: str, thread_id: str, decision: dict) -> None:
        decided.append((principal, thread_id, decision))
        store.decide(decision["approval_id"], decision["verdict"], by=principal)

    client = TestClient(
        AgentServer(agent=agent, allow_unauthenticated=True, decision_handler=decision_handler).app
    )
    first_a = _parse(client.post("/stream", json={"prompt": "A", "thread_id": "A"}).text)
    model._turns = [tool_call("book_hotel", call_id="cB", hotel_id="hB", nights=9)]
    _parse(client.post("/stream", json={"prompt": "B", "thread_id": "B"}).text)
    model._turns = [text("done")]

    # Undecided: 409, nothing performed, thread still paused.
    pending = client.post("/resume", json={"thread_id": "A"})
    assert pending.status_code == 409
    assert pending.json()["type"] == "approval_pending"
    assert pending.json()["interrupt_id"] == "cA"
    assert calls == []

    approval_a = first_a[-1]["metadata"]["approval_id"]
    resumed = client.post(
        "/resume",
        json={"thread_id": "A", "decision": {"approval_id": approval_a, "verdict": "approved"}},
    )
    assert resumed.status_code == 200
    events = _parse(resumed.text)
    assert [c["hotel_id"] for c in calls] == ["hA"]
    assert [e["tool_call_id"] for e in events if e["type"] == "tool_complete"] == ["cA"]
    assert events[-1]["type"] == "done"
    assert decided == [("anon", "A", {"approval_id": approval_a, "verdict": "approved"})]

    # Nothing left to resume on A; C never existed.
    assert client.post("/resume", json={"thread_id": "C"}).status_code == 404


def test_resume_rejects_decisions_without_a_handler() -> None:
    agent = Agent(model=ScriptedModel([text("x")]), reflexion=False)
    client = TestClient(AgentServer(agent=agent, allow_unauthenticated=True).app)
    r = client.post("/resume", json={"thread_id": "A", "decision": {"verdict": "approved"}})
    assert r.status_code == 400


def test_metadata_resolver_wins_over_client_metadata() -> None:
    store = InMemoryApprovals()
    agent = _gated_agent(
        store,
        ScriptedModel(
            [tool_call("book_hotel", call_id="c1", hotel_id="h1", nights=2), text("ok")],
        ),
    )

    async def resolver(request: Any, principal: str) -> dict[str, Any]:
        return {"user": request.headers.get("x-session-user", "unknown")}

    client = TestClient(
        AgentServer(agent=agent, allow_unauthenticated=True, metadata_resolver=resolver).app
    )
    client.post(
        "/stream",
        json={"prompt": "book", "thread_id": "t", "metadata": {"user": "mallory", "lang": "en"}},
        headers={"x-session-user": "alice"},
    )
    state_meta = agent._interrupts["anon:t"].metadata
    assert state_meta == {"user": "alice", "lang": "en"}

    [record] = store.pending()
    store.decide(record.approval_id, "approved", by="alice")
    seen: list[Any] = []

    class _Spy(HookProvider):
        @property
        def priority(self) -> int:
            return 100

        async def on_before_tool_call(self, event: Any) -> None:
            seen.append(dict(event.run.metadata))

    agent._hooks.append(_Spy())
    client.post(
        "/resume",
        json={"thread_id": "t", "metadata": {"user": "mallory"}},
        headers={"x-session-user": "alice"},
    )
    assert [c["hotel_id"] for c in calls] == ["h1"]
    assert seen == [{"user": "alice"}]


def test_invoke_applies_the_resolver_too() -> None:
    seen: list[dict[str, Any]] = []

    class _Stub:
        config = None

        async def run(self, prompt: str, **kwargs: Any) -> Any:
            seen.append(kwargs["metadata"])
            if False:  # pragma: no cover
                yield None

    client = TestClient(
        AgentServer(
            agent=_Stub(),
            allow_unauthenticated=True,
            metadata_resolver=lambda request, principal: {"user": principal},
        ).app
    )
    client.post("/invoke", json={"prompt": "x", "metadata": {"user": "mallory"}})
    assert seen == [{"user": "anon"}]


def test_custom_events_are_serialised_with_the_client_thread_id() -> None:
    from tulip.core.events import CustomEvent

    class _Stub:
        config = None

        async def run(self, prompt: str, **kwargs: Any) -> Any:
            yield CustomEvent(name="hotel_list", data={"n": 1}, thread_id=kwargs["thread_id"])

    client = TestClient(AgentServer(agent=_Stub(), allow_unauthenticated=True).app)
    events = _parse(client.post("/stream", json={"prompt": "x", "thread_id": "mine"}).text)
    assert events[0]["type"] == "custom"
    assert events[0]["name"] == "hotel_list"
    assert events[0]["data"] == {"n": 1}
    assert events[0]["thread_id"] == "mine"


def test_accepts_kwarg_handles_uninspectable_callables() -> None:
    from tulip.server.app import _accepts_kwarg

    assert _accepts_kwarg(1, "stream_tokens") is False
    assert _accepts_kwarg(lambda *, stream_tokens=False: None, "stream_tokens") is True
    assert _accepts_kwarg(lambda **kw: None, "stream_tokens") is True
    assert _accepts_kwarg(lambda prompt: None, "stream_tokens") is False


def test_stream_sanitises_a_run_that_raises_before_iterating() -> None:
    class _Broken:
        config = None

        def run(self, prompt: str, **kwargs: Any) -> Any:
            raise RuntimeError("DSN=postgres://leak/secret")

    client = TestClient(AgentServer(agent=_Broken(), allow_unauthenticated=True).app)
    body = client.post("/stream", json={"prompt": "x"}).text
    assert "DSN=postgres" not in body
    assert _parse(body)[0]["type"] == "error"
    assert body.rstrip().endswith("data: [DONE]")


def test_resume_edge_cases() -> None:
    class _NoResume:
        config = None

        async def run(self, prompt: str, **kwargs: Any) -> Any:
            if False:  # pragma: no cover
                yield None

    client = TestClient(AgentServer(agent=_NoResume(), allow_unauthenticated=True).app)
    assert client.post("/resume", json={"thread_id": "t"}).status_code == 404

    seen: list[dict[str, Any]] = []

    class _EmptyResume(_NoResume):
        async def resume(self, response: str, **kwargs: Any) -> Any:
            seen.append(kwargs)
            if False:  # pragma: no cover
                yield None

    async def handler(request: Any, principal: str, thread_id: str, decision: dict) -> None:
        seen.append({"decision": decision})

    client = TestClient(
        AgentServer(agent=_EmptyResume(), allow_unauthenticated=True, decision_handler=handler).app
    )
    r = client.post("/resume", json={"thread_id": "t", "decision": {"verdict": "approved"}})
    assert r.status_code == 200
    assert _parse(r.text) == []
    assert seen == [
        {"decision": {"verdict": "approved"}},
        {"thread_id": "anon:t", "perform_dangling": True},
    ]

    class _Exploding(_NoResume):
        async def resume(self, response: str, **kwargs: Any) -> Any:
            raise RuntimeError("database is on fire")
            yield None  # pragma: no cover

    client = TestClient(
        AgentServer(agent=_Exploding(), allow_unauthenticated=True).app,
        raise_server_exceptions=False,
    )
    assert client.post("/resume", json={"thread_id": "t"}).status_code == 500


def test_non_pydantic_events_fall_back_to_their_string_form() -> None:
    from tulip.server.app import _event_payload

    class _Odd:
        event_type = "odd"

        def __str__(self) -> str:
            return "odd-event"

    payload = _event_payload(_Odd(), scoped_thread_id=None, thread_id=None)
    assert payload == {"type": "odd", "data": "odd-event"}
