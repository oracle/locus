# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Realtime voice sessions against a scripted connection.

What matters: a tool the model calls by voice runs through the same gate as in
text, a held action is spoken back as pending and never performed, and a broken
call is reported to the model instead of ending the session.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest

from tulip.control import Action, ControlPolicy, InMemoryApprovals, gate_tool
from tulip.tools.decorator import tool
from tulip.voice import realtime as realtime_module
from tulip.voice.realtime import (
    ActionHeld,
    AudioDelta,
    RealtimeConnection,
    RealtimeSession,
    ResponseDone,
    SessionError,
    ToolRan,
    Transcript,
    connect_openai_realtime,
    realtime_tool_schema,
)


class _Socket:
    """Replays server events and records what the session sends."""

    def __init__(self, incoming: list[Any], *, close_with: BaseException | None = None) -> None:
        self.incoming = list(incoming)
        self.sent: list[dict[str, Any]] = []
        self.close_with = close_with

    async def send(self, event: dict[str, Any]) -> None:
        self.sent.append(event)

    async def recv(self) -> Any:
        if self.incoming:
            return self.incoming.pop(0)
        if self.close_with is not None:
            raise self.close_with
        return None


def _call(name: str, arguments: Any, call_id: str = "call-1") -> dict[str, Any]:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {
        "type": "response.function_call_arguments.done",
        "name": name,
        "call_id": call_id,
        "arguments": raw,
    }


async def _collect(session: RealtimeSession) -> list[Any]:
    return [event async for event in session.events()]


def _outputs(socket: _Socket) -> list[str]:
    return [
        e["item"]["output"]
        for e in socket.sent
        if e.get("item", {}).get("type") == "function_call_output"
    ]


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order."""
    return f"order {order_id}: delivered, damaged"


def test_a_connection_is_anything_with_send_and_recv() -> None:
    assert isinstance(_Socket([]), RealtimeConnection)


@pytest.mark.asyncio
async def test_start_declares_instructions_voice_and_tools() -> None:
    socket = _Socket([])
    session = RealtimeSession(
        socket, tools=[lookup_order], instructions="You handle refunds.", voice="marin"
    )

    await session.start()

    [update] = socket.sent
    assert update["type"] == "session.update"
    assert update["session"]["instructions"] == "You handle refunds."
    assert update["session"]["audio"] == {"output": {"voice": "marin"}}
    [declared] = update["session"]["tools"]
    assert declared == realtime_tool_schema(lookup_order)
    assert declared["name"] == "lookup_order"
    assert "order_id" in declared["parameters"]["properties"]


@pytest.mark.asyncio
async def test_audio_text_and_transcripts_flow_both_ways() -> None:
    socket = _Socket(
        [
            {
                "type": "response.output_audio.delta",
                "delta": base64.b64encode(b"\x01\x02").decode(),
            },
            {"type": "response.audio.delta", "delta": base64.b64encode(b"\x03").decode()},
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "refund it",
            },
            {"type": "response.output_audio_transcript.done", "transcript": "On it."},
            {"type": "response.done"},
            {"type": "session.created"},
        ]
    )
    session = RealtimeSession(socket)

    await session.send_audio(b"\x00\xff")
    await session.send_text("hello")
    events = await _collect(session)

    assert socket.sent[0] == {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(b"\x00\xff").decode(),
    }
    assert socket.sent[1]["item"]["content"] == [{"type": "input_text", "text": "hello"}]
    assert socket.sent[2] == {"type": "response.create"}
    assert events == [
        AudioDelta(b"\x01\x02"),
        AudioDelta(b"\x03"),
        Transcript("user", "refund it"),
        Transcript("assistant", "On it."),
        ResponseDone(),
    ]


@pytest.mark.asyncio
async def test_a_tool_call_runs_and_its_result_goes_back_to_the_model() -> None:
    socket = _Socket([_call("lookup_order", {"order_id": "4821"})])
    session = RealtimeSession(socket, tools=[lookup_order])

    [ran] = await _collect(session)

    assert ran == ToolRan("lookup_order", {"order_id": "4821"}, "order 4821: delivered, damaged")
    assert _outputs(socket) == ["order 4821: delivered, damaged"]
    assert socket.sent[-1] == {"type": "response.create"}
    assert socket.sent[-2]["item"]["call_id"] == "call-1"


@pytest.mark.asyncio
async def test_a_held_refund_is_spoken_back_as_pending_and_never_performed() -> None:
    refunds: list[float] = []

    @tool
    def issue_refund(order_id: str, amount_usd: float) -> str:
        """Refund an order."""
        refunds.append(amount_usd)
        return "refunded"

    store = InMemoryApprovals()
    gated = gate_tool(
        issue_refund,
        policy=ControlPolicy(
            require_verification_score=0.0, require_human_for=frozenset({"payment"})
        ),
        action=lambda name, args: Action(name=name, asset=args["order_id"], kind="payment"),
        approval=store,
        on_refusal="interrupt",
    )
    socket = _Socket([_call("issue_refund", {"order_id": "4821", "amount_usd": 80.0})])

    [held] = await _collect(RealtimeSession(socket, tools=[gated]))

    assert isinstance(held, ActionHeld)
    assert held.approval_id == store.pending()[0].approval_id
    assert held.arguments == {"order_id": "4821", "amount_usd": 80.0}
    assert refunds == []
    [told] = _outputs(socket)
    assert told.startswith("Not done yet")
    assert "pending" in told


@pytest.mark.asyncio
async def test_broken_calls_are_reported_to_the_model_not_raised() -> None:
    @tool
    def flaky(order_id: str) -> str:
        """Always fails."""
        raise RuntimeError("payment provider down")

    @tool
    def structured(order_id: str) -> dict[str, Any]:
        """Returns a mapping."""
        return {"order_id": order_id, "status": "ok"}

    socket = _Socket(
        [
            _call("no_such_tool", {}),
            _call("lookup_order", "{not json"),
            _call("lookup_order", "[1, 2]"),
            _call("flaky", {"order_id": "1"}),
            _call("structured", {"order_id": "2"}),
            {"type": "error", "error": {"message": "rate limited"}},
        ]
    )
    session = RealtimeSession(socket, tools=[lookup_order, flaky, structured])

    events = await _collect(session)

    assert isinstance(events[0], SessionError)
    assert "unknown tool" in events[0].message
    assert isinstance(events[1], SessionError)
    assert "malformed" in events[1].message
    assert isinstance(events[2], SessionError)
    assert isinstance(events[3], ToolRan)
    assert "payment provider down" in events[3].result
    assert isinstance(events[4], ToolRan)
    assert json.loads(events[4].result)["status"] == "ok"
    assert events[5] == SessionError("rate limited")
    outputs = _outputs(socket)
    assert len(outputs) == 5
    assert "no tool named" in outputs[0]


class ConnectionClosedError(Exception):
    """Stands in for websockets' clean-close exception."""


@pytest.mark.asyncio
async def test_a_closed_socket_ends_the_stream_and_other_errors_propagate() -> None:
    assert await _collect(RealtimeSession(_Socket([], close_with=ConnectionClosedError()))) == []
    with pytest.raises(RuntimeError, match="boom"):
        await _collect(RealtimeSession(_Socket([], close_with=RuntimeError("boom"))))


@pytest.mark.asyncio
async def test_connect_openai_realtime_opens_a_session_and_declares_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = _Socket([{"type": "response.done"}])
    opened: dict[str, Any] = {}

    class _Manager:
        async def __aenter__(self) -> _Socket:
            return socket

        async def __aexit__(self, *exc: object) -> None:
            opened["closed"] = True

    class _Realtime:
        def connect(self, *, model: str) -> _Manager:
            opened["model"] = model
            return _Manager()

    class _Client:
        def __init__(self, api_key: str | None = None) -> None:
            opened["api_key"] = api_key
            self.realtime = _Realtime()

    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", _Client)
    async with connect_openai_realtime(
        tools=[lookup_order], api_key="sk-test", model="gpt-realtime"
    ) as session:
        assert [e async for e in session.events()] == [ResponseDone()]

    assert opened == {"api_key": "sk-test", "model": "gpt-realtime", "closed": True}
    assert socket.sent[0]["type"] == "session.update"
    assert realtime_module.__all__
