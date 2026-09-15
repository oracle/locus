# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Gemini Live against a scripted live session.

What matters: Gemini's audio, transcripts and tool calls arrive as the same
realtime events as OpenAI's, a tool Gemini calls runs through the same gate,
and its result goes back to Gemini under the call's id and name.
"""

from __future__ import annotations

import base64
import contextlib
import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from tulip.control import Action, ControlPolicy, InMemoryApprovals, gate_tool
from tulip.tools.decorator import tool
from tulip.voice.gemini import (
    INPUT_AUDIO_MIME,
    GeminiLiveConnection,
    connect_gemini_live,
    gemini_live_config,
)
from tulip.voice.realtime import (
    ActionHeld,
    AudioDelta,
    RealtimeSession,
    ResponseDone,
    SessionError,
    ToolRan,
    Transcript,
)


class _Live:
    """A live session: each ``receive()`` replays one scripted turn."""

    def __init__(self, turns: list[list[Any]]) -> None:
        self.turns = list(turns)
        self.sent: list[tuple[str, dict[str, Any]]] = []

    async def send_realtime_input(self, **kwargs: Any) -> None:
        self.sent.append(("realtime_input", kwargs))

    async def send_client_content(self, **kwargs: Any) -> None:
        self.sent.append(("client_content", kwargs))

    async def send_tool_response(self, **kwargs: Any) -> None:
        self.sent.append(("tool_response", kwargs))

    def receive(self) -> Any:
        turn = self.turns.pop(0) if self.turns else []

        async def messages() -> Any:
            for message in turn:
                yield message

        return messages()


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order."""
    return f"order {order_id}: delivered, damaged"


def _tool_call(name: str, args: dict[str, Any], call_id: str | None = "fc-1") -> dict[str, Any]:
    return {"tool_call": {"function_calls": [{"id": call_id, "name": name, "args": args}]}}


async def _collect(session: RealtimeSession) -> list[Any]:
    return [event async for event in session.events()]


def test_config_declares_tools_instructions_and_voice() -> None:
    config = gemini_live_config(tools=[lookup_order], instructions="Be brief.", voice="Kore")
    assert config["response_modalities"] == ["AUDIO"]
    assert config["system_instruction"] == "Be brief."
    declaration = config["tools"][0]["function_declarations"][0]
    assert declaration["name"] == "lookup_order"
    assert declaration["parameters_json_schema"]["properties"]["order_id"]["type"] == "string"
    assert config["speech_config"]["voice_config"]["prebuilt_voice_config"] == {
        "voice_name": "Kore"
    }
    bare = gemini_live_config()
    assert "tools" not in bare
    assert "system_instruction" not in bare
    assert "speech_config" not in bare


def test_config_is_a_valid_gemini_connect_config() -> None:
    types = pytest.importorskip("google.genai.types")
    config = gemini_live_config(tools=[lookup_order], instructions="Be brief.", voice="Kore")
    parsed = types.LiveConnectConfig.model_validate(config)
    assert parsed.tools[0].function_declarations[0].name == "lookup_order"


@pytest.mark.asyncio
async def test_audio_text_and_nothing_for_session_events() -> None:
    live = _Live([])
    session = RealtimeSession(GeminiLiveConnection(live), tools=[lookup_order])
    await session.start()
    await session.send_audio(b"\x01\x02")
    await session.send_text("where is my order?")
    assert live.sent == [
        ("realtime_input", {"audio": {"data": b"\x01\x02", "mime_type": INPUT_AUDIO_MIME}}),
        (
            "client_content",
            {
                "turns": {"role": "user", "parts": [{"text": "where is my order?"}]},
                "turn_complete": True,
            },
        ),
    ]


@pytest.mark.asyncio
async def test_audio_transcripts_and_turn_end() -> None:
    live = _Live(
        [
            [
                {"server_content": {"input_transcription": {"text": "where is "}}},
                {"server_content": {"input_transcription": {"text": "my order"}}},
                SimpleNamespace(
                    server_content=SimpleNamespace(
                        model_turn=SimpleNamespace(
                            parts=[
                                SimpleNamespace(
                                    inline_data=SimpleNamespace(data=b"pcm"),
                                    text=None,
                                    thought=None,
                                ),
                                SimpleNamespace(inline_data=None, text="thinking", thought=True),
                            ]
                        ),
                        output_transcription=SimpleNamespace(text="It shipped."),
                        input_transcription=None,
                        turn_complete=None,
                        interrupted=None,
                    ),
                    tool_call=None,
                    go_away=None,
                ),
                {"server_content": {"turn_complete": True}},
            ]
        ]
    )
    events = await _collect(RealtimeSession(GeminiLiveConnection(live)))
    assert events == [
        Transcript("user", "where is my order"),
        AudioDelta(b"pcm"),
        Transcript("assistant", "It shipped."),
        ResponseDone(),
    ]


@pytest.mark.asyncio
async def test_text_parts_base64_audio_and_interruption() -> None:
    live = _Live(
        [
            [
                {
                    "server_content": {
                        "model_turn": {
                            "parts": [
                                {"inline_data": {"data": base64.b64encode(b"pcm").decode()}},
                                {"text": "Hello"},
                            ]
                        }
                    }
                },
                {"server_content": {"interrupted": True}},
            ]
        ]
    )
    events = await _collect(RealtimeSession(GeminiLiveConnection(live)))
    assert events == [AudioDelta(b"pcm"), Transcript("assistant", "Hello")]


@pytest.mark.asyncio
async def test_a_tool_call_runs_and_answers_gemini_across_turns() -> None:
    live = _Live(
        [
            [
                {"server_content": {"input_transcription": {"text": "order 4821?"}}},
                _tool_call("lookup_order", {"order_id": "4821"}),
            ],
            [
                {"server_content": {"output_transcription": {"text": "It arrived damaged."}}},
                {"server_content": {"turn_complete": True}},
            ],
        ]
    )
    session = RealtimeSession(GeminiLiveConnection(live), tools=[lookup_order])
    events = await _collect(session)
    assert events == [
        Transcript("user", "order 4821?"),
        ToolRan("lookup_order", {"order_id": "4821"}, "order 4821: delivered, damaged"),
        Transcript("assistant", "It arrived damaged."),
        ResponseDone(),
    ]
    assert live.sent == [
        (
            "tool_response",
            {
                "function_responses": [
                    {
                        "name": "lookup_order",
                        "response": {"output": "order 4821: delivered, damaged"},
                        "id": "fc-1",
                    }
                ]
            },
        )
    ]


@pytest.mark.asyncio
async def test_a_held_refund_is_never_performed() -> None:
    refunds: list[str] = []

    @tool
    def issue_refund(order_id: str, amount_usd: float) -> str:
        """Refund an order."""
        refunds.append(order_id)
        return "refunded"

    gated = gate_tool(
        issue_refund,
        policy=ControlPolicy(
            require_verification_score=0.0, require_human_for=frozenset({"payment"})
        ),
        action=lambda name, args: Action(name=name, asset=args["order_id"], kind="payment"),
        approval=InMemoryApprovals(),
        on_refusal="interrupt",
    )
    live = _Live([[_tool_call("issue_refund", {"order_id": "4821", "amount_usd": 80.0}, None)]])
    events = await _collect(RealtimeSession(GeminiLiveConnection(live), tools=[gated]))
    assert refunds == []
    assert isinstance(events[0], ActionHeld)
    assert events[0].approval_id
    response = live.sent[0][1]["function_responses"][0]
    assert "id" not in response  # Gemini sent no id, so none goes back
    assert response["name"] == "issue_refund"
    assert "waiting for a person to approve" in response["response"]["output"]


@pytest.mark.asyncio
async def test_go_away_is_reported() -> None:
    live = _Live([[{"go_away": {"time_left": "10s"}}]])
    events = await _collect(RealtimeSession(GeminiLiveConnection(live)))
    assert events == [SessionError("Gemini is closing the session (time left: 10s)")]


@pytest.mark.asyncio
async def test_connect_opens_a_configured_session(monkeypatch: pytest.MonkeyPatch) -> None:
    live = _Live([[{"server_content": {"turn_complete": True}}]])
    opened: dict[str, Any] = {}

    class _Client:
        def __init__(self, api_key: str | None = None) -> None:
            opened["api_key"] = api_key

            @contextlib.asynccontextmanager
            async def connect(*, model: str, config: dict[str, Any]) -> Any:
                opened.update(model=model, config=config)
                yield live

            self.aio = SimpleNamespace(live=SimpleNamespace(connect=connect))

    google = ModuleType("google")
    genai = ModuleType("google.genai")
    genai.Client = _Client  # type: ignore[attr-defined]
    google.genai = genai  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)

    async with connect_gemini_live(
        model="gemini-live-test",
        tools=[lookup_order],
        instructions="Be brief.",
        api_key="k",
        config={"temperature": 0.2},
    ) as session:
        assert [event async for event in session.events()] == [ResponseDone()]
    assert opened["model"] == "gemini-live-test"
    assert opened["api_key"] == "k"
    assert opened["config"]["temperature"] == 0.2
    assert opened["config"]["tools"][0]["function_declarations"][0]["name"] == "lookup_order"
    assert json.dumps(opened["config"])
