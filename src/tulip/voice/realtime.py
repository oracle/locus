# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Realtime voice sessions whose tool calls go through the same gate as text.

A speech-to-speech model calls tools mid-conversation. Here each call runs the
agent's :class:`~tulip.tools.decorator.Tool`, so hooks, ``gate_tool``, approval
stores and the audit trail apply exactly as in a text run. A held action is not
performed: the model is told it is waiting for a person, so it says so, and the
application receives :class:`ActionHeld` with the approval id::

    async with connect_openai_realtime(
        tools=[gated_refund, lookup_order]
    ) as session:
        await session.send_text("Refund order 4821, it arrived broken.")
        async for event in session.events():
            if isinstance(event, AudioDelta):
                speaker.write(event.audio)
            elif isinstance(event, ActionHeld):
                notify_approver(event.approval_id)

:class:`RealtimeSession` works over anything with ``send(event)`` and ``recv()``,
so another provider's realtime socket fits the same way.
:func:`connect_openai_realtime` opens OpenAI's Realtime API (the ``openai`` extra).
"""

from __future__ import annotations

import base64
import contextlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

    from tulip.tools.decorator import Tool


@runtime_checkable
class RealtimeConnection(Protocol):
    """A realtime socket: send JSON events, receive events (``None`` once closed)."""

    async def send(self, event: dict[str, Any]) -> None:
        """Send one client event."""
        ...

    async def recv(self) -> Any:
        """The next server event, or ``None`` when the connection has closed."""
        ...


@dataclass
class AudioDelta:
    """Model audio to play."""

    audio: bytes


@dataclass
class Transcript:
    """What was said, by ``user`` or ``assistant``."""

    role: str
    text: str


@dataclass
class ToolRan:
    """A tool call that ran; ``result`` is what the model was told."""

    name: str
    arguments: dict[str, Any]
    result: str


@dataclass
class ActionHeld:
    """A tool call held for a person; nothing was performed."""

    name: str
    arguments: dict[str, Any]
    question: str
    approval_id: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseDone:
    """The model finished a response."""


@dataclass
class SessionError:
    """The provider reported an error."""

    message: str


RealtimeEvent = AudioDelta | Transcript | ToolRan | ActionHeld | ResponseDone | SessionError

_AUDIO_DELTA = {"response.output_audio.delta", "response.audio.delta"}
_ASSISTANT_TRANSCRIPT = {"response.output_audio_transcript.done", "response.audio_transcript.done"}
_USER_TRANSCRIPT = "conversation.item.input_audio_transcription.completed"


def _get(event: Any, key: str, default: Any = None) -> Any:
    if isinstance(event, dict):
        return event.get(key, default)
    return getattr(event, key, default)


def realtime_tool_schema(tool: Tool) -> dict[str, Any]:
    """A tool as a realtime session declares it: the function fields at top level."""
    schema = tool.to_openai_schema()
    function = schema.get("function", schema)
    return {
        "type": "function",
        "name": function["name"],
        "description": function.get("description", ""),
        "parameters": function.get("parameters", {"type": "object", "properties": {}}),
    }


def _closed(exc: BaseException) -> bool:
    return type(exc).__name__.startswith("ConnectionClosed")


class RealtimeSession:
    """One realtime conversation over ``connection``, with gated tools.

    Args:
        connection: The provider socket.
        tools: Tools the model may call; wrap side effects with ``gate_tool``.
        instructions: The session's system instructions.
        voice: A provider voice name, when the provider takes one.
    """

    def __init__(
        self,
        connection: RealtimeConnection,
        *,
        tools: Iterable[Tool] = (),
        instructions: str = "",
        voice: str | None = None,
    ) -> None:
        self.connection = connection
        self.tools = {t.name: t for t in tools}
        self.instructions = instructions
        self.voice = voice

    async def start(self) -> None:
        """Declare the instructions and tools to the provider."""
        session: dict[str, Any] = {
            "type": "realtime",
            "instructions": self.instructions,
            "tools": [realtime_tool_schema(t) for t in self.tools.values()],
            "tool_choice": "auto",
        }
        if self.voice:
            session["audio"] = {"output": {"voice": self.voice}}
        await self.connection.send({"type": "session.update", "session": session})

    async def send_audio(self, pcm: bytes) -> None:
        """Append microphone audio to the input buffer."""
        await self.connection.send(
            {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode("ascii")}
        )

    async def send_text(self, text: str) -> None:
        """Say something as the user in text, and ask for a response."""
        await self.connection.send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )
        await self.connection.send({"type": "response.create"})

    async def events(self) -> AsyncIterator[RealtimeEvent]:
        """Server events as :data:`RealtimeEvent`s, running tool calls on the way."""
        while True:
            try:
                event = await self.connection.recv()
            except Exception as exc:
                if _closed(exc):
                    return
                raise
            if event is None:
                return
            kind = _get(event, "type", "")
            if kind in _AUDIO_DELTA:
                yield AudioDelta(base64.b64decode(_get(event, "delta", "")))
            elif kind in _ASSISTANT_TRANSCRIPT:
                yield Transcript("assistant", str(_get(event, "transcript", "")))
            elif kind == _USER_TRANSCRIPT:
                yield Transcript("user", str(_get(event, "transcript", "")))
            elif kind == "response.function_call_arguments.done":
                yield await self._call_tool(event)
            elif kind == "response.done":
                yield ResponseDone()
            elif kind == "error":
                error = _get(event, "error", {})
                yield SessionError(str(_get(error, "message", error)))

    async def _call_tool(self, event: Any) -> ToolRan | ActionHeld | SessionError:
        name = str(_get(event, "name", ""))
        call_id = str(_get(event, "call_id", ""))
        outcome: ToolRan | ActionHeld | SessionError
        try:
            arguments = json.loads(_get(event, "arguments", "") or "{}")
        except ValueError:
            arguments = None
        tool = self.tools.get(name)
        if tool is None:
            output = json.dumps({"error": f"no tool named {name!r}"})
            outcome = SessionError(f"the model called an unknown tool {name!r}")
        elif not isinstance(arguments, dict):
            output = json.dumps({"error": "the arguments were not a JSON object"})
            outcome = SessionError(f"the model sent malformed arguments to {name!r}")
        else:
            try:
                result = await tool.execute(**arguments)
            except Exception as exc:  # noqa: BLE001 — a tool failure is reported to the model, not raised
                output = json.dumps({"error": str(exc)})
                outcome = ToolRan(name, arguments, output)
            else:
                output = result if isinstance(result, str) else json.dumps(result, default=str)
                held = _held(output)
                if held is not None:
                    metadata = held.get("metadata", {}) or {}
                    question = str(held.get("question", "This action needs approval."))
                    output = (
                        f"Not done yet: this action is waiting for a person to approve it "
                        f"({question}). Tell the user it is pending."
                    )
                    outcome = ActionHeld(
                        name, arguments, question, metadata.get("approval_id"), metadata
                    )
                else:
                    outcome = ToolRan(name, arguments, output)
        await self.connection.send(
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": call_id, "output": output},
            }
        )
        await self.connection.send({"type": "response.create"})
        return outcome


def _held(output: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(output)
    except ValueError:
        return None
    if isinstance(payload, dict) and payload.get("__interrupt__") is True:
        return payload
    return None


@contextlib.asynccontextmanager
async def connect_openai_realtime(
    *,
    model: str = "gpt-realtime",
    tools: Iterable[Tool] = (),
    instructions: str = "",
    voice: str | None = None,
    api_key: str | None = None,
) -> AsyncIterator[RealtimeSession]:
    """Open an OpenAI Realtime session with ``tools`` declared and gated."""
    try:
        from openai import AsyncOpenAI  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError(
            'realtime voice needs the openai package: pip install "tulip-agents[openai]"'
        ) from exc
    client = AsyncOpenAI(api_key=api_key)
    async with client.realtime.connect(model=model) as connection:
        session = RealtimeSession(
            cast("RealtimeConnection", connection),
            tools=tools,
            instructions=instructions,
            voice=voice,
        )
        await session.start()
        yield session


__all__ = [
    "ActionHeld",
    "AudioDelta",
    "RealtimeConnection",
    "RealtimeEvent",
    "RealtimeSession",
    "ResponseDone",
    "SessionError",
    "ToolRan",
    "Transcript",
    "connect_openai_realtime",
    "realtime_tool_schema",
]
