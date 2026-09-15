# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Gemini Live as a realtime connection.

:class:`~tulip.voice.realtime.RealtimeSession` speaks realtime events over any
socket with ``send`` and ``recv``. :class:`GeminiLiveConnection` is that socket
for a Gemini Live session: it turns microphone audio, typed text and tool
results into Gemini Live calls, and Gemini's audio, transcripts, tool calls and
turn ends into the events the session reads. The tools are the same gated
:class:`~tulip.tools.decorator.Tool` objects, so a held action is held on Gemini
exactly as on OpenAI::

    async with connect_gemini_live(
        model="gemini-live-model-name", tools=[refund], instructions="..."
    ) as session:
        await session.send_audio(pcm_16khz)
        async for event in session.events():
            ...

Gemini fixes the instructions, tools and voice when the session opens, so they
are passed to :func:`connect_gemini_live` rather than sent later. Input audio is
16 kHz 16-bit mono PCM; output audio is 24 kHz. A tool call Gemini cancels after
it ran cannot be undone. Needs ``pip install "tulip-agents[gemini]"``.
"""

from __future__ import annotations

import base64
import contextlib
import json
from collections import deque
from typing import TYPE_CHECKING, Any, cast

from tulip.voice.realtime import RealtimeConnection, RealtimeSession, realtime_tool_schema


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

    from tulip.tools.decorator import Tool


#: The audio format :meth:`RealtimeSession.send_audio` is declared as.
INPUT_AUDIO_MIME = "audio/pcm;rate=16000"


def _get(value: Any, key: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def gemini_live_config(
    *,
    tools: Iterable[Tool] = (),
    instructions: str = "",
    voice: str | None = None,
) -> dict[str, Any]:
    """The Gemini Live connect config for ``tools``, ``instructions`` and ``voice``.

    Audio responses, with transcripts of both sides.
    """
    config: dict[str, Any] = {
        "response_modalities": ["AUDIO"],
        "input_audio_transcription": {},
        "output_audio_transcription": {},
    }
    if instructions:
        config["system_instruction"] = instructions
    declarations = [
        {
            "name": schema["name"],
            "description": schema["description"],
            "parameters_json_schema": schema["parameters"],
        }
        for schema in map(realtime_tool_schema, tools)
    ]
    if declarations:
        config["tools"] = [{"function_declarations": declarations}]
    if voice:
        config["speech_config"] = {"voice_config": {"prebuilt_voice_config": {"voice_name": voice}}}
    return config


class GeminiLiveConnection:
    """A Gemini Live session, as the ``send``/``recv`` socket a realtime session uses.

    Args:
        session: An open ``google.genai`` live session
            (``client.aio.live.connect(...)``).
        input_audio_mime: The format of the audio passed to ``send_audio``.
    """

    def __init__(self, session: Any, *, input_audio_mime: str = INPUT_AUDIO_MIME) -> None:
        self.session = session
        self.input_audio_mime = input_audio_mime
        self._events: deque[dict[str, Any]] = deque()
        self._calls: dict[str, tuple[str, str | None]] = {}
        self._user_text: list[str] = []
        self._assistant_text: list[str] = []
        self._turn: Any = None

    async def send(self, event: dict[str, Any]) -> None:
        """Carry out one realtime client event on the Gemini session.

        ``session.update`` and ``response.create`` need nothing: Gemini is
        configured when the session opens and answers without being asked.
        """
        kind = event.get("type")
        if kind == "input_audio_buffer.append":
            await self.session.send_realtime_input(
                audio={
                    "data": base64.b64decode(event.get("audio") or ""),
                    "mime_type": self.input_audio_mime,
                }
            )
        elif kind == "conversation.item.create":
            item = event.get("item") or {}
            if item.get("type") == "function_call_output":
                await self._send_tool_output(item)
            elif item.get("type") == "message":
                text = "".join(
                    str(part.get("text") or "")
                    for part in item.get("content") or []
                    if isinstance(part, dict)
                )
                await self.session.send_client_content(
                    turns={"role": "user", "parts": [{"text": text}]}, turn_complete=True
                )

    async def _send_tool_output(self, item: dict[str, Any]) -> None:
        call_id = str(item.get("call_id") or "")
        name, gemini_id = self._calls.pop(call_id, ("", None))
        response: dict[str, Any] = {"name": name, "response": {"output": item.get("output", "")}}
        if gemini_id is not None:
            response["id"] = gemini_id
        await self.session.send_tool_response(function_responses=[response])

    async def recv(self) -> dict[str, Any] | None:
        """The next realtime server event, or ``None`` when the session has ended."""
        while not self._events:
            message = await self._next_message()
            if message is None:
                return None
            self._translate(message)
        return self._events.popleft()

    async def _next_message(self) -> Any:
        # ``receive()`` ends after each model turn; a fresh one that ends at once
        # means the session is over.
        fresh = self._turn is None
        if fresh:
            self._turn = self.session.receive().__aiter__()
        try:
            return await self._turn.__anext__()
        except StopAsyncIteration:
            self._turn = None
            if fresh:
                return None
            return await self._next_message()

    def _translate(self, message: Any) -> None:
        content = _get(message, "server_content")
        if content is not None:
            self._translate_content(content)
        for call in _get(_get(message, "tool_call"), "function_calls") or []:
            self._flush_user()
            gemini_id = _get(call, "id")
            call_id = str(gemini_id or f"gemini_call_{len(self._calls)}")
            name = str(_get(call, "name") or "")
            self._calls[call_id] = (name, gemini_id)
            self._events.append(
                {
                    "type": "response.function_call_arguments.done",
                    "call_id": call_id,
                    "name": name,
                    "arguments": json.dumps(_get(call, "args") or {}),
                }
            )
        go_away = _get(message, "go_away")
        if go_away is not None:
            self._events.append(
                {
                    "type": "error",
                    "error": {
                        "message": "Gemini is closing the session "
                        f"(time left: {_get(go_away, 'time_left')})"
                    },
                }
            )

    def _translate_content(self, content: Any) -> None:
        user = _get(_get(content, "input_transcription"), "text")
        if user:
            self._user_text.append(str(user))
        for part in _get(_get(content, "model_turn"), "parts") or []:
            data = _get(_get(part, "inline_data"), "data")
            if data:
                self._flush_user()
                raw = data if isinstance(data, bytes) else base64.b64decode(data)
                self._events.append(
                    {
                        "type": "response.output_audio.delta",
                        "delta": base64.b64encode(raw).decode("ascii"),
                    }
                )
            text = _get(part, "text")
            if text and not _get(part, "thought"):
                self._assistant_text.append(str(text))
        spoken = _get(_get(content, "output_transcription"), "text")
        if spoken:
            self._assistant_text.append(str(spoken))
        if _get(content, "turn_complete") or _get(content, "interrupted"):
            self._flush_user()
            self._flush_assistant()
            if _get(content, "turn_complete"):
                self._events.append({"type": "response.done"})

    def _flush_user(self) -> None:
        if self._user_text:
            self._events.append(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "".join(self._user_text),
                }
            )
            self._user_text.clear()

    def _flush_assistant(self) -> None:
        if self._assistant_text:
            self._events.append(
                {
                    "type": "response.output_audio_transcript.done",
                    "transcript": "".join(self._assistant_text),
                }
            )
            self._assistant_text.clear()


@contextlib.asynccontextmanager
async def connect_gemini_live(
    *,
    model: str,
    tools: Iterable[Tool] = (),
    instructions: str = "",
    voice: str | None = None,
    api_key: str | None = None,
    config: dict[str, Any] | None = None,
) -> AsyncIterator[RealtimeSession]:
    """Open a Gemini Live session with ``tools`` declared and gated.

    Args:
        model: A Gemini Live model name.
        tools: Tools the model may call; wrap side effects with ``gate_tool``.
        instructions: The session's system instructions.
        voice: A Gemini prebuilt voice name.
        api_key: The Gemini API key; by default ``google-genai`` reads it from
            the environment.
        config: Extra connect config merged over :func:`gemini_live_config`.
    """
    try:
        from google import genai  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError(
            'Gemini Live needs the google-genai package: pip install "tulip-agents[gemini]"'
        ) from exc
    tools = list(tools)
    live_config = gemini_live_config(tools=tools, instructions=instructions, voice=voice)
    live_config.update(config or {})
    client = genai.Client(api_key=api_key)
    async with client.aio.live.connect(
        model=model, config=cast("Any", live_config)
    ) as live:  # pragma: no cover
        session = RealtimeSession(
            cast("RealtimeConnection", GeminiLiveConnection(live)),
            tools=tools,
            instructions=instructions,
            voice=voice,
        )
        await session.start()
        yield session


__all__ = [
    "INPUT_AUDIO_MIME",
    "GeminiLiveConnection",
    "connect_gemini_live",
    "gemini_live_config",
]
