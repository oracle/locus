# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""UI wire protocols over the server's event stream.

Every run the server drives publishes the ``/stream`` frame shapes
(``think``, ``tool_start``, ``tool_complete``, ``model_chunk``, ``interrupt``,
``done``, ``error``, …). An adapter here turns that one stream into the
frames a web agent UI already understands:

* :class:`LocusAdapter` — the frames unchanged (``protocol=locus``).
* :class:`AGUIAdapter` — AG-UI events: ``RUN_STARTED`` / ``RUN_FINISHED`` /
  ``RUN_ERROR``, ``TEXT_MESSAGE_START`` / ``CONTENT`` / ``END``,
  ``TOOL_CALL_START`` / ``ARGS`` / ``END`` / ``RESULT``, ``STATE_SNAPSHOT`` /
  ``STATE_DELTA`` and ``CUSTOM``.
* :class:`VercelDataStreamAdapter` — the Vercel AI SDK UI message stream
  (``x-vercel-ai-ui-message-stream: v1``): ``start``, ``start-step``,
  ``text-*``, ``reasoning-*``, ``tool-input-*``, ``tool-output-available``,
  ``data-*``, ``error``, ``finish-step``, ``finish``.

Adapters are small state machines: :meth:`~StreamAdapter.start` opens the
stream, :meth:`~StreamAdapter.feed` turns one payload into zero or more
frames, :meth:`~StreamAdapter.finish` closes anything left open when the
run ends without a terminal event. Frame ids are derived from the run id
and a counter, so a stream rebuilt from history on reconnect names the
same messages.

Where the published protocol pages leave room, the conservative reading
was taken and is recorded in ``docs/concepts/server.md``: assistant text
comes from token deltas when the server streams them and from the
``think`` content otherwise, never both; tool arguments are sent whole; a
tool error is a ``TOOL_CALL_RESULT`` / ``tool-output-available`` whose
content is an ``{"error": ...}`` object; an interrupt is a ``CUSTOM`` /
``data-locus-interrupt`` frame followed by the protocol's run-finished
frame; events neither protocol has a word for travel as ``CUSTOM`` /
``data-locus-event`` frames.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from locus.server.runs import REPLAY_GAP


SSE_MEDIA_TYPE = "text/event-stream"

#: Response headers every event stream carries, protocol or not.
_SSE_HEADERS: dict[str, str] = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

#: Query-parameter values of ``GET /runs/{id}/events?protocol=``.
PROTOCOL_LOCUS = "locus"
PROTOCOL_AG_UI = "ag-ui"
PROTOCOL_VERCEL_AI = "vercel-ai"
PROTOCOL_NAMES: frozenset[str] = frozenset({PROTOCOL_LOCUS, PROTOCOL_AG_UI, PROTOCOL_VERCEL_AI})


def encode_sse(frame: Mapping[str, Any], *, event_id: str | None = None) -> str:
    """One SSE message: an optional ``id:`` line and the frame as JSON."""
    head = f"id: {event_id}\n" if event_id is not None else ""
    return f"{head}data: {json.dumps(frame, default=str)}\n\n"


class StreamAdapter(Protocol):
    """What a wire-protocol adapter provides to the SSE and WebSocket routes."""

    media_type: str
    headers: Mapping[str, str]

    def start(self) -> list[dict[str, Any]]:
        """Frames that open the stream, before any run event."""

    def feed(self, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Frames for one run event payload."""

    def finish(self) -> list[dict[str, Any]]:
        """Frames that close the stream when no terminal event arrived."""


class LocusAdapter:
    """The server's own frames, passed through unchanged."""

    media_type = SSE_MEDIA_TYPE
    headers: Mapping[str, str] = dict(_SSE_HEADERS)

    def start(self) -> list[dict[str, Any]]:
        return []

    def feed(self, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [dict(payload)]

    def finish(self) -> list[dict[str, Any]]:
        return []


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _tool_output(payload: Mapping[str, Any]) -> Any:
    """A tool's result, or an ``{"error": ...}`` object when it failed."""
    error = payload.get("error")
    if error:
        return {"error": error}
    result = payload.get("result")
    return "" if result is None else result


class _TextTracker:
    """Shared bookkeeping for assistant text across both protocols.

    Text arrives twice on the server stream when tokens are streamed: as
    ``model_chunk`` deltas and again, whole, as the following ``think``
    content. A UI must see it once, so the tracker remembers whether deltas
    were streamed since the last ``think`` and what text was last emitted
    (the ``done`` message repeats the final answer).
    """

    def __init__(self) -> None:
        self.streamed = False
        self.last_text: str | None = None
        self.open_id: str | None = None
        self.count = 0
        self.buffer: list[str] = []


class AGUIAdapter:
    """AG-UI events for one run. See the module docstring for the mapping."""

    media_type = SSE_MEDIA_TYPE
    headers: Mapping[str, str] = dict(_SSE_HEADERS)

    def __init__(self, *, thread_id: str, run_id: str) -> None:
        self.thread_id = thread_id
        self.run_id = run_id
        self._state: dict[str, Any] = {"status": "running", "iteration": 0, "toolCalls": 0}
        self._text = _TextTracker()
        self._last_message_id: str | None = None
        self._tool_results = 0
        self._finished = False

    # -- frames --------------------------------------------------------

    def _run(self, kind: str, **fields: Any) -> dict[str, Any]:
        return {"type": kind, "threadId": self.thread_id, "runId": self.run_id, **fields}

    def _delta(self, path: str, value: Any) -> dict[str, Any]:
        self._state[path] = value
        return {
            "type": "STATE_DELTA",
            "delta": [{"op": "replace", "path": f"/{path}", "value": value}],
        }

    def _open_text(self) -> list[dict[str, Any]]:
        tracker = self._text
        if tracker.open_id is not None:
            return []
        tracker.count += 1
        tracker.open_id = f"{self.run_id}-msg-{tracker.count}"
        tracker.buffer = []
        self._last_message_id = tracker.open_id
        return [{"type": "TEXT_MESSAGE_START", "messageId": tracker.open_id, "role": "assistant"}]

    def _close_text(self) -> list[dict[str, Any]]:
        tracker = self._text
        if tracker.open_id is None:
            return []
        frame = {"type": "TEXT_MESSAGE_END", "messageId": tracker.open_id}
        tracker.last_text = "".join(tracker.buffer)
        tracker.open_id = None
        return [frame]

    def _text_message(self, content: str) -> list[dict[str, Any]]:
        frames = self._open_text()
        self._text.buffer.append(content)
        frames.append(
            {"type": "TEXT_MESSAGE_CONTENT", "messageId": self._text.open_id, "delta": content}
        )
        frames.extend(self._close_text())
        return frames

    # -- protocol ------------------------------------------------------

    def start(self) -> list[dict[str, Any]]:
        return [
            self._run("RUN_STARTED"),
            {"type": "STATE_SNAPSHOT", "snapshot": dict(self._state)},
        ]

    def feed(self, payload: Mapping[str, Any]) -> list[dict[str, Any]]:  # noqa: C901, PLR0912
        kind = payload.get("type")
        frames: list[dict[str, Any]] = []
        if kind == "model_chunk":
            reasoning = _text(payload.get("reasoning"))
            if reasoning:
                frames.append(
                    {"type": "CUSTOM", "name": "locus.reasoning", "value": {"delta": reasoning}}
                )
            content = _text(payload.get("content"))
            if content:
                frames.extend(self._open_text())
                self._text.buffer.append(content)
                self._text.streamed = True
                frames.append(
                    {
                        "type": "TEXT_MESSAGE_CONTENT",
                        "messageId": self._text.open_id,
                        "delta": content,
                    }
                )
        elif kind == "think":
            content = _text(payload.get("content"))
            if self._text.open_id is not None:
                frames.extend(self._close_text())
            elif content and not self._text.streamed:
                frames.extend(self._text_message(content))
            self._text.streamed = False
            frames.append(self._delta("iteration", int(self._state["iteration"]) + 1))
        elif kind == "tool_start":
            call_id = _text(payload.get("tool_call_id"))
            start: dict[str, Any] = {
                "type": "TOOL_CALL_START",
                "toolCallId": call_id,
                "toolCallName": _text(payload.get("tool")),
            }
            if self._last_message_id is not None:
                start["parentMessageId"] = self._last_message_id
            frames.extend(
                [
                    start,
                    {
                        "type": "TOOL_CALL_ARGS",
                        "toolCallId": call_id,
                        "delta": json.dumps(payload.get("arguments") or {}, default=str),
                    },
                    {"type": "TOOL_CALL_END", "toolCallId": call_id},
                ]
            )
        elif kind == "tool_complete":
            self._tool_results += 1
            output = _tool_output(payload)
            frames.append(
                {
                    "type": "TOOL_CALL_RESULT",
                    "messageId": f"{self.run_id}-tool-{self._tool_results}",
                    "toolCallId": _text(payload.get("tool_call_id")),
                    "content": output if isinstance(output, str) else json.dumps(output),
                    "role": "tool",
                }
            )
            frames.append(self._delta("toolCalls", int(self._state["toolCalls"]) + 1))
        elif kind == "interrupt":
            frames.extend(self._close_text())
            frames.append({"type": "CUSTOM", "name": "locus.interrupt", "value": dict(payload)})
            frames.append(self._delta("status", "interrupted"))
            frames.append(
                self._run(
                    "RUN_FINISHED",
                    outcome={
                        "type": "interrupt",
                        "interrupts": [
                            {
                                "id": _text(payload.get("interrupt_id")),
                                "reason": _text(payload.get("question")),
                            }
                        ],
                    },
                )
            )
            self._finished = True
        elif kind == "done":
            frames.extend(self._close_text())
            message = _text(payload.get("message"))
            if message and message != self._text.last_text:
                frames.extend(self._text_message(message))
            reason = _text(payload.get("reason")) or "complete"
            frames.append(
                self._delta("status", "cancelled" if reason == "cancelled" else "completed")
            )
            frames.append(
                self._run("RUN_FINISHED", result={"message": message, "stopReason": reason})
            )
            self._finished = True
        elif kind == "error":
            frames.extend(self._close_text())
            frames.append(self._delta("status", "failed"))
            frames.append(
                {
                    "type": "RUN_ERROR",
                    "message": _text(payload.get("error")) or "internal error",
                    "code": _text(payload.get("correlation_id")) or "internal_error",
                }
            )
            self._finished = True
        elif kind == "custom":
            frames.append(
                {
                    "type": "CUSTOM",
                    "name": _text(payload.get("name")) or "custom",
                    "value": payload.get("data"),
                }
            )
        elif kind == REPLAY_GAP:
            frames.append(
                {
                    "type": "CUSTOM",
                    "name": "locus.replay_gap",
                    "value": {"from": payload.get("from"), "to": payload.get("to")},
                }
            )
        else:
            frames.append({"type": "CUSTOM", "name": f"locus.{kind}", "value": dict(payload)})
        return frames

    def finish(self) -> list[dict[str, Any]]:
        if self._finished:
            return []
        self._finished = True
        frames = self._close_text()
        frames.append(self._delta("status", "completed"))
        frames.append(self._run("RUN_FINISHED"))
        return frames


class VercelDataStreamAdapter:
    """Vercel AI SDK UI message stream parts for one run."""

    media_type = SSE_MEDIA_TYPE
    headers: Mapping[str, str] = {**_SSE_HEADERS, "x-vercel-ai-ui-message-stream": "v1"}

    def __init__(self, *, run_id: str) -> None:
        self.run_id = run_id
        self._text = _TextTracker()
        self._reasoning_id: str | None = None
        self._reasoning_count = 0
        self._finished = False

    def _open_text(self) -> list[dict[str, Any]]:
        tracker = self._text
        if tracker.open_id is not None:
            return []
        tracker.count += 1
        tracker.open_id = f"{self.run_id}-text-{tracker.count}"
        tracker.buffer = []
        return [{"type": "text-start", "id": tracker.open_id}]

    def _close_text(self) -> list[dict[str, Any]]:
        tracker = self._text
        if tracker.open_id is None:
            return []
        frame = {"type": "text-end", "id": tracker.open_id}
        tracker.last_text = "".join(tracker.buffer)
        tracker.open_id = None
        return [frame]

    def _text_part(self, content: str) -> list[dict[str, Any]]:
        frames = self._open_text()
        self._text.buffer.append(content)
        frames.append({"type": "text-delta", "id": self._text.open_id, "delta": content})
        frames.extend(self._close_text())
        return frames

    def _close_reasoning(self) -> list[dict[str, Any]]:
        if self._reasoning_id is None:
            return []
        frame = {"type": "reasoning-end", "id": self._reasoning_id}
        self._reasoning_id = None
        return [frame]

    def _end(self) -> list[dict[str, Any]]:
        self._finished = True
        return [{"type": "finish-step"}, {"type": "finish"}]

    def start(self) -> list[dict[str, Any]]:
        return [{"type": "start", "messageId": self.run_id}, {"type": "start-step"}]

    def feed(self, payload: Mapping[str, Any]) -> list[dict[str, Any]]:  # noqa: C901, PLR0912
        kind = payload.get("type")
        frames: list[dict[str, Any]] = []
        if kind == "model_chunk":
            reasoning = _text(payload.get("reasoning"))
            if reasoning:
                if self._reasoning_id is None:
                    self._reasoning_count += 1
                    self._reasoning_id = f"{self.run_id}-reasoning-{self._reasoning_count}"
                    frames.append({"type": "reasoning-start", "id": self._reasoning_id})
                frames.append(
                    {"type": "reasoning-delta", "id": self._reasoning_id, "delta": reasoning}
                )
            content = _text(payload.get("content"))
            if content:
                frames.extend(self._close_reasoning())
                frames.extend(self._open_text())
                self._text.buffer.append(content)
                self._text.streamed = True
                frames.append({"type": "text-delta", "id": self._text.open_id, "delta": content})
        elif kind == "think":
            frames.extend(self._close_reasoning())
            content = _text(payload.get("content"))
            if self._text.open_id is not None:
                frames.extend(self._close_text())
            elif content and not self._text.streamed:
                frames.extend(self._text_part(content))
            self._text.streamed = False
        elif kind == "tool_start":
            call_id = _text(payload.get("tool_call_id"))
            name = _text(payload.get("tool"))
            frames.extend(
                [
                    {"type": "tool-input-start", "toolCallId": call_id, "toolName": name},
                    {
                        "type": "tool-input-available",
                        "toolCallId": call_id,
                        "toolName": name,
                        "input": payload.get("arguments") or {},
                    },
                ]
            )
        elif kind == "tool_complete":
            frames.append(
                {
                    "type": "tool-output-available",
                    "toolCallId": _text(payload.get("tool_call_id")),
                    "output": _tool_output(payload),
                }
            )
        elif kind == "interrupt":
            frames.extend(self._close_reasoning())
            frames.extend(self._close_text())
            frames.append({"type": "data-locus-interrupt", "data": dict(payload)})
            frames.extend(self._end())
        elif kind == "done":
            frames.extend(self._close_reasoning())
            frames.extend(self._close_text())
            message = _text(payload.get("message"))
            if message and message != self._text.last_text:
                frames.extend(self._text_part(message))
            frames.extend(self._end())
        elif kind == "error":
            frames.extend(self._close_reasoning())
            frames.extend(self._close_text())
            correlation = _text(payload.get("correlation_id"))
            text = _text(payload.get("error")) or "internal error"
            if correlation:
                text = f"{text} (correlation_id={correlation})"
            frames.append({"type": "error", "errorText": text})
            self._finished = True
        elif kind == "custom":
            name = _text(payload.get("name")) or "custom"
            frames.append({"type": f"data-{name}", "data": payload.get("data")})
        elif kind == REPLAY_GAP:
            frames.append(
                {
                    "type": "data-locus-replay-gap",
                    "data": {"from": payload.get("from"), "to": payload.get("to")},
                }
            )
        else:
            frames.append({"type": "data-locus-event", "data": dict(payload)})
        return frames

    def finish(self) -> list[dict[str, Any]]:
        if self._finished:
            return []
        frames = self._close_reasoning()
        frames.extend(self._close_text())
        frames.extend(self._end())
        return frames


def make_adapter(protocol: str, *, thread_id: str, run_id: str) -> StreamAdapter:
    """The adapter for a ``protocol=`` value; ``ValueError`` for an unknown one."""
    if protocol == PROTOCOL_LOCUS:
        return LocusAdapter()
    if protocol == PROTOCOL_AG_UI:
        return AGUIAdapter(thread_id=thread_id, run_id=run_id)
    if protocol == PROTOCOL_VERCEL_AI:
        return VercelDataStreamAdapter(run_id=run_id)
    raise ValueError(f"unknown protocol {protocol!r}; one of {sorted(PROTOCOL_NAMES)}")


# ----------------------------------------------------------------------
# Request bodies the protocols' clients send
# ----------------------------------------------------------------------


def _message_text(message: Mapping[str, Any]) -> str:
    """The text of one message: a string ``content`` or its text parts."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    pieces: list[str] = []
    parts: Any = content if isinstance(content, Sequence) else message.get("parts")
    if isinstance(parts, Sequence):
        for part in parts:
            if isinstance(part, Mapping) and part.get("type") == "text":
                pieces.append(_text(part.get("text")))
    return "".join(pieces)


def last_user_prompt(messages: Any) -> str:
    """The text of the last ``user`` message; ``ValueError`` when there is none."""
    if isinstance(messages, Sequence) and not isinstance(messages, str):
        for message in reversed(messages):
            if isinstance(message, Mapping) and message.get("role") == "user":
                text = _message_text(message)
                if text:
                    return text
    raise ValueError("the request carries no user message with text")


def _first_str(body: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = body.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def ag_ui_run_input(body: Mapping[str, Any]) -> tuple[str | None, str | None, str, dict[str, Any]]:
    """``(thread_id, run_id, prompt, metadata)`` from an AG-UI ``RunAgentInput``.

    ``threadId`` and ``runId`` are the client's; the prompt is the last user
    message; ``forwardedProps`` becomes the run's client metadata when it is
    an object. The input's ``tools`` (client-side tools) and ``state`` are
    not consumed: the server's tools run server-side.
    """
    thread_id = _first_str(body, "threadId", "thread_id")
    run_id = _first_str(body, "runId", "run_id")
    prompt = last_user_prompt(body.get("messages"))
    forwarded = body.get("forwardedProps", body.get("forwarded_props"))
    metadata = dict(forwarded) if isinstance(forwarded, Mapping) else {}
    return thread_id, run_id, prompt, metadata


def ui_message_input(body: Mapping[str, Any]) -> tuple[str | None, str, dict[str, Any]]:
    """``(thread_id, prompt, metadata)`` from a UI message request body.

    The chat ``id`` the client keeps per conversation is the thread id unless
    the body names ``thread_id`` explicitly; the prompt is the text parts of
    the last user message; a ``metadata`` object is passed through.
    """
    thread_id = _first_str(body, "thread_id", "threadId", "id")
    prompt = last_user_prompt(body.get("messages"))
    extra = body.get("metadata")
    metadata = dict(extra) if isinstance(extra, Mapping) else {}
    return thread_id, prompt, metadata


__all__ = [
    "PROTOCOL_AG_UI",
    "PROTOCOL_LOCUS",
    "PROTOCOL_NAMES",
    "PROTOCOL_VERCEL_AI",
    "SSE_MEDIA_TYPE",
    "AGUIAdapter",
    "LocusAdapter",
    "StreamAdapter",
    "VercelDataStreamAdapter",
    "ag_ui_run_input",
    "encode_sse",
    "last_user_prompt",
    "make_adapter",
    "ui_message_input",
]
