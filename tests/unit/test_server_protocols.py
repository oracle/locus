# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The AG-UI and Vercel AI SDK stream adapters, checked frame by frame.

Each test feeds a fixture transcript (the ``/stream`` payload shapes a run
publishes) through an adapter and compares against the frames written out
in full, so a change to either mapping shows up as a diff of wire frames.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from locus.server.protocols import (
    AGUIAdapter,
    LocusAdapter,
    VercelDataStreamAdapter,
    ag_ui_run_input,
    encode_sse,
    last_user_prompt,
    make_adapter,
    ui_message_input,
)


def _run(adapter: Any, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frames = list(adapter.start())
    for payload in payloads:
        frames.extend(adapter.feed(payload))
    frames.extend(adapter.finish())
    return frames


STREAMED_TOOL_RUN: list[dict[str, Any]] = [
    {"type": "model_chunk", "content": "Let me ", "done": False},
    {"type": "model_chunk", "content": "look.", "done": False},
    {"type": "model_chunk", "content": None, "done": True},
    {"type": "think", "content": "Let me look."},
    {"type": "tool_start", "tool": "lookup", "tool_call_id": "c1", "arguments": {"q": "x"}},
    {
        "type": "tool_complete",
        "tool": "lookup",
        "tool_call_id": "c1",
        "result": "42",
        "error": None,
    },
    {"type": "model_chunk", "content": "It is 42.", "done": False},
    {"type": "think", "content": "It is 42."},
    {"type": "done", "message": "It is 42.", "reason": "complete"},
]


def test_ag_ui_streamed_tool_run() -> None:
    frames = _run(AGUIAdapter(thread_id="t1", run_id="r1"), STREAMED_TOOL_RUN)
    assert frames == [
        {"type": "RUN_STARTED", "threadId": "t1", "runId": "r1"},
        {
            "type": "STATE_SNAPSHOT",
            "snapshot": {"status": "running", "iteration": 0, "toolCalls": 0},
        },
        {"type": "TEXT_MESSAGE_START", "messageId": "r1-msg-1", "role": "assistant"},
        {"type": "TEXT_MESSAGE_CONTENT", "messageId": "r1-msg-1", "delta": "Let me "},
        {"type": "TEXT_MESSAGE_CONTENT", "messageId": "r1-msg-1", "delta": "look."},
        {"type": "TEXT_MESSAGE_END", "messageId": "r1-msg-1"},
        {"type": "STATE_DELTA", "delta": [{"op": "replace", "path": "/iteration", "value": 1}]},
        {
            "type": "TOOL_CALL_START",
            "toolCallId": "c1",
            "toolCallName": "lookup",
            "parentMessageId": "r1-msg-1",
        },
        {"type": "TOOL_CALL_ARGS", "toolCallId": "c1", "delta": '{"q": "x"}'},
        {"type": "TOOL_CALL_END", "toolCallId": "c1"},
        {
            "type": "TOOL_CALL_RESULT",
            "messageId": "r1-tool-1",
            "toolCallId": "c1",
            "content": "42",
            "role": "tool",
        },
        {"type": "STATE_DELTA", "delta": [{"op": "replace", "path": "/toolCalls", "value": 1}]},
        {"type": "TEXT_MESSAGE_START", "messageId": "r1-msg-2", "role": "assistant"},
        {"type": "TEXT_MESSAGE_CONTENT", "messageId": "r1-msg-2", "delta": "It is 42."},
        {"type": "TEXT_MESSAGE_END", "messageId": "r1-msg-2"},
        {"type": "STATE_DELTA", "delta": [{"op": "replace", "path": "/iteration", "value": 2}]},
        {
            "type": "STATE_DELTA",
            "delta": [{"op": "replace", "path": "/status", "value": "completed"}],
        },
        {
            "type": "RUN_FINISHED",
            "threadId": "t1",
            "runId": "r1",
            "result": {"message": "It is 42.", "stopReason": "complete"},
        },
    ]


def test_vercel_streamed_tool_run() -> None:
    frames = _run(VercelDataStreamAdapter(run_id="r1"), STREAMED_TOOL_RUN)
    assert frames == [
        {"type": "start", "messageId": "r1"},
        {"type": "start-step"},
        {"type": "text-start", "id": "r1-text-1"},
        {"type": "text-delta", "id": "r1-text-1", "delta": "Let me "},
        {"type": "text-delta", "id": "r1-text-1", "delta": "look."},
        {"type": "text-end", "id": "r1-text-1"},
        {"type": "tool-input-start", "toolCallId": "c1", "toolName": "lookup"},
        {
            "type": "tool-input-available",
            "toolCallId": "c1",
            "toolName": "lookup",
            "input": {"q": "x"},
        },
        {"type": "tool-output-available", "toolCallId": "c1", "output": "42"},
        {"type": "text-start", "id": "r1-text-2"},
        {"type": "text-delta", "id": "r1-text-2", "delta": "It is 42."},
        {"type": "text-end", "id": "r1-text-2"},
        {"type": "finish-step"},
        {"type": "finish"},
    ]


UNSTREAMED_INTERRUPT: list[dict[str, Any]] = [
    {"type": "think", "content": "I need approval."},
    {"type": "tool_start", "tool": "refund", "tool_call_id": "c9", "arguments": {"amount": 5}},
    {
        "type": "interrupt",
        "question": "Approve refund?",
        "options": None,
        "fields": None,
        "interrupt_id": "c9",
        "metadata": {"approval_id": "appr-1"},
    },
]


def test_ag_ui_unstreamed_text_and_interrupt() -> None:
    frames = _run(AGUIAdapter(thread_id="t", run_id="r"), UNSTREAMED_INTERRUPT)
    assert frames == [
        {"type": "RUN_STARTED", "threadId": "t", "runId": "r"},
        {
            "type": "STATE_SNAPSHOT",
            "snapshot": {"status": "running", "iteration": 0, "toolCalls": 0},
        },
        {"type": "TEXT_MESSAGE_START", "messageId": "r-msg-1", "role": "assistant"},
        {"type": "TEXT_MESSAGE_CONTENT", "messageId": "r-msg-1", "delta": "I need approval."},
        {"type": "TEXT_MESSAGE_END", "messageId": "r-msg-1"},
        {"type": "STATE_DELTA", "delta": [{"op": "replace", "path": "/iteration", "value": 1}]},
        {
            "type": "TOOL_CALL_START",
            "toolCallId": "c9",
            "toolCallName": "refund",
            "parentMessageId": "r-msg-1",
        },
        {"type": "TOOL_CALL_ARGS", "toolCallId": "c9", "delta": '{"amount": 5}'},
        {"type": "TOOL_CALL_END", "toolCallId": "c9"},
        {"type": "CUSTOM", "name": "locus.interrupt", "value": UNSTREAMED_INTERRUPT[2]},
        {
            "type": "STATE_DELTA",
            "delta": [{"op": "replace", "path": "/status", "value": "interrupted"}],
        },
        {
            "type": "RUN_FINISHED",
            "threadId": "t",
            "runId": "r",
            "outcome": {
                "type": "interrupt",
                "interrupts": [{"id": "c9", "reason": "Approve refund?"}],
            },
        },
    ]


def test_vercel_unstreamed_text_and_interrupt() -> None:
    frames = _run(VercelDataStreamAdapter(run_id="r"), UNSTREAMED_INTERRUPT)
    assert frames == [
        {"type": "start", "messageId": "r"},
        {"type": "start-step"},
        {"type": "text-start", "id": "r-text-1"},
        {"type": "text-delta", "id": "r-text-1", "delta": "I need approval."},
        {"type": "text-end", "id": "r-text-1"},
        {"type": "tool-input-start", "toolCallId": "c9", "toolName": "refund"},
        {
            "type": "tool-input-available",
            "toolCallId": "c9",
            "toolName": "refund",
            "input": {"amount": 5},
        },
        {"type": "data-locus-interrupt", "data": UNSTREAMED_INTERRUPT[2]},
        {"type": "finish-step"},
        {"type": "finish"},
    ]


ODDS_AND_ENDS: list[dict[str, Any]] = [
    {"type": "model_chunk", "content": None, "reasoning": "hmm", "done": False},
    {"type": "model_chunk", "content": "Sure", "reasoning": None, "done": False},
    {"type": "think", "content": "Sure"},
    {"type": "tool_complete", "tool": "x", "tool_call_id": "c2", "result": None, "error": "boom"},
    {"type": "custom", "name": "hotel_list", "data": {"n": 1}},
    {"type": "reflect", "iteration": 1, "assessment": "on_track"},
    {"type": "replay_gap", "from": 3, "to": 5},
    {"type": "error", "error": "internal error", "correlation_id": "abc"},
]


def test_ag_ui_reasoning_errors_custom_and_gap() -> None:
    frames = _run(AGUIAdapter(thread_id="t", run_id="r"), ODDS_AND_ENDS)
    assert frames[2:] == [
        {"type": "CUSTOM", "name": "locus.reasoning", "value": {"delta": "hmm"}},
        {"type": "TEXT_MESSAGE_START", "messageId": "r-msg-1", "role": "assistant"},
        {"type": "TEXT_MESSAGE_CONTENT", "messageId": "r-msg-1", "delta": "Sure"},
        {"type": "TEXT_MESSAGE_END", "messageId": "r-msg-1"},
        {"type": "STATE_DELTA", "delta": [{"op": "replace", "path": "/iteration", "value": 1}]},
        {
            "type": "TOOL_CALL_RESULT",
            "messageId": "r-tool-1",
            "toolCallId": "c2",
            "content": '{"error": "boom"}',
            "role": "tool",
        },
        {"type": "STATE_DELTA", "delta": [{"op": "replace", "path": "/toolCalls", "value": 1}]},
        {"type": "CUSTOM", "name": "hotel_list", "value": {"n": 1}},
        {"type": "CUSTOM", "name": "locus.reflect", "value": ODDS_AND_ENDS[5]},
        {"type": "CUSTOM", "name": "locus.replay_gap", "value": {"from": 3, "to": 5}},
        {"type": "STATE_DELTA", "delta": [{"op": "replace", "path": "/status", "value": "failed"}]},
        {"type": "RUN_ERROR", "message": "internal error", "code": "abc"},
    ]


def test_vercel_reasoning_errors_custom_and_gap() -> None:
    frames = _run(VercelDataStreamAdapter(run_id="r"), ODDS_AND_ENDS)
    assert frames[2:] == [
        {"type": "reasoning-start", "id": "r-reasoning-1"},
        {"type": "reasoning-delta", "id": "r-reasoning-1", "delta": "hmm"},
        {"type": "reasoning-end", "id": "r-reasoning-1"},
        {"type": "text-start", "id": "r-text-1"},
        {"type": "text-delta", "id": "r-text-1", "delta": "Sure"},
        {"type": "text-end", "id": "r-text-1"},
        {"type": "tool-output-available", "toolCallId": "c2", "output": {"error": "boom"}},
        {"type": "data-hotel_list", "data": {"n": 1}},
        {"type": "data-locus-event", "data": ODDS_AND_ENDS[5]},
        {"type": "data-locus-replay-gap", "data": {"from": 3, "to": 5}},
        {"type": "error", "errorText": "internal error (correlation_id=abc)"},
    ]


def test_finish_closes_a_run_that_ended_without_a_terminal_event() -> None:
    ag = AGUIAdapter(thread_id="t", run_id="r")
    ag.start()
    ag.feed({"type": "model_chunk", "content": "half", "done": False})
    assert ag.finish() == [
        {"type": "TEXT_MESSAGE_END", "messageId": "r-msg-1"},
        {
            "type": "STATE_DELTA",
            "delta": [{"op": "replace", "path": "/status", "value": "completed"}],
        },
        {"type": "RUN_FINISHED", "threadId": "t", "runId": "r"},
    ]
    assert ag.finish() == []

    vercel = VercelDataStreamAdapter(run_id="r")
    vercel.start()
    vercel.feed({"type": "model_chunk", "content": "half", "done": False})
    assert vercel.finish() == [
        {"type": "text-end", "id": "r-text-1"},
        {"type": "finish-step"},
        {"type": "finish"},
    ]
    assert vercel.finish() == []


def test_done_message_is_not_repeated_after_streamed_text() -> None:
    payloads = [
        {"type": "model_chunk", "content": "Hi", "done": False},
        {"type": "think", "content": "Hi"},
        {"type": "done", "message": "Hi", "reason": "complete"},
    ]
    kinds = [f["type"] for f in _run(VercelDataStreamAdapter(run_id="r"), payloads)]
    assert kinds == [
        "start",
        "start-step",
        "text-start",
        "text-delta",
        "text-end",
        "finish-step",
        "finish",
    ]
    # A graph's progress lines differ from its final answer: both are shown.
    payloads = [
        {"type": "think", "content": "[g] node=draft produced data of type dict"},
        {"type": "done", "message": "Draft reply", "reason": "complete"},
    ]
    deltas = [
        f["delta"]
        for f in _run(AGUIAdapter(thread_id="t", run_id="r"), payloads)
        if f["type"] == "TEXT_MESSAGE_CONTENT"
    ]
    assert deltas == ["[g] node=draft produced data of type dict", "Draft reply"]


def test_locus_adapter_and_factory() -> None:
    assert LocusAdapter().feed({"type": "think", "content": "x"}) == [
        {"type": "think", "content": "x"}
    ]
    assert isinstance(make_adapter("locus", thread_id="t", run_id="r"), LocusAdapter)
    assert isinstance(make_adapter("ag-ui", thread_id="t", run_id="r"), AGUIAdapter)
    vercel = make_adapter("vercel-ai", thread_id="t", run_id="r")
    assert isinstance(vercel, VercelDataStreamAdapter)
    assert vercel.headers["x-vercel-ai-ui-message-stream"] == "v1"
    with pytest.raises(ValueError, match="unknown protocol"):
        make_adapter("graphql", thread_id="t", run_id="r")


def test_encode_sse_puts_the_id_line_first() -> None:
    assert encode_sse({"a": 1}) == 'data: {"a": 1}\n\n'
    assert encode_sse({"a": 1}, event_id="7") == 'id: 7\ndata: {"a": 1}\n\n'
    assert json.loads(encode_sse({"when": object()}).split("data: ")[1]) is not None


def test_request_body_parsers() -> None:
    thread_id, run_id, prompt, metadata = ag_ui_run_input(
        {
            "threadId": "t1",
            "runId": "r1",
            "messages": [
                {"id": "a", "role": "assistant", "content": "old"},
                {"id": "u", "role": "user", "content": [{"type": "text", "text": "hello"}]},
            ],
            "forwardedProps": {"user": "alice"},
        }
    )
    assert (thread_id, run_id, prompt, metadata) == ("t1", "r1", "hello", {"user": "alice"})

    thread_id, prompt, metadata = ui_message_input(
        {
            "id": "chat-1",
            "messages": [{"id": "m", "role": "user", "parts": [{"type": "text", "text": "hey"}]}],
        }
    )
    assert (thread_id, prompt, metadata) == ("chat-1", "hey", {})
    assert (
        ui_message_input({"thread_id": "x", "messages": [{"role": "user", "content": "c"}]})[0]
        == "x"
    )

    with pytest.raises(ValueError, match="no user message"):
        last_user_prompt([{"role": "assistant", "content": "only me"}])
    with pytest.raises(ValueError, match="no user message"):
        last_user_prompt("not a list")
