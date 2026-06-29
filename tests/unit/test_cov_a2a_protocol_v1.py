# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Coverage tests for ``tulip.a2a.protocol_v1`` — v1.0 handler error paths.

These exercise the validation-failure and lifecycle-guard branches in the
``A2AV1ServerMixin`` handlers plus the ISO timestamp parser, all of which
the happy-path suite skips.
"""

from __future__ import annotations

from datetime import UTC
from typing import Any

import pytest

from tulip.a2a.protocol import A2AServer
from tulip.a2a.protocol_v1 import A2AV1ProtocolError, _parse_iso_timestamp
from tulip.a2a.spec import (
    INVALID_PARAMS,
    TASK_NOT_FOUND,
    UNSUPPORTED_OPERATION,
    Message,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)


class _StubAgent:
    async def run(self, prompt: str) -> Any:
        return
        yield  # pragma: no cover — never reached, keeps this a generator


def _server() -> A2AServer:
    server = A2AServer(agent=_StubAgent(), allow_unauthenticated=True)
    _ = server.app
    return server


def _put_task(server: A2AServer, task_id: str, state: TaskState) -> Task:
    task = Task(
        id=task_id,
        contextId=f"ctx-{task_id}",
        status=TaskStatus(state=state, timestamp="2026-05-25T12:00:00Z"),
        history=[Message(role="user", parts=[TextPart(text="hi")], messageId="m1")],
    )
    server._store.put(task)
    return task


# ---------------------------------------------------------------------------
# _parse_iso_timestamp
# ---------------------------------------------------------------------------


def test_parse_iso_timestamp_rejects_garbage() -> None:
    with pytest.raises(A2AV1ProtocolError) as err:
        _parse_iso_timestamp("not-a-real-timestamp")
    assert err.value.code == INVALID_PARAMS
    assert "invalid timestamp" in err.value.message


def test_parse_iso_timestamp_assumes_utc_for_naive_value() -> None:
    parsed = _parse_iso_timestamp("2026-05-25T12:00:00")
    assert parsed.tzinfo is UTC
    assert parsed.hour == 12


# ---------------------------------------------------------------------------
# Handler validation failures (INVALID_PARAMS)
# ---------------------------------------------------------------------------


async def test_v1_send_message_rejects_invalid_params() -> None:
    server = _server()
    with pytest.raises(A2AV1ProtocolError) as err:
        await server._handle_v1_send_message({"message": "not-a-message"})
    assert err.value.code == INVALID_PARAMS
    assert "invalid params" in err.value.message


async def test_v1_list_tasks_rejects_invalid_status_enum() -> None:
    server = _server()
    with pytest.raises(A2AV1ProtocolError) as err:
        await server._handle_v1_list_tasks({"status": "NOT_A_STATE"})
    assert err.value.code == INVALID_PARAMS
    assert "invalid params" in err.value.message


async def test_v1_cancel_task_rejects_missing_id() -> None:
    server = _server()
    with pytest.raises(A2AV1ProtocolError) as err:
        await server._handle_v1_cancel_task({})
    assert err.value.code == INVALID_PARAMS
    assert "invalid params" in err.value.message


async def test_v1_preflight_subscription_rejects_invalid_params() -> None:
    server = _server()
    with pytest.raises(A2AV1ProtocolError) as err:
        server._preflight_v1_task_subscription({})
    assert err.value.code == INVALID_PARAMS


# ---------------------------------------------------------------------------
# _stream_v1_task_subscription lifecycle guards
# ---------------------------------------------------------------------------


async def test_stream_subscription_unknown_task_raises_not_found() -> None:
    server = _server()
    with pytest.raises(A2AV1ProtocolError) as err:
        [event async for event in server._stream_v1_task_subscription({"id": "ghost"})]
    assert err.value.code == TASK_NOT_FOUND


async def test_stream_subscription_terminal_task_raises_unsupported() -> None:
    server = _server()
    _put_task(server, "done", TaskState.completed)
    with pytest.raises(A2AV1ProtocolError) as err:
        [event async for event in server._stream_v1_task_subscription({"id": "done"})]
    assert err.value.code == UNSUPPORTED_OPERATION
