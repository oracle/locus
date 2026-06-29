# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Coverage tests for ``tulip.a2a.protocol`` — server + client edge paths.

Drives the error responses, JSON-RPC error codes, streaming/SSE branches,
task-lifecycle guards, and the v1/legacy client method splits that the
happy-path suite in ``test_a2a_protocol.py`` does not reach.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from tulip.a2a import (
    A2AClient,
    A2AServer,
    DataPart,
    FilePart,
    Message,
    Task,
    TaskState,
    TextPart,
)
from tulip.a2a.protocol import (
    _extract_user_text,
    _TaskStore,
)
from tulip.a2a.spec import (
    FileWithBytes,
    FileWithUri,
    TaskStatus,
)
from tulip.core.events import TerminateEvent, ThinkEvent, ToolStartEvent


# ---------------------------------------------------------------------------
# Stub agents.
# ---------------------------------------------------------------------------


class _StubAgent:
    def __init__(self, events: list[Any]) -> None:
        self._events = events

    async def run(self, prompt: str) -> Any:
        for event in self._events:
            yield event


class _BoomAgent:
    async def run(self, prompt: str) -> Any:
        raise RuntimeError("DSN=postgres://leak/secret")
        yield  # pragma: no cover — unreachable, keeps this a generator


def _terminate(*, final_message: str = "", reason: str = "complete") -> TerminateEvent:
    return TerminateEvent(
        reason=reason,
        iterations_used=1,
        final_confidence=1.0,
        total_tool_calls=0,
        final_message=final_message,
    )


def _think(reasoning: str) -> ThinkEvent:
    return ThinkEvent(iteration=0, reasoning=reasoning, tool_calls=[])


def _server_with(agent: Any, **kwargs: Any) -> A2AServer:
    server = A2AServer(agent=agent, allow_unauthenticated=True, **kwargs)
    _ = server.app
    return server


def _submitted_task(task_id: str = "t-open", context_id: str = "c-open") -> Task:
    return Task(
        id=task_id,
        contextId=context_id,
        status=TaskStatus(state=TaskState.submitted, timestamp="2026-05-25T12:00:00Z"),
        history=[],
    )


def _rpc(client: TestClient, method: str, params: dict[str, Any], **headers: str) -> dict[str, Any]:
    resp = client.post(
        "/",
        headers=headers or None,
        json={"jsonrpc": "2.0", "id": "rpc-1", "method": method, "params": params},
    )
    return resp.json()


# ---------------------------------------------------------------------------
# _extract_user_text — non-text parts.
# ---------------------------------------------------------------------------


class TestExtractUserText:
    def test_concatenates_text_file_and_data_parts(self) -> None:
        parts = [
            TextPart(text="hello"),
            FilePart(file=FileWithUri(uri="https://example.com/a.txt", name="a.txt")),
            FilePart(file=FileWithBytes(bytes="aGVsbG8=", mimeType="application/octet-stream")),
            DataPart(data={"k": "v"}),
        ]
        out = _extract_user_text(parts)
        lines = out.split("\n")
        assert lines[0] == "hello"
        assert lines[1] == "[file: a.txt]"
        # FileWithBytes has no name → falls back to uri sentinel "file".
        assert lines[2] == "[file: file]"
        assert json.loads(lines[3]) == {"k": "v"}


# ---------------------------------------------------------------------------
# _TaskStore.cancel
# ---------------------------------------------------------------------------


class TestTaskStoreCancel:
    def test_cancel_unknown_task_returns_false(self) -> None:
        store = _TaskStore()
        assert store.cancel("nope") is False

    def test_cancel_active_task_flips_to_canceled(self) -> None:
        store = _TaskStore()
        store.put(_submitted_task("t-1"))
        assert store.cancel("t-1") is True
        assert store.is_cancel_requested("t-1") is True
        assert store.get("t-1").status.state == TaskState.canceled

    def test_cancel_terminal_task_returns_false(self) -> None:
        store = _TaskStore()
        task = _submitted_task("t-done")
        task.status = TaskStatus(state=TaskState.completed, timestamp="2026-05-25T12:00:00Z")
        store.put(task)
        assert store.cancel("t-done") is False


# ---------------------------------------------------------------------------
# Skill normalisation + card building.
# ---------------------------------------------------------------------------


class TestNormaliseSkills:
    def test_plain_string_promoted_to_minimal_skill(self) -> None:
        skills = A2AServer._normalise_skills(["lookup"])
        assert len(skills) == 1
        assert skills[0].id == "lookup"
        assert skills[0].name == "lookup"
        assert skills[0].description == "lookup"


# ---------------------------------------------------------------------------
# Auth dependency.
# ---------------------------------------------------------------------------


class TestRequireAuth:
    async def test_dependency_returns_anon_when_no_key(self) -> None:
        server = A2AServer(agent=_StubAgent([]), allow_unauthenticated=True)
        dependency = server._require_auth()
        assert await dependency(authorization=None) == "anon"

    def test_invalid_bearer_token_returns_401(self) -> None:
        server = A2AServer(agent=_StubAgent([]), api_key="secret")
        client = TestClient(server.app)
        resp = client.get(
            "/.well-known/agent-card.json",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# _drive_agent cancellation.
# ---------------------------------------------------------------------------


class TestDriveAgentCancel:
    async def test_cancel_flag_short_circuits_run(self) -> None:
        server = _server_with(_StubAgent([_terminate(final_message="never")]))
        task = _submitted_task("t-cancel", "c-cancel")
        server._store.put(task)
        server._store._cancel_flags[task.id] = True
        events: list[Any] = []

        final = await server._drive_agent("hi", task=task, on_event=events.append)

        assert final == ""
        assert task.status.state == TaskState.canceled
        assert any(getattr(ev, "final", False) for ev in events)


# ---------------------------------------------------------------------------
# message/send error + lifecycle branches.
# ---------------------------------------------------------------------------


class TestMessageSend:
    def test_invalid_params_returns_invalid_params(self) -> None:
        client = TestClient(_server_with(_StubAgent([])).app)
        body = _rpc(client, "message/send", {})
        assert body["error"]["code"] == -32602  # INVALID_PARAMS

    def test_unknown_task_id_returns_task_not_found(self) -> None:
        client = TestClient(_server_with(_StubAgent([])).app)
        body = _rpc(
            client,
            "message/send",
            {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "hi"}],
                    "messageId": "m1",
                    "taskId": "ghost",
                }
            },
        )
        assert body["error"]["code"] == -32001  # TASK_NOT_FOUND

    def test_terminal_task_id_returns_not_cancelable(self) -> None:
        server = _server_with(_StubAgent([_terminate(final_message="done")]))
        client = TestClient(server.app)
        first = _rpc(
            client,
            "message/send",
            {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "hi"}],
                    "messageId": "m1",
                }
            },
        )
        task_id = first["result"]["id"]
        again = _rpc(
            client,
            "message/send",
            {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "more"}],
                    "messageId": "m2",
                    "taskId": task_id,
                }
            },
        )
        assert again["error"]["code"] == -32002  # TASK_NOT_CANCELABLE

    async def test_appends_to_active_task(self) -> None:
        server = _server_with(_StubAgent([_terminate(final_message="continued")]))
        existing = _submitted_task("t-live", "c-live")
        server._store.put(existing)
        result = await server._handle_message_send(
            {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "more"}],
                    "messageId": "m2",
                    "taskId": "t-live",
                }
            }
        )
        assert result["id"] == "t-live"
        assert result["contextId"] == "c-live"
        assert result["history"][-1]["parts"][0]["text"] == "more"
        assert result["status"]["state"] == "completed"

    def test_agent_failure_returns_internal_error_without_leak(self) -> None:
        client = TestClient(_server_with(_BoomAgent()).app)
        body = _rpc(
            client,
            "message/send",
            {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "hi"}],
                    "messageId": "m1",
                }
            },
        )
        assert body["error"]["code"] == -32603  # INTERNAL_ERROR
        assert "postgres" not in json.dumps(body)


# ---------------------------------------------------------------------------
# tasks/get + tasks/cancel branches.
# ---------------------------------------------------------------------------


class TestTasksGetCancel:
    def test_get_invalid_params(self) -> None:
        client = TestClient(_server_with(_StubAgent([])).app)
        body = _rpc(client, "tasks/get", {})
        assert body["error"]["code"] == -32602  # INVALID_PARAMS

    def test_get_history_length_trims(self) -> None:
        server = _server_with(_StubAgent([_terminate(final_message="ok")]))
        client = TestClient(server.app)
        sent = _rpc(
            client,
            "message/send",
            {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "hi"}],
                    "messageId": "m1",
                }
            },
        )
        task_id = sent["result"]["id"]
        body = _rpc(client, "tasks/get", {"id": task_id, "historyLength": 1})
        # The trimming branch ran; history keeps the single user turn.
        assert len(body["result"]["history"]) == 1

    def test_cancel_invalid_params(self) -> None:
        client = TestClient(_server_with(_StubAgent([])).app)
        body = _rpc(client, "tasks/cancel", {})
        assert body["error"]["code"] == -32602  # INVALID_PARAMS

    def test_cancel_unknown_returns_not_found(self) -> None:
        client = TestClient(_server_with(_StubAgent([])).app)
        body = _rpc(client, "tasks/cancel", {"id": "ghost"})
        assert body["error"]["code"] == -32001  # TASK_NOT_FOUND

    def test_cancel_active_task_returns_canceled_task(self) -> None:
        server = _server_with(_StubAgent([]))
        server._store.put(_submitted_task("t-active", "c-active"))
        client = TestClient(server.app)
        body = _rpc(client, "tasks/cancel", {"id": "t-active"})
        assert body["result"]["id"] == "t-active"
        assert body["result"]["status"]["state"] == "canceled"


# ---------------------------------------------------------------------------
# JSON-RPC dispatch — version + v1 method routing.
# ---------------------------------------------------------------------------


class TestJsonRpcDispatch:
    def test_unsupported_version_header_returns_version_not_supported(self) -> None:
        client = TestClient(_server_with(_StubAgent([])).app)
        body = _rpc(client, "message/send", {}, **{"A2A-Version": "9.9"})
        assert body["error"]["code"] == -32009  # VERSION_NOT_SUPPORTED

    def test_v1_get_task_round_trip(self) -> None:
        server = _server_with(_StubAgent([_terminate(final_message="hi")]))
        client = TestClient(server.app)
        sent = _rpc(
            client,
            "SendMessage",
            {"message": {"role": "ROLE_USER", "parts": [{"text": "hi"}], "messageId": "m1"}},
            **{"A2A-Version": "1.0"},
        )
        task_id = sent["result"]["task"]["id"]
        body = _rpc(client, "GetTask", {"id": task_id}, **{"A2A-Version": "1.0"})
        assert body["result"]["id"] == task_id

    def test_v1_cancel_task_on_terminal_returns_not_cancelable(self) -> None:
        server = _server_with(_StubAgent([_terminate(final_message="hi")]))
        client = TestClient(server.app)
        sent = _rpc(
            client,
            "SendMessage",
            {"message": {"role": "ROLE_USER", "parts": [{"text": "hi"}], "messageId": "m1"}},
            **{"A2A-Version": "1.0"},
        )
        task_id = sent["result"]["task"]["id"]
        body = _rpc(client, "CancelTask", {"id": task_id}, **{"A2A-Version": "1.0"})
        assert body["error"]["code"] == -32002  # TASK_NOT_CANCELABLE

    def test_v1_push_notification_unsupported(self) -> None:
        client = TestClient(_server_with(_StubAgent([])).app)
        body = _rpc(client, "CreateTaskPushNotificationConfig", {}, **{"A2A-Version": "1.0"})
        assert body["error"]["code"] == -32003  # PUSH_NOTIFICATION_NOT_SUPPORTED

    def test_v1_extended_agent_card_unsupported(self) -> None:
        client = TestClient(_server_with(_StubAgent([])).app)
        body = _rpc(client, "GetExtendedAgentCard", {}, **{"A2A-Version": "1.0"})
        assert body["error"]["code"] == -32004  # UNSUPPORTED_OPERATION

    def test_v1_subscribe_to_task_streams_snapshot(self) -> None:
        server = _server_with(_StubAgent([]))
        task = _submitted_task("t-sub", "c-sub")
        task.status = TaskStatus(state=TaskState.working, timestamp="2026-05-25T12:00:00Z")
        server._store.put(task)
        client = TestClient(server.app)
        with client.stream(
            "POST",
            "/",
            headers={"A2A-Version": "1.0"},
            json={
                "jsonrpc": "2.0",
                "id": "sub",
                "method": "SubscribeToTask",
                "params": {"id": "t-sub"},
            },
        ) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
        events = [
            json.loads(line.removeprefix("data: "))
            for line in body.split("\n\n")
            if line.startswith("data: ")
        ]
        assert events[0]["result"]["task"]["id"] == "t-sub"


# ---------------------------------------------------------------------------
# Streaming SSE paths (message/stream).
# ---------------------------------------------------------------------------


class TestStreamResponse:
    def test_message_stream_legacy_emits_events(self) -> None:
        server = _server_with(
            _StubAgent([_think(reasoning="thinking"), _terminate(final_message="final")])
        )
        client = TestClient(server.app)
        with client.stream(
            "POST",
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "ms",
                "method": "message/stream",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": "hi"}],
                        "messageId": "m1",
                    }
                },
            },
        ) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
        results = [
            json.loads(line.removeprefix("data: "))["result"]
            for line in body.split("\n\n")
            if line.startswith("data: ")
        ]
        # First event is the submitted task snapshot; later events carry updates.
        assert results[0]["status"]["state"] == "submitted"
        assert any(r.get("kind") == "artifact-update" for r in results)

    def test_message_stream_invalid_params_emits_rpc_error(self) -> None:
        client = TestClient(_server_with(_StubAgent([])).app)
        with client.stream(
            "POST",
            "/",
            json={"jsonrpc": "2.0", "id": "ms", "method": "message/stream", "params": {}},
        ) as resp:
            body = "".join(resp.iter_text())
        payload = next(
            json.loads(line.removeprefix("data: "))
            for line in body.split("\n\n")
            if line.startswith("data: ")
        )
        assert payload["error"]["code"] == -32602  # INVALID_PARAMS

    def test_stream_generic_error_is_sanitised(self) -> None:
        class _BoomStreamServer(A2AServer):
            async def _stream_message(self, params: dict[str, Any]) -> Any:
                raise ValueError("DSN=postgres://leak/secret")
                yield  # pragma: no cover — generator marker

        server = _BoomStreamServer(agent=_StubAgent([]), allow_unauthenticated=True)
        _ = server.app
        client = TestClient(server.app)
        with client.stream(
            "POST",
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "ms",
                "method": "message/stream",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": "hi"}],
                        "messageId": "m1",
                    }
                },
            },
        ) as resp:
            body = "".join(resp.iter_text())
        assert "postgres" not in body
        assert "internal error" in body
        assert "correlation_id" in body

    def test_runner_agent_error_marks_task_failed(self) -> None:
        server = _server_with(_BoomAgent())
        client = TestClient(server.app)
        with client.stream(
            "POST",
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "ms",
                "method": "message/stream",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": "hi"}],
                        "messageId": "m1",
                    }
                },
            },
        ) as resp:
            body = "".join(resp.iter_text())
        assert "postgres" not in body
        results = [
            json.loads(line.removeprefix("data: "))["result"]
            for line in body.split("\n\n")
            if line.startswith("data: ")
        ]
        assert any(r.get("status", {}).get("state") == "failed" for r in results)


# ---------------------------------------------------------------------------
# Legacy /a2a/stream non-think/terminate event branch.
# ---------------------------------------------------------------------------


class TestLegacyStreamOtherEvent:
    def test_other_event_type_passthrough(self) -> None:
        event = ToolStartEvent(tool_name="search", tool_call_id="c1", arguments={})
        server = _server_with(_StubAgent([event]))
        client = TestClient(server.app)
        with client.stream(
            "POST",
            "/a2a/stream",
            json={"messages": [{"role": "user", "content": "hi"}]},
        ) as resp:
            body = "".join(resp.iter_text())
        types_seen = [
            json.loads(line.removeprefix("data: "))["type"]
            for line in body.split("\n\n")
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        assert "tool_start" in types_seen


# ---------------------------------------------------------------------------
# App construction guards.
# ---------------------------------------------------------------------------


class TestAppConstruction:
    def test_no_api_key_logs_warning_and_builds_app(self) -> None:
        server = A2AServer(agent=_StubAgent([]))
        # api_key is None and allow_unauthenticated is False (default) — the
        # warning path runs but the app still builds for loopback use.
        assert server.app is not None

    def test_missing_fastapi_raises_clear_message(self) -> None:
        saved_fastapi = sys.modules.get("fastapi")
        saved_responses = sys.modules.get("fastapi.responses")
        sys.modules["fastapi"] = None  # type: ignore[assignment]
        try:
            server = A2AServer(agent=_StubAgent([]), allow_unauthenticated=True)
            with pytest.raises(ImportError, match="FastAPI required"):
                server._create_app()
        finally:
            if saved_fastapi is not None:
                sys.modules["fastapi"] = saved_fastapi
            else:
                sys.modules.pop("fastapi", None)
            if saved_responses is not None:
                sys.modules["fastapi.responses"] = saved_responses


# ---------------------------------------------------------------------------
# run() — uvicorn dispatch.
# ---------------------------------------------------------------------------


class TestRun:
    def test_run_invokes_uvicorn(self) -> None:
        captured: dict[str, Any] = {}
        fake = types.ModuleType("uvicorn")

        def fake_run(app: Any, **kwargs: Any) -> None:
            captured["app"] = app
            captured["kwargs"] = kwargs

        fake.run = fake_run  # type: ignore[attr-defined]
        saved = sys.modules.get("uvicorn")
        sys.modules["uvicorn"] = fake
        try:
            server = _server_with(_StubAgent([]))
            server.run(host="127.0.0.1", port=8123)
        finally:
            if saved is not None:
                sys.modules["uvicorn"] = saved
            else:
                sys.modules.pop("uvicorn", None)
        assert captured["kwargs"]["host"] == "127.0.0.1"
        assert captured["kwargs"]["port"] == 8123


# ---------------------------------------------------------------------------
# A2AClient — card fallback exhaustion + v1/legacy method splits.
# ---------------------------------------------------------------------------


_LEGACY_TASK_RESULT = {
    "id": "t-1",
    "contextId": "c-1",
    "status": {"state": "completed", "timestamp": "2026-01-01T00:00:00Z"},
    "history": [],
    "artifacts": [],
    "kind": "task",
}

_V1_TASK_RESULT = {
    "id": "t-1",
    "contextId": "c-1",
    "status": {"state": "TASK_STATE_COMPLETED", "timestamp": "2026-01-01T00:00:00Z"},
    "history": [],
    "artifacts": [],
}


def _rpc_response(result: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json={"jsonrpc": "2.0", "id": "x", "result": result})


class TestA2AClientCard:
    @respx.mock
    async def test_all_endpoints_unreachable_raises_runtime_error(self) -> None:
        respx.get("http://remote/.well-known/agent-card.json").mock(
            side_effect=httpx.ConnectError("down")
        )
        respx.get("http://remote/agent-card").mock(side_effect=httpx.ConnectError("down"))
        client = A2AClient(url="http://remote")
        with pytest.raises(RuntimeError, match="Could not fetch Agent Card"):
            await client.get_agent_card()


class TestA2AClientMethodSplits:
    @respx.mock
    async def test_send_message_legacy_protocol(self) -> None:
        route = respx.post("http://remote/").mock(return_value=_rpc_response(_LEGACY_TASK_RESULT))
        client = A2AClient(url="http://remote", protocol_version=None)
        msg = Message(role="user", parts=[TextPart(text="hi")], messageId="m1")
        task = await client.send_message(msg)
        sent = json.loads(route.calls.last.request.content)
        assert sent["method"] == "message/send"
        assert "A2A-Version" not in route.calls.last.request.headers
        assert isinstance(task, Task)
        assert task.status.state == TaskState.completed

    @respx.mock
    async def test_get_task_v1_with_history_length(self) -> None:
        route = respx.post("http://remote/").mock(return_value=_rpc_response(_V1_TASK_RESULT))
        client = A2AClient(url="http://remote")
        task = await client.get_task("t-1", history_length=5)
        sent = json.loads(route.calls.last.request.content)
        assert sent["method"] == "GetTask"
        assert sent["params"]["historyLength"] == 5
        assert task.id == "t-1"

    @respx.mock
    async def test_get_task_legacy_protocol(self) -> None:
        route = respx.post("http://remote/").mock(return_value=_rpc_response(_LEGACY_TASK_RESULT))
        client = A2AClient(url="http://remote", protocol_version=None)
        task = await client.get_task("t-1")
        sent = json.loads(route.calls.last.request.content)
        assert sent["method"] == "tasks/get"
        assert task.id == "t-1"

    @respx.mock
    async def test_list_tasks_forwards_page_token_and_history_length(self) -> None:
        route = respx.post("http://remote/").mock(
            return_value=_rpc_response(
                {"tasks": [], "nextPageToken": "", "pageSize": 1, "totalSize": 0}
            )
        )
        client = A2AClient(url="http://remote")
        tasks, token = await client.list_tasks(
            page_token="3",  # noqa: S106 — pagination cursor, not a credential
            history_length=2,
            page_size=1,
        )
        sent = json.loads(route.calls.last.request.content)
        assert sent["params"]["pageToken"] == "3"
        assert sent["params"]["historyLength"] == 2
        assert tasks == []
        assert token == ""

    async def test_list_tasks_requires_v1(self) -> None:
        client = A2AClient(url="http://remote", protocol_version=None)
        with pytest.raises(RuntimeError, match="requires A2A v1"):
            await client.list_tasks()

    @respx.mock
    async def test_cancel_task_v1(self) -> None:
        route = respx.post("http://remote/").mock(return_value=_rpc_response(_V1_TASK_RESULT))
        client = A2AClient(url="http://remote")
        task = await client.cancel_task("t-1")
        sent = json.loads(route.calls.last.request.content)
        assert sent["method"] == "CancelTask"
        assert task.id == "t-1"

    @respx.mock
    async def test_cancel_task_legacy(self) -> None:
        route = respx.post("http://remote/").mock(return_value=_rpc_response(_LEGACY_TASK_RESULT))
        client = A2AClient(url="http://remote", protocol_version=None)
        task = await client.cancel_task("t-1")
        sent = json.loads(route.calls.last.request.content)
        assert sent["method"] == "tasks/cancel"
        assert task.id == "t-1"


# ---------------------------------------------------------------------------
# A2AClient streaming.
# ---------------------------------------------------------------------------


class TestA2AClientStreaming:
    @respx.mock
    async def test_v1_streaming_consumes_sse_and_stops_on_done(self) -> None:
        status_update = {
            "statusUpdate": {
                "taskId": "t-1",
                "contextId": "c-1",
                "status": {
                    "state": "TASK_STATE_WORKING",
                    "timestamp": "2026-01-01T00:00:00Z",
                },
            }
        }
        env = {"jsonrpc": "2.0", "id": "x", "result": status_update}
        body = (
            f"data: {json.dumps(env)}\n\n"
            ": a comment line\n\n"
            "data: this-is-not-json\n\n"
            "data: [DONE]\n\n"
        )
        respx.post("http://remote/").mock(return_value=httpx.Response(200, content=body.encode()))
        client = A2AClient(url="http://remote")
        msg = Message(role="user", parts=[TextPart(text="hi")], messageId="m1")
        events = [ev async for ev in client.send_message_streaming(msg)]
        assert len(events) == 1
        assert events[0]["kind"] == "status-update"
        assert events[0]["status"]["state"] == "working"

    @respx.mock
    async def test_legacy_streaming_surfaces_error_envelope(self) -> None:
        success = {"jsonrpc": "2.0", "id": "x", "result": {"kind": "task", "id": "t-1"}}
        failure = {
            "jsonrpc": "2.0",
            "id": "x",
            "error": {"code": -32001, "message": "task t-1 not found"},
        }
        body = f"data: {json.dumps(success)}\n\ndata: {json.dumps(failure)}\n\n"
        respx.post("http://remote/").mock(return_value=httpx.Response(200, content=body.encode()))
        client = A2AClient(url="http://remote", protocol_version=None)
        msg = Message(role="user", parts=[TextPart(text="hi")], messageId="m1")
        events = [ev async for ev in client.send_message_streaming(msg)]
        assert events[0] == {"kind": "task", "id": "t-1"}
        assert "error" in events[1]


# ---------------------------------------------------------------------------
# A2AClient.as_tool — the wrapped callable actually round-trips.
# ---------------------------------------------------------------------------


class TestAsToolExecution:
    @respx.mock
    def test_call_remote_invokes_legacy_endpoint(self) -> None:
        respx.post("http://remote/a2a/invoke").mock(
            return_value=httpx.Response(
                200,
                json={
                    "messages": [{"role": "agent", "content": "tool answer", "metadata": {}}],
                    "status": "completed",
                    "metadata": {},
                },
            )
        )
        client = A2AClient(url="http://remote")
        tool = client.as_tool()
        result = tool("ping")
        assert result == "tool answer"
