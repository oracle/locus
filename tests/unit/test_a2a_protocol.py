# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Unit tests for ``tulip.a2a`` — the spec-compliant A2A transport.

Covers both the public A2A protocol surface (Agent Card at the
well-known URL, JSON-RPC 2.0 method dispatch, the eight-state task
lifecycle, message parts) and the backward-compat aliases
(``/agent-card``, ``/a2a/invoke``, ``/a2a/stream``) preserved from the
pre-spec implementation so peers that haven't picked up the new wire
shape keep working.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from tulip.a2a import (
    A2AClient,
    A2AMessage,
    A2ARequest,
    A2AResponse,
    A2AServer,
    A2AV1ListTasksRequest,
    A2AV1ListTasksResponse,
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    DataPart,
    FilePart,
    FileWithBytes,
    Message,
    Task,
    TaskState,
    TextPart,
)
from tulip.a2a.protocol import _is_loopback
from tulip.a2a.protocol_v1 import (
    A2AV1ProtocolError,
    A2AV1ServerMixin,
    legacy_event_to_v1_stream_response,
    legacy_message_to_v1,
    legacy_part_to_v1,
    legacy_task_to_v1,
    task_result_to_legacy_payload,
    v1_message_to_legacy,
    v1_part_to_legacy,
    v1_stream_response_to_legacy_payload,
    v1_task_to_legacy_payload,
)
from tulip.a2a.spec import (
    CONTENT_TYPE_NOT_SUPPORTED,
    INVALID_PARAMS,
    TASK_NOT_FOUND,
    UNSUPPORTED_OPERATION,
    Artifact,
    FileWithUri,
    TaskArtifactUpdateEvent,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from tulip.a2a.spec_v1 import (
    A2AV1Message,
    A2AV1Part,
    A2AV1Role,
    A2AV1TaskState,
    A2AV1TaskStatus,
    A2AV1TaskStatusUpdateEvent,
)
from tulip.core.events import TerminateEvent, ThinkEvent


# ---------------------------------------------------------------------------
# Stubs for the agent the server wraps.
# ---------------------------------------------------------------------------


class _StubAgent:
    def __init__(self, events: list[Any]) -> None:
        self._events = events

    async def run(self, prompt: str) -> Any:
        for event in self._events:
            yield event


def _think(reasoning: str) -> ThinkEvent:
    return ThinkEvent(iteration=0, reasoning=reasoning, tool_calls=[])


def _terminate(*, final_message: str = "", reason: str = "complete") -> TerminateEvent:
    return TerminateEvent(
        reason=reason,
        iterations_used=1,
        final_confidence=1.0,
        total_tool_calls=0,
        final_message=final_message,
    )


def _server_with(agent: Any, **kwargs: Any) -> A2AServer:
    server = A2AServer(agent=agent, allow_unauthenticated=True, **kwargs)
    _ = server.app  # eager-initialise
    return server


def _legacy_task(
    task_id: str = "t-1",
    *,
    context_id: str = "c-1",
    state: TaskState = TaskState.completed,
    timestamp: str = "2026-05-25T12:00:00Z",
) -> Task:
    user_message = Message(
        role="user",
        parts=[TextPart(text="hello", metadata={"source": "test"})],
        messageId=f"{task_id}-m-user",
        contextId=context_id,
        referenceTaskIds=["ref-1"],
        extensions=["urn:test"],
        metadata={"turn": 1},
    )
    agent_message = Message(
        role="agent",
        parts=[TextPart(text="done")],
        messageId=f"{task_id}-m-agent",
        contextId=context_id,
    )
    return Task(
        id=task_id,
        contextId=context_id,
        status=TaskStatus(state=state, message=agent_message, timestamp=timestamp),
        history=[user_message, agent_message],
        artifacts=[
            Artifact(
                artifactId=f"{task_id}-artifact",
                name="answer",
                description="final answer",
                parts=[
                    TextPart(text=f"artifact {task_id}"),
                    DataPart(data={"id": task_id}),
                    FilePart(
                        file=FileWithUri(
                            uri="https://example.com/file.txt",
                            mimeType="text/plain",
                            name="file.txt",
                        )
                    ),
                    FilePart(
                        file=FileWithBytes(
                            bytes="aGVsbG8=",
                            mimeType="application/octet-stream",
                            name="hello.bin",
                        )
                    ),
                ],
                metadata={"artifact": True},
                extensions=["urn:artifact"],
            )
        ],
        metadata={"task": task_id},
    )


class _ListStore:
    def __init__(self, tasks: list[Task]) -> None:
        self._tasks = tasks

    def list(self) -> list[Task]:
        return list(self._tasks)

    def get(self, task_id: str) -> Task | None:
        return next((task for task in self._tasks if task.id == task_id), None)


class _V1Host(A2AV1ServerMixin):
    def __init__(
        self,
        tasks: list[Task] | None = None,
        *,
        input_modes: list[str] | None = None,
        stream_events: list[dict[str, Any]] | None = None,
    ) -> None:
        self._store = _ListStore(tasks or [_legacy_task()])
        self._input_modes = input_modes or ["text/plain", "application/json"]
        self.sent_params: dict[str, Any] | None = None
        self.get_params: dict[str, Any] | None = None
        self.cancel_params: dict[str, Any] | None = None
        self.stream_params: dict[str, Any] | None = None
        self._stream_events = stream_events or []

    def _build_card(self) -> Any:
        return AgentCard(
            name="host",
            description="test",
            url="http://testserver",
            defaultInputModes=self._input_modes,
        )

    async def _handle_message_send(self, params: dict[str, Any]) -> dict[str, Any]:
        self.sent_params = params
        return self._store.list()[0].model_dump(exclude_none=True)

    async def _handle_tasks_get(self, params: dict[str, Any]) -> dict[str, Any]:
        self.get_params = params
        task = self._store.get(params["id"]) or self._store.list()[0]
        return task.model_dump(exclude_none=True)

    async def _handle_tasks_cancel(self, params: dict[str, Any]) -> dict[str, Any]:
        self.cancel_params = params
        task = self._store.get(params["id"]) or self._store.list()[0]
        return task.model_dump(exclude_none=True)

    async def _stream_message(self, params: dict[str, Any]) -> Any:
        self.stream_params = params
        for event in self._stream_events:
            yield event


# ---------------------------------------------------------------------------
# Spec models — round-trip, defaults, discriminated parts.
# ---------------------------------------------------------------------------


class TestSpecModels:
    def test_agent_card_full_shape_round_trip(self) -> None:
        card = AgentCard(
            name="research",
            description="researches things",
            url="https://research.example.com",
            skills=[
                AgentSkill(
                    id="search",
                    name="Search",
                    description="Look up facts",
                    tags=["web"],
                ),
            ],
            capabilities=AgentCapabilities(streaming=True),
        )
        payload = card.model_dump()
        assert payload["name"] == "research"
        assert payload["url"] == "https://research.example.com"
        assert payload["capabilities"]["streaming"] is True
        assert payload["skills"][0]["id"] == "search"
        # ``defaultInputModes`` / ``defaultOutputModes`` ship on every card.
        assert payload["defaultInputModes"] == ["text/plain"]
        assert payload["defaultOutputModes"] == ["text/plain"]

    def test_message_parts_discriminated(self) -> None:
        msg = Message(
            role="user",
            parts=[
                TextPart(text="hi"),
                DataPart(data={"k": "v"}),
                FilePart(file=FileWithBytes(bytes="aGVsbG8=", name="hello.txt")),
            ],
            messageId="m1",
        )
        roundtrip = Message.model_validate(msg.model_dump())
        assert roundtrip.parts[0].kind == "text"
        assert roundtrip.parts[1].kind == "data"
        assert roundtrip.parts[2].kind == "file"

    def test_task_state_enum_has_all_eight_states(self) -> None:
        # Spec §6.3 — the canonical lifecycle states.
        assert {s.value for s in TaskState} == {
            "submitted",
            "working",
            "input-required",
            "completed",
            "canceled",
            "failed",
            "rejected",
            "auth-required",
        }

    def test_v1_list_tasks_models_are_public(self) -> None:
        req = A2AV1ListTasksRequest(contextId="c-1", pageSize=10)
        resp = A2AV1ListTasksResponse(tasks=[], pageSize=10, totalSize=0)
        assert req.contextId == "c-1"
        assert resp.nextPageToken == ""


class TestLegacyFlatModels:
    """The flat ``A2AMessage`` / ``A2ARequest`` / ``A2AResponse`` shapes
    are kept around so the pre-spec wire surface still works."""

    def test_a2a_message_default_metadata_empty(self) -> None:
        m = A2AMessage(role="user", content="hi")
        assert m.metadata == {}

    def test_a2a_request_response_round_trip(self) -> None:
        req = A2ARequest(messages=[A2AMessage(role="user", content="hi")])
        assert req.metadata == {}
        resp = A2AResponse(messages=[A2AMessage(role="agent", content="ok")])
        assert resp.status == "completed"


class TestIsLoopback:
    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.5"])
    def test_recognises_loopback_hosts(self, host: str) -> None:
        assert _is_loopback(host) is True

    @pytest.mark.parametrize("host", ["8.8.8.8", "192.0.2.1", "example.com"])
    def test_rejects_non_loopback(self, host: str) -> None:
        assert _is_loopback(host) is False

    def test_invalid_host_string_returns_false(self) -> None:
        assert _is_loopback("not-a-real-host") is False


# ---------------------------------------------------------------------------
# Server bind gate
# ---------------------------------------------------------------------------


class TestA2AServerBindGate:
    def test_anonymous_run_on_non_loopback_refused(self) -> None:
        server = A2AServer(agent=_StubAgent([]))
        with pytest.raises(RuntimeError, match="Refusing to bind"):
            server.run(host="8.8.8.8", port=9999)

    def test_uvicorn_missing_raises_clear_message(self) -> None:
        import sys

        saved = sys.modules.pop("uvicorn", None)
        sys.modules["uvicorn"] = None  # type: ignore[assignment]
        try:
            server = A2AServer(agent=_StubAgent([]))
            with pytest.raises(ImportError, match="uvicorn required"):
                server.run(host="127.0.0.1")
        finally:
            sys.modules.pop("uvicorn", None)
            if saved is not None:
                sys.modules["uvicorn"] = saved


# ---------------------------------------------------------------------------
# Spec-compliant routes
# ---------------------------------------------------------------------------


class TestWellKnownAgentCard:
    def test_well_known_card_payload(self) -> None:
        server = _server_with(
            _StubAgent([]),
            name="ResearchAgent",
            description="Does research.",
            skills=[
                AgentSkill(
                    id="lookup",
                    name="Lookup",
                    description="Find a fact",
                    tags=["search"],
                ),
            ],
            url="https://research.example.com",
        )
        client = TestClient(server.app)
        resp = client.get("/.well-known/agent-card.json")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "ResearchAgent"
        assert body["url"] == "https://research.example.com"
        assert body["capabilities"]["streaming"] is True
        # Skills are spec objects, not strings.
        assert isinstance(body["skills"], list)
        assert body["skills"][0]["id"] == "lookup"
        assert body["skills"][0]["tags"] == ["search"]

    def test_legacy_card_emits_string_skills(self) -> None:
        # ``/agent-card`` keeps the old flat ``skills: list[str]`` shape
        # so peers that pre-date the spec rewrite keep parsing.
        server = _server_with(
            _StubAgent([]),
            name="X",
            description="d",
            skills=[AgentSkill(id="a", name="A", description="A")],
        )
        client = TestClient(server.app)
        resp = client.get("/agent-card")
        assert resp.status_code == 200
        body = resp.json()
        assert body["skills"] == ["A"]

    def test_card_advertises_bearer_auth_when_api_key_configured(self) -> None:
        """Closes #214 — when the server enforces bearer auth, the AgentCard
        must declare it via ``securitySchemes`` / ``security`` so peers can
        discover the requirement from the well-known URL instead of via a
        401 on the first call."""
        server = A2AServer(agent=_StubAgent([]), api_key="secret", name="N", description="d")
        _ = server.app
        client = TestClient(server.app)
        resp = client.get(
            "/.well-known/agent-card.json",
            headers={"Authorization": "Bearer secret"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["securitySchemes"] == {
            "bearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "description": (
                    "Bearer token required on every route. "
                    "Set via the ``api_key=`` constructor argument or "
                    "the ``TULIP_A2A_API_KEY`` environment variable."
                ),
            }
        }
        assert body["security"] == [{"bearerAuth": []}]

    def test_card_security_fields_empty_when_unauthenticated(self) -> None:
        """Mirror: in ``allow_unauthenticated=True`` mode the card stays
        explicit about having no auth requirements without emitting nulls."""
        server = _server_with(_StubAgent([]), name="N", description="d")
        client = TestClient(server.app)
        resp = client.get("/.well-known/agent-card.json")
        assert resp.status_code == 200
        body = resp.json()
        assert body["securitySchemes"] == {}
        assert "security" not in body
        assert body["securityRequirements"] == []


class TestJsonRpcMessageSend:
    def test_message_send_returns_completed_task(self) -> None:
        agent = _StubAgent([_terminate(final_message="answer")])
        server = _server_with(agent)
        client = TestClient(server.app)
        resp = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "req-1",
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": "what is AI?"}],
                        "messageId": "m1",
                    }
                },
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["jsonrpc"] == "2.0"
        assert body["id"] == "req-1"
        task = body["result"]
        assert task["kind"] == "task"
        assert task["status"]["state"] == "completed"
        assert task["artifacts"][0]["parts"][0]["text"] == "answer"

    def test_unknown_method_yields_method_not_found(self) -> None:
        server = _server_with(_StubAgent([]))
        client = TestClient(server.app)
        resp = client.post(
            "/",
            json={"jsonrpc": "2.0", "id": "x", "method": "nope/method", "params": {}},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["error"]["code"] == -32601  # METHOD_NOT_FOUND

    def test_invalid_request_returns_invalid_request_error(self) -> None:
        server = _server_with(_StubAgent([]))
        client = TestClient(server.app)
        resp = client.post("/", json={"this is": "not json-rpc"})
        body = resp.json()
        assert body["error"]["code"] == -32600  # INVALID_REQUEST

    def test_push_notifications_return_unsupported(self) -> None:
        server = _server_with(_StubAgent([]))
        client = TestClient(server.app)
        resp = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "1",
                "method": "tasks/pushNotificationConfig/set",
                "params": {},
            },
        )
        body = resp.json()
        assert body["error"]["code"] == -32003  # PUSH_NOTIFICATION_NOT_SUPPORTED


class TestJsonRpcTaskLifecycle:
    def _send_then_get(self, client: TestClient) -> dict[str, Any]:
        send = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "send-1",
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": "hi"}],
                        "messageId": "m1",
                    }
                },
            },
        )
        return send.json()["result"]

    def test_tasks_get_returns_known_task(self) -> None:
        server = _server_with(_StubAgent([_terminate(final_message="ok")]))
        client = TestClient(server.app)
        task = self._send_then_get(client)
        resp = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "g-1",
                "method": "tasks/get",
                "params": {"id": task["id"]},
            },
        )
        body = resp.json()
        assert body["result"]["id"] == task["id"]

    def test_tasks_get_unknown_returns_task_not_found(self) -> None:
        server = _server_with(_StubAgent([]))
        client = TestClient(server.app)
        resp = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "g",
                "method": "tasks/get",
                "params": {"id": "no-such-task"},
            },
        )
        body = resp.json()
        assert body["error"]["code"] == -32001  # TASK_NOT_FOUND

    def test_tasks_cancel_terminal_returns_not_cancelable(self) -> None:
        server = _server_with(_StubAgent([_terminate(final_message="done")]))
        client = TestClient(server.app)
        task = self._send_then_get(client)
        # Task is already in completed state — cancelling must error.
        resp = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "c",
                "method": "tasks/cancel",
                "params": {"id": task["id"]},
            },
        )
        body = resp.json()
        assert body["error"]["code"] == -32002  # TASK_NOT_CANCELABLE


class TestJsonRpcV1:
    def test_v1_send_message_returns_v1_task_wrapper(self) -> None:
        server = _server_with(_StubAgent([_terminate(final_message="answer")]))
        client = TestClient(server.app)
        resp = client.post(
            "/",
            headers={"A2A-Version": "1.0"},
            json={
                "jsonrpc": "2.0",
                "id": "req-v1",
                "method": "SendMessage",
                "params": {
                    "message": {
                        "role": "ROLE_USER",
                        "parts": [{"text": "what is AI?"}],
                        "messageId": "m1",
                    }
                },
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        task = body["result"]["task"]
        assert body["id"] == "req-v1"
        assert task["status"]["state"] == "TASK_STATE_COMPLETED"
        assert task["history"][0]["role"] == "ROLE_USER"
        assert task["artifacts"][0]["parts"][0]["text"] == "answer"
        assert "kind" not in task

    def test_v1_list_tasks_filters_and_paginates(self) -> None:
        server = _server_with(_StubAgent([_terminate(final_message="one")]))
        client = TestClient(server.app)
        client.post(
            "/",
            headers={"A2A-Version": "1.0"},
            json={
                "jsonrpc": "2.0",
                "id": "send-1",
                "method": "SendMessage",
                "params": {
                    "message": {
                        "role": "ROLE_USER",
                        "parts": [{"text": "hi"}],
                        "messageId": "m1",
                        "contextId": "c-1",
                    }
                },
            },
        ).json()["result"]["task"]
        server._agent = _StubAgent([_terminate(final_message="two")])
        client.post(
            "/",
            headers={"A2A-Version": "1.0"},
            json={
                "jsonrpc": "2.0",
                "id": "send-2",
                "method": "SendMessage",
                "params": {
                    "message": {
                        "role": "ROLE_USER",
                        "parts": [{"text": "hi again"}],
                        "messageId": "m2",
                        "contextId": "c-1",
                    }
                },
            },
        )
        resp = client.post(
            "/",
            headers={"A2A-Version": "1.0"},
            json={
                "jsonrpc": "2.0",
                "id": "list-1",
                "method": "ListTasks",
                "params": {
                    "contextId": "c-1",
                    "status": "TASK_STATE_COMPLETED",
                    "pageSize": 1,
                    "includeArtifacts": False,
                },
            },
        )
        body = resp.json()["result"]
        assert body["totalSize"] == 2
        assert body["pageSize"] == 1
        assert body["nextPageToken"] == "1"
        assert len(body["tasks"]) == 1
        assert body["tasks"][0]["status"]["state"] == "TASK_STATE_COMPLETED"
        assert body["tasks"][0]["artifacts"] == []

    def test_v1_stream_response_uses_stream_response_oneof(self) -> None:
        server = _server_with(
            _StubAgent([_think(reasoning="thinking..."), _terminate(final_message="final")])
        )
        client = TestClient(server.app)
        with client.stream(
            "POST",
            "/",
            headers={"A2A-Version": "1.0"},
            json={
                "jsonrpc": "2.0",
                "id": "stream-v1",
                "method": "SendStreamingMessage",
                "params": {
                    "message": {
                        "role": "ROLE_USER",
                        "parts": [{"text": "hi"}],
                        "messageId": "m1",
                    }
                },
            },
        ) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
        events = [
            json.loads(line.removeprefix("data: "))
            for line in body.split("\n\n")
            if line.startswith("data: ")
        ]
        results = [event["result"] for event in events]
        assert "task" in results[0]
        assert any("statusUpdate" in result for result in results)
        assert any("artifactUpdate" in result for result in results)
        assert all("final" not in result.get("statusUpdate", {}) for result in results)


class TestA2AV1ProtocolAdapters:
    def test_part_conversions_cover_text_data_url_raw_and_empty(self) -> None:
        text = v1_part_to_legacy(A2AV1Part(text="hi", metadata={"m": 1}))
        data = v1_part_to_legacy(A2AV1Part(data={"x": 1}, metadata={"m": 2}))
        url = v1_part_to_legacy(
            A2AV1Part(url="https://example.com/a.txt", filename="a.txt", mediaType="text/plain")
        )
        raw = v1_part_to_legacy(
            A2AV1Part(raw="aGVsbG8=", filename="a.bin", mediaType="application/octet-stream")
        )
        empty = v1_part_to_legacy(A2AV1Part(metadata={"empty": True}))

        assert isinstance(text, TextPart)
        assert isinstance(data, DataPart)
        assert isinstance(url, FilePart)
        assert isinstance(raw, FilePart)
        assert isinstance(empty, TextPart)
        assert empty.text == ""

        assert legacy_part_to_v1(text).text == "hi"
        assert legacy_part_to_v1(data).data == {"x": 1}
        assert legacy_part_to_v1(url).url == "https://example.com/a.txt"
        assert legacy_part_to_v1(raw).raw == "aGVsbG8="
        assert legacy_part_to_v1(object()).text == ""  # type: ignore[arg-type]

    def test_message_and_task_payload_conversions_round_trip(self) -> None:
        legacy_task = _legacy_task()
        v1_task = legacy_task_to_v1(legacy_task)
        legacy_payload = v1_task_to_legacy_payload(v1_task)

        assert v1_task["status"]["state"] == "TASK_STATE_COMPLETED"
        assert v1_task["history"][0]["role"] == "ROLE_USER"
        assert v1_task["history"][1]["role"] == "ROLE_AGENT"
        assert v1_task["artifacts"][0]["parts"][2]["url"] == "https://example.com/file.txt"
        assert legacy_payload["kind"] == "task"
        assert legacy_payload["status"]["state"] == "completed"
        assert legacy_payload["history"][0]["kind"] == "message"
        assert legacy_payload["artifacts"][0]["parts"][3]["kind"] == "file"

        v1_message = legacy_message_to_v1(legacy_task.history[0])
        assert v1_message.role == A2AV1Role.user
        assert v1_message.referenceTaskIds == ["ref-1"]
        assert v1_message.extensions == ["urn:test"]
        assert v1_message_to_legacy(v1_message).role == "user"

    def test_task_result_and_stream_response_conversions(self) -> None:
        legacy_task = _legacy_task()
        v1_task = legacy_task_to_v1(legacy_task)

        assert task_result_to_legacy_payload(legacy_task.model_dump(exclude_none=True))["kind"] == (
            "task"
        )
        assert task_result_to_legacy_payload({"task": v1_task})["status"]["state"] == "completed"

        task_stream = v1_stream_response_to_legacy_payload({"task": v1_task})
        status_stream = v1_stream_response_to_legacy_payload(
            {
                "statusUpdate": {
                    "taskId": "t-1",
                    "contextId": "c-1",
                    "status": {
                        "state": "TASK_STATE_WORKING",
                        "timestamp": "2026-05-25T12:01:00Z",
                    },
                    "metadata": {"step": 1},
                }
            }
        )
        artifact_stream = v1_stream_response_to_legacy_payload(
            {
                "artifactUpdate": {
                    "taskId": "t-1",
                    "contextId": "c-1",
                    "artifact": {
                        "artifactId": "a-1",
                        "parts": [{"text": "chunk"}],
                    },
                    "append": True,
                    "lastChunk": True,
                }
            }
        )
        message_stream = v1_stream_response_to_legacy_payload(
            {
                "message": {
                    "role": "ROLE_AGENT",
                    "parts": [{"text": "hello"}],
                    "messageId": "m-1",
                }
            }
        )

        assert task_stream["kind"] == "task"
        assert status_stream["kind"] == "status-update"
        assert status_stream["status"]["state"] == "working"
        assert artifact_stream["kind"] == "artifact-update"
        assert artifact_stream["artifact"]["parts"][0]["text"] == "chunk"
        assert message_stream["role"] == "agent"
        assert v1_stream_response_to_legacy_payload({"other": "value"}) == {"other": "value"}

    def test_legacy_event_conversions_cover_all_stream_variants(self) -> None:
        task_event = legacy_event_to_v1_stream_response(
            _legacy_task().model_dump(exclude_none=True)
        )
        status_event = legacy_event_to_v1_stream_response(
            TaskStatusUpdateEvent(
                taskId="t-1",
                contextId="c-1",
                status=TaskStatus(
                    state=TaskState.working,
                    timestamp="2026-05-25T12:01:00Z",
                ),
                metadata={"status": True},
            ).model_dump(exclude_none=True)
        )
        artifact_event = legacy_event_to_v1_stream_response(
            TaskArtifactUpdateEvent(
                taskId="t-1",
                contextId="c-1",
                artifact=Artifact(artifactId="a-1", parts=[TextPart(text="chunk")]),
                append=True,
                lastChunk=True,
                metadata={"artifact": True},
            ).model_dump(exclude_none=True)
        )

        assert task_event["task"]["status"]["state"] == "TASK_STATE_COMPLETED"
        assert status_event["statusUpdate"]["status"]["state"] == "TASK_STATE_WORKING"
        assert "final" not in status_event["statusUpdate"]
        assert artifact_event["artifactUpdate"]["artifact"]["parts"][0]["text"] == "chunk"
        assert legacy_event_to_v1_stream_response({"kind": "custom"}) == {"kind": "custom"}

    @pytest.mark.asyncio
    async def test_v1_send_get_cancel_and_stream_delegate_to_legacy_host(self) -> None:
        task = _legacy_task()
        host = _V1Host(
            [task],
            stream_events=[
                task.model_dump(exclude_none=True),
                TaskStatusUpdateEvent(
                    taskId=task.id,
                    contextId=task.contextId,
                    status=TaskStatus(
                        state=TaskState.working,
                        timestamp="2026-05-25T12:02:00Z",
                    ),
                ).model_dump(exclude_none=True),
            ],
        )
        params = {
            "message": {
                "role": "ROLE_USER",
                "parts": [{"text": "hello"}],
                "messageId": "m-send",
            },
            "configuration": {"history_length": 2, "blocking": True},
            "metadata": {"request": True},
        }

        sent = await host._handle_v1_send_message(params)
        got = await host._handle_v1_get_task(
            {"id": task.id, "history_length": 1, "metadata": {"get": True}}
        )
        canceled = await host._handle_v1_cancel_task({"id": task.id, "metadata": {"cancel": True}})
        streamed = [event async for event in host._stream_v1_message(params)]

        assert sent["task"]["id"] == task.id
        assert host.sent_params is not None
        assert host.sent_params["message"]["role"] == "user"
        assert host.sent_params["configuration"]["historyLength"] == 2
        # GetTask / CancelTask return the Task directly. SendMessage stays
        # wrapped because its response shape is a oneof {task | message}.
        assert got["id"] == task.id
        assert "task" not in got
        assert host.get_params == {
            "id": task.id,
            "historyLength": 1,
            "metadata": {"get": True},
        }
        assert canceled["id"] == task.id
        assert "task" not in canceled
        assert host.cancel_params == {"id": task.id, "metadata": {"cancel": True}}
        assert "task" in streamed[0]
        assert "statusUpdate" in streamed[1]

    @pytest.mark.asyncio
    async def test_v1_list_tasks_filters_paginates_and_trims(self) -> None:
        old = _legacy_task(
            "old",
            context_id="keep",
            state=TaskState.completed,
            timestamp="2026-05-25T12:00:00Z",
        )
        new = _legacy_task(
            "new",
            context_id="keep",
            state=TaskState.completed,
            timestamp="2026-05-25T13:00:00Z",
        )
        other_context = _legacy_task(
            "other",
            context_id="drop",
            state=TaskState.completed,
            timestamp="2026-05-25T14:00:00Z",
        )
        working = _legacy_task(
            "working",
            context_id="keep",
            state=TaskState.working,
            timestamp="2026-05-25T15:00:00Z",
        )
        host = _V1Host([old, new, other_context, working])

        first_page = await host._handle_v1_list_tasks(
            {
                "context_id": "keep",
                "status": "TASK_STATE_COMPLETED",
                "status_timestamp_after": "2026-05-25T11:00:00Z",
                "page_size": 1,
                "page_token": "0",
                "history_length": 1,
                "include_artifacts": False,
            }
        )
        second_page = await host._handle_v1_list_tasks(
            {
                "contextId": "keep",
                "status": "TASK_STATE_COMPLETED",
                "pageSize": 1,
                "pageToken": first_page["nextPageToken"],
            }
        )
        all_tasks = await host._handle_v1_list_tasks({})

        assert first_page["totalSize"] == 2
        assert first_page["tasks"][0]["id"] == "new"
        assert first_page["tasks"][0]["history"][0]["role"] == "ROLE_AGENT"
        assert first_page["tasks"][0]["artifacts"] == []
        assert first_page["nextPageToken"] == "1"
        assert second_page["tasks"][0]["id"] == "old"
        assert all_tasks["pageSize"] == 4
        assert all_tasks["nextPageToken"] == ""

    @pytest.mark.asyncio
    async def test_v1_list_tasks_compares_timestamp_offsets_by_instant(self) -> None:
        same_instant = _legacy_task(
            "same-instant",
            timestamp="2026-05-25T15:00:00+02:00",
        )
        later = _legacy_task(
            "later",
            timestamp="2026-05-25T13:30:00Z",
        )
        earliest = _legacy_task(
            "earliest",
            timestamp="2026-05-25T14:00:00+02:00",
        )
        host = _V1Host([same_instant, later, earliest])

        response = await host._handle_v1_list_tasks(
            {"status_timestamp_after": "2026-05-25T13:00:00+00:00"}
        )

        assert [task["id"] for task in response["tasks"]] == ["later"]

    @pytest.mark.asyncio
    async def test_v1_list_tasks_rejects_invalid_params_and_page_token(self) -> None:
        host = _V1Host()

        with pytest.raises(A2AV1ProtocolError) as missing_id:
            await host._handle_v1_get_task({})
        with pytest.raises(A2AV1ProtocolError) as invalid_page:
            await host._handle_v1_list_tasks({"pageToken": "not-an-int"})

        assert missing_id.value.code == INVALID_PARAMS
        assert invalid_page.value.code == INVALID_PARAMS
        assert invalid_page.value.message == "invalid pageToken"

    @pytest.mark.asyncio
    async def test_v1_subscribe_preflight_and_stream_errors(self) -> None:
        active = _legacy_task("active", state=TaskState.working)
        done = _legacy_task("done", state=TaskState.completed)
        host = _V1Host([active, done])

        host._preflight_v1_task_subscription({"id": "active"})
        streamed = [event async for event in host._stream_v1_task_subscription({"id": "active"})]
        assert streamed[0]["task"]["id"] == "active"

        with pytest.raises(A2AV1ProtocolError) as missing:
            host._preflight_v1_task_subscription({"id": "missing"})
        with pytest.raises(A2AV1ProtocolError) as terminal:
            host._preflight_v1_task_subscription({"id": "done"})
        with pytest.raises(A2AV1ProtocolError) as invalid:
            [event async for event in host._stream_v1_task_subscription({})]

        assert missing.value.code == TASK_NOT_FOUND
        assert terminal.value.code == UNSUPPORTED_OPERATION
        assert invalid.value.code == INVALID_PARAMS

    @pytest.mark.asyncio
    async def test_v1_input_mode_validation_rejects_unsupported_parts(self) -> None:
        host = _V1Host(input_modes=["text/plain"])

        with pytest.raises(A2AV1ProtocolError) as data_part:
            await host._handle_v1_send_message(
                {
                    "message": {
                        "role": "ROLE_USER",
                        "parts": [{"data": {"x": 1}, "mediaType": "application/json"}],
                        "messageId": "m-1",
                    }
                }
            )
        with pytest.raises(A2AV1ProtocolError) as raw_part:
            await host._handle_v1_send_message(
                {
                    "message": {
                        "role": "ROLE_USER",
                        "parts": [{"raw": "aGVsbG8="}],
                        "messageId": "m-2",
                    }
                }
            )
        with pytest.raises(A2AV1ProtocolError) as bad_stream:
            [event async for event in host._stream_v1_message({"message": "not-valid"})]

        assert data_part.value.code == CONTENT_TYPE_NOT_SUPPORTED
        assert raw_part.value.code == CONTENT_TYPE_NOT_SUPPORTED
        assert bad_stream.value.code == INVALID_PARAMS

    def test_v1_stream_response_status_model_accepts_v1_status(self) -> None:
        event = A2AV1TaskStatusUpdateEvent(
            taskId="t-1",
            contextId="c-1",
            status=A2AV1TaskStatus(
                state=A2AV1TaskState.auth_required,
                message=A2AV1Message(
                    role=A2AV1Role.agent,
                    parts=[A2AV1Part(text="auth please")],
                    messageId="m-auth",
                ),
            ),
        )

        legacy = v1_stream_response_to_legacy_payload(
            {"statusUpdate": event.model_dump(exclude_none=True)}
        )
        assert legacy["status"]["state"] == "auth-required"
        assert legacy["status"]["message"]["parts"][0]["text"] == "auth please"


# ---------------------------------------------------------------------------
# Backward-compat invoke / stream
# ---------------------------------------------------------------------------


class TestLegacyInvokeStream:
    def test_invoke_extracts_last_user_message(self) -> None:
        agent = _StubAgent([_terminate(final_message="answer", reason="complete")])
        server = _server_with(agent)
        client = TestClient(server.app)
        resp = client.post(
            "/a2a/invoke",
            json={
                "messages": [
                    {"role": "user", "content": "earlier"},
                    {"role": "agent", "content": "ignored"},
                    {"role": "user", "content": "latest question"},
                ]
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["messages"][0]["content"] == "answer"
        assert body["status"] == "completed"

    def test_legacy_stream_emits_text_done_and_terminator(self) -> None:
        agent = _StubAgent([_think(reasoning="thinking..."), _terminate(final_message="final")])
        server = _server_with(agent)
        client = TestClient(server.app)
        with client.stream(
            "POST",
            "/a2a/stream",
            json={"messages": [{"role": "user", "content": "hi"}]},
        ) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
        events = [
            json.loads(line.removeprefix("data: "))
            for line in body.split("\n\n")
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        types = [e["type"] for e in events]
        assert "text" in types
        assert "done" in types
        assert "[DONE]" in body

    def test_legacy_stream_sanitises_agent_errors(self) -> None:
        class _BoomAgent:
            async def run(self, prompt: str) -> Any:
                raise RuntimeError("DSN=postgres://leak/secret")
                yield  # pragma: no cover (unreachable)

        server = _server_with(_BoomAgent())
        client = TestClient(server.app)
        with client.stream(
            "POST",
            "/a2a/stream",
            json={"messages": [{"role": "user", "content": "hi"}]},
        ) as resp:
            body = "".join(resp.iter_text())
        assert "DSN=postgres" not in body
        assert "internal error" in body
        assert "correlation_id" in body


# ---------------------------------------------------------------------------
# Auth (every route, including the well-known card)
# ---------------------------------------------------------------------------


class TestA2AServerAuth:
    @pytest.mark.parametrize(
        ("method", "path", "kwargs"),
        [
            ("GET", "/.well-known/agent-card.json", {}),
            ("GET", "/agent-card", {}),
            (
                "POST",
                "/",
                {"json": {"jsonrpc": "2.0", "id": "1", "method": "message/send"}},
            ),
            ("POST", "/a2a/invoke", {"json": {"messages": []}}),
        ],
    )
    def test_missing_bearer_token_returns_401(
        self, method: str, path: str, kwargs: dict[str, Any]
    ) -> None:
        server = A2AServer(agent=_StubAgent([]), api_key="secret")
        client = TestClient(server.app)
        resp = client.request(method, path, **kwargs)
        assert resp.status_code == 401

    def test_valid_bearer_token_returns_200(self) -> None:
        server = A2AServer(agent=_StubAgent([]), api_key="secret")
        client = TestClient(server.app)
        resp = client.get(
            "/.well-known/agent-card.json", headers={"Authorization": "Bearer secret"}
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Settings docs gate
# ---------------------------------------------------------------------------


class TestDocsGate:
    def test_docs_disabled_when_settings_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom() -> None:
            raise RuntimeError("no settings")

        monkeypatch.setattr("tulip.core.config.get_settings", boom)
        server = A2AServer(agent=_StubAgent([]), allow_unauthenticated=True)
        assert server._resolve_docs_enabled() is False


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class TestA2AClient:
    @pytest.mark.asyncio
    @respx.mock
    async def test_get_agent_card_prefers_well_known(self) -> None:
        respx.get("http://remote/.well-known/agent-card.json").mock(
            return_value=httpx.Response(
                200,
                json={
                    "name": "x",
                    "description": "d",
                    "url": "http://remote",
                    "skills": [],
                    "capabilities": {"streaming": True},
                    "defaultInputModes": ["text/plain"],
                    "defaultOutputModes": ["text/plain"],
                },
            )
        )
        client = A2AClient(url="http://remote/")
        card = await client.get_agent_card()
        assert card.name == "x"
        assert card.capabilities.streaming is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_get_agent_card_falls_back_to_legacy(self) -> None:
        respx.get("http://remote/.well-known/agent-card.json").mock(
            return_value=httpx.Response(404)
        )
        respx.get("http://remote/agent-card").mock(
            return_value=httpx.Response(
                200,
                json={"name": "x", "description": "d", "skills": ["a", "b"]},
            )
        )
        client = A2AClient(url="http://remote")
        card = await client.get_agent_card()
        assert card.name == "x"
        # Legacy string skills got promoted to objects with id == name.
        assert [s.id for s in card.skills] == ["a", "b"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_send_message_round_trip(self) -> None:
        route = respx.post("http://remote/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": "x",
                    "result": {
                        "id": "t-1",
                        "contextId": "c-1",
                        "status": {"state": "completed", "timestamp": "2026-01-01T00:00:00Z"},
                        "history": [],
                        "artifacts": [
                            {
                                "artifactId": "a-1",
                                "parts": [{"kind": "text", "text": "answer"}],
                            }
                        ],
                        "kind": "task",
                    },
                },
            )
        )
        client = A2AClient(url="http://remote")
        msg = Message(role="user", parts=[TextPart(text="hi")], messageId="m1")
        task = await client.send_message(msg)
        assert route.calls.last.request.headers["A2A-Version"] == "1.0"
        assert json.loads(route.calls.last.request.content)["method"] == "SendMessage"
        assert isinstance(task, Task)
        assert task.status.state == TaskState.completed

    @pytest.mark.asyncio
    @respx.mock
    async def test_list_tasks_round_trip(self) -> None:
        route = respx.post("http://remote/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": "x",
                    "result": {
                        "tasks": [
                            {
                                "id": "t-1",
                                "contextId": "c-1",
                                "status": {
                                    "state": "TASK_STATE_COMPLETED",
                                    "timestamp": "2026-01-01T00:00:00Z",
                                },
                                "history": [],
                                "artifacts": [],
                            }
                        ],
                        "nextPageToken": "2",
                        "pageSize": 1,
                        "totalSize": 3,
                    },
                },
            )
        )
        client = A2AClient(url="http://remote")
        tasks, next_page = await client.list_tasks(
            context_id="c-1",
            status=TaskState.completed,
            page_size=1,
            include_artifacts=False,
        )
        sent = json.loads(route.calls.last.request.content)
        assert route.calls.last.request.headers["A2A-Version"] == "1.0"
        assert sent["method"] == "ListTasks"
        assert sent["params"]["status"] == "TASK_STATE_COMPLETED"
        assert sent["params"]["contextId"] == "c-1"
        assert tasks[0].id == "t-1"
        assert tasks[0].status.state == TaskState.completed
        assert next_page == "2"

    @pytest.mark.asyncio
    @respx.mock
    async def test_get_task_propagates_error(self) -> None:
        respx.post("http://remote/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": "x",
                    "error": {"code": -32001, "message": "task t-1 not found"},
                },
            )
        )
        client = A2AClient(url="http://remote")
        with pytest.raises(RuntimeError, match="task t-1 not found"):
            await client.get_task("t-1")

    @pytest.mark.asyncio
    @respx.mock
    async def test_legacy_invoke_returns_last_agent_message(self) -> None:
        respx.post("http://remote/a2a/invoke").mock(
            return_value=httpx.Response(
                200,
                json={
                    "messages": [
                        {"role": "user", "content": "hi", "metadata": {}},
                        {"role": "agent", "content": "answer", "metadata": {}},
                    ],
                    "status": "completed",
                    "metadata": {},
                },
            )
        )
        client = A2AClient(url="http://remote")
        result = await client.invoke("hi")
        assert result == "answer"

    @pytest.mark.asyncio
    @respx.mock
    async def test_auth_headers_propagated(self) -> None:
        route = respx.get("http://remote/.well-known/agent-card.json").mock(
            return_value=httpx.Response(
                200,
                json={
                    "name": "x",
                    "description": "d",
                    "url": "http://remote",
                    "skills": [],
                },
            )
        )
        client = A2AClient(url="http://remote", api_key="secret")
        await client.get_agent_card()
        assert route.called
        assert route.calls.last.request.headers["Authorization"] == "Bearer secret"

    @pytest.mark.asyncio
    async def test_no_api_key_yields_no_auth_header(self) -> None:
        client = A2AClient(url="http://remote")
        assert client._auth_headers() == {}

    def test_as_tool_returns_callable_tool(self) -> None:
        client = A2AClient(url="http://remote")
        tool = client.as_tool(name="custom", description="custom desc")
        assert tool.name == "custom"
        assert tool.description == "custom desc"

    def test_default_timeout_matches_class_constant(self) -> None:
        client = A2AClient(url="http://remote")
        assert client._timeout == A2AClient.DEFAULT_TIMEOUT

    def test_constructor_timeout_override_is_stored(self) -> None:
        client = A2AClient(url="http://remote", timeout=600.0)
        assert client._timeout == 600.0

    def test_constructor_accepts_httpx_timeout_object(self) -> None:
        t = httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=5.0)
        client = A2AClient(url="http://remote", timeout=t)
        assert client._timeout is t

    @pytest.mark.asyncio
    @respx.mock
    async def test_send_message_uses_configured_timeout(self) -> None:
        """Per-call timeout=None means the client's configured timeout flows
        through to httpx.AsyncClient."""
        captured: dict[str, object] = {}

        class _SpyAsyncClient(httpx.AsyncClient):
            def __init__(self, *args: object, **kwargs: object) -> None:
                captured["timeout"] = kwargs.get("timeout")
                super().__init__(*args, **kwargs)  # type: ignore[arg-type]

        respx.post("http://remote/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": "x",
                    "result": {
                        "id": "t-1",
                        "contextId": "c-1",
                        "status": {"state": "completed", "timestamp": "2026-01-01T00:00:00Z"},
                        "history": [],
                        "artifacts": [],
                        "kind": "task",
                    },
                },
            )
        )

        from unittest.mock import patch

        client = A2AClient(url="http://remote", timeout=600.0)
        msg = Message(role="user", parts=[TextPart(text="hi")], messageId="m1")
        with patch("httpx.AsyncClient", _SpyAsyncClient):
            await client.send_message(msg)
        assert captured["timeout"] == 600.0

    @pytest.mark.asyncio
    @respx.mock
    async def test_send_message_per_call_timeout_overrides_default(self) -> None:
        captured: dict[str, object] = {}

        class _SpyAsyncClient(httpx.AsyncClient):
            def __init__(self, *args: object, **kwargs: object) -> None:
                captured["timeout"] = kwargs.get("timeout")
                super().__init__(*args, **kwargs)  # type: ignore[arg-type]

        respx.post("http://remote/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": "x",
                    "result": {
                        "id": "t-1",
                        "contextId": "c-1",
                        "status": {"state": "completed", "timestamp": "2026-01-01T00:00:00Z"},
                        "history": [],
                        "artifacts": [],
                        "kind": "task",
                    },
                },
            )
        )

        from unittest.mock import patch

        client = A2AClient(url="http://remote")  # default 120s
        msg = Message(role="user", parts=[TextPart(text="hi")], messageId="m1")
        with patch("httpx.AsyncClient", _SpyAsyncClient):
            await client.send_message(msg, timeout=900.0)
        assert captured["timeout"] == 900.0

    def test_as_tool_default_name(self) -> None:
        tool = A2AClient(url="http://remote").as_tool()
        assert tool.name == "remote_agent"
