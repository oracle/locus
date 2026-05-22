# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for ``locus.a2a`` — the spec-compliant A2A transport.

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

from locus.a2a import (
    A2AClient,
    A2AMessage,
    A2ARequest,
    A2AResponse,
    A2AServer,
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
from locus.a2a.protocol import _is_loopback
from locus.core.events import TerminateEvent, ThinkEvent


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
                    "the ``LOCUS_A2A_API_KEY`` environment variable."
                ),
            }
        }
        assert body["security"] == [{"bearerAuth": []}]

    def test_card_security_fields_null_when_unauthenticated(self) -> None:
        """Mirror: in ``allow_unauthenticated=True`` mode the card stays
        silent on auth so clients detect the open mode by absence."""
        server = _server_with(_StubAgent([]), name="N", description="d")
        client = TestClient(server.app)
        resp = client.get("/.well-known/agent-card.json")
        assert resp.status_code == 200
        body = resp.json()
        assert body["securitySchemes"] is None
        assert body["security"] is None


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

        monkeypatch.setattr("locus.core.config.get_settings", boom)
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
        assert isinstance(task, Task)
        assert task.status.state == TaskState.completed

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
