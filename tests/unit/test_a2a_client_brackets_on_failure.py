# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""A2A client calls and the flat routes close their event brackets on failure too.

``a2a.client.send`` gets its ``a2a.client.received`` whether the call came
back, failed in transit or came back as an error status, and the
backwards-compatible ``/a2a/invoke`` and ``/a2a/stream`` routes emit the same
``a2a.task.received`` / ``a2a.task.completed`` pair as the JSON-RPC methods.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from locus.a2a import A2AClient, A2AServer
from locus.a2a.spec import INTERNAL_ERROR
from locus.a2a.transports import (
    A2AClientError,
    A2ATransportUnavailableError,
    HttpJsonClientTransport,
    JsonRpcClientTransport,
)
from locus.core.events import TerminateEvent, ThinkEvent
from locus.observability import run_context
from locus.observability.event_bus import get_event_bus


V1_MESSAGE = {"message": {"role": "ROLE_USER", "parts": [{"text": "hi"}], "messageId": "m1"}}
LEGACY_BODY = {"messages": [{"role": "user", "content": "hi"}]}


class _Events:
    """Collects the events a subscriber of the run saw, in order."""

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.seen: list[tuple[str, dict[str, Any]]] = []
        self._rid: str | None = None
        self._run: Any = None
        self._collector: asyncio.Task[None] | None = None

    async def __aenter__(self) -> _Events:
        bus = get_event_bus()
        self._run = run_context()
        self._rid = await self._run.__aenter__()

        async def collect() -> None:
            assert self._rid is not None
            async for event in bus.subscribe(self._rid):
                if event.event_type.startswith(self.prefix):
                    self.seen.append((event.event_type, dict(event.data)))

        self._collector = asyncio.create_task(collect())
        await asyncio.sleep(0)
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._rid is not None
        assert self._collector is not None
        await get_event_bus().close_stream(self._rid)
        await self._collector
        await self._run.__aexit__(None, None, None)

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.seen]

    def only(self, name: str) -> dict[str, Any]:
        matches = [data for event, data in self.seen if event == name]
        assert len(matches) == 1, self.seen
        return matches[0]


def _mock(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _refusing(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("refused", request=request)


def _status(code: int) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(code, json={"error": {"code": INTERNAL_ERROR, "message": "boom"}})

    return handler


def _ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"jsonrpc": "2.0", "id": "x", "result": {"ok": True}})


class TestClientTransports:
    async def test_jsonrpc_transport_closes_the_bracket_on_a_connection_error(self) -> None:
        transport = JsonRpcClientTransport("http://peer/", http_client=_mock(_refusing))
        async with _Events("a2a.client.") as events:
            with pytest.raises(A2ATransportUnavailableError):
                await transport.call("SendMessage", V1_MESSAGE)

        assert events.names == ["a2a.client.send", "a2a.client.received"]
        received = events.only("a2a.client.received")
        assert received["method"] == "SendMessage"
        assert received["binding"] == transport.binding
        assert received["status_code"] is None
        assert received["content_length"] == 0
        assert received["error"] == "ConnectError"
        assert received["duration_ms"] >= 0

    async def test_jsonrpc_transport_reports_an_error_status(self) -> None:
        transport = JsonRpcClientTransport("http://peer/", http_client=_mock(_status(503)))
        async with _Events("a2a.client.") as events:
            # 503 is not one of the statuses the transport maps itself, so
            # ``raise_for_status`` reports it; the bracket closes either way.
            with pytest.raises((A2AClientError, httpx.HTTPStatusError)):
                await transport.call("SendMessage", V1_MESSAGE)

        assert events.names == ["a2a.client.send", "a2a.client.received"]
        received = events.only("a2a.client.received")
        assert received["status_code"] == 503
        assert received["error"] == "HTTPStatusError"
        assert received["content_length"] > 0

    async def test_jsonrpc_transport_reports_a_response_without_an_error_key(self) -> None:
        transport = JsonRpcClientTransport("http://peer/", http_client=_mock(_ok))
        async with _Events("a2a.client.") as events:
            result = await transport.call("GetTask", {"id": "t1"})

        assert result == {"ok": True}
        received = events.only("a2a.client.received")
        assert received["status_code"] == 200
        assert "error" not in received

    async def test_http_json_transport_closes_the_bracket_on_a_connection_error(self) -> None:
        transport = HttpJsonClientTransport("http://peer/", http_client=_mock(_refusing))
        async with _Events("a2a.client.") as events:
            with pytest.raises(A2ATransportUnavailableError):
                await transport.call("SendMessage", V1_MESSAGE)

        assert events.names == ["a2a.client.send", "a2a.client.received"]
        received = events.only("a2a.client.received")
        assert received["binding"] == transport.binding
        assert received["status_code"] is None
        assert received["error"] == "ConnectError"

    async def test_http_json_transport_reports_an_error_status(self) -> None:
        transport = HttpJsonClientTransport("http://peer/", http_client=_mock(_status(500)))
        async with _Events("a2a.client.") as events:
            with pytest.raises(A2AClientError):
                await transport.call("SendMessage", V1_MESSAGE)

        received = events.only("a2a.client.received")
        assert received["status_code"] == 500
        assert received["error"] == "HTTPStatusError"


class TestGrpcTransport:
    async def test_grpc_transport_closes_the_bracket_when_the_peer_is_unreachable(self) -> None:
        pytest.importorskip("grpc")
        from locus.a2a.grpc_binding import GrpcClientTransport  # noqa: PLC0415

        # Port 1 answers nothing; the channel reports UNAVAILABLE at the deadline.
        transport = GrpcClientTransport("127.0.0.1:1", timeout=1.0)
        try:
            async with _Events("a2a.client.") as events:
                with pytest.raises(A2AClientError):
                    await transport.call("GetTask", {"id": "t1"})
        finally:
            await transport.aclose()

        assert events.names == ["a2a.client.send", "a2a.client.received"]
        received = events.only("a2a.client.received")
        assert received["binding"] == transport.binding
        assert received["status_code"] != 0
        assert received["error"] in {"UNAVAILABLE", "DEADLINE_EXCEEDED"}
        assert received["duration_ms"] >= 0


class TestPre10Client:
    async def test_the_pre_1_0_client_closes_the_bracket_on_a_connection_error(self) -> None:
        client = A2AClient("http://peer/", protocol_version=None, http_client=_mock(_refusing))
        async with _Events("a2a.client.") as events:
            with pytest.raises(httpx.ConnectError):
                await client._rpc("message/send", {})

        assert events.names == ["a2a.client.send", "a2a.client.received"]
        received = events.only("a2a.client.received")
        assert received["method"] == "message/send"
        assert received["status_code"] is None
        assert received["content_length"] == 0
        assert received["error"] == "ConnectError"

    async def test_the_pre_1_0_client_reports_an_error_status(self) -> None:
        client = A2AClient("http://peer/", protocol_version=None, http_client=_mock(_status(502)))
        async with _Events("a2a.client.") as events:
            with pytest.raises(httpx.HTTPStatusError):
                await client._rpc("message/send", {})

        received = events.only("a2a.client.received")
        assert received["status_code"] == 502
        assert received["error"] == "HTTPStatusError"
        assert received["content_length"] > 0


class _StubAgent:
    def __init__(self, events: list[Any]) -> None:
        self._events = events

    async def run(self, prompt: str) -> Any:
        for event in self._events:
            yield event


class _BoomAgent:
    async def run(self, prompt: str) -> Any:
        raise RuntimeError("boom")
        yield  # pragma: no cover


def _terminate() -> TerminateEvent:
    return TerminateEvent(
        reason="complete",
        iterations_used=1,
        final_confidence=1.0,
        total_tool_calls=0,
        final_message="final",
    )


def _happy_agent() -> _StubAgent:
    return _StubAgent([ThinkEvent(iteration=0, reasoning="thinking", tool_calls=[]), _terminate()])


async def _serve(agent: Any, path: str, events: _Events) -> tuple[int, str]:
    server = A2AServer(agent=agent, allow_unauthenticated=True)
    transport = httpx.ASGITransport(app=server.app, raise_app_exceptions=False)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://a2a") as client,
        client.stream("POST", path, json=LEGACY_BODY) as response,
    ):
        body = "".join([chunk async for chunk in response.aiter_text()])
        return response.status_code, body


class TestLegacyRoutes:
    async def test_invoke_emits_received_and_completed(self) -> None:
        async with _Events("a2a.task.") as events:
            status, body = await _serve(_happy_agent(), "/a2a/invoke", events)

        assert status == 200
        assert '"final"' in body
        assert events.names == ["a2a.task.received", "a2a.task.completed"]
        assert events.only("a2a.task.received") == {"method": "a2a/invoke", "rpc_id": None}
        completed = events.only("a2a.task.completed")
        assert completed["method"] == "a2a/invoke"
        assert completed["success"] is True
        assert "error_code" not in completed
        assert completed["duration_ms"] >= 0

    async def test_a_failing_invoke_completes_with_the_internal_error_code(self) -> None:
        async with _Events("a2a.task.") as events:
            status, _ = await _serve(_BoomAgent(), "/a2a/invoke", events)

        assert status == 500
        assert events.names == ["a2a.task.received", "a2a.task.completed"]
        completed = events.only("a2a.task.completed")
        assert completed["success"] is False
        assert completed["error_code"] == INTERNAL_ERROR

    async def test_stream_emits_received_and_completed_after_the_last_frame(self) -> None:
        async with _Events("a2a.task.") as events:
            status, body = await _serve(_happy_agent(), "/a2a/stream", events)

        assert status == 200
        assert body.rstrip().endswith("data: [DONE]")
        assert events.names == ["a2a.task.received", "a2a.task.completed"]
        assert events.only("a2a.task.received") == {"method": "a2a/stream", "rpc_id": None}
        completed = events.only("a2a.task.completed")
        assert completed["method"] == "a2a/stream"
        assert completed["success"] is True
        assert "error_code" not in completed

    async def test_a_failing_stream_completes_with_the_internal_error_code(self) -> None:
        async with _Events("a2a.task.") as events:
            status, body = await _serve(_BoomAgent(), "/a2a/stream", events)

        assert status == 200
        assert "internal error" in body
        assert body.rstrip().endswith("data: [DONE]")
        completed = events.only("a2a.task.completed")
        assert completed["success"] is False
        assert completed["error_code"] == INTERNAL_ERROR
