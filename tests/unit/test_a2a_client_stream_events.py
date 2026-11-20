# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""A streamed A2A client call is bracketed like a one-shot one.

``a2a.client.send`` opens before the request and ``a2a.client.received``
closes when the stream ends: after the last frame, when the consumer stops
reading, when the connection fails and when the peer answers an error
status. ``frames`` counts what the stream yielded.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from locus.a2a import A2AClient, Message, TextPart
from locus.a2a.spec import INTERNAL_ERROR
from locus.a2a.transports import (
    A2AClientError,
    A2ATransportUnavailableError,
    HttpJsonClientTransport,
    JsonRpcClientTransport,
)
from locus.observability import run_context
from locus.observability.event_bus import get_event_bus


V1_MESSAGE = {"message": {"role": "ROLE_USER", "parts": [{"text": "hi"}], "messageId": "m1"}}
SSE = {"content-type": "text/event-stream"}


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


def _sse(*payloads: Any, done: bool = True) -> Any:
    body = "".join(f"data: {json.dumps(p)}\n\n" for p in payloads)
    if done:
        body += "data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers=SSE, content=body.encode())

    return handler


def _rpc(result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": "x", "result": result}


class TestJsonRpcStream:
    async def test_a_stream_that_ends_closes_the_bracket_with_its_frame_count(self) -> None:
        transport = JsonRpcClientTransport(
            "http://peer/", http_client=_mock(_sse(_rpc({"n": 1}), _rpc({"n": 2})))
        )
        async with _Events("a2a.client.") as events:
            frames = [f async for f in transport.stream("SendStreamingMessage", V1_MESSAGE)]

        assert frames == [{"n": 1}, {"n": 2}]
        assert events.names == ["a2a.client.send", "a2a.client.received"]
        received = events.only("a2a.client.received")
        assert received["method"] == "SendStreamingMessage"
        assert received["binding"] == transport.binding
        assert received["status_code"] == 200
        assert received["frames"] == 2
        assert "error" not in received
        assert received["duration_ms"] >= 0

    async def test_a_consumer_that_stops_reading_still_closes_the_bracket(self) -> None:
        transport = JsonRpcClientTransport(
            "http://peer/", http_client=_mock(_sse(_rpc({"n": 1}), _rpc({"n": 2})))
        )
        async with _Events("a2a.client.") as events:
            stream = transport.stream("SendStreamingMessage", V1_MESSAGE)
            first = await stream.__anext__()
            await stream.aclose()  # type: ignore[attr-defined]  # an async generator at runtime

        assert first == {"n": 1}
        received = events.only("a2a.client.received")
        assert received["frames"] == 1
        assert received["status_code"] == 200
        assert "error" not in received

    async def test_a_refused_connection_closes_the_bracket(self) -> None:
        transport = JsonRpcClientTransport("http://peer/", http_client=_mock(_refusing))
        async with _Events("a2a.client.") as events:
            with pytest.raises(A2ATransportUnavailableError):
                _ = [f async for f in transport.stream("SendStreamingMessage", V1_MESSAGE)]

        assert events.names == ["a2a.client.send", "a2a.client.received"]
        received = events.only("a2a.client.received")
        assert received["status_code"] is None
        assert received["frames"] == 0
        assert received["error"] == "ConnectError"

    async def test_an_error_status_closes_the_bracket(self) -> None:
        transport = JsonRpcClientTransport("http://peer/", http_client=_mock(_status(503)))
        async with _Events("a2a.client.") as events:
            with pytest.raises((A2AClientError, httpx.HTTPStatusError)):
                _ = [f async for f in transport.stream("SendStreamingMessage", V1_MESSAGE)]

        received = events.only("a2a.client.received")
        assert received["status_code"] == 503
        assert received["error"] == "HTTPStatusError"


class TestHttpJsonStream:
    async def test_a_stream_that_ends_closes_the_bracket(self) -> None:
        transport = HttpJsonClientTransport(
            "http://peer/", http_client=_mock(_sse({"task": {"id": "t"}}, {"statusUpdate": {}}))
        )
        async with _Events("a2a.client.") as events:
            frames = [f async for f in transport.stream("SendStreamingMessage", V1_MESSAGE)]

        assert len(frames) == 2
        received = events.only("a2a.client.received")
        assert received["binding"] == transport.binding
        assert received["status_code"] == 200
        assert received["frames"] == 2
        assert "error" not in received

    async def test_an_error_body_closes_the_bracket(self) -> None:
        transport = HttpJsonClientTransport("http://peer/", http_client=_mock(_status(500)))
        async with _Events("a2a.client.") as events:
            with pytest.raises(A2AClientError):
                _ = [f async for f in transport.stream("SendStreamingMessage", V1_MESSAGE)]

        received = events.only("a2a.client.received")
        assert received["status_code"] == 500
        assert received["frames"] == 0
        assert received["error"] == "HTTPStatusError"


class TestGrpcStream:
    async def test_an_unreachable_peer_closes_the_bracket(self) -> None:
        pytest.importorskip("grpc")
        from locus.a2a.grpc_binding import GrpcClientTransport  # noqa: PLC0415

        transport = GrpcClientTransport("127.0.0.1:1", timeout=1.0)
        try:
            async with _Events("a2a.client.") as events:
                with pytest.raises(A2AClientError):
                    _ = [f async for f in transport.stream("SendStreamingMessage", V1_MESSAGE)]
        finally:
            await transport.aclose()

        assert events.names == ["a2a.client.send", "a2a.client.received"]
        received = events.only("a2a.client.received")
        assert received["binding"] == transport.binding
        assert received["status_code"] != 0
        assert received["frames"] == 0
        assert received["error"] in {"UNAVAILABLE", "DEADLINE_EXCEEDED"}


class TestPre10ClientStream:
    def _message(self) -> Message:
        return Message(role="user", parts=[TextPart(text="hi")], messageId="m1")

    async def test_the_pre_1_0_stream_is_bracketed(self) -> None:
        handler = _sse(
            _rpc({"type": "text", "content": "a"}), _rpc({"type": "done", "content": "b"})
        )
        client = A2AClient("http://peer/", protocol_version=None, http_client=_mock(handler))
        async with _Events("a2a.client.") as events:
            frames = [f async for f in client.send_message_streaming(self._message())]

        assert len(frames) == 2
        assert events.names == ["a2a.client.send", "a2a.client.received"]
        assert events.only("a2a.client.send")["method"] == "message/stream"
        received = events.only("a2a.client.received")
        assert received["method"] == "message/stream"
        assert received["status_code"] == 200
        assert received["frames"] == 2
        assert "error" not in received

    async def test_a_refused_pre_1_0_stream_closes_the_bracket(self) -> None:
        client = A2AClient("http://peer/", protocol_version=None, http_client=_mock(_refusing))
        async with _Events("a2a.client.") as events:
            with pytest.raises(httpx.ConnectError):
                _ = [f async for f in client.send_message_streaming(self._message())]

        received = events.only("a2a.client.received")
        assert received["status_code"] is None
        assert received["frames"] == 0
        assert received["error"] == "ConnectError"
