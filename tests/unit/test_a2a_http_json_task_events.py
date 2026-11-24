# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""The HTTP+JSON binding publishes the ``a2a.task.received`` / ``a2a.task.completed`` bracket.

The JSON-RPC binding brackets every request it serves; the HTTP+JSON routes
(``POST /message:send``, ``GET /tasks/{id}``, ...) serve the same operations
and report them the same way: ``method`` is the A2A 1.0 operation name,
``rpc_id`` is null (the binding has no request id), and ``completed`` carries
``success``, ``duration_ms`` and, on a failure, the ``error_code``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from locus.a2a import A2AServer
from locus.a2a.spec import INTERNAL_ERROR, VERSION_NOT_SUPPORTED
from locus.core.events import TerminateEvent, ThinkEvent
from locus.observability import run_context
from locus.observability.event_bus import get_event_bus


BODY: dict[str, Any] = {
    "message": {"role": "ROLE_USER", "parts": [{"text": "hi"}], "messageId": "m1"}
}


class _Agent:
    async def run(self, prompt: str) -> Any:
        yield ThinkEvent(iteration=0, reasoning="thinking", tool_calls=[])
        yield TerminateEvent(
            reason="complete",
            iterations_used=1,
            final_confidence=1.0,
            total_tool_calls=0,
            final_message="final",
        )


class _Served:
    """Drive one HTTP+JSON request through the ASGI app under a bound run id."""

    def __init__(self) -> None:
        self.server = A2AServer(agent=_Agent(), allow_unauthenticated=True)
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.status: int | None = None
        self.text = ""

    async def request(
        self,
        verb: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        bus = get_event_bus()
        async with run_context() as rid:

            async def collect() -> None:
                async for event in bus.subscribe(rid):
                    if event.event_type.startswith("a2a."):
                        self.events.append((event.event_type, dict(event.data)))

            collector = asyncio.create_task(collect())
            await asyncio.sleep(0)
            transport = httpx.ASGITransport(app=self.server.app)
            async with (
                httpx.AsyncClient(transport=transport, base_url="http://a2a") as client,
                client.stream(verb, path, json=json, headers=headers or {}) as response,
            ):
                self.status = response.status_code
                self.text = "".join([chunk async for chunk in response.aiter_text()])
            await bus.close_stream(rid)
            await collector

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.events]

    def only(self, name: str) -> dict[str, Any]:
        matches = [data for event, data in self.events if event == name]
        assert len(matches) == 1, self.events
        return matches[0]


class TestUnaryRoutes:
    async def test_send_message_emits_received_processing_completed(self) -> None:
        served = _Served()
        await served.request("POST", "/message:send", json=BODY)

        assert served.status == 200
        assert served.names == ["a2a.task.received", "a2a.task.processing", "a2a.task.completed"]
        assert served.only("a2a.task.received") == {"method": "SendMessage", "rpc_id": None}
        completed = served.only("a2a.task.completed")
        assert completed["method"] == "SendMessage"
        assert completed["success"] is True
        assert "error_code" not in completed
        assert completed["duration_ms"] > 0

    async def test_a_read_route_is_bracketed_too(self) -> None:
        served = _Served()
        await served.request("GET", "/tasks")

        assert served.status == 200
        assert served.names == ["a2a.task.received", "a2a.task.completed"]
        assert served.only("a2a.task.completed")["method"] == "ListTasks"

    async def test_a_protocol_error_completes_with_its_code(self) -> None:
        served = _Served()
        await served.request("POST", "/message:send", json=BODY, headers={"A2A-Version": "9.9"})

        assert served.status != 200
        assert served.names == ["a2a.task.received", "a2a.task.completed"]
        completed = served.only("a2a.task.completed")
        assert completed["success"] is False
        assert completed["error_code"] == VERSION_NOT_SUPPORTED

    async def test_an_internal_error_completes_with_the_internal_error_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        served = _Served()

        async def boom(method: str, params: dict[str, Any]) -> Any:
            raise RuntimeError("boom")

        monkeypatch.setattr(served.server, "handle_v1", boom)
        await served.request("POST", "/message:send", json=BODY)

        assert served.status == 500
        assert "boom" not in served.text
        completed = served.only("a2a.task.completed")
        assert completed["success"] is False
        assert completed["error_code"] == INTERNAL_ERROR


class TestStreamingRoutes:
    async def test_a_streamed_message_completes_after_its_last_frame(self) -> None:
        served = _Served()
        await served.request("POST", "/message:stream", json=BODY)

        assert served.status == 200
        assert served.text.count("data: ") >= 2
        assert served.names == ["a2a.task.received", "a2a.task.processing", "a2a.task.completed"]
        assert served.only("a2a.task.received") == {
            "method": "SendStreamingMessage",
            "rpc_id": None,
        }
        completed = served.only("a2a.task.completed")
        assert completed["method"] == "SendStreamingMessage"
        assert completed["success"] is True
        assert "error_code" not in completed

    async def test_a_protocol_error_inside_the_stream_completes_with_its_code(self) -> None:
        served = _Served()
        await served.request("POST", "/message:stream", json=BODY, headers={"A2A-Version": "9.9"})

        assert "event: error" in served.text
        completed = served.only("a2a.task.completed")
        assert completed["success"] is False
        assert completed["error_code"] == VERSION_NOT_SUPPORTED

    async def test_a_stream_that_errors_out_completes_with_the_internal_error_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        served = _Served()

        async def boom(method: str, params: dict[str, Any]) -> Any:
            raise RuntimeError("boom")
            yield  # pragma: no cover

        monkeypatch.setattr(served.server, "handle_v1_stream", boom)
        await served.request("POST", "/message:stream", json=BODY)

        assert "event: error" in served.text
        assert "boom" not in served.text
        completed = served.only("a2a.task.completed")
        assert completed["success"] is False
        assert completed["error_code"] == INTERNAL_ERROR

    async def test_subscribing_to_an_unknown_task_is_bracketed(self) -> None:
        served = _Served()
        await served.request("GET", "/tasks/nope:subscribe")

        assert served.names == ["a2a.task.received", "a2a.task.completed"]
        completed = served.only("a2a.task.completed")
        assert completed["method"] == "SubscribeToTask"
        assert completed["success"] is False
