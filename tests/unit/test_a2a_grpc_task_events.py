# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""The gRPC binding publishes the ``a2a.task.received`` / ``a2a.task.completed`` bracket.

The payload is the one the HTTP bindings use: ``method`` is the A2A 1.0
operation name, ``rpc_id`` is null, and ``completed`` carries ``success``,
``duration_ms`` and, on a failure, the ``error_code``. A gRPC call is served in
a task the ``grpc.aio`` server creates, which inherits the context the server
was started in, so the events land on the run that was bound at ``start()``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest


grpc = pytest.importorskip("grpc")
pytest.importorskip("grpc_status")

from locus.a2a import A2AClientError, A2AServer  # noqa: E402
from locus.a2a.grpc_binding import GrpcClientTransport  # noqa: E402
from locus.a2a.spec import INTERNAL_ERROR, TASK_NOT_FOUND  # noqa: E402
from locus.a2a.spec_v1 import BINDING_GRPC, BINDING_JSONRPC  # noqa: E402
from locus.core.events import TerminateEvent, ThinkEvent  # noqa: E402
from locus.observability import run_context  # noqa: E402
from locus.observability.event_bus import get_event_bus  # noqa: E402


PARAMS: dict[str, Any] = {
    "message": {"role": "ROLE_USER", "parts": [{"text": "hi"}], "messageId": "m1"}
}


class _Agent:
    def __init__(self, *, fail: bool = False, frames: int = 1) -> None:
        self._fail = fail
        self._frames = frames

    async def run(self, prompt: str) -> Any:
        for i in range(self._frames):
            yield ThinkEvent(iteration=i, reasoning="thinking", tool_calls=[])
            await asyncio.sleep(0)
        yield TerminateEvent(
            reason="complete",
            iterations_used=1,
            final_confidence=1.0,
            total_tool_calls=0,
            final_message="final",
        )


class _Served:
    """Serve one gRPC call from a server started under a bound run id."""

    def __init__(self, agent: Any | None = None) -> None:
        self.agent = agent or _Agent()
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def call(self, action: Any, *, server_patch: Any | None = None) -> Any:
        bus = get_event_bus()
        async with run_context() as rid:

            async def collect() -> None:
                async for event in bus.subscribe(rid):
                    if event.event_type.startswith("a2a.task."):
                        self.events.append((event.event_type, dict(event.data)))

            collector = asyncio.create_task(collect())
            await asyncio.sleep(0)
            server = A2AServer(
                agent=self.agent,
                allow_unauthenticated=True,
                url="http://127.0.0.1",
                transports=[BINDING_JSONRPC, BINDING_GRPC],
                grpc_url="set-below",
            )
            if server_patch is not None:
                server_patch(server)
            aio_server, port = server.grpc_server("127.0.0.1", 0)
            await aio_server.start()
            transport = GrpcClientTransport(f"127.0.0.1:{port}")
            try:
                return await action(transport)
            finally:
                await transport.aclose()
                await aio_server.stop(grace=None)
                await server.aclose()
                await bus.close_stream(rid)
                await collector

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.events]

    def only(self, name: str) -> dict[str, Any]:
        matches = [data for event, data in self.events if event == name]
        assert len(matches) == 1, self.events
        return matches[0]


async def _drain(
    transport: GrpcClientTransport, method: str, params: dict[str, Any]
) -> list[dict[str, Any]]:
    """Every frame of a stream; the transport reports a failed call as an error frame."""
    return [frame async for frame in transport.stream(method, params)]


class TestUnaryCalls:
    async def test_send_message_emits_received_processing_completed(self) -> None:
        served = _Served()
        await served.call(lambda t: t.call("SendMessage", PARAMS))

        assert served.names == ["a2a.task.received", "a2a.task.processing", "a2a.task.completed"]
        assert served.only("a2a.task.received") == {"method": "SendMessage", "rpc_id": None}
        completed = served.only("a2a.task.completed")
        assert completed["method"] == "SendMessage"
        assert completed["success"] is True
        assert completed["duration_ms"] >= 0
        assert "error_code" not in completed

    async def test_an_unknown_task_completes_with_its_code(self) -> None:
        served = _Served()
        with pytest.raises(A2AClientError) as err:
            await served.call(lambda t: t.call("GetTask", {"id": "nope"}))

        assert err.value.code == TASK_NOT_FOUND
        assert served.names == ["a2a.task.received", "a2a.task.completed"]
        completed = served.only("a2a.task.completed")
        assert completed["method"] == "GetTask"
        assert completed["success"] is False
        assert completed["error_code"] == TASK_NOT_FOUND

    async def test_an_internal_error_completes_with_the_internal_error_code(self) -> None:
        async def boom(method: str, params: dict[str, Any]) -> Any:
            raise RuntimeError("boom")

        served = _Served()
        with pytest.raises(A2AClientError):
            await served.call(
                lambda t: t.call("GetTask", {"id": "x"}),
                server_patch=lambda server: setattr(server, "handle_v1", boom),
            )

        completed = served.only("a2a.task.completed")
        assert completed["success"] is False
        assert completed["error_code"] == INTERNAL_ERROR


class TestStreamingCalls:
    async def test_a_streamed_message_completes_after_its_last_frame(self) -> None:
        served = _Served()
        frames = await served.call(lambda t: _drain(t, "SendStreamingMessage", PARAMS))

        assert frames
        assert "error" not in frames[-1]
        assert served.names == ["a2a.task.received", "a2a.task.processing", "a2a.task.completed"]
        assert served.only("a2a.task.received") == {
            "method": "SendStreamingMessage",
            "rpc_id": None,
        }
        completed = served.only("a2a.task.completed")
        assert completed["success"] is True
        assert "error_code" not in completed

    async def test_subscribing_to_an_unknown_task_completes_with_its_code(self) -> None:
        served = _Served()
        frames = await served.call(lambda t: _drain(t, "SubscribeToTask", {"id": "nope"}))

        assert frames[0]["error"]["code"] == TASK_NOT_FOUND

        assert served.names == ["a2a.task.received", "a2a.task.completed"]
        completed = served.only("a2a.task.completed")
        assert completed["success"] is False
        assert completed["error_code"] == TASK_NOT_FOUND

    async def test_a_stream_that_errors_out_completes_with_the_internal_error_code(self) -> None:
        async def boom(method: str, params: dict[str, Any]) -> Any:
            raise RuntimeError("boom")
            yield  # pragma: no cover - makes this an async generator

        served = _Served()
        frames = await served.call(
            lambda t: _drain(t, "SendStreamingMessage", PARAMS),
            server_patch=lambda server: setattr(server, "handle_v1_stream", boom),
        )

        assert frames[0]["error"]["code"] == INTERNAL_ERROR

        completed = served.only("a2a.task.completed")
        assert completed["success"] is False
        assert completed["error_code"] == INTERNAL_ERROR

    async def test_a_client_that_leaves_mid_stream_completes_without_a_code(self) -> None:
        served = _Served(_Agent(frames=2000))

        async def leave_early(transport: GrpcClientTransport) -> None:
            stream = transport.stream("SendStreamingMessage", PARAMS)
            await anext(stream)
            await stream.aclose()
            # The cancellation reaches the server over the wire; wait for it.
            for _ in range(500):
                if "a2a.task.completed" in served.names:
                    break
                await asyncio.sleep(0.01)

        await served.call(leave_early)

        completed = served.only("a2a.task.completed")
        assert completed["method"] == "SendStreamingMessage"
        assert completed["success"] is False
        assert "error_code" not in completed
