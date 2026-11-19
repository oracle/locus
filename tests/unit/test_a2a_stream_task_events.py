# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Streaming A2A requests close the ``a2a.task.received`` / ``a2a.task.completed`` bracket.

``message/send`` and the v1 ``SendMessage`` emit both events from the JSON-RPC
dispatcher. The streaming methods return a ``StreamingResponse`` before the
dispatcher's ``completed`` emit, so the stream itself has to publish it when
its generator ends: on success, on a protocol error, on an internal error and
when the client goes away mid-stream.
"""

from __future__ import annotations

import asyncio
import gc
from typing import Any

import httpx
import pytest

from locus.a2a import A2AServer, JsonRpcRequest
from locus.a2a.spec import INTERNAL_ERROR, INVALID_PARAMS
from locus.core.events import TerminateEvent, ThinkEvent
from locus.observability import run_context
from locus.observability.event_bus import get_event_bus


V1_HEADERS = {"A2A-Version": "1.0"}
LEGACY_PARAMS: dict[str, Any] = {
    "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "hi"}],
        "messageId": "m1",
    }
}
V1_PARAMS: dict[str, Any] = {
    "message": {"role": "ROLE_USER", "parts": [{"text": "hi"}], "messageId": "m1"}
}


class _StubAgent:
    def __init__(self, events: list[Any]) -> None:
        self._events = events

    async def run(self, prompt: str) -> Any:
        for event in self._events:
            yield event


class _BoomAgent:
    async def run(self, prompt: str) -> Any:
        yield ThinkEvent(iteration=0, reasoning="about to fail", tool_calls=[])
        raise RuntimeError("boom")


class _SlowAgent:
    """Yields one event, then waits until the test releases it."""

    def __init__(self) -> None:
        self.release = asyncio.Event()

    async def run(self, prompt: str) -> Any:
        yield ThinkEvent(iteration=0, reasoning="first", tool_calls=[])
        await self.release.wait()
        yield _terminate("late")


def _terminate(final_message: str = "final") -> TerminateEvent:
    return TerminateEvent(
        reason="complete",
        iterations_used=1,
        final_confidence=1.0,
        total_tool_calls=0,
        final_message=final_message,
    )


def _happy_agent() -> _StubAgent:
    return _StubAgent([ThinkEvent(iteration=0, reasoning="thinking", tool_calls=[]), _terminate()])


class _Served:
    """Drive one JSON-RPC request through the ASGI app under a bound run id.

    Collects the ``a2a.*`` events a subscriber of that run saw, in order.
    """

    def __init__(self, agent: Any) -> None:
        self.server = A2AServer(agent=agent, allow_unauthenticated=True)
        self.app = self.server.app
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.status: int | None = None
        self.frames: list[str] = []

    async def request(
        self,
        method: str,
        params: dict[str, Any],
        *,
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
            transport = httpx.ASGITransport(app=self.app)
            payload = {"jsonrpc": "2.0", "id": "r1", "method": method, "params": params}
            async with (
                httpx.AsyncClient(transport=transport, base_url="http://a2a") as client,
                client.stream("POST", "/", json=payload, headers=headers or {}) as response,
            ):
                self.status = response.status_code
                async for chunk in response.aiter_text():
                    self.frames.extend(
                        line for line in chunk.split("\n\n") if line.startswith("data: ")
                    )
            await bus.close_stream(rid)
            await collector

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.events]

    def completed(self) -> dict[str, Any]:
        matches = [data for name, data in self.events if name == "a2a.task.completed"]
        assert len(matches) == 1, self.events
        return matches[0]


class _StreamUnderRun:
    """The SSE body iterator of one ``message/stream`` request, under a run id.

    Calls ``_stream_response`` directly, as the dispatcher does, so the test
    holds the generator and decides how it ends.
    """

    def __init__(self, agent: Any) -> None:
        self.server = A2AServer(agent=agent, allow_unauthenticated=True)
        _ = self.server.app
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.iterator: Any = None
        self._response: Any = None
        self._collector: asyncio.Task[None] | None = None
        self._run: Any = None

    async def __aenter__(self) -> _StreamUnderRun:
        bus = get_event_bus()
        self._run = run_context()
        rid = await self._run.__aenter__()
        self._rid = rid

        async def collect() -> None:
            async for event in bus.subscribe(rid):
                if event.event_type.startswith("a2a."):
                    self.events.append((event.event_type, dict(event.data)))

        self._collector = asyncio.create_task(collect())
        await asyncio.sleep(0)
        request = JsonRpcRequest(id="r1", method="message/stream", params=LEGACY_PARAMS)
        self._response = await self.server._stream_response(request)
        self.iterator = self._response.body_iterator
        return self

    def drop(self) -> None:
        self.iterator = None
        self._response = None

    async def __aexit__(self, *exc: object) -> None:
        # ``close_stream`` lets the publishes ``emit_sync`` scheduled land
        # first, then ends the subscriber's iterator.
        await get_event_bus().close_stream(self._rid)
        assert self._collector is not None
        await self._collector
        await self._run.__aexit__(None, None, None)

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.events]

    def completed(self) -> dict[str, Any]:
        matches = [data for name, data in self.events if name == "a2a.task.completed"]
        assert len(matches) == 1, self.events
        return matches[0]


class TestStreamingRequestsCloseTheBracket:
    @pytest.mark.parametrize(
        ("method", "params", "headers"),
        [
            ("message/stream", LEGACY_PARAMS, {}),
            ("SendStreamingMessage", V1_PARAMS, V1_HEADERS),
        ],
    )
    async def test_a_streamed_run_emits_received_then_completed(
        self, method: str, params: dict[str, Any], headers: dict[str, str]
    ) -> None:
        served = _Served(_happy_agent())
        await served.request(method, params, headers=headers)

        assert served.status == 200
        assert len(served.frames) >= 2
        assert served.names == ["a2a.task.received", "a2a.task.completed"]
        completed = served.completed()
        assert completed["method"] == method
        assert completed["success"] is True
        assert "error_code" not in completed
        assert completed["duration_ms"] > 0

    @pytest.mark.parametrize(
        ("method", "params", "headers"),
        [
            ("message/stream", LEGACY_PARAMS, {}),
            ("SendStreamingMessage", V1_PARAMS, V1_HEADERS),
        ],
    )
    async def test_an_agent_failure_inside_the_stream_is_a_served_request(
        self, method: str, params: dict[str, Any], headers: dict[str, str]
    ) -> None:
        # The stream turns the agent's exception into a final ``failed``
        # status update for the client, so the request itself was served;
        # ``success`` reports the transport outcome, as it does for a
        # ``message/send`` whose task failed with an error response.
        served = _Served(_BoomAgent())
        await served.request(method, params, headers=headers)

        assert served.status == 200
        assert "failed" in served.frames[-1].lower()
        assert served.names == ["a2a.task.received", "a2a.task.completed"]
        completed = served.completed()
        assert completed["method"] == method
        assert completed["success"] is True
        assert "error_code" not in completed

    @pytest.mark.parametrize(
        ("method", "params", "headers", "attribute"),
        [
            ("message/stream", LEGACY_PARAMS, {}, "_stream_message"),
            ("SendStreamingMessage", V1_PARAMS, V1_HEADERS, "handle_v1_stream"),
        ],
    )
    async def test_a_stream_that_errors_out_completes_with_the_internal_error_code(
        self,
        monkeypatch: pytest.MonkeyPatch,
        method: str,
        params: dict[str, Any],
        headers: dict[str, str],
        attribute: str,
    ) -> None:
        served = _Served(_happy_agent())

        def boom_stream(*args: Any, **kwargs: Any) -> Any:
            async def gen() -> Any:
                raise RuntimeError("boom")
                yield {}  # pragma: no cover

            return gen()

        monkeypatch.setattr(served.server, attribute, boom_stream)
        await served.request(method, params, headers=headers)

        assert served.status == 200
        assert "correlation_id" in served.frames[-1]
        assert served.names == ["a2a.task.received", "a2a.task.completed"]
        completed = served.completed()
        assert completed["method"] == method
        assert completed["success"] is False
        assert completed["error_code"] == INTERNAL_ERROR
        assert completed["duration_ms"] > 0

    async def test_invalid_stream_params_complete_with_the_protocol_error_code(self) -> None:
        served = _Served(_happy_agent())
        await served.request("message/stream", {})

        assert served.names == ["a2a.task.received", "a2a.task.completed"]
        completed = served.completed()
        assert completed["success"] is False
        assert completed["error_code"] == INVALID_PARAMS

    async def test_a_client_that_leaves_mid_stream_completes_without_success(self) -> None:
        # Starlette does not close the body iterator when the client goes
        # away; whoever does close it (here: the test) gets the completion.
        async with _StreamUnderRun(_SlowAgent()) as run:
            await run.iterator.__anext__()
            await run.iterator.aclose()

        assert run.names == ["a2a.task.completed"]
        completed = run.completed()
        assert completed["method"] == "message/stream"
        assert completed["success"] is False
        assert "error_code" not in completed
        assert completed["duration_ms"] > 0

    async def test_an_abandoned_stream_still_completes_when_the_loop_finalises_it(self) -> None:
        # The production disconnect path: the response and its generator are
        # dropped without ``aclose()``; the loop's async-generator finaliser
        # closes them in a context of its own, and the completion still has
        # to carry this request's run id.
        async with _StreamUnderRun(_SlowAgent()) as run:
            await run.iterator.__anext__()
            run.drop()
            gc.collect()
            for _ in range(100):
                await asyncio.sleep(0.01)
                if "a2a.task.completed" in run.names:
                    break

        assert run.names == ["a2a.task.completed"]
        completed = run.completed()
        assert completed["success"] is False
        assert "error_code" not in completed

    @pytest.mark.parametrize(
        ("method", "params", "headers"),
        [
            ("message/send", LEGACY_PARAMS, {}),
            ("SendMessage", V1_PARAMS, V1_HEADERS),
        ],
    )
    async def test_the_non_streaming_paths_are_unchanged(
        self, method: str, params: dict[str, Any], headers: dict[str, str]
    ) -> None:
        served = _Served(_happy_agent())
        await served.request(method, params, headers=headers)

        assert served.status == 200
        assert served.names == ["a2a.task.received", "a2a.task.completed"]
        completed = served.completed()
        assert completed["method"] == method
        assert completed["success"] is True
