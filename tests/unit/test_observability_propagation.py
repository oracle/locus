# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""W3C Trace Context on the wire: the propagation helpers and the A2A transport.

Outbound, an A2A request made from inside an agent run carries
``traceparent`` with the agent's tool span as its parent; inbound, a served
A2A request runs the agent under the caller's trace, so its ``invoke_agent``
span is a child of the caller's span.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from locus.observability.propagation import (
    OTEL_AVAILABLE,
    current_traceparent,
    extract_trace_context,
    inject_trace_headers,
    trace_scope,
)


def _otel_sdk_available() -> bool:
    try:
        import opentelemetry.sdk.trace  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not OTEL_AVAILABLE or not _otel_sdk_available(), reason="OpenTelemetry API + SDK required"
)

HEADER = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"


@pytest.fixture
def exporter() -> Any:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    if not isinstance(otel_trace.get_tracer_provider(), TracerProvider):
        otel_trace.set_tracer_provider(TracerProvider())
    exp = InMemorySpanExporter()
    otel_trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exp))
    return exp


def _tracer() -> Any:
    from opentelemetry import trace as otel_trace

    return otel_trace.get_tracer("locus.propagation.test")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_inject_writes_nothing_without_a_span(self) -> None:
        assert inject_trace_headers({"Accept": "x"}) == {"Accept": "x"}
        assert inject_trace_headers() == {}

    def test_inject_carries_the_given_span(self, exporter: Any) -> None:
        with _tracer().start_span("outer") as span:
            headers = inject_trace_headers(span=span)
        parts = headers["traceparent"].split("-")
        assert parts[0] == "00"
        assert parts[1] == format(span.get_span_context().trace_id, "032x")
        assert parts[2] == format(span.get_span_context().span_id, "016x")

    def test_inject_defaults_to_the_current_span(self, exporter: Any) -> None:
        with _tracer().start_as_current_span("outer") as span:
            headers = inject_trace_headers()
            assert current_traceparent() == headers["traceparent"]
        assert headers["traceparent"].split("-")[2] == format(
            span.get_span_context().span_id, "016x"
        )
        assert current_traceparent() is None

    def test_extract_is_case_insensitive_and_rejects_garbage(self) -> None:
        assert extract_trace_context(None) is None
        assert extract_trace_context({}) is None
        assert extract_trace_context({"traceparent": "nonsense"}) is None
        assert extract_trace_context({"TraceParent": HEADER}) is not None

    def test_trace_scope_attaches_and_detaches(self, exporter: Any) -> None:
        from opentelemetry import trace as otel_trace

        with trace_scope({"traceparent": HEADER}):
            ctx = otel_trace.get_current_span().get_span_context()
            assert format(ctx.trace_id, "032x") == "0af7651916cd43dd8448eb211c80319c"
            assert ctx.is_remote
        assert otel_trace.get_current_span().get_span_context().span_id == 0

    def test_trace_scope_defers_to_a_span_of_the_same_trace(self, exporter: Any) -> None:
        """A framework that already extracted the headers keeps its span current."""
        from opentelemetry import trace as otel_trace

        with (
            trace_scope({"traceparent": HEADER}),
            _tracer().start_as_current_span("framework") as framework,
            trace_scope({"traceparent": HEADER}),
        ):
            current = otel_trace.get_current_span()
            assert current.get_span_context().span_id == framework.get_span_context().span_id


# ---------------------------------------------------------------------------
# A2A client: traceparent on every request it sends
# ---------------------------------------------------------------------------


def _ok(result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": "1", "result": result}


class TestA2AClient:
    async def test_rpc_carries_the_current_span(self, exporter: Any) -> None:
        from locus.a2a import A2AClient

        client = A2AClient("http://remote.invalid")
        with respx.mock(base_url="http://remote.invalid") as mock:
            route = mock.post("/").mock(return_value=httpx.Response(200, json=_ok({"ok": True})))
            with _tracer().start_as_current_span("caller") as caller:
                await client._rpc("tasks/get", {"id": "t-1"})
        sent = route.calls.last.request.headers
        assert sent["traceparent"].split("-")[2] == format(
            caller.get_span_context().span_id, "016x"
        )

    async def test_rpc_sends_no_header_outside_a_trace(self, exporter: Any) -> None:
        from locus.a2a import A2AClient

        client = A2AClient("http://remote.invalid")
        with respx.mock(base_url="http://remote.invalid") as mock:
            route = mock.post("/").mock(return_value=httpx.Response(200, json=_ok({})))
            await client._rpc("tasks/get", {"id": "t-1"})
        assert "traceparent" not in route.calls.last.request.headers

    async def test_legacy_invoke_carries_the_header(self, exporter: Any) -> None:
        from locus.a2a import A2AClient

        client = A2AClient("http://remote.invalid")
        body = {"messages": [{"role": "agent", "content": "pong"}], "status": "completed"}
        with respx.mock(base_url="http://remote.invalid") as mock:
            route = mock.post("/a2a/invoke").mock(return_value=httpx.Response(200, json=body))
            with _tracer().start_as_current_span("caller"):
                assert await client.invoke("ping") == "pong"
        assert "traceparent" in route.calls.last.request.headers


# ---------------------------------------------------------------------------
# A2A server: the served run files under the caller's trace
# ---------------------------------------------------------------------------


def _server_app(**kwargs: Any) -> tuple[Any, Any, str]:
    from locus.a2a import A2AServer
    from locus.agent import Agent
    from locus.hooks.builtin.telemetry import TelemetryHook
    from locus.testing import ScriptedModel, text

    scope = "locus.propagation.test-a2a-server"
    agent = Agent(
        model=ScriptedModel([text("done")]),
        hooks=[TelemetryHook(tracer_name=scope)],
        reflexion=False,
        grounding=False,
    )
    server = A2AServer(agent, name="traced", allow_unauthenticated=True, **kwargs)
    return server, server.app, scope


V1_SEND = {"message": {"role": "ROLE_USER", "parts": [{"text": "hi"}], "messageId": "m-1"}}


def _root(exporter: Any, scope: str) -> Any:
    spans = [s for s in exporter.get_finished_spans() if s.instrumentation_scope.name == scope]
    roots = [s for s in spans if s.name.startswith("invoke_agent")]
    assert len(roots) == 1, [s.name for s in spans]
    return roots[0]


def _remote_ancestor(exporter: Any, span: Any) -> Any:
    """The remote span context the span descends from, if any.

    FastAPI emits spans of its own when OpenTelemetry is installed, so the
    agent's root may sit under a framework span that in turn carries the
    caller's ``traceparent``; the lineage, not the direct parent, is what
    matters.
    """
    by_id = {s.context.span_id: s for s in exporter.get_finished_spans()}
    parent = span.parent
    while parent is not None:
        if parent.is_remote:
            return parent
        owner = by_id.get(parent.span_id)
        parent = owner.parent if owner is not None else None
    return None


REMOTE_TRACE = "0af7651916cd43dd8448eb211c80319c"
REMOTE_SPAN = "b7ad6b7169203331"


class TestA2AServer:
    def test_message_send_inherits_traceparent(self, exporter: Any) -> None:
        from fastapi.testclient import TestClient

        _, app, scope = _server_app()
        body = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "hi"}],
                    "messageId": "m-1",
                }
            },
        }
        response = TestClient(app).post("/", json=body, headers={"traceparent": HEADER})
        assert response.status_code == 200, response.text
        assert "result" in response.json()
        root = _root(exporter, scope)
        assert format(root.context.trace_id, "032x") == REMOTE_TRACE
        assert format(_remote_ancestor(exporter, root).span_id, "016x") == REMOTE_SPAN

    def test_legacy_invoke_inherits_traceparent(self, exporter: Any) -> None:
        from fastapi.testclient import TestClient

        _, app, scope = _server_app()
        response = TestClient(app).post(
            "/a2a/invoke",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"traceparent": HEADER},
        )
        assert response.status_code == 200, response.text
        root = _root(exporter, scope)
        assert format(root.context.trace_id, "032x") == REMOTE_TRACE
        assert format(_remote_ancestor(exporter, root).span_id, "016x") == REMOTE_SPAN

    def test_without_a_header_the_run_is_its_own_root(self, exporter: Any) -> None:
        from fastapi.testclient import TestClient

        _, app, scope = _server_app()
        response = TestClient(app).post(
            "/a2a/invoke", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert response.status_code == 200, response.text
        root = _root(exporter, scope)
        assert format(root.context.trace_id, "032x") != REMOTE_TRACE
        assert _remote_ancestor(exporter, root) is None

    def test_streaming_inherits_traceparent(self, exporter: Any) -> None:
        from fastapi.testclient import TestClient

        _, app, scope = _server_app()
        with TestClient(app).stream(
            "POST",
            "/a2a/stream",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"traceparent": HEADER},
        ) as response:
            assert response.status_code == 200
            for _ in response.iter_lines():
                pass
        root = _root(exporter, scope)
        assert format(root.context.trace_id, "032x") == REMOTE_TRACE
        assert format(_remote_ancestor(exporter, root).span_id, "016x") == REMOTE_SPAN


# ---------------------------------------------------------------------------
# The other bindings: HTTP+JSON routes, the transports' headers, gRPC metadata
# ---------------------------------------------------------------------------


class TestA2AHttpJson:
    def test_message_send_inherits_traceparent(self, exporter: Any) -> None:
        from fastapi.testclient import TestClient

        _, app, scope = _server_app(transports=["jsonrpc", "http+json"])
        response = TestClient(app).post(
            "/message:send", json=V1_SEND, headers={"traceparent": HEADER}
        )
        assert response.status_code == 200, response.text
        root = _root(exporter, scope)
        assert format(root.context.trace_id, "032x") == REMOTE_TRACE
        assert format(_remote_ancestor(exporter, root).span_id, "016x") == REMOTE_SPAN

    def test_message_stream_inherits_traceparent(self, exporter: Any) -> None:
        from fastapi.testclient import TestClient

        _, app, scope = _server_app(transports=["jsonrpc", "http+json"])
        with TestClient(app).stream(
            "POST", "/message:stream", json=V1_SEND, headers={"traceparent": HEADER}
        ) as response:
            assert response.status_code == 200
            for _ in response.iter_lines():
                pass
        root = _root(exporter, scope)
        assert format(root.context.trace_id, "032x") == REMOTE_TRACE
        assert format(_remote_ancestor(exporter, root).span_id, "016x") == REMOTE_SPAN

    def test_without_a_header_the_run_is_its_own_root(self, exporter: Any) -> None:
        from fastapi.testclient import TestClient

        _, app, scope = _server_app(transports=["http+json"])
        assert TestClient(app).post("/message:send", json=V1_SEND).status_code == 200
        root = _root(exporter, scope)
        assert format(root.context.trace_id, "032x") != REMOTE_TRACE
        assert _remote_ancestor(exporter, root) is None


class TestTransports:
    def test_http_transport_headers_carry_the_current_span(self, exporter: Any) -> None:
        from locus.a2a.transports import HttpJsonClientTransport, JsonRpcClientTransport

        for cls in (JsonRpcClientTransport, HttpJsonClientTransport):
            transport = cls("http://remote.invalid", api_key="k")
            assert "traceparent" not in transport.headers()
            with _tracer().start_as_current_span("caller") as caller:
                headers = transport.headers(Accept="text/event-stream")
            assert headers["Authorization"] == "Bearer k"
            assert headers["Accept"] == "text/event-stream"
            assert headers["traceparent"].split("-")[2] == format(
                caller.get_span_context().span_id, "016x"
            )

    def test_grpc_metadata_carries_the_current_span(self, exporter: Any) -> None:
        pytest.importorskip("grpc")
        from locus.a2a.grpc_binding import GrpcClientTransport

        transport = GrpcClientTransport("127.0.0.1:1", api_key="k")
        assert "traceparent" not in dict(transport._metadata())
        with _tracer().start_as_current_span("caller") as caller:
            metadata = dict(transport._metadata())
        assert metadata["authorization"] == "Bearer k"
        assert metadata["traceparent"].split("-")[2] == format(
            caller.get_span_context().span_id, "016x"
        )


class TestA2AGrpc:
    async def test_served_call_runs_under_the_callers_trace(self, exporter: Any) -> None:
        pytest.importorskip("grpc")
        from locus.a2a import A2AClient
        from locus.a2a.spec import Message, TextPart

        server, _, scope = _server_app(
            url="http://127.0.0.1", transports=["jsonrpc", "grpc"], grpc_url="set-below"
        )
        aio_server, port = server.grpc_server("127.0.0.1", 0)
        server._grpc_url = f"127.0.0.1:{port}"
        await aio_server.start()
        client = A2AClient(f"127.0.0.1:{port}", binding="grpc")
        try:
            with _tracer().start_as_current_span("caller") as caller:
                task = await client.send_message(
                    Message(role="user", parts=[TextPart(text="hi")], messageId="m-1")
                )
            assert task.status.state.value == "completed"
        finally:
            await client.aclose()
            await aio_server.stop(grace=None)
            await server.aclose()
        root = _root(exporter, scope)
        assert root.context.trace_id == caller.get_span_context().trace_id
        assert format(_remote_ancestor(exporter, root).span_id, "016x") == format(
            caller.get_span_context().span_id, "016x"
        )
