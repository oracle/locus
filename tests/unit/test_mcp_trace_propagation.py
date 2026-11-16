# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""W3C trace context in MCP ``_meta``, both directions.

Client side: a ``tools/call`` the agent makes carries ``traceparent`` with the
telemetry hook's tool span as the parent, so the server's work files under
the agent's tool span. Server side: a call served by ``LocusMCPServer`` runs
under the context its ``_meta`` carries, so an agent exposed over MCP files
its run under the caller's span.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest

from tests._mcp_deps import require_server_deps


require_server_deps()

from locus.agent import Agent  # noqa: E402
from locus.core.events import ToolCompleteEvent  # noqa: E402
from locus.integrations.fastmcp import LocusMCPServer, MCPClient  # noqa: E402
from locus.integrations.mcp_tracing import (  # noqa: E402
    OTEL_AVAILABLE,
    context_from_meta,
    trace_meta,
    trace_scope,
)
from locus.testing import FunctionModel, ScriptedModel, text, tool_call  # noqa: E402
from locus.tools.decorator import tool  # noqa: E402
from tests.unit.test_mcp_fidelity import _LoopbackServer  # noqa: E402


def _otel_sdk_available() -> bool:
    try:
        import opentelemetry.sdk.trace  # noqa: F401
        import opentelemetry.sdk.trace.export.in_memory_span_exporter  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not OTEL_AVAILABLE or not _otel_sdk_available(), reason="OpenTelemetry API + SDK required"
)


@pytest.fixture
def exporter() -> Any:
    """An in-memory exporter on the (once-installed) SDK tracer provider."""
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    current = otel_trace.get_tracer_provider()
    if not isinstance(current, TracerProvider):
        otel_trace.set_tracer_provider(TracerProvider())
    provider = otel_trace.get_tracer_provider()
    exp = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    return exp


@pytest.fixture(scope="module")
def server() -> Iterator[_LoopbackServer]:
    srv = _LoopbackServer().start()
    yield srv
    srv.stop()


def _telemetry_hook() -> Any:
    from locus.hooks.builtin.telemetry import TelemetryHook

    return TelemetryHook(tracer_name="locus.mcp.trace-test")


def _spans(exporter: Any, scope: str = "locus.mcp.trace-test") -> dict[str, Any]:
    return {
        s.name: s for s in exporter.get_finished_spans() if s.instrumentation_scope.name == scope
    }


def _span_by_id(exporter: Any, span_id: int) -> Any:
    return next(s for s in exporter.get_finished_spans() if s.context.span_id == span_id)


# ---------------------------------------------------------------------------
# Client side
# ---------------------------------------------------------------------------


async def test_a_call_carries_the_tool_span_as_its_parent(
    server: _LoopbackServer, exporter: Any
) -> None:
    client = MCPClient(base_url=server.url, verify_url=False)
    model = ScriptedModel([tool_call("echo_meta", call_id="m1"), text("ok")])
    agent = Agent(
        model=model,
        mcp_servers=[client],
        hooks=[_telemetry_hook()],
        reflexion=False,
        grounding=False,
    )
    events = [event async for event in agent.run("trace me")]
    await client.close()

    done = next(e for e in events if isinstance(e, ToolCompleteEvent))
    meta = json.loads(done.result or "{}")
    assert "traceparent" in meta
    version, trace_id, parent_id, flags = meta["traceparent"].split("-")
    assert version == "00"
    assert int(flags, 16) & 0x01  # sampled

    spans = _spans(exporter)
    tool_span = spans["tool.echo_meta"]
    invocation = spans["agent.invocation"]
    assert trace_id == format(tool_span.context.trace_id, "032x")
    assert tool_span.context.trace_id == invocation.context.trace_id
    # What the server received names either the tool span itself (an SDK
    # that leaves ``_meta`` alone) or the SDK's own client span for the
    # request, which must then be a child of the tool span.
    sender = _span_by_id(exporter, int(parent_id, 16))
    if sender is not tool_span:
        assert sender.name.startswith("MCP send tools/call")
        assert sender.parent is not None
        assert sender.parent.span_id == tool_span.context.span_id


async def test_propagation_can_be_turned_off(server: _LoopbackServer, exporter: Any) -> None:
    client = MCPClient(base_url=server.url, verify_url=False, propagate_trace=False)
    model = ScriptedModel([tool_call("echo_meta", call_id="m2"), text("ok")])
    agent = Agent(
        model=model,
        mcp_servers=[client],
        hooks=[_telemetry_hook()],
        reflexion=False,
        grounding=False,
    )
    events = [event async for event in agent.run("no trace")]
    await client.close()
    done = next(e for e in events if isinstance(e, ToolCompleteEvent))
    meta = json.loads(done.result or "{}")
    # Nothing of the agent's trace left the process: either no traceparent
    # at all, or (an SDK that always opens a client span) one from a trace
    # that is not the agent's.
    invocation = _spans(exporter)["agent.invocation"]
    if "traceparent" in meta:
        assert meta["traceparent"].split("-")[1] != format(invocation.context.trace_id, "032x")


def test_trace_meta_is_empty_outside_any_span() -> None:
    assert trace_meta() == {}


def test_trace_meta_from_an_explicit_span(exporter: Any) -> None:
    from opentelemetry import trace as otel_trace

    tracer = otel_trace.get_tracer("locus.mcp.trace-test")
    with tracer.start_as_current_span("outer") as span:
        meta = trace_meta(span)
    assert meta["traceparent"].split("-")[2] == format(span.get_span_context().span_id, "016x")


# ---------------------------------------------------------------------------
# Server side
# ---------------------------------------------------------------------------


def test_context_from_meta_rejects_what_is_not_a_trace() -> None:
    assert context_from_meta(None) is None
    assert context_from_meta({}) is None
    assert context_from_meta({"traceparent": "garbage"}) is None
    assert context_from_meta({"progressToken": 1}) is None
    with trace_scope({"traceparent": "garbage"}):  # a no-op, not an error
        pass


def test_trace_scope_attaches_a_remote_parent_and_defers_to_a_live_one(exporter: Any) -> None:
    from opentelemetry import trace as otel_trace

    remote = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    with trace_scope({"traceparent": remote}):
        ctx = otel_trace.get_current_span().get_span_context()
        assert format(ctx.trace_id, "032x") == "0af7651916cd43dd8448eb211c80319c"
        assert format(ctx.span_id, "016x") == "b7ad6b7169203331"
    # A server stack that extracted the context already has a span of that
    # trace current; the scope leaves it in place as the parent.
    tracer = otel_trace.get_tracer("locus.mcp.trace-test")
    with (
        trace_scope({"traceparent": remote}),
        tracer.start_as_current_span("server") as server_span,
        trace_scope({"traceparent": remote}),
    ):
        assert otel_trace.get_current_span() is server_span


async def test_a_served_tool_runs_under_the_callers_trace(exporter: Any) -> None:
    from fastmcp import Client
    from opentelemetry import trace as otel_trace

    seen: dict[str, Any] = {}

    @tool
    def where_am_i() -> str:
        """Record the current trace."""
        ctx = otel_trace.get_current_span().get_span_context()
        seen["trace_id"] = ctx.trace_id
        seen["span_id"] = ctx.span_id
        return "ok"

    agent = Agent(
        model=FunctionModel(lambda messages, tools: text("hi")),
        tools=[where_am_i],
        reflexion=False,
        grounding=False,
    )
    mcp = LocusMCPServer(agent=agent, name="traced").build()
    remote = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    async with Client(mcp) as client:
        # The caller's trace: carried in ``_meta`` by the SDK from the
        # current context, and passed explicitly for an SDK that does not.
        with trace_scope({"traceparent": remote}):
            result = await client.call_tool("where_am_i", {}, meta={"traceparent": remote})
    assert result.content[0].text == "ok"
    assert format(seen["trace_id"], "032x") == "0af7651916cd43dd8448eb211c80319c"
    assert seen["span_id"] != 0


async def test_an_agent_served_over_mcp_files_its_run_under_the_caller(exporter: Any) -> None:
    from fastmcp import Client

    agent = Agent(
        model=FunctionModel(lambda messages, tools: text("served")),
        tools=[],
        hooks=[_telemetry_hook()],
        reflexion=False,
        grounding=False,
    )
    mcp = LocusMCPServer(agent=agent, name="traced-agent").build()
    remote = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    async with Client(mcp) as client:
        with trace_scope({"traceparent": remote}):
            result = await client.call_tool(
                "run_agent_stream", {"prompt": "go"}, meta={"traceparent": remote}
            )
    assert result.content[0].text == "served"
    invocation = _spans(exporter)["agent.invocation"]
    assert format(invocation.context.trace_id, "032x") == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert invocation.parent is not None
    assert invocation.parent.trace_id == invocation.context.trace_id
