# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``Agent.run_sync`` inside a running loop keeps the caller's context.

The run is driven on a worker thread; without ``contextvars.copy_context()``
every ``ContextVar`` — the OpenTelemetry context among them — was lost there,
so an agent served over MCP through ``run_agent`` filed its ``invoke_agent``
span in a fresh trace although ``LocusMCPServer`` had set the caller's.
"""

from __future__ import annotations

import contextvars
from typing import Any

import pytest

from locus.agent import Agent
from locus.core.messages import Role
from locus.testing import FunctionModel, text, tool_call
from locus.tools import tool


_caller: contextvars.ContextVar[str] = contextvars.ContextVar("locus_test_caller", default="nobody")

TRACE_ID = "0af7651916cd43dd8448eb211c80319c"
PARENT_SPAN_ID = "b7ad6b7169203331"
TRACEPARENT = f"00-{TRACE_ID}-{PARENT_SPAN_ID}-01"


@tool
def whoami() -> str:
    """The caller's identity, read from a context variable."""
    return _caller.get()


def _decide(messages: list[Any], tools: list[Any]) -> Any:
    answers = [m for m in messages if m.role == Role.TOOL]
    if answers:
        return text(f"caller: {answers[-1].content}")
    return tool_call("whoami")


def _agent(**overrides: Any) -> Agent:
    settings: dict[str, Any] = {"tools": [whoami], "reflexion": False, "grounding": False}
    settings.update(overrides)
    return Agent(model=FunctionModel(_decide), **settings)


async def test_run_sync_inside_a_running_loop_sees_the_callers_context_variables() -> None:
    token = _caller.set("ada")
    try:
        result = _agent().run_sync("who am I?")
    finally:
        _caller.reset(token)
    assert result.message == "caller: ada"


def test_run_sync_with_no_loop_sees_them_too() -> None:
    token = _caller.set("grace")
    try:
        result = _agent().run_sync("who am I?")
    finally:
        _caller.reset(token)
    assert result.message == "caller: grace"


def _otel_sdk_available() -> bool:
    try:
        import opentelemetry.sdk.trace  # noqa: F401
    except ImportError:
        return False
    return True


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


def _invoke_span(exporter: Any, agent_name: str) -> Any:
    spans = [s for s in exporter.get_finished_spans() if s.name == f"invoke_agent {agent_name}"]
    assert spans, [s.name for s in exporter.get_finished_spans()]
    return spans[-1]


@pytest.mark.skipif(not _otel_sdk_available(), reason="OpenTelemetry SDK required")
async def test_run_sync_inside_a_loop_keeps_the_trace_parent(exporter: Any) -> None:
    from locus.hooks.builtin.telemetry import TelemetryHook
    from locus.observability.propagation import trace_scope

    agent = _agent(name="context-desk", hooks=[TelemetryHook()])
    with trace_scope({"traceparent": TRACEPARENT}):
        result = agent.run_sync("who am I?")
    assert result.message.startswith("caller:")
    span = _invoke_span(exporter, "context-desk")
    assert f"{span.context.trace_id:032x}" == TRACE_ID
    assert span.parent is not None
    assert f"{span.parent.span_id:016x}" == PARENT_SPAN_ID


@pytest.mark.skipif(not _otel_sdk_available(), reason="OpenTelemetry SDK required")
@pytest.mark.parametrize("tool_name", ["run_agent", "run_agent_stream"])
async def test_mcp_served_agent_runs_in_the_callers_trace(exporter: Any, tool_name: str) -> None:
    """The served agent's ``invoke_agent`` shares the caller's trace and has a
    parent — through ``run_agent`` (``run_sync`` on a worker thread) as much as
    through ``run_agent_stream`` (in the loop). The ``mcp`` client writes the
    span of its own send into ``_meta``, so the caller's span is what the
    served side inherits."""
    pytest.importorskip("fastmcp")
    from fastmcp import Client
    from opentelemetry import trace as otel_trace

    from locus.hooks.builtin.telemetry import TelemetryHook
    from locus.integrations.fastmcp import LocusMCPServer

    agent = _agent(name="mcp-desk", hooks=[TelemetryHook()])
    server = LocusMCPServer(agent=agent, name="context-test")
    tracer = otel_trace.get_tracer("test.caller")
    async with Client(server.build()) as client:
        with tracer.start_as_current_span("caller") as caller:
            result = await client.call_tool(tool_name, {"prompt": "who am I?"})
    assert result.content[0].text.startswith("caller:")
    span = _invoke_span(exporter, "mcp-desk")
    assert span.context.trace_id == caller.get_span_context().trace_id
    assert span.parent is not None
