# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""W3C trace context in MCP ``_meta``.

An MCP request carries an open ``_meta`` object. Locus puts the W3C
``traceparent`` / ``tracestate`` pair there on every ``tools/call`` it sends,
so a server that reads it files its own spans under the agent's tool span; and
reads the pair off every call it serves, so an agent exposed over MCP files its
run under the caller's span. Both ends use OpenTelemetry when it is installed
and do nothing when it is not.

:func:`trace_meta` is the client side, :func:`incoming_trace_scope` the server
side. The telemetry hook (:mod:`locus.hooks.builtin.telemetry`) owns the tool
span; :func:`trace_meta` asks it for the span of the tool call in flight, so
the context sent is the tool span's, not merely the iteration's.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any


logger = logging.getLogger(__name__)

TRACEPARENT = "traceparent"
TRACESTATE = "tracestate"

try:
    from opentelemetry import context as _otel_context
    from opentelemetry import trace as _otel_trace
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised where the extra is absent
    OTEL_AVAILABLE = False
    _otel_context = None  # type: ignore[assignment]
    _otel_trace = None  # type: ignore[assignment]
    TraceContextTextMapPropagator = None  # type: ignore[assignment,misc]


def _current_tool_span() -> Any | None:
    """The telemetry hook's span for the tool call running in this task."""
    try:
        from locus.hooks.builtin.telemetry import current_tool_span
    except ImportError:  # pragma: no cover - the hook module always imports
        return None
    return current_tool_span()


def trace_meta(span: Any | None = None) -> dict[str, str]:
    """The ``_meta`` entries that carry the current trace to an MCP server.

    Args:
        span: The span to propagate. Defaults to the telemetry hook's span for
            the tool call in flight, else whatever span is current.

    Returns:
        ``{"traceparent": ..., "tracestate": ...}`` (``tracestate`` only when
        set), or ``{}`` when OpenTelemetry is absent or no span is active.
    """
    if not OTEL_AVAILABLE:
        return {}
    if span is None:
        span = _current_tool_span()
    context = _otel_trace.set_span_in_context(span) if span is not None else None
    carrier: dict[str, str] = {}
    TraceContextTextMapPropagator().inject(carrier, context=context)
    return carrier


def context_from_meta(meta: Mapping[str, Any] | None) -> Any | None:
    """The OpenTelemetry context an incoming request's ``_meta`` carries.

    ``None`` when there is no ``traceparent`` in it or OpenTelemetry is absent.
    """
    if not OTEL_AVAILABLE or not meta:
        return None
    carrier = {k: str(v) for k, v in meta.items() if k in (TRACEPARENT, TRACESTATE)}
    if TRACEPARENT not in carrier:
        return None
    extracted = TraceContextTextMapPropagator().extract(carrier=carrier)
    if not _otel_trace.get_current_span(extracted).get_span_context().is_valid:
        return None
    return extracted


@contextmanager
def trace_scope(meta: Mapping[str, Any] | None) -> Iterator[None]:
    """Run the block under the trace context ``meta`` carries, if any.

    A server stack that already extracted it (the ``mcp`` package does, from
    2.x) has a span of that trace current; the block then runs as it is, so
    the server's own span stays the parent. Only when nothing has picked the
    context up is it attached here.
    """
    context = context_from_meta(meta)
    if context is None:
        yield
        return
    incoming = _otel_trace.get_current_span(context).get_span_context()
    current = _otel_trace.get_current_span().get_span_context()
    if current.is_valid and current.trace_id == incoming.trace_id:
        yield
        return
    token = _otel_context.attach(context)
    try:
        yield
    finally:
        _otel_context.detach(token)


@contextmanager
def outgoing_trace_scope(propagate: bool = True) -> Iterator[None]:
    """Make the agent's tool span current while an MCP request is sent.

    The ``mcp`` package (2.x) opens a client span for every request under the
    current context and writes that span into ``_meta`` itself; with the
    telemetry hook's tool span current, the server's work files under the
    agent's tool call. ``propagate=False`` runs the block under an empty
    context instead, so nothing of the agent's trace leaves the process.
    """
    if not OTEL_AVAILABLE:
        yield
        return
    if not propagate:
        context: Any = _otel_context.Context()
    else:
        span = _current_tool_span()
        if span is None:
            yield
            return
        context = _otel_trace.set_span_in_context(span)
    token = _otel_context.attach(context)
    try:
        yield
    finally:
        _otel_context.detach(token)


def incoming_meta() -> dict[str, Any] | None:
    """The ``_meta`` of the MCP request this task is serving, if any.

    Reads the FastMCP request context, so it works inside any tool a
    :class:`~locus.integrations.LocusMCPServer` registers; ``None`` outside a
    request or when FastMCP is not the server.
    """
    try:
        from fastmcp.server.context import _current_context
    except ImportError:
        return None
    ctx = _current_context.get()
    request_context = getattr(ctx, "request_context", None) if ctx is not None else None
    meta = getattr(request_context, "meta", None)
    return dict(meta) if isinstance(meta, Mapping) else None


@contextmanager
def incoming_trace_scope() -> Iterator[None]:
    """Run a served MCP call under the trace context its ``_meta`` carries."""
    with trace_scope(incoming_meta()):
        yield


__all__ = [
    "OTEL_AVAILABLE",
    "TRACEPARENT",
    "TRACESTATE",
    "context_from_meta",
    "incoming_meta",
    "incoming_trace_scope",
    "outgoing_trace_scope",
    "trace_meta",
    "trace_scope",
]
