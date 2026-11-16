# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""W3C Trace Context on the wire, both directions.

The telemetry hook (:mod:`locus.hooks.builtin.telemetry`) files an agent run
as one trace. This module carries that trace across process boundaries the
SDK talks over — HTTP headers of an A2A request, the ``_meta`` object of an
MCP call, or any other string-keyed carrier — and picks up a trace a caller
sends in, so the run files under the caller's span rather than as a new root.

Outbound, :func:`inject_trace_headers` writes ``traceparent`` (and
``tracestate`` when set) for the span of the tool call in flight, else for
whatever span is current. Inbound, :func:`trace_scope` runs a block under the
context a carrier holds. Both do nothing when OpenTelemetry is not installed,
so the SDK's transports call them unconditionally.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from typing import Any


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


def _span_to_propagate(span: Any | None) -> Any | None:
    """``span`` if given, else the telemetry hook's span for the running tool call."""
    if span is not None:
        return span
    try:
        from locus.hooks.builtin.telemetry import current_tool_span  # noqa: PLC0415
    except ImportError:  # pragma: no cover - the hook module always imports
        return None
    return current_tool_span()


def inject_trace_headers(
    headers: MutableMapping[str, str] | None = None,
    *,
    span: Any | None = None,
) -> dict[str, str]:
    """Add ``traceparent`` / ``tracestate`` for the current trace to ``headers``.

    Args:
        headers: The carrier to write into; a new dict when ``None``.
        span: The span to propagate. Defaults to the telemetry hook's span for
            the tool call in flight, else the span current on this task.

    Returns:
        The carrier, as a plain ``dict``. Unchanged when OpenTelemetry is
        absent or no span is active, so a request never carries a dead
        ``traceparent``.
    """
    out: dict[str, str] = dict(headers) if headers is not None else {}
    if not OTEL_AVAILABLE:
        return out
    target = _span_to_propagate(span)
    context = _otel_trace.set_span_in_context(target) if target is not None else None
    TraceContextTextMapPropagator().inject(out, context=context)
    return out


def current_traceparent() -> str | None:
    """The ``traceparent`` of the span current on this task, or ``None``.

    Surfacing this to a user (a support ticket, a log line) lets them find
    the run in whichever backend the spans were exported to.
    """
    return inject_trace_headers(
        span=_otel_trace.get_current_span() if OTEL_AVAILABLE else None
    ).get(TRACEPARENT)


def extract_trace_context(carrier: Mapping[str, Any] | None) -> Any | None:
    """The OpenTelemetry context a carrier holds.

    ``None`` when there is no valid ``traceparent`` in it or OpenTelemetry is
    absent. Header names are matched case-insensitively, as HTTP requires.
    """
    if not OTEL_AVAILABLE or not carrier:
        return None
    lowered = {
        str(k).lower(): str(v)
        for k, v in carrier.items()
        if str(k).lower() in (TRACEPARENT, TRACESTATE)
    }
    if TRACEPARENT not in lowered:
        return None
    extracted = TraceContextTextMapPropagator().extract(carrier=lowered)
    if not _otel_trace.get_current_span(extracted).get_span_context().is_valid:
        return None
    return extracted


@contextmanager
def trace_scope(carrier: Mapping[str, Any] | None) -> Iterator[None]:
    """Run the block under the trace context ``carrier`` holds, if any.

    When a span of that same trace is already current (a server framework
    that extracted the headers itself), the block runs as it is and the
    framework's span stays the parent. Only when nothing has picked the
    context up is it attached here, and detached on the way out.
    """
    context = extract_trace_context(carrier)
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


__all__ = [
    "OTEL_AVAILABLE",
    "TRACEPARENT",
    "TRACESTATE",
    "current_traceparent",
    "extract_trace_context",
    "inject_trace_headers",
    "trace_scope",
]
