# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``validate_spans`` and spans from other instrumentation scopes.

An exporter shared with a web framework or the ``mcp`` package also holds
their spans (``POST /``, ``MCP send tools/call …``); the checker reported
them as a foreign root and extra traces. Spans from more than one scope are
narrowed to the telemetry hook's own by default.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import pytest

from locus.observability import semconv as sc


FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "semconv_trace_notebook60.json"
FASTAPI_SCOPE = "opentelemetry.instrumentation.fastapi"


def _hook_trace(scope: str | None = sc.TELEMETRY_SCOPE) -> list[sc.SpanRecord]:
    return [dataclasses.replace(r, scope=scope) for r in sc.load_spans(FIXTURE.read_text())]


def _foreign_root(scope: str = FASTAPI_SCOPE) -> sc.SpanRecord:
    return sc.SpanRecord(
        name="POST /",
        kind="SERVER",
        trace_id="f" * 32,
        span_id="e" * 16,
        parent_span_id=None,
        status="UNSET",
        attributes={"http.request.method": "POST"},
        scope=scope,
    )


def test_foreign_scope_spans_are_left_out_by_default() -> None:
    mixed: list[Any] = [_foreign_root(), *_hook_trace()]
    assert sc.validate_spans(mixed) == []
    sc.assert_valid_spans(mixed)

    everything = sc.validate_spans(mixed, scope=None)
    assert any("root span 'POST /' is not an invoke_agent span" in p for p in everything)
    assert any("2 traces" in p for p in everything)

    only_fastapi = sc.validate_spans(mixed, scope=FASTAPI_SCOPE)
    assert any("root span 'POST /'" in p for p in only_fastapi)
    with pytest.raises(AssertionError, match="POST /"):
        sc.assert_valid_spans(mixed, scope=None)


def test_a_single_scope_is_checked_as_it_is() -> None:
    """A hook with its own ``tracer_name`` needs no ``scope=`` argument."""
    own = _hook_trace("my.app.telemetry")
    assert sc.validate_spans(own) == []
    # Alone, the foreign spans are checked too — and found wanting.
    assert any("POST /" in p for p in sc.validate_spans([_foreign_root()]))


def test_unknown_scopes_are_always_kept() -> None:
    """Records loaded from a fixture carry no scope; they count in every mode."""
    from_fixture = _hook_trace(None)
    assert sc.validate_spans([*from_fixture, _foreign_root()], scope=None) != []
    # Known hook spans plus unknown ones plus a foreign root: the unknown ones stay.
    mixed: list[Any] = [_foreign_root(), *from_fixture[:1], *_hook_trace()[1:]]
    assert sc.validate_spans(mixed) == []


def test_filtering_everything_out_says_so() -> None:
    mixed = [_foreign_root(), _foreign_root("mcp")]
    assert sc.validate_spans(mixed) == [f"no spans recorded in scope {sc.TELEMETRY_SCOPE!r}"]
    assert sc.validate_spans([]) == ["no spans recorded"]


def test_span_records_carry_the_scope() -> None:
    record = _foreign_root()
    assert sc.SpanRecord.from_dict(record.to_dict()) == record
    assert sc.SpanRecord.from_dict({**record.to_dict(), "scope": None}).scope is None
    # A fixture written before the field existed loads with scope=None.
    assert sc.load_spans(FIXTURE.read_text())[0].scope is None


def test_readable_spans_report_their_instrumentation_scope() -> None:
    pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider.get_tracer("my.scope").start_span("probe").end()
    (span,) = exporter.get_finished_spans()
    assert sc.SpanRecord.from_readable_span(span).scope == "my.scope"
    provider.shutdown()
