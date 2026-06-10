# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Telemetry hook provider for OpenTelemetry integration."""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from tulip.hooks.provider import (
    AfterToolCallEvent,
    BeforeToolCallEvent,
    HookPriority,
    HookProvider,
)


if TYPE_CHECKING:
    from tulip.core.state import AgentState

# Optional OpenTelemetry imports
try:
    from opentelemetry import context as otel_context
    from opentelemetry import metrics, trace
    from opentelemetry.trace import Span, Status, StatusCode

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None  # type: ignore[assignment]
    metrics = None  # type: ignore[assignment]
    otel_context = None  # type: ignore[assignment]
    Span = None  # type: ignore[assignment,misc]
    Status = None  # type: ignore[assignment,misc]
    StatusCode = None  # type: ignore[assignment,misc]


class TelemetryHook(HookProvider):
    """Hook provider for OpenTelemetry tracing and metrics.

    Provides automatic instrumentation for:
    - Trace spans for agent invocations and iterations
    - Trace spans for tool calls
    - Metrics for invocation duration, tool call counts, etc.

    Requires the `telemetry` extra: `pip install tulip[telemetry]`

    Example:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor

        # Configure OpenTelemetry
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)

        # Add telemetry hook
        registry.add_provider(TelemetryHook())
    """

    def __init__(
        self,
        service_name: str = "tulip-agent",
        tracer_name: str = "tulip.hooks.telemetry",
        meter_name: str = "tulip.hooks.telemetry",
        record_arguments: bool = False,
        record_results: bool = False,
        priority: int = HookPriority.OBSERVABILITY_MIN + 10,
    ) -> None:
        """Initialize telemetry hook.

        Args:
            service_name: Service name for telemetry
            tracer_name: Name for the OpenTelemetry tracer
            meter_name: Name for the OpenTelemetry meter
            record_arguments: Whether to record tool arguments as span attributes
            record_results: Whether to record tool results as span attributes
            priority: Hook priority (default: early in observability range)

        Raises:
            ImportError: If OpenTelemetry is not installed
        """
        if not OTEL_AVAILABLE:
            msg = "OpenTelemetry is not installed. Install with: pip install tulip[telemetry]"
            raise ImportError(msg)

        self._service_name = service_name
        self._tracer = trace.get_tracer(tracer_name)
        self._meter = metrics.get_meter(meter_name)
        self._record_arguments = record_arguments
        self._record_results = record_results
        self._priority = priority

        # Active spans tracking + OTel context tokens for proper parent-
        # child propagation. Without attaching each span to OTel's current
        # context, exporters see every span as a new trace root — Langfuse
        # and similar backends then render N separate traces instead of
        # one tree per invocation. See issue #244.
        self._invocation_span: Span | None = None
        self._invocation_token: Any = None
        self._iteration_spans: dict[int, Span] = {}
        self._iteration_tokens: dict[int, Any] = {}
        self._current_iteration: int | None = None
        self._tool_spans: dict[str, tuple[Span, float]] = {}

        # Metrics
        self._invocation_counter = self._meter.create_counter(
            "tulip.invocations",
            description="Number of agent invocations",
            unit="1",
        )
        self._invocation_duration = self._meter.create_histogram(
            "tulip.invocation.duration",
            description="Duration of agent invocations",
            unit="ms",
        )
        self._iteration_counter = self._meter.create_counter(
            "tulip.iterations",
            description="Number of agent iterations",
            unit="1",
        )
        self._tool_call_counter = self._meter.create_counter(
            "tulip.tool_calls",
            description="Number of tool calls",
            unit="1",
        )
        self._tool_call_duration = self._meter.create_histogram(
            "tulip.tool_call.duration",
            description="Duration of tool calls",
            unit="ms",
        )
        self._tool_error_counter = self._meter.create_counter(
            "tulip.tool_errors",
            description="Number of tool call errors",
            unit="1",
        )

    @property
    def priority(self) -> int:
        """Return hook priority."""
        return self._priority

    @property
    def name(self) -> str:
        """Return hook name."""
        return "TelemetryHook"

    @contextmanager
    def _span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Generator[Span, None, None]:
        """Create a span context manager.

        Args:
            name: Span name
            attributes: Span attributes

        Yields:
            The active span
        """
        with self._tracer.start_as_current_span(name, attributes=attributes) as span:
            yield span

    async def on_before_invocation(
        self,
        prompt: str,
        state: AgentState,
    ) -> AgentState:
        """Start invocation span and attach it as the current OTel context.

        Args:
            prompt: User prompt
            state: Agent state

        Returns:
            Unchanged state
        """
        self._invocation_span = self._tracer.start_span(
            "agent.invocation",
            attributes={
                "tulip.run_id": state.run_id,
                "tulip.agent_id": state.agent_id or "",
                "tulip.prompt_length": len(prompt),
                "tulip.max_iterations": state.max_iterations,
                "service.name": self._service_name,
            },
        )
        # Attach the span to OTel's current context so every span the
        # agent creates inside this invocation is automatically parented
        # to it. Without this, child spans become disconnected trace
        # roots (issue #244).
        ctx = trace.set_span_in_context(self._invocation_span)
        self._invocation_token = otel_context.attach(ctx)
        self._invocation_counter.add(1, {"agent_id": state.agent_id or "default"})
        return state

    async def on_after_invocation(
        self,
        state: AgentState,
        success: bool,
    ) -> None:
        """End invocation span.

        Args:
            state: Final agent state
            success: Whether execution succeeded
        """
        if self._invocation_span:
            duration_ms = (state.updated_at - state.started_at).total_seconds() * 1000

            self._invocation_span.set_attributes(
                {
                    "tulip.success": success,
                    "tulip.iterations": state.iteration,
                    "tulip.confidence": state.confidence,
                    "tulip.tool_calls": len(state.tool_executions),
                    "tulip.errors": len(state.errors),
                    "tulip.duration_ms": duration_ms,
                }
            )

            if success:
                self._invocation_span.set_status(Status(StatusCode.OK))
            else:
                self._invocation_span.set_status(
                    Status(StatusCode.ERROR, "Agent invocation failed")
                )

            self._invocation_span.end()
            self._invocation_span = None

            # Detach the invocation context — pair to the attach in
            # `on_before_invocation`. Detaching restores whatever
            # context was active before tulip took over.
            if self._invocation_token is not None:
                otel_context.detach(self._invocation_token)
                self._invocation_token = None

            # Record duration metric
            self._invocation_duration.record(
                duration_ms,
                {
                    "agent_id": state.agent_id or "default",
                    "success": str(success),
                },
            )

    async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        """Start tool call span.

        Args:
            event: Write-protected event carrying ``tool_name`` and
                ``arguments``. The hook only inspects them.
        """
        tool_name = event.tool_name
        span_attrs: dict[str, Any] = {
            "tulip.tool_name": tool_name,
        }

        if self._record_arguments:
            # Sanitize arguments for span attributes
            for key, value in event.arguments.items():
                attr_key = f"tulip.tool.arg.{key}"
                try:
                    span_attrs[attr_key] = str(value)[:1000]  # Limit length
                except Exception:  # noqa: BLE001 — arbitrary user values; fall back to placeholder
                    span_attrs[attr_key] = "<non-serializable>"

        span = self._tracer.start_span(f"tool.{tool_name}", attributes=span_attrs)
        self._tool_spans[tool_name] = (span, time.perf_counter())

        self._tool_call_counter.add(1, {"tool_name": tool_name})

    async def on_after_tool_call(self, event: AfterToolCallEvent) -> None:
        """End tool call span.

        Args:
            event: Write-protected event carrying ``tool_name``,
                ``result``, and ``error``.
        """
        tool_name = event.tool_name
        error = event.error
        result = event.result
        if tool_name in self._tool_spans:
            span, start_time = self._tool_spans.pop(tool_name)
            duration_ms = (time.perf_counter() - start_time) * 1000

            span.set_attribute("tulip.duration_ms", duration_ms)

            if error:
                span.set_status(Status(StatusCode.ERROR, error))
                span.set_attribute("tulip.error", error[:1000])
                self._tool_error_counter.add(1, {"tool_name": tool_name})
            else:
                span.set_status(Status(StatusCode.OK))
                if self._record_results and result is not None:
                    result_str = str(result)
                    span.set_attribute("tulip.result_preview", result_str[:500])

            span.end()

            self._tool_call_duration.record(
                duration_ms,
                {
                    "tool_name": tool_name,
                    "success": str(error is None),
                },
            )

    async def on_iteration_start(
        self,
        iteration: int,
        state: AgentState,
    ) -> None:
        """Start iteration span as a child of the invocation span.

        Args:
            iteration: Iteration number
            state: Current state
        """
        span = self._tracer.start_span(
            f"agent.iteration.{iteration}",
            attributes={
                "tulip.iteration": iteration,
                "tulip.confidence": state.confidence,
                "tulip.messages": len(state.messages),
            },
        )
        # Push this iteration's context onto OTel's stack so any tool
        # spans created during the iteration become children of it
        # (which is itself a child of the invocation).
        ctx = trace.set_span_in_context(span)
        self._iteration_spans[iteration] = span
        self._iteration_tokens[iteration] = otel_context.attach(ctx)
        self._current_iteration = iteration
        self._iteration_counter.add(1, {"agent_id": state.agent_id or "default"})

    async def on_iteration_end(
        self,
        iteration: int,
        state: AgentState,
    ) -> None:
        """End iteration span.

        Args:
            iteration: Iteration number
            state: Current state
        """
        if iteration in self._iteration_spans:
            span = self._iteration_spans.pop(iteration)
            span.set_attributes(
                {
                    "tulip.confidence_after": state.confidence,
                    "tulip.messages_after": len(state.messages),
                }
            )
            span.set_status(Status(StatusCode.OK))
            span.end()
            # Detach the iteration context — pair to attach in
            # `on_iteration_start`. After detach, subsequent spans
            # parent to the invocation again.
            token = self._iteration_tokens.pop(iteration, None)
            if token is not None:
                otel_context.detach(token)
            if self._current_iteration == iteration:
                self._current_iteration = None


class NoOpTelemetryHook(HookProvider):
    """No-op telemetry hook for when OpenTelemetry is not available.

    This hook does nothing but can be used as a drop-in replacement
    for TelemetryHook when telemetry is disabled.
    """

    def __init__(self, priority: int = HookPriority.OBSERVABILITY_MIN + 10) -> None:
        """Initialize no-op hook.

        Args:
            priority: Hook priority
        """
        self._priority = priority

    @property
    def priority(self) -> int:
        """Return hook priority."""
        return self._priority

    @property
    def name(self) -> str:
        """Return hook name."""
        return "NoOpTelemetryHook"


def create_telemetry_hook(
    enabled: bool = True,
    **kwargs: Any,
) -> HookProvider:
    """Factory to create a telemetry hook.

    Creates TelemetryHook if enabled and OpenTelemetry is available,
    otherwise creates NoOpTelemetryHook.

    Args:
        enabled: Whether telemetry should be enabled
        **kwargs: Arguments to pass to TelemetryHook

    Returns:
        TelemetryHook or NoOpTelemetryHook
    """
    if not enabled:
        return NoOpTelemetryHook()

    if not OTEL_AVAILABLE:
        import logging

        logging.getLogger(__name__).warning(
            "OpenTelemetry not available, using no-op telemetry hook"
        )
        return NoOpTelemetryHook()

    return TelemetryHook(**kwargs)
