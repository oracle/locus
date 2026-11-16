# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Telemetry hook provider for OpenTelemetry integration.

One agent run is one trace. Under the default GenAI semantic conventions
(:data:`~locus.observability.semconv.SEMCONV_GENAI`) the run is an
``invoke_agent`` span, each model call a ``chat`` span and each tool call an
``execute_tool`` span, carrying the ``gen_ai.*`` attributes and feeding the
``gen_ai.client.inference.usage.*`` token counters and the duration
histograms the conventions define. ``semconv="locus-legacy"`` keeps the
``agent.invocation`` / ``tool.<name>`` / ``locus.*`` shape the SDK emitted
before, for dashboards built on it.

The root span is parented to whatever OpenTelemetry context is current when
the run starts, so a caller's ``traceparent`` — attached by
:func:`locus.observability.propagation.trace_scope` on an A2A or MCP server —
is inherited, and the tool span in flight is what
:func:`locus.observability.propagation.inject_trace_headers` and the MCP
client carry onward.
"""

from __future__ import annotations

import json
import time
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from locus.hooks.provider import (
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
    HookPriority,
    HookProvider,
)
from locus.observability import semconv as sc


if TYPE_CHECKING:
    from locus.core.state import AgentState

# Optional OpenTelemetry imports
try:
    from opentelemetry import context as otel_context
    from opentelemetry import metrics, trace
    from opentelemetry.trace import Span, SpanKind, Status, StatusCode

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None  # type: ignore[assignment]
    metrics = None  # type: ignore[assignment]
    otel_context = None  # type: ignore[assignment]
    Span = None  # type: ignore[assignment,misc]
    SpanKind = None  # type: ignore[assignment,misc]
    Status = None  # type: ignore[assignment,misc]
    StatusCode = None  # type: ignore[assignment,misc]


#: Tool spans in flight, by ``tool_call_id``, across every ``TelemetryHook``.
#: An integration running inside the tool (the MCP client, say) reads the span
#: of ITS call through :func:`current_tool_span` to propagate it onward.
_TOOL_CALL_SPANS: dict[str, Any] = {}

#: Longest opt-in payload (tool arguments, tool result) recorded on a span.
_PAYLOAD_LIMIT = 4000


def current_tool_span() -> Span | None:
    """The ``TelemetryHook`` span of the tool call running in this task.

    ``None`` outside a tool call, when no ``TelemetryHook`` is installed, or
    when OpenTelemetry is not available. Lets a tool — or a client library it
    calls, like the MCP client — carry the agent's trace to a remote system
    with this call's span as the parent, not merely the iteration's.
    """
    if not OTEL_AVAILABLE or not _TOOL_CALL_SPANS:
        return None
    from locus.tools.context import current_tool_context

    ctx = current_tool_context()
    if ctx is None:
        return None
    span: Span | None = _TOOL_CALL_SPANS.get(ctx.tool_call_id)
    return span


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _preview(value: Any, limit: int = _PAYLOAD_LIMIT) -> str:
    """A bounded string form of a payload, JSON when it serialises."""
    try:
        text = json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        try:
            text = str(value)
        except Exception:  # noqa: BLE001 — arbitrary user values; fall back to placeholder
            return "<non-serializable>"
    return text[:limit]


def _attrs(**pairs: Any) -> dict[str, Any]:
    """Attribute dict without the ``None`` values OpenTelemetry would drop with a warning."""
    return {k: v for k, v in pairs.items() if v is not None}


class TelemetryHook(HookProvider):
    """Hook provider for OpenTelemetry tracing and metrics.

    Spans (GenAI conventions, the default):

    - ``invoke_agent <agent name>`` — the run, INTERNAL, root unless a caller's
      trace context is current; ``gen_ai.usage.*`` totals on completion.
    - ``locus.iteration`` — one ReAct iteration (the SDK's own span; the
      index is the ``locus.iteration`` attribute).
    - ``chat <model>`` — one model call, CLIENT, with ``gen_ai.provider.name``,
      ``gen_ai.request.model``, usage, finish reasons and the response id.
    - ``execute_tool <tool>`` — one tool call, INTERNAL, with
      ``gen_ai.tool.name`` / ``gen_ai.tool.call.id``; arguments and result
      only when asked for.

    Metrics (GenAI conventions): ``gen_ai.client.inference.usage.input_tokens``
    and ``output_tokens`` counters, ``gen_ai.client.inference.duration``,
    ``gen_ai.invoke_agent.duration`` / ``inference_calls`` / ``tool_calls`` and
    ``gen_ai.execute_tool.duration`` histograms, in seconds with the
    recommended buckets.

    ``semconv="locus-legacy"`` emits ``agent.invocation``, ``agent.iteration.<n>``
    and ``tool.<name>`` spans with ``locus.*`` attributes, the ``locus.*``
    counters and millisecond histograms, and no model-call spans.

    Requires the `telemetry` extra: `pip install locus-sdk[telemetry]`

    Example:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor

        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)

        agent = Agent(model="oci:openai.gpt-4.1", hooks=[TelemetryHook()])
    """

    def __init__(
        self,
        service_name: str = "locus-agent",
        tracer_name: str = "locus.hooks.telemetry",
        meter_name: str = "locus.hooks.telemetry",
        record_arguments: bool = False,
        record_results: bool = False,
        priority: int = HookPriority.OBSERVABILITY_MIN + 10,
        *,
        semconv: str | None = None,
        model: Any = None,
        provider_name: str | None = None,
        agent_name: str | None = None,
    ) -> None:
        """Initialize telemetry hook.

        Args:
            service_name: Service name for telemetry. Under the GenAI
                conventions the service is a Resource attribute, so set
                ``OTEL_SERVICE_NAME`` or the provider's Resource instead;
                the legacy shape records it on the invocation span.
            tracer_name: Name for the OpenTelemetry tracer
            meter_name: Name for the OpenTelemetry meter
            record_arguments: Record tool arguments on tool spans
                (``gen_ai.tool.call.arguments``; ``locus.tool.arg.*`` legacy)
            record_results: Record tool results on tool spans
                (``gen_ai.tool.call.result``; ``locus.result_preview`` legacy)
            priority: Hook priority (default: early in observability range)
            semconv: Which conventions to emit — ``"genai/1.45"`` (default,
                alias ``"genai"``) or ``"locus-legacy"``.
            model: The agent's model, as the ``"provider:model"`` string or
                the model object, for ``gen_ai.request.model`` and
                ``gen_ai.provider.name``. Not needed when the hook is passed
                to ``Agent(hooks=[...])``: the agent lends it its model.
            provider_name: ``gen_ai.provider.name`` to record, when it cannot
                be read off the model (an OpenAI-compatible endpoint behind a
                custom ``base_url``).
            agent_name: ``gen_ai.agent.name``. Defaults to the agent's
                configured name or id.

        Raises:
            ImportError: If OpenTelemetry is not installed
            ValueError: An unknown ``semconv`` value
        """
        if not OTEL_AVAILABLE:
            msg = "OpenTelemetry is not installed. Install with: pip install locus[telemetry]"
            raise ImportError(msg)

        self._semconv = sc.resolve_semconv(semconv)
        self._genai = self._semconv == sc.SEMCONV_GENAI
        self._service_name = service_name
        self._tracer = trace.get_tracer(tracer_name)
        self._meter = metrics.get_meter(meter_name)
        self._record_arguments = record_arguments
        self._record_results = record_results
        self._priority = priority

        self._explicit_model = model is not None
        self._explicit_provider = provider_name is not None
        self._request_model: str | None = sc.request_model(model)
        self._provider_name: str | None = (
            provider_name or sc.provider_name(model) or sc.fallback_provider_name(model)
        )
        self._server_address: str | None = None
        self._server_port: int | None = None
        if model is not None and not isinstance(model, str):
            self._server_address, self._server_port = sc.server_address(model)
        self._agent_name = agent_name
        self._tool_descriptions: dict[str, str] = {}

        # Active spans tracking + OTel context tokens for proper parent-
        # child propagation. Without attaching each span to OTel's current
        # context, exporters see every span as a new trace root — backends
        # then render N separate traces instead of one tree per invocation.
        # See issue #244.
        self._invocation_span: Span | None = None
        self._invocation_token: Any = None
        self._invocation_started: float | None = None
        self._iteration_spans: dict[int, Span] = {}
        self._iteration_tokens: dict[int, Any] = {}
        self._current_iteration: int | None = None
        self._tool_spans: dict[str, tuple[Span, float, str]] = {}
        self._model_span: Span | None = None
        self._model_token: Any = None
        self._model_started: float = 0.0
        self._inference_calls = 0
        self._run_agent_name: str | None = None
        self._invocation_agent_name: str | None = None
        self._invocation_unnamed = False
        self._conversation_id: str | None = None

        if self._genai:
            self._inference_duration = self._histogram(
                sc.METRIC_INFERENCE_DURATION,
                "s",
                "GenAI client inference operation duration",
                sc.DURATION_BUCKETS_S,
            )
            self._input_tokens = self._meter.create_counter(
                sc.METRIC_INPUT_TOKENS,
                unit="{token}",
                description="Input (prompt) tokens used, including cached tokens",
            )
            self._output_tokens = self._meter.create_counter(
                sc.METRIC_OUTPUT_TOKENS,
                unit="{token}",
                description="Output (completion) tokens used, including reasoning tokens",
            )
            self._invoke_duration = self._histogram(
                sc.METRIC_INVOKE_AGENT_DURATION,
                "s",
                "End-to-end duration of an agent invocation",
                sc.INVOKE_AGENT_DURATION_BUCKETS_S,
            )
            self._invoke_inference_calls = self._histogram(
                sc.METRIC_INVOKE_AGENT_INFERENCE_CALLS,
                "{inference_call}",
                "Model calls made during one agent invocation",
                sc.CALL_COUNT_BUCKETS,
            )
            self._invoke_tool_calls = self._histogram(
                sc.METRIC_INVOKE_AGENT_TOOL_CALLS,
                "{tool_call}",
                "Tool calls made during one agent invocation",
                sc.CALL_COUNT_BUCKETS,
            )
            self._tool_duration = self._histogram(
                sc.METRIC_EXECUTE_TOOL_DURATION,
                "s",
                "Duration of a single tool execution",
                sc.DURATION_BUCKETS_S,
            )
        else:
            self._invocation_counter = self._meter.create_counter(
                "locus.invocations",
                description="Number of agent invocations",
                unit="1",
            )
            self._invocation_duration = self._meter.create_histogram(
                "locus.invocation.duration",
                description="Duration of agent invocations",
                unit="ms",
            )
            self._iteration_counter = self._meter.create_counter(
                "locus.iterations",
                description="Number of agent iterations",
                unit="1",
            )
            self._tool_call_counter = self._meter.create_counter(
                "locus.tool_calls",
                description="Number of tool calls",
                unit="1",
            )
            self._tool_call_duration = self._meter.create_histogram(
                "locus.tool_call.duration",
                description="Duration of tool calls",
                unit="ms",
            )
            self._tool_error_counter = self._meter.create_counter(
                "locus.tool_errors",
                description="Number of tool call errors",
                unit="1",
            )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def priority(self) -> int:
        """Return hook priority."""
        return self._priority

    @property
    def name(self) -> str:
        """Return hook name."""
        return "TelemetryHook"

    @property
    def semconv(self) -> str:
        """The conventions this hook emits: ``"genai/1.45"`` or ``"locus-legacy"``."""
        return self._semconv

    @property
    def request_model(self) -> str | None:
        """The ``gen_ai.request.model`` the hook records, if known."""
        return self._request_model

    @property
    def provider_name(self) -> str | None:
        """The ``gen_ai.provider.name`` the hook records, if known."""
        return self._provider_name

    def bind_agent(self, *, model: Any, tools: Iterable[Any]) -> None:
        """Learn the agent's model and tools (called by ``Agent`` at construction).

        Fills ``gen_ai.request.model`` / ``gen_ai.provider.name`` from the
        model unless the constructor was given them, and keeps each tool's
        description for ``gen_ai.tool.description``.
        """
        if not self._explicit_model:
            self._request_model = sc.request_model(model) or self._request_model
            self._server_address, self._server_port = sc.server_address(model)
            if not self._explicit_provider:
                self._provider_name = (
                    sc.provider_name(model)
                    or self._provider_name
                    or sc.fallback_provider_name(model)
                )
        for tool in tools:
            name = getattr(tool, "name", None)
            description = getattr(tool, "description", None)
            if isinstance(name, str) and isinstance(description, str) and description:
                self._tool_descriptions[name] = description

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _histogram(self, name: str, unit: str, description: str, buckets: tuple[float, ...]) -> Any:
        """A histogram with the conventions' bucket advisory where the API supports it."""
        try:
            return self._meter.create_histogram(
                name,
                unit=unit,
                description=description,
                explicit_bucket_boundaries_advisory=list(buckets),
            )
        except TypeError:  # OpenTelemetry API before the advisory parameter
            return self._meter.create_histogram(name, unit=unit, description=description)

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

    def _agent_label(self, state: AgentState | None = None) -> str | None:
        if self._agent_name:
            return self._agent_name
        if self._run_agent_name:
            return self._run_agent_name
        agent_id = getattr(state, "agent_id", None) if state is not None else None
        if isinstance(agent_id, str) and agent_id:
            return agent_id
        return self._invocation_agent_name

    def _absorb_run(self, run: Any) -> None:
        """Pick the conversation id and agent name off a hook event's ``run``."""
        if run is None:
            return
        thread_id = getattr(run, "thread_id", None)
        if self._conversation_id is None and isinstance(thread_id, str) and thread_id:
            self._conversation_id = thread_id
            if self._genai and self._invocation_span is not None:
                self._invocation_span.set_attribute(sc.GEN_AI_CONVERSATION_ID, thread_id)
        agent_name = getattr(run, "agent_name", None)
        if (
            self._agent_name is None
            and self._run_agent_name is None
            and isinstance(agent_name, str)
            and agent_name
        ):
            self._run_agent_name = agent_name
            span = self._invocation_span
            if self._genai and span is not None and self._invocation_unnamed:
                span.update_name(f"{sc.OP_INVOKE_AGENT} {agent_name}")
                span.set_attribute(sc.GEN_AI_AGENT_NAME, agent_name)
                self._invocation_unnamed = False

    def _close_model_span(self, error: str | None = None) -> None:
        """End the open ``chat`` span, as a failure when ``error`` is given."""
        span = self._model_span
        if span is None:
            return
        self._model_span = None
        if error is not None:
            span.set_status(Status(StatusCode.ERROR, error))
            span.set_attribute(sc.ERROR_TYPE, sc.ERROR_TYPE_OTHER)
        span.end()
        if self._model_token is not None:
            otel_context.detach(self._model_token)
            self._model_token = None

    def _close_open_iterations(self) -> None:
        """End iteration spans a failed run left open, newest first."""
        for iteration in sorted(self._iteration_spans, reverse=True):
            span = self._iteration_spans.pop(iteration)
            span.set_status(Status(StatusCode.ERROR, "iteration did not complete"))
            span.end()
            token = self._iteration_tokens.pop(iteration, None)
            if token is not None:
                otel_context.detach(token)
        self._current_iteration = None

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------

    async def on_before_invocation(
        self,
        prompt: str,
        state: AgentState,
    ) -> AgentState:
        """Start the run's root span and attach it as the current OTel context.

        The span is parented to whatever context is current, so a caller's
        ``traceparent`` (attached by ``trace_scope``) is inherited.

        Args:
            prompt: User prompt
            state: Agent state

        Returns:
            Unchanged state
        """
        self._inference_calls = 0
        self._conversation_id = None
        self._run_agent_name = None
        self._invocation_agent_name = None
        self._invocation_started = time.perf_counter()
        if self._genai:
            agent_name = self._agent_label(state)
            self._invocation_agent_name = agent_name
            self._invocation_unnamed = agent_name is None
            self._invocation_span = self._tracer.start_span(
                f"{sc.OP_INVOKE_AGENT} {agent_name}" if agent_name else sc.OP_INVOKE_AGENT,
                kind=SpanKind.INTERNAL,
                attributes=_attrs(
                    **{
                        sc.GEN_AI_OPERATION_NAME: sc.OP_INVOKE_AGENT,
                        sc.GEN_AI_AGENT_NAME: agent_name,
                        sc.GEN_AI_AGENT_ID: state.agent_id or None,
                        sc.GEN_AI_REQUEST_MODEL: self._request_model,
                        "locus.run_id": state.run_id,
                        "locus.prompt_length": len(prompt),
                        "locus.max_iterations": state.max_iterations,
                    }
                ),
            )
        else:
            self._invocation_span = self._tracer.start_span(
                "agent.invocation",
                attributes={
                    "locus.run_id": state.run_id,
                    "locus.agent_id": state.agent_id or "",
                    "locus.prompt_length": len(prompt),
                    "locus.max_iterations": state.max_iterations,
                    "service.name": self._service_name,
                },
            )
            self._invocation_counter.add(1, {"agent_id": state.agent_id or "default"})
        # Attach the span to OTel's current context so every span the
        # agent creates inside this invocation is automatically parented
        # to it. Without this, child spans become disconnected trace
        # roots (issue #244).
        ctx = trace.set_span_in_context(self._invocation_span)
        self._invocation_token = otel_context.attach(ctx)
        return state

    async def on_after_invocation(
        self,
        state: AgentState,
        success: bool,
    ) -> None:
        """End the run's root span and record the invocation metrics.

        Args:
            state: Final agent state
            success: Whether execution succeeded
        """
        if self._invocation_span is None:
            return
        # A run that raised may leave a model call or an iteration open;
        # close them (innermost first, so context tokens detach in order).
        self._close_model_span("model call did not complete")
        self._close_open_iterations()

        span = self._invocation_span
        started = self._invocation_started
        elapsed_s = (time.perf_counter() - started) if started is not None else 0.0
        duration_ms = (state.updated_at - state.started_at).total_seconds() * 1000
        errors = getattr(state, "errors", ()) or ()
        error_kind = sc.error_type(errors[-1] if errors else None) if not success else None

        if self._genai:
            cache_read = (_int_or_none(getattr(state, "cache_read_tokens_used", None)) or 0) + (
                _int_or_none(getattr(state, "cached_tokens_used", None)) or 0
            )
            cache_write = (
                _int_or_none(getattr(state, "cache_creation_tokens_used", None)) or 0
            ) + (_int_or_none(getattr(state, "cache_write_tokens_used", None)) or 0)
            span.set_attributes(
                _attrs(
                    **{
                        sc.GEN_AI_USAGE_INPUT_TOKENS: _int_or_none(
                            getattr(state, "prompt_tokens_used", None)
                        ),
                        sc.GEN_AI_USAGE_OUTPUT_TOKENS: _int_or_none(
                            getattr(state, "completion_tokens_used", None)
                        ),
                        sc.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS: cache_read or None,
                        sc.GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS: cache_write or None,
                        sc.ERROR_TYPE: error_kind,
                        "locus.success": success,
                        "locus.iterations": _int_or_none(state.iteration),
                        "locus.inference_calls": self._inference_calls,
                        "locus.tool_calls": len(state.tool_executions),
                        "locus.errors": len(errors),
                        "locus.confidence": state.confidence
                        if isinstance(state.confidence, (int, float))
                        else None,
                    }
                )
            )
        else:
            span.set_attributes(
                {
                    "locus.success": success,
                    "locus.iterations": state.iteration,
                    "locus.confidence": state.confidence,
                    "locus.tool_calls": len(state.tool_executions),
                    "locus.errors": len(errors),
                    "locus.duration_ms": duration_ms,
                }
            )

        if success:
            span.set_status(Status(StatusCode.OK))
        else:
            span.set_status(Status(StatusCode.ERROR, "Agent invocation failed"))
        span.end()
        self._invocation_span = None

        # Detach the invocation context — pair to the attach in
        # `on_before_invocation`. Detaching restores whatever
        # context was active before locus took over.
        if self._invocation_token is not None:
            otel_context.detach(self._invocation_token)
            self._invocation_token = None

        if self._genai:
            agent_name = self._agent_label(state)
            invoke_attrs = _attrs(
                **{
                    sc.GEN_AI_OPERATION_NAME: sc.OP_INVOKE_AGENT,
                    sc.GEN_AI_AGENT_NAME: agent_name,
                    sc.GEN_AI_REQUEST_MODEL: self._request_model,
                    sc.ERROR_TYPE: error_kind,
                }
            )
            self._invoke_duration.record(elapsed_s, invoke_attrs)
            count_attrs = _attrs(
                **{sc.GEN_AI_OPERATION_NAME: sc.OP_INVOKE_AGENT, sc.GEN_AI_AGENT_NAME: agent_name}
            )
            self._invoke_inference_calls.record(self._inference_calls, count_attrs)
            self._invoke_tool_calls.record(len(state.tool_executions), count_attrs)
        else:
            self._invocation_duration.record(
                duration_ms,
                {
                    "agent_id": state.agent_id or "default",
                    "success": str(success),
                },
            )

    # ------------------------------------------------------------------
    # Model calls
    # ------------------------------------------------------------------

    def _inference_attrs(self, response_model: str | None = None) -> dict[str, Any]:
        return _attrs(
            **{
                sc.GEN_AI_OPERATION_NAME: sc.OP_CHAT,
                sc.GEN_AI_PROVIDER_NAME: self._provider_name,
                sc.GEN_AI_REQUEST_MODEL: self._request_model,
                sc.GEN_AI_RESPONSE_MODEL: response_model,
                sc.SERVER_ADDRESS: self._server_address,
                sc.SERVER_PORT: self._server_port,
            }
        )

    async def on_before_model_call(self, event: BeforeModelCallEvent) -> None:
        """Start a ``chat`` span for the model call (GenAI conventions only).

        The span is made current, so an instrumented HTTP client inside the
        provider files its request under it.
        """
        if not self._genai:
            return
        self._close_model_span("model call did not complete")
        self._absorb_run(getattr(event, "run", None))
        model = self._request_model
        attrs = self._inference_attrs()
        attrs.update(
            _attrs(
                **{
                    sc.GEN_AI_CONVERSATION_ID: self._conversation_id,
                    "locus.messages": len(event.messages),
                    "locus.tools": len(event.tools or []),
                }
            )
        )
        span = self._tracer.start_span(
            f"{sc.OP_CHAT} {model}" if model else sc.OP_CHAT,
            kind=SpanKind.CLIENT,
            attributes=attrs,
        )
        self._model_span = span
        self._model_token = otel_context.attach(trace.set_span_in_context(span))
        self._model_started = time.perf_counter()

    async def on_after_model_call(self, event: AfterModelCallEvent) -> None:
        """End the ``chat`` span with usage, finish reason and response ids."""
        if not self._genai or self._model_span is None:
            return
        span = self._model_span
        elapsed_s = time.perf_counter() - self._model_started
        response = event.response
        usage = getattr(response, "usage", None)
        usage = usage if isinstance(usage, dict) else {}
        metadata = getattr(response, "metadata", None)
        metadata = metadata if isinstance(metadata, dict) else {}
        provider_state = getattr(response, "provider_state", None)
        provider_state = provider_state if isinstance(provider_state, dict) else {}

        input_tokens = _int_or_none(usage.get("prompt_tokens"))
        output_tokens = _int_or_none(usage.get("completion_tokens"))
        cache_read = sum(
            _int_or_none(usage.get(k)) or 0 for k in ("cache_read_input_tokens", "cached_tokens")
        )
        cache_write = sum(
            _int_or_none(usage.get(k)) or 0
            for k in ("cache_creation_input_tokens", "cache_write_tokens")
        )
        stop_reason = getattr(response, "stop_reason", None)
        response_id = (
            metadata.get("response_id")
            or metadata.get("id")
            or provider_state.get("previous_response_id")
        )
        response_model = metadata.get("model")
        span.set_attributes(
            _attrs(
                **{
                    sc.GEN_AI_USAGE_INPUT_TOKENS: input_tokens,
                    sc.GEN_AI_USAGE_OUTPUT_TOKENS: output_tokens,
                    sc.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS: cache_read or None,
                    sc.GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS: cache_write or None,
                    sc.GEN_AI_RESPONSE_FINISH_REASONS: [stop_reason]
                    if isinstance(stop_reason, str) and stop_reason
                    else None,
                    sc.GEN_AI_RESPONSE_ID: str(response_id) if response_id else None,
                    sc.GEN_AI_RESPONSE_MODEL: response_model
                    if isinstance(response_model, str) and response_model
                    else None,
                }
            )
        )
        span.set_status(Status(StatusCode.OK))
        self._close_model_span()
        self._inference_calls += 1

        metric_attrs = self._inference_attrs(
            response_model if isinstance(response_model, str) else None
        )
        self._inference_duration.record(elapsed_s, metric_attrs)
        token_attrs = metric_attrs | {sc.GEN_AI_TOKEN_MODALITY: sc.TOKEN_MODALITY_TEXT}
        # The conventions count only known, strictly positive contributions.
        if input_tokens:
            self._input_tokens.add(input_tokens, token_attrs)
        if output_tokens:
            self._output_tokens.add(output_tokens, token_attrs)

    # ------------------------------------------------------------------
    # Tool calls
    # ------------------------------------------------------------------

    async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        """Start tool call span.

        Args:
            event: Write-protected event carrying ``tool_name`` and
                ``arguments``. The hook only inspects them.
        """
        tool_name = event.tool_name
        call_id = getattr(event, "tool_call_id", "") or ""
        self._absorb_run(getattr(event, "run", None))

        if self._genai:
            span_attrs = _attrs(
                **{
                    sc.GEN_AI_OPERATION_NAME: sc.OP_EXECUTE_TOOL,
                    sc.GEN_AI_TOOL_NAME: tool_name,
                    sc.GEN_AI_TOOL_CALL_ID: call_id or None,
                    sc.GEN_AI_TOOL_TYPE: sc.TOOL_TYPE_FUNCTION,
                    sc.GEN_AI_TOOL_DESCRIPTION: self._tool_descriptions.get(tool_name),
                    sc.GEN_AI_AGENT_NAME: self._agent_label(),
                    sc.GEN_AI_CONVERSATION_ID: self._conversation_id,
                }
            )
            if self._record_arguments:
                span_attrs[sc.GEN_AI_TOOL_CALL_ARGUMENTS] = _preview(event.arguments)
            span = self._tracer.start_span(
                f"{sc.OP_EXECUTE_TOOL} {tool_name}",
                kind=SpanKind.INTERNAL,
                attributes=span_attrs,
            )
        else:
            span_attrs = {"locus.tool_name": tool_name}
            if self._record_arguments:
                # Sanitize arguments for span attributes
                for key, value in event.arguments.items():
                    attr_key = f"locus.tool.arg.{key}"
                    try:
                        span_attrs[attr_key] = str(value)[:1000]  # Limit length
                    except Exception:  # noqa: BLE001 — arbitrary user values; fall back to placeholder
                        span_attrs[attr_key] = "<non-serializable>"
            span = self._tracer.start_span(f"tool.{tool_name}", attributes=span_attrs)
            self._tool_call_counter.add(1, {"tool_name": tool_name})

        self._tool_spans[call_id or tool_name] = (span, time.perf_counter(), tool_name)
        if call_id:
            _TOOL_CALL_SPANS[call_id] = span

    def _pop_tool_span(self, call_id: str, tool_name: str) -> tuple[Span, float] | None:
        """The open span for a finished call: by id, else the oldest for the tool."""
        entry = self._tool_spans.pop(call_id, None) if call_id else None
        if entry is None:
            for key, (span, started, name) in self._tool_spans.items():
                if name == tool_name:
                    del self._tool_spans[key]
                    entry = (span, started, name)
                    break
        if entry is None:
            return None
        span, started, _ = entry
        if call_id:
            _TOOL_CALL_SPANS.pop(call_id, None)
        else:  # an event built without the id: drop the entry by identity
            for key, active in list(_TOOL_CALL_SPANS.items()):
                if active is span:
                    del _TOOL_CALL_SPANS[key]
        return span, started

    async def on_after_tool_call(self, event: AfterToolCallEvent) -> None:
        """End tool call span.

        Args:
            event: Write-protected event carrying ``tool_name``,
                ``result``, and ``error``.
        """
        tool_name = event.tool_name
        error = event.error
        result = event.result
        popped = self._pop_tool_span(getattr(event, "tool_call_id", "") or "", tool_name)
        if popped is None:
            return
        span, start_time = popped
        elapsed_s = time.perf_counter() - start_time
        duration_ms = elapsed_s * 1000

        if self._genai:
            error_kind = sc.error_type(error) if error else None
            span.set_attribute("locus.duration_ms", duration_ms)
            if error:
                span.set_status(Status(StatusCode.ERROR, error))
                span.set_attribute(sc.ERROR_TYPE, error_kind or sc.ERROR_TYPE_OTHER)
            else:
                span.set_status(Status(StatusCode.OK))
                if self._record_results and result is not None:
                    span.set_attribute(sc.GEN_AI_TOOL_CALL_RESULT, _preview(result))
            span.end()
            self._tool_duration.record(
                elapsed_s,
                _attrs(
                    **{
                        sc.GEN_AI_TOOL_NAME: tool_name,
                        sc.GEN_AI_TOOL_TYPE: sc.TOOL_TYPE_FUNCTION,
                        sc.GEN_AI_AGENT_NAME: self._agent_label(),
                        sc.ERROR_TYPE: error_kind,
                    }
                ),
            )
            return

        span.set_attribute("locus.duration_ms", duration_ms)
        if error:
            span.set_status(Status(StatusCode.ERROR, error))
            span.set_attribute("locus.error", error[:1000])
            self._tool_error_counter.add(1, {"tool_name": tool_name})
        else:
            span.set_status(Status(StatusCode.OK))
            if self._record_results and result is not None:
                result_str = str(result)
                span.set_attribute("locus.result_preview", result_str[:500])
        span.end()
        self._tool_call_duration.record(
            duration_ms,
            {
                "tool_name": tool_name,
                "success": str(error is None),
            },
        )

    # ------------------------------------------------------------------
    # Iterations
    # ------------------------------------------------------------------

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
        attributes = {
            "locus.iteration": iteration,
            "locus.confidence": state.confidence,
            "locus.messages": len(state.messages),
        }
        if self._genai:
            span = self._tracer.start_span(
                sc.ITERATION_SPAN_NAME, kind=SpanKind.INTERNAL, attributes=attributes
            )
        else:
            span = self._tracer.start_span(f"agent.iteration.{iteration}", attributes=attributes)
            self._iteration_counter.add(1, {"agent_id": state.agent_id or "default"})
        # Push this iteration's context onto OTel's stack so any tool
        # spans created during the iteration become children of it
        # (which is itself a child of the invocation).
        ctx = trace.set_span_in_context(span)
        self._iteration_spans[iteration] = span
        self._iteration_tokens[iteration] = otel_context.attach(ctx)
        self._current_iteration = iteration

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
        if iteration not in self._iteration_spans:
            return
        # A model call that raised never reached ``on_after_model_call``;
        # close it first so context tokens detach in attach order.
        self._close_model_span("model call did not complete")
        span = self._iteration_spans.pop(iteration)
        span.set_attributes(
            {
                "locus.confidence_after": state.confidence,
                "locus.messages_after": len(state.messages),
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
