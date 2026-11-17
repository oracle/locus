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

The hook keeps its state per run, keyed by the run id the hook events carry,
so one hook — one ``Agent`` — serves concurrent runs and nested ones (a
sub-agent exposed as a tool) with a trace each.

The root span is parented to the tool call in flight when the run starts
inside one (a sub-agent's run sits under its caller's ``execute_tool``), else
to whatever OpenTelemetry context is current, so a caller's ``traceparent`` —
attached by :func:`locus.observability.propagation.trace_scope` on an A2A or
MCP server — is inherited. The tool span in flight is what
:func:`locus.observability.propagation.inject_trace_headers` and the MCP
client carry onward.
"""

from __future__ import annotations

import contextvars
import json
import time
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

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
#: of ITS call through :func:`current_tool_span` to propagate it onward. An
#: entry leaves with its call, or with its run when the run ends first.
_TOOL_CALL_SPANS: dict[str, Any] = {}

#: The run current on this task (its id): set by ``on_before_invocation``,
#: restored by ``on_after_invocation``. How a hook event that names no run —
#: one built by hand — finds the run it belongs to.
_CURRENT_RUN: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "locus_telemetry_run", default=None
)

#: Longest opt-in payload (tool arguments, tool result) recorded on a span.
_PAYLOAD_LIMIT = 4000

#: Per-modality token counts a provider may report inside its usage details,
#: under the keys OpenAI-compatible usage blocks use for them.
_TOKEN_MODALITY_KEYS: tuple[tuple[str, str], ...] = (
    ("text_tokens", sc.TOKEN_MODALITY_TEXT),
    ("audio_tokens", sc.TOKEN_MODALITY_AUDIO),
    ("image_tokens", sc.TOKEN_MODALITY_IMAGE),
    ("video_tokens", sc.TOKEN_MODALITY_VIDEO),
)
_INPUT_DETAIL_KEYS = ("prompt_tokens_details", "input_tokens_details", "input_token_details")
_OUTPUT_DETAIL_KEYS = (
    "completion_tokens_details",
    "output_tokens_details",
    "output_token_details",
)


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


def _token_points(
    total: int | None, usage: dict[str, Any], detail_keys: tuple[str, ...]
) -> list[tuple[str, int]]:
    """``(modality, count)`` pairs for one usage figure.

    The conventions want ``gen_ai.token.modality`` set to ``unknown`` when the
    modality cannot be determined; only a per-modality breakdown inside the
    usage details (``audio_tokens``, ``text_tokens``, …) names one, and the
    tokens it leaves unaccounted for stay ``unknown``.
    """
    if not total or total <= 0:
        return []
    details = next((usage[k] for k in detail_keys if isinstance(usage.get(k), dict)), None)
    if details is None:
        return [(sc.TOKEN_MODALITY_UNKNOWN, total)]
    points: list[tuple[str, int]] = []
    for key, modality in _TOKEN_MODALITY_KEYS:
        count = _int_or_none(details.get(key))
        if count is not None and count > 0:
            points.append((modality, count))
    if not points:
        return [(sc.TOKEN_MODALITY_UNKNOWN, total)]
    rest = total - sum(count for _, count in points)
    if rest > 0:
        points.append((sc.TOKEN_MODALITY_UNKNOWN, rest))
    return points


@dataclass
class _ToolCall:
    """One tool call's open span."""

    span: Any
    started: float
    name: str
    #: The context token when this call's span is current on the task.
    token: Any = None


@dataclass
class _ModelCall:
    """One model call's open ``chat`` span."""

    span: Any
    token: Any
    started: float
    #: 1 for the call the before-model hooks saw; one more per hook retry.
    attempt: int = 1


@dataclass
class _Run:
    """The hook's state for one agent run: its root span and what is open under it."""

    run_id: str
    span: Any
    token: Any
    started: float
    #: The agent label the root span was started with; ``unnamed`` when none.
    agent_name: str | None
    unnamed: bool
    #: The run current on this task before this one, and the token to restore it.
    previous: str | None
    context_token: Any = None
    run_agent_name: str | None = None
    conversation_id: str | None = None
    inference_calls: int = 0
    iterations: dict[int, Any] = field(default_factory=dict)
    iteration_tokens: dict[int, Any] = field(default_factory=dict)
    current_iteration: int | None = None
    model: _ModelCall | None = None
    tools: dict[str, _ToolCall] = field(default_factory=dict)
    #: Key of the tool call whose span is current on the task, if one is.
    current_tool: str | None = None


class TelemetryHook(HookProvider):
    """Hook provider for OpenTelemetry tracing and metrics.

    Spans (GenAI conventions, the default):

    - ``invoke_agent <agent name>`` — the run, INTERNAL, root unless a caller's
      trace context is current; ``gen_ai.usage.*`` totals on completion.
    - ``locus.iteration`` — one ReAct iteration (the SDK's own span; the
      index is the ``locus.iteration`` attribute).
    - ``chat <model>`` — one model call, CLIENT, with ``gen_ai.provider.name``,
      ``gen_ai.request.model``, usage, finish reasons and the response id.
      A hook that asks for a retry ends the span and the retried call gets
      one of its own, so every attempt is one span.
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

        # One record per run in flight, keyed by run id. Each span is attached
        # to OTel's current context while open, so the spans the agent creates
        # inside a run are parented to it — without that, exporters see every
        # span as a new trace root (issue #244). ``_loose`` collects spans of
        # hook events that arrive outside any run (a hook driven by hand).
        self._runs: dict[str, _Run] = {}
        self._loose: _Run | None = None

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
    # Run records
    # ------------------------------------------------------------------

    @staticmethod
    def _run_id_of(source: Any) -> str | None:
        """The run id a hook event's ``run`` or an agent state names, if any."""
        run_id = getattr(source, "run_id", None) if source is not None else None
        return run_id if isinstance(run_id, str) and run_id else None

    def _run_for(self, run_id: str | None) -> _Run | None:
        """The record ``run_id`` names, else the run current on this task.

        An event that names no run at all (one built by hand) falls back to
        the newest run in flight.
        """
        if run_id is not None and run_id in self._runs:
            return self._runs[run_id]
        current = _CURRENT_RUN.get()
        if current is not None and current in self._runs:
            return self._runs[current]
        if run_id is None and self._runs:
            return next(reversed(self._runs.values()))
        return None

    def _run_or_loose(self, run_id: str | None) -> _Run:
        """As :meth:`_run_for`, with the record for out-of-run events as the last resort."""
        run = self._run_for(run_id)
        if run is not None:
            return run
        if self._loose is None:
            self._loose = _Run(
                run_id="",
                span=None,
                token=None,
                started=time.perf_counter(),
                agent_name=None,
                unnamed=False,
                previous=None,
            )
        return self._loose

    def _peek(self) -> _Run | None:
        """The run a hand-built event would land in, without creating one."""
        return self._run_for(None) or self._loose

    # Views of the current run's open spans, for tests and debugging.
    @property
    def _invocation_span(self) -> Span | None:
        run = self._run_for(None)
        return run.span if run is not None else None

    @property
    def _iteration_spans(self) -> dict[int, Span]:
        run = self._peek()
        return run.iterations if run is not None else {}

    @property
    def _tool_spans(self) -> dict[str, tuple[Span, float, str]]:
        run = self._peek()
        if run is None:
            return {}
        return {key: (call.span, call.started, call.name) for key, call in run.tools.items()}

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

    def _agent_label(self, run: _Run | None, state: AgentState | None = None) -> str | None:
        if self._agent_name:
            return self._agent_name
        if run is not None and run.run_agent_name:
            return run.run_agent_name
        agent_id = getattr(state, "agent_id", None) if state is not None else None
        if isinstance(agent_id, str) and agent_id:
            return agent_id
        return run.agent_name if run is not None else None

    def _absorb_run(self, run: _Run, info: Any) -> None:
        """Pick the conversation id and agent name off a hook event's ``run``."""
        if info is None:
            return
        thread_id = getattr(info, "thread_id", None)
        if run.conversation_id is None and isinstance(thread_id, str) and thread_id:
            run.conversation_id = thread_id
            if self._genai and run.span is not None:
                run.span.set_attribute(sc.GEN_AI_CONVERSATION_ID, thread_id)
        agent_name = getattr(info, "agent_name", None)
        if (
            self._agent_name is None
            and run.run_agent_name is None
            and isinstance(agent_name, str)
            and agent_name
        ):
            run.run_agent_name = agent_name
            if self._genai and run.span is not None and run.unnamed:
                run.span.update_name(f"{sc.OP_INVOKE_AGENT} {agent_name}")
                run.span.set_attribute(sc.GEN_AI_AGENT_NAME, agent_name)
                run.unnamed = False

    def _close_model_span(
        self, run: _Run, error: str | None = None, error_type: str | None = None
    ) -> None:
        """End the open ``chat`` span, as a failure when ``error`` is given.

        A call that ended in an error is still an inference call: its duration
        is recorded with ``error.type``, as the conventions ask, and it counts
        towards the run's ``inference_calls``.
        """
        call = run.model
        if call is None:
            return
        run.model = None
        span = call.span
        if error is not None:
            kind = error_type or sc.ERROR_TYPE_OTHER
            span.set_status(Status(StatusCode.ERROR, error))
            span.set_attribute(sc.ERROR_TYPE, kind)
            run.inference_calls += 1
            self._inference_duration.record(
                time.perf_counter() - call.started,
                self._inference_attrs() | {sc.ERROR_TYPE: kind},
            )
        span.end()
        if call.token is not None:
            otel_context.detach(call.token)

    def _close_open_tools(self, run: _Run, error_type: str | None) -> None:
        """End tool spans a run that ended left open, newest first."""
        for key in list(reversed(run.tools)):
            call = run.tools.pop(key)
            call.span.set_status(Status(StatusCode.ERROR, "tool call did not complete"))
            if self._genai:
                call.span.set_attribute(sc.ERROR_TYPE, error_type or sc.ERROR_TYPE_OTHER)
            else:
                call.span.set_attribute("locus.error", "tool call did not complete")
            call.span.end()
            self._release_tool(run, call, key)

    def _close_open_iterations(self, run: _Run, error_type: str | None) -> None:
        """End iteration spans a failed run left open, newest first."""
        for iteration in sorted(run.iterations, reverse=True):
            span = run.iterations.pop(iteration)
            span.set_status(Status(StatusCode.ERROR, "iteration did not complete"))
            span.set_attribute(sc.ERROR_TYPE, error_type or sc.ERROR_TYPE_OTHER)
            span.end()
            token = run.iteration_tokens.pop(iteration, None)
            if token is not None:
                otel_context.detach(token)
        run.current_iteration = None

    @staticmethod
    def _last_error_type(state: Any) -> str | None:
        """``error.type`` for the last error an agent state recorded, if any."""
        errors = getattr(state, "errors", ()) or ()
        return sc.error_type(errors[-1]) if errors else None

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------

    async def on_before_invocation(
        self,
        prompt: str,
        state: AgentState,
    ) -> AgentState:
        """Start the run's root span and attach it as the current OTel context.

        The span is parented to the tool call in flight when the run starts
        inside one (a sub-agent run from a tool), else to whatever context is
        current, so a caller's ``traceparent`` (attached by ``trace_scope``)
        is inherited.

        Args:
            prompt: User prompt
            state: Agent state

        Returns:
            Unchanged state
        """
        run_id = self._run_id_of(state) or uuid4().hex
        started = time.perf_counter()
        agent_name = self._agent_label(None, state)
        parent = current_tool_span()
        context = trace.set_span_in_context(parent) if parent is not None else None
        if self._genai:
            span = self._tracer.start_span(
                f"{sc.OP_INVOKE_AGENT} {agent_name}" if agent_name else sc.OP_INVOKE_AGENT,
                context=context,
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
            span = self._tracer.start_span(
                "agent.invocation",
                context=context,
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
        token = otel_context.attach(trace.set_span_in_context(span))
        run = _Run(
            run_id=run_id,
            span=span,
            token=token,
            started=started,
            agent_name=agent_name,
            unnamed=agent_name is None,
            previous=_CURRENT_RUN.get(),
        )
        run.context_token = _CURRENT_RUN.set(run_id)
        self._runs[run_id] = run
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
        run = self._run_for(self._run_id_of(state))
        if run is None or run.span is None:
            return
        errors = getattr(state, "errors", ()) or ()
        error_kind = sc.error_type(errors[-1] if errors else None) if not success else None
        # A run that raised may leave a model call, tool calls or an iteration
        # open; close them (innermost first, so context tokens detach in order).
        self._close_model_span(run, "model call did not complete", error_kind)
        self._close_open_tools(run, error_kind)
        self._close_open_iterations(run, error_kind)

        span = run.span
        elapsed_s = time.perf_counter() - run.started
        duration_ms = (state.updated_at - state.started_at).total_seconds() * 1000

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
                        "locus.inference_calls": run.inference_calls,
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
        self._runs.pop(run.run_id, None)

        # Detach the invocation context — pair to the attach in
        # `on_before_invocation`. Detaching restores whatever
        # context was active before locus took over.
        if run.token is not None:
            otel_context.detach(run.token)
        try:
            _CURRENT_RUN.reset(run.context_token)
        except (ValueError, RuntimeError):  # ended on another task than it started
            _CURRENT_RUN.set(run.previous)

        if self._genai:
            agent_name = self._agent_label(run, state)
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
            self._invoke_inference_calls.record(run.inference_calls, count_attrs)
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

    def _start_model_span(
        self, run: _Run, *, messages: int | None, tools: int | None, attempt: int
    ) -> None:
        """Open a ``chat`` span and make it current.

        An instrumented HTTP client inside the provider then files its request
        under it.
        """
        model = self._request_model
        attrs = self._inference_attrs()
        attrs.update(
            _attrs(
                **{
                    sc.GEN_AI_CONVERSATION_ID: run.conversation_id,
                    "locus.messages": messages,
                    "locus.tools": tools,
                    "locus.retry_attempt": attempt if attempt > 1 else None,
                }
            )
        )
        span = self._tracer.start_span(
            f"{sc.OP_CHAT} {model}" if model else sc.OP_CHAT,
            kind=SpanKind.CLIENT,
            attributes=attrs,
        )
        run.model = _ModelCall(
            span=span,
            token=otel_context.attach(trace.set_span_in_context(span)),
            started=time.perf_counter(),
            attempt=attempt,
        )

    async def on_before_model_call(self, event: BeforeModelCallEvent) -> None:
        """Start a ``chat`` span for the model call (GenAI conventions only)."""
        if not self._genai:
            return
        info = getattr(event, "run", None)
        run = self._run_or_loose(self._run_id_of(info))
        self._close_model_span(run, "model call did not complete")
        self._absorb_run(run, info)
        self._start_model_span(
            run, messages=len(event.messages), tools=len(event.tools or []), attempt=1
        )

    async def on_after_model_call(self, event: AfterModelCallEvent) -> None:
        """End the ``chat`` span with usage, finish reason and response ids.

        When a hook asked for a retry (``event.retry``), the loop calls the
        model again without the before-model hooks, so the retried attempt's
        ``chat`` span is opened here: every attempt is one span and counts
        once.
        """
        if not self._genai:
            return
        run = self._run_for(self._run_id_of(getattr(event, "run", None)))
        if run is None or run.model is None:
            return
        call = run.model
        span = call.span
        elapsed_s = time.perf_counter() - call.started
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
        retry = bool(getattr(event, "retry", False))
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
                    "locus.retry_requested": True if retry else None,
                }
            )
        )
        span.set_status(Status(StatusCode.OK))
        self._close_model_span(run)
        run.inference_calls += 1

        metric_attrs = self._inference_attrs(
            response_model if isinstance(response_model, str) else None
        )
        self._inference_duration.record(elapsed_s, metric_attrs)
        # The conventions count only known, strictly positive contributions,
        # each under its modality — ``unknown`` when the provider gives no
        # per-modality breakdown.
        for modality, count in _token_points(input_tokens, usage, _INPUT_DETAIL_KEYS):
            self._input_tokens.add(count, metric_attrs | {sc.GEN_AI_TOKEN_MODALITY: modality})
        for modality, count in _token_points(output_tokens, usage, _OUTPUT_DETAIL_KEYS):
            self._output_tokens.add(count, metric_attrs | {sc.GEN_AI_TOKEN_MODALITY: modality})

        if retry:
            feedback = getattr(event, "retry_feedback", None)
            messages = len(event.messages) + (1 if isinstance(feedback, str) and feedback else 0)
            self._start_model_span(run, messages=messages, tools=None, attempt=call.attempt + 1)

    # ------------------------------------------------------------------
    # Tool calls
    # ------------------------------------------------------------------

    async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        """Start tool call span.

        The span is made current while the call runs, so a run started inside
        the tool (a sub-agent) and an instrumented client the tool uses file
        under it. When several calls of one batch are open at once — the loop
        runs every call's before-tool hooks before any call starts — no single
        span can be current; the tool context (:func:`current_tool_span`)
        still names each call's span for what runs inside it.

        Args:
            event: Write-protected event carrying ``tool_name`` and
                ``arguments``. The hook only inspects them.
        """
        tool_name = event.tool_name
        call_id = getattr(event, "tool_call_id", "") or ""
        info = getattr(event, "run", None)
        run = self._run_or_loose(self._run_id_of(info))
        self._absorb_run(run, info)

        if self._genai:
            span_attrs = _attrs(
                **{
                    sc.GEN_AI_OPERATION_NAME: sc.OP_EXECUTE_TOOL,
                    sc.GEN_AI_TOOL_NAME: tool_name,
                    sc.GEN_AI_TOOL_CALL_ID: call_id or None,
                    sc.GEN_AI_TOOL_TYPE: sc.TOOL_TYPE_FUNCTION,
                    sc.GEN_AI_TOOL_DESCRIPTION: self._tool_descriptions.get(tool_name),
                    sc.GEN_AI_AGENT_NAME: self._agent_label(run),
                    sc.GEN_AI_CONVERSATION_ID: run.conversation_id,
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

        key = call_id or tool_name
        call = _ToolCall(span=span, started=time.perf_counter(), name=tool_name)
        if not run.tools:
            call.token = otel_context.attach(trace.set_span_in_context(span))
            run.current_tool = key
        elif run.current_tool is not None:
            # A second call before the first ran: a batch. Neither span can
            # be current for the tools' bodies, so neither is.
            current = run.tools.get(run.current_tool)
            if current is not None and current.token is not None:
                otel_context.detach(current.token)
                current.token = None
            run.current_tool = None
        run.tools[key] = call
        if call_id:
            _TOOL_CALL_SPANS[call_id] = span

    @staticmethod
    def _release_tool(run: _Run, call: _ToolCall, key: str) -> None:
        """Drop a finished call from the current context and the in-flight index."""
        if call.token is not None:
            otel_context.detach(call.token)
            call.token = None
        if run.current_tool == key:
            run.current_tool = None
        for call_id, active in list(_TOOL_CALL_SPANS.items()):
            if active is call.span:
                del _TOOL_CALL_SPANS[call_id]

    def _pop_tool_span(self, run: _Run, call_id: str, tool_name: str) -> _ToolCall | None:
        """The open call for a finished one: by id, else the oldest for the tool in this run."""
        key = call_id if call_id and call_id in run.tools else None
        if key is None:
            key = next((k for k, call in run.tools.items() if call.name == tool_name), None)
        if key is None:
            return None
        call = run.tools.pop(key)
        self._release_tool(run, call, key)
        return call

    async def on_after_tool_call(self, event: AfterToolCallEvent) -> None:
        """End tool call span.

        Args:
            event: Write-protected event carrying ``tool_name``,
                ``result``, and ``error``.
        """
        tool_name = event.tool_name
        error = event.error
        result = event.result
        run = self._run_for(self._run_id_of(getattr(event, "run", None))) or self._loose
        if run is None:
            return
        call = self._pop_tool_span(run, getattr(event, "tool_call_id", "") or "", tool_name)
        if call is None:
            return
        span = call.span
        elapsed_s = time.perf_counter() - call.started
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
                        sc.GEN_AI_AGENT_NAME: self._agent_label(run),
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
        run = self._run_or_loose(self._run_id_of(state))
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
        run.iterations[iteration] = span
        run.iteration_tokens[iteration] = otel_context.attach(trace.set_span_in_context(span))
        run.current_iteration = iteration

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
        run = self._run_for(self._run_id_of(state)) or self._loose
        if run is None or iteration not in run.iterations:
            return
        # A model call that raised never reached ``on_after_model_call``;
        # close it first so context tokens detach in attach order.
        self._close_model_span(run, "model call did not complete", self._last_error_type(state))
        span = run.iterations.pop(iteration)
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
        token = run.iteration_tokens.pop(iteration, None)
        if token is not None:
            otel_context.detach(token)
        if run.current_iteration == iteration:
            run.current_iteration = None


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
