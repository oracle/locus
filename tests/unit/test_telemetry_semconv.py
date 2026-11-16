# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``TelemetryHook`` under the GenAI semantic conventions.

Span names, kinds and ``gen_ai.*`` attributes per the convention switch, the
token and duration metrics, propagation into tool spans, the caller's trace
being inherited, and the validator that checks a recorded trace.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from locus.core.events import RunInfo
from locus.hooks.builtin import telemetry as telemetry_mod
from locus.hooks.builtin.telemetry import OTEL_AVAILABLE, TelemetryHook
from locus.hooks.provider import (
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
)
from locus.observability import semconv as sc


def _otel_sdk_available() -> bool:
    try:
        import opentelemetry.sdk.metrics  # noqa: F401
        import opentelemetry.sdk.trace  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not OTEL_AVAILABLE or not _otel_sdk_available(), reason="OpenTelemetry API + SDK required"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def exporter() -> Any:
    """An in-memory exporter on the (once-installed) SDK tracer provider."""
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    if not isinstance(otel_trace.get_tracer_provider(), TracerProvider):
        otel_trace.set_tracer_provider(TracerProvider())
    exp = InMemorySpanExporter()
    otel_trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exp))
    return exp


@pytest.fixture
def metric_reader(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A private SDK meter provider the hook's ``get_meter`` is pointed at."""
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    monkeypatch.setattr(
        telemetry_mod.metrics, "get_meter", lambda name, *a, **kw: provider.get_meter(name)
    )
    return reader


def _state(**overrides: Any) -> SimpleNamespace:
    now = datetime.now(UTC)
    base: dict[str, Any] = {
        "run_id": "run-1",
        "agent_id": "support",
        "max_iterations": 5,
        "iteration": 1,
        "confidence": 0.9,
        "tool_executions": [object()],
        "errors": (),
        "messages": [],
        "started_at": now,
        "updated_at": now + timedelta(seconds=1),
        "prompt_tokens_used": 30,
        "completion_tokens_used": 12,
        "cache_read_tokens_used": 0,
        "cached_tokens_used": 0,
        "cache_creation_tokens_used": 0,
        "cache_write_tokens_used": 0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _run(thread_id: str | None = "thread-7", agent_name: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(info=RunInfo(run_id="run-1", thread_id=thread_id, agent_name=agent_name))


def _spans(exporter: Any, scope: str) -> list[Any]:
    return [s for s in exporter.get_finished_spans() if s.instrumentation_scope.name == scope]


def _by_name(exporter: Any, scope: str) -> dict[str, Any]:
    return {s.name: s for s in _spans(exporter, scope)}


def _points(reader: Any) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    data = reader.get_metrics_data()
    for rm in data.resource_metrics if data else []:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                out.setdefault(metric.name, []).extend(metric.data.data_points)
    return out


async def _drive(hook: TelemetryHook, *, run: Any = None, error: str | None = None) -> None:
    """One invocation: one iteration, one model call, one tool call."""
    from locus.testing import text

    run = run or _run()
    await hook.on_before_invocation("Look up CS-1042", _state())
    await hook.on_iteration_start(0, _state())
    await hook.on_before_model_call(BeforeModelCallEvent(messages=[1, 2], tools=[{}], run=run))
    await hook.on_after_model_call(
        AfterModelCallEvent(response=text("ok", stop_reason="tool_use"), messages=[], run=run)
    )
    await hook.on_before_tool_call(
        BeforeToolCallEvent(
            tool_name="lookup_order",
            tool_call_id="call-1",
            arguments={"order_id": "CS-1042"},
            run=run,
        )
    )
    await hook.on_after_tool_call(
        AfterToolCallEvent(
            tool_name="lookup_order",
            result="delivered",
            error=error,
            tool_call_id="call-1",
            run=run,
        )
    )
    await hook.on_iteration_end(0, _state())
    await hook.on_after_invocation(_state(), success=True)


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------


class TestSwitch:
    def test_default_is_the_genai_conventions(self) -> None:
        assert TelemetryHook().semconv == sc.SEMCONV_GENAI == "genai/1.45"

    @pytest.mark.parametrize("value", ["genai", "genai/latest", "GenAI/1.45"])
    def test_aliases_resolve_to_genai(self, value: str) -> None:
        assert TelemetryHook(semconv=value).semconv == sc.SEMCONV_GENAI

    @pytest.mark.parametrize("value", ["legacy", "locus-legacy"])
    def test_legacy_aliases(self, value: str) -> None:
        assert TelemetryHook(semconv=value).semconv == sc.SEMCONV_LEGACY

    def test_unknown_convention_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"genai/1\.45"):
            TelemetryHook(semconv="genai/9.99")


# ---------------------------------------------------------------------------
# Span names and attributes
# ---------------------------------------------------------------------------


class TestGenAISpans:
    async def test_names_kinds_attributes_and_tree(self, exporter: Any) -> None:
        from opentelemetry.trace import SpanKind

        scope = "locus.semconv.test-tree"
        hook = TelemetryHook(tracer_name=scope, model="oci:openai.gpt-4.1")
        await _drive(hook)

        spans = _by_name(exporter, scope)
        assert set(spans) == {
            "invoke_agent support",
            sc.ITERATION_SPAN_NAME,
            "chat openai.gpt-4.1",
            "execute_tool lookup_order",
        }
        root = spans["invoke_agent support"]
        iteration = spans[sc.ITERATION_SPAN_NAME]
        chat = spans["chat openai.gpt-4.1"]
        tool = spans["execute_tool lookup_order"]

        assert root.parent is None
        assert iteration.parent.span_id == root.context.span_id
        assert chat.parent.span_id == iteration.context.span_id
        assert tool.parent.span_id == iteration.context.span_id
        assert len({s.context.trace_id for s in spans.values()}) == 1

        assert root.kind is SpanKind.INTERNAL
        assert chat.kind is SpanKind.CLIENT
        assert tool.kind is SpanKind.INTERNAL

        assert root.attributes[sc.GEN_AI_OPERATION_NAME] == "invoke_agent"
        assert root.attributes[sc.GEN_AI_AGENT_NAME] == "support"
        assert root.attributes[sc.GEN_AI_AGENT_ID] == "support"
        assert root.attributes[sc.GEN_AI_REQUEST_MODEL] == "openai.gpt-4.1"
        assert root.attributes[sc.GEN_AI_CONVERSATION_ID] == "thread-7"
        assert root.attributes[sc.GEN_AI_USAGE_INPUT_TOKENS] == 30
        assert root.attributes[sc.GEN_AI_USAGE_OUTPUT_TOKENS] == 12
        assert root.attributes["locus.inference_calls"] == 1
        assert root.status.status_code.name == "OK"

        assert chat.attributes[sc.GEN_AI_OPERATION_NAME] == "chat"
        assert chat.attributes[sc.GEN_AI_PROVIDER_NAME] == "oci.generative_ai"
        assert chat.attributes[sc.GEN_AI_REQUEST_MODEL] == "openai.gpt-4.1"
        assert chat.attributes[sc.GEN_AI_CONVERSATION_ID] == "thread-7"
        assert chat.attributes[sc.GEN_AI_USAGE_INPUT_TOKENS] == 10
        assert chat.attributes[sc.GEN_AI_USAGE_OUTPUT_TOKENS] == 20
        assert list(chat.attributes[sc.GEN_AI_RESPONSE_FINISH_REASONS]) == ["tool_use"]

        assert tool.attributes[sc.GEN_AI_OPERATION_NAME] == "execute_tool"
        assert tool.attributes[sc.GEN_AI_TOOL_NAME] == "lookup_order"
        assert tool.attributes[sc.GEN_AI_TOOL_CALL_ID] == "call-1"
        assert tool.attributes[sc.GEN_AI_TOOL_TYPE] == "function"
        assert tool.attributes[sc.GEN_AI_AGENT_NAME] == "support"
        assert sc.GEN_AI_TOOL_CALL_ARGUMENTS not in tool.attributes
        assert sc.GEN_AI_TOOL_CALL_RESULT not in tool.attributes

        assert sc.validate_spans(spans.values()) == []

    async def test_arguments_and_results_are_opt_in(self, exporter: Any) -> None:
        scope = "locus.semconv.test-optin"
        hook = TelemetryHook(
            tracer_name=scope, model="openai:gpt-4.1", record_arguments=True, record_results=True
        )
        await _drive(hook)
        tool = _by_name(exporter, scope)["execute_tool lookup_order"]
        assert tool.attributes[sc.GEN_AI_TOOL_CALL_ARGUMENTS] == '{"order_id": "CS-1042"}'
        assert tool.attributes[sc.GEN_AI_TOOL_CALL_RESULT] == '"delivered"'

    async def test_tool_error_sets_status_and_error_type(self, exporter: Any) -> None:
        scope = "locus.semconv.test-error"
        hook = TelemetryHook(tracer_name=scope, model="openai:gpt-4.1")
        await _drive(hook, error="ValueError: unknown order")
        tool = _by_name(exporter, scope)["execute_tool lookup_order"]
        assert tool.status.status_code.name == "ERROR"
        assert tool.attributes[sc.ERROR_TYPE] == "ValueError"
        assert sc.validate_spans(_spans(exporter, scope)) == []

    async def test_unnamed_agent_takes_the_name_the_run_carries(self, exporter: Any) -> None:
        scope = "locus.semconv.test-rename"
        hook = TelemetryHook(tracer_name=scope, model="openai:gpt-4.1")
        await hook.on_before_invocation("hi", _state(agent_id=None))
        await hook.on_iteration_start(0, _state(agent_id=None))
        run = _run(agent_name="billing")
        await hook.on_before_model_call(BeforeModelCallEvent(messages=[], tools=None, run=run))
        from locus.testing import text

        await hook.on_after_model_call(
            AfterModelCallEvent(response=text("x"), messages=[], run=run)
        )
        await hook.on_iteration_end(0, _state(agent_id=None))
        await hook.on_after_invocation(_state(agent_id=None), success=True)
        spans = _by_name(exporter, scope)
        assert "invoke_agent billing" in spans
        assert spans["invoke_agent billing"].attributes[sc.GEN_AI_AGENT_NAME] == "billing"

    async def test_failed_invocation_records_error_type(self, exporter: Any) -> None:
        scope = "locus.semconv.test-failed"
        hook = TelemetryHook(tracer_name=scope, model="openai:gpt-4.1")
        await hook.on_before_invocation("hi", _state())
        await hook.on_after_invocation(_state(errors=("TimeoutError: model",)), success=False)
        root = _by_name(exporter, scope)["invoke_agent support"]
        assert root.status.status_code.name == "ERROR"
        assert root.attributes[sc.ERROR_TYPE] == "TimeoutError"

    async def test_model_call_that_never_returns_is_closed_as_an_error(self, exporter: Any) -> None:
        from opentelemetry import trace as otel_trace

        scope = "locus.semconv.test-dangling"
        hook = TelemetryHook(tracer_name=scope, model="openai:gpt-4.1")
        await hook.on_before_invocation("hi", _state())
        await hook.on_iteration_start(0, _state())
        await hook.on_before_model_call(BeforeModelCallEvent(messages=[], tools=None, run=_run()))
        # The provider raised: no on_after_model_call. The iteration still ends.
        await hook.on_iteration_end(0, _state())
        await hook.on_after_invocation(_state(), success=False)
        chat = _by_name(exporter, scope)["chat gpt-4.1"]
        assert chat.status.status_code.name == "ERROR"
        assert chat.attributes[sc.ERROR_TYPE] == "_OTHER"
        assert otel_trace.get_current_span().get_span_context().span_id == 0

    async def test_open_iteration_is_closed_with_the_run(self, exporter: Any) -> None:
        from opentelemetry import trace as otel_trace

        scope = "locus.semconv.test-open-iteration"
        hook = TelemetryHook(tracer_name=scope, model="openai:gpt-4.1")
        await hook.on_before_invocation("hi", _state())
        await hook.on_iteration_start(0, _state())
        await hook.on_after_invocation(_state(), success=False)
        names = {s.name for s in _spans(exporter, scope)}
        assert names == {"invoke_agent support", sc.ITERATION_SPAN_NAME}
        assert otel_trace.get_current_span().get_span_context().span_id == 0


class TestLegacySpans:
    async def test_legacy_switch_keeps_the_old_names(self, exporter: Any) -> None:
        scope = "locus.semconv.test-legacy"
        hook = TelemetryHook(tracer_name=scope, semconv="locus-legacy")
        await _drive(hook)
        spans = _by_name(exporter, scope)
        assert set(spans) == {"agent.invocation", "agent.iteration.0", "tool.lookup_order"}
        assert spans["tool.lookup_order"].attributes["locus.tool_name"] == "lookup_order"
        assert spans["agent.invocation"].attributes["service.name"] == "locus-agent"
        assert not any(k.startswith("gen_ai.") for s in spans.values() for k in s.attributes)


# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------


class _FakeConfig:
    def __init__(self, model: str, base_url: str | None = None) -> None:
        self.model = model
        self.base_url = base_url


class OpenAIModel:  # the hook looks at class names, not the real classes
    def __init__(self, model: str, base_url: str | None = None) -> None:
        self.config = _FakeConfig(model, base_url)


class ScriptedThing:
    pass


class TestModelIdentity:
    @pytest.mark.parametrize(
        ("model", "provider", "name"),
        [
            ("oci:openai.gpt-4.1", "oci.generative_ai", "openai.gpt-4.1"),
            ("openai:gpt-4.1", "openai", "gpt-4.1"),
            ("anthropic:claude-sonnet-4-5", "anthropic", "claude-sonnet-4-5"),
            ("xai:grok-3", "x_ai", "grok-3"),
            ("gemini:gemini-2.5-flash", "gcp.gemini", "gemini-2.5-flash"),
            ("vllm:qwen2.5", "vllm", "qwen2.5"),
        ],
    )
    def test_model_strings(self, model: str, provider: str, name: str) -> None:
        hook = TelemetryHook(model=model)
        assert hook.provider_name == provider
        assert hook.request_model == name

    def test_openai_object_and_compatible_endpoints(self) -> None:
        assert sc.provider_name(OpenAIModel("gpt-4.1")) == "openai"
        assert sc.provider_name(OpenAIModel("llama", "https://api.groq.com/openai/v1")) == "groq"
        assert sc.provider_name(OpenAIModel("qwen", "http://vllm.internal:8000/v1")) == "openai"
        # Local servers are told apart by port, not host: 8000 is vLLM's default,
        # 11434 Ollama's, and an unlisted port is plain OpenAI-compatible.
        assert sc.provider_name(OpenAIModel("qwen", "http://localhost:8000/v1")) == "vllm"
        assert sc.provider_name(OpenAIModel("llama3", "http://localhost:11434/v1")) == "ollama"
        assert sc.provider_name(OpenAIModel("qwen", "http://localhost:9999/v1")) == "openai"
        assert sc.server_address(OpenAIModel("qwen", "http://vllm.internal:8000/v1")) == (
            "vllm.internal",
            8000,
        )

    def test_bind_agent_fills_model_and_tool_descriptions(self) -> None:
        hook = TelemetryHook()
        tools = [SimpleNamespace(name="lookup_order", description="Look up an order.")]
        hook.bind_agent(model=OpenAIModel("gpt-4.1", "http://vllm.internal:8000/v1"), tools=tools)
        assert hook.request_model == "gpt-4.1"
        assert hook.provider_name == "openai"
        assert hook._server_address == "vllm.internal"
        assert hook._tool_descriptions == {"lookup_order": "Look up an order."}

    def test_explicit_values_win_over_bind(self) -> None:
        hook = TelemetryHook(model="oci:openai.gpt-4.1", provider_name="custom")
        hook.bind_agent(model=OpenAIModel("other"), tools=[])
        assert hook.request_model == "openai.gpt-4.1"
        assert hook.provider_name == "custom"

    def test_unknown_model_class_falls_back_to_its_name(self) -> None:
        hook = TelemetryHook()
        hook.bind_agent(model=ScriptedThing(), tools=[])
        assert hook.provider_name == "scripted_thing"
        assert hook.request_model is None

    def test_error_type_values(self) -> None:
        assert sc.error_type("ValueError: bad") == "ValueError"
        assert sc.error_type("httpx.ReadTimeout: slow") == "ReadTimeout"
        assert sc.error_type(KeyError("k")) == "KeyError"
        assert sc.error_type("boom") == "_OTHER"
        assert sc.error_type(None) == "_OTHER"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestGenAIMetrics:
    async def test_token_counters_and_duration_histograms(
        self, exporter: Any, metric_reader: Any
    ) -> None:
        hook = TelemetryHook(tracer_name="locus.semconv.test-metrics", model="oci:openai.gpt-4.1")
        await _drive(hook)
        await _drive(hook)
        points = _points(metric_reader)

        inputs = points[sc.METRIC_INPUT_TOKENS]
        assert sum(p.value for p in inputs) == 20  # 10 per call, two calls
        assert all(p.attributes[sc.GEN_AI_TOKEN_MODALITY] == "unknown" for p in inputs)
        assert all(p.attributes[sc.GEN_AI_OPERATION_NAME] == "chat" for p in inputs)
        assert all(p.attributes[sc.GEN_AI_PROVIDER_NAME] == "oci.generative_ai" for p in inputs)
        assert all(p.attributes[sc.GEN_AI_REQUEST_MODEL] == "openai.gpt-4.1" for p in inputs)
        assert sum(p.value for p in points[sc.METRIC_OUTPUT_TOKENS]) == 40

        assert sum(p.count for p in points[sc.METRIC_INFERENCE_DURATION]) == 2
        assert sum(p.count for p in points[sc.METRIC_INVOKE_AGENT_DURATION]) == 2
        assert sum(p.sum for p in points[sc.METRIC_INVOKE_AGENT_INFERENCE_CALLS]) == 2
        assert sum(p.sum for p in points[sc.METRIC_INVOKE_AGENT_TOOL_CALLS]) == 2
        tool = points[sc.METRIC_EXECUTE_TOOL_DURATION]
        assert sum(p.count for p in tool) == 2
        assert all(p.attributes[sc.GEN_AI_TOOL_NAME] == "lookup_order" for p in tool)
        assert all(p.attributes[sc.GEN_AI_AGENT_NAME] == "support" for p in tool)
        assert not any(name.startswith("locus.") for name in points)

    async def test_legacy_metrics_keep_their_names(self, exporter: Any, metric_reader: Any) -> None:
        hook = TelemetryHook(
            tracer_name="locus.semconv.test-legacy-metrics", semconv="locus-legacy"
        )
        await _drive(hook)
        names = set(_points(metric_reader))
        assert {"locus.invocations", "locus.tool_calls", "locus.tool_call.duration"} <= names
        assert not any(name.startswith("gen_ai.") for name in names)


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------


class TestPropagation:
    async def test_run_inherits_the_callers_span(self, exporter: Any) -> None:
        from opentelemetry import trace as otel_trace

        scope = "locus.semconv.test-inherit"
        hook = TelemetryHook(tracer_name=scope, model="openai:gpt-4.1")
        tracer = otel_trace.get_tracer("locus.semconv.test-caller")
        with tracer.start_as_current_span("http request") as caller:
            await _drive(hook)
        root = _by_name(exporter, scope)["invoke_agent support"]
        assert root.parent.span_id == caller.get_span_context().span_id
        assert root.context.trace_id == caller.get_span_context().trace_id

    async def test_run_inherits_a_traceparent_header(self, exporter: Any) -> None:
        from locus.observability.propagation import trace_scope

        scope = "locus.semconv.test-header"
        hook = TelemetryHook(tracer_name=scope, model="openai:gpt-4.1")
        header = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        with trace_scope({"Traceparent": header}):
            await _drive(hook)
        root = _by_name(exporter, scope)["invoke_agent support"]
        assert format(root.context.trace_id, "032x") == "0af7651916cd43dd8448eb211c80319c"
        assert format(root.parent.span_id, "016x") == "b7ad6b7169203331"
        # The run is still a valid trace even though its root has a remote parent.
        assert sc.validate_spans(_spans(exporter, scope)) == []

    async def test_tool_span_is_what_leaves_the_process(self, exporter: Any) -> None:
        from locus.observability.propagation import inject_trace_headers

        scope = "locus.semconv.test-outbound"
        hook = TelemetryHook(tracer_name=scope, model="openai:gpt-4.1")
        await hook.on_before_invocation("hi", _state())
        await hook.on_before_tool_call(
            BeforeToolCallEvent(tool_name="remote", tool_call_id="c-9", arguments={}, run=_run())
        )
        span = telemetry_mod._TOOL_CALL_SPANS["c-9"]
        headers = inject_trace_headers({"Accept": "application/json"}, span=span)
        assert headers["Accept"] == "application/json"
        assert headers["traceparent"].split("-")[2] == format(
            span.get_span_context().span_id, "016x"
        )
        await hook.on_after_tool_call(
            AfterToolCallEvent(tool_name="remote", result="", error=None, tool_call_id="c-9")
        )
        await hook.on_after_invocation(_state(), success=True)
        assert "c-9" not in telemetry_mod._TOOL_CALL_SPANS


# ---------------------------------------------------------------------------
# A whole agent run, with the agent lending the hook its model and tools
# ---------------------------------------------------------------------------


class TestAgentRun:
    async def test_real_run_validates(self, exporter: Any) -> None:
        from locus.agent import Agent
        from locus.testing import ScriptedModel, text, tool_call
        from locus.tools import tool

        @tool
        def lookup_order(order_id: str) -> str:
            """Look up an order's shipping status."""
            return f"{order_id}: delivered"

        scope = "locus.semconv.test-agent"
        hook = TelemetryHook(tracer_name=scope)
        agent = Agent(
            model=ScriptedModel([tool_call("lookup_order", order_id="CS-1042"), text("done")]),
            tools=[lookup_order],
            hooks=[hook],
            reflexion=False,
            grounding=False,
        )
        await agent.arun("Look up CS-1042", thread_id="ticket-1042")

        spans = _spans(exporter, scope)
        assert sc.validate_spans(spans) == []
        names = {s.name for s in spans}
        assert "invoke_agent" in names
        assert "execute_tool lookup_order" in names
        assert "chat" in names  # ScriptedModel has no model id
        by_name = {s.name: s for s in spans}
        tool_span = by_name["execute_tool lookup_order"]
        assert tool_span.attributes[sc.GEN_AI_TOOL_DESCRIPTION].startswith("Look up an order")
        assert tool_span.attributes[sc.GEN_AI_CONVERSATION_ID] == "ticket-1042"
        chat = by_name["chat"]
        assert chat.attributes[sc.GEN_AI_PROVIDER_NAME] == "scripted_model"
        assert chat.attributes[sc.GEN_AI_USAGE_INPUT_TOKENS] == 10
        root = by_name["invoke_agent"]
        assert root.attributes[sc.GEN_AI_USAGE_INPUT_TOKENS] == 20  # two model calls
        assert root.attributes[sc.GEN_AI_CONVERSATION_ID] == "ticket-1042"


# ---------------------------------------------------------------------------
# The validator on hand-made traces
# ---------------------------------------------------------------------------


def _rec(name: str, op: str | None, **attrs: Any) -> sc.SpanRecord:
    attributes = {sc.GEN_AI_OPERATION_NAME: op} if op else {}
    attributes.update(attrs)
    kind = {"chat": "CLIENT"}.get(op or "", "INTERNAL")
    return sc.SpanRecord(
        name=name,
        kind=kind,
        trace_id="t1",
        span_id=name,
        parent_span_id=None if op == "invoke_agent" else "invoke_agent",
        status="OK",
        attributes=attributes,
    )


class TestValidator:
    def test_minimal_valid_trace(self) -> None:
        spans = [
            _rec(
                "invoke_agent",
                "invoke_agent",
                **{sc.GEN_AI_USAGE_INPUT_TOKENS: 1, sc.GEN_AI_USAGE_OUTPUT_TOKENS: 1},
            ),
            _rec(
                "chat m",
                "chat",
                **{
                    sc.GEN_AI_PROVIDER_NAME: "openai",
                    sc.GEN_AI_REQUEST_MODEL: "m",
                    sc.GEN_AI_USAGE_INPUT_TOKENS: 1,
                    sc.GEN_AI_USAGE_OUTPUT_TOKENS: 1,
                },
            ),
            _rec("execute_tool t", "execute_tool", **{sc.GEN_AI_TOOL_NAME: "t"}),
        ]
        assert sc.validate_spans(spans) == []
        sc.assert_valid_spans(spans)

    def test_problems_are_named(self) -> None:
        spans = [
            _rec("invoke_agent", "invoke_agent"),
            _rec("chat", "chat", **{"gen_ai.usage.prompt_tokens": 3}),
            _rec("execute_tool", "execute_tool"),
            _rec("mystery", None),
        ]
        problems = sc.validate_spans(spans)
        text = "\n".join(problems)
        assert "unknown attribute 'gen_ai.usage.prompt_tokens'" in text
        assert "missing gen_ai.provider.name" in text
        assert "missing gen_ai.tool.name" in text
        assert "missing gen_ai.operation.name" in text
        assert "missing gen_ai.usage.input_tokens" in text
        with pytest.raises(AssertionError, match="GenAI conventions"):
            sc.assert_valid_spans(spans)

    def test_wrong_kind_and_two_roots(self) -> None:
        chat = _rec(
            "chat m", "chat", **{sc.GEN_AI_PROVIDER_NAME: "p", sc.GEN_AI_REQUEST_MODEL: "m"}
        )
        chat.kind = "INTERNAL"
        chat.parent_span_id = None
        problems = sc.validate_spans(
            [_rec("invoke_agent", "invoke_agent"), chat], require_usage=False
        )
        assert any("2 root spans" in p for p in problems)
        assert any("chat spans are CLIENT" in p for p in problems)

    def test_round_trip_through_json(self) -> None:
        spans = [_rec("invoke_agent", "invoke_agent")]
        loaded = sc.load_spans(sc.dump_spans(spans))
        assert loaded == spans
        assert sc.validate_spans([]) == ["no spans recorded"]


# ---------------------------------------------------------------------------
# One hook, many runs: concurrent, nested, retried, failed
# ---------------------------------------------------------------------------


def _parent_name(exporter: Any, span: Any) -> str | None:
    by_id = {s.context.span_id: s for s in exporter.get_finished_spans()}
    parent = by_id.get(span.parent.span_id) if span.parent is not None else None
    return parent.name if parent is not None else None


class TestRuns:
    async def test_concurrent_runs_on_one_agent_are_two_traces(self, exporter: Any) -> None:
        """Two ``arun`` on one Agent: two roots, each with its own iterations and totals."""
        import asyncio

        from locus.agent import Agent
        from locus.testing import ScriptedModel, text, tool_call
        from locus.tools import tool

        @tool
        async def slow(x: str) -> str:
            """Wait, then echo."""
            await asyncio.sleep(0.02)
            return x

        scope = "locus.semconv.test-concurrent"
        agent = Agent(
            model=ScriptedModel(
                [tool_call("slow", x="1"), text("a"), tool_call("slow", x="2"), text("b")]
            ),
            tools=[slow],
            hooks=[TelemetryHook(tracer_name=scope)],
            name="shared",
            reflexion=False,
            grounding=False,
        )
        await asyncio.gather(agent.arun("one"), agent.arun("two"))

        spans = _spans(exporter, scope)
        roots = [s for s in spans if s.name == "invoke_agent shared"]
        assert len(roots) == 2
        assert len({s.context.trace_id for s in spans}) == 2
        # The script is shared, so the runs split its four replies between them.
        assert sum(r.attributes["locus.inference_calls"] for r in roots) == 4
        assert sum(r.attributes["locus.tool_calls"] for r in roots) == 2
        for root in roots:
            own = [s for s in spans if s.context.trace_id == root.context.trace_id]
            assert sc.validate_spans(own) == []
            calls = root.attributes["locus.inference_calls"]
            assert sum(1 for s in own if s.name == sc.ITERATION_SPAN_NAME) == calls
            assert sum(1 for s in own if s.name == "chat") == calls
            assert (
                sum(1 for s in own if s.name == "execute_tool slow")
                == (root.attributes["locus.tool_calls"])
            )
        assert all(s.status.status_code.name == "OK" for s in spans)

    async def test_sub_agent_run_sits_under_the_tool_span(self, exporter: Any) -> None:
        """A sub-agent exposed as a tool parents its root on the orchestrator's execute_tool."""
        from locus.agent import Agent
        from locus.testing import ScriptedModel, text, tool_call
        from locus.tools import tool

        scope = "locus.semconv.test-subagent"
        specialist = Agent(
            model=ScriptedModel([text("spec done")]),
            hooks=[TelemetryHook(tracer_name=scope)],
            name="specialist",
            reflexion=False,
            grounding=False,
        )

        @tool
        async def ask_specialist(question: str) -> str:
            """Delegate a question."""
            result = await specialist.arun(question)
            return str(result.message)

        orchestrator = Agent(
            model=ScriptedModel([tool_call("ask_specialist", question="q"), text("done")]),
            tools=[ask_specialist],
            hooks=[TelemetryHook(tracer_name=scope)],
            name="orchestrator",
            reflexion=False,
            grounding=False,
        )
        await orchestrator.arun("go")

        spans = _spans(exporter, scope)
        assert len({s.context.trace_id for s in spans}) == 1
        by_name = {s.name: s for s in spans}
        assert _parent_name(exporter, by_name["invoke_agent specialist"]) == (
            "execute_tool ask_specialist"
        )
        assert _parent_name(exporter, by_name["execute_tool ask_specialist"]) == (
            sc.ITERATION_SPAN_NAME
        )
        assert sc.validate_spans(spans) == []

    async def test_nested_runs_on_one_hook_each_get_a_record(self, exporter: Any) -> None:
        """One hook instance driven for an outer and an inner run keeps them apart."""
        from opentelemetry import trace as otel_trace

        from locus.testing import text

        scope = "locus.semconv.test-nested"
        hook = TelemetryHook(tracer_name=scope, model="openai:gpt-4.1")
        outer = _state(run_id="outer", agent_id="outer")
        inner = _state(run_id="inner", agent_id="inner")
        outer_run = SimpleNamespace(info=RunInfo(run_id="outer", thread_id="t-outer"))
        inner_run = SimpleNamespace(info=RunInfo(run_id="inner", thread_id="t-inner"))
        await hook.on_before_invocation("outer", outer)
        await hook.on_iteration_start(0, outer)
        await hook.on_before_invocation("inner", inner)
        assert hook._invocation_span.name == "invoke_agent inner"
        await hook.on_before_model_call(
            BeforeModelCallEvent(messages=[], tools=None, run=inner_run)
        )
        await hook.on_after_model_call(
            AfterModelCallEvent(response=text("ok"), messages=[], run=inner_run)
        )
        await hook.on_after_invocation(inner, success=True)
        assert hook._invocation_span.name == "invoke_agent outer"
        assert otel_trace.get_current_span().name == sc.ITERATION_SPAN_NAME
        await hook.on_before_model_call(
            BeforeModelCallEvent(messages=[], tools=None, run=outer_run)
        )
        await hook.on_after_model_call(
            AfterModelCallEvent(response=text("ok"), messages=[], run=outer_run)
        )
        await hook.on_iteration_end(0, outer)
        await hook.on_after_invocation(outer, success=True)

        by_name = _by_name(exporter, scope)
        assert by_name["invoke_agent inner"].attributes["locus.inference_calls"] == 1
        assert by_name["invoke_agent outer"].attributes["locus.inference_calls"] == 1
        assert by_name["invoke_agent inner"].attributes[sc.GEN_AI_CONVERSATION_ID] == "t-inner"
        assert by_name["invoke_agent outer"].attributes[sc.GEN_AI_CONVERSATION_ID] == "t-outer"
        assert _parent_name(exporter, by_name["invoke_agent inner"]) == sc.ITERATION_SPAN_NAME
        assert hook._invocation_span is None

    async def test_hook_retry_gives_every_attempt_a_chat_span(
        self, exporter: Any, metric_reader: Any
    ) -> None:
        """A hook's ``retry`` ends the chat span; the retried call gets its own and counts."""
        from locus.agent import Agent
        from locus.hooks.provider import HookPriority, HookProvider
        from locus.testing import ScriptedModel, text

        class RetryOnce(HookProvider):
            def __init__(self) -> None:
                self.calls = 0

            @property
            def priority(self) -> int:
                return HookPriority.OBSERVABILITY_MIN + 50

            @property
            def name(self) -> str:
                return "RetryOnce"

            async def on_after_model_call(self, event: AfterModelCallEvent) -> None:
                self.calls += 1
                if self.calls == 1:
                    event.retry = True
                    event.retry_feedback = "try again"

        scope = "locus.semconv.test-retry"
        agent = Agent(
            model=ScriptedModel([text("first"), text("second")]),
            hooks=[TelemetryHook(tracer_name=scope), RetryOnce()],
            name="retrying",
            reflexion=False,
            grounding=False,
        )
        result = await agent.arun("go")
        assert result.message == "second"

        spans = _spans(exporter, scope)
        chats = [s for s in spans if s.name == "chat"]
        assert len(chats) == 2
        assert all(s.status.status_code.name == "OK" for s in chats)
        assert all(s.attributes[sc.GEN_AI_USAGE_INPUT_TOKENS] == 10 for s in chats)
        assert chats[0].attributes["locus.retry_requested"] is True
        assert "locus.retry_attempt" not in chats[0].attributes
        assert chats[1].attributes["locus.retry_attempt"] == 2
        assert "locus.retry_requested" not in chats[1].attributes
        assert _parent_name(exporter, chats[1]) == sc.ITERATION_SPAN_NAME
        root = next(s for s in spans if s.name == "invoke_agent retrying")
        assert root.attributes["locus.inference_calls"] == 2
        assert sc.validate_spans(spans) == []
        points = _points(metric_reader)
        assert sum(p.count for p in points[sc.METRIC_INFERENCE_DURATION]) == 2
        assert sum(p.value for p in points[sc.METRIC_INPUT_TOKENS]) == 20

    async def test_failed_run_closes_everything_with_error_type(
        self, exporter: Any, metric_reader: Any
    ) -> None:
        """A run that raised mid-call leaves a trace the validator accepts."""
        scope = "locus.semconv.test-failed"
        hook = TelemetryHook(tracer_name=scope, model="openai:gpt-4.1")
        run = _run()
        failed = _state(errors=("RuntimeError: boom",))
        await hook.on_before_invocation("hi", failed)
        await hook.on_iteration_start(0, failed)
        await hook.on_before_model_call(BeforeModelCallEvent(messages=[], tools=None, run=run))
        await hook.on_after_invocation(failed, success=False)

        by_name = _by_name(exporter, scope)
        for name in ("chat gpt-4.1", sc.ITERATION_SPAN_NAME, "invoke_agent support"):
            assert by_name[name].status.status_code.name == "ERROR", name
            assert by_name[name].attributes[sc.ERROR_TYPE] == "RuntimeError", name
        assert by_name["invoke_agent support"].attributes["locus.inference_calls"] == 1
        assert sc.validate_spans(_spans(exporter, scope), require_usage=False) == []
        # The failed call's duration is recorded, with error.type, as the conventions ask.
        durations = _points(metric_reader)[sc.METRIC_INFERENCE_DURATION]
        assert sum(p.count for p in durations) == 1
        assert durations[0].attributes[sc.ERROR_TYPE] == "RuntimeError"
        assert sc.METRIC_INPUT_TOKENS not in _points(metric_reader)

    async def test_model_call_that_raised_closes_at_iteration_end(self, exporter: Any) -> None:
        scope = "locus.semconv.test-model-raised"
        hook = TelemetryHook(tracer_name=scope, model="openai:gpt-4.1")
        run = _run()
        await hook.on_before_invocation("hi", _state())
        await hook.on_iteration_start(0, _state())
        await hook.on_before_model_call(BeforeModelCallEvent(messages=[], tools=None, run=run))
        await hook.on_iteration_end(0, _state(errors=("TimeoutError: slow",)))
        await hook.on_after_invocation(_state(), success=True)
        chat = _by_name(exporter, scope)["chat gpt-4.1"]
        assert chat.status.status_code.name == "ERROR"
        assert chat.attributes[sc.ERROR_TYPE] == "TimeoutError"

    async def test_tool_spans_open_at_run_end_are_closed_and_forgotten(self, exporter: Any) -> None:
        """A cancelled run ends its tool spans as errors and leaves no in-flight entry."""
        scope = "locus.semconv.test-open-tools"
        hook = TelemetryHook(tracer_name=scope, model="openai:gpt-4.1")
        run = _run()
        await hook.on_before_invocation("hi", _state())
        await hook.on_iteration_start(0, _state())
        await hook.on_before_tool_call(
            BeforeToolCallEvent(tool_name="slow", tool_call_id="c-1", arguments={}, run=run)
        )
        assert "c-1" in telemetry_mod._TOOL_CALL_SPANS
        await hook.on_after_invocation(_state(errors=("CancelledError",)), success=False)
        assert "c-1" not in telemetry_mod._TOOL_CALL_SPANS
        assert hook._tool_spans == {}
        tool = _by_name(exporter, scope)["execute_tool slow"]
        assert tool.status.status_code.name == "ERROR"
        assert tool.attributes[sc.ERROR_TYPE] == "CancelledError"
        assert sc.validate_spans(_spans(exporter, scope), require_usage=False) == []

    async def test_by_name_fallback_stays_inside_the_run(self, exporter: Any) -> None:
        """An after-tool event without an id closes a span of its own run, never another's."""
        scope = "locus.semconv.test-by-name"
        hook = TelemetryHook(tracer_name=scope, model="openai:gpt-4.1")
        first, second = _state(run_id="r-1"), _state(run_id="r-2")
        run_1 = SimpleNamespace(info=RunInfo(run_id="r-1"))
        run_2 = SimpleNamespace(info=RunInfo(run_id="r-2"))
        await hook.on_before_invocation("one", first)
        await hook.on_before_tool_call(
            BeforeToolCallEvent(tool_name="lookup", tool_call_id="a", arguments={}, run=run_1)
        )
        await hook.on_before_invocation("two", second)
        # Names the tool but not the call, from the second run: the first run's span stays open.
        await hook.on_after_tool_call(
            AfterToolCallEvent(tool_name="lookup", result=None, error=None, run=run_2)
        )
        assert "a" in telemetry_mod._TOOL_CALL_SPANS
        await hook.on_after_tool_call(
            AfterToolCallEvent(tool_name="lookup", result="ok", error=None, run=run_1)
        )
        assert "a" not in telemetry_mod._TOOL_CALL_SPANS
        await hook.on_after_invocation(second, success=True)
        await hook.on_after_invocation(first, success=True)
        tool = _by_name(exporter, scope)["execute_tool lookup"]
        assert tool.status.status_code.name == "OK"
        roots = [s for s in _spans(exporter, scope) if s.name == "invoke_agent support"]
        root_1 = next(s for s in roots if s.attributes["locus.run_id"] == "r-1")
        assert tool.context.trace_id == root_1.context.trace_id

    async def test_tool_span_is_current_while_the_call_runs(self, exporter: Any) -> None:
        """One call open: its span is current; a batch of two: none is, the tool context tells."""
        from opentelemetry import trace as otel_trace

        scope = "locus.semconv.test-tool-current"
        hook = TelemetryHook(tracer_name=scope, model="openai:gpt-4.1")
        run = _run()
        await hook.on_before_invocation("hi", _state())
        await hook.on_iteration_start(0, _state())
        await hook.on_before_tool_call(
            BeforeToolCallEvent(tool_name="one", tool_call_id="c-1", arguments={}, run=run)
        )
        assert otel_trace.get_current_span().name == "execute_tool one"
        await hook.on_before_tool_call(
            BeforeToolCallEvent(tool_name="two", tool_call_id="c-2", arguments={}, run=run)
        )
        assert otel_trace.get_current_span().name == sc.ITERATION_SPAN_NAME
        for call_id, name in (("c-1", "one"), ("c-2", "two")):
            await hook.on_after_tool_call(
                AfterToolCallEvent(
                    tool_name=name, result="", error=None, tool_call_id=call_id, run=run
                )
            )
        assert otel_trace.get_current_span().name == sc.ITERATION_SPAN_NAME
        await hook.on_iteration_end(0, _state())
        await hook.on_after_invocation(_state(), success=True)
        assert otel_trace.get_current_span().get_span_context().span_id == 0


class TestTokenModality:
    async def test_unknown_without_a_breakdown(self, exporter: Any, metric_reader: Any) -> None:
        hook = TelemetryHook(tracer_name="locus.semconv.test-modality", model="openai:gpt-4.1")
        await _drive(hook)
        for name in (sc.METRIC_INPUT_TOKENS, sc.METRIC_OUTPUT_TOKENS):
            points = _points(metric_reader)[name]
            assert [p.attributes[sc.GEN_AI_TOKEN_MODALITY] for p in points] == ["unknown"]

    async def test_breakdown_in_the_usage_details_names_the_modality(
        self, exporter: Any, metric_reader: Any
    ) -> None:
        from locus.testing import text

        hook = TelemetryHook(tracer_name="locus.semconv.test-modality-2", model="openai:gpt-4.1")
        response = text("ok")
        response.usage = {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_tokens_details": {"audio_tokens": 30, "cached_tokens": 10},
            "completion_tokens_details": {"text_tokens": 20},
        }
        run = _run()
        await hook.on_before_invocation("hi", _state())
        await hook.on_before_model_call(BeforeModelCallEvent(messages=[], tools=None, run=run))
        await hook.on_after_model_call(AfterModelCallEvent(response=response, messages=[], run=run))
        await hook.on_after_invocation(_state(), success=True)
        points = _points(metric_reader)
        inputs = {
            p.attributes[sc.GEN_AI_TOKEN_MODALITY]: p.value for p in points[sc.METRIC_INPUT_TOKENS]
        }
        assert inputs == {"audio": 30, "unknown": 70}  # cached_tokens is no modality
        outputs = {
            p.attributes[sc.GEN_AI_TOKEN_MODALITY]: p.value for p in points[sc.METRIC_OUTPUT_TOKENS]
        }
        assert outputs == {"text": 20}

    def test_token_points(self) -> None:
        points = telemetry_mod._token_points
        keys = telemetry_mod._INPUT_DETAIL_KEYS
        assert points(None, {}, keys) == []
        assert points(0, {}, keys) == []
        assert points(5, {"prompt_tokens_details": {"cached_tokens": 5}}, keys) == [("unknown", 5)]
        assert points(5, {"input_tokens_details": {"image_tokens": 5}}, keys) == [("image", 5)]
        assert points(5, {"input_tokens_details": {"text_tokens": 9}}, keys) == [("text", 9)]


# ---------------------------------------------------------------------------
# Edges: the legacy shape's tool paths, the API without bucket advisories,
# a run ended on another task, events outside any run
# ---------------------------------------------------------------------------


class TestEdges:
    async def test_legacy_tool_spans_record_arguments_results_and_errors(
        self, exporter: Any, metric_reader: Any
    ) -> None:
        scope = "locus.semconv.test-legacy-tools"
        hook = TelemetryHook(
            tracer_name=scope,
            semconv="locus-legacy",
            record_arguments=True,
            record_results=True,
        )

        class Unprintable:
            def __str__(self) -> str:
                raise ValueError("no")

        await hook.on_before_invocation("hi", _state())
        await hook.on_before_tool_call(
            BeforeToolCallEvent(
                tool_name="lookup",
                tool_call_id="c-1",
                arguments={"order": "CS-1", "blob": Unprintable()},
            )
        )
        await hook.on_after_tool_call(
            AfterToolCallEvent(tool_name="lookup", result="found", error=None, tool_call_id="c-1")
        )
        await hook.on_before_tool_call(
            BeforeToolCallEvent(tool_name="lookup", tool_call_id="c-2", arguments={})
        )
        await hook.on_after_tool_call(
            AfterToolCallEvent(tool_name="lookup", result=None, error="boom", tool_call_id="c-2")
        )
        # A third call is still open when the run ends.
        await hook.on_before_tool_call(
            BeforeToolCallEvent(tool_name="lookup", tool_call_id="c-3", arguments={})
        )
        await hook.on_after_invocation(_state(), success=False)

        spans = [s for s in _spans(exporter, scope) if s.name == "tool.lookup"]
        assert len(spans) == 3
        first, second, third = spans
        assert first.attributes["locus.tool.arg.order"] == "CS-1"
        assert first.attributes["locus.tool.arg.blob"] == "<non-serializable>"
        assert first.attributes["locus.result_preview"] == "found"
        assert second.status.status_code.name == "ERROR"
        assert second.attributes["locus.error"] == "boom"
        assert third.status.status_code.name == "ERROR"
        assert third.attributes["locus.error"] == "tool call did not complete"
        points = _points(metric_reader)
        assert sum(p.value for p in points["locus.tool_errors"]) == 1
        assert sum(p.value for p in points["locus.tool_calls"]) == 3

    def test_histograms_without_the_bucket_advisory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An OpenTelemetry API that predates ``explicit_bucket_boundaries_advisory``."""
        from opentelemetry import metrics as otel_metrics

        meter = otel_metrics.get_meter("locus.semconv.test-old-api")
        real = meter.create_histogram
        seen: list[dict[str, Any]] = []

        def create_histogram(name: str, **kwargs: Any) -> Any:
            seen.append(kwargs)
            if "explicit_bucket_boundaries_advisory" in kwargs:
                raise TypeError("unexpected keyword argument")
            return real(name, **kwargs)

        monkeypatch.setattr(meter, "create_histogram", create_histogram)
        monkeypatch.setattr(otel_metrics, "get_meter", lambda name: meter)
        TelemetryHook(tracer_name="locus.semconv.test-old-api")
        assert any("explicit_bucket_boundaries_advisory" in k for k in seen)
        assert any("explicit_bucket_boundaries_advisory" not in k for k in seen)

    async def test_run_ended_on_another_task_still_closes(self, exporter: Any) -> None:
        """The record goes even when the ending hook runs where the start did not."""
        import asyncio

        scope = "locus.semconv.test-other-task"
        hook = TelemetryHook(tracer_name=scope, model="openai:gpt-4.1")
        await hook.on_before_invocation("hi", _state())
        await asyncio.create_task(hook.on_after_invocation(_state(), success=True))
        assert hook._invocation_span is None
        assert _by_name(exporter, scope)["invoke_agent support"].status.status_code.name == "OK"

    async def test_events_outside_any_run(self, exporter: Any) -> None:
        """Hook events with no run in flight: no error, no span for what never started."""
        scope = "locus.semconv.test-no-run"
        hook = TelemetryHook(tracer_name=scope, model="openai:gpt-4.1")
        unknown = SimpleNamespace(info=RunInfo(run_id="never-started"))
        await hook.on_after_model_call(AfterModelCallEvent(response=None, messages=[], run=unknown))
        await hook.on_after_tool_call(
            AfterToolCallEvent(tool_name="x", result=None, error=None, run=unknown)
        )
        await hook.on_iteration_end(0, _state(run_id="never-started"))
        await hook.on_after_invocation(_state(run_id="never-started"), success=True)
        assert hook._run_for("never-started") is None
        assert hook._tool_spans == {}
        assert hook._iteration_spans == {}
        assert _spans(exporter, scope) == []

    async def test_explicit_agent_name_wins_on_every_span(self, exporter: Any) -> None:
        scope = "locus.semconv.test-agent-name"
        hook = TelemetryHook(tracer_name=scope, model="openai:gpt-4.1", agent_name="desk")
        await _drive(hook, run=_run(agent_name="other"))
        by_name = _by_name(exporter, scope)
        assert "invoke_agent desk" in by_name
        assert by_name["execute_tool lookup_order"].attributes[sc.GEN_AI_AGENT_NAME] == "desk"


# ---------------------------------------------------------------------------
# The naming helpers and the validator on the less common shapes
# ---------------------------------------------------------------------------


class TestSemconvEdges:
    def test_identity_helpers_on_odd_inputs(self) -> None:
        class OCIModel:
            pass

        class AnthropicModel:
            pass

        class Bare:
            model_id = "qwen-7b"

        assert sc.request_model(object()) is None
        assert sc.request_model(Bare()) == "qwen-7b"
        assert sc.provider_name("gpt-4.1") is None  # no provider prefix
        assert sc.provider_name(OCIModel()) == sc.PROVIDER_OCI_GENERATIVE_AI
        assert sc.provider_name(AnthropicModel()) == "anthropic"
        assert sc.provider_name(OpenAIModel("x", "not a url")) == "openai"
        assert sc.server_address(OpenAIModel("x", "not a url")) == (None, None)
        assert sc.server_address(object()) == (None, None)
        assert sc.fallback_provider_name(None) is None

    def test_records_from_dicts_and_span_records(self) -> None:
        record = _rec("invoke_agent", "invoke_agent")
        loaded = sc.records_from_spans([record, record.to_dict()])
        assert loaded == [record, record]

    def test_validator_names_every_other_problem(self) -> None:
        chat = _rec(
            "chat other",
            "chat",
            **{
                sc.GEN_AI_PROVIDER_NAME: "p",
                sc.GEN_AI_REQUEST_MODEL: "m",
                sc.GEN_AI_RESPONSE_FINISH_REASONS: "stop",
            },
        )
        chat.status = "ERROR"
        odd_kind = _rec("execute_tool t", "execute_tool", **{sc.GEN_AI_TOOL_NAME: "t"})
        odd_kind.kind = "SIDEWAYS"
        odd_kind.span_id = "odd"
        misnamed_tool = _rec(
            "tool t", "execute_tool", **{sc.GEN_AI_TOOL_NAME: "t", sc.GEN_AI_TOOL_CALL_ID: 7}
        )
        misnamed_tool.span_id = "misnamed"
        unknown_op = _rec("x", "dance")
        iteration = _rec(sc.ITERATION_SPAN_NAME, None)
        iteration.parent_span_id = None
        other_trace = _rec("invoke_agent", "invoke_agent")
        other_trace.trace_id = "t2"
        other_trace.span_id = "root-2"
        root = _rec(
            "invoke_agent misnamed",
            "invoke_agent",
            **{sc.GEN_AI_USAGE_INPUT_TOKENS: -1, sc.GEN_AI_USAGE_OUTPUT_TOKENS: 2},
        )
        root.span_id = "invoke_agent"

        problems = sc.validate_spans(
            [root, chat, odd_kind, misnamed_tool, unknown_op, iteration, other_trace]
        )
        text = "\n".join(problems)
        assert "belong to 2 traces" in text
        assert "root spans, expected 1" in text
        assert "expected name 'invoke_agent'" in text
        assert "status ERROR without error.type" in text
        assert "expected name 'chat m'" in text
        assert "must be a string list" in text
        assert "must be a non-negative integer" in text
        assert "unknown span kind 'SIDEWAYS'" in text
        assert "expected name 'execute_tool t'" in text
        assert "gen_ai.tool.call.id must be a string" in text
        assert "unknown operation 'dance'" in text
        assert "iteration span has no parent" in text
        # A root that is not an invoke_agent span is named as such.
        lone = _rec(
            "chat m", "chat", **{sc.GEN_AI_PROVIDER_NAME: "p", sc.GEN_AI_REQUEST_MODEL: "m"}
        )
        lone.parent_span_id = None
        assert any(
            "is not an invoke_agent span" in p
            for p in sc.validate_spans([lone], require_usage=False)
        )
