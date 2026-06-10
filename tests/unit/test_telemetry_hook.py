# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for telemetry hook."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from tulip.hooks.builtin.telemetry import (
    OTEL_AVAILABLE,
    NoOpTelemetryHook,
    TelemetryHook,
    create_telemetry_hook,
)
from tulip.hooks.provider import (
    AfterToolCallEvent,
    BeforeToolCallEvent,
    HookPriority,
)


def _before(tool_name: str, arguments: dict) -> BeforeToolCallEvent:
    return BeforeToolCallEvent(
        tool_name=tool_name, tool_call_id=f"{tool_name}-call", arguments=arguments
    )


def _after(tool_name: str, *, result, error: str | None) -> AfterToolCallEvent:
    return AfterToolCallEvent(tool_name=tool_name, result=result, error=error)


class TestNoOpTelemetryHook:
    """Tests for NoOpTelemetryHook."""

    def test_create_with_default_priority(self):
        """Test creating hook with default priority."""
        hook = NoOpTelemetryHook()
        assert hook.priority == HookPriority.OBSERVABILITY_MIN + 10

    def test_create_with_custom_priority(self):
        """Test creating hook with custom priority."""
        hook = NoOpTelemetryHook(priority=100)
        assert hook.priority == 100

    def test_name(self):
        """Test hook name."""
        hook = NoOpTelemetryHook()
        assert hook.name == "NoOpTelemetryHook"


@pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not installed")
class TestTelemetryHook:
    """Tests for TelemetryHook (requires OpenTelemetry)."""

    @pytest.fixture
    def hook(self):
        """Create a telemetry hook."""
        return TelemetryHook(
            service_name="test-service",
            record_arguments=True,
            record_results=True,
        )

    @pytest.fixture
    def mock_state(self):
        """Create a mock agent state."""
        state = MagicMock()
        state.run_id = "test-run-123"
        state.agent_id = "test-agent"
        state.max_iterations = 10
        state.iteration = 3
        state.confidence = 0.85
        state.tool_executions = []
        state.errors = []
        state.messages = []
        state.started_at = datetime.now(UTC)
        state.updated_at = datetime.now(UTC) + timedelta(seconds=5)
        return state

    def test_create_hook(self):
        """Test creating telemetry hook."""
        hook = TelemetryHook()
        assert hook._service_name == "tulip-agent"
        assert hook._record_arguments is False
        assert hook._record_results is False

    def test_create_hook_custom(self):
        """Test creating telemetry hook with custom settings."""
        hook = TelemetryHook(
            service_name="custom-service",
            tracer_name="custom.tracer",
            meter_name="custom.meter",
            record_arguments=True,
            record_results=True,
            priority=50,
        )
        assert hook._service_name == "custom-service"
        assert hook._record_arguments is True
        assert hook._record_results is True
        assert hook.priority == 50

    def test_hook_name(self, hook):
        """Test hook name."""
        assert hook.name == "TelemetryHook"

    def test_hook_priority(self, hook):
        """Test hook priority."""
        assert hook.priority == HookPriority.OBSERVABILITY_MIN + 10

    @pytest.mark.asyncio
    async def test_on_before_invocation(self, hook, mock_state):
        """Test on_before_invocation starts span."""
        result = await hook.on_before_invocation("Test prompt", mock_state)

        assert result is mock_state
        assert hook._invocation_span is not None

    @pytest.mark.asyncio
    async def test_on_after_invocation_success(self, hook, mock_state):
        """Test on_after_invocation with success."""
        # Start the span first
        await hook.on_before_invocation("Test prompt", mock_state)

        # End the span
        await hook.on_after_invocation(mock_state, success=True)

        assert hook._invocation_span is None

    @pytest.mark.asyncio
    async def test_on_after_invocation_failure(self, hook, mock_state):
        """Test on_after_invocation with failure."""
        await hook.on_before_invocation("Test prompt", mock_state)
        await hook.on_after_invocation(mock_state, success=False)

        assert hook._invocation_span is None

    @pytest.mark.asyncio
    async def test_on_after_invocation_no_span(self, hook, mock_state):
        """Test on_after_invocation when no span exists."""
        # Call without starting span first
        await hook.on_after_invocation(mock_state, success=True)
        # Should not raise

    @pytest.mark.asyncio
    async def test_on_before_tool_call(self, hook):
        """Test on_before_tool_call starts span."""
        args = {"query": "test", "limit": 10}
        event = _before("search", args)
        await hook.on_before_tool_call(event)

        # Hook is observe-only — event.arguments is unmodified.
        assert event.arguments == args
        assert "search" in hook._tool_spans

    @pytest.mark.asyncio
    async def test_on_before_tool_call_no_record_args(self):
        """Test on_before_tool_call without recording arguments."""
        hook = TelemetryHook(record_arguments=False)
        args = {"query": "test"}
        event = _before("search", args)
        await hook.on_before_tool_call(event)

        assert event.arguments == args
        assert "search" in hook._tool_spans

    @pytest.mark.asyncio
    async def test_on_after_tool_call_success(self, hook):
        """Test on_after_tool_call with success."""
        # Start tool span
        await hook.on_before_tool_call(_before("search", {}))

        # End tool span
        await hook.on_after_tool_call(_after("search", result="Found 5 items", error=None))

        assert "search" not in hook._tool_spans

    @pytest.mark.asyncio
    async def test_on_after_tool_call_with_error(self, hook):
        """Test on_after_tool_call with error."""
        await hook.on_before_tool_call(_before("search", {}))
        await hook.on_after_tool_call(_after("search", result=None, error="Connection failed"))

        assert "search" not in hook._tool_spans

    @pytest.mark.asyncio
    async def test_on_after_tool_call_no_span(self, hook):
        """Test on_after_tool_call when no span exists."""
        # Call without starting span
        await hook.on_after_tool_call(_after("missing_tool", result="data", error=None))
        # Should not raise

    @pytest.mark.asyncio
    async def test_on_after_tool_call_no_record_results(self):
        """Test on_after_tool_call without recording results."""
        hook = TelemetryHook(record_results=False)
        await hook.on_before_tool_call(_before("search", {}))
        await hook.on_after_tool_call(_after("search", result="Result data", error=None))

        assert "search" not in hook._tool_spans

    @pytest.mark.asyncio
    async def test_on_iteration_start(self, hook, mock_state):
        """Test on_iteration_start creates span."""
        await hook.on_iteration_start(1, mock_state)

        assert 1 in hook._iteration_spans

    @pytest.mark.asyncio
    async def test_on_iteration_end(self, hook, mock_state):
        """Test on_iteration_end closes span."""
        await hook.on_iteration_start(1, mock_state)
        await hook.on_iteration_end(1, mock_state)

        assert 1 not in hook._iteration_spans

    @pytest.mark.asyncio
    async def test_on_iteration_end_no_span(self, hook, mock_state):
        """Test on_iteration_end when no span exists."""
        # Call without starting span
        await hook.on_iteration_end(999, mock_state)
        # Should not raise

    def test_span_context_manager(self, hook):
        """Test _span context manager."""
        with hook._span("test.span", {"key": "value"}) as span:
            assert span is not None

    @pytest.mark.asyncio
    async def test_tool_call_with_non_serializable_arg(self, hook):
        """Test tool call with non-serializable argument."""

        class NonSerializable:
            def __str__(self):
                raise ValueError("Cannot serialize")

        args = {"obj": NonSerializable()}
        # Should not raise
        event = _before("test_tool", args)
        await hook.on_before_tool_call(event)
        assert event.arguments == args


class TestCreateTelemetryHook:
    """Tests for create_telemetry_hook factory."""

    def test_create_disabled(self):
        """Test creating disabled telemetry hook."""
        hook = create_telemetry_hook(enabled=False)
        assert isinstance(hook, NoOpTelemetryHook)

    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not installed")
    def test_create_enabled(self):
        """Test creating enabled telemetry hook."""
        hook = create_telemetry_hook(enabled=True)
        assert isinstance(hook, TelemetryHook)

    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not installed")
    def test_create_with_kwargs(self):
        """Test creating hook with custom kwargs."""
        hook = create_telemetry_hook(
            enabled=True,
            service_name="custom",
            record_arguments=True,
        )
        assert isinstance(hook, TelemetryHook)
        assert hook._service_name == "custom"
        assert hook._record_arguments is True

    def test_create_otel_not_available(self):
        """Test creating hook when OpenTelemetry is not available."""
        with patch("tulip.hooks.builtin.telemetry.OTEL_AVAILABLE", False):
            # Reimport to get patched behavior
            from tulip.hooks.builtin import telemetry

            original_otel = telemetry.OTEL_AVAILABLE
            telemetry.OTEL_AVAILABLE = False

            try:
                hook = telemetry.create_telemetry_hook(enabled=True)
                assert isinstance(hook, NoOpTelemetryHook)
            finally:
                telemetry.OTEL_AVAILABLE = original_otel


@pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not installed")
class TestTelemetryHookMetrics:
    """Tests for telemetry hook metrics."""

    @pytest.fixture
    def hook(self):
        """Create telemetry hook."""
        return TelemetryHook()

    def test_metrics_created(self, hook):
        """Test that metrics are created."""
        assert hook._invocation_counter is not None
        assert hook._invocation_duration is not None
        assert hook._iteration_counter is not None
        assert hook._tool_call_counter is not None
        assert hook._tool_call_duration is not None
        assert hook._tool_error_counter is not None


class TestOtelNotAvailable:
    """Tests for when OpenTelemetry is not available."""

    def test_telemetry_hook_raises_import_error(self):
        """Test TelemetryHook raises ImportError when OTEL not available."""
        with patch("tulip.hooks.builtin.telemetry.OTEL_AVAILABLE", False):
            from tulip.hooks.builtin import telemetry

            original_otel = telemetry.OTEL_AVAILABLE
            telemetry.OTEL_AVAILABLE = False

            try:
                with pytest.raises(ImportError, match="OpenTelemetry is not installed"):
                    telemetry.TelemetryHook()
            finally:
                telemetry.OTEL_AVAILABLE = original_otel


def _otel_sdk_available() -> bool:
    """The SDK (TracerProvider, InMemorySpanExporter) is a separate dep
    from the OTel API. The TelemetryHook only needs the API at runtime,
    but trace-tree assertions need the SDK's in-memory exporter."""
    try:
        import opentelemetry.sdk.trace  # noqa: F401
        import opentelemetry.sdk.trace.export.in_memory_span_exporter  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(
    not OTEL_AVAILABLE or not _otel_sdk_available(),
    reason="OpenTelemetry API + SDK required",
)
class TestTraceContextPropagation:
    """Regression tests for issue #244 — every span must share a trace tree.

    Before the fix, ``TelemetryHook`` used ``tracer.start_span`` without
    attaching the span to OTel's current context, so each invocation /
    iteration / tool span emerged as its own trace root. Langfuse, Jaeger,
    and any other OTLP backend then rendered N separate traces instead of
    a single tree per agent run.
    """

    @pytest.fixture
    def exporter(self):
        """Wire an InMemorySpanExporter onto the global TracerProvider.

        OTel's global TracerProvider can only be set once per process —
        attempting to override it logs a warning and silently keeps the
        original. So we install our SDK provider on first use and attach
        a fresh processor + exporter per test, clearing between runs.
        """
        from opentelemetry import trace as otel_trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        # Install a real SDK provider once; the default no-op provider
        # has no `add_span_processor` so we'd silently drop spans.
        current = otel_trace.get_tracer_provider()
        if not isinstance(current, TracerProvider):
            otel_trace.set_tracer_provider(TracerProvider())
        provider = otel_trace.get_tracer_provider()

        exp = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exp))
        return exp

    @pytest.fixture
    def mock_state(self):
        """Mock AgentState used by the hook."""
        state = MagicMock()
        state.run_id = "trace-test"
        state.agent_id = "test-agent"
        state.max_iterations = 3
        state.iteration = 0
        state.confidence = 0.0
        state.tool_executions = []
        state.errors = []
        state.messages = []
        state.started_at = datetime.now(UTC)
        state.updated_at = datetime.now(UTC) + timedelta(seconds=1)
        return state

    @pytest.mark.asyncio
    async def test_all_spans_share_one_trace_id(self, exporter, mock_state):
        """invocation + iteration + tool spans must share a single trace_id."""
        hook = TelemetryHook(tracer_name="tulip.telemetry.test-244")

        await hook.on_before_invocation("Question?", mock_state)
        await hook.on_iteration_start(0, mock_state)
        await hook.on_before_tool_call(_before("calculator", {"x": 1}))
        await hook.on_after_tool_call(_after("calculator", result=2, error=None))
        await hook.on_iteration_end(0, mock_state)
        await hook.on_after_invocation(mock_state, success=True)

        spans = [
            s
            for s in exporter.get_finished_spans()
            if s.instrumentation_scope.name == "tulip.telemetry.test-244"
        ]
        # We expect: 1 invocation + 1 iteration + 1 tool = 3 spans.
        assert len(spans) == 3, f"expected 3 spans, got {len(spans)}: {[s.name for s in spans]}"

        trace_ids = {s.context.trace_id for s in spans}
        assert len(trace_ids) == 1, (
            f"all spans must share one trace, got {len(trace_ids)} "
            f"distinct trace_ids — see issue #244"
        )

    @pytest.mark.asyncio
    async def test_tool_span_is_child_of_iteration_span(self, exporter, mock_state):
        """Tool span's parent must be the iteration span (not the invocation)."""
        hook = TelemetryHook(tracer_name="tulip.telemetry.test-244-parent")

        await hook.on_before_invocation("Q", mock_state)
        await hook.on_iteration_start(0, mock_state)
        await hook.on_before_tool_call(_before("ping", {}))
        await hook.on_after_tool_call(_after("ping", result="pong", error=None))
        await hook.on_iteration_end(0, mock_state)
        await hook.on_after_invocation(mock_state, success=True)

        by_name = {
            s.name: s
            for s in exporter.get_finished_spans()
            if s.instrumentation_scope.name == "tulip.telemetry.test-244-parent"
        }
        assert "agent.invocation" in by_name
        assert "agent.iteration.0" in by_name
        assert "tool.ping" in by_name

        # iteration's parent = invocation
        assert by_name["agent.iteration.0"].parent is not None
        assert (
            by_name["agent.iteration.0"].parent.span_id
            == by_name["agent.invocation"].context.span_id
        ), "iteration span must be a child of invocation span"

        # tool's parent = iteration
        assert by_name["tool.ping"].parent is not None
        assert (
            by_name["tool.ping"].parent.span_id == by_name["agent.iteration.0"].context.span_id
        ), "tool span must be a child of iteration span"

        # invocation has no parent (it IS the root)
        assert by_name["agent.invocation"].parent is None, "invocation must be the trace root"

    @pytest.mark.asyncio
    async def test_invocation_context_detaches_after_run(self, exporter, mock_state):
        """A span started after `on_after_invocation` must not parent to ours.

        Verifies we don't leak the invocation context into whatever the
        caller does next (e.g. another framework's span).
        """
        from opentelemetry import trace as otel_trace

        hook = TelemetryHook(tracer_name="tulip.telemetry.test-244-detach")
        await hook.on_before_invocation("Q", mock_state)
        await hook.on_after_invocation(mock_state, success=True)

        # After detach, the current span should be INVALID (no active span).
        current = otel_trace.get_current_span()
        # INVALID_SPAN.context.span_id == 0 — the "no span" sentinel.
        assert current.get_span_context().span_id == 0, (
            "invocation context wasn't detached — caller's spans would still be parented to ours"
        )
