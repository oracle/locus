# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for tools executor module."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from locus.core.messages import ToolCall
from locus.tools.executor import (
    CircuitBreakerExecutor,
    ConcurrentExecutor,
    SequentialExecutor,
    ToolContextFactory,
)


class TestToolContextFactory:
    """Tests for ToolContextFactory."""

    def test_create_factory(self):
        """Test creating a context factory."""
        factory = ToolContextFactory(
            run_id="run123",
            agent_id="agent1",
            iteration=5,
        )
        assert factory.run_id == "run123"
        assert factory.agent_id == "agent1"
        assert factory.iteration == 5

    def test_create_context(self):
        """Test creating a context from factory."""
        factory = ToolContextFactory(
            run_id="run123",
            agent_id="agent1",
            iteration=5,
            state={"key": "value"},
            invocation_metadata={"meta": "data"},
        )

        tool_call = ToolCall(
            id="call1",
            name="test_tool",
            arguments={"arg": "value"},
        )

        ctx = factory.create(tool_call, "test_tool")

        assert ctx.tool_call_id == "call1"
        assert ctx.tool_name == "test_tool"
        assert ctx.agent_id == "agent1"
        assert ctx.run_id == "run123"
        assert ctx.iteration == 5
        assert ctx.state == {"key": "value"}
        assert ctx.invocation_metadata == {"meta": "data"}


class TestSequentialExecutor:
    """Tests for SequentialExecutor."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock tool registry."""
        registry = MagicMock()

        mock_tool = MagicMock()
        mock_tool.execute = AsyncMock(return_value="result")

        registry.get = MagicMock(return_value=mock_tool)
        return registry, mock_tool

    @pytest.mark.asyncio
    async def test_execute_single_tool(self, mock_registry):
        """Test executing a single tool."""
        registry, mock_tool = mock_registry
        executor = SequentialExecutor()

        tool_calls = [
            ToolCall(id="call1", name="test_tool", arguments={"arg": "value"}),
        ]

        results = await executor.execute(tool_calls, registry)

        assert len(results) == 1
        assert results[0].tool_call_id == "call1"
        assert results[0].name == "test_tool"
        assert results[0].content == "result"
        assert results[0].error is None

    @pytest.mark.asyncio
    async def test_execute_multiple_tools(self, mock_registry):
        """Test executing multiple tools sequentially."""
        registry, mock_tool = mock_registry
        executor = SequentialExecutor()

        tool_calls = [
            ToolCall(id="call1", name="tool1", arguments={}),
            ToolCall(id="call2", name="tool2", arguments={}),
            ToolCall(id="call3", name="tool3", arguments={}),
        ]

        results = await executor.execute(tool_calls, registry)

        assert len(results) == 3
        assert all(r.error is None for r in results)

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, mock_registry):
        """Test executing unknown tool returns error."""
        registry, mock_tool = mock_registry
        registry.get = MagicMock(return_value=None)

        executor = SequentialExecutor()
        tool_calls = [
            ToolCall(id="call1", name="unknown_tool", arguments={}),
        ]

        results = await executor.execute(tool_calls, registry)

        assert len(results) == 1
        assert results[0].error == "Unknown tool: unknown_tool"
        assert results[0].content == ""

    @pytest.mark.asyncio
    async def test_execute_with_exception(self, mock_registry):
        """Test handling tool execution exception."""
        registry, mock_tool = mock_registry
        mock_tool.execute = AsyncMock(side_effect=ValueError("Tool failed"))

        executor = SequentialExecutor()
        tool_calls = [
            ToolCall(id="call1", name="test_tool", arguments={}),
        ]

        results = await executor.execute(tool_calls, registry)

        assert len(results) == 1
        assert results[0].error == "ValueError: Tool failed"
        assert results[0].content == ""

    @pytest.mark.asyncio
    async def test_execute_with_context_factory(self, mock_registry):
        """Test execution with context factory."""
        registry, mock_tool = mock_registry
        executor = SequentialExecutor()

        factory = ToolContextFactory(run_id="run123", agent_id="agent1")
        tool_calls = [
            ToolCall(id="call1", name="test_tool", arguments={"x": 1}),
        ]

        results = await executor.execute(tool_calls, registry, factory)

        assert len(results) == 1
        # Verify tool was called with context
        mock_tool.execute.assert_called_once()
        call_kwargs = mock_tool.execute.call_args.kwargs
        assert "ctx" in call_kwargs
        assert call_kwargs["ctx"] is not None

    @pytest.mark.asyncio
    async def test_execute_duration_tracking(self, mock_registry):
        """Test that execution tracks duration."""
        registry, mock_tool = mock_registry
        executor = SequentialExecutor()

        tool_calls = [
            ToolCall(id="call1", name="test_tool", arguments={}),
        ]

        results = await executor.execute(tool_calls, registry)

        assert results[0].duration_ms is not None
        assert results[0].duration_ms >= 0


class TestConcurrentExecutor:
    """Tests for ConcurrentExecutor."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock tool registry."""
        registry = MagicMock()

        mock_tool = MagicMock()
        mock_tool.execute = AsyncMock(return_value="result")

        registry.get = MagicMock(return_value=mock_tool)
        return registry, mock_tool

    def test_default_concurrency(self):
        """Test default max concurrency."""
        executor = ConcurrentExecutor()
        assert executor.max_concurrency == 10

    def test_custom_concurrency(self):
        """Test custom max concurrency."""
        executor = ConcurrentExecutor(max_concurrency=5)
        assert executor.max_concurrency == 5

    @pytest.mark.asyncio
    async def test_execute_concurrent(self, mock_registry):
        """Test concurrent execution."""
        registry, mock_tool = mock_registry
        executor = ConcurrentExecutor(max_concurrency=3)

        tool_calls = [ToolCall(id=f"call{i}", name="test_tool", arguments={}) for i in range(5)]

        results = await executor.execute(tool_calls, registry)

        assert len(results) == 5
        assert all(r.error is None for r in results)

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, mock_registry):
        """Test executing unknown tool returns error."""
        registry, mock_tool = mock_registry
        registry.get = MagicMock(return_value=None)

        executor = ConcurrentExecutor()
        tool_calls = [
            ToolCall(id="call1", name="unknown", arguments={}),
        ]

        results = await executor.execute(tool_calls, registry)

        assert len(results) == 1
        assert "Unknown tool" in results[0].error

    @pytest.mark.asyncio
    async def test_execute_with_exception(self, mock_registry):
        """Test handling concurrent execution exception."""
        registry, mock_tool = mock_registry
        mock_tool.execute = AsyncMock(side_effect=RuntimeError("Failed"))

        executor = ConcurrentExecutor()
        tool_calls = [
            ToolCall(id="call1", name="test_tool", arguments={}),
        ]

        results = await executor.execute(tool_calls, registry)

        assert results[0].error == "RuntimeError: Failed"


class TestCircuitBreakerExecutor:
    """Tests for CircuitBreakerExecutor."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock tool registry."""
        registry = MagicMock()

        mock_tool = MagicMock()
        mock_tool.execute = AsyncMock(return_value="result")

        registry.get = MagicMock(return_value=mock_tool)
        return registry, mock_tool

    def test_default_threshold(self):
        """Test default failure threshold."""
        executor = CircuitBreakerExecutor()
        assert executor.failure_threshold == 3

    def test_custom_threshold(self):
        """Test custom failure threshold."""
        executor = CircuitBreakerExecutor(failure_threshold=5)
        assert executor.failure_threshold == 5

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_registry):
        """Test successful execution."""
        registry, mock_tool = mock_registry
        executor = CircuitBreakerExecutor()

        tool_calls = [
            ToolCall(id="call1", name="test_tool", arguments={}),
        ]

        results = await executor.execute(tool_calls, registry)

        assert len(results) == 1
        assert results[0].error is None

    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self, mock_registry):
        """Test circuit opens after consecutive failures."""
        registry, mock_tool = mock_registry
        mock_tool.execute = AsyncMock(side_effect=ValueError("Failed"))

        executor = CircuitBreakerExecutor(failure_threshold=2)

        # First two calls fail but circuit stays closed
        tool_calls = [
            ToolCall(id="call1", name="failing_tool", arguments={}),
        ]
        results = await executor.execute(tool_calls, registry)
        assert results[0].error == "ValueError: Failed"

        results = await executor.execute(tool_calls, registry)
        assert results[0].error == "ValueError: Failed"

        # Third call should be blocked by circuit breaker
        tool_calls = [
            ToolCall(id="call3", name="failing_tool", arguments={}),
        ]
        results = await executor.execute(tool_calls, registry)
        assert "Circuit breaker open" in results[0].error

    @pytest.mark.asyncio
    async def test_reset_circuit(self, mock_registry):
        """Test resetting circuit breaker."""
        registry, mock_tool = mock_registry
        mock_tool.execute = AsyncMock(side_effect=ValueError("Failed"))

        executor = CircuitBreakerExecutor(failure_threshold=1)

        # Fail once to open circuit
        tool_calls = [
            ToolCall(id="call1", name="failing_tool", arguments={}),
        ]
        await executor.execute(tool_calls, registry)

        # Reset the circuit
        executor.reset("failing_tool")

        # Now should be able to call again
        mock_tool.execute = AsyncMock(return_value="success")
        results = await executor.execute(tool_calls, registry)
        assert results[0].content == "success"

    @pytest.mark.asyncio
    async def test_reset_all_circuits(self, mock_registry):
        """Test resetting all circuit breakers."""
        registry, mock_tool = mock_registry
        mock_tool.execute = AsyncMock(side_effect=ValueError("Failed"))

        executor = CircuitBreakerExecutor(failure_threshold=1)

        # Fail to open circuit
        tool_calls = [
            ToolCall(id="call1", name="failing_tool", arguments={}),
        ]
        await executor.execute(tool_calls, registry)

        # Reset all circuits
        executor.reset()

        # Should be able to call again
        mock_tool.execute = AsyncMock(return_value="success")
        results = await executor.execute(tool_calls, registry)
        assert results[0].content == "success"

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self, mock_registry):
        """Test that success resets failure count."""
        registry, mock_tool = mock_registry
        executor = CircuitBreakerExecutor(failure_threshold=2)

        tool_calls = [
            ToolCall(id="call1", name="test_tool", arguments={}),
        ]

        # Fail once
        mock_tool.execute = AsyncMock(side_effect=ValueError("Failed"))
        await executor.execute(tool_calls, registry)

        # Succeed - should reset counter
        mock_tool.execute = AsyncMock(return_value="success")
        await executor.execute(tool_calls, registry)

        # Fail once more - should not open circuit
        mock_tool.execute = AsyncMock(side_effect=ValueError("Failed"))
        await executor.execute(tool_calls, registry)

        # Should still be able to call (not open)
        results = await executor.execute(tool_calls, registry)
        assert results[0].error == "ValueError: Failed"
        assert "Circuit breaker" not in results[0].error


# =============================================================================
# execute_streaming — added with the completion-order / interrupt-cancel
# work in the #210 follow-up. The streaming variant is the path the runtime
# loop now takes; ``execute`` stays for back-compat with anyone who already
# constructs the executor directly.
# =============================================================================


class _Tool:
    """Tiny stand-in for a registered tool whose body sleeps then echoes."""

    def __init__(self, sleep_ms: float = 0.0, name: str = "stub") -> None:
        import asyncio as _asyncio

        async def _body(**kwargs: object) -> str:
            await _asyncio.sleep(sleep_ms / 1000.0)
            return f"{name}:{kwargs}"

        self.execute = _body
        self.name = name


def _registry_with(*tools_by_name: tuple[str, _Tool]) -> Any:
    """Return a duck-typed registry that ``.get(name)`` resolves to a tool.

    Typed as ``Any`` because the executor's signature expects
    ``ToolRegistry`` — these tests intentionally use lightweight
    stand-ins to keep the focus on streaming/scheduling behaviour
    rather than registry plumbing.
    """
    lookup = dict(tools_by_name)

    class _R:
        def get(self, name: str) -> _Tool | None:
            return lookup.get(name)

    return _R()


class TestToolExecutorAbstractStreaming:
    """The ABC's default ``execute_streaming`` impl — falls back to
    ``execute`` and yields in input order. Reached only by subclasses that
    don't override it (the two real executors below do)."""

    @pytest.mark.asyncio
    async def test_default_streaming_falls_back_to_execute(self) -> None:
        from locus.core.messages import ToolResult
        from locus.tools.executor import ToolExecutor

        class _Stub(ToolExecutor):
            async def execute(
                self,
                tool_calls: list,  # type: ignore[type-arg]
                registry: object,
                ctx_factory: object | None = None,
            ) -> list[ToolResult]:
                return [
                    ToolResult(
                        tool_call_id=tc.id,
                        name=tc.name,
                        content=f"stub:{tc.id}",
                    )
                    for tc in tool_calls
                ]

        registry = _registry_with(("t", _Tool(name="t")))
        calls = [
            ToolCall(id="c0", name="t", arguments={}),
            ToolCall(id="c1", name="t", arguments={}),
        ]

        yielded: list[tuple[int, str]] = []
        async for input_idx, result in _Stub().execute_streaming(calls, registry):
            yielded.append((input_idx, result.tool_call_id))

        assert yielded == [(0, "c0"), (1, "c1")]


class TestSequentialExecutorStreaming:
    @pytest.mark.asyncio
    async def test_streaming_yields_in_input_order(self) -> None:
        """SequentialExecutor.execute_streaming yields tuples in input order
        (input order == completion order under sequential semantics)."""
        registry = _registry_with(("t", _Tool(sleep_ms=10, name="t")))
        calls = [ToolCall(id=f"c{i}", name="t", arguments={"i": i}) for i in range(3)]

        yielded: list[tuple[int, str]] = []
        async for input_idx, result in SequentialExecutor().execute_streaming(calls, registry):
            yielded.append((input_idx, result.tool_call_id))

        assert yielded == [(0, "c0"), (1, "c1"), (2, "c2")]


class TestConcurrentExecutorStreaming:
    @pytest.mark.asyncio
    async def test_streaming_empty_input_yields_nothing(self) -> None:
        """Empty ``tool_calls`` is a clean no-op — the runtime loop only
        enters Phase 2 when ``to_execute_calls`` is non-empty, but defensive
        early-return keeps the executor safe to call directly with []."""
        registry = _registry_with(("t", _Tool(name="t")))
        async for _ in ConcurrentExecutor(max_concurrency=5).execute_streaming([], registry):
            raise AssertionError("execute_streaming([]) yielded an item")

    @pytest.mark.asyncio
    async def test_streaming_yields_in_completion_order(self) -> None:
        """Fast tools surface before slow ones — completion-order delivery."""
        registry = _registry_with(
            ("slow", _Tool(sleep_ms=200, name="slow")),
            ("med", _Tool(sleep_ms=100, name="med")),
            ("fast", _Tool(sleep_ms=20, name="fast")),
        )
        calls = [
            ToolCall(id="c-slow", name="slow", arguments={}),
            ToolCall(id="c-med", name="med", arguments={}),
            ToolCall(id="c-fast", name="fast", arguments={}),
        ]

        order: list[str] = []
        async for _input_idx, result in ConcurrentExecutor(max_concurrency=5).execute_streaming(
            calls, registry
        ):
            order.append(result.tool_call_id)

        # fast (20ms) → med (100ms) → slow (200ms), opposite of input order.
        assert order == ["c-fast", "c-med", "c-slow"]

    @pytest.mark.asyncio
    async def test_streaming_respects_max_concurrency(self) -> None:
        """In-flight count never exceeds ``max_concurrency``."""
        import asyncio as _asyncio

        in_flight = 0
        peak = 0
        lock = _asyncio.Lock()

        class _Counting:
            name = "ct"

            async def execute(self, **_kwargs: object) -> str:
                nonlocal in_flight, peak
                async with lock:
                    in_flight += 1
                    peak = max(peak, in_flight)
                try:
                    await _asyncio.sleep(0.05)
                finally:
                    async with lock:
                        in_flight -= 1
                return "ok"

        registry = _registry_with(("ct", _Counting()))  # type: ignore[arg-type]
        calls = [ToolCall(id=f"c{i}", name="ct", arguments={}) for i in range(10)]

        async for _ in ConcurrentExecutor(max_concurrency=3).execute_streaming(calls, registry):
            pass

        assert peak <= 3, f"in-flight peaked at {peak} with max_concurrency=3"

    @pytest.mark.asyncio
    async def test_consumer_break_cancels_in_flight_siblings(self) -> None:
        """Breaking the ``async for`` triggers cleanup that cancels remaining
        tasks. This is the property the runtime loop relies on for
        interrupt-driven sibling cancellation."""
        import asyncio as _asyncio

        executed: list[str] = []

        class _Tracking:
            name = "track"

            def __init__(self, sleep_ms: float, tag: str) -> None:
                self._sleep_ms = sleep_ms
                self._tag = tag

            async def execute(self, **_kwargs: object) -> str:
                await _asyncio.sleep(self._sleep_ms / 1000.0)
                executed.append(self._tag)
                return self._tag

        # First tool finishes fast; siblings take much longer. Consumer breaks
        # after seeing the first → siblings must be cancelled before they get
        # the chance to append to ``executed``.
        lookup = {
            "fast": _Tracking(sleep_ms=20, tag="fast"),
            "slow1": _Tracking(sleep_ms=500, tag="slow1"),
            "slow2": _Tracking(sleep_ms=500, tag="slow2"),
        }

        class _Reg:
            def get(self, name: str) -> object:
                return lookup[name]

        calls = [
            ToolCall(id="c0", name="fast", arguments={}),
            ToolCall(id="c1", name="slow1", arguments={}),
            ToolCall(id="c2", name="slow2", arguments={}),
        ]

        reg: Any = _Reg()
        async for _input_idx, result in ConcurrentExecutor(max_concurrency=5).execute_streaming(
            calls, reg
        ):
            if result.tool_call_id == "c0":
                break

        # Give the event loop one tick to settle cancellations.
        await _asyncio.sleep(0.05)
        assert executed == ["fast"], f"expected siblings cancelled; got {executed}"
