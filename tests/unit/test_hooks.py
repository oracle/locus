# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for hooks module."""

from unittest.mock import MagicMock

import pytest

from locus.hooks import (
    HookPriority,
    HookProvider,
    HookRegistry,
    HookResult,
    IterationEndEvent,
    IterationStartEvent,
    create_registry,
)


class TestHookPriority:
    """Tests for HookPriority constants."""

    def test_security_range(self):
        """Test security priority range."""
        assert HookPriority.SECURITY_MIN == 0
        assert HookPriority.SECURITY_MAX == 99
        assert HookPriority.SECURITY_DEFAULT == 50

    def test_observability_range(self):
        """Test observability priority range."""
        assert HookPriority.OBSERVABILITY_MIN == 100
        assert HookPriority.OBSERVABILITY_MAX == 199
        assert HookPriority.OBSERVABILITY_DEFAULT == 150

    def test_business_range(self):
        """Test business priority range."""
        assert HookPriority.BUSINESS_MIN == 200
        assert HookPriority.BUSINESS_MAX == 299
        assert HookPriority.BUSINESS_DEFAULT == 250

    def test_default_priority(self):
        """Test default priority."""
        assert HookPriority.DEFAULT == 300


class TestHookResult:
    """Tests for HookResult."""

    def test_create_success_result(self):
        """Test creating successful result."""
        result = HookResult(
            provider_name="TestProvider",
            success=True,
            result={"data": "value"},
        )
        assert result.provider_name == "TestProvider"
        assert result.success is True
        assert result.result == {"data": "value"}
        assert result.error is None

    def test_create_error_result(self):
        """Test creating error result."""
        result = HookResult(
            provider_name="TestProvider",
            success=False,
            error="Something failed",
        )
        assert result.success is False
        assert result.error == "Something failed"

    def test_repr_success(self):
        """Test string representation for success."""
        result = HookResult(provider_name="Test", success=True)
        repr_str = repr(result)
        assert "Test" in repr_str
        assert "success" in repr_str

    def test_repr_error(self):
        """Test string representation for error."""
        result = HookResult(provider_name="Test", success=False, error="oops")
        repr_str = repr(result)
        assert "Test" in repr_str
        assert "error" in repr_str
        assert "oops" in repr_str


class TestIterationEvents:
    """Tests for iteration events."""

    def test_iteration_start_event(self):
        """Test creating iteration start event."""
        event = IterationStartEvent(iteration=5, agent_id="agent1")
        assert event.event_type == "iteration_start"
        assert event.iteration == 5
        assert event.agent_id == "agent1"

    def test_iteration_start_event_no_agent(self):
        """Test iteration start event without agent ID."""
        event = IterationStartEvent(iteration=1)
        assert event.iteration == 1
        assert event.agent_id is None

    def test_iteration_end_event(self):
        """Test creating iteration end event."""
        event = IterationEndEvent(
            iteration=3,
            agent_id="agent1",
            tool_calls_made=5,
            confidence=0.85,
        )
        assert event.event_type == "iteration_end"
        assert event.iteration == 3
        assert event.tool_calls_made == 5
        assert event.confidence == 0.85

    def test_iteration_end_event_defaults(self):
        """Test iteration end event with defaults."""
        event = IterationEndEvent(iteration=1)
        assert event.tool_calls_made == 0
        assert event.confidence == 0.0


class ConcreteHookProvider(HookProvider):
    """Concrete implementation for testing."""

    def __init__(self, priority: int = HookPriority.DEFAULT):
        self._priority = priority

    @property
    def priority(self) -> int:
        return self._priority


class TestHookProvider:
    """Tests for HookProvider base class."""

    def test_provider_name(self):
        """Test provider name is class name."""
        provider = ConcreteHookProvider()
        assert provider.name == "ConcreteHookProvider"

    def test_provider_priority(self):
        """Test provider priority."""
        provider = ConcreteHookProvider(priority=50)
        assert provider.priority == 50

    @pytest.mark.asyncio
    async def test_default_before_invocation(self):
        """Test default before_invocation returns state unchanged."""
        provider = ConcreteHookProvider()
        mock_state = MagicMock()

        result = await provider.on_before_invocation("prompt", mock_state)
        assert result is mock_state

    @pytest.mark.asyncio
    async def test_default_after_invocation(self):
        """Test default after_invocation does nothing."""
        provider = ConcreteHookProvider()
        mock_state = MagicMock()

        # Should not raise
        await provider.on_after_invocation(mock_state, True)

    @pytest.mark.asyncio
    async def test_default_before_tool_call(self):
        """Test default before_tool_call accepts event."""
        from locus.hooks.provider import BeforeToolCallEvent

        provider = ConcreteHookProvider()
        event = BeforeToolCallEvent(tool_name="tool", tool_call_id="c1", arguments={"key": "value"})

        # Should not raise
        await provider.on_before_tool_call(event)

    @pytest.mark.asyncio
    async def test_default_after_tool_call(self):
        """Test default after_tool_call accepts event."""
        from locus.hooks.provider import AfterToolCallEvent

        provider = ConcreteHookProvider()
        event = AfterToolCallEvent(tool_name="tool", result="result", error=None)

        # Should not raise
        await provider.on_after_tool_call(event)

    @pytest.mark.asyncio
    async def test_default_iteration_start(self):
        """Test default iteration_start does nothing."""
        provider = ConcreteHookProvider()
        mock_state = MagicMock()

        # Should not raise
        await provider.on_iteration_start(0, mock_state)

    @pytest.mark.asyncio
    async def test_default_iteration_end(self):
        """Test default iteration_end does nothing."""
        provider = ConcreteHookProvider()
        mock_state = MagicMock()

        # Should not raise
        await provider.on_iteration_end(0, mock_state)

    def test_register_hooks(self):
        """Test register_hooks returns all hooks."""
        provider = ConcreteHookProvider()
        hooks = provider.register_hooks()

        assert hooks["on_before_invocation"] is True
        assert hooks["on_after_invocation"] is True
        assert hooks["on_before_tool_call"] is True
        assert hooks["on_after_tool_call"] is True
        assert hooks["on_iteration_start"] is True
        assert hooks["on_iteration_end"] is True


class TestHookRegistry:
    """Tests for HookRegistry."""

    def test_create_empty_registry(self):
        """Test creating empty registry."""
        registry = HookRegistry()
        assert len(registry) == 0

    def test_add_provider(self):
        """Test adding provider to registry."""
        registry = HookRegistry()
        provider = ConcreteHookProvider()

        registry.add_provider(provider)

        assert len(registry) == 1
        assert "ConcreteHookProvider" in registry

    def test_add_duplicate_provider(self):
        """Test adding duplicate provider raises error."""
        registry = HookRegistry()
        provider1 = ConcreteHookProvider()
        provider2 = ConcreteHookProvider()

        registry.add_provider(provider1)

        with pytest.raises(ValueError, match="already registered"):
            registry.add_provider(provider2)

    def test_remove_provider(self):
        """Test removing provider from registry."""
        registry = HookRegistry()
        provider = ConcreteHookProvider()
        registry.add_provider(provider)

        result = registry.remove_provider("ConcreteHookProvider")

        assert result is True
        assert len(registry) == 0

    def test_remove_nonexistent_provider(self):
        """Test removing nonexistent provider returns False."""
        registry = HookRegistry()

        result = registry.remove_provider("NonexistentProvider")

        assert result is False

    def test_get_provider(self):
        """Test getting provider by name."""
        registry = HookRegistry()
        provider = ConcreteHookProvider()
        registry.add_provider(provider)

        result = registry.get_provider("ConcreteHookProvider")

        assert result is provider

    def test_get_nonexistent_provider(self):
        """Test getting nonexistent provider returns None."""
        registry = HookRegistry()

        result = registry.get_provider("NonexistentProvider")

        assert result is None

    def test_providers_sorted_by_priority(self):
        """Test providers are sorted by priority."""
        registry = HookRegistry()

        class HighPriorityProvider(HookProvider):
            @property
            def priority(self):
                return 100

        class LowPriorityProvider(HookProvider):
            @property
            def priority(self):
                return 10

        registry.add_provider(HighPriorityProvider())
        registry.add_provider(LowPriorityProvider())

        providers = registry.providers
        assert providers[0].priority < providers[1].priority

    def test_contains(self):
        """Test __contains__ method."""
        registry = HookRegistry()
        provider = ConcreteHookProvider()
        registry.add_provider(provider)

        assert "ConcreteHookProvider" in registry
        assert "OtherProvider" not in registry

    @pytest.mark.asyncio
    async def test_emit_before_invocation(self):
        """Test emitting before_invocation event."""
        registry = HookRegistry()

        class TestProvider(HookProvider):
            @property
            def priority(self):
                return 100

            async def on_before_invocation(self, prompt, state):
                state.modified = True
                return state

        registry.add_provider(TestProvider())
        mock_state = MagicMock()

        result = await registry.emit_before_invocation("test prompt", mock_state)

        assert result.modified is True

    @pytest.mark.asyncio
    async def test_emit_after_invocation(self):
        """Test emitting after_invocation event."""
        registry = HookRegistry()
        called = []

        class TestProvider(HookProvider):
            @property
            def priority(self):
                return 100

            async def on_after_invocation(self, state, success):
                called.append(success)

        registry.add_provider(TestProvider())
        mock_state = MagicMock()

        await registry.emit_after_invocation(mock_state, True)

        assert called == [True]

    @pytest.mark.asyncio
    async def test_emit_before_tool_call(self):
        """Test emitting before_tool_call event."""
        registry = HookRegistry()

        class TestProvider(HookProvider):
            @property
            def priority(self):
                return 100

            async def on_before_tool_call(self, event):
                event.arguments["modified"] = True

        registry.add_provider(TestProvider())

        result = await registry.emit_before_tool_call("test_tool", {"x": 1})

        assert result["modified"] is True
        assert result["x"] == 1

    @pytest.mark.asyncio
    async def test_emit_after_tool_call(self):
        """Test emitting after_tool_call event."""
        registry = HookRegistry()
        called = []

        class TestProvider(HookProvider):
            @property
            def priority(self):
                return 100

            async def on_after_tool_call(self, event):
                called.append((event.tool_name, event.result, event.error))

        registry.add_provider(TestProvider())

        await registry.emit_after_tool_call("test_tool", "result", None)

        assert called == [("test_tool", "result", None)]

    @pytest.mark.asyncio
    async def test_emit_iteration_start(self):
        """Test emitting iteration_start event."""
        registry = HookRegistry()
        called = []

        class TestProvider(HookProvider):
            @property
            def priority(self):
                return 100

            async def on_iteration_start(self, iteration, state):
                called.append(iteration)

        registry.add_provider(TestProvider())
        mock_state = MagicMock()

        await registry.emit_iteration_start(5, mock_state)

        assert called == [5]

    @pytest.mark.asyncio
    async def test_emit_iteration_end(self):
        """Test emitting iteration_end event."""
        registry = HookRegistry()
        called = []

        class TestProvider(HookProvider):
            @property
            def priority(self):
                return 100

            async def on_iteration_end(self, iteration, state):
                called.append(iteration)

        registry.add_provider(TestProvider())
        mock_state = MagicMock()

        await registry.emit_iteration_end(3, mock_state)

        assert called == [3]

    @pytest.mark.asyncio
    async def test_emit_generic(self):
        """Test generic emit method."""
        registry = HookRegistry()

        class TestProvider(HookProvider):
            @property
            def priority(self):
                return 100

            async def custom_hook(self, value):
                return value * 2

        registry.add_provider(TestProvider())

        result = await registry.emit("custom_hook", 5)

        assert result == 10

    @pytest.mark.asyncio
    async def test_emit_before_invocation_error_propagates(self):
        """Test that errors in before_invocation propagate."""
        registry = HookRegistry()

        class FailingProvider(HookProvider):
            @property
            def priority(self):
                return 100

            async def on_before_invocation(self, prompt, state):
                raise ValueError("Hook failed")

        registry.add_provider(FailingProvider())
        mock_state = MagicMock()

        with pytest.raises(ValueError, match="Hook failed"):
            await registry.emit_before_invocation("test", mock_state)

    @pytest.mark.asyncio
    async def test_emit_after_invocation_error_wrapped(self):
        """Test that errors in after_invocation are wrapped."""
        registry = HookRegistry()

        class FailingProvider(HookProvider):
            @property
            def priority(self):
                return 100

            async def on_after_invocation(self, state, success):
                raise ValueError("Hook failed")

        registry.add_provider(FailingProvider())
        mock_state = MagicMock()

        with pytest.raises(RuntimeError, match="failed in on_after_invocation"):
            await registry.emit_after_invocation(mock_state, True)


class TestCreateRegistry:
    """Tests for create_registry helper function."""

    def test_create_empty_registry(self):
        """Test creating registry with no providers."""
        registry = create_registry()
        assert len(registry) == 0

    def test_create_registry_with_providers(self):
        """Test creating registry with multiple providers."""

        class Provider1(HookProvider):
            @property
            def priority(self):
                return 50

        class Provider2(HookProvider):
            @property
            def priority(self):
                return 100

        registry = create_registry(Provider1(), Provider2())

        assert len(registry) == 2
        assert "Provider1" in registry
        assert "Provider2" in registry
