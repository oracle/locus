# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for hooks registry."""

import pytest

from locus.core.state import AgentState
from locus.hooks.provider import HookProvider
from locus.hooks.registry import HookRegistry, create_registry


class MockHookProvider(HookProvider):
    """Mock hook provider for testing."""

    def __init__(self, name: str = "mock", priority: int = 100):
        self._name = name
        self._priority = priority
        self.before_invocation_called = False
        self.after_invocation_called = False
        self.before_tool_called = False
        self.after_tool_called = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    async def on_before_invocation(self, prompt, state):
        self.before_invocation_called = True
        return state

    async def on_after_invocation(self, state, success):
        self.after_invocation_called = True

    async def on_before_tool_call(self, event):
        self.before_tool_called = True

    async def on_after_tool_call(self, event):
        self.after_tool_called = True

    async def on_iteration_start(self, iteration, state):
        pass

    async def on_iteration_end(self, iteration, state):
        pass


class FailingHookProvider(HookProvider):
    """Hook provider that raises errors."""

    @property
    def name(self) -> str:
        return "failing"

    @property
    def priority(self) -> int:
        return 100

    async def on_before_invocation(self, prompt, state):
        return state

    async def on_after_invocation(self, state, success):
        pass

    async def on_before_tool_call(self, event):
        raise ValueError("Before tool error")

    async def on_after_tool_call(self, event):
        raise ValueError("After tool error")

    async def on_iteration_start(self, iteration, state):
        pass

    async def on_iteration_end(self, iteration, state):
        pass


class TestHookRegistry:
    """Tests for HookRegistry."""

    @pytest.fixture
    def registry(self):
        """Create empty registry."""
        return HookRegistry()

    @pytest.fixture
    def mock_provider(self):
        """Create mock provider."""
        return MockHookProvider()

    def test_add_provider(self, registry, mock_provider):
        """Add a hook provider."""
        registry.add_provider(mock_provider)
        assert len(registry._providers) == 1

    def test_add_duplicate_provider_raises(self, registry, mock_provider):
        """Adding duplicate provider raises ValueError."""
        registry.add_provider(mock_provider)
        with pytest.raises(ValueError, match="already registered"):
            registry.add_provider(mock_provider)

    def test_remove_provider(self, registry, mock_provider):
        """Remove a hook provider by name."""
        registry.add_provider(mock_provider)
        result = registry.remove_provider("mock")
        assert result is True
        assert len(registry._providers) == 0

    def test_remove_nonexistent_provider(self, registry):
        """Remove nonexistent provider returns False."""
        result = registry.remove_provider("nonexistent")
        assert result is False

    def test_get_provider(self, registry, mock_provider):
        """Get provider by name."""
        registry.add_provider(mock_provider)
        found = registry.get_provider("mock")
        assert found is mock_provider

    def test_get_nonexistent_provider(self, registry):
        """Get nonexistent provider returns None."""
        found = registry.get_provider("nonexistent")
        assert found is None

    def test_providers_sorted_by_priority(self, registry):
        """Providers are sorted by priority (ascending)."""
        low = MockHookProvider("low", priority=10)
        high = MockHookProvider("high", priority=200)
        medium = MockHookProvider("medium", priority=100)

        registry.add_provider(low)
        registry.add_provider(high)
        registry.add_provider(medium)

        # Get sorted providers
        sorted_providers = registry.providers

        # Lower priority comes first (ascending order)
        assert sorted_providers[0] is low
        assert sorted_providers[1] is medium
        assert sorted_providers[2] is high

    def test_len(self, registry, mock_provider):
        """Test __len__."""
        assert len(registry) == 0
        registry.add_provider(mock_provider)
        assert len(registry) == 1

    def test_contains(self, registry, mock_provider):
        """Test __contains__."""
        assert "mock" not in registry
        registry.add_provider(mock_provider)
        assert "mock" in registry

    @pytest.mark.asyncio
    async def test_emit_before_invocation(self, registry, mock_provider):
        """Emit before_invocation to all providers."""
        registry.add_provider(mock_provider)
        state = AgentState()

        result = await registry.emit_before_invocation("test prompt", state)

        assert mock_provider.before_invocation_called
        assert result is state

    @pytest.mark.asyncio
    async def test_emit_after_invocation(self, registry, mock_provider):
        """Emit after_invocation to all providers."""
        registry.add_provider(mock_provider)
        state = AgentState()

        await registry.emit_after_invocation(state, success=True)

        assert mock_provider.after_invocation_called

    @pytest.mark.asyncio
    async def test_emit_before_tool_call(self, registry, mock_provider):
        """Emit before_tool_call to all providers."""
        registry.add_provider(mock_provider)

        result = await registry.emit_before_tool_call("test_tool", {"arg": "value"})

        assert mock_provider.before_tool_called
        assert result == {"arg": "value"}

    @pytest.mark.asyncio
    async def test_emit_after_tool_call(self, registry, mock_provider):
        """Emit after_tool_call to all providers."""
        registry.add_provider(mock_provider)

        await registry.emit_after_tool_call("test_tool", "result", None)

        assert mock_provider.after_tool_called

    @pytest.mark.asyncio
    async def test_emit_before_tool_call_error(self, registry):
        """Error in before_tool_call is propagated."""
        failing = FailingHookProvider()
        registry.add_provider(failing)

        with pytest.raises(ValueError, match="Before tool error"):
            await registry.emit_before_tool_call("test_tool", {})

    @pytest.mark.asyncio
    async def test_emit_after_tool_call_error(self, registry):
        """Error in after_tool_call is collected and raised."""
        failing = FailingHookProvider()
        registry.add_provider(failing)

        with pytest.raises(RuntimeError, match="failed in on_after_tool_call"):
            await registry.emit_after_tool_call("test_tool", "result", None)

    @pytest.mark.asyncio
    async def test_emit_iteration_start(self, registry, mock_provider):
        """Emit iteration_start to all providers."""
        registry.add_provider(mock_provider)
        state = AgentState()

        await registry.emit_iteration_start(1, state)

    @pytest.mark.asyncio
    async def test_emit_iteration_end(self, registry, mock_provider):
        """Emit iteration_end to all providers."""
        registry.add_provider(mock_provider)
        state = AgentState()

        await registry.emit_iteration_end(1, state)

    @pytest.mark.asyncio
    async def test_emit_generic_event(self, registry, mock_provider):
        """Emit generic event through dynamic dispatch."""
        registry.add_provider(mock_provider)
        state = AgentState()

        # This uses the emit() method for arbitrary events
        result = await registry.emit("on_before_invocation", "test", state)

        assert result is state


class TestCreateRegistry:
    """Tests for create_registry helper."""

    def test_create_empty(self):
        """Create empty registry."""
        registry = create_registry()
        assert len(registry._providers) == 0

    def test_create_with_providers(self):
        """Create registry with providers."""
        p1 = MockHookProvider("p1")
        p2 = MockHookProvider("p2")

        registry = create_registry(p1, p2)

        assert len(registry._providers) == 2
