# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for the hooks Plugin base class and PluginAdapter."""

from __future__ import annotations

import pytest

from locus.hooks.plugin import Plugin, PluginAdapter, hook
from locus.tools.decorator import tool


class _StubPlugin(Plugin):
    """Concrete plugin used to exercise the discovery helpers."""

    @property
    def name(self) -> str:
        return "stub"

    @hook
    async def on_before_model_call(self, event: object) -> None:
        return None

    @hook
    async def on_after_tool_call(self, event: object) -> None:
        return None

    async def not_a_hook(self, event: object) -> None:
        """Plain method — must be ignored by ``get_hooks``."""
        return

    @tool(name="stub_tool", description="A tool exposed by the plugin.")
    def my_tool(self, message: str) -> str:
        return f"echoed: {message}"


class TestHookDecorator:
    """The ``@hook`` decorator marks methods for discovery."""

    def test_decorator_marks_method(self) -> None:
        @hook
        def my_method() -> None:
            return None

        assert getattr(my_method, "_is_hook", False) is True

    def test_decorator_returns_same_function(self) -> None:
        def my_method() -> None:
            return None

        assert hook(my_method) is my_method


class TestPluginCannotInstantiateAbstract:
    """Plugin is ABC — must implement ``name``."""

    def test_missing_name_raises(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            Plugin()  # type: ignore[abstract]


class TestPluginGetHooks:
    """``get_hooks`` discovers ``@hook``-decorated methods."""

    def test_discovers_decorated_methods(self) -> None:
        plugin = _StubPlugin()
        hooks = plugin.get_hooks()

        assert "on_before_model_call" in hooks
        assert "on_after_tool_call" in hooks

    def test_skips_undecorated_methods(self) -> None:
        plugin = _StubPlugin()
        hooks = plugin.get_hooks()

        assert "not_a_hook" not in hooks

    def test_skips_dunder_attrs(self) -> None:
        plugin = _StubPlugin()
        hooks = plugin.get_hooks()
        for name in hooks:
            assert not name.startswith("_")

    def test_returns_bound_methods(self) -> None:
        """The discovered hooks must be bound to the plugin instance."""
        plugin = _StubPlugin()
        hooks = plugin.get_hooks()
        # Bound methods carry the plugin as ``__self__``.
        assert hooks["on_before_model_call"].__self__ is plugin


class TestPluginGetTools:
    """``get_tools`` discovers ``@tool``-decorated methods."""

    def test_discovers_tool(self) -> None:
        plugin = _StubPlugin()
        tools = plugin.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "stub_tool"

    def test_returns_empty_when_no_tools(self) -> None:
        class NoToolsPlugin(Plugin):
            @property
            def name(self) -> str:
                return "no_tools"

        plugin = NoToolsPlugin()
        assert plugin.get_tools() == []


class TestPluginInitAgent:
    """``init_agent`` is a hook for plugins that need the agent reference."""

    def test_default_init_is_noop(self) -> None:
        plugin = _StubPlugin()
        # Default implementation must not raise; agent reference is opaque.
        plugin.init_agent(agent=object())


class TestPluginAdapter:
    """``PluginAdapter`` exposes plugin hooks as attribute access."""

    def test_attribute_access_returns_hook(self) -> None:
        plugin = _StubPlugin()
        adapter = PluginAdapter(plugin)
        # Bound methods compare equal but are not ``is``-identical
        # (Python re-binds ``__get__`` on each descriptor access).
        assert adapter.on_before_model_call == plugin.on_before_model_call
        # Same function under the hood.
        assert adapter.on_before_model_call.__func__ is _StubPlugin.on_before_model_call

    def test_missing_hook_raises_attribute_error(self) -> None:
        plugin = _StubPlugin()
        adapter = PluginAdapter(plugin)
        with pytest.raises(AttributeError, match="has no hook 'on_iteration_start'"):
            _ = adapter.on_iteration_start

    def test_repr_includes_plugin_name_and_hooks(self) -> None:
        plugin = _StubPlugin()
        adapter = PluginAdapter(plugin)
        rep = repr(adapter)
        assert "stub" in rep
        assert "on_before_model_call" in rep
