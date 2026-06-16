# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Plugin system for composable agent extensions.

Plugins bundle hooks and tools into a single reusable unit.
Methods decorated with @hook are auto-discovered and registered.
Methods decorated with @tool are auto-discovered and added to the agent.

Example:
    from locus.hooks.plugin import Plugin, hook

    class LoggingPlugin(Plugin):
        name = "logging"

        @hook
        async def on_before_model_call(self, event):
            print(f"Calling model with {len(event.messages)} messages")

        @hook
        async def on_after_tool_call(self, event):
            print(f"Tool {event.tool_name} returned {len(event.result or '')} chars")

    agent = Agent(config=AgentConfig(
        model=model,
        plugins=[LoggingPlugin()],
    ))
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


def hook(fn: Any) -> Any:
    """Mark a method as a hook callback.

    The event type is inferred from the method name (on_before_model_call,
    on_after_tool_call, etc.) or from the type annotation of the first
    parameter.
    """
    fn._is_hook = True
    return fn


class Plugin(ABC):
    """Base class for composable agent plugins.

    Plugins bundle related hooks and tools into a single unit.
    All methods decorated with @hook are auto-discovered and registered
    as hook callbacks when the plugin is attached to an agent.

    Subclasses must define a `name` property.

    Example:
        class MyPlugin(Plugin):
            name = "my_plugin"

            @hook
            async def on_before_model_call(self, event):
                event.messages = event.messages[-10:]  # Trim context

            @hook
            async def on_after_tool_call(self, event):
                if event.error:
                    event.retry = True  # Auto-retry failures
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name for identification."""
        ...

    def get_hooks(self) -> dict[str, Any]:
        """Discover all @hook decorated methods.

        Returns:
            Dict mapping hook method names to bound methods.
        """
        hooks: dict[str, Any] = {}
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr = getattr(self, attr_name, None)
            if attr is not None and callable(attr) and getattr(attr, "_is_hook", False):
                hooks[attr_name] = attr
        return hooks

    def get_tools(self) -> list[Any]:
        """Discover all @tool decorated methods.

        Returns:
            List of Tool instances found on the plugin.
        """
        from locus.tools.decorator import Tool

        tools: list[Any] = []
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr = getattr(self, attr_name, None)
            if isinstance(attr, Tool):
                tools.append(attr)
        return tools

    def init_agent(self, agent: Any) -> None:
        """Called when plugin is attached to an agent.

        Override to perform setup that requires the agent instance.

        Args:
            agent: The agent this plugin is being attached to.
        """


class PluginAdapter:
    """Adapts a Plugin into hook callbacks compatible with the agent's hook system.

    The agent stores hooks as a list of objects with on_before_model_call,
    on_after_tool_call, etc. methods. This adapter wraps a Plugin's
    @hook methods to match that interface.
    """

    def __init__(self, plugin: Plugin) -> None:
        self._plugin = plugin
        self._hooks = plugin.get_hooks()

    def __getattr__(self, name: str) -> Any:
        if name in self._hooks:
            return self._hooks[name]
        raise AttributeError(f"Plugin '{self._plugin.name}' has no hook '{name}'")

    def __repr__(self) -> str:
        return f"PluginAdapter({self._plugin.name}, hooks={list(self._hooks.keys())})"
