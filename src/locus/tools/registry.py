# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Tool registry for Locus - 100% Pydantic."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, Field

from locus.tools.decorator import Tool


class ToolRegistry(BaseModel):
    """
    Registry for managing available tools.

    Handles tool registration, lookup, and schema generation.
    """

    tools: dict[str, Tool] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        if tool.name in self.tools:
            msg = f"Tool already registered: {tool.name}"
            raise ValueError(msg)
        self.tools[tool.name] = tool

    def register_many(self, tools: list[Tool]) -> None:
        """Register multiple tools."""
        for tool in tools:
            self.register(tool)

    def unregister(self, name: str) -> Tool | None:
        """Unregister a tool by name."""
        return self.tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self.tools.get(name)

    def get_or_raise(self, name: str) -> Tool:
        """Get a tool by name, raising if not found."""
        tool = self.tools.get(name)
        if tool is None:
            available = list(self.tools.keys())
            msg = f"Tool not found: {name}. Available: {available}"
            raise KeyError(msg)
        return tool

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self.tools.keys())

    def to_openai_schemas(self) -> list[dict[str, Any]]:
        """Get all tools as OpenAI-compatible schemas."""
        return [tool.to_openai_schema() for tool in self.tools.values()]

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self.tools

    def __len__(self) -> int:
        """Number of registered tools."""
        return len(self.tools)

    # Pydantic's BaseModel.__iter__ yields ``(field_name, value)`` tuples
    # for each model field — this override changes the semantic to
    # iterate over registered Tool instances. The Liskov mismatch is
    # intentional and predates strict typing in this module.
    def __iter__(self) -> Iterator[Tool]:  # type: ignore[override]
        """Iterate over tools."""
        return iter(self.tools.values())


def create_registry(*tools: Tool) -> ToolRegistry:
    """Create a registry with the given tools."""
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry
