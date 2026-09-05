# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Tool registry for Tulip - 100% Pydantic."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, Field

from tulip.tools.decorator import Tool


class ToolRegistry(BaseModel):
    """
    Registry for managing available tools.

    Handles tool registration, lookup, and schema generation.
    """

    tools: dict[str, Tool] = Field(default_factory=dict)

    activated: set[str] = Field(default_factory=set)
    """Names of deferred tools whose schemas have been surfaced to the model
    (via ``tool_search`` or :meth:`activate`). Deferral is visibility only:
    an unactivated deferred tool is still registered, still executable, and
    still gated — it just costs no context until the model asks for it."""

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
        """Schemas for the tools the model may currently see.

        Eager tools always; deferred tools only once activated. This is the
        single choke point the run loop reads, so deferral needs no loop
        changes — an unactivated deferred tool simply never reaches the
        request payload (#177).
        """
        return [
            tool.to_openai_schema()
            for tool in self.tools.values()
            if not tool.deferred or tool.name in self.activated
        ]

    def deferred_pending(self) -> list[Tool]:
        """Deferred tools whose schemas the model has not been shown yet."""
        return [t for t in self.tools.values() if t.deferred and t.name not in self.activated]

    def activate(self, name: str) -> Tool:
        """Surface a deferred tool's schema to subsequent model calls.

        Activation changes visibility and nothing else — gating, labels and
        sandbox requirements ride on the Tool object and were in force the
        whole time. Raises ``KeyError`` for an unknown name; activating an
        eager or already-active tool is a no-op.
        """
        tool = self.get_or_raise(name)
        self.activated.add(name)
        return tool

    def search(self, query: str, *, limit: int = 5) -> list[Tool]:
        """Rank deferred, not-yet-activated tools against a keyword query.

        Deliberately boring lexical scoring (token overlap on name and
        description, substring bonus): deterministic, offline, and cheap —
        the model supplies the intelligence; this just narrows the catalog.
        """
        terms = [t for t in query.lower().split() if t]
        if not terms:
            return []
        scored: list[tuple[float, str, Tool]] = []
        for candidate in self.deferred_pending():
            haystack = f"{candidate.name} {candidate.description}".lower()
            name_l = candidate.name.lower()
            score = 0.0
            for term in terms:
                if term in name_l:
                    score += 3.0
                elif term in haystack:
                    score += 1.0
            if score > 0:
                scored.append((score, candidate.name, candidate))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [t for _, _, t in scored[:limit]]

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
