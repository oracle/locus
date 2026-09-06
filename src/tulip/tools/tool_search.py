# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""The ``tool_search`` builtin — deferred tool loading (#177).

Tool definitions dominate the window on MCP-heavy agents; the field measured
~85% token reduction from loading schemas on demand. Tulip's version keeps the
control plane intact: deferral changes *visibility only*. A deferred tool is
registered, gated, labelled and sandbox-checked from the start — ``tool_search``
merely surfaces its schema to the next model call, and the search itself goes
through the ordinary tool seam, so hooks see what the model went looking for
(that is signal an audit wants).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from tulip.tools.decorator import Tool, tool


if TYPE_CHECKING:
    from tulip.tools.registry import ToolRegistry


def deferred_catalog_note(registry: ToolRegistry) -> str:
    """One system-prompt line telling the model the catalog exists.

    Without it the model has no reason to search: a deferred tool it was
    never told about is indistinguishable from a tool that does not exist.
    """
    pending = registry.deferred_pending()
    if not pending:
        return ""
    return (
        f"\n\nBeyond the tools listed, {len(pending)} more are available on "
        "demand. Call tool_search with keywords describing what you need "
        "(e.g. 'refund payment', 'dns lookup') to load them; matching tools "
        "become callable on your next turn."
    )


def create_tool_search_tool(registry: ToolRegistry) -> Tool:
    """Build the ``tool_search`` tool bound to ``registry``.

    Auto-registered by the agent initializer whenever any registered tool is
    deferred. Matching tools are activated as a side effect — their schemas
    join the next model call.
    """

    @tool(
        name="tool_search",
        description=(
            "Search the catalog of not-yet-loaded tools by keyword and load "
            "the matches. Returns the tools now available; call them on your "
            "next turn. Use when no visible tool fits the task."
        ),
    )
    def tool_search(query: str) -> str:
        matches = registry.search(query)
        if not matches:
            remaining = len(registry.deferred_pending())
            return json.dumps(
                {
                    "matches": [],
                    "note": (
                        f"No tools matched {query!r}. {remaining} unloaded "
                        "tool(s) remain — try different keywords."
                    ),
                }
            )
        loaded = []
        for match in matches:
            registry.activate(match.name)
            loaded.append({"name": match.name, "description": match.description})
        return json.dumps(
            {
                "matches": loaded,
                "note": "These tools are now loaded and callable from your next turn.",
            }
        )

    return tool_search
