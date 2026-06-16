# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Capability metadata layered on top of an existing :class:`ToolRegistry`.

Capabilities are *tools plus a domain/risk overlay* — no parallel tool
storage. :class:`CapabilityIndex` holds a reference to the surrounding
:class:`ToolRegistry` and simply annotates known tools with the
metadata the router needs (domain, risk).

Non-tool capabilities (e.g. a "human-approval" step) get the sentinel
``tool_name="$human"``; they're resolved by the policy gate, not the
registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from locus.router.goal_frame import Risk


if TYPE_CHECKING:
    from locus.tools.decorator import Tool
    from locus.tools.registry import ToolRegistry


HUMAN_SENTINEL = "$human"
"""Sentinel value for ``tool_name`` when the capability is human-only."""


class Capability(BaseModel):
    """A tool (or human action) tagged with router-relevant metadata."""

    id: str = Field(..., description="Stable identifier referenced by GoalFrame.")
    description: str = Field(..., description="Free-text description for the LLM.")
    domain: str = Field(..., description="Domain tag for scoping (e.g. 'observability').")
    risk: Risk = Field(default=Risk.LOW, description="Per-capability risk floor.")
    tool_name: str = Field(
        ...,
        description=(
            "Name of the underlying Tool in ToolRegistry, or ``$human`` for non-tool capabilities."
        ),
    )

    model_config = {"frozen": True}

    @property
    def is_human(self) -> bool:
        """True if this capability is the human-approval sentinel."""
        return self.tool_name == HUMAN_SENTINEL


class CapabilityIndex:
    """View over a :class:`ToolRegistry` that adds router metadata.

    Not a replacement for ``ToolRegistry`` — the underlying ``Tool``
    instances live there. ``CapabilityIndex`` only holds the metadata
    overlay (domain, risk) and resolves capabilities back to tools on
    demand.
    """

    def __init__(self, tools: ToolRegistry) -> None:
        self._tools = tools
        self._caps: dict[str, Capability] = {}

    def annotate(
        self,
        cap_id: str,
        *,
        tool_name: str,
        description: str,
        domain: str,
        risk: Risk = Risk.LOW,
    ) -> Capability:
        """Register router metadata for an existing tool (or a human step).

        ``tool_name`` must already exist in the underlying
        :class:`ToolRegistry`, except for the ``$human`` sentinel which is
        treated as a non-tool capability.
        """
        if tool_name != HUMAN_SENTINEL and tool_name not in self._tools:
            available = sorted(self._tools.list_tools())
            raise KeyError(
                f"Capability {cap_id!r} references unknown tool {tool_name!r}. "
                f"Register the tool first. Available: {available}",
            )
        if cap_id in self._caps:
            raise ValueError(f"Capability already registered: {cap_id!r}")
        cap = Capability(
            id=cap_id,
            description=description,
            domain=domain,
            risk=risk,
            tool_name=tool_name,
        )
        self._caps[cap_id] = cap
        return cap

    def lookup(self, ids: list[str]) -> list[Capability]:
        """Resolve a list of capability ids; raises if any are missing."""
        missing = [i for i in ids if i not in self._caps]
        if missing:
            available = sorted(self._caps.keys())
            raise KeyError(
                f"Unknown capability ids: {missing}. Available: {available}",
            )
        return [self._caps[i] for i in ids]

    def for_domain(self, domain: str) -> list[Capability]:
        """All capabilities tagged with ``domain``."""
        return [c for c in self._caps.values() if c.domain == domain]

    def all(self) -> list[Capability]:
        """All registered capabilities, in registration order."""
        return list(self._caps.values())

    def resolve_tool(self, cap: Capability) -> Tool:
        """Return the underlying :class:`Tool`. Raises for human capabilities."""
        if cap.is_human:
            raise ValueError(
                f"Capability {cap.id!r} is a human-approval step, not a Tool.",
            )
        return self._tools.get_or_raise(cap.tool_name)

    def __contains__(self, cap_id: str) -> bool:
        return cap_id in self._caps

    def __len__(self) -> int:
        return len(self._caps)
