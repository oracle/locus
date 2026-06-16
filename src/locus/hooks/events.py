# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Hook event types and wrappers.

This module re-exports hook events from locus.core.events and provides
additional hook-specific utilities.
"""

from __future__ import annotations

from typing import Any

from locus.core.events import (
    AfterInvocationEvent,
    AfterToolCallEvent,
    BeforeInvocationEvent,
    BeforeToolCallEvent,
    HookEvent,
    LocusEvent,
)


# Re-export all hook events
__all__ = [
    "AfterInvocationEvent",
    "AfterToolCallEvent",
    "BeforeInvocationEvent",
    "BeforeToolCallEvent",
    "HookEvent",
    "HookResult",
    "IterationEndEvent",
    "IterationStartEvent",
    "LocusEvent",
]


class IterationStartEvent(HookEvent):
    """Fired at the start of an agent iteration."""

    event_type: str = "iteration_start"
    iteration: int
    agent_id: str | None = None


class IterationEndEvent(HookEvent):
    """Fired at the end of an agent iteration."""

    event_type: str = "iteration_end"
    iteration: int
    agent_id: str | None = None
    tool_calls_made: int = 0
    confidence: float = 0.0


class HookResult:
    """Container for hook execution results.

    Used to capture results and errors from hook provider execution.
    """

    def __init__(
        self,
        provider_name: str,
        success: bool,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        """Initialize hook result.

        Args:
            provider_name: Name of the hook provider
            success: Whether the hook executed successfully
            result: Return value from the hook (if any)
            error: Error message (if failed)
        """
        self.provider_name = provider_name
        self.success = success
        self.result = result
        self.error = error

    def __repr__(self) -> str:
        """Return string representation."""
        status = "success" if self.success else f"error: {self.error}"
        return f"HookResult(provider={self.provider_name!r}, {status})"
