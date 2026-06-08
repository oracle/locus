# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Lifecycle hooks for Tulip.

This module provides a hook system for observing and modifying
agent behavior at key lifecycle points.

Example:
    from tulip.hooks import HookRegistry, HookProvider, HookPriority
    from tulip.hooks.builtin import LoggingHook, GuardrailsHook

    # Create registry with hooks
    registry = HookRegistry()
    registry.add_provider(GuardrailsHook())  # Priority 50 (security)
    registry.add_provider(LoggingHook())     # Priority 150 (observability)

    # Use in agent
    agent = Agent(
        model="openai:gpt-4o",
        hooks=registry,
    )
"""

from tulip.hooks.events import (
    AfterInvocationEvent,
    BeforeInvocationEvent,
    HookEvent,
    HookResult,
    IterationEndEvent,
    IterationStartEvent,
)
from tulip.hooks.provider import (
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
    HookPriority,
    HookProvider,
    ProtectedEvent,
)
from tulip.hooks.registry import HookRegistry, create_registry


__all__ = [
    # Core classes
    "HookProvider",
    "HookPriority",
    "HookRegistry",
    "ProtectedEvent",
    "create_registry",
    # Events - write-protected (from provider)
    "AfterModelCallEvent",
    "AfterToolCallEvent",
    "BeforeModelCallEvent",
    "BeforeToolCallEvent",
    # Events - info (from events)
    "AfterInvocationEvent",
    "BeforeInvocationEvent",
    "HookEvent",
    "HookResult",
    "IterationEndEvent",
    "IterationStartEvent",
]
