# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Hook registry for managing lifecycle hook providers."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from locus.hooks.provider import HookProvider


if TYPE_CHECKING:
    from locus.core.state import AgentState


logger = logging.getLogger(__name__)


class HookRegistry:
    """Registry for managing hook providers.

    The registry maintains a priority-ordered list of hook providers
    and dispatches lifecycle events to them in order.

    Example:
        registry = HookRegistry()
        registry.add_provider(LoggingHook())
        registry.add_provider(GuardrailsHook())

        # During agent execution
        state = await registry.emit_before_invocation(prompt, state)
        # ... agent runs ...
        await registry.emit_after_invocation(state, success=True)
    """

    def __init__(self) -> None:
        """Initialize empty hook registry."""
        self._providers: list[HookProvider] = []
        self._sorted = True

    def add_provider(self, provider: HookProvider) -> None:
        """Register a hook provider.

        Args:
            provider: Hook provider to register

        Raises:
            ValueError: If provider with same name already registered
        """
        for existing in self._providers:
            if existing.name == provider.name:
                msg = f"Hook provider '{provider.name}' already registered"
                raise ValueError(msg)

        self._providers.append(provider)
        self._sorted = False
        logger.debug(
            "Registered hook provider '%s' with priority %d",
            provider.name,
            provider.priority,
        )

    def remove_provider(self, name: str) -> bool:
        """Remove a hook provider by name.

        Args:
            name: Name of the provider to remove

        Returns:
            True if provider was removed, False if not found
        """
        for i, provider in enumerate(self._providers):
            if provider.name == name:
                self._providers.pop(i)
                logger.debug("Removed hook provider '%s'", name)
                return True
        return False

    def get_provider(self, name: str) -> HookProvider | None:
        """Get a hook provider by name.

        Args:
            name: Name of the provider to find

        Returns:
            The provider if found, None otherwise
        """
        for provider in self._providers:
            if provider.name == name:
                return provider
        return None

    def _ensure_sorted(self) -> None:
        """Ensure providers are sorted by priority."""
        if not self._sorted:
            self._providers.sort(key=lambda p: p.priority)
            self._sorted = True

    @property
    def providers(self) -> list[HookProvider]:
        """Get all registered providers in priority order."""
        self._ensure_sorted()
        return list(self._providers)

    def __len__(self) -> int:
        """Return number of registered providers."""
        return len(self._providers)

    def __contains__(self, name: str) -> bool:
        """Check if a provider with given name is registered."""
        return any(p.name == name for p in self._providers)

    # =========================================================================
    # Event Emission
    # =========================================================================

    async def emit_before_invocation(
        self,
        prompt: str,
        state: AgentState,
    ) -> AgentState:
        """Emit before_invocation event to all providers.

        Args:
            prompt: User prompt being processed
            state: Current agent state

        Returns:
            Potentially modified agent state
        """
        self._ensure_sorted()
        for provider in self._providers:
            try:
                state = await provider.on_before_invocation(prompt, state)
            except Exception:
                logger.exception(
                    "Error in hook provider '%s' on_before_invocation",
                    provider.name,
                )
                raise
        return state

    async def emit_after_invocation(
        self,
        state: AgentState,
        success: bool,
    ) -> None:
        """Emit after_invocation event to all providers.

        Args:
            state: Final agent state
            success: Whether execution completed successfully
        """
        self._ensure_sorted()
        errors: list[tuple[str, Exception]] = []
        # Reverse order: last-registered-first for proper teardown
        for provider in reversed(self._providers):
            try:
                await provider.on_after_invocation(state, success)
            except Exception as e:
                logger.exception(
                    "Error in hook provider '%s' on_after_invocation",
                    provider.name,
                )
                errors.append((provider.name, e))

        # Re-raise first error if any occurred
        if errors:
            name, error = errors[0]
            msg = f"Hook provider '{name}' failed in on_after_invocation: {error}"
            raise RuntimeError(msg) from error

    async def emit_before_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Emit before_tool_call event to all providers.

        Args:
            tool_name: Name of the tool being called
            arguments: Tool arguments

        Returns:
            Potentially modified arguments
        """
        from locus.hooks.provider import BeforeToolCallEvent

        self._ensure_sorted()
        event = BeforeToolCallEvent(tool_name=tool_name, tool_call_id="", arguments=arguments)
        for provider in self._providers:
            try:
                await provider.on_before_tool_call(event)
            except Exception:
                logger.exception(
                    "Error in hook provider '%s' on_before_tool_call",
                    provider.name,
                )
                raise
        modified_arguments: dict[str, Any] = event.arguments
        return modified_arguments

    async def emit_after_tool_call(
        self,
        tool_name: str,
        result: Any,
        error: str | None,
        *,
        tool_call_id: str = "",
        arguments: dict[str, Any] | None = None,
    ) -> None:
        """Emit after_tool_call event to all providers.

        Args:
            tool_name: Name of the tool that was called
            result: Tool result (if successful)
            error: Error message (if failed)
            tool_call_id: ID of the tool call (correlates with BeforeToolCallEvent).
            arguments: Arguments the tool was invoked with (post-hook mutation).
        """
        from locus.hooks.provider import AfterToolCallEvent

        self._ensure_sorted()
        event = AfterToolCallEvent(
            tool_name=tool_name,
            result=result,
            error=error,
            tool_call_id=tool_call_id,
            arguments=arguments,
        )
        errors: list[tuple[str, Exception]] = []
        # Reverse order for proper teardown
        for provider in reversed(self._providers):
            try:
                await provider.on_after_tool_call(event)
            except Exception as e:
                logger.exception(
                    "Error in hook provider '%s' on_after_tool_call",
                    provider.name,
                )
                errors.append((provider.name, e))

        if errors:
            name, error_exc = errors[0]
            msg = f"Hook provider '{name}' failed in on_after_tool_call: {error_exc}"
            raise RuntimeError(msg) from error_exc

    async def emit_iteration_start(
        self,
        iteration: int,
        state: AgentState,
    ) -> None:
        """Emit iteration_start event to all providers.

        Args:
            iteration: Current iteration number
            state: Current agent state
        """
        self._ensure_sorted()
        tasks = [provider.on_iteration_start(iteration, state) for provider in self._providers]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def emit_iteration_end(
        self,
        iteration: int,
        state: AgentState,
    ) -> None:
        """Emit iteration_end event to all providers.

        Args:
            iteration: Current iteration number
            state: Current agent state
        """
        self._ensure_sorted()
        tasks = [provider.on_iteration_end(iteration, state) for provider in self._providers]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def emit_before_model_call(
        self,
        messages: list[Any],
        tools: list[dict[str, Any]] | None,
    ) -> list[Any]:
        """Emit before_model_call event to all providers.

        Args:
            messages: Messages about to be sent to the model
            tools: Tool schemas (if any)

        Returns:
            Potentially modified messages list
        """
        from locus.hooks.provider import BeforeModelCallEvent

        self._ensure_sorted()
        event = BeforeModelCallEvent(messages=messages, tools=tools)
        for provider in self._providers:
            try:
                await provider.on_before_model_call(event)
            except Exception:
                logger.exception(
                    "Error in hook provider '%s' on_before_model_call",
                    provider.name,
                )
                raise
        messages_out: list[Any] = event.messages
        return messages_out

    async def emit_after_model_call(
        self,
        response: Any,
        messages: list[Any],
    ) -> Any:
        """Emit after_model_call event to all providers.

        Args:
            response: The ModelResponse from the model
            messages: The messages that were sent

        Returns:
            Potentially modified response
        """
        from locus.hooks.provider import AfterModelCallEvent

        self._ensure_sorted()
        event = AfterModelCallEvent(response=response, messages=messages)
        # Reverse order for proper teardown
        for provider in reversed(self._providers):
            try:
                await provider.on_after_model_call(event)
            except Exception:
                logger.exception(
                    "Error in hook provider '%s' on_after_model_call",
                    provider.name,
                )
                raise
        return event.response

    async def emit(
        self,
        event_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Generic event emission for custom hook points.

        Args:
            event_name: Name of the hook method to call
            *args: Positional arguments to pass
            **kwargs: Keyword arguments to pass

        Returns:
            Result from the last provider that returned a non-None value
        """
        self._ensure_sorted()
        result = None
        for provider in self._providers:
            method = getattr(provider, event_name, None)
            if method is not None and callable(method):
                try:
                    ret = await method(*args, **kwargs)
                    if ret is not None:
                        result = ret
                except Exception:
                    logger.exception(
                        "Error in hook provider '%s' %s",
                        provider.name,
                        event_name,
                    )
                    raise
        return result


def create_registry(*providers: HookProvider) -> HookRegistry:
    """Create a registry with the given providers.

    Args:
        *providers: Hook providers to register

    Returns:
        New HookRegistry with all providers registered

    Example:
        registry = create_registry(
            LoggingHook(),
            TelemetryHook(),
            GuardrailsHook(),
        )
    """
    registry = HookRegistry()
    for provider in providers:
        registry.add_provider(provider)
    return registry
