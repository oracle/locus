# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Hook provider protocol and base class for Locus lifecycle hooks.

Includes write-protected event objects for safe hook interaction.
Only explicitly writable fields can be modified — attempting to set
a read-only field raises AttributeError.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from locus.core.state import AgentState


# =============================================================================
# Write-Protected Event Base
# =============================================================================


class ProtectedEvent:
    """Base class for write-protected hook events.

    Subclasses declare _writable as a set of field names that hooks
    may modify. All other attributes are read-only after __init__.

    Setting a read-only field raises AttributeError with a clear message.

    Example:
        class BeforeToolCallEvent(ProtectedEvent):
            _writable = {"arguments", "cancel", "cancel_reason"}

            def __init__(self, tool_name, arguments):
                self._init("tool_name", tool_name)
                self._init("arguments", arguments)
                self._init("cancel", False)
                self._init("cancel_reason", "")
    """

    _writable: set[str] = set()

    def _init(self, name: str, value: Any) -> None:
        """Set a field during __init__ (bypasses protection)."""
        object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        """Only allow setting writable fields."""
        if name.startswith("_") or name in self._writable:
            object.__setattr__(self, name, value)
        else:
            writable = ", ".join(sorted(self._writable)) or "none"
            msg = (
                f"Cannot set '{name}' on {type(self).__name__} — "
                f"read-only. Writable fields: {writable}"
            )
            raise AttributeError(msg)

    def __repr__(self) -> str:
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        pairs = ", ".join(f"{k}={v!r}" for k, v in attrs.items())
        return f"{type(self).__name__}({pairs})"


# =============================================================================
# Hook Events
# =============================================================================


class BeforeModelCallEvent(ProtectedEvent):
    """Event fired before each model.complete() call.

    Writable fields:
        messages: Modify/trim messages before they reach the model.

    Read-only fields:
        tools: Tool schemas (inspect only).

    Example:
        async def on_before_model_call(self, event):
            # Trim to last 10 messages to fit context window
            event.messages = event.messages[-10:]
    """

    _writable = {"messages"}

    # Class-level annotations make the dynamically-set fields visible to
    # mypy. The actual values are bound by ``self._init(...)`` in
    # ``__init__``; ``ProtectedEvent.__setattr__`` enforces the
    # writable / read-only split at runtime.
    messages: list[Any]
    tools: list[Any] | None

    def __init__(self, messages: list[Any], tools: list[Any] | None) -> None:
        self._init("messages", messages)
        self._init("tools", tools)


class AfterModelCallEvent(ProtectedEvent):
    """Event fired after each model.complete() call.

    Writable fields:
        retry: Set True to discard response and re-call the model.
        response: Replace the model response.

    Read-only fields:
        messages: The messages that were sent.

    Example:
        async def on_after_model_call(self, event):
            if not event.response.message.content:
                event.retry = True  # Empty response, retry
    """

    _writable = {"retry", "response"}

    response: Any
    messages: list[Any]
    retry: bool

    def __init__(self, response: Any, messages: list[Any]) -> None:
        self._init("response", response)
        self._init("messages", messages)
        self._init("retry", False)


class BeforeToolCallEvent(ProtectedEvent):
    """Event fired before each tool execution.

    Writable fields:
        arguments: Modify tool arguments.
        cancel: Set True (or a string reason) to skip this tool call.

    Read-only fields:
        tool_name: Name of the tool being called.
        tool_call_id: ID of the tool call.

    Example:
        async def on_before_tool_call(self, event):
            if event.tool_name == "delete_file":
                event.cancel = "Blocked by security policy"
    """

    _writable = {"arguments", "cancel"}

    tool_name: str
    tool_call_id: str
    arguments: dict[str, Any]
    cancel: bool | str

    def __init__(self, tool_name: str, tool_call_id: str, arguments: dict[str, Any]) -> None:
        self._init("tool_name", tool_name)
        self._init("tool_call_id", tool_call_id)
        self._init("arguments", arguments)
        self._init("cancel", False)


class AfterToolCallEvent(ProtectedEvent):
    """Event fired after each tool execution.

    Writable fields:
        retry: Set True to discard result and re-execute the tool.
        result: Replace the tool result.

    Read-only fields:
        tool_name: Name of the tool that was called.
        tool_call_id: ID correlating this event with the matching
            BeforeToolCallEvent. Empty string if not supplied by the
            caller (e.g. tests constructing the event directly).
        arguments: The arguments the tool was invoked with (after any
            BeforeToolCallEvent mutations). Empty dict if not supplied.
        error: Error message (if failed).

    Example:
        async def on_after_tool_call(self, event):
            # Mirror every tool call into an action queue keyed by id.
            self._queue.append({
                "id": event.tool_call_id,
                "tool": event.tool_name,
                "args": event.arguments,
                "result": event.result,
            })
    """

    _writable = {"retry", "result"}

    tool_name: str
    tool_call_id: str
    arguments: dict[str, Any]
    result: Any
    error: str | None
    retry: bool

    def __init__(
        self,
        tool_name: str,
        result: Any,
        error: str | None,
        *,
        tool_call_id: str = "",
        arguments: dict[str, Any] | None = None,
    ) -> None:
        self._init("tool_name", tool_name)
        self._init("tool_call_id", tool_call_id)
        self._init("arguments", arguments if arguments is not None else {})
        self._init("result", result)
        self._init("error", error)
        self._init("retry", False)


class HookPriority:
    """Standard priority ranges for hook ordering.

    Lower priority = earlier execution.

    Ranges:
    - SECURITY (0-99): Security checks, input validation, rate limiting
    - OBSERVABILITY (100-199): Logging, metrics, tracing
    - BUSINESS (200-299): Business logic, custom transformations
    - DEFAULT (300+): General purpose hooks
    """

    SECURITY_MIN = 0
    SECURITY_MAX = 99
    SECURITY_DEFAULT = 50

    OBSERVABILITY_MIN = 100
    OBSERVABILITY_MAX = 199
    OBSERVABILITY_DEFAULT = 150

    BUSINESS_MIN = 200
    BUSINESS_MAX = 299
    BUSINESS_DEFAULT = 250

    DEFAULT = 300


class HookProvider(ABC):
    """Abstract base class for hook providers.

    Hook providers implement lifecycle callbacks that are invoked
    during agent execution. Multiple providers can be registered,
    with execution order determined by priority (lower = earlier).

    Example:
        class MyLoggingHook(HookProvider):
            @property
            def priority(self) -> int:
                return HookPriority.OBSERVABILITY_DEFAULT

            async def on_before_invocation(
                self, prompt: str, state: AgentState
            ) -> AgentState:
                print(f"Starting: {prompt[:50]}...")
                return state

            async def on_after_invocation(
                self, state: AgentState, success: bool
            ) -> None:
                print(f"Completed: success={success}")
    """

    @property
    @abstractmethod
    def priority(self) -> int:
        """Hook priority (lower = earlier execution).

        Use HookPriority constants for standard ranges.
        """
        ...

    @property
    def name(self) -> str:
        """Hook provider name for identification."""
        return self.__class__.__name__

    async def on_before_invocation(
        self,
        prompt: str,
        state: AgentState,
    ) -> AgentState:
        """Called before agent starts processing.

        Args:
            prompt: The user prompt being processed
            state: Current agent state

        Returns:
            Potentially modified agent state
        """
        return state

    async def on_after_invocation(
        self,
        state: AgentState,
        success: bool,
    ) -> None:
        """Called after agent completes processing.

        Args:
            state: Final agent state
            success: Whether execution completed successfully
        """

    async def on_before_tool_call(
        self,
        event: BeforeToolCallEvent,
    ) -> None:
        """Called before tool execution.

        Modify event.arguments to change tool inputs.
        Set event.cancel = True or a string reason to skip execution.
        event.tool_name and event.tool_call_id are read-only.

        Args:
            event: Write-protected event. Writable: arguments, cancel.
        """

    async def on_after_tool_call(
        self,
        event: AfterToolCallEvent,
    ) -> None:
        """Called after tool execution.

        Set event.retry = True to re-execute the tool.
        Set event.result to replace the tool result.
        event.tool_name and event.error are read-only.

        Args:
            event: Write-protected event. Writable: result, retry.
        """

    async def on_iteration_start(
        self,
        iteration: int,
        state: AgentState,
    ) -> None:
        """Called at the start of each agent iteration.

        Args:
            iteration: Current iteration number (0-indexed)
            state: Current agent state
        """

    async def on_iteration_end(
        self,
        iteration: int,
        state: AgentState,
    ) -> None:
        """Called at the end of each agent iteration.

        Args:
            iteration: Current iteration number (0-indexed)
            state: Current agent state
        """

    async def on_before_model_call(
        self,
        event: BeforeModelCallEvent,
    ) -> None:
        """Called before each model.complete() call.

        Modify event.messages to change what the model sees.
        event.tools is read-only (inspect only).

        Args:
            event: Write-protected event. Writable: messages.
        """

    async def on_after_model_call(
        self,
        event: AfterModelCallEvent,
    ) -> None:
        """Called after each model.complete() call.

        Set event.retry = True to discard response and re-call.
        Set event.response to replace the response.
        event.messages is read-only.

        Args:
            event: Write-protected event. Writable: response, retry.
        """

    def register_hooks(self) -> dict[str, bool]:
        """Return which hooks this provider implements.

        Returns:
            Dictionary mapping hook names to whether they are implemented.
            Useful for optimization - registry can skip calling unimplemented hooks.
        """
        return {
            "on_before_invocation": True,
            "on_after_invocation": True,
            "on_before_tool_call": True,
            "on_after_tool_call": True,
            "on_iteration_start": True,
            "on_iteration_end": True,
            "on_before_model_call": True,
            "on_after_model_call": True,
        }
