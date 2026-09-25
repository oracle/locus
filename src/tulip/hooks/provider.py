# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Hook provider protocol and base class for Tulip lifecycle hooks.

Includes write-protected event objects for safe hook interaction.
Only explicitly writable fields can be modified — attempting to set
a read-only field raises AttributeError.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from tulip.core.events import CustomEvent, RunInfo
    from tulip.core.state import AgentState


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

    #: Fields whose values never appear in ``repr()`` (only their keys do).
    _redacted: frozenset[str] = frozenset()

    #: The run this event belongs to; bound by the runtime, ``None`` when the
    #: event was constructed by hand (tests, custom dispatchers).
    _run_ctx: Any = None

    @property
    def run(self) -> RunInfo | None:
        """Read-only identity of the run this hook fired in.

        ``run.thread_id``, ``run.run_id`` and ``run.metadata`` (the run's
        invocation metadata, read-only) let a hook on a shared agent tell
        concurrent users apart without globals or context variables.
        ``None`` when the event was not dispatched by a running agent.
        """
        ctx = self._run_ctx
        info: RunInfo | None = ctx.info if ctx is not None else None
        return info

    def emit(self, event: CustomEvent) -> None:
        """Send a UI-only :class:`~tulip.core.events.CustomEvent` to the run's stream.

        The event is yielded from ``Agent.run()`` right after this hook
        returns. It never reaches the model, the conversation, or the
        checkpoint — use it for widgets and progress, and ``event.result`` for
        what the model should read.

        Raises:
            RuntimeError: The event is not bound to a run.
            TypeError: ``event`` is not a ``CustomEvent``.
        """
        ctx = self._run_ctx
        if ctx is None:
            raise RuntimeError(
                f"{type(self).__name__}.emit() needs a running agent; this event "
                "was not dispatched by one"
            )
        ctx.emit(event, tool_call_id=getattr(self, "tool_call_id", None) or None)

    def _bind_run(self, run: Any) -> None:
        """Attach the runtime's per-run context (internal)."""
        object.__setattr__(self, "_run_ctx", run)

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
        for name in self._redacted:
            value = attrs.get(name)
            if isinstance(value, dict):
                attrs[name] = dict.fromkeys(value, "***")
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

    def __init__(self, messages: list[Any], tools: list[Any] | None, *, run: Any = None) -> None:
        self._init("messages", messages)
        self._init("tools", tools)
        self._bind_run(run)


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

    def __init__(self, response: Any, messages: list[Any], *, run: Any = None) -> None:
        self._init("response", response)
        self._init("messages", messages)
        self._init("retry", False)
        self._bind_run(run)


class BeforeToolCallEvent(ProtectedEvent):
    """Event fired before each tool execution.

    Writable fields:
        arguments: Modify tool arguments. Modified arguments are what the
            run records: they are checkpointed with the call
            (``state.tool_executions``) and passed to ``on_after_tool_call``.
        secret_arguments: Extra arguments merged over ``arguments`` for the
            tool invocation ONLY — never checkpointed, never written to the
            conversation or the event stream, never shown to after-hooks. Use
            it for anything that must reach the tool but must not be
            persisted: a confirmation token, a per-user credential.
        cancel: Set True (or a string reason) to skip this tool call.

    Read-only fields:
        tool_name: Name of the tool being called.
        tool_call_id: ID of the tool call.

    Example:
        async def on_before_tool_call(self, event):
            if event.tool_name == "delete_file":
                event.cancel = "Blocked by security policy"
            if event.tool_name == "book":
                event.secret_arguments = {"confirm_token": mint_token(event.run)}
    """

    _writable = {"arguments", "secret_arguments", "cancel"}
    _redacted = frozenset({"secret_arguments"})

    tool_name: str
    tool_call_id: str
    arguments: dict[str, Any]
    secret_arguments: dict[str, Any]
    cancel: bool | str

    def __init__(
        self,
        tool_name: str,
        tool_call_id: str,
        arguments: dict[str, Any],
        *,
        run: Any = None,
    ) -> None:
        self._init("tool_name", tool_name)
        self._init("tool_call_id", tool_call_id)
        self._init("arguments", arguments)
        self._init("secret_arguments", {})
        self._init("cancel", False)
        self._bind_run(run)


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
        run: Any = None,
    ) -> None:
        self._init("tool_name", tool_name)
        self._init("tool_call_id", tool_call_id)
        self._init("arguments", arguments if arguments is not None else {})
        self._init("result", result)
        self._init("error", error)
        self._init("retry", False)
        self._bind_run(run)


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
