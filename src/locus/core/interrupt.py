# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Human-in-the-loop interrupt/resume mechanism.

This module provides the ability to pause graph execution
for human input and resume with the provided value.

Example - Basic interrupt:
    from locus.core.interrupt import interrupt

    async def review_node(inputs):
        # Pause and wait for human approval
        approval = interrupt({
            "action": "delete_user",
            "user_id": inputs["user_id"],
            "message": "Please approve this deletion"
        })

        if approval == "approved":
            return {"status": "deleted"}
        return {"status": "cancelled"}

Example - Resume:
    from locus.core.command import Command

    # Resume the interrupted graph
    result = await graph.invoke(
        Command(resume="approved"),
        config={"thread_id": "my-thread"}
    )
"""

from __future__ import annotations

import contextvars
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


# =============================================================================
# Context for tracking current execution
# =============================================================================

# Context variable to track current node during execution
_current_node_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_node_id",
    default=None,
)

# Context variable to track current graph execution
_current_graph_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_graph_id",
    default=None,
)

# Context variable for resume value when resuming from interrupt
_resume_value: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "_resume_value",
    default=None,
)

# Context variable to track if we're in resume mode
_is_resuming: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_is_resuming",
    default=False,
)


# =============================================================================
# Interrupt Models
# =============================================================================


class InterruptValue(BaseModel):
    """
    Value captured when an interrupt occurs.

    Contains all information needed to display to the user
    and resume execution later.

    Attributes:
        interrupt_id: Unique identifier for this interrupt
        payload: Data to present to the human (question, context, etc.)
        node_id: ID of the node that raised the interrupt
        graph_id: ID of the graph being executed
        created_at: When the interrupt occurred
        metadata: Additional context for the interrupt
    """

    interrupt_id: str = Field(default_factory=lambda: f"int_{uuid4().hex[:8]}")
    payload: Any
    node_id: str | None = None
    graph_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    def to_display(self) -> dict[str, Any]:
        """Convert to a display-friendly format."""
        return {
            "interrupt_id": self.interrupt_id,
            "payload": self.payload,
            "node_id": self.node_id,
            "graph_id": self.graph_id,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


class InterruptState(BaseModel):
    """
    State saved when graph is interrupted.

    This is stored in the checkpoint to enable resumption.

    Attributes:
        interrupt: The interrupt value that paused execution
        node_id: Node to resume from
        pending_nodes: Nodes that were scheduled but not yet executed
        partial_results: Results from nodes completed before interrupt
    """

    interrupt: InterruptValue
    node_id: str
    pending_nodes: list[str] = Field(default_factory=list)
    partial_results: dict[str, Any] = Field(default_factory=dict)
    state_snapshot: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


# =============================================================================
# Interrupt Exception
# =============================================================================


class InterruptException(Exception):
    """
    Exception raised to pause graph execution for human input.

    This is caught by the graph executor, which saves the current
    state and returns the interrupt value to the caller.

    The caller can then present the interrupt to a human and
    resume execution with their response.
    """

    def __init__(self, value: InterruptValue):
        self.value = value
        super().__init__(f"Interrupt requested: {value.interrupt_id}")


class GraphInterrupted(Exception):
    """
    Raised when graph execution is paused for human input.

    Contains all information needed to resume execution.
    """

    def __init__(
        self,
        interrupt_state: InterruptState,
        checkpoint_id: str | None = None,
    ):
        self.interrupt_state = interrupt_state
        self.checkpoint_id = checkpoint_id
        super().__init__(
            f"Graph interrupted at node '{interrupt_state.node_id}': "
            f"{interrupt_state.interrupt.interrupt_id}"
        )


# =============================================================================
# Interrupt Function
# =============================================================================


def interrupt(payload: Any, **metadata: Any) -> Any:
    """
    Pause graph execution and wait for human input.

    When called, this function:
    1. If resuming: Returns the resume value immediately
    2. If not resuming: Raises InterruptException to pause execution

    The graph executor catches the exception, saves state,
    and returns the interrupt to the caller. When the caller
    provides a response and resumes, this function returns
    that response.

    Args:
        payload: Data to present to the human. Can be:
            - str: Simple message/question
            - dict: Structured data (action details, options, etc.)
            - Any serializable value
        **metadata: Additional context (not displayed, for tracking)

    Returns:
        The value passed when resuming (via Command(resume=...))

    Raises:
        InterruptException: When not resuming, to pause execution

    Example - Simple approval:
        >>> approval = interrupt("Approve this action?")
        >>> if approval == "yes":
        ...     execute_action()

    Example - Structured data:
        >>> response = interrupt(
        ...     {
        ...         "type": "confirmation",
        ...         "action": "delete_account",
        ...         "account_id": "12345",
        ...         "options": ["confirm", "cancel", "modify"],
        ...     }
        ... )

    Example - With metadata:
        >>> result = interrupt(
        ...     {"question": "Select priority"}, urgency="high", deadline="2024-01-01"
        ... )
    """
    # Check if we're resuming from an interrupt
    if _is_resuming.get():
        resume_val = _resume_value.get()
        # Clear resume state after use
        _is_resuming.set(False)
        _resume_value.set(None)
        return resume_val

    # Not resuming - create and raise interrupt
    node_id = _current_node_id.get()
    graph_id = _current_graph_id.get()

    value = InterruptValue(
        payload=payload,
        node_id=node_id,
        graph_id=graph_id,
        metadata=metadata,
    )

    raise InterruptException(value)


# =============================================================================
# Context Management
# =============================================================================


class NodeExecutionContext:
    """
    Context manager for node execution.

    Sets up context variables for interrupt handling.

    Example:
        async with NodeExecutionContext(node_id="my_node", graph_id="my_graph"):
            result = await node.execute(inputs)
    """

    def __init__(
        self,
        node_id: str,
        graph_id: str | None = None,
        resume_value: Any = None,
        is_resuming: bool = False,
    ):
        self.node_id = node_id
        self.graph_id = graph_id
        self.resume_value = resume_value
        self.is_resuming = is_resuming
        self._tokens: list[contextvars.Token[Any]] = []

    def __enter__(self) -> NodeExecutionContext:
        self._tokens.append(_current_node_id.set(self.node_id))
        self._tokens.append(_current_graph_id.set(self.graph_id))
        self._tokens.append(_resume_value.set(self.resume_value))
        self._tokens.append(_is_resuming.set(self.is_resuming))
        return self

    def __exit__(self, *args: Any) -> None:
        for token in reversed(self._tokens):
            # Reset to previous value
            try:
                token.var.reset(token)
            except ValueError:
                pass  # Token already reset

    async def __aenter__(self) -> NodeExecutionContext:
        return self.__enter__()

    async def __aexit__(self, *args: Any) -> None:
        self.__exit__(*args)


def set_resume_context(value: Any) -> None:
    """
    Set resume context for the next interrupt call.

    Used by graph executor when resuming from interrupt.

    Args:
        value: Value to return from interrupt()
    """
    _resume_value.set(value)
    _is_resuming.set(True)


def clear_resume_context() -> None:
    """Clear resume context after handling."""
    _resume_value.set(None)
    _is_resuming.set(False)


def get_current_node_id() -> str | None:
    """Get the current node ID from context."""
    return _current_node_id.get()


def get_current_graph_id() -> str | None:
    """Get the current graph ID from context."""
    return _current_graph_id.get()


# =============================================================================
# Interrupt Handlers
# =============================================================================


class InterruptHandler:
    """
    Base class for handling interrupts.

    Subclass this to create custom interrupt handlers
    (e.g., CLI prompts, web callbacks, message queues).

    Example:
        class CLIInterruptHandler(InterruptHandler):
            async def handle(self, interrupt: InterruptValue) -> Any:
                print(f"Interrupt: {interrupt.payload}")
                return input("Your response: ")
    """

    async def handle(self, interrupt: InterruptValue) -> Any:
        """
        Handle an interrupt and return the response.

        Args:
            interrupt: The interrupt value to handle

        Returns:
            Response value to pass back to the interrupted node
        """
        raise NotImplementedError("Subclasses must implement handle()")

    async def can_handle(self, interrupt: InterruptValue) -> bool:
        """Check if this handler can process the interrupt."""
        return True


class AutoApproveHandler(InterruptHandler):
    """
    Interrupt handler that auto-approves everything.

    Useful for testing and automated pipelines.
    """

    def __init__(self, response: Any = "approved"):
        self.response = response

    async def handle(self, interrupt: InterruptValue) -> Any:
        """Return configured response."""
        return self.response


class CallbackInterruptHandler(InterruptHandler):
    """
    Interrupt handler using a callback function.

    Example:
        async def my_callback(interrupt):
            # Custom logic
            return await get_user_input(interrupt.payload)

        handler = CallbackInterruptHandler(my_callback)
    """

    def __init__(self, callback: Any):
        self.callback = callback

    async def handle(self, interrupt: InterruptValue) -> Any:
        """Call the callback with the interrupt."""
        import asyncio

        if asyncio.iscoroutinefunction(self.callback):
            return await self.callback(interrupt)
        return self.callback(interrupt)
