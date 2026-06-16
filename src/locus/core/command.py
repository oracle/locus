# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Command primitive for unified state updates and control flow.

The Command class combines state updates with routing control,
enabling nodes to both modify state and direct execution flow
in a single return value.

Example:
    from locus.core.command import Command

    async def router_node(inputs):
        if inputs["urgency"] == "high":
            return Command(
                update={"priority": 1},
                goto="fast_track"
            )
        return Command(goto="standard_queue")

    async def approval_node(inputs):
        # After human approves via interrupt
        return Command(
            update={"approved": True, "approved_by": inputs["user"]},
            goto="execute_action"
        )
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Command(BaseModel):
    """
    Control flow command for graph execution.

    A Command can:
    1. Update state with new values
    2. Direct execution to specific node(s)
    3. Resume from an interrupt with a value

    This enables complex control flow patterns like:
    - Dynamic routing based on state
    - Branching to multiple parallel nodes
    - Returning from interrupts with data

    Attributes:
        update: State updates to apply (merged with reducers if defined)
        goto: Target node(s) to execute next. Can be:
            - str: Single node ID
            - list[str]: Multiple nodes (parallel execution)
            - None: Continue with normal graph flow
        resume: Value to pass back when resuming from interrupt
        graph: For subgraph commands, which graph to route to

    Example - Simple routing:
        >>> Command(goto="next_node")

    Example - State update with routing:
        >>> Command(update={"processed": True, "result": data}, goto="output_node")

    Example - Parallel fan-out:
        >>> Command(goto=["worker_1", "worker_2", "worker_3"])

    Example - Resume from interrupt:
        >>> Command(resume="approved")
    """

    update: dict[str, Any] = Field(default_factory=dict)
    goto: str | list[str] | None = None
    resume: Any = None
    graph: str | None = None  # For subgraph routing

    model_config = {"frozen": True}

    @property
    def has_update(self) -> bool:
        """Whether this command includes state updates."""
        return bool(self.update)

    @property
    def has_goto(self) -> bool:
        """Whether this command specifies routing."""
        return self.goto is not None

    @property
    def has_resume(self) -> bool:
        """Whether this command is resuming from interrupt."""
        return self.resume is not None

    @property
    def is_parallel_goto(self) -> bool:
        """Whether goto targets multiple nodes."""
        return isinstance(self.goto, list)

    @property
    def goto_nodes(self) -> list[str]:
        """Get list of target nodes (normalizes single/list)."""
        if self.goto is None:
            return []
        if isinstance(self.goto, str):
            return [self.goto]
        return list(self.goto)

    def with_update(self, **kwargs: Any) -> Command:
        """Return new Command with additional updates merged."""
        new_update = {**self.update, **kwargs}
        return self.model_copy(update={"update": new_update})

    def with_goto(self, target: str | list[str]) -> Command:
        """Return new Command with different goto target."""
        return self.model_copy(update={"goto": target})


# =============================================================================
# Special Command Constants
# =============================================================================


class End(Command):
    """
    Special command indicating graph completion.

    Use this to explicitly terminate graph execution.

    Example:
        async def final_node(inputs):
            return End(update={"final_result": inputs["data"]})
    """

    goto: str = "__END__"

    model_config = {"frozen": True}


class Continue(Command):
    """
    Special command indicating normal flow continuation.

    Useful when you want to update state but continue
    with default routing logic.

    Example:
        async def process_node(inputs):
            result = process(inputs)
            return Continue(update={"processed": result})
    """

    goto: None = None

    model_config = {"frozen": True}


# =============================================================================
# Command Result Handling
# =============================================================================


def is_command(value: Any) -> bool:
    """Check if a value is a Command instance."""
    return isinstance(value, Command)


def normalize_node_output(output: Any) -> tuple[dict[str, Any], Command | None]:
    """
    Normalize node output to (state_update, command).

    Nodes can return:
    - dict: Treated as state update, no routing
    - Command: Extract update and routing
    - None: No update, no routing
    - Other: Wrapped as {"result": value}

    Args:
        output: Raw output from node execution

    Returns:
        Tuple of (state_update_dict, optional_command)
    """
    if output is None:
        return {}, None

    if isinstance(output, Command):
        return dict(output.update), output

    if isinstance(output, dict):
        return output, None

    # Pydantic BaseModel → treat as state update dict (like LangGraph does)
    try:
        from pydantic import BaseModel as _BaseModel

        if isinstance(output, _BaseModel):
            return output.model_dump(mode="python", exclude_none=False), None
    except Exception:  # noqa: BLE001
        pass

    # Wrap other values
    return {"result": output}, None


# =============================================================================
# Convenience Constructors
# =============================================================================


def goto(target: str | list[str], **updates: Any) -> Command:
    """
    Create a Command that routes to target node(s).

    Args:
        target: Node ID or list of node IDs
        **updates: Optional state updates

    Returns:
        Command with goto and optional updates

    Example:
        >>> goto("next_node")
        >>> goto(["worker_1", "worker_2"], task_id=123)
    """
    return Command(update=updates, goto=target)


def end(**updates: Any) -> End:
    """
    Create an End command to terminate graph.

    Args:
        **updates: Final state updates

    Returns:
        End command

    Example:
        >>> end(result="success", data=processed_data)
    """
    return End(update=updates)


def resume_with(value: Any, **updates: Any) -> Command:
    """
    Create a Command to resume from interrupt.

    Args:
        value: Value to pass to interrupted node
        **updates: Optional state updates

    Returns:
        Command with resume value

    Example:
        >>> resume_with("approved")
        >>> resume_with({"action": "modify", "changes": data})
    """
    return Command(update=updates, resume=value)
