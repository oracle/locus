# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Conditional routing logic for ReAct loop - 100% Pydantic."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from locus.core.state import AgentState


class NodeType(StrEnum):
    """Types of nodes in the ReAct loop."""

    THINK = "think"
    EXECUTE = "execute"
    REFLECT = "reflect"
    TERMINATE = "terminate"


class RouteDecision(BaseModel):
    """Result of a routing decision."""

    next_node: NodeType
    reason: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


class Router(BaseModel):
    """
    Conditional routing logic for the ReAct loop.

    Determines which node to execute next based on the current state.
    """

    # Whether to include reflect step in the loop
    enable_reflection: bool = True

    # Reflect every N iterations (if enabled)
    reflect_interval: int = 1

    # Skip reflection when no tools were called
    skip_reflect_without_tools: bool = True

    model_config = {"frozen": True}

    def route_from_think(self, state: AgentState) -> RouteDecision:
        """
        Route from the Think node.

        After thinking:
        - If tool calls exist -> Execute
        - If no tool calls and should terminate -> Terminate
        - If no tool calls -> Reflect (if enabled) or Terminate
        """
        # Check for termination conditions first
        should_stop, reason = state.should_terminate
        if should_stop:
            return RouteDecision(
                next_node=NodeType.TERMINATE,
                reason=f"Termination condition met: {reason}",
                metadata={"termination_reason": reason},
            )

        # If there are tool calls, execute them
        if state.last_tool_calls:
            return RouteDecision(
                next_node=NodeType.EXECUTE,
                reason=f"Executing {len(state.last_tool_calls)} tool call(s)",
                metadata={"tool_count": len(state.last_tool_calls)},
            )

        # No tool calls - this is a potential termination point
        # The agent has responded without requesting any actions
        if self.enable_reflection and not self.skip_reflect_without_tools:
            return RouteDecision(
                next_node=NodeType.REFLECT,
                reason="No tool calls - reflecting on response",
            )

        return RouteDecision(
            next_node=NodeType.TERMINATE,
            reason="No tool calls - completing",
            metadata={"termination_reason": "no_tools"},
        )

    def route_from_execute(self, state: AgentState) -> RouteDecision:
        """
        Route from the Execute node.

        After executing tools:
        - If should terminate -> Terminate
        - If reflection enabled and interval met -> Reflect
        - Otherwise -> Think
        """
        # Check termination
        should_stop, reason = state.should_terminate
        if should_stop:
            return RouteDecision(
                next_node=NodeType.TERMINATE,
                reason=f"Termination condition met: {reason}",
                metadata={"termination_reason": reason},
            )

        # Check if we should reflect
        if self._should_reflect(state):
            return RouteDecision(
                next_node=NodeType.REFLECT,
                reason="Reflecting on progress",
                metadata={"iteration": state.iteration},
            )

        # Continue to think
        return RouteDecision(
            next_node=NodeType.THINK,
            reason="Continuing to next thought",
        )

    def route_from_reflect(self, state: AgentState) -> RouteDecision:
        """
        Route from the Reflect node.

        After reflecting:
        - If should terminate -> Terminate
        - Otherwise -> Think (with new iteration)
        """
        # Check termination
        should_stop, reason = state.should_terminate
        if should_stop:
            return RouteDecision(
                next_node=NodeType.TERMINATE,
                reason=f"Termination condition met: {reason}",
                metadata={"termination_reason": reason},
            )

        # Continue to think
        return RouteDecision(
            next_node=NodeType.THINK,
            reason="Starting next iteration",
        )

    def route(self, current_node: NodeType, state: AgentState) -> RouteDecision:
        """
        Route from the given node based on current state.

        Args:
            current_node: The node that just completed
            state: Current agent state

        Returns:
            Decision about which node to execute next
        """
        if current_node == NodeType.THINK:
            return self.route_from_think(state)
        if current_node == NodeType.EXECUTE:
            return self.route_from_execute(state)
        if current_node == NodeType.REFLECT:
            return self.route_from_reflect(state)
        # Terminate node - stay terminated
        return RouteDecision(
            next_node=NodeType.TERMINATE,
            reason="Already terminated",
        )

    def _should_reflect(self, state: AgentState) -> bool:
        """Check if we should reflect at this point."""
        if not self.enable_reflection:
            return False

        # Check interval
        if state.iteration > 0 and state.iteration % self.reflect_interval == 0:
            return True

        # Reflect on errors
        if state.tool_executions:
            last_exec = state.tool_executions[-1]
            if not last_exec.success:
                return True

        # Reflect on potential loops
        if state.has_tool_loop:
            return True

        return False


class ConditionalRouter(Router):
    """
    Router with custom condition functions.

    Allows injecting custom routing logic for advanced use cases.
    """

    custom_conditions: list[tuple[str, Any]] = Field(default_factory=list)

    def add_condition(
        self,
        name: str,
        condition: Any,  # Callable[[AgentState], RouteDecision | None]
    ) -> ConditionalRouter:
        """
        Add a custom routing condition.

        The condition function receives the state and returns either:
        - A RouteDecision to override default routing
        - None to continue with default routing

        Returns a new router with the condition added (immutable).
        """
        new_conditions = [*self.custom_conditions, (name, condition)]
        return self.model_copy(update={"custom_conditions": new_conditions})

    def route(self, current_node: NodeType, state: AgentState) -> RouteDecision:
        """
        Route with custom conditions checked first.

        Custom conditions are checked in order. The first one to return
        a RouteDecision wins.
        """
        # Check custom conditions first
        for name, condition in self.custom_conditions:
            try:
                result = condition(state)
                if result is not None:
                    updated: RouteDecision = result.model_copy(
                        update={"metadata": {**result.metadata, "custom_condition": name}}
                    )
                    return updated
            except Exception:  # noqa: BLE001
                # Custom condition failed, continue with others
                continue

        # Fall back to default routing
        return super().route(current_node, state)
