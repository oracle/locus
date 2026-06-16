# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Core ReAct loop implementation - 100% Pydantic."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from locus.core.events import (
    LoopEvent,
    TerminateEvent,
)
from locus.core.messages import Message
from locus.core.state import AgentState
from locus.loop.nodes import ExecuteNode, Node, ReflectNode, ThinkNode
from locus.loop.router import NodeType, Router


if TYPE_CHECKING:
    from locus.core.protocols import ModelProtocol
    from locus.tools.registry import ToolRegistry


class ReActLoopConfig(BaseModel):
    """Configuration for the ReAct loop."""

    # Maximum iterations before forced termination
    max_iterations: int = Field(default=20, ge=1)

    # Confidence threshold for completion
    confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)

    # Enable reflection step
    enable_reflection: bool = True

    # Reflect every N iterations
    reflect_interval: int = Field(default=1, ge=1)

    # System prompt for the agent
    system_prompt: str | None = None

    # Terminal tool names
    terminal_tools: frozenset[str] = Field(
        default_factory=lambda: frozenset({"submit", "done", "finish", "complete"})
    )

    model_config = {"frozen": True}


class ReActLoop(BaseModel):
    """
    ReAct (Reason + Act) loop implementation.

    Implements the Think -> Execute -> Reflect cycle with:
    - Streaming events via AsyncIterator
    - Conditional routing based on state
    - Confidence-based termination
    - Tool loop detection

    Usage:
        loop = ReActLoop(model=my_model, registry=my_tools)

        async for event in loop.run("Solve this problem"):
            match event:
                case ThinkEvent():
                    print(f"Thinking: {event.reasoning}")
                case ToolCompleteEvent():
                    print(f"Tool {event.tool_name}: {event.result}")
                case TerminateEvent():
                    print(f"Done: {event.reason}")
    """

    model: Any  # ModelProtocol - Any for Pydantic compatibility
    registry: Any  # ToolRegistry
    config: ReActLoopConfig = Field(default_factory=ReActLoopConfig)

    model_config = {"arbitrary_types_allowed": True}

    def _create_nodes(self) -> dict[NodeType, Node]:
        """Create the nodes for the loop."""
        return {
            NodeType.THINK: ThinkNode(
                model=self.model,
                registry=self.registry,
                system_prompt=self.config.system_prompt,
            ),
            NodeType.EXECUTE: ExecuteNode(registry=self.registry),
            NodeType.REFLECT: ReflectNode(),
        }

    def _create_router(self) -> Router:
        """Create the router for the loop."""
        return Router(
            enable_reflection=self.config.enable_reflection,
            reflect_interval=self.config.reflect_interval,
        )

    def _create_initial_state(self, prompt: str, **kwargs: Any) -> AgentState:
        """Create the initial agent state."""
        state = AgentState(
            max_iterations=self.config.max_iterations,
            confidence_threshold=self.config.confidence_threshold,
            terminal_tools=self.config.terminal_tools,
            **kwargs,
        )

        # Add user message
        user_message = Message.user(prompt)
        state = state.with_message(user_message)

        return state

    async def run(
        self,
        prompt: str,
        initial_state: AgentState | None = None,
        **state_kwargs: Any,
    ) -> AsyncIterator[LoopEvent]:
        """
        Run the ReAct loop.

        Args:
            prompt: User prompt to process
            initial_state: Optional pre-configured state
            **state_kwargs: Additional state configuration

        Yields:
            Loop events (ThinkEvent, ToolStartEvent, ToolCompleteEvent,
                        ReflectEvent, TerminateEvent)
        """
        # Initialize
        if initial_state is not None:
            state = initial_state.with_message(Message.user(prompt))
        else:
            state = self._create_initial_state(prompt, **state_kwargs)

        nodes = self._create_nodes()
        router = self._create_router()

        # Start with Think
        current_node = NodeType.THINK

        while current_node != NodeType.TERMINATE:
            # Execute current node
            node = nodes.get(current_node)

            if node is not None:
                result = await node.execute(state)
                state = result.state

                # Yield all events from this node
                for event in result.events:
                    yield event

            # Route to next node
            decision = router.route(current_node, state)
            current_node = decision.next_node

            # Increment iteration when going back to Think
            if current_node == NodeType.THINK:
                state = state.next_iteration()

        # Emit termination event
        _should_stop, reason = state.should_terminate
        yield TerminateEvent(
            reason=reason or "complete",
            iterations_used=state.iteration,
            final_confidence=state.confidence,
            total_tool_calls=len(state.tool_executions),
        )

    async def run_to_completion(
        self,
        prompt: str,
        initial_state: AgentState | None = None,
        **state_kwargs: Any,
    ) -> tuple[AgentState, list[LoopEvent]]:
        """
        Run the loop and collect all events.

        Convenience method that collects all events and returns
        the final state along with the event history.

        Args:
            prompt: User prompt to process
            initial_state: Optional pre-configured state
            **state_kwargs: Additional state configuration

        Returns:
            Tuple of (final_state, events)
        """
        events: list[LoopEvent] = []

        # Initialize state to track
        if initial_state is not None:
            state = initial_state.with_message(Message.user(prompt))
        else:
            state = self._create_initial_state(prompt, **state_kwargs)

        nodes = self._create_nodes()
        router = self._create_router()
        current_node = NodeType.THINK

        while current_node != NodeType.TERMINATE:
            node = nodes.get(current_node)

            if node is not None:
                result = await node.execute(state)
                state = result.state
                events.extend(result.events)

            decision = router.route(current_node, state)
            current_node = decision.next_node

            if current_node == NodeType.THINK:
                state = state.next_iteration()

        # Add termination event
        _should_stop, reason = state.should_terminate
        terminate_event = TerminateEvent(
            reason=reason or "complete",
            iterations_used=state.iteration,
            final_confidence=state.confidence,
            total_tool_calls=len(state.tool_executions),
        )
        events.append(terminate_event)

        return state, events

    def with_config(self, **updates: Any) -> ReActLoop:
        """
        Create a new loop with updated configuration.

        Returns a new ReActLoop instance (immutable).
        """
        new_config = self.config.model_copy(update=updates)
        return self.model_copy(update={"config": new_config})


def create_react_loop(
    model: ModelProtocol,
    registry: ToolRegistry,
    *,
    max_iterations: int = 20,
    confidence_threshold: float = 0.85,
    enable_reflection: bool = True,
    system_prompt: str | None = None,
) -> ReActLoop:
    """
    Factory function to create a ReActLoop.

    Args:
        model: LLM model to use for reasoning
        registry: Tool registry with available tools
        max_iterations: Maximum iterations before forced termination
        confidence_threshold: Confidence level for completion
        enable_reflection: Whether to include reflection step
        system_prompt: Optional system prompt

    Returns:
        Configured ReActLoop instance
    """
    config = ReActLoopConfig(
        max_iterations=max_iterations,
        confidence_threshold=confidence_threshold,
        enable_reflection=enable_reflection,
        system_prompt=system_prompt,
    )

    return ReActLoop(
        model=model,
        registry=registry,
        config=config,
    )
