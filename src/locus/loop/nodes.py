# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""ReAct loop nodes - Think, Execute, Reflect - 100% Pydantic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from locus.core.events import (
    ReflectEvent,
    ThinkEvent,
    ToolCompleteEvent,
    ToolStartEvent,
)
from locus.core.messages import Message
from locus.core.state import AgentState, ReasoningStep, ToolExecution
from locus.tools.executor import ConcurrentExecutor, ToolContextFactory, ToolExecutor


if TYPE_CHECKING:
    from locus.models.base import ModelResponse


class NodeResult(BaseModel):
    """Result from executing a node."""

    state: AgentState
    events: list[ThinkEvent | ToolStartEvent | ToolCompleteEvent | ReflectEvent] = Field(
        default_factory=list
    )

    model_config = {"arbitrary_types_allowed": True}


class Node(BaseModel, ABC):
    """Base class for ReAct loop nodes."""

    model_config = {"arbitrary_types_allowed": True}

    @abstractmethod
    async def execute(self, state: AgentState) -> NodeResult:
        """
        Execute the node.

        Args:
            state: Current agent state

        Returns:
            Updated state and any events produced
        """
        ...


class ThinkNode(Node):
    """
    Node that invokes the LLM to generate reasoning and/or tool calls.

    This is the "Reason" part of ReAct.
    """

    model: Any  # ModelProtocol - Any for Pydantic compatibility
    registry: Any  # ToolRegistry
    system_prompt: str | None = None

    async def execute(self, state: AgentState) -> NodeResult:
        """
        Generate the next thought and/or tool calls.

        Args:
            state: Current agent state

        Returns:
            Updated state with assistant message and ThinkEvent
        """
        # Build messages for model
        messages = list(state.messages)

        # Add system prompt if provided and not already present
        if self.system_prompt and (not messages or messages[0].role.value != "system"):
            messages.insert(0, Message.system(self.system_prompt))

        # Get tool schemas
        tool_schemas = self.registry.to_openai_schemas() if len(self.registry) > 0 else None

        # Call model
        response: ModelResponse = await self.model.complete(
            messages=messages,
            tools=tool_schemas,
        )

        # Extract response content
        assistant_message = response.message
        reasoning = assistant_message.content
        tool_calls = list(assistant_message.tool_calls)

        # Create ThinkEvent
        event = ThinkEvent(
            iteration=state.iteration,
            reasoning=reasoning,
            tool_calls=tool_calls,
        )

        # Update state with assistant message
        new_state = state.with_message(assistant_message)

        return NodeResult(state=new_state, events=[event])


def _find_matching_execution(
    state: AgentState, tool_name: str, arguments: dict
) -> ToolExecution | None:
    """Return a prior ToolExecution on ``state`` matching the given tool
    and arguments, or None if no match exists.

    Used by ExecuteNode to dedupe calls for tools marked ``idempotent=True``.
    Argument equality is a structural dict comparison, so a model legitimately
    re-calling a tool with different args (e.g. a new date) will not hit the
    cache.
    """
    for prior in reversed(state.tool_executions):
        if prior.tool_name != tool_name:
            continue
        try:
            if dict(prior.arguments) == arguments:
                return prior
        except (TypeError, ValueError):
            continue
    return None


class ExecuteNode(Node):
    """
    Node that executes tool calls.

    This is the "Act" part of ReAct.
    """

    registry: Any  # ToolRegistry - Any for Pydantic compatibility
    executor: ToolExecutor = Field(default_factory=ConcurrentExecutor)

    async def execute(self, state: AgentState) -> NodeResult:
        """
        Execute pending tool calls.

        This is the "Act" part of ReAct. Tool calls whose tool declared
        ``idempotent=True`` are de-duplicated against prior executions on
        the current state: if the same (tool_name, arguments) pair has
        already been executed during this agent run, the prior result is
        reused and the tool function is NOT invoked again. This prevents
        models that re-emit the same call from causing duplicate
        side-effects (double bookings, double transfers, etc.).

        Args:
            state: Current agent state with tool calls

        Returns:
            Updated state with tool results and events
        """
        from locus.tools.executor import ToolResult

        tool_calls = state.last_tool_calls
        # ``NodeResult.events`` is invariantly typed as the broader event
        # union; declare the local list with that union so the eventual
        # ``return NodeResult(events=events)`` type-checks.
        events: list[ThinkEvent | ToolStartEvent | ToolCompleteEvent | ReflectEvent] = []

        if not tool_calls:
            return NodeResult(state=state, events=[])

        # Create context factory
        ctx_factory = ToolContextFactory(
            run_id=state.run_id,
            agent_id=state.agent_id,
            iteration=state.iteration,
        )

        # Emit start events (dedup is transparent to observers)
        for tc in tool_calls:
            events.append(
                ToolStartEvent(
                    tool_name=tc.name,
                    tool_call_id=tc.id,
                    arguments=tc.arguments,
                )
            )

        # Split into fresh vs. cached (idempotent tools with a prior match).
        cached_results: dict[str, ToolResult] = {}
        fresh_calls = []
        for tc in tool_calls:
            tool = self.registry.get(tc.name) if hasattr(self.registry, "get") else None
            if tool is not None and getattr(tool, "idempotent", False):
                prior = _find_matching_execution(state, tc.name, dict(tc.arguments))
                if prior is not None:
                    cached_results[tc.id] = ToolResult(
                        tool_call_id=tc.id,
                        name=tc.name,
                        content=prior.result if prior.result is not None else "",
                        error=prior.error,
                        duration_ms=0.0,
                    )
                    continue
            fresh_calls.append(tc)

        # Execute the fresh ones; cached ones re-use prior output.
        fresh_results = (
            await self.executor.execute(
                tool_calls=fresh_calls,
                registry=self.registry,
                ctx_factory=ctx_factory,
            )
            if fresh_calls
            else []
        )
        results_by_id: dict[str, ToolResult] = {r.tool_call_id: r for r in fresh_results}
        results_by_id.update(cached_results)
        results = [results_by_id[tc.id] for tc in tool_calls if tc.id in results_by_id]

        # Process results and update state
        new_state = state
        tool_messages: list[Message] = []

        for result in results:
            # Record tool execution
            execution = ToolExecution(
                tool_name=result.name,
                tool_call_id=result.tool_call_id,
                arguments=next(
                    (tc.arguments for tc in tool_calls if tc.id == result.tool_call_id),
                    {},
                ),
                result=result.content if result.success else None,
                error=result.error,
                duration_ms=result.duration_ms,
            )
            new_state = new_state.with_tool_execution(execution)

            # Create tool message
            tool_messages.append(Message.tool(result))

            # Emit complete event
            events.append(
                ToolCompleteEvent(
                    tool_name=result.name,
                    tool_call_id=result.tool_call_id,
                    result=result.content if result.success else None,
                    error=result.error,
                    duration_ms=result.duration_ms,
                )
            )

        # Add all tool result messages
        new_state = new_state.with_messages(tool_messages)

        return NodeResult(state=new_state, events=events)


class ReflectNode(Node):
    """
    Node that evaluates progress and adjusts confidence.

    This implements a simplified Reflexion-style self-evaluation.
    """

    # Confidence adjustments for different assessments
    confidence_adjustments: dict[str, float] = Field(
        default_factory=lambda: {
            "on_track": 0.1,
            "new_findings": 0.15,
            "stuck": -0.1,
            "loop_detected": -0.2,
            "error": -0.15,
        }
    )

    async def execute(self, state: AgentState) -> NodeResult:
        """
        Reflect on the current progress and update confidence.

        Args:
            state: Current agent state

        Returns:
            Updated state with adjusted confidence and ReflectEvent
        """
        assessment, guidance = self._assess_progress(state)

        # Get confidence delta
        delta = self.confidence_adjustments.get(assessment, 0.0)

        # Update state with new confidence
        new_state = state.adjust_confidence(delta, diminishing=True)

        # Create reasoning step record
        step = ReasoningStep(
            iteration=state.iteration,
            thought=self._get_last_thought(state),
            tool_calls=state.last_tool_calls,
            tool_results=list(state.tool_executions[-len(state.last_tool_calls) :])
            if state.last_tool_calls
            else [],
            reflection=guidance,
            confidence_delta=delta,
        )
        new_state = new_state.with_reasoning_step(step)

        # Create event
        event = ReflectEvent(
            iteration=state.iteration,
            assessment=assessment,
            confidence_delta=delta,
            new_confidence=new_state.confidence,
            guidance=guidance,
        )

        return NodeResult(state=new_state, events=[event])

    def _assess_progress(self, state: AgentState) -> tuple[str, str | None]:
        """
        Assess the agent's progress.

        Returns:
            Tuple of (assessment, optional guidance message)
        """
        # Check for tool loop
        if state.has_tool_loop:
            return "loop_detected", "Breaking out of repetitive pattern - try a different approach"

        # Check for errors in recent executions
        recent_executions = state.tool_executions[-3:] if state.tool_executions else []
        recent_errors = sum(1 for e in recent_executions if not e.success)

        if recent_errors >= 2:
            return "error", "Multiple recent errors - consider adjusting approach"

        if recent_errors == 1:
            return "stuck", "Tool error occurred - may need to retry or try alternative"

        # Check for progress indicators
        if state.last_tool_calls:
            # Tools were called, which usually indicates progress
            last_results = state.tool_executions[-len(state.last_tool_calls) :]
            if all(e.success for e in last_results):
                # Check if we got meaningful results
                has_content = any(e.result and len(e.result) > 10 for e in last_results)
                if has_content:
                    return "new_findings", "Retrieved useful information"
                return "on_track", None

        # Default: on track
        return "on_track", None

    def _get_last_thought(self, state: AgentState) -> str | None:
        """Get the last thought/reasoning from messages."""
        for msg in reversed(state.messages):
            if msg.role.value == "assistant" and msg.content:
                return msg.content
        return None
