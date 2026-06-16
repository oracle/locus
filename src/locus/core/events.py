# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Event types for streaming and hooks - 100% Pydantic."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from locus.core.messages import ToolCall


class LocusEvent(BaseModel):
    """Base class for all Locus events."""

    event_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = {"frozen": True}


# =============================================================================
# Loop Events
# =============================================================================


class ThinkEvent(LocusEvent):
    """Agent produced reasoning and/or tool calls."""

    event_type: Literal["think"] = "think"
    iteration: int
    reasoning: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ToolStartEvent(LocusEvent):
    """Tool execution started."""

    event_type: Literal["tool_start"] = "tool_start"
    tool_name: str
    tool_call_id: str
    arguments: dict[str, Any]


class ToolCompleteEvent(LocusEvent):
    """Tool execution completed."""

    event_type: Literal["tool_complete"] = "tool_complete"
    tool_name: str
    tool_call_id: str
    result: str | None = None
    error: str | None = None
    duration_ms: float | None = None

    @property
    def success(self) -> bool:
        """Whether the tool execution succeeded."""
        return self.error is None


class ReflectEvent(LocusEvent):
    """Reflexion evaluation completed."""

    event_type: Literal["reflect"] = "reflect"
    iteration: int
    assessment: str  # "on_track", "stuck", "new_findings", "loop_detected"
    confidence_delta: float
    new_confidence: float
    guidance: str | None = None


class GroundingEvent(LocusEvent):
    """Grounding evaluation completed."""

    event_type: Literal["grounding"] = "grounding"
    score: float
    claims_evaluated: int
    ungrounded_claims: list[str] = Field(default_factory=list)
    requires_replan: bool = False


class TerminateEvent(LocusEvent):
    """Agent execution terminated."""

    event_type: Literal["terminate"] = "terminate"
    reason: (
        str  # "complete", "max_iterations", "confidence_met", "terminal_tool", "tool_loop", "error"
    )
    iterations_used: int
    final_confidence: float
    total_tool_calls: int
    final_message: str | None = None  # Final assistant message content


class InterruptEvent(LocusEvent):
    """Agent paused for user input.

    When a tool calls interrupt() (e.g., ask_user), the agent yields this
    event and pauses. The caller should present the question to the user
    and call agent.resume(response) to continue.
    """

    event_type: Literal["interrupt"] = "interrupt"
    question: str
    options: list[str] | None = None
    interrupt_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Model Events
# =============================================================================


class ModelChunkEvent(LocusEvent):
    """Streaming chunk from model."""

    event_type: Literal["model_chunk"] = "model_chunk"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    done: bool = False


class ModelCompleteEvent(LocusEvent):
    """Model completion finished."""

    event_type: Literal["model_complete"] = "model_complete"
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    stop_reason: str | None = None


# =============================================================================
# Multi-Agent Events
# =============================================================================


class SpecialistStartEvent(LocusEvent):
    """Specialist agent started."""

    event_type: Literal["specialist_start"] = "specialist_start"
    specialist_id: str
    specialist_type: str
    task: str


class SpecialistCompleteEvent(LocusEvent):
    """Specialist agent completed."""

    event_type: Literal["specialist_complete"] = "specialist_complete"
    specialist_id: str
    specialist_type: str
    result: str | None = None
    confidence: float
    duration_ms: float


class OrchestratorDecisionEvent(LocusEvent):
    """Orchestrator made a routing decision."""

    event_type: Literal["orchestrator_decision"] = "orchestrator_decision"
    decision: str  # "invoke_specialist", "correlate", "summarize", "finalize"
    specialists_selected: list[str] = Field(default_factory=list)
    reasoning: str | None = None


# =============================================================================
# Causal Events
# =============================================================================


class CausalNodeEvent(LocusEvent):
    """Causal inference node identified."""

    event_type: Literal["causal_node"] = "causal_node"
    node_id: str
    label: str
    node_type: str  # "root_cause", "symptom", "intermediate"
    evidence: list[str] = Field(default_factory=list)


class CausalEdgeEvent(LocusEvent):
    """Causal relationship identified."""

    event_type: Literal["causal_edge"] = "causal_edge"
    source_id: str
    target_id: str
    relationship: str  # "causes", "correlates_with", "precedes"
    confidence: float


# =============================================================================
# Hook Events
# =============================================================================


class HookEvent(LocusEvent):
    """Base class for hook lifecycle events."""


class BeforeInvocationEvent(HookEvent):
    """Fired before agent invocation starts."""

    event_type: Literal["before_invocation"] = "before_invocation"
    prompt: str
    agent_id: str | None = None


class AfterInvocationEvent(HookEvent):
    """Fired after agent invocation completes."""

    event_type: Literal["after_invocation"] = "after_invocation"
    success: bool
    iterations: int
    confidence: float
    duration_ms: float


class BeforeToolCallEvent(HookEvent):
    """Fired before a tool is called."""

    event_type: Literal["before_tool_call"] = "before_tool_call"
    tool_name: str
    arguments: dict[str, Any]
    # Writable: hooks can modify arguments
    modified_arguments: dict[str, Any] | None = None


class AfterToolCallEvent(HookEvent):
    """Fired after a tool call completes."""

    event_type: Literal["after_tool_call"] = "after_tool_call"
    tool_name: str
    tool_call_id: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str | None = None
    error: str | None = None
    duration_ms: float


# =============================================================================
# Type aliases
# =============================================================================

LoopEvent = (
    ThinkEvent | ToolStartEvent | ToolCompleteEvent | ReflectEvent | GroundingEvent | TerminateEvent
)
AgentEvent = LoopEvent | SpecialistStartEvent | SpecialistCompleteEvent | OrchestratorDecisionEvent
AllEvents = (
    AgentEvent
    | ModelChunkEvent
    | ModelCompleteEvent
    | CausalNodeEvent
    | CausalEdgeEvent
    | HookEvent
)
