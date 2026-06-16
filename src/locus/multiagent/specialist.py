# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Specialist agents with domain-specific capabilities."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, Field

from locus.core.events import SpecialistCompleteEvent, SpecialistStartEvent
from locus.core.messages import Message
from locus.core.state import AgentState
from locus.tools.decorator import Tool


if TYPE_CHECKING:
    from locus.tools.registry import ToolRegistry


class SpecialistResult(BaseModel):
    """Result from a specialist agent execution."""

    specialist_id: str
    specialist_type: str
    output: str | None = None
    confidence: float = 0.0
    duration_ms: float = 0.0
    state: AgentState | None = None
    error: str | None = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def success(self) -> bool:
        """Whether the specialist completed successfully."""
        return self.error is None


class PlaybookStep(BaseModel):
    """A step in a playbook procedure."""

    instruction: str
    required_tools: list[str] = Field(default_factory=list)
    expected_output: str | None = None
    on_failure: str | None = None


class Playbook(BaseModel):
    """
    A predefined procedure for a specialist to follow.

    Playbooks provide structured guidance for domain-specific tasks.
    """

    name: str
    description: str
    steps: list[PlaybookStep] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    success_criteria: str | None = None

    def to_prompt(self) -> str:
        """Convert playbook to a prompt for the specialist."""
        lines = [
            f"## Playbook: {self.name}",
            "",
            self.description,
            "",
        ]

        if self.preconditions:
            lines.append("### Preconditions:")
            for pre in self.preconditions:
                lines.append(f"- {pre}")
            lines.append("")

        lines.append("### Steps:")
        for i, step in enumerate(self.steps, 1):
            lines.append(f"{i}. {step.instruction}")
            if step.required_tools:
                lines.append(f"   Tools: {', '.join(step.required_tools)}")
            if step.expected_output:
                lines.append(f"   Expected: {step.expected_output}")
            if step.on_failure:
                lines.append(f"   On failure: {step.on_failure}")

        if self.success_criteria:
            lines.append("")
            lines.append(f"### Success Criteria: {self.success_criteria}")

        return "\n".join(lines)


class Specialist(BaseModel):
    """
    A specialist agent focused on a specific domain.

    Features:
    - Domain-specific system prompt
    - Focused tool set
    - Optional playbook integration
    - Confidence-based execution
    """

    id: str = Field(default_factory=lambda: f"specialist_{uuid4().hex[:8]}")
    name: str
    specialist_type: str
    description: str

    # Domain-specific configuration
    system_prompt: str
    tools: list[Tool] = Field(default_factory=list)
    playbooks: list[Playbook] = Field(default_factory=list)

    # Execution configuration
    max_iterations: int = 10
    confidence_threshold: float = 0.85

    # The model to use (injected)
    model: Any = None

    model_config = {"arbitrary_types_allowed": True}

    def _build_system_prompt(self, task: str, playbook: Playbook | None = None) -> str:
        """Build the complete system prompt for the specialist."""
        parts = [
            f"You are a {self.name} specialist.",
            "",
            self.description,
            "",
            "## System Instructions:",
            self.system_prompt,
        ]

        if playbook:
            parts.extend(
                [
                    "",
                    playbook.to_prompt(),
                ]
            )

        parts.extend(
            [
                "",
                "## Current Task:",
                task,
            ]
        )

        return "\n".join(parts)

    def select_playbook(self, task: str) -> Playbook | None:
        """
        Select the most appropriate playbook for a task.

        Args:
            task: The task description

        Returns:
            Best matching playbook or None
        """
        # Simple keyword matching - could be enhanced with embeddings
        task_lower = task.lower()

        best_match: Playbook | None = None
        best_score = 0

        for playbook in self.playbooks:
            # Count matching keywords
            score = 0
            playbook_words = set(playbook.name.lower().split())
            playbook_words.update(playbook.description.lower().split())

            for word in task_lower.split():
                if word in playbook_words:
                    score += 1

            if score > best_score:
                best_score = score
                best_match = playbook

        return best_match

    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        registry: ToolRegistry | None = None,
    ) -> SpecialistResult:
        """
        Execute the specialist on a task.

        Args:
            task: The task to perform
            context: Optional context from orchestrator or other specialists
            registry: Tool registry (uses self.tools if not provided)

        Returns:
            SpecialistResult with output and confidence
        """
        if self.model is None:
            return SpecialistResult(
                specialist_id=self.id,
                specialist_type=self.specialist_type,
                error="No model configured for specialist",
            )

        # Local import — observability is optional; this is a no-op
        # outside a run_context.
        from locus.observability.emit import (  # noqa: PLC0415
            EV_SPECIALIST_COMPLETED,
            EV_SPECIALIST_STARTED,
            emit,
        )

        start_time = time.perf_counter()

        await emit(
            EV_SPECIALIST_STARTED,
            specialist_id=self.id,
            specialist_type=self.specialist_type,
            task_preview=task[:160],
        )

        # Construct the typed event for back-compat consumers that iterate
        # the agent loop's event stream. Stored under a leading underscore
        # because nothing currently consumes the local — the observability
        # emit above is the live publication path.
        _start_event = SpecialistStartEvent(
            specialist_id=self.id,
            specialist_type=self.specialist_type,
            task=task,
        )

        # Select appropriate playbook
        playbook = self.select_playbook(task)

        # Build system prompt
        system_prompt = self._build_system_prompt(task, playbook)

        # Initialize state
        state = AgentState(
            agent_id=self.id,
            max_iterations=self.max_iterations,
            confidence_threshold=self.confidence_threshold,
        )

        # Add system message
        state = state.with_message(Message.system(system_prompt))

        # Add context if provided
        if context:
            context_str = self._format_context(context)
            state = state.with_message(Message.user(context_str))

        # Add task message
        state = state.with_message(Message.user(task))

        # Get tool schemas
        tool_schemas = None
        if self.tools:
            tool_schemas = [tool.to_openai_schema() for tool in self.tools]

        try:
            # Simple single-turn execution for now
            # Full agentic loop would integrate with the main loop system
            response = await self.model.complete(
                messages=list(state.messages),
                tools=tool_schemas,
            )

            # Update state with response
            state = state.with_message(response.message)

            # Extract confidence from response (simple heuristic)
            confidence = self._estimate_confidence(response.message.content or "")

            duration_ms = (time.perf_counter() - start_time) * 1000

            await emit(
                EV_SPECIALIST_COMPLETED,
                specialist_id=self.id,
                specialist_type=self.specialist_type,
                output_preview=(response.message.content or "")[:200],
                confidence=confidence,
                duration_ms=duration_ms,
                success=True,
            )

            # Local construction kept for parity with other typed-event sites.
            complete_event = SpecialistCompleteEvent(  # noqa: F841
                specialist_id=self.id,
                specialist_type=self.specialist_type,
                result=response.message.content,
                confidence=confidence,
                duration_ms=duration_ms,
            )

            return SpecialistResult(
                specialist_id=self.id,
                specialist_type=self.specialist_type,
                output=response.message.content,
                confidence=confidence,
                duration_ms=duration_ms,
                state=state,
            )

        except Exception as e:  # noqa: BLE001
            duration_ms = (time.perf_counter() - start_time) * 1000
            await emit(
                EV_SPECIALIST_COMPLETED,
                specialist_id=self.id,
                specialist_type=self.specialist_type,
                error=str(e),
                duration_ms=duration_ms,
                success=False,
            )
            return SpecialistResult(
                specialist_id=self.id,
                specialist_type=self.specialist_type,
                error=str(e),
                duration_ms=duration_ms,
                state=state,
            )

    def _format_context(self, context: dict[str, Any]) -> str:
        """Format context dictionary as a message."""
        lines = ["## Context from previous analysis:"]
        for key, value in context.items():
            lines.append(f"### {key}:")
            lines.append(str(value))
            lines.append("")
        return "\n".join(lines)

    def _estimate_confidence(self, response: str) -> float:
        """
        Estimate confidence from response text.

        This is a simple heuristic - could be enhanced with
        model-based confidence estimation.
        """
        response_lower = response.lower()

        # Indicators of high confidence
        high_confidence_markers = [
            "definitely",
            "certainly",
            "clearly",
            "confirmed",
            "verified",
            "established",
        ]

        # Indicators of low confidence
        low_confidence_markers = [
            "might",
            "possibly",
            "perhaps",
            "unclear",
            "uncertain",
            "unsure",
            "need more",
            "requires further",
        ]

        high_count = sum(1 for m in high_confidence_markers if m in response_lower)
        low_count = sum(1 for m in low_confidence_markers if m in response_lower)

        # Base confidence
        confidence = 0.5

        # Adjust based on markers
        confidence += high_count * 0.1
        confidence -= low_count * 0.1

        # Clamp to valid range
        return max(0.0, min(1.0, confidence))

    def with_model(self, model: Any) -> Specialist:
        """Return a copy of this specialist with the given model."""
        return self.model_copy(update={"model": model})


# =============================================================================
# Pre-built Specialist Types
# =============================================================================


def create_log_analyst(
    model: Any = None,
    tools: list[Tool] | None = None,
) -> Specialist:
    """Create a log analysis specialist."""
    return Specialist(
        name="Log Analyst",
        specialist_type="log_analyst",
        description="Specializes in analyzing log files, identifying patterns, and extracting insights from system logs.",
        system_prompt="""You are an expert log analyst. Your responsibilities:
1. Parse and understand various log formats (syslog, JSON, application logs)
2. Identify error patterns and anomalies
3. Correlate events across log entries
4. Extract actionable insights from log data
5. Summarize findings clearly

When analyzing logs:
- Look for error codes, stack traces, and exception messages
- Note timestamps and event sequences
- Identify recurring patterns
- Highlight severity levels""",
        tools=tools or [],
        model=model,
    )


def create_metrics_analyst(
    model: Any = None,
    tools: list[Tool] | None = None,
) -> Specialist:
    """Create a metrics analysis specialist."""
    return Specialist(
        name="Metrics Analyst",
        specialist_type="metrics_analyst",
        description="Specializes in analyzing system metrics, identifying anomalies, and understanding performance trends.",
        system_prompt="""You are an expert metrics analyst. Your responsibilities:
1. Analyze time-series metrics data
2. Identify anomalies and deviations from baselines
3. Understand correlations between different metrics
4. Assess system performance and health
5. Provide actionable recommendations

When analyzing metrics:
- Compare against historical baselines
- Look for sudden spikes or drops
- Identify correlating metrics
- Consider seasonality and trends""",
        tools=tools or [],
        model=model,
    )


def create_trace_analyst(
    model: Any = None,
    tools: list[Tool] | None = None,
) -> Specialist:
    """Create a distributed trace analysis specialist."""
    return Specialist(
        name="Trace Analyst",
        specialist_type="trace_analyst",
        description="Specializes in analyzing distributed traces, understanding service dependencies, and identifying latency issues.",
        system_prompt="""You are an expert distributed systems analyst. Your responsibilities:
1. Analyze distributed traces across services
2. Identify latency bottlenecks
3. Map service dependencies
4. Detect failed spans and error propagation
5. Understand request flow through the system

When analyzing traces:
- Follow the request path through services
- Identify slow spans and their causes
- Look for retry patterns
- Map the dependency graph""",
        tools=tools or [],
        model=model,
    )


def create_code_analyst(
    model: Any = None,
    tools: list[Tool] | None = None,
) -> Specialist:
    """Create a code analysis specialist."""
    return Specialist(
        name="Code Analyst",
        specialist_type="code_analyst",
        description="Specializes in analyzing source code, understanding implementations, and identifying potential issues.",
        system_prompt="""You are an expert code analyst. Your responsibilities:
1. Analyze source code for bugs and issues
2. Understand code flow and logic
3. Identify potential performance problems
4. Review error handling
5. Suggest improvements

When analyzing code:
- Trace execution paths
- Look for error handling gaps
- Identify resource leaks
- Check for common antipatterns""",
        tools=tools or [],
        model=model,
    )
