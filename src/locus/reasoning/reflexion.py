# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Reflexion pattern implementation for iterative self-improvement.

The Reflexion pattern enables agents to evaluate their own progress and adjust
strategy based on tool results, detecting loops, and building confidence.
"""

from __future__ import annotations

from collections import Counter
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from locus.core.state import AgentState, ToolExecution


class AssessmentCategory(StrEnum):
    """Categories for agent progress assessment."""

    ON_TRACK = "on_track"
    STUCK = "stuck"
    NEW_FINDINGS = "new_findings"
    LOOP_DETECTED = "loop_detected"


class ReflectionResult(BaseModel):
    """Result of reflecting on agent progress.

    Attributes:
        confidence_delta: Adjustment to confidence score (-1.0 to 1.0).
        assessment: Category of agent's current progress.
        guidance: Suggestions for the next iteration.
        loop_pattern: Detected loop pattern if assessment is loop_detected.
        findings_summary: Summary of new information discovered.
    """

    confidence_delta: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Adjustment to confidence score",
    )
    assessment: AssessmentCategory = Field(
        default=AssessmentCategory.ON_TRACK,
        description="Category of agent's current progress",
    )
    guidance: str | None = Field(
        default=None,
        description="Suggestions for the next iteration",
    )
    loop_pattern: str | None = Field(
        default=None,
        description="Detected loop pattern if assessment is loop_detected",
    )
    findings_summary: str | None = Field(
        default=None,
        description="Summary of new information discovered",
    )

    model_config = {"frozen": True}


class Reflector:
    """Evaluates agent progress after each iteration.

    The Reflector analyzes tool execution patterns, results, and state
    to determine if the agent is making progress toward its goal.

    Attributes:
        loop_threshold: Number of repeated tool calls to consider a loop.
        success_weight: Weight for successful tool executions in confidence.
        error_penalty: Penalty for failed tool executions.
        diminishing_returns: Whether to apply diminishing returns to confidence.
        min_progress_delta: Minimum confidence delta for "on_track" assessment.
    """

    def __init__(
        self,
        loop_threshold: int = 3,
        success_weight: float = 0.15,
        error_penalty: float = 0.2,
        diminishing_returns: bool = True,
        min_progress_delta: float = 0.05,
        completion_bonus: float = 0.05,
    ) -> None:
        """Initialize the Reflector.

        Args:
            loop_threshold: Number of repeated tool calls to consider a loop.
            success_weight: Base confidence increase per successful tool call.
            error_penalty: Confidence decrease per failed tool call.
            diminishing_returns: Apply diminishing returns to positive deltas.
            min_progress_delta: Minimum delta to consider making progress.
            completion_bonus: Small confidence bump applied when an
                iteration produces an assistant turn but no tools fired
                (i.e. a successful chat reply). Without this, tool-less
                chat agents would never raise their confidence above
                ``0.0``. Set to ``0.0`` to opt out.
        """
        self.loop_threshold = loop_threshold
        self.success_weight = success_weight
        self.error_penalty = error_penalty
        self.diminishing_returns = diminishing_returns
        self.min_progress_delta = min_progress_delta
        self.completion_bonus = completion_bonus

    def reflect(
        self,
        state: AgentState,
        iteration_executions: list[ToolExecution] | None = None,
    ) -> ReflectionResult:
        """Evaluate agent progress and produce reflection result.

        Args:
            state: Current agent state with history.
            iteration_executions: Tool executions from the current iteration.
                If None, uses the most recent executions from state.

        Returns:
            ReflectionResult with assessment and guidance.
        """
        # Get executions for this iteration if not provided
        if iteration_executions is None:
            iteration_executions = self._get_recent_executions(state)

        # Check for loops first (highest priority)
        loop_result = self._detect_loop(state)
        if loop_result is not None:
            return loop_result

        # Analyze tool execution results
        success_count, error_count, results_content = self._analyze_executions(iteration_executions)

        # Calculate base confidence delta
        confidence_delta = self._calculate_confidence_delta(
            success_count,
            error_count,
            state.confidence,
        )

        # Determine assessment category and guidance
        assessment, guidance, findings = self._assess_progress(
            confidence_delta,
            success_count,
            error_count,
            results_content,
            state,
        )

        return ReflectionResult(
            confidence_delta=confidence_delta,
            assessment=assessment,
            guidance=guidance,
            findings_summary=findings,
        )

    def _get_recent_executions(
        self,
        state: AgentState,
    ) -> list[ToolExecution]:
        """Get executions from the most recent iteration."""
        if not state.tool_executions:
            return []

        # Find executions from the current iteration
        # (those with timestamps matching the most recent)
        all_executions = list(state.tool_executions)
        if not all_executions:
            return []

        # Group by approximate time (same iteration)
        # For simplicity, return executions that match the current iteration count
        # Note: state.iteration is available if needed for filtering
        recent: list[ToolExecution] = []

        # Walk backwards to find executions from this iteration
        # This is approximate; in production you'd track iteration per execution
        for execution in reversed(all_executions):
            recent.append(execution)
            # Assume each iteration has at most a few tool calls
            if len(recent) >= 5:
                break

        return list(reversed(recent))

    def _detect_loop(self, state: AgentState) -> ReflectionResult | None:
        """Detect if the agent is stuck in a tool loop.

        Checks across iterations, not within a single iteration's parallel
        calls. Multiple calls to the same tool in ONE turn (parallel) is
        normal research behavior, not a loop.

        Returns ReflectionResult if loop detected, None otherwise.
        """
        # Build per-iteration tool sets to detect cross-iteration loops
        # A loop is: the SAME set of tools called across consecutive iterations
        if state.iteration < self.loop_threshold:
            return None

        # Group tool executions by approximate iteration
        # Use reasoning_steps as iteration markers (one per iteration)
        if len(state.reasoning_steps) < self.loop_threshold:
            # Fallback to simple tool_history check for backward compat
            if len(state.tool_history) < self.loop_threshold:
                return None
            recent_tools = state.tool_history[-self.loop_threshold :]
            tool_counts = Counter(recent_tools)
            most_common = tool_counts.most_common(1)
            if most_common and most_common[0][1] == self.loop_threshold:
                # But only flag if these are from DIFFERENT iterations
                # (not parallel calls in one turn)
                if len(state.reasoning_steps) >= self.loop_threshold:
                    pattern = f"Tool '{most_common[0][0]}' called across {self.loop_threshold} consecutive iterations"
                    return ReflectionResult(
                        confidence_delta=-0.3,
                        assessment=AssessmentCategory.LOOP_DETECTED,
                        guidance=self._generate_loop_guidance(most_common[0][0], state),
                        loop_pattern=pattern,
                    )
            return None

        # Check if the last N iterations used the exact same tool set
        recent_steps = state.reasoning_steps[-self.loop_threshold :]
        tool_sets = []
        for step in recent_steps:
            if step.tool_calls:
                tool_set = frozenset(tc.name for tc in step.tool_calls)
                tool_sets.append(tool_set)

        if len(tool_sets) >= self.loop_threshold and len(set(tool_sets)) == 1:
            tool_names = ", ".join(sorted(tool_sets[0]))
            pattern = f"Same tools ({tool_names}) called across {self.loop_threshold} consecutive iterations"
            return ReflectionResult(
                confidence_delta=-0.3,
                assessment=AssessmentCategory.LOOP_DETECTED,
                guidance=self._generate_loop_guidance(tool_names, state),
                loop_pattern=pattern,
            )

        # Check for alternating pattern (A->B->A->B)
        if self.loop_threshold >= 4 and len(tool_sets) >= 4:
            if (
                tool_sets[-4] == tool_sets[-2]
                and tool_sets[-3] == tool_sets[-1]
                and tool_sets[-4] != tool_sets[-3]
            ):
                pattern = (
                    f"Alternating pattern: {sorted(tool_sets[-4])} <-> {sorted(tool_sets[-3])}"
                )
                return ReflectionResult(
                    confidence_delta=-0.25,
                    assessment=AssessmentCategory.LOOP_DETECTED,
                    guidance="Detected alternating tool pattern across iterations. Consider a different approach.",
                    loop_pattern=pattern,
                )

        return None

    def _detect_alternating_pattern(
        self,
        state: AgentState,
    ) -> ReflectionResult | None:
        """Detect alternating tool patterns like A->B->A->B."""
        recent = state.tool_history[-4:]
        if len(recent) < 4:
            return None

        # Check for A-B-A-B pattern
        if recent[0] == recent[2] and recent[1] == recent[3] and recent[0] != recent[1]:
            pattern = f"Alternating pattern: {recent[0]} <-> {recent[1]}"
            return ReflectionResult(
                confidence_delta=-0.25,
                assessment=AssessmentCategory.LOOP_DETECTED,
                guidance=(
                    f"Detected alternating loop between '{recent[0]}' and '{recent[1]}'. "
                    "Consider a different approach or gathering additional context before proceeding."
                ),
                loop_pattern=pattern,
            )

        return None

    def _analyze_executions(
        self,
        executions: list[ToolExecution],
    ) -> tuple[int, int, list[str]]:
        """Analyze tool executions to count successes, errors, and gather content.

        Returns:
            Tuple of (success_count, error_count, result_contents).
        """
        success_count = 0
        error_count = 0
        results_content: list[str] = []

        for execution in executions:
            if execution.success:
                success_count += 1
                if execution.result:
                    results_content.append(execution.result)
            else:
                error_count += 1

        return success_count, error_count, results_content

    def _calculate_confidence_delta(
        self,
        success_count: int,
        error_count: int,
        current_confidence: float,
    ) -> float:
        """Calculate the confidence adjustment based on execution results.

        Args:
            success_count: Number of successful tool executions.
            error_count: Number of failed tool executions.
            current_confidence: Current confidence level (0.0 to 1.0).

        Returns:
            Confidence delta (-1.0 to 1.0).
        """
        # Base delta from successes and errors. Iterations with no tool
        # activity get a small completion_bonus so chat-only agents still
        # show a rising confidence trajectory over a multi-turn dialogue.
        if success_count == 0 and error_count == 0:
            raw_delta = self.completion_bonus
        else:
            raw_delta = (success_count * self.success_weight) - (error_count * self.error_penalty)

        # Apply diminishing returns for positive deltas
        if self.diminishing_returns and raw_delta > 0:
            # As confidence increases, gains decrease
            effective_delta = raw_delta * (1.0 - current_confidence)
        else:
            effective_delta = raw_delta

        # Clamp to valid range
        return max(-1.0, min(1.0, effective_delta))

    def _assess_progress(
        self,
        confidence_delta: float,
        success_count: int,
        error_count: int,
        results_content: list[str],
        state: AgentState,
    ) -> tuple[AssessmentCategory, str | None, str | None]:
        """Determine assessment category and generate guidance.

        Returns:
            Tuple of (assessment, guidance, findings_summary).
        """
        # Check if we got new findings (substantial results)
        has_findings = self._has_new_findings(results_content)

        # Assess based on delta and results
        if has_findings and success_count > 0:
            findings_summary = self._summarize_findings(results_content)
            return (
                AssessmentCategory.NEW_FINDINGS,
                "New information discovered. Continue analyzing the findings.",
                findings_summary,
            )

        if confidence_delta >= self.min_progress_delta:
            return (
                AssessmentCategory.ON_TRACK,
                None,  # No guidance needed when on track
                None,
            )

        if error_count > success_count or confidence_delta < -self.min_progress_delta:
            guidance = self._generate_stuck_guidance(error_count, state)
            return (
                AssessmentCategory.STUCK,
                guidance,
                None,
            )

        # Default to on_track with minimal progress
        return (
            AssessmentCategory.ON_TRACK,
            "Progress is slow. Consider alternative approaches if no improvement.",
            None,
        )

    def _has_new_findings(self, results_content: list[str]) -> bool:
        """Determine if results contain substantial new findings."""
        if not results_content:
            return False

        # Check for non-trivial content
        total_content = "".join(results_content)
        # Heuristic: significant findings have meaningful content
        return len(total_content) > 100

    def _summarize_findings(self, results_content: list[str]) -> str:
        """Create a brief summary of findings."""
        if not results_content:
            return ""

        # Simple summary: first 200 chars of combined content
        combined = " ".join(results_content)
        if len(combined) <= 200:
            return combined
        return combined[:197] + "..."

    def _generate_loop_guidance(self, tool_name: str, state: AgentState) -> str:
        """Generate guidance for escaping a tool loop."""
        guidance_parts = [
            f"The tool '{tool_name}' has been called repeatedly without progress.",
            "Consider the following:",
            "1. Use a different tool to gather new information",
            "2. Review the tool arguments for potential issues",
            "3. If the task cannot be completed, report findings and limitations",
        ]

        # Check if there are errors in recent executions
        recent_errors = [e for e in state.tool_executions[-self.loop_threshold :] if e.error]
        if recent_errors:
            guidance_parts.append(f"4. Address the error: {recent_errors[-1].error}")

        return " ".join(guidance_parts)

    def _generate_stuck_guidance(self, error_count: int, state: AgentState) -> str:
        """Generate guidance when the agent is stuck."""
        if error_count > 0:
            # Get the most recent error
            recent_errors = [e for e in state.tool_executions if e.error]
            if recent_errors:
                return (
                    f"Encountering errors ({error_count} in this iteration). "
                    f"Last error: {recent_errors[-1].error}. "
                    "Consider adjusting approach or trying alternative tools."
                )

        return (
            "Progress has stalled. Consider: "
            "1) Using different tools, "
            "2) Reformulating the approach, "
            "3) Breaking the problem into smaller steps."
        )

    def adjust_state_confidence(
        self,
        state: AgentState,
        reflection: ReflectionResult,
    ) -> AgentState:
        """Apply reflection result to update agent state confidence.

        Uses the AgentState.adjust_confidence pattern for consistency.

        Args:
            state: Current agent state.
            reflection: Reflection result with confidence delta.

        Returns:
            New state with updated confidence.
        """
        return state.adjust_confidence(
            reflection.confidence_delta,
            diminishing=self.diminishing_returns,
        )

    def create_guidance_message(
        self,
        reflection: ReflectionResult,
    ) -> str | None:
        """Create a guidance message to inject into the next iteration.

        Args:
            reflection: Reflection result with assessment and guidance.

        Returns:
            Formatted guidance message or None if no guidance needed.
        """
        if reflection.guidance is None:
            return None

        parts = [f"[Reflection - {reflection.assessment.value}]"]
        parts.append(reflection.guidance)

        if reflection.loop_pattern:
            parts.append(f"Pattern detected: {reflection.loop_pattern}")

        if reflection.findings_summary:
            parts.append(f"Key findings: {reflection.findings_summary}")

        return "\n".join(parts)


def evaluate_progress(
    state: AgentState,
    executions: list[ToolExecution] | None = None,
    **reflector_kwargs: Any,
) -> ReflectionResult:
    """Convenience function to evaluate agent progress.

    Args:
        state: Current agent state.
        executions: Optional list of executions from current iteration.
        **reflector_kwargs: Arguments passed to Reflector constructor.

    Returns:
        ReflectionResult with assessment and guidance.
    """
    reflector = Reflector(**reflector_kwargs)
    return reflector.reflect(state, executions)
