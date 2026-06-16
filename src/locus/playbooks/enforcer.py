# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Playbook execution enforcement."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

from locus.playbooks.models import (
    Playbook,
    PlaybookPlan,
    PlaybookStep,
    StepExecution,
    StepStatus,
)


class EnforcementViolation(BaseModel):
    """Record of an enforcement violation."""

    violation_type: str = Field(..., description="Type of violation")
    step_id: str | None = Field(default=None, description="Step ID where violation occurred")
    tool_name: str | None = Field(default=None, description="Tool that caused violation")
    message: str = Field(..., description="Human-readable message")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    blocked: bool = Field(default=False, description="Whether the action was blocked")


class EnforcementResult(BaseModel):
    """Result of an enforcement check."""

    allowed: bool = Field(..., description="Whether the action is allowed")
    violation: EnforcementViolation | None = Field(
        default=None,
        description="Violation details if not allowed",
    )
    hints: list[str] = Field(
        default_factory=list,
        description="Hints for the agent",
    )
    current_step: PlaybookStep | None = Field(
        default=None,
        description="Current step being executed",
    )


class PlaybookEnforcer(BaseModel):
    """Enforces playbook execution sequence and constraints.

    The enforcer tracks progress through a playbook, validates tool calls,
    and provides hints to guide the agent through the execution plan.

    Features:
    - Track completed steps
    - Validate tool calls match current step's expected tools
    - Provide hints for the next step
    - Block out-of-sequence execution when strict_sequence is True
    - Record violations for auditing
    """

    plan: PlaybookPlan = Field(..., description="Active execution plan")
    block_violations: bool = Field(
        default=True,
        description="Whether to block violating tool calls",
    )
    record_violations: bool = Field(
        default=True,
        description="Whether to record violations",
    )

    _violations: list[EnforcementViolation] = PrivateAttr(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_playbook(
        cls,
        playbook: Playbook,
        block_violations: bool = True,
        record_violations: bool = True,
    ) -> PlaybookEnforcer:
        """Create an enforcer from a playbook.

        Args:
            playbook: The playbook to enforce
            block_violations: Whether to block violating tool calls
            record_violations: Whether to record violations

        Returns:
            Configured PlaybookEnforcer
        """
        plan = PlaybookPlan(playbook=playbook)
        return cls(
            plan=plan,
            block_violations=block_violations,
            record_violations=record_violations,
        )

    @property
    def violations(self) -> list[EnforcementViolation]:
        """Get recorded violations."""
        return list(self._violations)

    @property
    def current_step(self) -> PlaybookStep | None:
        """Get the current step."""
        return self.plan.current_step

    @property
    def current_step_hints(self) -> list[str]:
        """Get hints for the current step."""
        step = self.current_step
        return list(step.hints) if step else []

    @property
    def progress(self) -> float:
        """Get execution progress (0.0 to 1.0)."""
        return self.plan.progress

    @property
    def is_complete(self) -> bool:
        """Check if the playbook execution is complete."""
        return self.plan.completed

    def validate_tool_call(self, tool_name: str) -> EnforcementResult:
        """Validate a tool call against the current step.

        Args:
            tool_name: Name of the tool being called

        Returns:
            EnforcementResult indicating whether the call is allowed
        """
        step = self.current_step

        # No more steps - check if extra tools are allowed
        if step is None:
            if self.plan.completed:
                return EnforcementResult(
                    allowed=self.plan.playbook.allow_extra_tools,
                    violation=self._maybe_record_violation(
                        "playbook_complete",
                        None,
                        tool_name,
                        f"Playbook is complete, tool '{tool_name}' called after completion",
                        blocked=self.block_violations and not self.plan.playbook.allow_extra_tools,
                    )
                    if not self.plan.playbook.allow_extra_tools
                    else None,
                    hints=["Playbook execution is complete"],
                )
            return EnforcementResult(allowed=True)

        # Check if tool is in expected tools
        if step.expected_tools and tool_name not in step.expected_tools:
            # Tool not expected for this step
            if self.plan.playbook.allow_extra_tools:
                return EnforcementResult(
                    allowed=True,
                    hints=step.hints,
                    current_step=step,
                )

            violation = self._maybe_record_violation(
                "unexpected_tool",
                step.id,
                tool_name,
                f"Tool '{tool_name}' not expected for step '{step.id}'. "
                f"Expected: {step.expected_tools}",
                blocked=self.block_violations,
            )

            return EnforcementResult(
                allowed=not self.block_violations,
                violation=violation,
                hints=[
                    f"Current step expects: {', '.join(step.expected_tools)}",
                    *step.hints,
                ],
                current_step=step,
            )

        # Check max tool calls for step
        step_exec = self.plan.step_executions.get(step.id)
        if step.max_tool_calls is not None and step_exec:
            if step_exec.tool_call_count >= step.max_tool_calls:
                violation = self._maybe_record_violation(
                    "max_tool_calls",
                    step.id,
                    tool_name,
                    f"Step '{step.id}' has reached max tool calls ({step.max_tool_calls})",
                    blocked=self.block_violations,
                )
                return EnforcementResult(
                    allowed=not self.block_violations,
                    violation=violation,
                    hints=["Consider moving to the next step"],
                    current_step=step,
                )

        return EnforcementResult(
            allowed=True,
            hints=step.hints,
            current_step=step,
        )

    def record_tool_call(self, tool_name: str) -> None:
        """Record that a tool was called.

        Updates the step execution tracking.

        Args:
            tool_name: Name of the tool that was called
        """
        step = self.current_step
        if step is None:
            self.plan.total_tool_calls += 1
            return

        # Get or create step execution
        if step.id not in self.plan.step_executions:
            self.plan.step_executions[step.id] = StepExecution(
                step_id=step.id,
                status=StepStatus.IN_PROGRESS,
                started_at=datetime.now(UTC),
            )

        step_exec = self.plan.step_executions[step.id]
        step_exec.tool_calls.append(tool_name)
        step_exec.tool_call_count += 1
        self.plan.total_tool_calls += 1

    def complete_current_step(self, result: str | None = None) -> bool:
        """Mark the current step as complete and advance.

        Args:
            result: Optional result to record for the step

        Returns:
            True if advanced to next step, False if playbook is complete
        """
        step = self.current_step
        if step is None:
            return False

        # Get or create step execution
        if step.id not in self.plan.step_executions:
            self.plan.step_executions[step.id] = StepExecution(
                step_id=step.id,
                status=StepStatus.COMPLETED,
                started_at=datetime.now(UTC),
            )

        step_exec = self.plan.step_executions[step.id]
        step_exec.status = StepStatus.COMPLETED
        step_exec.completed_at = datetime.now(UTC)
        step_exec.result = result

        # Advance to next step
        self.plan.current_step_index += 1

        # Check if playbook is complete
        if self.plan.current_step_index >= len(self.plan.playbook.steps):
            self.plan.completed = True
            return False

        return True

    def skip_current_step(self, reason: str | None = None) -> bool:
        """Skip the current step.

        Only works for non-required steps.

        Args:
            reason: Optional reason for skipping

        Returns:
            True if step was skipped, False if step is required
        """
        step = self.current_step
        if step is None:
            return False

        if step.required:
            return False

        # Record as skipped
        if step.id not in self.plan.step_executions:
            self.plan.step_executions[step.id] = StepExecution(
                step_id=step.id,
                status=StepStatus.SKIPPED,
            )
        else:
            self.plan.step_executions[step.id].status = StepStatus.SKIPPED

        if reason:
            self.plan.step_executions[step.id].result = reason

        # Advance
        self.plan.current_step_index += 1

        if self.plan.current_step_index >= len(self.plan.playbook.steps):
            self.plan.completed = True
            return True

        return True

    def fail_current_step(self, error: str) -> None:
        """Mark the current step as failed.

        Args:
            error: Error message
        """
        step = self.current_step
        if step is None:
            return

        if step.id not in self.plan.step_executions:
            self.plan.step_executions[step.id] = StepExecution(
                step_id=step.id,
                status=StepStatus.FAILED,
                started_at=datetime.now(UTC),
            )

        step_exec = self.plan.step_executions[step.id]
        step_exec.status = StepStatus.FAILED
        step_exec.completed_at = datetime.now(UTC)
        step_exec.error = error

        self.plan.errors.append(f"Step {step.id}: {error}")

    def get_next_step_hints(self) -> list[str]:
        """Get hints for the next step after current.

        Useful for looking ahead during execution.

        Returns:
            List of hints for the next step, or empty if no next step
        """
        next_index = self.plan.current_step_index + 1
        if next_index < len(self.plan.playbook.steps):
            return list(self.plan.playbook.steps[next_index].hints)
        return []

    def get_step_summary(self) -> dict[str, Any]:
        """Get a summary of step execution status.

        Returns:
            Dictionary with step status summary
        """
        steps = self.plan.playbook.steps
        return {
            "total_steps": len(steps),
            "current_step_index": self.plan.current_step_index,
            "completed": len(
                [s for s in self.plan.step_executions.values() if s.status == StepStatus.COMPLETED]
            ),
            "skipped": len(
                [s for s in self.plan.step_executions.values() if s.status == StepStatus.SKIPPED]
            ),
            "failed": len(
                [s for s in self.plan.step_executions.values() if s.status == StepStatus.FAILED]
            ),
            "pending": len(steps) - len(self.plan.step_executions),
            "progress": self.progress,
            "is_complete": self.is_complete,
        }

    def _maybe_record_violation(
        self,
        violation_type: str,
        step_id: str | None,
        tool_name: str | None,
        message: str,
        blocked: bool,
    ) -> EnforcementViolation | None:
        """Record a violation if recording is enabled."""
        if not self.record_violations:
            return None

        violation = EnforcementViolation(
            violation_type=violation_type,
            step_id=step_id,
            tool_name=tool_name,
            message=message,
            blocked=blocked,
        )
        self._violations.append(violation)
        return violation

    def reset(self) -> None:
        """Reset the enforcer to start over."""
        self.plan.current_step_index = 0
        self.plan.step_executions.clear()
        self.plan.completed = False
        self.plan.total_tool_calls = 0
        self.plan.errors.clear()
        self._violations.clear()
