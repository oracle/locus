# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Pydantic models for playbook definitions and execution plans."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class StepStatus(StrEnum):
    """Status of a playbook step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class PlaybookStep(BaseModel):
    """Individual step in a playbook.

    Defines what tools are expected, hints for the agent,
    and optional validation criteria.
    """

    id: str = Field(..., description="Unique identifier for the step")
    description: str = Field(..., description="Human-readable description of the step")
    expected_tools: list[str] = Field(
        default_factory=list,
        description="Tools expected to be called during this step",
    )
    hints: list[str] = Field(
        default_factory=list,
        description="Hints to provide the agent for this step",
    )
    required: bool = Field(
        default=True,
        description="Whether this step is required or optional",
    )
    validation: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional validation criteria for step completion",
    )
    max_tool_calls: int | None = Field(
        default=None,
        description="Maximum number of tool calls allowed for this step",
    )
    timeout_seconds: float | None = Field(
        default=None,
        description="Optional timeout for this step in seconds",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata for the step",
    )

    model_config = {"frozen": True}


class Playbook(BaseModel):
    """Collection of steps that define an execution plan.

    A playbook provides structure for agent execution, defining
    the expected sequence of operations and validation criteria.
    """

    id: str = Field(..., description="Unique identifier for the playbook")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(default="", description="Detailed description of the playbook")
    version: str = Field(default="1.0.0", description="Semantic version of the playbook")
    steps: list[PlaybookStep] = Field(
        default_factory=list,
        description="Ordered list of steps to execute",
    )
    strict_sequence: bool = Field(
        default=True,
        description="Whether steps must be executed in order",
    )
    allow_extra_tools: bool = Field(
        default=False,
        description="Whether tools not in expected_tools are allowed",
    )
    max_iterations: int | None = Field(
        default=None,
        description="Maximum iterations for the entire playbook",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata for the playbook",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorization",
    )

    model_config = {"frozen": True}

    def get_step(self, step_id: str) -> PlaybookStep | None:
        """Get a step by its ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def get_step_index(self, step_id: str) -> int | None:
        """Get the index of a step by its ID."""
        for i, step in enumerate(self.steps):
            if step.id == step_id:
                return i
        return None


class StepExecution(BaseModel):
    """Record of a single step's execution."""

    step_id: str = Field(..., description="ID of the step")
    status: StepStatus = Field(default=StepStatus.PENDING, description="Current status")
    started_at: datetime | None = Field(default=None, description="When execution started")
    completed_at: datetime | None = Field(default=None, description="When execution completed")
    tool_calls: list[str] = Field(
        default_factory=list,
        description="Tool calls made during this step",
    )
    tool_call_count: int = Field(default=0, description="Number of tool calls made")
    error: str | None = Field(default=None, description="Error message if failed")
    result: str | None = Field(default=None, description="Result of the step")


class PlaybookPlan(BaseModel):
    """Active execution plan for a playbook.

    Tracks progress through the playbook, including which steps
    have been completed, current step, and any deviations.
    """

    playbook: Playbook = Field(..., description="The playbook being executed")
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When execution started",
    )
    current_step_index: int = Field(default=0, description="Index of current step")
    step_executions: dict[str, StepExecution] = Field(
        default_factory=dict,
        description="Execution records by step ID",
    )
    completed: bool = Field(default=False, description="Whether the plan is complete")
    total_tool_calls: int = Field(default=0, description="Total tool calls across all steps")
    errors: list[str] = Field(default_factory=list, description="Accumulated errors")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Runtime metadata",
    )

    model_config = {"arbitrary_types_allowed": True}

    @property
    def current_step(self) -> PlaybookStep | None:
        """Get the current step."""
        if 0 <= self.current_step_index < len(self.playbook.steps):
            return self.playbook.steps[self.current_step_index]
        return None

    @property
    def progress(self) -> float:
        """Calculate progress as a percentage (0.0 to 1.0)."""
        if not self.playbook.steps:
            return 1.0
        completed = sum(
            1 for se in self.step_executions.values() if se.status == StepStatus.COMPLETED
        )
        return completed / len(self.playbook.steps)

    @property
    def completed_steps(self) -> list[str]:
        """Get IDs of completed steps."""
        return [
            step_id
            for step_id, se in self.step_executions.items()
            if se.status == StepStatus.COMPLETED
        ]

    @property
    def pending_steps(self) -> list[str]:
        """Get IDs of pending steps."""
        completed = set(self.completed_steps)
        return [step.id for step in self.playbook.steps if step.id not in completed]

    def get_step_execution(self, step_id: str) -> StepExecution | None:
        """Get execution record for a step."""
        return self.step_executions.get(step_id)

    def is_step_complete(self, step_id: str) -> bool:
        """Check if a step is complete."""
        se = self.step_executions.get(step_id)
        return se is not None and se.status == StepStatus.COMPLETED
