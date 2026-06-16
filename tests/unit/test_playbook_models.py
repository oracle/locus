# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for playbook models."""

from datetime import UTC, datetime

import pytest

from locus.playbooks.models import (
    Playbook,
    PlaybookPlan,
    PlaybookStep,
    StepExecution,
    StepStatus,
)


class TestStepStatus:
    """Tests for StepStatus enum."""

    def test_all_statuses_exist(self):
        """Test all status values exist."""
        assert StepStatus.PENDING == "pending"
        assert StepStatus.IN_PROGRESS == "in_progress"
        assert StepStatus.COMPLETED == "completed"
        assert StepStatus.SKIPPED == "skipped"
        assert StepStatus.FAILED == "failed"


class TestPlaybookStep:
    """Tests for PlaybookStep model."""

    def test_create_minimal_step(self):
        """Test creating step with minimal fields."""
        step = PlaybookStep(
            id="step1",
            description="Test step",
        )
        assert step.id == "step1"
        assert step.description == "Test step"
        assert step.expected_tools == []
        assert step.hints == []
        assert step.required is True

    def test_create_full_step(self):
        """Test creating step with all fields."""
        step = PlaybookStep(
            id="step1",
            description="Full step",
            expected_tools=["tool_a", "tool_b"],
            hints=["Hint 1", "Hint 2"],
            required=False,
            validation={"type": "string"},
            max_tool_calls=5,
            timeout_seconds=30.0,
            metadata={"key": "value"},
        )
        assert step.expected_tools == ["tool_a", "tool_b"]
        assert step.hints == ["Hint 1", "Hint 2"]
        assert step.required is False
        assert step.max_tool_calls == 5
        assert step.timeout_seconds == 30.0

    def test_step_is_frozen(self):
        """Test that step is immutable."""
        from pydantic import ValidationError

        step = PlaybookStep(id="step1", description="Test")
        with pytest.raises(ValidationError, match="frozen"):
            step.id = "new_id"


class TestPlaybook:
    """Tests for Playbook model."""

    def test_create_minimal_playbook(self):
        """Test creating playbook with minimal fields."""
        playbook = Playbook(id="rb1", name="Test Playbook")
        assert playbook.id == "rb1"
        assert playbook.name == "Test Playbook"
        assert playbook.description == ""
        assert playbook.version == "1.0.0"
        assert playbook.steps == []
        assert playbook.strict_sequence is True
        assert playbook.allow_extra_tools is False

    def test_create_full_playbook(self):
        """Test creating playbook with all fields."""
        steps = [
            PlaybookStep(id="s1", description="Step 1"),
            PlaybookStep(id="s2", description="Step 2"),
        ]
        playbook = Playbook(
            id="rb1",
            name="Full Playbook",
            description="A complete playbook",
            version="2.0.0",
            steps=steps,
            strict_sequence=False,
            allow_extra_tools=True,
            max_iterations=10,
            metadata={"author": "test"},
            tags=["test", "demo"],
        )
        assert len(playbook.steps) == 2
        assert playbook.strict_sequence is False
        assert playbook.max_iterations == 10
        assert playbook.tags == ["test", "demo"]

    def test_get_step(self):
        """Test getting step by ID."""
        steps = [
            PlaybookStep(id="s1", description="Step 1"),
            PlaybookStep(id="s2", description="Step 2"),
        ]
        playbook = Playbook(id="rb1", name="Test", steps=steps)

        step = playbook.get_step("s1")
        assert step is not None
        assert step.id == "s1"

        step = playbook.get_step("nonexistent")
        assert step is None

    def test_get_step_index(self):
        """Test getting step index by ID."""
        steps = [
            PlaybookStep(id="s1", description="Step 1"),
            PlaybookStep(id="s2", description="Step 2"),
        ]
        playbook = Playbook(id="rb1", name="Test", steps=steps)

        index = playbook.get_step_index("s1")
        assert index == 0

        index = playbook.get_step_index("s2")
        assert index == 1

        index = playbook.get_step_index("nonexistent")
        assert index is None

    def test_playbook_is_frozen(self):
        """Test that playbook is immutable."""
        from pydantic import ValidationError

        playbook = Playbook(id="rb1", name="Test")
        with pytest.raises(ValidationError, match="frozen"):
            playbook.id = "new_id"


class TestStepExecution:
    """Tests for StepExecution model."""

    def test_create_default_execution(self):
        """Test creating execution with defaults."""
        execution = StepExecution(step_id="s1")
        assert execution.step_id == "s1"
        assert execution.status == StepStatus.PENDING
        assert execution.started_at is None
        assert execution.tool_calls == []
        assert execution.tool_call_count == 0
        assert execution.error is None

    def test_create_completed_execution(self):
        """Test creating completed execution."""
        now = datetime.now(UTC)
        execution = StepExecution(
            step_id="s1",
            status=StepStatus.COMPLETED,
            started_at=now,
            completed_at=now,
            tool_calls=["tool_a", "tool_b"],
            tool_call_count=2,
            result="Success",
        )
        assert execution.status == StepStatus.COMPLETED
        assert len(execution.tool_calls) == 2

    def test_create_failed_execution(self):
        """Test creating failed execution."""
        execution = StepExecution(
            step_id="s1",
            status=StepStatus.FAILED,
            error="Something went wrong",
        )
        assert execution.status == StepStatus.FAILED
        assert execution.error == "Something went wrong"


class TestPlaybookPlan:
    """Tests for PlaybookPlan model."""

    @pytest.fixture
    def playbook(self):
        """Create a test playbook."""
        return Playbook(
            id="rb1",
            name="Test Playbook",
            steps=[
                PlaybookStep(id="s1", description="Step 1"),
                PlaybookStep(id="s2", description="Step 2"),
                PlaybookStep(id="s3", description="Step 3"),
            ],
        )

    def test_create_plan(self, playbook):
        """Test creating a plan."""
        plan = PlaybookPlan(playbook=playbook)
        assert plan.playbook is playbook
        assert plan.current_step_index == 0
        assert plan.completed is False
        assert plan.total_tool_calls == 0

    def test_current_step(self, playbook):
        """Test getting current step."""
        plan = PlaybookPlan(playbook=playbook)
        assert plan.current_step is not None
        assert plan.current_step.id == "s1"

        plan = PlaybookPlan(playbook=playbook, current_step_index=1)
        assert plan.current_step.id == "s2"

    def test_current_step_out_of_bounds(self, playbook):
        """Test current step when index is out of bounds."""
        plan = PlaybookPlan(playbook=playbook, current_step_index=10)
        assert plan.current_step is None

    def test_progress_no_steps(self):
        """Test progress with no steps."""
        playbook = Playbook(id="rb1", name="Empty")
        plan = PlaybookPlan(playbook=playbook)
        assert plan.progress == 1.0

    def test_progress_no_completions(self, playbook):
        """Test progress with no completed steps."""
        plan = PlaybookPlan(playbook=playbook)
        assert plan.progress == 0.0

    def test_progress_partial(self, playbook):
        """Test progress with some completed steps."""
        plan = PlaybookPlan(
            playbook=playbook,
            step_executions={
                "s1": StepExecution(step_id="s1", status=StepStatus.COMPLETED),
            },
        )
        assert plan.progress == pytest.approx(1 / 3)

    def test_progress_all_complete(self, playbook):
        """Test progress with all steps completed."""
        plan = PlaybookPlan(
            playbook=playbook,
            step_executions={
                "s1": StepExecution(step_id="s1", status=StepStatus.COMPLETED),
                "s2": StepExecution(step_id="s2", status=StepStatus.COMPLETED),
                "s3": StepExecution(step_id="s3", status=StepStatus.COMPLETED),
            },
        )
        assert plan.progress == 1.0

    def test_completed_steps(self, playbook):
        """Test getting completed steps."""
        plan = PlaybookPlan(
            playbook=playbook,
            step_executions={
                "s1": StepExecution(step_id="s1", status=StepStatus.COMPLETED),
                "s2": StepExecution(step_id="s2", status=StepStatus.IN_PROGRESS),
            },
        )
        completed = plan.completed_steps
        assert "s1" in completed
        assert "s2" not in completed

    def test_pending_steps(self, playbook):
        """Test getting pending steps."""
        plan = PlaybookPlan(
            playbook=playbook,
            step_executions={
                "s1": StepExecution(step_id="s1", status=StepStatus.COMPLETED),
            },
        )
        pending = plan.pending_steps
        assert "s1" not in pending
        assert "s2" in pending
        assert "s3" in pending

    def test_get_step_execution(self, playbook):
        """Test getting step execution."""
        execution = StepExecution(step_id="s1", status=StepStatus.COMPLETED)
        plan = PlaybookPlan(
            playbook=playbook,
            step_executions={"s1": execution},
        )

        result = plan.get_step_execution("s1")
        assert result is execution

        result = plan.get_step_execution("nonexistent")
        assert result is None

    def test_is_step_complete(self, playbook):
        """Test checking if step is complete."""
        plan = PlaybookPlan(
            playbook=playbook,
            step_executions={
                "s1": StepExecution(step_id="s1", status=StepStatus.COMPLETED),
                "s2": StepExecution(step_id="s2", status=StepStatus.IN_PROGRESS),
            },
        )

        assert plan.is_step_complete("s1") is True
        assert plan.is_step_complete("s2") is False
        assert plan.is_step_complete("s3") is False
