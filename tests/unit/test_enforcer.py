# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for playbook enforcer."""

from datetime import datetime

import pytest

from locus.playbooks.enforcer import (
    EnforcementResult,
    EnforcementViolation,
    PlaybookEnforcer,
)
from locus.playbooks.models import (
    Playbook,
    PlaybookStep,
    StepStatus,
)


class TestEnforcementViolation:
    """Tests for EnforcementViolation."""

    def test_create_minimal(self):
        """Test creating violation with minimal fields."""
        violation = EnforcementViolation(
            violation_type="test",
            message="Test message",
        )
        assert violation.violation_type == "test"
        assert violation.message == "Test message"
        assert violation.step_id is None
        assert violation.tool_name is None
        assert violation.blocked is False
        assert isinstance(violation.timestamp, datetime)

    def test_create_full(self):
        """Test creating violation with all fields."""
        violation = EnforcementViolation(
            violation_type="unexpected_tool",
            step_id="step1",
            tool_name="bad_tool",
            message="Tool not allowed",
            blocked=True,
        )
        assert violation.step_id == "step1"
        assert violation.tool_name == "bad_tool"
        assert violation.blocked is True


class TestEnforcementResult:
    """Tests for EnforcementResult."""

    def test_create_allowed(self):
        """Test creating allowed result."""
        result = EnforcementResult(allowed=True)
        assert result.allowed is True
        assert result.violation is None
        assert result.hints == []
        assert result.current_step is None

    def test_create_with_violation(self):
        """Test creating result with violation."""
        violation = EnforcementViolation(
            violation_type="test",
            message="Test",
        )
        result = EnforcementResult(
            allowed=False,
            violation=violation,
            hints=["Fix this"],
        )
        assert result.allowed is False
        assert result.violation is violation
        assert "Fix this" in result.hints


class TestPlaybookEnforcer:
    """Tests for PlaybookEnforcer."""

    @pytest.fixture
    def simple_playbook(self):
        """Create a simple playbook for testing."""
        return Playbook(
            id="test_playbook",
            name="test_playbook",
            description="Test playbook",
            steps=[
                PlaybookStep(
                    id="step1",
                    description="First step",
                    expected_tools=frozenset({"tool_a", "tool_b"}),
                    hints=["Use tool_a first"],
                ),
                PlaybookStep(
                    id="step2",
                    description="Second step",
                    expected_tools=frozenset({"tool_c"}),
                    required=False,
                ),
                PlaybookStep(
                    id="step3",
                    description="Third step",
                    expected_tools=frozenset({"tool_d"}),
                    max_tool_calls=2,
                ),
            ],
        )

    @pytest.fixture
    def enforcer(self, simple_playbook):
        """Create an enforcer for testing."""
        return PlaybookEnforcer.from_playbook(simple_playbook)

    def test_create_from_playbook(self, simple_playbook):
        """Test creating enforcer from playbook."""
        enforcer = PlaybookEnforcer.from_playbook(simple_playbook)
        assert enforcer.plan.playbook is simple_playbook
        assert enforcer.block_violations is True
        assert enforcer.record_violations is True

    def test_create_with_options(self, simple_playbook):
        """Test creating enforcer with custom options."""
        enforcer = PlaybookEnforcer.from_playbook(
            simple_playbook,
            block_violations=False,
            record_violations=False,
        )
        assert enforcer.block_violations is False
        assert enforcer.record_violations is False

    def test_current_step(self, enforcer):
        """Test current step property."""
        assert enforcer.current_step.id == "step1"

    def test_current_step_hints(self, enforcer):
        """Test current step hints."""
        hints = enforcer.current_step_hints
        assert "Use tool_a first" in hints

    def test_progress(self, enforcer):
        """Test progress calculation."""
        assert enforcer.progress == 0.0

        enforcer.complete_current_step()
        assert enforcer.progress == pytest.approx(1 / 3, rel=0.01)

    def test_is_complete(self, enforcer):
        """Test is_complete property."""
        assert enforcer.is_complete is False

        enforcer.complete_current_step()
        enforcer.complete_current_step()
        enforcer.complete_current_step()
        assert enforcer.is_complete is True

    def test_validate_expected_tool(self, enforcer):
        """Test validating expected tool."""
        result = enforcer.validate_tool_call("tool_a")
        assert result.allowed is True
        assert result.violation is None
        assert result.current_step.id == "step1"

    def test_validate_unexpected_tool_blocked(self, enforcer):
        """Test validating unexpected tool with blocking."""
        result = enforcer.validate_tool_call("tool_x")
        assert result.allowed is False
        assert result.violation is not None
        assert result.violation.violation_type == "unexpected_tool"
        assert result.violation.blocked is True

    def test_validate_unexpected_tool_allowed(self):
        """Test validating unexpected tool when extra tools allowed."""
        playbook = Playbook(
            id="test_playbook",
            name="test_playbook",
            description="Test playbook",
            allow_extra_tools=True,
            steps=[
                PlaybookStep(
                    id="step1",
                    description="First step",
                    expected_tools=frozenset({"tool_a"}),
                ),
            ],
        )
        enforcer = PlaybookEnforcer.from_playbook(playbook)

        result = enforcer.validate_tool_call("tool_x")
        assert result.allowed is True

    def test_validate_after_complete(self, enforcer):
        """Test validation after playbook complete."""
        # Complete all steps
        enforcer.complete_current_step()
        enforcer.complete_current_step()
        enforcer.complete_current_step()

        result = enforcer.validate_tool_call("any_tool")
        assert result.allowed is False
        assert "playbook_complete" in result.violation.violation_type

    def test_validate_max_tool_calls(self, enforcer):
        """Test max tool calls enforcement."""
        # Advance to step3 which has max_tool_calls=2
        enforcer.complete_current_step()
        enforcer.complete_current_step()

        # Record tool calls
        enforcer.record_tool_call("tool_d")
        enforcer.record_tool_call("tool_d")

        # Third call should be blocked
        result = enforcer.validate_tool_call("tool_d")
        assert result.allowed is False
        assert result.violation.violation_type == "max_tool_calls"

    def test_record_tool_call(self, enforcer):
        """Test recording tool calls."""
        enforcer.record_tool_call("tool_a")
        assert enforcer.plan.total_tool_calls == 1

        step_exec = enforcer.plan.step_executions["step1"]
        assert step_exec.tool_call_count == 1
        assert "tool_a" in step_exec.tool_calls

    def test_complete_current_step(self, enforcer):
        """Test completing current step."""
        result = enforcer.complete_current_step("Step 1 done")

        assert result is True
        assert enforcer.current_step.id == "step2"
        step_exec = enforcer.plan.step_executions["step1"]
        assert step_exec.status == StepStatus.COMPLETED
        assert step_exec.result == "Step 1 done"

    def test_complete_last_step(self, enforcer):
        """Test completing the last step."""
        enforcer.complete_current_step()
        enforcer.complete_current_step()
        result = enforcer.complete_current_step()

        assert result is False
        assert enforcer.is_complete is True

    def test_skip_current_step(self, enforcer):
        """Test skipping current step."""
        # Step 1 is required, can't skip
        result = enforcer.skip_current_step("Don't want to")
        assert result is False
        assert enforcer.current_step.id == "step1"

        # Advance to step2 which is optional
        enforcer.complete_current_step()
        result = enforcer.skip_current_step("Not needed")
        assert result is True
        assert enforcer.current_step.id == "step3"

    def test_fail_current_step(self, enforcer):
        """Test failing current step."""
        enforcer.fail_current_step("Something went wrong")

        step_exec = enforcer.plan.step_executions["step1"]
        assert step_exec.status == StepStatus.FAILED
        assert step_exec.error == "Something went wrong"
        assert "Something went wrong" in enforcer.plan.errors[0]

    def test_get_next_step_hints(self, enforcer):
        """Test getting next step hints."""
        hints = enforcer.get_next_step_hints()
        # Step 2 has no explicit hints, returns empty
        assert isinstance(hints, list)

    def test_get_step_summary(self, enforcer):
        """Test getting step summary."""
        summary = enforcer.get_step_summary()

        assert summary["total_steps"] == 3
        assert summary["current_step_index"] == 0
        assert summary["completed"] == 0
        assert summary["pending"] == 3
        assert summary["progress"] == 0.0
        assert summary["is_complete"] is False

    def test_violations_property(self, enforcer):
        """Test violations property."""
        # Trigger a violation
        enforcer.validate_tool_call("unknown_tool")

        violations = enforcer.violations
        assert len(violations) == 1
        assert violations[0].tool_name == "unknown_tool"

    def test_reset(self, enforcer):
        """Test reset enforcer."""
        # Make some progress
        enforcer.record_tool_call("tool_a")
        enforcer.complete_current_step()
        enforcer.validate_tool_call("bad_tool")  # Creates violation

        # Reset
        enforcer.reset()

        assert enforcer.current_step.id == "step1"
        assert enforcer.plan.total_tool_calls == 0
        assert len(enforcer.violations) == 0
        assert not enforcer.is_complete

    def test_record_tool_call_after_complete(self, enforcer):
        """Test recording tool call after playbook complete."""
        enforcer.complete_current_step()
        enforcer.complete_current_step()
        enforcer.complete_current_step()

        # Should not crash, just increment total
        enforcer.record_tool_call("any_tool")
        assert enforcer.plan.total_tool_calls == 1

    def test_fail_step_when_none(self, enforcer):
        """Test failing step when none current."""
        enforcer.complete_current_step()
        enforcer.complete_current_step()
        enforcer.complete_current_step()

        # Should not crash
        enforcer.fail_current_step("Error")

    def test_complete_step_when_none(self, enforcer):
        """Test completing step when none current."""
        enforcer.complete_current_step()
        enforcer.complete_current_step()
        enforcer.complete_current_step()

        result = enforcer.complete_current_step()
        assert result is False

    def test_skip_step_when_none(self, enforcer):
        """Test skipping step when none current."""
        enforcer.complete_current_step()
        enforcer.complete_current_step()
        enforcer.complete_current_step()

        result = enforcer.skip_current_step()
        assert result is False

    def test_no_record_violation_when_disabled(self, simple_playbook):
        """Test violations not recorded when disabled."""
        enforcer = PlaybookEnforcer.from_playbook(
            simple_playbook,
            record_violations=False,
        )

        result = enforcer.validate_tool_call("bad_tool")
        assert result.allowed is False
        assert result.violation is None
        assert len(enforcer.violations) == 0

    def test_no_block_when_disabled(self, simple_playbook):
        """Test violations not blocked when disabled."""
        enforcer = PlaybookEnforcer.from_playbook(
            simple_playbook,
            block_violations=False,
        )

        result = enforcer.validate_tool_call("bad_tool")
        assert result.allowed is True
        assert result.violation is not None
        assert result.violation.blocked is False
