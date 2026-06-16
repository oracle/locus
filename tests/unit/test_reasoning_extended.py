# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Extended unit tests for reasoning module."""

import pytest

from locus.reasoning.grounding import (
    ClaimEvaluation,
    GroundingEvaluator,
    GroundingResult,
)
from locus.reasoning.reflexion import (
    AssessmentCategory,
    ReflectionResult,
    Reflector,
)


class TestAssessmentCategory:
    """Tests for AssessmentCategory enum."""

    def test_assessment_values(self):
        """Test assessment enum values exist."""
        # Check that the enum has expected values
        values = [e.value for e in AssessmentCategory]
        assert len(values) > 0


class TestReflectionResult:
    """Tests for ReflectionResult."""

    def test_create_result(self):
        """Test creating reflection result."""
        result = ReflectionResult(
            assessment=AssessmentCategory.ON_TRACK,
            confidence_delta=0.1,
            guidance="Keep going",
        )
        assert result.assessment == AssessmentCategory.ON_TRACK
        assert result.confidence_delta == 0.1
        assert result.guidance == "Keep going"

    def test_result_negative_delta(self):
        """Test result with negative confidence delta."""
        result = ReflectionResult(
            assessment=AssessmentCategory.STUCK,
            confidence_delta=-0.15,
            guidance="Try different approach",
        )
        assert result.confidence_delta < 0


class TestReflector:
    """Tests for Reflector class."""

    @pytest.fixture
    def reflector(self):
        """Create reflector instance."""
        return Reflector()

    def test_reflector_creation(self, reflector):
        """Test reflector creation."""
        assert reflector is not None

    def test_reflector_has_loop_threshold(self, reflector):
        """Test reflector has loop threshold attribute."""
        assert hasattr(reflector, "loop_threshold")
        assert reflector.loop_threshold > 0

    def test_reflector_has_success_weight(self, reflector):
        """Test reflector has success weight attribute."""
        assert hasattr(reflector, "success_weight")

    def test_reflector_has_error_penalty(self, reflector):
        """Test reflector has error penalty attribute."""
        assert hasattr(reflector, "error_penalty")


class TestClaimEvaluation:
    """Tests for ClaimEvaluation model."""

    def test_create_evaluation(self):
        """Test creating claim evaluation."""
        evaluation = ClaimEvaluation(
            claim="The sky is blue",
            score=0.95,
            supporting_evidence=["Source says sky is blue"],
            reasoning="Matches evidence",
        )
        assert evaluation.claim == "The sky is blue"
        assert evaluation.score == 0.95
        assert len(evaluation.supporting_evidence) == 1

    def test_low_score_evaluation(self):
        """Test low score claim evaluation."""
        evaluation = ClaimEvaluation(
            claim="Unverified statement",
            score=0.1,
            supporting_evidence=[],
            reasoning="No evidence found",
        )
        assert evaluation.score == 0.1
        assert len(evaluation.supporting_evidence) == 0


class TestGroundingResult:
    """Tests for GroundingResult."""

    def test_create_result(self):
        """Test creating grounding result."""
        claim_eval = ClaimEvaluation(
            claim="Test claim",
            score=0.9,
            supporting_evidence=["evidence"],
            reasoning="Good match",
        )
        result = GroundingResult(
            score=0.9,
            claims=[claim_eval],
            ungrounded_claims=[],
        )
        assert result.score == 0.9
        assert len(result.claims) == 1

    def test_result_with_ungrounded(self):
        """Test result with ungrounded claims."""
        grounded = ClaimEvaluation(
            claim="Grounded claim",
            score=0.9,
            supporting_evidence=["evidence"],
            reasoning="Good",
        )
        ungrounded = ClaimEvaluation(
            claim="Ungrounded claim",
            score=0.1,
            supporting_evidence=[],
            reasoning="No evidence",
        )
        result = GroundingResult(
            score=0.5,
            claims=[grounded, ungrounded],
            ungrounded_claims=["Ungrounded claim"],
        )
        assert len(result.ungrounded_claims) == 1


class TestGroundingEvaluator:
    """Tests for GroundingEvaluator."""

    @pytest.fixture
    def evaluator(self):
        """Create grounding evaluator."""
        return GroundingEvaluator()

    def test_evaluator_creation(self, evaluator):
        """Test evaluator creation."""
        assert evaluator is not None


class TestReflectorReflect:
    """Tests for Reflector.reflect method."""

    @pytest.fixture
    def reflector(self):
        """Create reflector."""
        return Reflector(loop_threshold=3)

    def test_reflect_no_executions(self, reflector):
        """Test reflect with no tool executions."""
        from locus.core.state import AgentState

        state = AgentState(run_id="test", messages=[], tool_executions=())
        result = reflector.reflect(state)

        assert result.assessment in [
            AssessmentCategory.ON_TRACK,
            AssessmentCategory.STUCK,
        ]

    def test_reflect_with_successful_execution(self, reflector):
        """Test reflect with successful tool execution."""
        from locus.core.state import AgentState, ToolExecution

        execution = ToolExecution(
            tool_name="search",
            tool_call_id="call_1",
            arguments={"query": "test"},
            result="Found results",
            success=True,
        )
        state = AgentState(
            run_id="test",
            messages=[],
            tool_executions=(execution,),
            tool_history=("search",),
        )
        result = reflector.reflect(state)

        # Should have positive confidence delta for success
        assert result.confidence_delta >= 0

    def test_reflect_with_mixed_executions(self, reflector):
        """Test reflect with mixed success and failure."""
        from locus.core.state import AgentState, ToolExecution

        success_exec = ToolExecution(
            tool_name="search",
            tool_call_id="call_1",
            arguments={"query": "test"},
            result="Found results",
            success=True,
        )
        fail_exec = ToolExecution(
            tool_name="write",
            tool_call_id="call_2",
            arguments={"data": "test"},
            result="Write failed",
            success=False,
        )
        state = AgentState(
            run_id="test",
            messages=[],
            tool_executions=(success_exec, fail_exec),
            tool_history=("search", "write"),
        )
        result = reflector.reflect(state)

        # Result should be computed (may be positive or negative depending on weights)
        assert result.assessment is not None


class TestReflectorLoopDetection:
    """Tests for Reflector loop detection."""

    @pytest.fixture
    def reflector(self):
        """Create reflector with loop threshold of 3."""
        return Reflector(loop_threshold=3)

    def test_detect_single_tool_loop(self, reflector):
        """Test detecting repeated same tool call across iterations."""
        from locus.core.messages import ToolCall
        from locus.core.state import AgentState, ReasoningStep

        state = AgentState(run_id="test")
        for i in range(3):
            step = ReasoningStep(
                iteration=i + 1,
                thought=f"Search {i}",
                tool_calls=[ToolCall(name="search", arguments={})],
            )
            state = state.with_reasoning_step(step)
            state = state.next_iteration()
        result = reflector.reflect(state)

        assert result.assessment == AssessmentCategory.LOOP_DETECTED
        assert "search" in result.loop_pattern

    def test_detect_alternating_loop(self):
        """Test detecting alternating tool pattern across iterations."""
        from locus.core.messages import ToolCall
        from locus.core.state import AgentState, ReasoningStep

        reflector = Reflector(loop_threshold=4)

        state = AgentState(run_id="test")
        for i, name in enumerate(["read", "write", "read", "write"]):
            step = ReasoningStep(
                iteration=i + 1,
                thought=f"Step {i}",
                tool_calls=[ToolCall(name=name, arguments={})],
            )
            state = state.with_reasoning_step(step)
            state = state.next_iteration()
        result = reflector.reflect(state)

        assert result.assessment == AssessmentCategory.LOOP_DETECTED

    def test_no_loop_with_varied_tools(self, reflector):
        """Test no loop detected with varied tool calls."""
        from locus.core.state import AgentState

        state = AgentState(
            run_id="test",
            messages=[],
            tool_history=("search", "read", "write"),
        )
        result = reflector.reflect(state)

        assert result.assessment != AssessmentCategory.LOOP_DETECTED


class TestReflectorConfidenceDelta:
    """Tests for confidence delta calculation."""

    def test_diminishing_returns(self):
        """Test diminishing returns at high confidence."""
        reflector = Reflector(
            success_weight=0.15,
            diminishing_returns=True,
        )

        from locus.core.state import AgentState, ToolExecution

        execution = ToolExecution(
            tool_name="search",
            tool_call_id="call_1",
            arguments={},
            result="Found",
            success=True,
        )

        # At low confidence, gains are higher
        state_low = AgentState(
            run_id="test",
            messages=[],
            tool_executions=(execution,),
            tool_history=("search",),
            confidence=0.2,
        )
        result_low = reflector.reflect(state_low)

        # At high confidence, gains are lower (diminishing returns)
        state_high = AgentState(
            run_id="test",
            messages=[],
            tool_executions=(execution,),
            tool_history=("search",),
            confidence=0.8,
        )
        result_high = reflector.reflect(state_high)

        # Low confidence state should have higher delta
        assert result_low.confidence_delta > result_high.confidence_delta

    def test_no_diminishing_returns(self):
        """Test without diminishing returns."""
        reflector = Reflector(
            success_weight=0.15,
            diminishing_returns=False,
        )

        from locus.core.state import AgentState, ToolExecution

        execution = ToolExecution(
            tool_name="search",
            tool_call_id="call_1",
            arguments={},
            result="Found",
            success=True,
        )

        state = AgentState(
            run_id="test",
            messages=[],
            tool_executions=(execution,),
            tool_history=("search",),
            confidence=0.8,
        )
        result = reflector.reflect(state)

        # Without diminishing returns, gain is not reduced
        assert result.confidence_delta > 0


class TestReflectorCustomParameters:
    """Tests for Reflector with custom parameters."""

    def test_custom_loop_threshold(self):
        """Test custom loop threshold."""
        reflector = Reflector(loop_threshold=5)
        assert reflector.loop_threshold == 5

        from locus.core.state import AgentState

        # 3 repeated calls should not trigger loop with threshold of 5
        state = AgentState(
            run_id="test",
            messages=[],
            tool_history=("search", "search", "search"),
        )
        result = reflector.reflect(state)

        # Should not be a loop since threshold is 5
        assert result.assessment != AssessmentCategory.LOOP_DETECTED

    def test_custom_weights(self):
        """Test custom success weight and error penalty."""
        reflector = Reflector(
            success_weight=0.25,
            error_penalty=0.3,
        )

        from locus.core.state import AgentState, ToolExecution

        # With higher success weight, same success gives higher delta
        execution = ToolExecution(
            tool_name="search",
            tool_call_id="call_1",
            arguments={},
            result="Found",
            success=True,
        )
        state = AgentState(
            run_id="test",
            messages=[],
            tool_executions=(execution,),
            tool_history=("search",),
            confidence=0.1,  # Low confidence for maximum effect
        )
        result = reflector.reflect(state)

        # Should have positive delta
        assert result.confidence_delta > 0
