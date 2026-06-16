# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for grounding evaluation."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from locus.reasoning.grounding import (
    ClaimEvaluation,
    GroundingEvaluator,
    GroundingResult,
    evaluate_grounding,
)


class TestClaimEvaluation:
    """Tests for ClaimEvaluation model."""

    def test_create_claim_evaluation(self):
        """Test creating a claim evaluation."""
        evaluation = ClaimEvaluation(
            claim="The sky is blue",
            score=0.9,
            supporting_evidence=["Evidence about sky color"],
            reasoning="Strong evidence support",
        )

        assert evaluation.claim == "The sky is blue"
        assert evaluation.score == 0.9
        assert len(evaluation.supporting_evidence) == 1
        assert evaluation.reasoning == "Strong evidence support"

    def test_is_grounded_high_score(self):
        """Test is_grounded property for high score."""
        evaluation = ClaimEvaluation(
            claim="Test claim",
            score=0.8,
        )
        assert evaluation.is_grounded is True

    def test_is_grounded_low_score(self):
        """Test is_grounded property for low score."""
        evaluation = ClaimEvaluation(
            claim="Test claim",
            score=0.3,
        )
        assert evaluation.is_grounded is False

    def test_is_grounded_threshold(self):
        """Test is_grounded at threshold (0.5)."""
        evaluation = ClaimEvaluation(
            claim="Test claim",
            score=0.5,
        )
        assert evaluation.is_grounded is True

    def test_claim_evaluation_frozen(self):
        """Test that claim evaluation is frozen."""
        from pydantic import ValidationError

        evaluation = ClaimEvaluation(claim="Test", score=0.5)
        with pytest.raises(ValidationError, match="frozen"):
            evaluation.score = 0.9


class TestGroundingResult:
    """Tests for GroundingResult model."""

    def test_create_grounding_result(self):
        """Test creating grounding result."""
        result = GroundingResult(
            score=0.85,
            claims=[
                ClaimEvaluation(claim="Claim 1", score=0.9),
                ClaimEvaluation(claim="Claim 2", score=0.8),
            ],
            ungrounded_claims=[],
            requires_replan=False,
        )

        assert result.score == 0.85
        assert len(result.claims) == 2
        assert result.requires_replan is False

    def test_grounded_claims_property(self):
        """Test grounded_claims property."""
        result = GroundingResult(
            score=0.6,
            claims=[
                ClaimEvaluation(claim="Grounded claim", score=0.8),
                ClaimEvaluation(claim="Ungrounded claim", score=0.3),
            ],
            ungrounded_claims=["Ungrounded claim"],
        )

        grounded = result.grounded_claims
        assert len(grounded) == 1
        assert grounded[0].claim == "Grounded claim"

    def test_grounding_ratio_property(self):
        """Test grounding_ratio property."""
        result = GroundingResult(
            score=0.6,
            claims=[
                ClaimEvaluation(claim="C1", score=0.8),
                ClaimEvaluation(claim="C2", score=0.3),
                ClaimEvaluation(claim="C3", score=0.7),
            ],
            ungrounded_claims=["C2"],
        )

        # 2 out of 3 claims are grounded
        assert result.grounding_ratio == pytest.approx(2 / 3)

    def test_grounding_ratio_empty_claims(self):
        """Test grounding_ratio with no claims."""
        result = GroundingResult(score=1.0, claims=[])
        assert result.grounding_ratio == 1.0


class TestGroundingEvaluator:
    """Tests for GroundingEvaluator."""

    @pytest.fixture
    def evaluator(self):
        """Create an evaluator with defaults."""
        return GroundingEvaluator()

    def test_create_evaluator_defaults(self):
        """Test creating evaluator with defaults."""
        evaluator = GroundingEvaluator()
        assert evaluator.replan_threshold == 0.65
        assert evaluator.claim_threshold == 0.5
        assert evaluator.require_evidence is True

    def test_create_evaluator_custom(self):
        """Test creating evaluator with custom settings."""
        evaluator = GroundingEvaluator(
            replan_threshold=0.8,
            claim_threshold=0.6,
            require_evidence=False,
        )
        assert evaluator.replan_threshold == 0.8
        assert evaluator.claim_threshold == 0.6
        assert evaluator.require_evidence is False

    def test_evaluate_empty_claims(self, evaluator):
        """Test evaluating empty claims list."""
        result = evaluator.evaluate([], ["some evidence"])

        assert result.score == 1.0
        assert result.claims == []
        assert result.ungrounded_claims == []
        assert result.requires_replan is False
        assert result.evaluation_details.get("reason") == "no_claims_to_evaluate"

    def test_evaluate_exact_match(self, evaluator):
        """Test evaluation with exact evidence match."""
        claims = ["The temperature is 72F"]
        evidence = ["The temperature is 72F"]

        result = evaluator.evaluate(claims, evidence)

        assert result.score == 1.0
        assert len(result.claims) == 1
        assert result.claims[0].reasoning == "Exact match in evidence"

    def test_evaluate_substring_match(self, evaluator):
        """Test evaluation with substring match."""
        claims = ["weather is sunny"]
        evidence = ["Today the weather is sunny and warm"]

        result = evaluator.evaluate(claims, evidence)

        assert result.score == 0.9
        assert result.claims[0].reasoning == "Claim found as substring in evidence"

    def test_evaluate_keyword_overlap(self, evaluator):
        """Test evaluation with keyword overlap."""
        claims = ["Python programming language features"]
        evidence = ["Python is a versatile programming language with many features"]

        result = evaluator.evaluate(claims, evidence)

        # Should have some overlap score
        assert result.score > 0.0
        assert "overlap" in result.claims[0].reasoning.lower()

    def test_evaluate_no_match(self, evaluator):
        """Test evaluation with no evidence match."""
        claims = ["Completely unrelated claim"]
        evidence = ["Evidence about something else entirely"]

        result = evaluator.evaluate(claims, evidence)

        assert result.score == 0.0
        assert "Completely unrelated claim" in result.ungrounded_claims

    def test_evaluate_without_require_evidence(self):
        """Test evaluation when evidence not required."""
        evaluator = GroundingEvaluator(require_evidence=False)
        claims = ["Unmatched claim"]
        evidence = ["Different content"]

        result = evaluator.evaluate(claims, evidence)

        # Should get benefit of doubt score
        assert result.score == 0.3
        assert "not required" in result.claims[0].reasoning.lower()

    def test_evaluate_multiple_claims(self, evaluator):
        """Test evaluation with multiple claims."""
        claims = [
            "The sky is blue",
            "Water is wet",
            "Fire is hot",
        ]
        evidence = [
            "The sky is blue on sunny days",
            "Fire produces heat and is hot",
        ]

        result = evaluator.evaluate(claims, evidence)

        assert len(result.claims) == 3
        assert result.evaluation_details["claim_count"] == 3

    def test_evaluate_triggers_replan(self, evaluator):
        """Test that low score triggers replan."""
        claims = ["Unsubstantiated claim 1", "Unsubstantiated claim 2"]
        evidence = ["Unrelated evidence"]

        result = evaluator.evaluate(claims, evidence)

        assert result.requires_replan is True

    def test_evaluate_no_replan_above_threshold(self):
        """Test that high score doesn't trigger replan."""
        evaluator = GroundingEvaluator(replan_threshold=0.3)
        claims = ["Some claim"]
        evidence = ["Some claim"]

        result = evaluator.evaluate(claims, evidence)

        assert result.requires_replan is False

    def test_should_replan(self, evaluator):
        """Test should_replan method."""
        result_needs_replan = GroundingResult(
            score=0.5,
            requires_replan=True,
        )
        result_ok = GroundingResult(
            score=0.9,
            requires_replan=False,
        )

        assert evaluator.should_replan(result_needs_replan) is True
        assert evaluator.should_replan(result_ok) is False

    def test_get_replan_guidance_with_ungrounded(self, evaluator):
        """Test replan guidance with ungrounded claims."""
        result = GroundingResult(
            score=0.4,
            ungrounded_claims=["Claim 1", "Claim 2"],
            requires_replan=True,
        )

        guidance = evaluator.get_replan_guidance(result)

        assert "Claim 1" in guidance
        assert "Claim 2" in guidance
        assert "below threshold" in guidance
        assert "Recommendations" in guidance

    def test_get_replan_guidance_all_grounded(self, evaluator):
        """Test replan guidance when all claims grounded."""
        result = GroundingResult(
            score=0.9,
            ungrounded_claims=[],
            requires_replan=False,
        )

        guidance = evaluator.get_replan_guidance(result)

        assert "All claims are grounded" in guidance

    def test_calculate_overlap_score_stop_words(self, evaluator):
        """Test overlap calculation with stop words only."""
        # Claim with only stop words
        claim_words = {"the", "is", "a", "to"}
        evidence_text = "unrelated content"

        score = evaluator._calculate_overlap_score(claim_words, evidence_text)

        # Should return neutral score for claims with only stop words
        assert score == 0.5

    def test_find_supporting_evidence(self, evaluator):
        """Test finding supporting evidence."""
        claim = "Python programming language"
        evidence_set = {
            "Python is a programming language",
            "Java is also popular",
            "Python programming notebooks available",
        }

        supporting = evaluator._find_supporting_evidence(claim, evidence_set)

        # Should find evidence with matching words
        assert len(supporting) > 0
        assert len(supporting) <= 3  # Limited to 3


class TestEvaluateWithLLM:
    """Tests for LLM-based evaluation."""

    @pytest.fixture
    def evaluator(self):
        """Create an evaluator."""
        return GroundingEvaluator()

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = MagicMock()
        model.complete = AsyncMock()
        return model

    @pytest.mark.asyncio
    async def test_evaluate_with_llm_empty_claims(self, evaluator, mock_model):
        """Test LLM evaluation with empty claims."""
        result = await evaluator.evaluate_with_llm([], ["evidence"], mock_model)

        assert result.score == 1.0
        assert result.evaluation_details.get("method") == "llm"
        mock_model.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_with_llm(self, evaluator, mock_model):
        """Test LLM evaluation."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.message.content = """
CLAIM 1: 0.9 - Strongly supported by evidence
CLAIM 2: 0.4 - Partially supported
"""
        mock_model.complete.return_value = mock_response

        claims = ["Claim one", "Claim two"]
        evidence = ["Evidence one", "Evidence two"]

        result = await evaluator.evaluate_with_llm(claims, evidence, mock_model)

        assert result.evaluation_details.get("method") == "llm"
        assert len(result.claims) == 2

    @pytest.mark.asyncio
    async def test_evaluate_with_llm_context(self, evaluator, mock_model):
        """Test LLM evaluation with context."""
        mock_response = MagicMock()
        mock_response.message.content = "CLAIM 1: 0.8 - Supported"
        mock_model.complete.return_value = mock_response

        await evaluator.evaluate_with_llm(
            ["Claim"], ["Evidence"], mock_model, context="Extra context"
        )

        # Verify model was called
        mock_model.complete.assert_called_once()

    def test_build_evaluation_prompt(self, evaluator):
        """Test building evaluation prompt."""
        claims = ["First claim", "Second claim"]
        evidence = ["Evidence A", "Evidence B"]

        prompt = evaluator._build_evaluation_prompt(claims, evidence, None)

        assert "EVIDENCE:" in prompt
        assert "Evidence A" in prompt
        assert "CLAIMS TO EVALUATE:" in prompt
        assert "First claim" in prompt
        assert "Example:" in prompt

    def test_build_evaluation_prompt_with_context(self, evaluator):
        """Test building prompt with context."""
        prompt = evaluator._build_evaluation_prompt(["Claim"], ["Evidence"], "Some context")

        assert "CONTEXT: Some context" in prompt

    def test_parse_llm_response(self, evaluator):
        """Test parsing LLM response."""
        claims = ["Claim one", "Claim two"]
        response = """
CLAIM 1: 0.9 - Strong support
CLAIM 2: 0.3 - Weak support
"""

        evaluations = evaluator._parse_llm_response(claims, response)

        assert len(evaluations) == 2
        # Find evaluations by claim
        eval_dict = {e.claim: e for e in evaluations}
        assert eval_dict["Claim one"].score == 0.9
        assert eval_dict["Claim two"].score == 0.3

    def test_parse_llm_response_no_reasoning(self, evaluator):
        """Test parsing response without reasoning."""
        claims = ["Claim one"]
        response = "CLAIM 1: 0.75"

        evaluations = evaluator._parse_llm_response(claims, response)

        assert len(evaluations) == 1
        assert evaluations[0].score == 0.75
        assert evaluations[0].reasoning is None

    def test_parse_llm_response_invalid_lines(self, evaluator):
        """Test parsing response with invalid lines."""
        claims = ["Claim one"]
        response = """
Some preamble text
CLAIM 1: 0.8 - Valid
This is not a claim
Another random line
"""

        evaluations = evaluator._parse_llm_response(claims, response)

        # Should only parse valid claim lines
        assert len(evaluations) == 1

    def test_parse_llm_response_missing_claims(self, evaluator):
        """Test parsing response with missing claim evaluations."""
        claims = ["Claim one", "Claim two", "Claim three"]
        response = "CLAIM 1: 0.8 - Supported"

        evaluations = evaluator._parse_llm_response(claims, response)

        # Should fill in missing claims with score 0
        assert len(evaluations) == 3
        eval_dict = {e.claim: e for e in evaluations}
        assert eval_dict["Claim two"].score == 0.0
        assert "Failed to parse" in eval_dict["Claim two"].reasoning

    def test_parse_llm_response_score_clamping(self, evaluator):
        """Test that scores are clamped to valid range."""
        claims = ["Claim one", "Claim two"]
        response = """
CLAIM 1: 1.5 - Score too high
CLAIM 2: -0.5 - Score too low
"""

        evaluations = evaluator._parse_llm_response(claims, response)

        eval_dict = {e.claim: e for e in evaluations}
        assert eval_dict["Claim one"].score == 1.0  # Clamped to max
        assert eval_dict["Claim two"].score == 0.0  # Clamped to min

    def test_parse_llm_response_invalid_claim_number(self, evaluator):
        """Test parsing with out of range claim numbers."""
        claims = ["Claim one"]
        response = """
CLAIM 0: 0.8 - Invalid (0-indexed)
CLAIM 5: 0.8 - Invalid (too high)
"""

        evaluations = evaluator._parse_llm_response(claims, response)

        # Should fill in with default
        assert len(evaluations) == 1
        assert evaluations[0].score == 0.0


class TestEvaluateGroundingFunction:
    """Tests for evaluate_grounding convenience function."""

    def test_evaluate_grounding_function(self):
        """Test convenience function."""
        result = evaluate_grounding(
            claims=["Test claim"],
            evidence=["Test claim"],
            threshold=0.65,
        )

        assert result.score == 1.0
        assert result.requires_replan is False

    def test_evaluate_grounding_custom_threshold(self):
        """Test convenience function with custom threshold."""
        result = evaluate_grounding(
            claims=["Unmatched claim"],
            evidence=["Different evidence"],
            threshold=0.1,  # Very low threshold
        )

        # Even with low threshold, ungrounded claim should fail
        assert result.score < 0.5
