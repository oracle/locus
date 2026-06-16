# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Grounding evaluation using LLM-as-judge pattern.

Grounding ensures that claims made by the agent are supported by evidence
gathered during tool execution. This module provides evaluation and
replan triggering capabilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from collections.abc import Sequence


class ClaimEvaluation(BaseModel):
    """Evaluation of a single claim against evidence.

    Attributes:
        claim: The claim being evaluated.
        score: Grounding score (0.0 = ungrounded, 1.0 = fully grounded).
        supporting_evidence: Evidence that supports this claim.
        reasoning: Explanation of the evaluation.
    """

    claim: str = Field(..., description="The claim being evaluated")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Grounding score (0.0 = ungrounded, 1.0 = fully grounded)",
    )
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence that supports this claim",
    )
    reasoning: str | None = Field(
        default=None,
        description="Explanation of the evaluation",
    )

    model_config = {"frozen": True}

    @property
    def is_grounded(self) -> bool:
        """Whether this claim is sufficiently grounded (score >= 0.5)."""
        return self.score >= 0.5


class GroundingResult(BaseModel):
    """Result of grounding evaluation.

    Attributes:
        score: Overall grounding score (0.0 to 1.0).
        claims: List of all evaluated claims.
        ungrounded_claims: Claims that scored below threshold.
        requires_replan: Whether the agent should replan.
        evaluation_details: Additional details about the evaluation.
    """

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall grounding score",
    )
    claims: list[ClaimEvaluation] = Field(
        default_factory=list,
        description="All evaluated claims with their scores",
    )
    ungrounded_claims: list[str] = Field(
        default_factory=list,
        description="Claims that failed grounding check",
    )
    requires_replan: bool = Field(
        default=False,
        description="Whether the agent should replan based on grounding",
    )
    evaluation_details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional evaluation metadata",
    )

    model_config = {"frozen": True}

    @property
    def grounded_claims(self) -> list[ClaimEvaluation]:
        """Get claims that passed grounding check."""
        return [c for c in self.claims if c.is_grounded]

    @property
    def grounding_ratio(self) -> float:
        """Ratio of grounded claims to total claims."""
        if not self.claims:
            return 1.0
        return len(self.grounded_claims) / len(self.claims)


class GroundingEvaluator:
    """Evaluates if claims are grounded in evidence using LLM-as-judge pattern.

    The GroundingEvaluator analyzes claims against evidence gathered during
    tool execution to determine if the claims are factually supported.

    Attributes:
        replan_threshold: Score below which replanning is triggered.
        claim_threshold: Minimum score for a claim to be considered grounded.
        require_evidence: Whether claims without explicit evidence are penalized.
    """

    def __init__(
        self,
        replan_threshold: float = 0.65,
        claim_threshold: float = 0.5,
        require_evidence: bool = True,
    ) -> None:
        """Initialize the GroundingEvaluator.

        Args:
            replan_threshold: Score below which replanning is triggered.
            claim_threshold: Minimum score for individual claim grounding.
            require_evidence: Penalize claims without explicit evidence.
        """
        self.replan_threshold = replan_threshold
        self.claim_threshold = claim_threshold
        self.require_evidence = require_evidence

    def evaluate(
        self,
        claims: Sequence[str],
        evidence: Sequence[str],
        context: str | None = None,
    ) -> GroundingResult:
        """Evaluate claims against evidence.

        This is a rule-based evaluation. For LLM-based evaluation, use
        evaluate_with_llm which integrates with a model provider.

        Args:
            claims: List of claims to evaluate.
            evidence: List of evidence strings from tool executions.
            context: Optional context for evaluation.

        Returns:
            GroundingResult with scores and ungrounded claims.
        """
        if not claims:
            return GroundingResult(
                score=1.0,
                claims=[],
                ungrounded_claims=[],
                requires_replan=False,
                evaluation_details={"reason": "no_claims_to_evaluate"},
            )

        evaluated_claims: list[ClaimEvaluation] = []
        evidence_set = set(evidence)
        evidence_text = " ".join(evidence).lower()

        for claim in claims:
            evaluation = self._evaluate_single_claim(
                claim,
                evidence_set,
                evidence_text,
            )
            evaluated_claims.append(evaluation)

        # Calculate overall score
        if evaluated_claims:
            overall_score = sum(c.score for c in evaluated_claims) / len(evaluated_claims)
        else:
            overall_score = 1.0

        # Identify ungrounded claims
        ungrounded = [c.claim for c in evaluated_claims if c.score < self.claim_threshold]

        # Determine if replan is needed
        requires_replan = overall_score < self.replan_threshold

        return GroundingResult(
            score=overall_score,
            claims=evaluated_claims,
            ungrounded_claims=ungrounded,
            requires_replan=requires_replan,
            evaluation_details={
                "evidence_count": len(evidence),
                "claim_count": len(claims),
                "grounded_count": len(claims) - len(ungrounded),
            },
        )

    def _evaluate_single_claim(
        self,
        claim: str,
        evidence_set: set[str],
        evidence_text: str,
    ) -> ClaimEvaluation:
        """Evaluate a single claim against evidence.

        Uses heuristic matching for rule-based evaluation.

        Args:
            claim: The claim to evaluate.
            evidence_set: Set of evidence strings.
            evidence_text: Concatenated lowercase evidence for matching.

        Returns:
            ClaimEvaluation for this claim.
        """
        claim_lower = claim.lower()
        claim_words = set(claim_lower.split())

        # Direct match in evidence
        if claim in evidence_set:
            return ClaimEvaluation(
                claim=claim,
                score=1.0,
                supporting_evidence=[claim],
                reasoning="Exact match in evidence",
            )

        # Check for substring match
        if claim_lower in evidence_text:
            return ClaimEvaluation(
                claim=claim,
                score=0.9,
                supporting_evidence=self._find_supporting_evidence(claim, evidence_set),
                reasoning="Claim found as substring in evidence",
            )

        # Check for keyword overlap
        overlap_score = self._calculate_overlap_score(claim_words, evidence_text)
        if overlap_score >= 0.7:
            return ClaimEvaluation(
                claim=claim,
                score=overlap_score,
                supporting_evidence=self._find_supporting_evidence(claim, evidence_set),
                reasoning=f"High keyword overlap ({overlap_score:.0%})",
            )

        if overlap_score >= 0.4:
            return ClaimEvaluation(
                claim=claim,
                score=overlap_score,
                supporting_evidence=self._find_supporting_evidence(claim, evidence_set),
                reasoning=f"Partial keyword overlap ({overlap_score:.0%})",
            )

        # No evidence found
        if self.require_evidence:
            return ClaimEvaluation(
                claim=claim,
                score=0.0,
                supporting_evidence=[],
                reasoning="No supporting evidence found",
            )

        # Without require_evidence, give benefit of doubt
        return ClaimEvaluation(
            claim=claim,
            score=0.3,
            supporting_evidence=[],
            reasoning="No evidence found, but not required",
        )

    def _calculate_overlap_score(
        self,
        claim_words: set[str],
        evidence_text: str,
    ) -> float:
        """Calculate keyword overlap between claim and evidence.

        Args:
            claim_words: Set of words in the claim.
            evidence_text: Lowercase evidence text.

        Returns:
            Overlap score (0.0 to 1.0).
        """
        # Filter out common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "and",
            "but",
            "or",
            "nor",
            "so",
            "yet",
            "both",
            "either",
            "neither",
            "not",
            "only",
            "own",
            "same",
            "than",
            "too",
            "very",
            "just",
            "also",
            "now",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "every",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "any",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
        }

        meaningful_words = claim_words - stop_words
        if not meaningful_words:
            return 0.5  # Neutral score for claims with only stop words

        found_count = sum(1 for word in meaningful_words if word in evidence_text)
        return found_count / len(meaningful_words)

    def _find_supporting_evidence(
        self,
        claim: str,
        evidence_set: set[str],
    ) -> list[str]:
        """Find evidence items that support a claim.

        Args:
            claim: The claim to find support for.
            evidence_set: Set of evidence strings.

        Returns:
            List of supporting evidence strings.
        """
        claim_lower = claim.lower()
        claim_words = set(claim_lower.split())
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "for", "on"}
        meaningful_words = claim_words - stop_words

        supporting: list[str] = []
        for evidence in evidence_set:
            evidence_lower = evidence.lower()
            # Check if evidence contains meaningful claim words
            matches = sum(1 for word in meaningful_words if word in evidence_lower)
            if matches >= len(meaningful_words) * 0.5:
                supporting.append(evidence)

        return supporting[:3]  # Return at most 3 pieces of evidence

    async def evaluate_with_llm(
        self,
        claims: Sequence[str],
        evidence: Sequence[str],
        model: Any,
        context: str | None = None,
    ) -> GroundingResult:
        """Evaluate claims using an LLM as judge.

        This method uses a language model to evaluate whether claims
        are grounded in the provided evidence.

        Args:
            claims: List of claims to evaluate.
            evidence: List of evidence strings.
            model: Model instance implementing ModelProtocol.
            context: Optional context for evaluation.

        Returns:
            GroundingResult with LLM-based evaluations.
        """
        from locus.core.messages import Message  # noqa: PLC0415

        if not claims:
            return GroundingResult(
                score=1.0,
                claims=[],
                ungrounded_claims=[],
                requires_replan=False,
                evaluation_details={"reason": "no_claims_to_evaluate", "method": "llm"},
            )

        # Build the evaluation prompt
        prompt = self._build_evaluation_prompt(claims, evidence, context)

        # Call the model
        messages = [Message.user(prompt)]
        response = await model.complete(messages)

        # Parse the response
        evaluated_claims = self._parse_llm_response(
            claims,
            response.message.content or "",
        )

        # Calculate overall score
        if evaluated_claims:
            overall_score = sum(c.score for c in evaluated_claims) / len(evaluated_claims)
        else:
            overall_score = 1.0

        # Identify ungrounded claims
        ungrounded = [c.claim for c in evaluated_claims if c.score < self.claim_threshold]

        return GroundingResult(
            score=overall_score,
            claims=evaluated_claims,
            ungrounded_claims=ungrounded,
            requires_replan=overall_score < self.replan_threshold,
            evaluation_details={
                "evidence_count": len(evidence),
                "claim_count": len(claims),
                "method": "llm",
            },
        )

    def _build_evaluation_prompt(
        self,
        claims: Sequence[str],
        evidence: Sequence[str],
        context: str | None,
    ) -> str:
        """Build the prompt for LLM-based evaluation.

        Args:
            claims: Claims to evaluate.
            evidence: Evidence to check against.
            context: Optional context.

        Returns:
            Formatted prompt string.
        """
        parts = [
            "You are evaluating whether claims are grounded in evidence.",
            "For each claim, assign a score from 0.0 (completely ungrounded) to 1.0 (fully grounded).",
            "",
            "EVIDENCE:",
        ]

        for i, ev in enumerate(evidence, 1):
            parts.append(f"{i}. {ev}")

        parts.extend(["", "CLAIMS TO EVALUATE:"])

        for i, claim in enumerate(claims, 1):
            parts.append(f"{i}. {claim}")

        if context:
            parts.extend(["", f"CONTEXT: {context}"])

        parts.extend(
            [
                "",
                "For each claim, respond with:",
                "CLAIM [number]: [score] - [brief reasoning]",
                "",
                "Example:",
                "CLAIM 1: 0.9 - Directly supported by evidence item 2",
                "CLAIM 2: 0.3 - Only partially mentioned, key details missing",
            ]
        )

        return "\n".join(parts)

    def _parse_llm_response(
        self,
        claims: Sequence[str],
        response: str,
    ) -> list[ClaimEvaluation]:
        """Parse LLM response into claim evaluations.

        Args:
            claims: Original claims for reference.
            response: LLM response text.

        Returns:
            List of ClaimEvaluation objects.
        """
        evaluations: list[ClaimEvaluation] = []
        lines = response.strip().split("\n")

        # Create a mapping from claim index to claim text
        claim_list = list(claims)

        for line in lines:
            stripped_line = line.strip()
            if not stripped_line.upper().startswith("CLAIM"):
                continue

            try:
                # Parse "CLAIM N: SCORE - REASONING"
                parts = stripped_line.split(":", 1)
                if len(parts) < 2:
                    continue

                # Extract claim number
                claim_num_str = parts[0].upper().replace("CLAIM", "").strip()
                claim_num = int(claim_num_str) - 1  # 0-indexed

                if claim_num < 0 or claim_num >= len(claim_list):
                    continue

                # Extract score and reasoning
                score_reasoning = parts[1].strip()
                if "-" in score_reasoning:
                    score_str, reasoning = score_reasoning.split("-", 1)
                    score = float(score_str.strip())
                    reasoning = reasoning.strip()
                else:
                    score = float(score_reasoning.strip())
                    reasoning = None

                # Clamp score to valid range
                score = max(0.0, min(1.0, score))

                evaluations.append(
                    ClaimEvaluation(
                        claim=claim_list[claim_num],
                        score=score,
                        supporting_evidence=[],
                        reasoning=reasoning,
                    )
                )

            except (ValueError, IndexError):
                continue

        # Fill in any missing claims with low scores
        evaluated_indices = {
            claim_list.index(e.claim) for e in evaluations if e.claim in claim_list
        }

        for i, claim in enumerate(claim_list):
            if i not in evaluated_indices:
                evaluations.append(
                    ClaimEvaluation(
                        claim=claim,
                        score=0.0,
                        supporting_evidence=[],
                        reasoning="Failed to parse evaluation",
                    )
                )

        return evaluations

    def should_replan(self, result: GroundingResult) -> bool:
        """Check if replanning is recommended based on grounding result.

        Args:
            result: GroundingResult from evaluation.

        Returns:
            True if replanning is recommended.
        """
        return result.requires_replan

    def get_replan_guidance(self, result: GroundingResult) -> str:
        """Generate guidance for replanning based on grounding failures.

        Args:
            result: GroundingResult with ungrounded claims.

        Returns:
            Guidance string for the agent.
        """
        if not result.ungrounded_claims:
            return "All claims are grounded. No replanning needed."

        parts = [
            f"Grounding score ({result.score:.0%}) is below threshold ({self.replan_threshold:.0%}).",
            "",
            "Ungrounded claims that need evidence:",
        ]

        for claim in result.ungrounded_claims[:5]:  # Limit to first 5
            parts.append(f"- {claim}")

        parts.extend(
            [
                "",
                "Recommendations:",
                "1. Gather additional evidence for ungrounded claims",
                "2. Revise claims that cannot be substantiated",
                "3. Focus on verifiable facts from tool results",
            ]
        )

        return "\n".join(parts)


def evaluate_grounding(
    claims: Sequence[str],
    evidence: Sequence[str],
    threshold: float = 0.65,
) -> GroundingResult:
    """Convenience function to evaluate grounding.

    Args:
        claims: List of claims to evaluate.
        evidence: List of evidence strings.
        threshold: Replan threshold.

    Returns:
        GroundingResult with scores and recommendations.
    """
    evaluator = GroundingEvaluator(replan_threshold=threshold)
    return evaluator.evaluate(claims, evidence)
