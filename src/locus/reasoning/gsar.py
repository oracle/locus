# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""GSAR — Grounding-Stratified Adaptive Replanning.

Implements the typed-grounding scoring layer and three-tier decision
function from Kamelhar 2026 (`arXiv:2604.23366`). The framework
extends ``locus.reasoning.grounding`` from a binary scalar to a
four-way claim partition with cost-asymmetric recovery actions.

What's in this module
---------------------

- :class:`EvidenceType` — eight-element taxonomy from the paper's
  reference instantiation (Appendix B).
- :data:`DEFAULT_WEIGHT_MAP` — the production-calibrated weights from
  Appendix B (``tool_match`` 1.00 down to ``inference``/``domain`` 0.60).
- :class:`Claim`, :class:`Partition` — frozen Pydantic types matching
  the paper's Definition 2 and Definition 4.
- :func:`partition_weight` — Equation (1).
- :func:`gsar_score` — Equation (2), the centre of gravity. Returns
  ``S`` in ``[0, 1]``; falls back to ``0.5`` on the empty partition
  (the paper's epistemic-indifference convention).
- :class:`Decision` — ``proceed``/``regenerate``/``replan``/``abstain``.
- :func:`decide` — Equation (3) with the paper's reference thresholds
  ``τ_proceed = 0.80``, ``τ_regenerate = 0.65`` (Appendix B).

What's *not* in this module
---------------------------

- The LLM-as-judge protocol — see :mod:`locus.reasoning.gsar_judge`.
- The Algorithm-1 outer loop with ``K_max`` budget — see
  :mod:`locus.reasoning.gsar_evaluator`.
- ``Agent`` integration — wire a configured evaluator on
  ``AgentConfig`` once it stabilises; the scoring layer here is
  deliberately deployment-agnostic so it can be re-used at training
  time too (the paper sketches RLHF / PRM uses in §11.1).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class EvidenceType(StrEnum):
    """Provenance taxonomy from the paper's reference instantiation.

    Order is intentional: descending epistemic strength under the
    Appendix-B default weights. ``tool_match`` is the strongest (a
    claim is directly traceable to a structured tool output);
    ``domain`` and ``inference`` are the weakest (model-internal).
    """

    TOOL_MATCH = "tool_match"
    """Direct tool output verification (e.g. a metrics query row, log line)."""

    SPECIFIC_DATA = "specific_data"
    """Structured step output — a typed value lifted from a specialist's
    response object."""

    SIGNAL_MATCH = "signal_match"
    """A field on the originating signal (alert, anomaly envelope,
    operator request) referenced by the claim."""

    COMPLEMENTARY_FINDING = "complementary_finding"
    """A non-redundant alternative perspective that doesn't conflict
    with anything in :attr:`Partition.grounded`."""

    SYNTHESIS = "synthesis"
    """A cross-specialist derivation — one specialist's output combined
    with another's."""

    NEG_EVIDENCE = "neg_evidence"
    """An absence-of-signal observation (e.g. ``no errors in the last 5
    minutes``)."""

    INFERENCE = "inference"
    """A model-internal inference not anchored in any tool output."""

    DOMAIN = "domain"
    """A domain-knowledge assertion (textbook fact, runbook rule)."""


# Reference weight table from Appendix B. These are one well-calibrated
# point in ``(w, ρ, τ)``-space, not universally optimal — the paper
# explicitly recommends per-deployment recalibration on a small
# human-graded held-out set (Table B / §11.1 / §10).
DEFAULT_WEIGHT_MAP: dict[EvidenceType, float] = {
    EvidenceType.TOOL_MATCH: 1.00,
    EvidenceType.SPECIFIC_DATA: 0.95,
    EvidenceType.SIGNAL_MATCH: 0.90,
    EvidenceType.COMPLEMENTARY_FINDING: 0.85,
    EvidenceType.SYNTHESIS: 0.80,
    EvidenceType.NEG_EVIDENCE: 0.70,
    EvidenceType.INFERENCE: 0.60,
    EvidenceType.DOMAIN: 0.60,
}

# Default contradiction penalty ρ — the paper's reference value
# (Appendix B). Setting ρ=0 trips the score-inflation failure mode
# documented in §8.5 (P5 ablation).
DEFAULT_CONTRADICTION_PENALTY: float = 0.5

# Default decision thresholds τ_regenerate < τ_proceed (Appendix B).
DEFAULT_TAU_PROCEED: float = 0.80
DEFAULT_TAU_REGENERATE: float = 0.65

# Default replan budget K_max — recommendation from §11 ("Concrete
# recommendations for multi-agent orchestration systems", item 3).
# Beyond 3 the paper's empirical runs never recovered; higher just
# inflated latency and compute.
DEFAULT_K_MAX: int = 2

# Score the framework returns for an empty partition (C = ∅). The
# paper's Definition / §4.1 edge case: epistemic indifference, rather
# than a degenerate 0/0.
NEUTRAL_SCORE_ON_EMPTY: float = 0.5

# Weight assigned to claims whose ``EvidenceType`` is missing from the
# weight map. We default to ``w(inference)`` so unknown-provenance
# claims behave like model-internal ones until calibrated otherwise.
DEFAULT_UNKNOWN_TYPE_WEIGHT: float = 0.60


# ---------------------------------------------------------------------------
# Frozen claim + partition types (Definitions 2 + 4 in the paper)
# ---------------------------------------------------------------------------


_Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class Claim(BaseModel):
    """One atomic claim — the paper's Definition 2.

    A ``Claim`` is the unit the judge partitions, scores, and explains
    against. The ``evidence_refs`` field is a free-form list of strings
    (e.g. ``"tool:query_metrics:row=14:cpu_pct"``) so the framework
    isn't tied to any specific evidence-reference grammar.
    """

    text: str = Field(description="Natural-language statement.")
    type: EvidenceType = Field(
        description="Provenance / epistemic-strength label.",
    )
    evidence_refs: list[str] = Field(
        default_factory=list,
        description=(
            "Opaque references to the underlying evidence (tool output "
            "row, signal field, prior claim). Free-form strings — the "
            "framework doesn't parse them, but they're persisted "
            "alongside the partition for audit."
        ),
    )

    model_config = {"frozen": True}


class Partition(BaseModel):
    """The four-way partition C = G ⊔ U ⊔ X ⊔ K (Definition 4).

    The paper makes :attr:`complementary` first-class: a claim in K is
    not redundant with any claim in G *and* doesn't contradict any
    claim in G ∪ X. It carries positive weight, distinct from
    grounded.
    """

    grounded: list[Claim] = Field(default_factory=list, description="G.")
    ungrounded: list[Claim] = Field(default_factory=list, description="U.")
    contradicted: list[Claim] = Field(default_factory=list, description="X.")
    complementary: list[Claim] = Field(default_factory=list, description="K.")

    model_config = {"frozen": True}

    @property
    def is_empty(self) -> bool:
        """Whether C = ∅ — triggers the neutral-score convention."""
        return not (self.grounded or self.ungrounded or self.contradicted or self.complementary)

    @property
    def total_claims(self) -> int:
        return (
            len(self.grounded)
            + len(self.ungrounded)
            + len(self.contradicted)
            + len(self.complementary)
        )

    def all_claims(self) -> list[Claim]:
        """Flattened claim list across all four buckets."""
        return [
            *self.grounded,
            *self.ungrounded,
            *self.contradicted,
            *self.complementary,
        ]


class GSARThresholds(BaseModel):
    """Decision thresholds ``τ_regenerate < τ_proceed`` (§5.1 + Eq. 3)."""

    proceed: _Probability = Field(
        default=DEFAULT_TAU_PROCEED,
        description="τ_proceed — at or above this, δ = proceed.",
    )
    regenerate: _Probability = Field(
        default=DEFAULT_TAU_REGENERATE,
        description=(
            "τ_regenerate — at or above this and below τ_proceed, "
            "δ = regenerate; below this, δ = replan."
        ),
    )

    model_config = {"frozen": True}

    @field_validator("regenerate")
    @classmethod
    def _ordered(cls, v: float, info: object) -> float:
        # Pydantic v2 field-validator API: read sibling via ValidationInfo.
        proceed = getattr(info, "data", {}).get("proceed", DEFAULT_TAU_PROCEED)
        if v >= proceed:
            raise ValueError(
                f"τ_regenerate ({v}) must be strictly less than τ_proceed ({proceed})."
            )
        return v


# ---------------------------------------------------------------------------
# Equations 1, 2, 3 from §4-§5
# ---------------------------------------------------------------------------


def _claim_weight(
    claim: Claim,
    weight_map: dict[EvidenceType, float],
    default_unknown: float,
) -> float:
    """Look up the type weight, defaulting for unknown types.

    Mirrors Algorithm 2 lines 3 / 7 / 11 / 15 (``w(TYPE(c)) if … else w₀``).
    """
    return weight_map.get(claim.type, default_unknown)


def partition_weight(
    claims: list[Claim],
    *,
    weight_map: dict[EvidenceType, float] | None = None,
    default_unknown: float = DEFAULT_UNKNOWN_TYPE_WEIGHT,
) -> float:
    """Equation (1): ``W(P) = Σ_{c ∈ P} w(type(c))``.

    Args:
        claims: A list of :class:`Claim` (e.g. ``partition.grounded``).
        weight_map: Override the Appendix-B default weights.
        default_unknown: Weight for types missing from ``weight_map``.

    Returns:
        Sum of the per-claim weights — non-negative real.
    """
    weights = weight_map if weight_map is not None else DEFAULT_WEIGHT_MAP
    return sum(_claim_weight(c, weights, default_unknown) for c in claims)


def gsar_score(
    partition: Partition,
    *,
    weight_map: dict[EvidenceType, float] | None = None,
    contradiction_penalty: float = DEFAULT_CONTRADICTION_PENALTY,
    default_unknown: float = DEFAULT_UNKNOWN_TYPE_WEIGHT,
) -> float:
    """Equation (2) — the GSAR grounding score ``S``.

    ``S = (W(G) + W(K)) / (W(G) + W(U) + ρ · W(X) + W(K))``.

    On the empty partition the framework returns the paper's
    epistemic-indifference convention :data:`NEUTRAL_SCORE_ON_EMPTY`
    (§4.1) rather than a degenerate ``0/0``.

    Args:
        partition: Four-way partition produced by a judge.
        weight_map: Override Appendix-B weights.
        contradiction_penalty: ``ρ ∈ [0, 1]``. Defaults to ``0.5``;
            ``ρ = 0`` reproduces the score-inflation failure mode of
            the paper's P5 ablation.
        default_unknown: Weight for unknown evidence types.

    Returns:
        Scalar score in ``[0, 1]``.

    Raises:
        ValueError: When ``contradiction_penalty`` is outside
            ``[0, 1]``.
    """
    if not 0.0 <= contradiction_penalty <= 1.0:
        raise ValueError(f"contradiction_penalty must be in [0, 1], got {contradiction_penalty}")
    if partition.is_empty:
        return NEUTRAL_SCORE_ON_EMPTY

    weights = weight_map if weight_map is not None else DEFAULT_WEIGHT_MAP
    w_g = partition_weight(partition.grounded, weight_map=weights, default_unknown=default_unknown)
    w_u = partition_weight(
        partition.ungrounded, weight_map=weights, default_unknown=default_unknown
    )
    w_x = partition_weight(
        partition.contradicted, weight_map=weights, default_unknown=default_unknown
    )
    w_k = partition_weight(
        partition.complementary, weight_map=weights, default_unknown=default_unknown
    )

    numerator = w_g + w_k
    denominator = w_g + w_u + contradiction_penalty * w_x + w_k

    # Denominator can still be 0 if every claim's weight is zero (an
    # extreme calibration, e.g. ``w(domain) = 0`` and a partition
    # containing only domain claims). Treat as the empty-partition case.
    if denominator == 0.0:
        return NEUTRAL_SCORE_ON_EMPTY
    return numerator / denominator


class Decision(StrEnum):
    """Three-tier decision plus the abstain channel.

    ``abstain`` is the paper's judge-side decision-status flag (§6,
    Definition 4). The outer loop treats abstain as equivalent to
    ``replan``, but the framework keeps the distinction so observers
    can tell "the judge couldn't decide" apart from "the judge said
    the report needs a fresh investigation".
    """

    PROCEED = "proceed"
    REGENERATE = "regenerate"
    REPLAN = "replan"
    ABSTAIN = "abstain"


def decide(
    score: float,
    *,
    thresholds: GSARThresholds | None = None,
) -> Decision:
    """Equation (3) — the three-tier decision function ``δ``.

    ``δ(s) = proceed  if s ≥ τ_proceed``;
    ``δ(s) = regenerate  if τ_regenerate ≤ s < τ_proceed``;
    ``δ(s) = replan  if s < τ_regenerate``.

    Args:
        score: ``S`` from :func:`gsar_score` (or the judge's own scalar
            when reconciling).
        thresholds: Override the Appendix-B reference thresholds.

    Returns:
        :class:`Decision`. Never returns :attr:`Decision.ABSTAIN` —
        that signal comes from the judge, not the score function.

    Raises:
        ValueError: When ``score`` is outside ``[0, 1]``.
    """
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"score must be in [0, 1], got {score}")
    th = thresholds if thresholds is not None else GSARThresholds()
    if score >= th.proceed:
        return Decision.PROCEED
    if score >= th.regenerate:
        return Decision.REGENERATE
    return Decision.REPLAN


__all__ = [
    "DEFAULT_CONTRADICTION_PENALTY",
    "DEFAULT_K_MAX",
    "DEFAULT_TAU_PROCEED",
    "DEFAULT_TAU_REGENERATE",
    "DEFAULT_UNKNOWN_TYPE_WEIGHT",
    "DEFAULT_WEIGHT_MAP",
    "NEUTRAL_SCORE_ON_EMPTY",
    "Claim",
    "Decision",
    "EvidenceType",
    "GSARThresholds",
    "Partition",
    "decide",
    "gsar_score",
    "partition_weight",
]
