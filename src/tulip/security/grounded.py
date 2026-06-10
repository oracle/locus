# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""The grounding bridge — emit a finding only when its evidence clears GSAR.

This is the primitive that makes Tulip a *security* SDK rather than a
general one. :func:`ground_finding` takes a candidate finding plus the
GSAR :class:`~tulip.reasoning.gsar.Partition` of its claims, scores the
partition with the existing :func:`~tulip.reasoning.gsar.gsar_score` /
:func:`~tulip.reasoning.gsar.decide` functions, and:

- returns a :class:`~tulip.security.findings.Finding` **only** when the
  decision is :attr:`~tulip.reasoning.gsar.Decision.PROCEED`;
- otherwise returns an :class:`Abstention` recording why it did not ship.

An ungrounded finding is a false positive by construction — the caller
literally cannot get a ``Finding`` back for one. Abstentions are kept
(not discarded) so the non-finding is auditable: a SOC can review what
the agent declined to assert and why.
"""

from __future__ import annotations

from typing import TypeGuard

from pydantic import BaseModel, Field

from tulip.reasoning.gsar import (
    DEFAULT_CONTRADICTION_PENALTY,
    Decision,
    EvidenceType,
    GSARThresholds,
    Partition,
    decide,
    gsar_score,
)
from tulip.security.findings import (
    Confidence,
    Finding,
    FingerprintFinding,
    FingerprintVerdict,
    Indicator,
)
from tulip.security.taxonomy import Severity, TaxonomyTag


class Abstention(BaseModel):
    """The audit record for a candidate finding that did NOT ship.

    Carries the GSAR decision (never :attr:`Decision.PROCEED`), the
    score, the would-be title for triage, and a human-readable reason.
    """

    decision: Decision = Field(description="GSAR decision — regenerate / replan / abstain.")
    gsar_score: Confidence = Field(description="The grounding score that fell short.")
    candidate_title: str = Field(description="Title of the finding that was withheld.")
    reason: str = Field(description="Why the finding was withheld.")

    model_config = {"frozen": True}


# A grounding call yields either a shipped finding or an audited
# abstention; callers branch on the type (use :func:`is_finding`).
GroundedFinding = Finding | Abstention


def is_finding(result: GroundedFinding) -> TypeGuard[Finding]:
    """Narrow a :data:`GroundedFinding` to :class:`Finding` (vs Abstention)."""
    return isinstance(result, Finding)


def _evidence_refs(partition: Partition) -> list[str]:
    """Flatten every claim's evidence refs across the partition."""
    return [ref for claim in partition.all_claims() for ref in claim.evidence_refs]


def _abstention_reason(partition: Partition, decision: Decision) -> str:
    """Explain a withheld finding, distinguishing contradiction from weakness."""
    if partition.contradicted:
        return (
            f"withheld ({decision.value}): {len(partition.contradicted)} claim(s) "
            "contradicted by evidence"
        )
    return (
        f"withheld ({decision.value}): grounding below the proceed threshold "
        f"({len(partition.ungrounded)} ungrounded of {partition.total_claims} claims)"
    )


def ground_finding(
    *,
    title: str,
    description: str,
    severity: Severity,
    asset: str,
    remediation: str,
    partition: Partition,
    indicators: list[Indicator] | None = None,
    taxonomy: list[TaxonomyTag] | None = None,
    confidence: float = 1.0,
    thresholds: GSARThresholds | None = None,
    weight_map: dict[EvidenceType, float] | None = None,
    contradiction_penalty: float = DEFAULT_CONTRADICTION_PENALTY,
) -> GroundedFinding:
    """Emit a :class:`Finding` only if its evidence clears the GSAR threshold.

    Scores ``partition`` with :func:`~tulip.reasoning.gsar.gsar_score` and
    routes it through :func:`~tulip.reasoning.gsar.decide`. On
    :attr:`~tulip.reasoning.gsar.Decision.PROCEED` returns a
    :class:`Finding` carrying the score and the partition's flattened
    evidence refs; otherwise returns an :class:`Abstention`.

    Args:
        title: One-line finding summary.
        description: What was observed and why it matters.
        severity: Severity band.
        asset: Affected asset / host / service / endpoint.
        remediation: Recommended remediation.
        partition: The GSAR four-way partition of the finding's claims.
        indicators: Optional indicators of compromise.
        taxonomy: Optional MITRE ATLAS / OWASP tags.
        confidence: Analyst-facing confidence (distinct from grounding).
        thresholds: Override the GSAR reference thresholds.
        weight_map: Override the GSAR evidence-type weights.
        contradiction_penalty: GSAR ``ρ`` — see
            :func:`~tulip.reasoning.gsar.gsar_score`.

    Returns:
        A :class:`Finding` when grounded, else an :class:`Abstention`.
    """
    score = gsar_score(
        partition,
        weight_map=weight_map,
        contradiction_penalty=contradiction_penalty,
    )
    decision = decide(score, thresholds=thresholds)
    if decision is not Decision.PROCEED:
        return Abstention(
            decision=decision,
            gsar_score=score,
            candidate_title=title,
            reason=_abstention_reason(partition, decision),
        )
    return Finding(
        title=title,
        description=description,
        severity=severity,
        asset=asset,
        remediation=remediation,
        gsar_score=score,
        confidence=confidence,
        indicators=indicators or [],
        taxonomy=taxonomy or [],
        evidence_refs=_evidence_refs(partition),
    )


def ground_fingerprint(
    *,
    verdict: FingerprintVerdict,
    asset: str,
    partition: Partition,
    title: str | None = None,
    description: str | None = None,
    severity: Severity = Severity.MEDIUM,
    remediation: str = "Confirm the served model/engine against the approved inventory.",
    indicators: list[Indicator] | None = None,
    taxonomy: list[TaxonomyTag] | None = None,
    thresholds: GSARThresholds | None = None,
    weight_map: dict[EvidenceType, float] | None = None,
    contradiction_penalty: float = DEFAULT_CONTRADICTION_PENALTY,
) -> FingerprintFinding | Abstention:
    """Ground a timing side-channel fingerprint into a finding, or abstain.

    Same admit/abstain contract as :func:`ground_finding`, threading the
    :class:`~tulip.security.findings.FingerprintVerdict` through on the
    PROCEED path. Low feature coverage drives a weak partition, so an
    under-observed endpoint abstains rather than asserting a fingerprint.

    Args:
        verdict: The classifier verdict (model / engine / hardware).
        asset: The fingerprinted endpoint.
        partition: GSAR partition of the fingerprint's claims (the timing
            feature vector is its evidence).
        title: Optional override; defaults to a verdict summary.
        description: Optional override; defaults to a verdict summary.
        severity: Severity band (default ``MEDIUM``).
        remediation: Recommended remediation.
        indicators: Optional indicators (e.g. the endpoint).
        taxonomy: Optional threat tags.
        thresholds: Override the GSAR reference thresholds.
        weight_map: Override the GSAR evidence-type weights.
        contradiction_penalty: GSAR ``ρ``.

    Returns:
        A :class:`FingerprintFinding` when grounded, else an
        :class:`Abstention`.
    """
    auto = f"{verdict.model} on {verdict.engine} / {verdict.hardware}"
    score = gsar_score(
        partition,
        weight_map=weight_map,
        contradiction_penalty=contradiction_penalty,
    )
    decision = decide(score, thresholds=thresholds)
    if decision is not Decision.PROCEED:
        return Abstention(
            decision=decision,
            gsar_score=score,
            candidate_title=title or f"Inference fingerprint: {auto}",
            reason=_abstention_reason(partition, decision),
        )
    return FingerprintFinding(
        title=title or f"Inference fingerprint: {auto}",
        description=description or f"Endpoint {asset} fingerprinted as {auto}.",
        severity=severity,
        asset=asset,
        remediation=remediation,
        gsar_score=score,
        confidence=verdict.confidence,
        indicators=indicators or [],
        taxonomy=taxonomy or [],
        evidence_refs=_evidence_refs(partition),
        verdict=verdict,
    )


__all__ = [
    "Abstention",
    "GroundedFinding",
    "ground_finding",
    "ground_fingerprint",
    "is_finding",
]
