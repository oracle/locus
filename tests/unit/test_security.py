# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``tulip.security`` — evidence-grounded findings.

Covers the core invariant — a ``Evidence`` only exists when its evidence
clears the GSAR proceed threshold, otherwise an ``Abstention`` is
returned — plus the schema, taxonomy, and fingerprint surfaces.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tulip.reasoning.gsar import (
    DEFAULT_TAU_PROCEED,
    Claim,
    Decision,
    EvidenceType,
    Partition,
)
from tulip.security import (
    Abstention,
    AtlasTechnique,
    Evidence,
    FingerprintClassifier,
    FingerprintFinding,
    FingerprintVerdict,
    Indicator,
    IndicatorType,
    OwaspASI,
    OwaspLLM,
    Severity,
    ground_finding,
    ground_fingerprint,
    is_finding,
    severity_at_least,
)


def _grounded_partition() -> Partition:
    """A partition that scores well above τ_proceed (all tool-match)."""
    return Partition(
        grounded=[
            Claim(
                text="TLS certificate expired",
                type=EvidenceType.TOOL_MATCH,
                evidence_refs=["tool:scan_endpoint:tls_expiry=2026-01-02"],
            ),
            Claim(
                text="Endpoint reachable on 443",
                type=EvidenceType.TOOL_MATCH,
                evidence_refs=["tool:scan_endpoint:port=443"],
            ),
        ],
    )


def _ungrounded_partition() -> Partition:
    """A partition dominated by ungrounded inference (below τ_regenerate)."""
    return Partition(
        ungrounded=[
            Claim(text="Host is probably exploitable", type=EvidenceType.INFERENCE),
            Claim(text="Likely an APT", type=EvidenceType.INFERENCE),
        ],
    )


def _contradicted_partition() -> Partition:
    """A partition with contradicted claims."""
    return Partition(
        grounded=[Claim(text="Cert present", type=EvidenceType.TOOL_MATCH)],
        contradicted=[
            Claim(text="Patched last quarter", type=EvidenceType.DOMAIN),
            Claim(text="No exposure", type=EvidenceType.INFERENCE),
        ],
    )


# ---------------------------------------------------------------------------
# Schema invariants
# ---------------------------------------------------------------------------


def test_finding_is_frozen() -> None:
    result = ground_finding(
        title="t",
        description="d",
        severity=Severity.HIGH,
        asset="192.0.2.10",
        remediation="rotate",
        partition=_grounded_partition(),
    )
    assert is_finding(result)
    with pytest.raises(ValidationError):
        result.title = "mutated"  # type: ignore[misc]


def test_no_finding_without_score() -> None:
    """A Evidence cannot be constructed without a grounding score."""
    with pytest.raises(ValidationError):
        Evidence(  # type: ignore[call-arg]
            title="t",
            description="d",
            severity=Severity.LOW,
            asset="a",
            remediation="r",
        )


def test_indicator_type_serializes_as_string() -> None:
    ind = Indicator(type=IndicatorType.IP, value="192.0.2.5")
    assert ind.model_dump()["type"] == "ip"


# ---------------------------------------------------------------------------
# The grounding bridge
# ---------------------------------------------------------------------------


def test_ground_finding_proceeds_above_tau() -> None:
    result = ground_finding(
        title="Expired TLS certificate",
        description="d",
        severity=Severity.HIGH,
        asset="192.0.2.10:443",
        remediation="rotate",
        partition=_grounded_partition(),
        taxonomy=[OwaspLLM.SENSITIVE_INFORMATION_DISCLOSURE],
    )
    assert is_finding(result)
    assert result.gsar_score >= DEFAULT_TAU_PROCEED
    # Evidence refs are flattened from the partition's claims.
    assert "tool:scan_endpoint:tls_expiry=2026-01-02" in result.evidence_refs
    assert "tool:scan_endpoint:port=443" in result.evidence_refs


def test_ground_finding_abstains_below_tau() -> None:
    result = ground_finding(
        title="Maybe exploitable",
        description="d",
        severity=Severity.MEDIUM,
        asset="192.0.2.11",
        remediation="investigate",
        partition=_ungrounded_partition(),
    )
    assert not is_finding(result)
    assert isinstance(result, Abstention)
    assert result.decision in {Decision.REGENERATE, Decision.REPLAN}
    assert result.candidate_title == "Maybe exploitable"


def test_ground_finding_contradiction_abstains_with_reason() -> None:
    result = ground_finding(
        title="Contested finding",
        description="d",
        severity=Severity.LOW,
        asset="a",
        remediation="r",
        partition=_contradicted_partition(),
    )
    assert isinstance(result, Abstention)
    assert "contradicted" in result.reason


def test_abstention_carries_audit_fields() -> None:
    result = ground_finding(
        title="Withheld",
        description="d",
        severity=Severity.LOW,
        asset="a",
        remediation="r",
        partition=_ungrounded_partition(),
    )
    assert isinstance(result, Abstention)
    assert result.candidate_title == "Withheld"
    assert 0.0 <= result.gsar_score <= 1.0
    assert result.reason


def test_ground_finding_threads_taxonomy_and_indicators() -> None:
    result = ground_finding(
        title="t",
        description="d",
        severity=Severity.CRITICAL,
        asset="a",
        remediation="r",
        partition=_grounded_partition(),
        indicators=[Indicator(type=IndicatorType.DOMAIN, value="phish.example")],
        taxonomy=[AtlasTechnique.PROMPT_INJECTION, OwaspASI.TOOL_MISUSE],
    )
    assert is_finding(result)
    assert result.indicators[0].value == "phish.example"
    dumped = result.model_dump()
    assert "AML.T0051" in dumped["taxonomy"]
    assert "ASI02" in dumped["taxonomy"]


# ---------------------------------------------------------------------------
# Fingerprint surface
# ---------------------------------------------------------------------------


def _mock_classifier(features: dict[str, float]) -> FingerprintVerdict:
    """Deterministic stand-in for a real fingerprinting service."""
    coverage = min(1.0, len(features) / 5.0)
    return FingerprintVerdict(
        model="open-weights-7b",
        engine="vllm",
        hardware="datacenter-gpu",
        confidence=0.92,
        feature_coverage=coverage,
    )


def test_mock_classifier_satisfies_protocol() -> None:
    assert isinstance(_mock_classifier, FingerprintClassifier)


def test_ground_fingerprint_threads_verdict() -> None:
    verdict = _mock_classifier({"ttft": 0.1, "tps": 42.0, "p50": 0.2, "p99": 0.4, "cadence": 1.0})
    result = ground_fingerprint(
        verdict=verdict,
        asset="192.0.2.20:8000",
        partition=_grounded_partition(),
    )
    assert isinstance(result, FingerprintFinding)
    assert result.verdict.engine == "vllm"
    assert result.confidence == verdict.confidence


def test_ground_fingerprint_abstains_when_ungrounded() -> None:
    verdict = _mock_classifier({"ttft": 0.1})
    result = ground_fingerprint(
        verdict=verdict,
        asset="192.0.2.20:8000",
        partition=_ungrounded_partition(),
    )
    assert isinstance(result, Abstention)


# ---------------------------------------------------------------------------
# Severity + taxonomy
# ---------------------------------------------------------------------------


def test_severity_ordering() -> None:
    assert severity_at_least(Severity.CRITICAL, Severity.LOW)
    assert severity_at_least(Severity.HIGH, Severity.HIGH)
    assert not severity_at_least(Severity.LOW, Severity.HIGH)


def test_taxonomy_ids_are_canonical() -> None:
    assert OwaspLLM.PROMPT_INJECTION == "LLM01"
    assert OwaspASI.ROGUE_AGENTS == "ASI10"
    assert AtlasTechnique.PROMPT_INJECTION.startswith("AML.T")


def test_taxonomy_sizes_lock_against_drift() -> None:
    # OWASP lists are exactly 10; ATLAS subset is a curated 10.
    assert len(OwaspLLM) == 10
    assert len(OwaspASI) == 10
    assert len(AtlasTechnique) == 10


def test_top_level_reexports() -> None:
    """The security surface is importable from the package root."""
    import tulip

    assert tulip.Severity is Severity
    assert tulip.ground_finding is ground_finding
