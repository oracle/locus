# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ControlPolicy + approve() — safe-before-action."""

from __future__ import annotations

from tulip.control import (
    Action,
    ApprovalOutcome,
    ControlPolicy,
    approve,
)
from tulip.security import Evidence, Severity, VerificationResult


def _verdict(confidence: float, *, survives: bool = True) -> VerificationResult:
    return VerificationResult(survives=survives, confidence=confidence, evidence_quality=confidence)


def _finding(severity: Severity = Severity.HIGH) -> Evidence:
    return Evidence(
        title="t",
        description="d",
        severity=severity,
        asset="host-1",
        remediation="r",
        gsar_score=1.0,
        confidence=1.0,
        evidence_refs=["e1", "e2"],
    )


def test_clean_low_risk_action_is_allowed() -> None:
    d = approve(
        Action(name="enrich", asset="host-1", blast_radius=1, environment="staging"),
        policy=ControlPolicy(),
        finding=_finding(),
        verdict=_verdict(0.95),
    )
    assert d.outcome == ApprovalOutcome.ALLOW
    assert d.allowed
    assert d.checks == []


def test_no_verdict_requires_human() -> None:
    d = approve(Action(name="isolate", environment="staging"), policy=ControlPolicy())
    assert d.outcome == ApprovalOutcome.REQUIRE_HUMAN
    assert "no verification" in d.reason


def test_refuted_verdict_is_denied() -> None:
    d = approve(
        Action(name="isolate", environment="staging"),
        policy=ControlPolicy(),
        verdict=_verdict(0.0, survives=False),
    )
    assert d.outcome == ApprovalOutcome.DENY
    assert "survive verification" in d.reason


def test_low_confidence_requires_human() -> None:
    d = approve(
        Action(name="isolate", environment="staging"),
        policy=ControlPolicy(require_verification_score=0.9),
        verdict=_verdict(0.7),
    )
    assert d.outcome == ApprovalOutcome.REQUIRE_HUMAN
    assert "below the bar" in d.reason


def test_blast_radius_requires_human() -> None:
    d = approve(
        Action(name="isolate", blast_radius=12, environment="staging"),
        policy=ControlPolicy(max_blast_radius=10),
        verdict=_verdict(0.95),
    )
    assert d.outcome == ApprovalOutcome.REQUIRE_HUMAN
    assert "blast radius" in d.reason


def test_production_requires_human() -> None:
    d = approve(
        Action(name="isolate", environment="production"),
        policy=ControlPolicy(),
        verdict=_verdict(0.95),
    )
    assert d.outcome == ApprovalOutcome.REQUIRE_HUMAN
    assert "require human approval" in d.reason


def test_deny_for_label_hard_denies() -> None:
    d = approve(
        Action(name="delete_bucket", kind="destructive", environment="staging"),
        policy=ControlPolicy(deny_for=frozenset({"destructive"})),
        verdict=_verdict(0.99),
    )
    assert d.outcome == ApprovalOutcome.DENY
    assert "denied by policy" in d.reason


def test_finding_below_min_severity_is_denied() -> None:
    d = approve(
        Action(name="isolate", environment="staging"),
        policy=ControlPolicy(min_severity=Severity.HIGH),
        finding=_finding(Severity.LOW),
        verdict=_verdict(0.95),
    )
    assert d.outcome == ApprovalOutcome.DENY
    assert "below the policy minimum" in d.reason


def test_strongest_outcome_wins_and_all_checks_recorded() -> None:
    # production (require_human) + refuted verdict (deny) -> DENY wins, both recorded.
    d = approve(
        Action(name="isolate", environment="production", blast_radius=50),
        policy=ControlPolicy(),
        verdict=_verdict(0.0, survives=False),
    )
    assert d.outcome == ApprovalOutcome.DENY
    assert len(d.checks) >= 2  # multiple rules fired
