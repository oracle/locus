# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Tests for verify() — independent challenge of a finding.

A well-grounded finding survives; an unsupported or over-claimed one is refuted.
Works on Tulip Findings and on finding-shaped mappings from any other framework.
"""

from __future__ import annotations

from tulip.security import (
    EvidenceQualitySkeptic,
    Finding,
    Refutation,
    Severity,
    Skeptic,
    Verdict,
    verify,
)
from tulip.security.verify import FindingLike


def _strong_finding() -> Finding:
    # Two tool-backed evidence refs, cleared grounding at 1.0 — should survive.
    return Finding(
        title="Direct prompt injection on bot",
        description="Target echoed the injected canary.",
        severity=Severity.HIGH,
        asset="bot",
        remediation="Separate untrusted input from instructions.",
        gsar_score=1.0,
        confidence=1.0,
        evidence_refs=["probe:inj:payload", "probe:inj:response_contains_canary"],
    )


async def test_strong_finding_survives() -> None:
    verdict = await verify(_strong_finding())
    assert isinstance(verdict, Verdict)
    assert verdict.survives
    assert verdict.confidence >= 0.8
    assert verdict.refutations == []


async def test_unsupported_finding_is_refuted_fatally() -> None:
    # No evidence references -> fatal refutation -> cannot survive.
    verdict = await verify({"title": "host compromised", "severity": "critical"})
    assert not verdict.survives
    assert any(r.weight == "fatal" for r in verdict.refutations)
    assert verdict.confidence == 0.0


async def test_ungrounded_external_finding_does_not_survive() -> None:
    # Has refs but no grounding score (external agent didn't ground it).
    verdict = await verify(
        {"title": "suspicious login", "severity": "high", "evidence_refs": ["log:42"]}
    )
    assert not verdict.survives
    assert any("below the proceed threshold" in r.reason for r in verdict.refutations)


async def test_single_ref_high_severity_raises_concern_but_can_survive() -> None:
    finding = Finding(
        title="Expired TLS cert",
        description="…",
        severity=Severity.HIGH,
        asset="host:443",
        remediation="Rotate cert.",
        gsar_score=1.0,
        confidence=1.0,
        evidence_refs=["tool:scan:tls_expiry"],  # only one ref
    )
    verdict = await verify(finding)
    assert any("single evidence reference" in r.reason for r in verdict.refutations)
    assert verdict.survives  # 1.0 - 0.2 = 0.8 >= 0.6


async def test_custom_skeptic_panel_can_kill_a_strong_finding() -> None:
    class _Paranoid:
        name = "paranoid"

        async def challenge(self, finding: FindingLike) -> list[Refutation]:
            return [Refutation("I refuse to trust this.", "fatal")]

    skeptic = _Paranoid()
    assert isinstance(skeptic, Skeptic)  # satisfies the protocol
    verdict = await verify(_strong_finding(), skeptics=[skeptic])
    assert not verdict.survives


async def test_evidence_quality_skeptic_is_a_skeptic() -> None:
    assert isinstance(EvidenceQualitySkeptic(), Skeptic)
    refs = await EvidenceQualitySkeptic().challenge(_strong_finding())
    assert refs == []  # nothing to object to


async def test_confidence_threshold_is_respected() -> None:
    # A strong finding with a strict threshold above its confidence is refuted.
    verdict = await verify(_strong_finding(), threshold=0.99)
    # gsar 1.0, no refutations -> confidence 1.0 -> survives even at 0.99
    assert verdict.survives
    # but a single-ref high-sev (confidence 0.8) fails a 0.9 bar
    one_ref = Finding(
        title="x",
        description="x",
        severity=Severity.HIGH,
        asset="a",
        remediation="x",
        gsar_score=1.0,
        confidence=1.0,
        evidence_refs=["r1"],
    )
    assert not (await verify(one_ref, threshold=0.9)).survives
