# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Integration: the full trust pipeline, end to end.

red_team a target → ground findings → verify each → approve a response action
against policy → record every step in a tamper-evident audit trail. This is
"the SDK that prevents security hallucinations" exercised as one flow, offline
(no model/credentials), so it proves the pillars compose:

    grounded → verified → safe-before-action → auditable.
"""

from __future__ import annotations

import pytest

from tulip.security import (
    Action,
    ApprovalOutcome,
    AuditTrail,
    SecurityPolicy,
    Target,
    approve,
    is_finding,
    red_team,
    verify,
)


pytestmark = pytest.mark.integration


def _vulnerable() -> Target:
    return Target.from_callable(lambda p: p, name="vuln-bot")


def _hardened() -> Target:
    return Target.from_callable(lambda _p: "I can't help with that.", name="hardened-bot")


async def test_trust_pipeline_allows_a_verified_finding_and_audits_it() -> None:
    trail = AuditTrail()
    policy = SecurityPolicy(require_verification_score=0.8, max_blast_radius=1)

    # 1. Attack → grounded findings.
    report = await red_team(_vulnerable(), suite="owasp-asi")
    findings = [r for r in report if is_finding(r)]
    assert findings, "vulnerable target should yield grounded findings"
    trail.record("red_team.complete", {"findings": len(findings)})

    # 2. Verify the first finding → it should survive (strong, grounded).
    finding = findings[0]
    verdict = await verify(finding)
    trail.record("verify", {"survives": verdict.survives, "confidence": verdict.confidence})
    assert verdict.survives

    # 3. Propose a low-blast staging action → policy allows it.
    action = Action(name="quarantine", asset=finding.asset, blast_radius=1, environment="staging")
    decision = approve(action, policy=policy, finding=finding, verdict=verdict)
    trail.record("approve", {"outcome": decision.outcome, "reason": decision.reason})
    assert decision.outcome == ApprovalOutcome.ALLOW

    # 4. The whole investigation is a verifiable, tamper-evident record.
    assert len(trail) == 3
    assert trail.verify()


async def test_trust_pipeline_abstains_on_a_hardened_target() -> None:
    report = await red_team(_hardened(), suite="owasp-asi")
    assert not any(is_finding(r) for r in report)  # nothing to verify or act on


async def test_pipeline_blocks_unverifiable_external_finding() -> None:
    # A finding-shaped claim from an external agent, with no evidence, must not
    # be allowed to drive an action.
    verdict = await verify({"title": "host compromised", "severity": "critical"})
    assert not verdict.survives
    decision = approve(
        Action(name="isolate_host", environment="production"),
        policy=SecurityPolicy(),
        verdict=verdict,
    )
    assert decision.outcome == ApprovalOutcome.DENY
