# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Live integration test: the grounded SOC analyst on a real model + real AWS.

Runs :func:`create_soc_analyst` against an actual AWS account through the
read-only ``tulip-security-audit`` profile, using a real LLM (Anthropic or
OpenAI). Auto-skips unless BOTH a model key and an AWS identity are present, so
it never runs in a bare CI environment.

The account under test genuinely carries account-level posture issues (e.g.
root access keys), so a competent model produces at least one finding; the
load-bearing assertion is the grounding invariant — every *shipped* finding
cleared the GSAR threshold, and the rest abstained.
"""

from __future__ import annotations

import os

import pytest

from tulip.security import (
    Abstention,
    Finding,
    PostureReport,
    SecurityControls,
    create_soc_analyst,
    ground_report,
    is_finding,
)


def _aws_available() -> bool:
    try:
        import boto3

        profile = os.environ.get("TULIP_AWS_PROFILE", "tulip-security-audit")
        return boto3.Session(profile_name=profile).get_credentials() is not None
    except Exception:
        return False


def _model() -> str | None:
    if os.getenv("ANTHROPIC_API_KEY"):
        return os.getenv("TULIP_ANTHROPIC_TEST_MODEL", "anthropic:claude-haiku-4-5-20251001")
    if os.getenv("OPENAI_API_KEY"):
        return os.getenv("TULIP_OPENAI_TEST_MODEL", "openai:gpt-4o-mini")
    return None


skip_without_stack = pytest.mark.skipif(
    not (_aws_available() and _model()),
    reason="needs both an AWS identity and a model key (ANTHROPIC_API_KEY / OPENAI_API_KEY)",
)


@skip_without_stack
def test_soc_analyst_grounds_account_posture_live() -> None:
    controls = SecurityControls(min_gsar=0.6, min_confidence=0.6)
    analyst = create_soc_analyst(
        model=_model(),  # type: ignore[arg-type]
        controls=controls,
        max_iterations=14,
    )

    result = analyst.run_sync(
        "Review only the account-level IAM posture of this AWS account: whether "
        "the root user has access keys and whether MFA is enabled. Make a few "
        "read-only calls (start with iam GetAccountSummary), cite the exact API "
        "facts as evidence, then submit your report."
    )

    report = result.parsed
    assert isinstance(report, PostureReport)

    grounded = ground_report(report, controls)
    assert len(grounded) == len(report.findings)

    # The grounding invariant: anything that shipped cleared the threshold;
    # everything else is a typed abstention. This must hold regardless of what
    # the model chose to report.
    for g in grounded:
        if is_finding(g):
            assert isinstance(g, Finding)
            assert g.gsar_score >= controls.min_gsar
        else:
            assert isinstance(g, Abstention)
            assert g.reason
