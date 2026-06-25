#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 73: Grounded AWS cloud-posture agent.

A SOC-analyst-shaped agent that audits an AWS account read-only, then
*grounds* every finding it proposes against the API facts it actually
observed. ``create_soc_analyst`` composes two spec-driven, read-only tools —
``describe_aws`` (discover the shape of AWS from botocore's service models)
and ``use_aws`` (run one read-only operation, return the raw response as
evidence) — behind a ``create_deepagent`` core. The agent proposes findings;
``ground_report`` decides which survive: a proposed finding becomes a typed
``Evidence`` only if its cited evidence clears the GSAR threshold, otherwise it
abstains. The model gathers and proposes; Python decides what ships.

This is the differentiator. A commodity "AWS agent" will confidently narrate
misconfigurations it never actually observed. Here, an ungrounded claim cannot
become a Evidence — it abstains — so the report is trustworthy by construction.

Maps to OWASP ASI: Identity & Privilege Abuse (the root-access-key class of
finding); the read-only-by-construction tooling is the control that keeps the
auditor itself from becoming a liability.

Run it:
    python examples/notebook_73_cloud_posture_agent.py

Part 1 (the grounding decision) runs fully offline and deterministically — no
model, no cloud account. Part 2 builds the live agent; it runs against a real
account only when BOTH a real model provider (``TULIP_MODEL_PROVIDER=openai`` /
``anthropic``) and AWS credentials (the read-only ``tulip-security-audit``
profile, or ``TULIP_AWS_PROFILE``) are present. With neither, it prints the
bring-your-own-credentials note and exits cleanly.

Prerequisites:
- Notebook 29 (DeepAgent) — the core this factory wraps.
- For the live Part 2 only: a structured-output-capable provider + an AWS
  identity. The agent is strictly read-only; ``use_aws`` refuses writes.
"""

from __future__ import annotations

import os

from config import get_model

from tulip.security import (
    PostureEvidence,
    PostureFinding,
    PostureReport,
    SecurityControls,
    Severity,
    create_soc_analyst,
    ground_report,
    is_finding,
)
from tulip.security.taxonomy import OwaspASI


# =============================================================================
# Part 1 — the grounding decision, offline and deterministic.
#
# We hand-build the report exactly as the agent would submit it (three proposed
# findings) and run it through ``ground_report``. Two rest on real API
# observations; one is pure speculation. Watch the speculative one abstain.
# =============================================================================


def _sample_report() -> PostureReport:
    return PostureReport(
        summary="Account-level IAM review of 000000000000.",
        confidence=0.9,
        findings=[
            # Grounded in a concrete GetAccountSummary fact → will ship.
            PostureFinding(
                title="Root account has an active access key",
                description=(
                    "The account root user has a long-lived access key, against "
                    "CIS AWS Foundations 1.4. Root keys cannot be scoped and are a "
                    "standing compromise of the entire account."
                ),
                severity=Severity.CRITICAL,
                asset="aws-account:000000000000:root",
                remediation="Delete the root access key; use scoped IAM roles instead.",
                taxonomy=[OwaspASI.IDENTITY_AND_PRIVILEGE_ABUSE],
                evidence=[
                    PostureEvidence(
                        statement="GetAccountSummary reports AccountAccessKeysPresent=1",
                        ref="aws:iam:GetAccountSummary:AccountAccessKeysPresent",
                        grounded=True,
                    )
                ],
            ),
            # Grounded positive observation → ships as an informational Evidence.
            PostureFinding(
                title="Root account MFA is enabled",
                description="The root user has an MFA device, per the account summary.",
                severity=Severity.INFO,
                asset="aws-account:000000000000:root",
                remediation="Maintain MFA; periodically verify the device.",
                evidence=[
                    PostureEvidence(
                        statement="GetAccountSummary reports AccountMFAEnabled=1",
                        ref="aws:iam:GetAccountSummary:AccountMFAEnabled",
                        grounded=True,
                    )
                ],
            ),
            # No observation behind it — only an inference. → abstains.
            PostureFinding(
                title="Possible lateral-movement path via over-broad roles",
                description="The account may have over-privileged roles enabling pivoting.",
                severity=Severity.MEDIUM,
                asset="aws-account:000000000000",
                remediation="Review role trust policies and attached permissions.",
                evidence=[
                    PostureEvidence(
                        statement="An attacker could pivot if such roles existed",
                        ref="inference:no-call-made",
                        grounded=False,
                    )
                ],
            ),
        ],
    )


def part1_grounding_offline() -> None:
    print("\n--- Part 1: the grounding decision (offline, deterministic) ---")
    controls = SecurityControls(min_gsar=0.6)
    report = _sample_report()
    print(f"agent proposed {len(report.findings)} finding(s); grounding each:\n")

    shipped = 0
    for grounded in ground_report(report, controls):
        if is_finding(grounded):
            shipped += 1
            print(f"  SHIP    [{grounded.severity:<8}] {grounded.title}")
            print(f"          gsar={grounded.gsar_score:.2f}  evidence={grounded.evidence_refs}")
        else:
            print(f"  ABSTAIN  {grounded.reason}")
    print(f"\n{shipped}/{len(report.findings)} proposed findings cleared grounding and shipped.")
    print("The speculative finding abstained — it cited no observation. That is the moat.")


# =============================================================================
# Part 2 — the live agent.
#
# Build the analyst with ``create_soc_analyst`` (read-only AWS tools baked in).
# Run it against a real account only when a real model AND AWS creds are
# present; otherwise show the shape and exit cleanly.
# =============================================================================


def _aws_available() -> bool:
    try:
        import boto3

        profile = os.environ.get("TULIP_AWS_PROFILE", "tulip-security-audit")
        return boto3.Session(profile_name=profile).get_credentials() is not None
    except Exception:
        return False


def part2_live_agent() -> None:
    print("\n--- Part 2: the live cloud-posture agent ---")
    controls = SecurityControls(min_gsar=0.6, min_confidence=0.6)
    analyst = create_soc_analyst(model=get_model(), controls=controls, max_iterations=14)

    tool_names = sorted(getattr(t, "name", "?") for t in analyst.config.tools)
    print(f"analyst tools: {tool_names}")
    print("read-only by construction: use_aws refuses any non-describe/list/get call.")

    provider = os.environ.get("TULIP_MODEL_PROVIDER", "mock").lower()
    if provider == "mock" or not _aws_available():
        print(
            "\nLive run skipped (bring-your-own-credentials).\n"
            "  Set TULIP_MODEL_PROVIDER=openai|anthropic (+ the API key) and configure\n"
            "  the read-only tulip-security-audit AWS profile to audit a real account.\n"
            "  Offline, Part 1 already showed the grounding decision that gates the report."
        )
        return

    print("\nrunning the analyst against the live account (read-only)…")
    result = analyst.run_sync(
        "Review the account-level IAM posture: root access keys and root MFA. "
        "Start with iam GetAccountSummary, cite the exact API facts as evidence, "
        "then submit your report."
    )
    report = result.parsed
    if report is None:
        print(f"no structured report (stop_reason={result.stop_reason}).")
        return
    print(f"\nsummary: {report.summary[:200]}")
    for grounded in ground_report(report, controls):
        if is_finding(grounded):
            print(
                f"  SHIP    [{grounded.severity:<8}] {grounded.title} (gsar={grounded.gsar_score:.2f})"
            )
        else:
            print(f"  ABSTAIN  {grounded.reason[:80]}")


def main() -> None:
    part1_grounding_offline()
    part2_live_agent()


if __name__ == "__main__":
    main()
