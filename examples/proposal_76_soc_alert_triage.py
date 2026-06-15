#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""PROPOSAL — Notebook 76: SOC alert triage with SIEM-grounded verdicts.

STATUS: PROPOSAL — not yet promoted to a full notebook.
Demonstrates the grounded/abstain differentiation in the Pillar A (SOC) context.

The core idea
─────────────
A commodity AI SOC agent emits a verdict for every alert it sees, whether or
not it can back that verdict with evidence.  Here, the SOC analyst grounded
every verdict through ``ground_report``: a proposed Finding only ships if
the evidence the agent actually cited clears the GSAR threshold.  When the
agent opines without evidence, the result is an ``Abstention`` — nothing is
filed, and the analyst knows to review it manually.

The false-positive case matters most.  When the agent sees a noisy / benign
alert and correctly finds no corroborating evidence, an ``Abstention`` is the
right output: don't file a finding you can't prove.

Scenario
────────
Four SIEM alerts arrive in a 1-hour window:
  1. Phishing link click        → SIEM corroboration found → HIGH Finding
  2. Lateral movement (PsExec)  → EDR timeline + SIEM corroboration → CRITICAL Finding
  3. C2 beacon (known bad IP)   → Threat-intel enrichment → HIGH Finding
  4. benign process noise       → No corroborating evidence → Abstention

This runs fully offline using mock SIEM/EDR/intel adapters.  Swap the mocks
for ``security_toolset(siem=True, edr=True, threat_intel=True)`` and a real
model to run against a live environment.

Run:
    python examples/proposal_76_soc_alert_triage.py
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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
from tulip.security.taxonomy import OwaspLLM


# ---------------------------------------------------------------------------
# Mock alert data — stands in for SIEM alert queue
# ---------------------------------------------------------------------------


@dataclass
class Alert:
    id: str
    title: str
    raw: dict


_ALERTS: list[Alert] = [
    Alert(
        id="ALT-001",
        title="Phishing link clicked by user jsmith",
        raw={
            "src_ip": "198.51.100.42",
            "user": "jsmith",
            "url": "http://login.phish.example.net/reset",
            "siem_corroboration": True,  # mock: SIEM found 3 related events
            "edr_corroboration": False,
        },
    ),
    Alert(
        id="ALT-002",
        title="PsExec lateral movement from WKSTN-04 to SRV-FINANCE",
        raw={
            "src_host": "WKSTN-04",
            "dst_host": "SRV-FINANCE",
            "process": "psexec.exe",
            "siem_corroboration": True,
            "edr_corroboration": True,  # mock: EDR timeline shows execution chain
        },
    ),
    Alert(
        id="ALT-003",
        title="Outbound connection to known C2 IP 203.0.113.99",
        raw={
            "dst_ip": "203.0.113.99",
            "intel_verdict": "malicious",  # mock: threat intel confirms bad IP
            "siem_corroboration": True,
            "edr_corroboration": False,
        },
    ),
    Alert(
        id="ALT-004",
        title="Unusual process: msiexec.exe spawned by Teams.exe",
        raw={
            "process": "msiexec.exe",
            "parent": "Teams.exe",
            "siem_corroboration": False,  # mock: no corroborating SIEM events
            "edr_corroboration": False,  # mock: EDR shows benign update path
            "known_fp": True,
        },
    ),
]


# ---------------------------------------------------------------------------
# Mock evidence builder
# The real version calls query_siem(), fetch_host_timeline(), enrich_indicator()
# and returns actual API responses as evidence refs.
# ---------------------------------------------------------------------------


def _build_mock_evidence(alert: Alert) -> tuple[list[PostureFinding], SecurityControls]:
    """
    In production this is where the SOC analyst agent runs:
      tools = security_toolset(siem=True, edr=True, threat_intel=True)
      agent = create_soc_analyst(model=get_model(), tools=tools)
      response = agent.run_sync(f"Investigate alert: {alert.title}\n{alert.raw}")

    For this offline demo we hand-craft the proposed findings and evidence
    that a real agent would produce from tool call results.
    """
    raw = alert.raw
    proposed: list[PostureFinding] = []
    controls = SecurityControls(siem=True, edr=True, threat_intel=True)

    if alert.id == "ALT-001":
        proposed.append(
            PostureFinding(
                asset=alert.id,
                title="User clicked confirmed phishing URL",
                description=(
                    "User jsmith clicked http://login.phish.example.net/reset. "
                    "SIEM corroborates 3 follow-on DNS lookups to the same domain. "
                    "URL not in allowlist."
                ),
                severity=Severity.HIGH,
                evidence=[
                    PostureEvidence(
                        ref="siem:ALT-001:dns_lookups=3",
                        statement="3 DNS lookups to phish.example.net post-click",
                        grounded=True,
                    ),
                    PostureEvidence(
                        ref="siem:ALT-001:url_not_in_allowlist",
                        statement="URL absent from approved domain list",
                        grounded=True,
                    ),
                ],
                remediation="Reset jsmith credentials; block phish.example.net at DNS filter.",
                taxonomy=[OwaspLLM.PROMPT_INJECTION],
            )
        )

    elif alert.id == "ALT-002":
        proposed.append(
            PostureFinding(
                asset=alert.id,
                title="PsExec lateral movement: WKSTN-04 → SRV-FINANCE",
                description=(
                    "PsExec executed on WKSTN-04 targeting SRV-FINANCE. "
                    "EDR timeline shows process tree: explorer.exe → cmd.exe → psexec.exe. "
                    "SIEM corroborates SMB admin share access within same minute."
                ),
                severity=Severity.CRITICAL,
                evidence=[
                    PostureEvidence(
                        ref="edr:WKSTN-04:process_tree:psexec",
                        statement="EDR process tree shows psexec spawned from cmd.exe",
                        grounded=True,
                    ),
                    PostureEvidence(
                        ref="siem:ALT-002:smb_admin_share_access",
                        statement="SMB admin share WKSTN-04→SRV-FINANCE within 60s of psexec",
                        grounded=True,
                    ),
                ],
                remediation="Isolate WKSTN-04 and SRV-FINANCE; rotate all service account credentials.",
                taxonomy=[],
            )
        )

    elif alert.id == "ALT-003":
        proposed.append(
            PostureFinding(
                asset=alert.id,
                title="Outbound C2 beacon to 203.0.113.99 (threat-intel confirmed malicious)",
                description=(
                    "Three hosts beaconed to 203.0.113.99 over 90 minutes. "
                    "Threat intel: IP listed in Emerging Threats + VirusTotal 47/72. "
                    "SIEM shows beaconing pattern (60-second intervals)."
                ),
                severity=Severity.HIGH,
                evidence=[
                    PostureEvidence(
                        ref="intel:203.0.113.99:emerging_threats",
                        statement="IP listed in Emerging Threats ruleset",
                        grounded=True,
                    ),
                    PostureEvidence(
                        ref="intel:203.0.113.99:vt_score=47/72",
                        statement="VirusTotal: 47/72 engines flagged as malicious",
                        grounded=True,
                    ),
                    PostureEvidence(
                        ref="siem:ALT-003:beacon_pattern_60s",
                        statement="60-second beaconing interval observed over 90 min",
                        grounded=True,
                    ),
                ],
                remediation="Block 203.0.113.99 at perimeter; image affected hosts; rotate keys.",
                taxonomy=[],
            )
        )

    elif alert.id == "ALT-004":
        # No corroboration — the agent correctly finds nothing to cite
        proposed.append(
            PostureFinding(
                asset=alert.id,
                title="msiexec.exe spawned by Teams.exe — possible software update",
                description=(
                    "msiexec.exe observed as child of Teams.exe. "
                    "No SIEM corroboration; EDR shows standard Microsoft Teams auto-update path. "
                    "No network IOCs observed."
                ),
                severity=Severity.INFO,
                evidence=[
                    PostureEvidence(
                        ref="edr:WKSTN-06:msiexec-teams-no-ioc",
                        statement="EDR shows standard Teams auto-update path; no network IOCs",
                        grounded=False,  # agent found no corroborating evidence
                    ),
                ],
                remediation="No action required. Monitor for recurrence with unexpected network IOCs.",
                taxonomy=[],
            )
        )

    return proposed, controls  # noqa: RET504


# ---------------------------------------------------------------------------
# Main: triage each alert, ground the report, print results
# ---------------------------------------------------------------------------


def triage_alerts(alerts: Sequence[Alert]) -> None:
    total_findings = 0
    total_abstentions = 0

    for alert in alerts:
        print(f"\n{'─' * 60}")
        print(f"Alert {alert.id}: {alert.title}")

        proposed_findings, controls = _build_mock_evidence(alert)

        report = PostureReport(
            summary=f"SOC triage: {alert.title}",
            findings=proposed_findings,
            confidence=0.95,
        )

        # ground_report applies the GSAR grounding decision to each proposed
        # finding: evidence-backed ones become Finding; ungrounded ones abstain.
        grounded = ground_report(report, controls)

        for result in grounded:
            if is_finding(result):
                total_findings += 1
                sev = result.severity.value.upper()
                tags = ", ".join(str(t) for t in result.taxonomy) or "—"
                print(f"  [FINDING ] {sev:<8} {result.title}")
                print(f"             grounded @ {result.gsar_score:.2f}")
                print(f"             evidence : {result.evidence_refs}")
                print(f"             taxonomy : {tags}")
            else:
                total_abstentions += 1
                print(f"  [ABSTAIN ] {result.candidate_title}")
                print(f"             reason   : {result.reason}")
                print(f"             gsar     : {result.gsar_score:.2f} (below threshold)")

    print(f"\n{'═' * 60}")
    print(f"Triage complete: {total_findings} findings filed, {total_abstentions} abstentions.")
    print(
        "The abstention(s) above are alerts a commodity AI SOC would have filed as findings.\n"
        "Here they were withheld because no corroborating evidence was cited — reducing\n"
        "analyst noise and false-positive rate."
    )


if __name__ == "__main__":
    triage_alerts(_ALERTS)
