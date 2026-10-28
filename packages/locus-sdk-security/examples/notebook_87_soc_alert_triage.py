# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Notebook 87: SOC alert triage with SIEM-grounded verdicts.

Demonstrates the grounded/abstain differentiation in the SOC-triage setting.

The core idea
─────────────
A commodity AI SOC agent emits a verdict for every alert it sees, whether or
not it can back that verdict with evidence.  Here, the SOC analyst grounded
every verdict through ``ground_report``: a proposed Evidence only ships if
the evidence the agent actually cited clears the GSAR threshold.  When the
agent opines without evidence, the result is an ``Abstention`` — nothing is
filed, and the analyst knows to review it manually.

The false-positive case matters most.  When the agent sees a noisy / benign
alert and correctly finds no corroborating evidence, an ``Abstention`` is the
right output: don't file a finding you can't prove.

Scenario
────────
Four SIEM alerts arrive in a 1-hour window:
  1. Phishing link click        → SIEM corroboration found → HIGH Evidence
  2. Lateral movement (PsExec)  → EDR timeline + SIEM corroboration → CRITICAL Evidence
  3. C2 beacon (known bad IP)   → Threat-intel enrichment → HIGH Evidence
  4. benign process noise       → No corroborating evidence → Abstention

This runs fully offline using mock SIEM/EDR/intel adapters.  Swap the mocks
for ``security_toolset(siem=True, edr=True, threat_intel=True)`` and a real
model to run against a live environment.

Run it:
    python packages/locus-sdk-security/examples/notebook_87_soc_alert_triage.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from locus_security import (
    PostureEvidence,
    PostureFinding,
    PostureReport,
    SecurityControls,
    Severity,
    ground_report,
    is_finding,
)


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

# The grounding bar is the same for every alert, so build it once.
_CONTROLS = SecurityControls(siem=True, edr=True, threat_intel=True)


def _build_mock_evidence(alert: Alert) -> list[PostureFinding]:
    """
    In production this is where the SOC analyst agent runs:
      tools = security_toolset(siem=True, edr=True, threat_intel=True)
      agent = create_soc_analyst(model=get_model(), tools=tools)
      response = await agent.arun(f"Investigate alert: {alert.title}\n{alert.raw}")

    For this offline demo the proposed finding is derived from the alert's
    raw fields — each corroboration flag stands in for one tool result the
    agent would have cited. No flag set means nothing to cite.
    """
    raw = alert.raw
    evidence: list[PostureEvidence] = []

    if raw.get("siem_corroboration"):
        evidence.append(
            PostureEvidence(
                ref=f"siem:{alert.id}:related_events",
                statement="SIEM shows related events in the alert window",
                grounded=True,
            )
        )
    if raw.get("edr_corroboration"):
        evidence.append(
            PostureEvidence(
                ref=f"edr:{raw.get('src_host', alert.id)}:process_tree",
                statement="EDR timeline shows the execution chain behind the alert",
                grounded=True,
            )
        )
    if raw.get("intel_verdict") == "malicious":
        evidence.append(
            PostureEvidence(
                ref=f"intel:{raw.get('dst_ip', alert.id)}:malicious",
                statement="Threat intel classifies the indicator as malicious",
                grounded=True,
            )
        )
    if not evidence:
        # Nothing corroborates: the agent still writes down what it saw, but
        # cannot mark it grounded — this is the row that should abstain.
        evidence.append(
            PostureEvidence(
                ref=f"edr:{alert.id}:no_ioc",
                statement="EDR shows a benign path; no corroborating IOCs",
                grounded=False,
            )
        )

    grounded_count = sum(1 for e in evidence if e.grounded)
    if raw.get("known_fp") or grounded_count == 0:
        severity = Severity.INFO
        remediation = "No action required. Monitor for recurrence with unexpected network IOCs."
    elif grounded_count >= 2 and raw.get("edr_corroboration"):
        severity = Severity.CRITICAL
        remediation = "Isolate the affected hosts; rotate exposed credentials; open an IR case."
    else:
        severity = Severity.HIGH
        remediation = "Contain per playbook: block the indicator and reset affected credentials."

    return [
        PostureFinding(
            asset=alert.id,
            title=alert.title,
            description="; ".join(e.statement for e in evidence) + ".",
            severity=severity,
            evidence=evidence,
            remediation=remediation,
            taxonomy=[],
        )
    ]


# ---------------------------------------------------------------------------
# Main: triage each alert, ground the report, print results
# ---------------------------------------------------------------------------


def triage_alerts(alerts: Sequence[Alert]) -> None:
    total_findings = 0
    total_abstentions = 0

    for alert in alerts:
        print(f"\n{'─' * 60}")
        print(f"Alert {alert.id}: {alert.title}")

        proposed_findings = _build_mock_evidence(alert)

        report = PostureReport(
            summary=f"SOC triage: {alert.title}",
            findings=proposed_findings,
            confidence=0.95,
        )

        # ground_report applies the GSAR grounding decision to each proposed
        # finding: evidence-backed ones become Evidence; ungrounded ones abstain.
        grounded = ground_report(report, _CONTROLS)

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


async def main() -> None:
    triage_alerts(_ALERTS)


if __name__ == "__main__":
    asyncio.run(main())
