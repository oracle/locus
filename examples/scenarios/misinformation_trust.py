# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Misinformation and human-agent trust exploitation.

Threat: an agent states a confident, plausible, but unsupported conclusion
("contain host WS-0007 — it is exfiltrating data") and a human, trusting
the fluent output, acts on it. The danger is not a tool call — it's a
*recommendation* with no evidence that a person rubber-stamps.

Defense (built-in SDK primitive): route every assertion through
tulip.security.ground_finding. A confident-but-ungrounded conclusion
ABSTAINS, so it never reaches the analyst as an actionable finding — the
human is shown the abstention and its reason, not a false certainty. A
conclusion backed by evidence ships with its grounding score.

Taxonomy: OWASP LLM09 (Misinformation) · OWASP ASI09 (Human-Agent Trust
Exploitation).
"""

from __future__ import annotations

from tulip.reasoning.gsar import Claim, EvidenceType, Partition
from tulip.security import Severity, ground_finding, is_finding


def main() -> None:
    print("Scenario: misinformation / trust exploitation  [LLM09 · ASI09]\n")

    # Fluent, confident, and unsupported — exactly what a human over-trusts.
    unsupported = ground_finding(
        title="Contain WS-0007 immediately — active data exfiltration",
        description="High-confidence narrative with no backing telemetry.",
        severity=Severity.CRITICAL,
        asset="WS-0007",
        remediation="Network-contain the host.",
        partition=Partition(
            ungrounded=[
                Claim(
                    text="the host is probably exfiltrating data",
                    type=EvidenceType.INFERENCE,
                    evidence_refs=["llm:assertion"],
                )
            ],
        ),
    )
    _show("confident, unsupported recommendation", unsupported)

    # The same conclusion, but with evidence — now it is safe to surface.
    supported = ground_finding(
        title="WS-0007 beaconing to a known-malicious host",
        description="DNS + netflow corroborate egress to a flagged C2 IP.",
        severity=Severity.HIGH,
        asset="WS-0007",
        remediation="Network-contain the host; preserve volatile evidence.",
        partition=Partition(
            grounded=[
                Claim(
                    text="netflow shows 41 connections to 198.51.100.23 in 5 min",
                    type=EvidenceType.TOOL_MATCH,
                    evidence_refs=["siem:netflow:WS-0007"],
                ),
                Claim(
                    text="198.51.100.23 is on the threat-intel C2 list",
                    type=EvidenceType.SIGNAL_MATCH,
                    evidence_refs=["intel:c2:198.51.100.23"],
                ),
            ],
        ),
    )
    _show("evidence-backed recommendation", supported)

    print("\nThe human is never handed a confident claim the agent could not ground.")


def _show(label: str, result: object) -> None:
    if is_finding(result):
        print(f"  [SURFACED] {label}: gsar={result.gsar_score:.2f} — {result.title}")
    else:
        print(f"  [ABSTAIN]  {label}: {result.reason}")


if __name__ == "__main__":
    main()
