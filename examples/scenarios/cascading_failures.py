# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Cascading failures across a multi-agent pipeline.

Threat: agent A emits a wrong-but-confident result; agent B builds on it;
agent C acts on B's output. One ungrounded conclusion at the top compounds
into a confident, wholly-wrong action at the bottom. Multi-agent systems
amplify a single bad inference.

Defense (built-in SDK primitive): put a grounding gate between stages.
Each stage's output must clear tulip.security.ground_finding before the
next stage consumes it; an ungrounded intermediate ABSTAINS and the
pipeline halts there instead of propagating the error downstream — a
circuit-breaker built from evidence.

Taxonomy: OWASP ASI08 (Cascading Failures).
"""

from __future__ import annotations

from tulip.reasoning.gsar import Claim, EvidenceType, Partition
from tulip.security import Severity, ground_finding, is_finding


def stage_gate(name: str, partition: Partition) -> bool:
    """A stage may propagate only if its claim grounds; else the pipeline stops."""
    result = ground_finding(
        title=f"{name} intermediate",
        description="stage output offered to the next stage",
        severity=Severity.MEDIUM,
        asset="pipeline",
        remediation="n/a",
        partition=partition,
    )
    if is_finding(result):
        print(f"  [PASS]  {name}: grounded (gsar={result.gsar_score:.2f}) → propagate")
        return True
    print(f"  [HALT]  {name}: {result.reason} → pipeline stops, no cascade")
    return False


def main() -> None:
    print("Scenario: cascading failures  [ASI08]\n")

    print("Pipeline A → B → C, with a grounding gate between stages:\n")

    # Stage A produces a confident but unsupported inference.
    a_ok = stage_gate(
        "stage-A (classify)",
        Partition(
            ungrounded=[
                Claim(
                    text="the alert is probably a false positive",
                    type=EvidenceType.INFERENCE,
                    evidence_refs=["A:hunch"],
                )
            ],
        ),
    )
    if not a_ok:
        print("\n  stage-B and stage-C never run on A's bad output. Cascade averted.")

    # For contrast: a grounded stage that is allowed to propagate.
    print()
    stage_gate(
        "stage-A' (classify, evidenced)",
        Partition(
            grounded=[
                Claim(
                    text="the alerting rule fired on a sanctioned scanner (CMDB-confirmed)",
                    type=EvidenceType.TOOL_MATCH,
                    evidence_refs=["cmdb:asset:WS-0142"],
                )
            ],
        ),
    )

    print("\nThe cascade is broken at the first stage that cannot ground its claim.")


if __name__ == "__main__":
    main()
