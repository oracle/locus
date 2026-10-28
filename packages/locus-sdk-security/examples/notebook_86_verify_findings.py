# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Notebook 86: verify findings — the SDK that prevents security hallucinations.

A `Evidence` is a *claim*. Before you act on it, `verify()` puts it through an
independent skeptic that challenges the evidence and scores confidence. A
well-grounded finding **survives**; an unsupported or fabricated one is
**refuted** — so a hallucinated "critical" never drives a real action.

`verify()` is framework-agnostic: it takes a Locus `Evidence` *or* a
finding-shaped dict produced by any other agent, which is what lets Locus
sit above the stack as the verification layer.

Runs fully offline.

Run it:
    python packages/locus-sdk-security/examples/notebook_86_verify_findings.py
"""

from __future__ import annotations

import asyncio

from locus_security import Target, VerificationResult, is_finding, red_team, verify


def _vulnerable_bot(prompt: str) -> str:
    return prompt  # naive bot: obeys injected instructions (see notebook 84)


def _show(label: str, v: VerificationResult) -> None:
    status = "SURVIVES" if v.survives else "REFUTED"
    print(f"\n[{status}] {label}  (confidence {v.confidence:.2f})")
    for r in v.refutations:
        print(f"    - ({r.weight}) {r.reason}")


async def main() -> int:
    print("verify() — does the evidence actually support the claim?")

    # 1. A REAL finding from red-teaming a vulnerable bot — well grounded.
    report = await red_team(Target.from_callable(_vulnerable_bot, name="bot"), suite="owasp-asi")
    real_finding = next(r for r in report if is_finding(r))
    _show(f"real finding: {real_finding.title}", await verify(real_finding))

    # 2. A FABRICATED finding from some other agent — no evidence at all.
    hallucination = {"title": "Host 10.0.0.5 is compromised", "severity": "critical"}
    _show("hallucinated finding (no evidence)", await verify(hallucination))

    # 3. An external finding that has refs but was never grounded.
    ungrounded = {
        "title": "Possible data exfiltration",
        "severity": "high",
        "evidence_refs": ["siem:alert:9931"],
    }
    _show("external finding (ungrounded)", await verify(ungrounded))

    print(
        "\nOnly the grounded finding survives. Locus refuses to act on a claim whose "
        "evidence it cannot independently stand behind — that is how it prevents "
        "security hallucinations."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
