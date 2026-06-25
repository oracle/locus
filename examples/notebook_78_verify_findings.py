#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 78: verify findings — the SDK that prevents security hallucinations.

A `Evidence` is a *claim*. Before you act on it, `verify()` puts it through an
independent skeptic that challenges the evidence and scores confidence. A
well-grounded finding **survives**; an unsupported or fabricated one is
**refuted** — so a hallucinated "critical" never drives a real action.

`verify()` is framework-agnostic: it takes a Tulip `Evidence` *or* a
finding-shaped dict produced by any other agent (LangGraph/CrewAI/anything),
which is what lets Tulip sit above the stack as the verification layer.

Runs fully offline.

Run:
    python examples/notebook_78_verify_findings.py
"""

from __future__ import annotations

import asyncio

from tulip.security import Target, is_finding, red_team, verify


def _vulnerable_bot(prompt: str) -> str:
    return prompt  # naive bot: obeys injected instructions (see notebook 76)


def _show(label: str, verdict: object) -> None:
    v = verdict  # VerificationResult
    status = "SURVIVES" if v.survives else "REFUTED"  # type: ignore[attr-defined]
    print(f"\n[{status}] {label}  (confidence {v.confidence:.2f})")  # type: ignore[attr-defined]
    for r in v.refutations:  # type: ignore[attr-defined]
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
        "\nOnly the grounded finding survives. Tulip refuses to act on a claim whose "
        "evidence it cannot independently stand behind — that is how it prevents "
        "security hallucinations."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
