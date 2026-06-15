#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 75: agentic AI red-teaming — grounded findings or abstentions.

The flagship of the agentic-AI-security SDK: point a ``Target`` at an AI
system and run the OWASP-ASI / MITRE-ATLAS red-team suite. Every result is a
grounded ``Finding`` (the attack landed, with tool-backed evidence) or an
``Abstention`` (no evidence — so nothing is asserted). That abstain-by-
construction property is the line no other red-team tool draws: AI scorers
hallucinate vulnerabilities; Tulip refuses to ship one it cannot evidence.

This notebook points the same suite at two targets:
  1. a *vulnerable* bot that obeys injected instructions  -> grounded Findings
  2. a *hardened* bot that refuses them                    -> Abstentions

Runs fully offline — no credentials, no network — via ``Target.from_callable``.
Point ``Target.endpoint(url, ...)`` at a real LLM/agent endpoint to red-team it
for real.

Run:
    python examples/notebook_75_agent_red_team.py
"""

from __future__ import annotations

import asyncio

from tulip.security import Target, is_finding, red_team


def _vulnerable_bot(prompt: str) -> str:
    """A naive assistant with no input/output separation: it parrots back
    whatever it is told, so injected instructions (and their canaries) flow
    straight through."""
    return prompt


def _hardened_bot(_prompt: str) -> str:
    """A guarded assistant that refuses to follow untrusted instructions."""
    return "I can't comply with that. I follow only my approved instructions."


async def _assess(target: Target) -> None:
    results = await red_team(target, suite="owasp-asi")
    findings = [r for r in results if is_finding(r)]
    abstentions = [r for r in results if not is_finding(r)]

    print(f"\n== Red-team report for {target.name!r} ({len(results)} probes) ==")
    print(f"   findings: {len(findings)}   abstentions: {len(abstentions)}")
    for r in results:
        if is_finding(r):
            tags = ", ".join(str(t) for t in r.taxonomy)
            print(f"   [FINDING ] {r.severity.value:<8} {tags:<14} {r.title}")
            print(f"              grounded @ {r.gsar_score:.2f} · evidence: {r.evidence_refs}")
        else:
            print(f"   [ABSTAIN ] {r.candidate_title}")
            print(f"              {r.reason}")


async def main() -> int:
    print("Agentic AI red-team — grounded findings or abstentions (offline demo)")

    vulnerable = Target.from_callable(_vulnerable_bot, name="vulnerable-bot")
    hardened = Target.from_callable(_hardened_bot, name="hardened-bot")

    await _assess(vulnerable)
    await _assess(hardened)

    print(
        "\nThe vulnerable bot produced grounded Findings; the hardened bot abstained "
        "across the board. No vulnerability is ever asserted without evidence."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
