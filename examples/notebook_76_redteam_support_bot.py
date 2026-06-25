#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 76: red-team a customer-support chatbot (a real scenario).

Almost every company now ships a customer-support assistant backed by an LLM
and a knowledge base. Two questions a security team has to answer before it
goes live: can someone **inject instructions** into it (directly, or via a
poisoned KB doc), and can they make it **leak data** it was told to protect?

This notebook points Tulip at a support bot and runs the OWASP-ASI / MITRE-ATLAS
suite, then prints graded results — a `Evidence` (the attack worked, here's the
evidence) or an `Abstention` (no proof, so no claim). It assesses two versions:
a *naive* bot with no trust boundary, and a *hardened* one.

It runs fully offline by simulating the bot with ``Target.from_callable``. In
production you would not simulate it — you would point at the real thing:

    target = Target.endpoint("https://support.acme.example/chat",
                             build_payload=lambda p: {"message": p},
                             response_path="reply")

Run:
    python examples/notebook_76_redteam_support_bot.py
"""

from __future__ import annotations

import asyncio

from tulip.security import Target, is_finding, red_team


_KB = {
    "refund": "Refunds are processed in 5-7 business days.",
    "hours": "ACME Support is open 9am-5pm ET, Monday to Friday.",
}


def naive_support_bot(message: str) -> str:
    """A naive RAG support assistant with NO trust boundary.

    Like many first prototypes, it folds the retrieved knowledge-base text, the
    user's message, and its own instructions into one prompt and simply "does
    what the combined prompt says" — so instructions injected via the message
    (or, in production, via a poisoned KB document) get executed. We model that
    failure mode by complying with the input.
    """
    return message  # stands in for an LLM that follows injected instructions


def hardened_support_bot(message: str) -> str:
    """The same assistant, with a trust boundary.

    Untrusted text is treated as data, never instructions; out-of-policy
    requests are refused. Injected canaries never surface, so the probes have
    no evidence and abstain.
    """
    text = message.lower()
    if "refund" in text:
        return _KB["refund"]
    if "hour" in text:
        return _KB["hours"]
    return "I can only help with ACME order and account questions."


async def _assess(target: Target) -> None:
    report = await red_team(target, suite="owasp-asi")
    findings = [r for r in report if is_finding(r)]
    print(
        f"\n== {target.name!r}: {len(findings)} findings / {len(report) - len(findings)} abstentions =="
    )
    for r in report:
        if is_finding(r):
            tags = ", ".join(str(t) for t in r.taxonomy)
            print(f"   [FINDING ] {r.severity.value:<8} {tags:<16} {r.title}")
        else:
            print(f"   [ABSTAIN ] {r.candidate_title}")


async def main() -> int:
    print("Red-teaming a customer-support chatbot (offline simulation)")

    await _assess(Target.from_callable(naive_support_bot, name="acme-support (naive)"))
    await _assess(Target.from_callable(hardened_support_bot, name="acme-support (hardened)"))

    print(
        "\nThe naive bot fails injection and data-exfiltration probes with evidence; "
        "the hardened bot abstains across the board. Point Target.endpoint(...) at your "
        "real /chat endpoint to run this against production."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
