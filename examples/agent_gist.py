# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Tulip Agent — IOC verdict in 15 lines.

A one-call triage agent: hand it an indicator of compromise, get back a
benign / suspicious / malicious verdict with the single strongest reason.
This is the smallest useful shape of the SOC's tier-1 triage agent.

Uses the shared ``config.get_model`` helper, so it runs offline against
the bundled mock model by default and upgrades to a live provider when
``TULIP_MODEL_PROVIDER`` (plus the matching API key) is set. See
``examples/config.py`` and ``docs/concepts/models.md``.
"""

from config import get_model

from tulip.agent import Agent


def main():
    agent = Agent(
        model=get_model(),
        system_prompt=(
            "You are an IOC triage assistant. Give a one-line verdict for each "
            "indicator — benign, suspicious, or malicious — with the single "
            "strongest reason. If the evidence is insufficient, say so rather "
            "than guessing: an unproven verdict is a false positive."
        ),
    )

    # 198.51.100.0/24 is RFC 5737 documentation space — a safe stand-in IOC.
    result = agent.run_sync(
        "A quarantined email links to http://phish.example.net/reset, served "
        "from 198.51.100.23. Verdict?"
    )
    print(f"Verdict: {result.message}")
    print(f"Iterations: {result.metrics.iterations}")


if __name__ == "__main__":
    main()
