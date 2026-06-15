#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 82: investigate an incident with SecurityContext — no vendor names.

The point of `SecurityContext` is that you reason in *domains*, not vendors. The
whole investigation below never says "Splunk" or "Okta" or "CrowdStrike" — it
says logs, identity, threat-intel, endpoint, actions. Swap a real vendor in by
injecting a provider (`SecurityContext(logs=SplunkLogs())`); the investigation
code does not change.

Runs offline on the bundled reference providers.

Run:
    python examples/notebook_82_investigate_with_ctx.py
"""

from __future__ import annotations

import asyncio

from tulip.security import Action, SecurityContext, Verdict


async def main() -> int:
    ctx = SecurityContext()  # zero config — offline reference providers
    print("Investigating a suspected account compromise (vendor-agnostic)\n")

    # 1. Logs: what happened around the alert.
    logs = await ctx.logs.search("failed login spike", window="6h")
    print(f"1. logs.search -> {logs['count']} event(s) from {logs['source']}")

    # 2. Identity: is the user risky?
    risk = await ctx.identity.risk("mallory@example.com")
    print(f"2. identity.risk -> {risk['risk']} (impossible_travel={risk['impossible_travel']})")

    # 3. Threat intel: enrich an observed indicator.
    intel = await ctx.threat_intel.enrich("198.51.100.23")
    print(
        f"3. threat_intel.enrich -> {intel.get('classification', intel.get('verdict', 'see record'))}"
    )

    # 4. Endpoint: pull the host's recent forensics.
    host = await ctx.endpoint.get_host("WS-0142")
    print(f"4. endpoint.get_host -> timeline from {host.get('source', 'edr')}")

    # 5. Action: propose containment — but gate it through policy first.
    #    (High-confidence verification stands in for a verified finding.)
    verdict = Verdict(survives=True, confidence=0.93, evidence_quality=0.93)
    decision = ctx.actions.request_approval(
        Action(name="disable_user", asset="mallory@example.com", environment="production"),
        verdict=verdict,
    )
    print(f"5. actions.request_approval -> {decision.outcome.upper()}: {decision.reason}")

    print(
        "\nOne investigation, six domains, zero vendor names. The same code runs "
        "against real Splunk / Okta / CrowdStrike once you inject those providers."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
