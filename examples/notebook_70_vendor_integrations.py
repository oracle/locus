#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 70: Live vendor integrations — IOC intel, SIEM, GPU probe dispatch.

The earlier notebooks used inline mock tools to keep the focus on agent
mechanics. Real SOC work calls real systems: a threat-intel feed to score
an indicator, a SIEM to pull the events behind an alert, a GPU cloud to run
an inference-fingerprint probe. This notebook wires three *worked* vendor
integrations (``examples/integrations/``) into a triage agent.

Every integration follows one convention: read the vendor credential from
the environment and call the live API when it's set; otherwise return a
deterministic, benign sample so the cookbook runs offline with no account.
The return shape is identical either way, so the agent's reasoning doesn't
change between this offline demo and a live deployment.

- ``enrich_indicator`` — VirusTotal/GreyNoise-shaped IOC reputation
  (``VT_API_KEY``).
- ``query_siem`` — Splunk/Elastic-shaped log/alert search
  (``SIEM_URL`` + ``SIEM_TOKEN``).
- ``dispatch_timing_probe_reference`` — the *offline reference* GPU probe; the
  live RunPod/Lambda probe is ``tulip_integrations.compute.dispatch_timing_probe``
  (``RUNPOD_API_KEY`` / ``LAMBDA_API_KEY``); see notebook 27 for grounding
  the verdict.

Run it:
    .venv/bin/python examples/notebook_70_vendor_integrations.py

The default provider is the bundled mock model, and every vendor tool falls
back to its offline sample, so this runs end-to-end with no credentials. See
``examples/integrations/README.md`` for the live-credential contract.

Prerequisites:
- Notebook 07 (Agent with tools).
- Notebook 27 (CURATOR) — grounds the fingerprint the probe feeds.
"""

import asyncio

from config import get_model, print_config

from tulip.multiagent.specialist import Specialist

# These adapters are now first-class in the SDK (graduated from the example
# cookbook). The old `from integrations.X import ...` paths still work via
# back-compat shims, but the SDK path is the one to use.
from tulip.security import (
    FEATURE_KEYS,
    dispatch_timing_probe_reference,
    enrich_indicator,
    enrich_indicator_tool,
    query_siem,
    siem_query_tool,
)


# The EICAR test hash — a safe, well-known indicator for the offline demo.
_EICAR = "275a021bbfb6489e54d471899f7db9d1663fc695ec2fe2a2c4538aabf651fd0f"


async def main() -> None:
    print("=" * 60)
    print("Notebook 70: Live vendor integrations")
    print("=" * 60)
    print()
    print_config()

    model = get_model()

    # =========================================================================
    # Part 1: the vendor tools, called directly
    # =========================================================================
    # Each integration is a plain callable with a live path and an offline
    # sample. Here we exercise the offline path so the output is deterministic.
    print("\n=== Part 1: Vendor Tools (offline sample path) ===\n")

    print("Threat-intel enrichment:")
    for ioc in (_EICAR, "198.51.100.23", "phish.example.net"):
        rep = enrich_indicator(ioc)
        print(f"  {ioc[:24]:<24} -> {rep['verdict']} (malicious={rep['malicious']})")

    print("\nSIEM query ('powershell', last 6h):")
    hits = query_siem("powershell", window="6h")
    print(f"  {hits['count']} event(s) from {hits['source']}")
    for ev in hits["events"]:
        print(f"    [{ev['severity']}] {ev['host']}: {ev['detail']}")

    print("\nGPU inference-fingerprint probe dispatch (offline reference):")
    feats = dispatch_timing_probe_reference("203.0.113.10:443")
    observed = sum(1 for k in FEATURE_KEYS if k in feats)
    print(f"  features ({observed}/{len(FEATURE_KEYS)} observed): {feats}")

    # =========================================================================
    # Part 2: hand the tools to a triage agent
    # =========================================================================
    # The same integrations, this time as Tulip @tools the agent can call.
    # Under the mock model the agent narrates rather than truly tool-calls;
    # point TULIP_MODEL_PROVIDER at a live model to see it drive the tools.
    print("\n=== Part 2: A triage agent with vendor tools ===\n")

    sentinel = Specialist(
        name="SENTINEL",
        specialist_type="triage",
        description="First-line SOC triage with live vendor tools",
        system_prompt=(
            "You are SENTINEL, a SOC triage analyst. Given an alert, enrich "
            "the indicators (enrich_indicator), pull the supporting events "
            "(query_siem), and state a severity with the evidence behind it. "
            "Never assert a verdict you cannot back with a tool result."
        ),
        tools=[enrich_indicator_tool, siem_query_tool],
        confidence_threshold=0.8,
        max_iterations=6,
        model=model,
    )

    print(f"Specialist: {sentinel.name}  tools={[t.name for t in sentinel.tools]}")

    result = await sentinel.execute(
        task=(
            "Triage this alert: EDR flagged encoded PowerShell spawned by "
            "winword.exe on WS-0142, beaconing to 198.51.100.23. Enrich the "
            "IP and pull the related events."
        ),
        context={"case_id": "IR-2026-070", "host": "WS-0142"},
    )

    print(f"  success={result.success}  confidence={result.confidence:.0%}")
    if result.output:
        print(f"  verdict: {result.output[:240]}")
    if result.error:
        print(f"  error: {result.error}")

    print("\n" + "=" * 60)
    print("Vendor integrations live in examples/integrations/ — set the")
    print("matching credential to swap any offline sample for the live API.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
