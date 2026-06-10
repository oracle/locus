#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 34: emergent alert routing — the model picks the specialist protocol.

The default router (Notebook 58) is deterministic: the LLM fills a
``GoalFrame``, then ``_rank_key`` picks a protocol via tuple
comparison. Reproducible, auditable, rule-based — exactly what an
auditor wants from a SOC router.

This notebook covers the opt-in second mode. When multiple protocols
pass the filter — say a phishing alert (ATT&CK T1566) that could go to
the email-security specialist or the network-forensics specialist — an
``LLMProtocolPicker`` asks the model to make the last-mile choice and
records its rationale on the ``router.protocol.selected`` event. Moving
*only* the disambiguation step to the model keeps the routing audit
trail intact, which matters when the router itself is part of the SOC's
attack surface (LLM06 Excessive Agency).

- Filtering, policy gating, capability binding, and builder dispatch
  stay rule-based. Only the disambiguation step moves to the model.
- The picker short-circuits when only one candidate survives the
  filter — no extra LLM call, no extra token spend.
- If the picker raises or returns an unknown protocol id, the compiler
  falls back to ``_rank_key`` and emits
  ``router.protocol.picker_fallback`` so the degradation is
  observable.
- The same five analyst requests run through both routers side by
  side. Rows marked ``≠`` are where the two modes disagreed.

Run it:
    .venv/bin/python examples/notebook_34_emergent_routing.py

The default provider is the bundled mock model. Set TULIP_MODEL_PROVIDER=openai
(or anthropic) and the matching credentials to use a live model. Set
``TULIP_MODEL_PROVIDER=mock`` for offline runs — the picker will
emerge with its mock rationale.

Prerequisites:
- Notebook 35 (structured output).
- Notebook 58 (cognitive router — the default rule-based path).
"""

from __future__ import annotations

import asyncio

from config import get_model

from tulip.agent import Agent
from tulip.router import (
    CapabilityIndex,
    CognitiveCompiler,
    GoalFrame,
    LLMProtocolPicker,
    PolicyGate,
    ProtocolRegistry,
    Router,
    builtin_protocols,
)
from tulip.tools import tool
from tulip.tools.registry import create_registry


# ---------------------------------------------------------------------------
# Three small SOC tools shared by both routers
# ---------------------------------------------------------------------------


@tool
def intel_search(query: str) -> str:
    """Search the threat-intel knowledge base for an alert family."""
    canned = {
        "phishing": (
            "Phishing (ATT&CK T1566) = credential-lure family; route to the "
            "email-security specialist."
        ),
        "beaconing": (
            "Beaconing (ATT&CK T1071) = periodic outbound callbacks; route to "
            "network forensics."
        ),
    }
    return canned.get(query.lower(), f"No intel KB entry for {query!r}.")


@tool
def get_detection_metric(name: str) -> str:
    """Return the latest value of a named detection metric."""
    return {
        "failed_logins": "failed_logins=4200/hr (baseline 300/hr — spike)",
        "edr_alerts": "edr_alerts=17 (warn 10)",
    }.get(name.lower(), f"no metric {name!r}")


@tool
def list_alerts(window_minutes: int = 30) -> str:
    """List recent SOC alerts."""
    return (
        "alert_id=A-101 sev=high family=phishing user=jdoe credential-lure email\n"
        "alert_id=A-102 sev=medium family=beaconing host=ws-0231 dns to evil.example"
    )


def _build_compiler(model, capabilities, protocols, picker=None) -> CognitiveCompiler:
    return CognitiveCompiler(
        protocols=protocols,
        capabilities=capabilities,
        policy=PolicyGate(),
        model=model,
        protocol_picker=picker,
    )


def _build_extractor(model) -> Agent:
    return Agent(
        model=model,
        system_prompt=(
            "Fill the GoalFrame schema based on the analyst's verb and intent. "
            "required_capabilities may include: intel_search, metric_probe, alert_list."
        ),
        output_schema=GoalFrame,
    )


def build_routers() -> tuple[Router, Router]:
    """Return (default, emergent) — same registry, only the picker differs."""
    model = get_model()

    tools = create_registry(intel_search, get_detection_metric, list_alerts)
    capabilities = CapabilityIndex(tools)
    capabilities.annotate(
        "intel_search",
        tool_name="intel_search",
        description="Threat-intel knowledge-base lookup.",
        domain="threat-intel",
    )
    capabilities.annotate(
        "metric_probe",
        tool_name="get_detection_metric",
        description="Latest value of a named detection metric.",
        domain="soc",
    )
    capabilities.annotate(
        "alert_list",
        tool_name="list_alerts",
        description="Recent SOC alerts in a window.",
        domain="soc",
    )

    protocols = ProtocolRegistry()
    protocols.register_many(builtin_protocols())

    default = Router(
        extractor=_build_extractor(model),
        compiler=_build_compiler(model, capabilities, protocols),
    )
    emergent = Router(
        extractor=_build_extractor(model),
        compiler=_build_compiler(
            model,
            capabilities,
            protocols,
            picker=LLMProtocolPicker(model=model),
        ),
    )
    return default, emergent


# ---------------------------------------------------------------------------
# Five analyst requests. Most route the same way under both modes. The
# COMPARE one is where the picker earns its keep — debate vs
# specialist_fanout both qualify, and the picker can decide with
# intent-level reasoning that _rank_key can't see.
# ---------------------------------------------------------------------------


PROMPTS = [
    "What does the threat-intel KB say about beaconing as an alert family?",
    "Triage the failed-login spike: pull detection metrics, list recent alerts, correlate.",
    "Outline a three-step hardening plan for our detection rules. Read-only — no prod changes.",
    "Compare email-gateway quarantine vs DNS sinkholing for containing a phishing campaign.",
    "Generate a Python function that counts failed logins per source IP, with tests.",
]


async def main() -> None:
    default, emergent = build_routers()

    print(f"{'─' * 90}")
    print(f"  REQUEST                                         | DEFAULT             | EMERGENT")
    print(f"{'─' * 90}")

    for prompt in PROMPTS:
        try:
            d_res = await default.dispatch(prompt)
            d_pid = d_res.protocol_id
        except Exception as exc:  # noqa: BLE001
            d_pid = f"ERR: {type(exc).__name__}"

        try:
            e_res = await emergent.dispatch(prompt)
            e_pid = e_res.protocol_id
        except Exception as exc:  # noqa: BLE001
            e_pid = f"ERR: {type(exc).__name__}"

        marker = "  " if d_pid == e_pid else "≠ "
        print(f"{marker}{prompt[:48]:<48} | {d_pid:<19} | {e_pid}")

    print(f"{'─' * 90}")
    print(
        "Read each row left-to-right: analyst request → default protocol → emergent\n"
        "protocol. When the two columns differ (≠), the picker's rationale field on\n"
        "the router.protocol.selected event tells you why it chose differently. The\n"
        "filter, policy gate, and builder dispatch are identical in both modes —\n"
        "only the disambiguation step changed."
    )


if __name__ == "__main__":
    asyncio.run(main())
