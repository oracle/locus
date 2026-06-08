#!/usr/bin/env python3
# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Notebook 35: emergent routing — let the model pick the protocol when it's ambiguous.

The default router (Notebook 53) is deterministic: the LLM fills a
``GoalFrame``, then ``_rank_key`` picks a protocol via tuple
comparison. Reproducible, auditable, rule-based.

This notebook covers the opt-in second mode. When multiple protocols
pass the filter, an ``LLMProtocolPicker`` asks the model to make the
last-mile choice and records its rationale on the
``router.protocol.selected`` event.

- Filtering, policy gating, capability binding, and builder dispatch
  stay rule-based. Only the disambiguation step moves to the model.
- The picker short-circuits when only one candidate survives the
  filter — no extra LLM call, no extra token spend.
- If the picker raises or returns an unknown protocol id, the compiler
  falls back to ``_rank_key`` and emits
  ``router.protocol.picker_fallback`` so the degradation is
  observable.
- The same five prompts run through both routers side by side. Rows
  marked ``≠`` are where the two modes disagreed.

Run it:
    .venv/bin/python examples/notebook_40_emergent_routing.py

The default provider is the bundled mock model. Set TULIP_MODEL_PROVIDER=openai
(or anthropic) and the matching credentials to use a live model. Set
``TULIP_MODEL_PROVIDER=mock`` for offline runs — the picker will
emerge with its mock rationale.

Prerequisites:
- Notebook 14 (structured output).
- Notebook 53 (cognitive router — the default rule-based path).
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
# Three small tools shared by both routers
# ---------------------------------------------------------------------------


@tool
def kb_search(query: str) -> str:
    """Search the knowledge base for a topic."""
    canned = {
        "swarm": "Swarm = self-organizing pool claiming tasks from a shared queue.",
        "orchestrator": "Orchestrator = one coordinator picks specialists per sub-task.",
    }
    return canned.get(query.lower(), f"No KB entry for {query!r}.")


@tool
def get_metric(name: str) -> str:
    """Return the latest value of a named metric."""
    return {
        "latency_p99": "latency_p99=420ms (slo 300ms — breach)",
        "cpu": "cpu=87% (warn 80%)",
    }.get(name.lower(), f"no metric {name!r}")


@tool
def list_alerts(window_minutes: int = 30) -> str:
    """List recent alerts."""
    return (
        "alert_id=A-101 sev=high svc=checkout latency_p99 breach\n"
        "alert_id=A-102 sev=medium svc=catalog cpu_warn"
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
            "Fill the GoalFrame schema based on the user's verb and intent. "
            "required_capabilities may include: kb_search, metric_probe, alert_list."
        ),
        output_schema=GoalFrame,
    )


def build_routers() -> tuple[Router, Router]:
    """Return (default, emergent) — same registry, only the picker differs."""
    model = get_model()

    tools = create_registry(kb_search, get_metric, list_alerts)
    capabilities = CapabilityIndex(tools)
    capabilities.annotate(
        "kb_search",
        tool_name="kb_search",
        description="Knowledge-base lookup.",
        domain="research",
    )
    capabilities.annotate(
        "metric_probe",
        tool_name="get_metric",
        description="Latest value of a named metric.",
        domain="observability",
    )
    capabilities.annotate(
        "alert_list",
        tool_name="list_alerts",
        description="Recent alerts in a window.",
        domain="observability",
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
# Five prompts. Most route the same way under both modes. The COMPARE
# one is where the picker earns its keep — debate vs specialist_fanout
# both qualify, and the picker can decide with intent-level reasoning
# that _rank_key can't see.
# ---------------------------------------------------------------------------


PROMPTS = [
    "What does the tulip router do in the context of this SDK?",
    "Diagnose the checkout API latency spike: pull metrics, list alerts, correlate findings.",
    "Outline a three-step refactor plan for our agent test suite. Read-only — no production changes.",
    "Compare swarm vs orchestrator patterns for open-ended research.",
    "Generate a Python function that returns the nth Fibonacci number, with tests.",
]


async def main() -> None:
    default, emergent = build_routers()

    print(f"{'─' * 90}")
    print(f"  PROMPT                                          | DEFAULT             | EMERGENT")
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
        "Read each row left-to-right: prompt → default protocol → emergent protocol.\n"
        "When the two columns differ (≠), the picker's rationale field on the\n"
        "router.protocol.selected event tells you why it chose differently. The\n"
        "filter, policy gate, and builder dispatch are identical in both modes —\n"
        "only the disambiguation step changed."
    )


if __name__ == "__main__":
    asyncio.run(main())
