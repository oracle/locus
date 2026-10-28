# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Notebook 28: emergent incident routing — the model picks the specialist protocol.

The default router (Notebook 29) is deterministic: the LLM fills a
``GoalFrame``, then ``_rank_key`` picks a protocol via tuple
comparison. Reproducible, auditable, rule-based — exactly what an
on-call lead wants from an incident router.

This notebook covers the opt-in second mode. When multiple protocols
pass the filter — say a latency regression that could go to the
service-mesh specialist or the capacity-planning specialist — an
``LLMProtocolPicker`` asks the model to make the last-mile choice and
records its rationale on the ``router.protocol.selected`` event. Moving
*only* the disambiguation step to the model keeps the routing audit
trail intact, which matters when the router itself can fan work out to
agents that touch production infrastructure.

Key ideas:
- Filtering, policy gating, capability binding, and builder dispatch
  stay rule-based. Only the disambiguation step moves to the model.
- The picker short-circuits when only one candidate survives the
  filter — no extra LLM call, no extra token spend.
- If the picker raises or returns an unknown protocol id, the compiler
  falls back to ``_rank_key`` and emits
  ``router.protocol.picker_fallback`` so the degradation is
  observable.
- The same five on-call requests run through both routers side by
  side. Rows marked ``≠`` are where the two modes disagreed.

Run it:
    .venv/bin/python examples/notebook_28_emergent_routing.py

It needs a live model: the router's first step fills a structured
``GoalFrame``, which the bundled mock model cannot produce, so under mock the
notebook explains that and exits. Set LOCUS_MODEL_PROVIDER=openai (or
anthropic, or any registry provider with LOCUS_MODEL_ID) and the matching
credentials.

Prerequisites:
- Notebook 07 (structured output).
- Notebook 29 (cognitive router — the default rule-based path).
"""

from __future__ import annotations

import asyncio
from typing import Any

from config import check_structured_output_capable, get_model, print_config

from locus.agent import Agent
from locus.router import (
    CapabilityIndex,
    CognitiveCompiler,
    GoalFrame,
    LLMProtocolPicker,
    PolicyGate,
    ProtocolRegistry,
    Router,
    builtin_protocols,
)
from locus.tools import tool
from locus.tools.registry import create_registry


# ---------------------------------------------------------------------------
# Three small SRE tools shared by both routers
# ---------------------------------------------------------------------------


@tool
def runbook_search(query: str) -> str:
    """Search the on-call runbook knowledge base for an incident family."""
    canned = {
        "latency": (
            "Latency regression (RED method) = elevated p99 on a request path; "
            "route to the service-mesh specialist."
        ),
        "saturation": (
            "Saturation (USE method) = resource exhaustion on a node; route to capacity planning."
        ),
    }
    return canned.get(query.lower(), f"No runbook entry for {query!r}.")


@tool
def get_slo_metric(name: str) -> str:
    """Return the latest value of a named SLO / telemetry metric."""
    return {
        "error_rate": "error_rate=4.2%/min (baseline 0.3%/min — spike)",
        "cpu_throttle": "cpu_throttle=17 pods (warn 10)",
    }.get(name.lower(), f"no metric {name!r}")


@tool
def list_incidents(window_minutes: int = 30) -> str:
    """List recent on-call incidents."""
    return (
        "inc_id=I-101 sev=high family=latency service=checkout p99 spike on /charge\n"
        "inc_id=I-102 sev=medium family=saturation host=node-0231 memory pressure"
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
            "Fill the GoalFrame schema based on the engineer's verb and intent. "
            "required_capabilities may include: runbook_search, metric_probe, incident_list."
        ),
        output_schema=GoalFrame,
    )


def build_routers() -> tuple[Router, Router, Any]:
    """Return (default, emergent, model) — same registry, only the picker differs."""
    model = get_model()

    tools = create_registry(runbook_search, get_slo_metric, list_incidents)
    capabilities = CapabilityIndex(tools)
    capabilities.annotate(
        "runbook_search",
        tool_name="runbook_search",
        description="On-call runbook knowledge-base lookup.",
        domain="runbook",
    )
    capabilities.annotate(
        "metric_probe",
        tool_name="get_slo_metric",
        description="Latest value of a named SLO / telemetry metric.",
        domain="observability",
    )
    capabilities.annotate(
        "incident_list",
        tool_name="list_incidents",
        description="Recent on-call incidents in a window.",
        domain="sre",
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
    return default, emergent, model


# ---------------------------------------------------------------------------
# Five on-call requests. Most route the same way under both modes. The
# COMPARE one is where the picker earns its keep — debate vs
# specialist_fanout both qualify, and the picker can decide with
# intent-level reasoning that _rank_key can't see.
# ---------------------------------------------------------------------------


PROMPTS = [
    "What does the runbook KB say about saturation as an incident family?",
    "Triage the error-rate spike: pull SLO metrics, list recent incidents, correlate.",
    "Outline a three-step hardening plan for our autoscaling policies. Read-only — no prod changes.",
    "Compare connection draining vs pod eviction for shedding load during a latency incident.",
    "Generate a Python function that counts 5xx responses per upstream host, with tests.",
]


async def _selected(router: Router, prompt: str, frame: GoalFrame) -> str:
    """The protocol ``router`` picks for ``frame``, without running it."""
    try:
        runnable = await router.compile(prompt, frame=frame)
    except Exception as exc:  # noqa: BLE001
        return f"ERR: {type(exc).__name__}"
    inner = getattr(runnable, "inner", None)
    return getattr(runnable, "protocol_id", None) or getattr(inner, "protocol_id", "?")


async def main() -> None:
    print("=" * 60)
    print("Notebook 28: emergent incident routing — the model picks the specialist protocol")
    print("=" * 60)
    print()

    print_config()
    print()

    check_structured_output_capable()
    default, emergent, model = build_routers()

    print("─" * 90)
    print("  REQUEST                                         | DEFAULT             | EMERGENT")
    print("─" * 90)

    for prompt in PROMPTS:
        # Routing is the subject here, so compile() — choose and build the
        # topology — rather than dispatch(), which would also run it: ten
        # full multi-agent protocols to print ten protocol ids. Both routers
        # get the same frame, so a ≠ comes from the picker alone, not from
        # two extractions reading the request differently.
        try:
            frame = await default.extract(prompt)
        except Exception as exc:  # noqa: BLE001
            d_pid = e_pid = f"ERR: {type(exc).__name__}"
        else:
            d_pid = await _selected(default, prompt, frame)
            e_pid = await _selected(emergent, prompt, frame)

        marker = "  " if d_pid == e_pid else "≠ "
        print(f"{marker}{prompt[:48]:<48} | {d_pid:<19} | {e_pid}")

    print("─" * 90)
    print(
        "Read each row left-to-right: on-call request → default protocol → emergent\n"
        "protocol. When the two columns differ (≠), the picker's rationale field on\n"
        "the router.protocol.selected event tells you why it chose differently. The\n"
        "filter, policy gate, and builder dispatch are identical in both modes —\n"
        "only the disambiguation step changed."
    )

    close = getattr(model, "close", None)
    if close is not None:
        await close()

    print("\n" + "=" * 60)
    print("Next: Notebook 29 — Cognitive router — risk-gated routing for infra/devops requests")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
