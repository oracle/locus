#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 58: Cognitive router — risk-gated routing for SOC requests.

tulip.router compiles a natural-language security request onto existing
Tulip primitives. The LLM never picks topology — it fills a typed
GoalFrame; the router selects a protocol deterministically, and the
compiler emits a real Agent / SequentialPipeline / ParallelPipeline /
LoopAgent from a curated registry. PolicyGate is the star, and it is the
control that keeps an autonomous router from over-reaching (OWASP LLM06
Excessive Agency / OWASP ASI02 Tool Misuse): the frame's risk band, not
the model's confidence, decides whether an action runs. A LOW-risk
"summarize this advisory" runs straight through to a direct answer; a
HIGH-risk "scan this subnet and isolate beaconing hosts" compiles behind
an approval interrupt and does not execute until a human says so.

- Define a small SOC capability set as annotated tools.
- Register all 8 built-in protocols.
- Load SKILL.md packages from examples/skills/ and tag them by domain
  so the compiler attaches the right ones to every emitted Agent.
- Stand up a Router with an Agent(output_schema=GoalFrame) extractor
  plus a CognitiveCompiler.
- Dispatch six requests: five hit five different protocols (answer /
  plan / diagnose / debate / codegen) and print which protocol fired
  and the runtime shape; the sixth is HIGH risk and PolicyGate holds
  it for approval instead of executing.

Run it
    # Default: the bundled mock model (set TULIP_MODEL_PROVIDER for a live provider)
    python examples/notebook_58_cognitive_router.py

    # Offline / no credentials (uses fallback frames):
    TULIP_MODEL_PROVIDER=mock python examples/notebook_58_cognitive_router.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from config import get_model

from tulip.agent import Agent
from tulip.router import (
    CapabilityIndex,
    CognitiveCompiler,
    Complexity,
    GoalFrame,
    PolicyDeniedError,
    PolicyGate,
    ProtocolRegistry,
    Risk,
    Router,
    SkillIndex,
    TaskType,
    builtin_protocols,
)
from tulip.router.runtime import FrameExtractionError
from tulip.skills import Skill
from tulip.tools import tool
from tulip.tools.registry import create_registry


# Per-protocol-id description of the runtime shape the builder emits.
# Kept identical to the workbench Protocols tab so the notebook doesn't
# drift from the UI.
RUNTIME_SHAPES: dict[str, str] = {
    "direct_response": "Agent (single call)",
    "plan_execute_validate": "SequentialPipeline of 3 Agents (planner → executor → validator)",
    "specialist_fanout": "ParallelPipeline of N tool-bound Agents",
    "debate": "ParallelPipeline of 2 debaters + judge Agent",
    "codegen_test_validate": "LoopAgent (PASS/FAIL stop)",
    "approval_gated_execution": "Agent wrapped in approval interrupt",
    "a2a_delegate": "A2AClient.invoke against remote endpoint",
    "handoff_chain": "SequentialPipeline of one-tool Agents",
}


# Three small SOC tools the router can bind to specialist agents.
# All outputs are invented demo data.


@tool
def kb_search(query: str) -> str:
    """Search the security knowledge base for a topic and return a short summary."""
    canned = {
        "phishing": "Phishing: credential-lure messages; mitigate with MFA and reporting drills.",
        "ransomware": "Ransomware advisories recommend offline backups and rapid host isolation.",
        "cve-2024-99999": "CVE-2024-99999 (fictional): patch released; no exploitation observed.",
    }
    return canned.get(query.lower(), f"No KB entry for {query!r}; suggest a richer query.")


@tool
def get_metric(name: str) -> str:
    """Return the latest value of a named security metric (mocked)."""
    metrics = {
        "failed_logins": "failed_logins=312/h (warn threshold 100/h — breach)",
        "phishing_reports": "phishing_reports=12/day (baseline 3/day)",
        "edr_alerts": "edr_alerts=4 open (1 critical, host ws-0042)",
    }
    return metrics.get(name.lower(), f"No metric named {name!r}.")


@tool
def list_alerts(window_minutes: int = 30) -> str:
    """List recent SIEM alerts inside the given window (mocked)."""
    if window_minutes <= 0:
        return "no window provided"
    return (
        "alert_id=A-101 sev=high host=ws-0042 impossible-travel login from 198.51.100.7\n"
        "alert_id=A-102 sev=medium host=db-7 failed-login burst (312/h)"
    )


# Router setup: capabilities, protocols, policy gate, compiler, router.


def _load_skills() -> SkillIndex:
    """Load every SKILL.md package under examples/skills/ and tag each
    skill with the domain its frontmatter declares (or "global" when
    absent — those then appear in every domain's catalogue)."""
    idx = SkillIndex()
    skills_dir = (Path(__file__).resolve().parent / "skills").resolve()
    if not skills_dir.is_dir():
        return idx
    for skill in Skill.from_directory(skills_dir):
        domain = (skill.metadata or {}).get("domain", "")
        idx.register(skill, domain=domain)
    return idx


def build_router() -> Router:
    """Stand up a fully configured Router for this demo."""
    model = get_model()

    tools = create_registry(kb_search, get_metric, list_alerts)
    capabilities = CapabilityIndex(tools)
    capabilities.annotate(
        "kb_search",
        tool_name="kb_search",
        description="Look up a topic in the security knowledge base.",
        domain="research",
    )
    capabilities.annotate(
        "metric_probe",
        tool_name="get_metric",
        description="Return the latest value of a named security metric.",
        domain="observability",
    )
    capabilities.annotate(
        "alert_list",
        tool_name="list_alerts",
        description="List recent SIEM alerts in a time window.",
        domain="observability",
    )

    protocols = ProtocolRegistry()
    protocols.register_many(builtin_protocols())

    skills = _load_skills()

    extractor = Agent(
        model=model,
        system_prompt=(
            "You are a goal-frame extractor for a SOC cognitive router. Given "
            "the analyst's request, fill the provided schema.\n\n"
            "Rules:\n"
            "1. Pick the single best primary_goal that matches the analyst's verb.\n"
            "2. Risk: LOW for read-only/info (summaries, lookups); MEDIUM for "
            "plan/build/modify tasks; HIGH for intrusive or irreversible actions "
            "(active scans, host isolation, blocking traffic).\n"
            "3. Complexity: LOW for one-step; MEDIUM for multi-step; HIGH for "
            "fan-out across multiple specialists.\n"
            "4. required_capabilities must come from this set; do not invent ids:\n"
            "     - kb_search   (security knowledge-base lookup)\n"
            "     - metric_probe (latest value of a named security metric)\n"
            "     - alert_list  (recent SIEM alerts)\n"
            "If the request needs a tool that isn't in this list, leave "
            "required_capabilities empty."
        ),
        output_schema=GoalFrame,
    )
    compiler = CognitiveCompiler(
        protocols=protocols,
        capabilities=capabilities,
        policy=PolicyGate(),
        model=model,
        skills=skills,
    )
    print(
        f"[router] {len(protocols.all())} protocols · {len(capabilities)} capabilities · "
        f"{len(skills)} skills",
    )
    return Router(
        extractor=extractor,
        compiler=compiler,
        on_frame=lambda f: print(
            f"  [frame] goal={f.primary_goal.value} domain={f.domain} "
            f"complexity={f.complexity.value} risk={f.risk.value} "
            f"caps={f.required_capabilities}",
        ),
    )


# Demo cases. Each entry: (prompt, fallback_frame). The fallback is used
# only when the extractor can't produce a valid GoalFrame — i.e. against
# the mock provider. A real provider fills the frame from the prompt.
# The first case is the LOW-risk read-only path (summarize an advisory):
# it routes to a direct answer and runs straight through. The final case
# is its deliberate opposite — a HIGH-risk active scan + host isolation.
# PolicyGate routes that to approval_gated_execution and the default
# approval callback denies, so nothing intrusive runs from a notebook.
DEMO_CASES: tuple[tuple[str, GoalFrame], ...] = (
    (
        "Summarize what the advisory for CVE-2024-99999 means for our exposure.",
        GoalFrame(
            primary_goal=TaskType.ANSWER,
            domain="research",
            complexity=Complexity.LOW,
            risk=Risk.LOW,
        ),
    ),
    (
        "Write a three-step plan to roll out MFA to every admin account, "
        "including a validation step.",
        GoalFrame(
            primary_goal=TaskType.PLAN,
            domain="engineering",
            complexity=Complexity.MEDIUM,
            risk=Risk.MEDIUM,
            success_criteria=["plan has 3 numbered steps", "validation step present"],
        ),
    ),
    (
        "Diagnose the spike in failed logins — pull recent alerts and the "
        "failed_logins metric, correlate them.",
        GoalFrame(
            primary_goal=TaskType.DIAGNOSE,
            domain="observability",
            complexity=Complexity.HIGH,
            risk=Risk.MEDIUM,
            required_capabilities=["metric_probe", "alert_list"],
        ),
    ),
    (
        "Compare two containment strategies for a compromised workstation: "
        "full network isolation vs EDR process quarantine. Which preserves "
        "more forensic evidence?",
        GoalFrame(
            primary_goal=TaskType.COMPARE,
            domain="engineering",
            complexity=Complexity.HIGH,
            risk=Risk.LOW,
        ),
    ),
    (
        "Generate a Python function that returns the SHA-256 of a file for "
        "IOC matching, with a doctest that verifies it on empty input.",
        GoalFrame(
            primary_goal=TaskType.GENERATE_CODE,
            domain="engineering",
            complexity=Complexity.MEDIUM,
            risk=Risk.MEDIUM,
        ),
    ),
    (
        "Run an authorized vulnerability scan of subnet 192.0.2.0/24 and "
        "isolate any host beaconing to evil.example.",
        GoalFrame(
            primary_goal=TaskType.REMEDIATE,
            domain="observability",
            complexity=Complexity.MEDIUM,
            risk=Risk.HIGH,
        ),
    ),
)


async def _dispatch_with_fallback(
    router: Router, prompt: str, fallback: GoalFrame
) -> tuple[str, str]:
    """Run router.dispatch; fall back to the hand-built frame on parse failure.

    The deterministic core (protocol selection, policy gate, compile,
    execute) works even when the extractor can't produce a valid frame,
    so the notebook runs end-to-end against the mock provider.
    """
    try:
        result = await router.dispatch(prompt)
        return result.protocol_id, result.text
    except FrameExtractionError as exc:
        print(f"  [extractor unavailable: {exc}; using fallback frame]")
        runnable = await router.compiler.compile(fallback)
        result = await runnable.execute(prompt)
        return result.protocol_id, result.text


async def run_demo() -> None:
    router = build_router()
    fired: dict[str, int] = {}
    for i, (prompt, fallback) in enumerate(DEMO_CASES, start=1):
        print(f"\n=== Prompt {i} ===\n  {prompt}")
        try:
            protocol_id, text = await _dispatch_with_fallback(router, prompt, fallback)
        except PolicyDeniedError as exc:
            # PolicyGate held a HIGH-risk action for human approval; the
            # default approval callback denies, so the scan never ran.
            print("  [policy]   held for approval — not executed")
            print(f"  [verdict]  {exc}")
            fired["(held for approval)"] = fired.get("(held for approval)", 0) + 1
            continue
        except Exception as exc:  # noqa: BLE001 — surface every failure for the demo
            print(f"  [error] {type(exc).__name__}: {exc}")
            continue
        fired[protocol_id] = fired.get(protocol_id, 0) + 1
        shape = RUNTIME_SHAPES.get(protocol_id, "(unknown shape)")
        print(f"  [protocol] {protocol_id}")
        print(f"  [shape]    {shape}")
        head = "\n  ".join(text.strip().splitlines()[:6])
        print(f"  [result]\n  {head}")
    print("\n=== Protocol histogram ===")
    for pid, n in sorted(fired.items()):
        print(f"  {pid:28s} × {n}")


if __name__ == "__main__":
    asyncio.run(run_demo())
