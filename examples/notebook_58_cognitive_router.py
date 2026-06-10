#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 53: Cognitive router — natural language to bounded graph.

tulip.router compiles a natural-language request onto existing Tulip
primitives. The LLM never picks topology — it fills a typed GoalFrame;
the router selects a protocol deterministically, and the compiler emits
a real Agent / SequentialPipeline / ParallelPipeline / LoopAgent from a
curated registry.

- Define a small capability set as annotated tools.
- Register all 8 built-in protocols.
- Load SKILL.md packages from examples/skills/ and tag them by domain
  so the compiler attaches the right ones to every emitted Agent.
- Stand up a Router with an Agent(output_schema=GoalFrame) extractor
  plus a CognitiveCompiler.
- Dispatch five distinct user inputs that hit five different protocols
  (answer / plan / diagnose / debate / codegen) and print which
  protocol fired, the runtime shape it compiled to, and the result.

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


# Three small tools the router can bind to specialist agents.


@tool
def kb_search(query: str) -> str:
    """Search the knowledge base for a topic and return a short summary."""
    canned = {
        "rag": "RAG = retrieve documents from a vector store and condition the LLM on them.",
        "router": "tulip.router compiles a typed GoalFrame onto bounded orchestration shapes.",
        "tulip": "Tulip is a zero-LangChain agentic SDK with native multi-agent shapes.",
    }
    return canned.get(query.lower(), f"No KB entry for {query!r}; suggest a richer query.")


@tool
def get_metric(name: str) -> str:
    """Return the latest value of a named metric (mocked)."""
    metrics = {
        "cpu": "cpu=87% (warn threshold 80%)",
        "latency_p99": "latency_p99=420ms (slo 300ms — breach)",
        "errors_5xx": "errors_5xx=0.4% (within 1% budget)",
    }
    return metrics.get(name.lower(), f"No metric named {name!r}.")


@tool
def list_alerts(window_minutes: int = 30) -> str:
    """List recent alerts inside the given window."""
    if window_minutes <= 0:
        return "no window provided"
    return (
        "alert_id=A-101 sev=high svc=checkout latency_p99 breach\n"
        "alert_id=A-102 sev=medium svc=catalog cpu_warn"
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
        description="Look up a topic in the knowledge base.",
        domain="research",
    )
    capabilities.annotate(
        "metric_probe",
        tool_name="get_metric",
        description="Return the latest value of a named metric.",
        domain="observability",
    )
    capabilities.annotate(
        "alert_list",
        tool_name="list_alerts",
        description="List recent alerts in a time window.",
        domain="observability",
    )

    protocols = ProtocolRegistry()
    protocols.register_many(builtin_protocols())

    skills = _load_skills()

    extractor = Agent(
        model=model,
        system_prompt=(
            "You are a goal-frame extractor for the cognitive router. Given the "
            "user's request, fill the provided schema.\n\n"
            "Rules:\n"
            "1. Pick the single best primary_goal that matches the user's verb.\n"
            "2. Risk: LOW for read-only/info; MEDIUM for build/modify/plan tasks; "
            "HIGH only for irreversible production operations.\n"
            "3. Complexity: LOW for one-step; MEDIUM for multi-step; HIGH for "
            "fan-out across multiple specialists.\n"
            "4. required_capabilities must come from this set; do not invent ids:\n"
            "     - kb_search   (knowledge-base lookup)\n"
            "     - metric_probe (latest value of a named metric)\n"
            "     - alert_list  (recent alerts)\n"
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
DEMO_CASES: tuple[tuple[str, GoalFrame], ...] = (
    (
        "What does the tulip router do in the context of this SDK?",
        GoalFrame(
            primary_goal=TaskType.ANSWER,
            domain="research",
            complexity=Complexity.LOW,
            risk=Risk.LOW,
        ),
    ),
    (
        "Write a three-step plan to migrate our checkout service to a new "
        "payment provider, including a validation step.",
        GoalFrame(
            primary_goal=TaskType.PLAN,
            domain="engineering",
            complexity=Complexity.MEDIUM,
            risk=Risk.MEDIUM,
            success_criteria=["plan has 3 numbered steps", "validation step present"],
        ),
    ),
    (
        "Diagnose what's slowing down checkout right now — pull recent "
        "alerts and the latency_p99 metric, correlate them.",
        GoalFrame(
            primary_goal=TaskType.DIAGNOSE,
            domain="observability",
            complexity=Complexity.HIGH,
            risk=Risk.MEDIUM,
            required_capabilities=["metric_probe", "alert_list"],
        ),
    ),
    (
        "Compare two approaches to handling rate limits: token bucket vs "
        "sliding window. Which is better for a high-traffic API?",
        GoalFrame(
            primary_goal=TaskType.COMPARE,
            domain="engineering",
            complexity=Complexity.HIGH,
            risk=Risk.LOW,
        ),
    ),
    (
        "Generate a Python function that returns the SHA-256 of a string, "
        "with a doctest that verifies it on the empty string.",
        GoalFrame(
            primary_goal=TaskType.GENERATE_CODE,
            domain="engineering",
            complexity=Complexity.MEDIUM,
            risk=Risk.MEDIUM,
        ),
    ),
)


async def _dispatch_with_fallback(
    router: Router, prompt: str, fallback: GoalFrame
) -> tuple[str, str]:
    """Run router.dispatch; fall back to the hand-built frame on parse failure.

    The deterministic core (protocol selection, compile, execute) works
    even when the extractor can't produce a valid frame, so the notebook
    runs end-to-end against the mock provider.
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
