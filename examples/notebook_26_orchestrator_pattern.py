# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL
"""
Notebook 27: orchestrator — one supervisor, many specialists, parallel fan-out.

An orchestrator routes a task to a chosen set of specialist agents, runs
them in parallel under a semaphore, then correlates their outputs into a
single summary. Compared with a swarm (Notebook 25), the decision of who
does what is centralised here instead of emerging from capability tags.

- ``Specialist`` is a domain-focused agent with tools, a system prompt,
  and a confidence threshold. Tulip ships pre-built ones for logs,
  metrics, traces, and code.
- ``Orchestrator`` registers specialists, emits ``RoutingDecision``
  objects, and runs the chosen specialists concurrently behind
  ``max_parallel_specialists`` (an asyncio.Semaphore).
- ``RoutingDecision`` is the typed object the orchestrator's planner
  returns — which specialists, which sub-task per specialist, and the
  reasoning.
- The final ``OrchestrationResult`` carries each specialist's output,
  the decisions trail, and a correlated summary.

Run it:
    .venv/bin/python examples/notebook_32_orchestrator_pattern.py

The default provider is the bundled mock model. Set TULIP_MODEL_PROVIDER=openai
(or anthropic) and the matching credentials to use a live model. Set
``TULIP_MODEL_PROVIDER=mock`` for offline runs.

This notebook uses Tulip's "Model B" slot (``TULIP_MODEL_ID_B``) for the
short commentary calls — set a cheaper model there to cut runtime; falls
back to Model A when unset.

Prerequisites:
- Notebook 08 (Agent basics).
- Notebook 25 (Swarm) for the unsupervised counterpoint.
"""

import asyncio
import time

from config import get_model, get_model_b, print_config

from tulip.agent import Agent
from tulip.multiagent import (
    Orchestrator,
    RoutingDecision,
    Specialist,
    create_code_analyst,
    create_log_analyst,
    create_metrics_analyst,
    create_orchestrator,
    create_trace_analyst,
)
from tulip.tools.decorator import tool


def _llm_call(
    prompt: str, *, system: str = "Reply in one short sentence.", max_tokens: int = 80
) -> str:
    """One model call with a timing/token banner — used for commentary."""
    # Slot B (a cheaper/faster model) is enough for short commentary;
    # falls back to slot A when TULIP_MODEL_ID_B is unset.
    agent = Agent(model=get_model_b(max_tokens=max_tokens), system_prompt=system)
    t0 = time.perf_counter()
    res = agent.run_sync(prompt)
    dt = time.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · {res.metrics.prompt_tokens}→{res.metrics.completion_tokens} tokens]"
    )
    return res.message.strip()


async def main():
    print("=" * 60)
    print("Notebook 27: orchestrator — supervisor + specialists + parallel fan-out")
    print("=" * 60)
    print()
    print_config()

    model = get_model()

    # =========================================================================
    # Part 1: pre-built specialists
    # =========================================================================
    print("\n=== Part 1: Pre-built Specialists ===\n")

    log_analyst = create_log_analyst(model=model)
    metrics_analyst = create_metrics_analyst(model=model)
    trace_analyst = create_trace_analyst(model=model)
    code_analyst = create_code_analyst(model=model)

    print("Pre-built Specialists:")
    for specialist in [log_analyst, metrics_analyst, trace_analyst, code_analyst]:
        print(f"  - {specialist.name}")
        print(f"    Type: {specialist.specialist_type}")
        print(f"    Description: {specialist.description[:60]}...")
        print()

    t0 = time.perf_counter()
    p1 = await log_analyst.execute(task="In one sentence, summarise what a log analyst does.")
    dt = time.perf_counter() - t0
    print(f"  [model call: {dt:.2f}s · log_analyst.execute()]")
    if p1.output:
        print(f"  Output: {p1.output[:160]}")

    # =========================================================================
    # Part 2: build a custom specialist with its own tools
    # =========================================================================
    print("\n=== Part 2: Custom Specialists ===\n")

    @tool(name="check_database", description="Check database health and connections")
    async def check_database() -> str:
        return "Database: 45/50 connections used, avg query time 250ms"

    @tool(name="check_cache", description="Check cache hit rates")
    async def check_cache() -> str:
        return "Cache hit rate: 85%, memory usage: 2.1GB/4GB"

    database_specialist = Specialist(
        name="Database Specialist",
        specialist_type="database_analyst",
        description="Analyzes database performance, connections, and queries",
        system_prompt="""You are a database specialist. Your expertise includes:
- Analyzing query performance
- Monitoring connection pools
- Identifying slow queries
- Recommending optimizations

When analyzing, look for connection leaks, slow queries, and lock contention.""",
        tools=[check_database, check_cache],
        max_iterations=5,
        confidence_threshold=0.8,
        model=model,
    )

    print(f"Custom Specialist: {database_specialist.name}")
    print(f"  Tools: {[t.name for t in database_specialist.tools]}")
    print(
        f"AI commentary: {_llm_call('In one sentence, why is a custom Specialist with domain tools better than a generic Agent for DB diagnostics?')}"
    )

    # =========================================================================
    # Part 3: run one specialist on its own
    # =========================================================================
    print("\n=== Part 3: Executing a Specialist ===\n")

    result = await database_specialist.execute(
        task="Analyze current database performance and identify issues",
        context={"incident_id": "INC-12345", "reported_issue": "Slow API responses"},
    )

    print("Specialist Result:")
    print(f"  Success: {result.success}")
    print(f"  Confidence: {result.confidence:.0%}")
    print(f"  Duration: {result.duration_ms:.0f}ms")
    if result.output:
        print(f"  Output: {result.output[:300]}...")

    # =========================================================================
    # Part 4: wire the orchestrator
    # =========================================================================
    print("\n=== Part 4: Creating an Orchestrator ===\n")

    orchestrator = create_orchestrator(
        name="Incident Analysis Orchestrator",
        specialists=[log_analyst, metrics_analyst, database_specialist],
        model=model,
    )

    print(f"Orchestrator: {orchestrator.name}")
    print("Registered specialists:")
    for spec_id, spec in orchestrator.specialists.items():
        print(f"  - {spec.name} ({spec_id})")

    # =========================================================================
    # Part 5: orchestrator tuning knobs
    # =========================================================================
    print("\n=== Part 5: Orchestrator Configuration ===\n")

    # max_parallel_specialists caps the asyncio.Semaphore that bounds
    # fan-out. Drop to 1 to serialise (useful when debugging a flaky
    # specialist).
    orchestrator.max_parallel_specialists = 3
    orchestrator.correlation_threshold = 0.7

    print(f"Max parallel specialists: {orchestrator.max_parallel_specialists}")
    print(f"Correlation threshold: {orchestrator.correlation_threshold}")

    custom_orchestrator = Orchestrator(
        name="Custom Orchestrator",
        description="Orchestrates analysis with custom logic",
        system_prompt="""You coordinate specialist agents for incident analysis.

When routing:
1. For performance issues -> metrics + database specialists
2. For error spikes -> log + trace specialists
3. For unknown issues -> all specialists

Prioritize based on urgency indicated in the task.""",
        model=model,
    )
    custom_orchestrator.register_specialists([log_analyst, metrics_analyst])

    print(f"\nCustom orchestrator with {len(custom_orchestrator.specialists)} specialists")

    # =========================================================================
    # Part 6: anatomy of a RoutingDecision
    # =========================================================================
    print("\n=== Part 6: Routing Decisions ===\n")

    routing = RoutingDecision(
        decision_type="invoke",
        specialists=["log_analyst", "metrics_analyst"],
        reasoning="Performance issue requires log and metrics analysis",
        context={
            "subtasks": {
                "log_analyst": "Search for timeout errors in the last hour",
                "metrics_analyst": "Check CPU and memory trends",
            }
        },
    )

    print("Routing Decision:")
    print(f"  Type: {routing.decision_type}")
    print(f"  Specialists: {routing.specialists}")
    print(f"  Reasoning: {routing.reasoning}")
    print(f"  Subtasks: {routing.context.get('subtasks', {})}")

    # =========================================================================
    # Part 7: end-to-end orchestration
    # =========================================================================
    print("\n=== Part 7: Full Orchestration ===\n")

    orch_result = await orchestrator.execute(
        task="API response times have increased from 200ms to 2000ms in the last 30 minutes",
        context={"severity": "high", "affected_services": ["api-gateway", "user-service"]},
    )

    print("Orchestration Result:")
    print(f"  Success: {orch_result.success}")
    print(f"  Duration: {orch_result.duration_ms:.0f}ms")
    # The three specialists ran concurrently behind the semaphore — not
    # serially. Per-specialist budgets average ~5s, so parallel ~= 5s
    # vs serial ~= 15s.
    print(f"  Parallel cap: max_parallel_specialists={orchestrator.max_parallel_specialists}")
    print(f"  Decisions made: {len(orch_result.decisions)}")

    for i, decision in enumerate(orch_result.decisions):
        print(f"\n  Decision {i + 1}: {decision.decision_type}")
        if decision.specialists:
            print(f"    Specialists: {decision.specialists}")

    print("\nSpecialist Results:")
    for spec_id, spec_result in orch_result.specialist_results.items():
        status = "OK" if spec_result.success else f"ERROR: {spec_result.error}"
        print(f"  {spec_id}: {status}")
        if spec_result.output:
            print(f"    Output preview: {spec_result.output[:100]}...")

    if orch_result.summary:
        print("\nFinal Summary:")
        print(f"  {orch_result.summary[:500]}...")

    # =========================================================================
    # Part 8: register a specialist at runtime
    # =========================================================================
    print("\n=== Part 8: Dynamic Specialist Registration ===\n")

    network_specialist = Specialist(
        name="Network Analyst",
        specialist_type="network_analyst",
        description="Analyzes network connectivity and latency",
        system_prompt="You analyze network issues including DNS, latency, and connectivity.",
        model=model,
    )

    orchestrator.register_specialist(network_specialist)
    print(f"Added specialist: {network_specialist.name}")
    print(f"Total specialists: {len(orchestrator.specialists)}")

    t0 = time.perf_counter()
    p8 = await network_specialist.execute(
        task="In one short sentence, what would you check first if a service had intermittent timeouts?",
    )
    dt = time.perf_counter() - t0
    print(f"  [model call: {dt:.2f}s · network_specialist.execute()]")
    if p8.output:
        print(f"  Output: {p8.output[:160]}")

    # =========================================================================
    # Part 9: common orchestration shapes
    # =========================================================================
    print("\n=== Part 9: Common Patterns ===\n")

    print("Pattern 1: Parallel Analysis")
    print("  Invoke multiple specialists at once, correlate, produce one summary.")
    print()

    print("Pattern 2: Sequential Refinement")
    print("  Broad pass first, then route to a specific specialist based on findings.")
    print()

    print("Pattern 3: Hierarchical Routing")
    print("  Top-level orchestrator routes to sub-orchestrators per domain.")
    print()

    print("Pattern 4: Consensus Analysis")
    print("  Multiple specialists analyse the same data; flag disagreements.")

    # =========================================================================
    # Part 10: things to keep in mind
    # =========================================================================
    print("\n=== Part 10: Best Practices ===\n")

    print("1. Give specialists focused, non-overlapping domains.")
    print("2. Use clear specialist_type names — they show up in audit logs.")
    print("3. Write domain-specific system prompts.")
    print("4. Set max_parallel_specialists to match your provider quota.")
    print("5. Include correlation logic in the summariser.")
    print("6. Handle specialist failures gracefully — one bad specialist shouldn't kill the run.")
    print("7. Track per-specialist confidence and duration.")

    print("\n" + "=" * 60)
    print("Next: Notebook 28 — Specialist Agents")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
