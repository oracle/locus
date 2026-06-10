# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 26: MARSHAL — one incident commander, many security specialists.

MARSHAL is the SOC's named incident-commander agent. An orchestrator
routes an incident to a chosen set of specialist agents, runs them in
parallel under a semaphore, then correlates their outputs into a single
situation report. Compared with a swarm (Notebook 24), the decision of
who investigates what is centralised in the commander instead of emerging
from capability tags — which means the routing step is a single,
auditable choke point. That matters for agentic risk: a central router
constrains tool reach and makes goal-hijack or rogue-specialist behaviour
(OWASP ASI01 Agent Goal Hijack, ASI10 Rogue Agents) far easier to detect
than the same fleet self-organizing.

- ``Specialist`` is a domain-focused agent with tools, a system prompt,
  and a confidence threshold. Tulip ships pre-built ones for logs,
  metrics, traces, and code — all useful evidence sources in an incident.
- ``Orchestrator`` registers specialists, emits ``RoutingDecision``
  objects, and runs the chosen specialists concurrently behind
  ``max_parallel_specialists`` (an asyncio.Semaphore).
- ``RoutingDecision`` is the typed object the orchestrator's planner
  returns — which specialists, which sub-task per specialist, and the
  reasoning.
- The final ``OrchestrationResult`` carries each specialist's output,
  the decisions trail, and a correlated summary.

Run it:
    .venv/bin/python examples/notebook_26_orchestrator_pattern.py

The default provider is the bundled mock model. Set TULIP_MODEL_PROVIDER=openai
(or anthropic) and the matching credentials to use a live model. Set
``TULIP_MODEL_PROVIDER=mock`` for offline runs.

This notebook uses Tulip's "Model B" slot (``TULIP_MODEL_ID_B``) for the
short commentary calls — set a cheaper model there to cut runtime; falls
back to Model A when unset.

Prerequisites:
- Notebook 06 (Agent basics).
- Notebook 24 (Swarm) for the unsupervised counterpoint.
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
    print("Notebook 26: MARSHAL incident commander — orchestrator + specialists + fan-out")
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
    p1 = await log_analyst.execute(
        task="In one sentence, what does a log analyst contribute to incident response?"
    )
    dt = time.perf_counter() - t0
    print(f"  [model call: {dt:.2f}s · log_analyst.execute()]")
    if p1.output:
        print(f"  Output: {p1.output[:160]}")

    # =========================================================================
    # Part 2: build a custom specialist with its own tools
    # =========================================================================
    print("\n=== Part 2: Custom Specialists ===\n")

    @tool(name="query_edr", description="Query EDR for open detections on a host")
    async def query_edr() -> str:
        return "EDR: 3 open detections on WS-0142, latest: encoded PowerShell from winword.exe"

    @tool(name="check_auth_logs", description="Check recent authentication failures")
    async def check_auth_logs() -> str:
        return "Auth logs: 17 failed logins for svc-backup from 192.0.2.55 in the last hour"

    endpoint_specialist = Specialist(
        name="Endpoint Specialist",
        specialist_type="endpoint_analyst",
        description="Analyzes EDR detections, process activity, and host state",
        system_prompt="""You are an endpoint security specialist. Your expertise includes:
- Reviewing EDR detections and process trees
- Spotting suspicious parent-child process pairs
- Correlating host activity with authentication logs
- Recommending containment steps

When analyzing, look for unsigned binaries, encoded command lines, and odd logon patterns.""",
        tools=[query_edr, check_auth_logs],
        max_iterations=5,
        confidence_threshold=0.8,
        model=model,
    )

    print(f"Custom Specialist: {endpoint_specialist.name}")
    print(f"  Tools: {[t.name for t in endpoint_specialist.tools]}")
    print(
        f"AI commentary: {_llm_call('In one sentence, why is a custom Specialist with EDR tools better than a generic Agent for endpoint triage?')}"
    )

    # =========================================================================
    # Part 3: run one specialist on its own
    # =========================================================================
    print("\n=== Part 3: Executing a Specialist ===\n")

    result = await endpoint_specialist.execute(
        task="Analyze the open EDR detections on WS-0142 and identify the likely entry point",
        context={"incident_id": "IR-2026-042", "reported_issue": "Encoded PowerShell alert"},
    )

    print("Specialist Result:")
    print(f"  Success: {result.success}")
    print(f"  Confidence: {result.confidence:.0%}")
    print(f"  Duration: {result.duration_ms:.0f}ms")
    if result.output:
        print(f"  Output: {result.output[:300]}...")

    # =========================================================================
    # Part 4: wire the incident commander
    # =========================================================================
    print("\n=== Part 4: Creating an Orchestrator ===\n")

    orchestrator = create_orchestrator(
        name="MARSHAL Incident Commander",
        specialists=[log_analyst, metrics_analyst, endpoint_specialist],
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
        name="SOC Commander",
        description="Coordinates security specialists for incident analysis",
        system_prompt="""You coordinate specialist agents for security-incident analysis.

When routing:
1. For endpoint alerts -> endpoint + log specialists
2. For credential abuse -> log + metrics specialists
3. For unknown alerts -> all specialists

Prioritize based on the severity indicated in the task.""",
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
        specialists=["log_analyst", "endpoint_analyst"],
        reasoning="Encoded-PowerShell alert needs host context plus log corroboration",
        context={
            "subtasks": {
                "log_analyst": "Search auth logs for activity from 192.0.2.55 in the last 24h",
                "endpoint_analyst": "Pull the process tree for the flagged PowerShell on WS-0142",
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
        task="EDR raised a high-severity alert: encoded PowerShell spawned by winword.exe "
        "on workstation WS-0142 at 09:14 UTC",
        context={"severity": "high", "affected_assets": ["WS-0142", "fileserver FS-03"]},
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
        name="Network Forensics Analyst",
        specialist_type="network_forensics",
        description="Analyzes firewall logs, DNS activity, and beaconing patterns",
        system_prompt="You analyze network evidence: DNS queries, firewall denies, beaconing.",
        model=model,
    )

    orchestrator.register_specialist(network_specialist)
    print(f"Added specialist: {network_specialist.name}")
    print(f"Total specialists: {len(orchestrator.specialists)}")

    t0 = time.perf_counter()
    p8 = await network_specialist.execute(
        task="In one short sentence, what would you check first if a host made periodic connections to an unknown domain?",
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
    print("  Invoke multiple specialists at once, correlate, produce one sitrep.")
    print()

    print("Pattern 2: Sequential Refinement")
    print("  Broad triage first, then route to a specific specialist based on findings.")
    print()

    print("Pattern 3: Hierarchical Routing")
    print("  Top-level commander routes to sub-orchestrators per domain (endpoint, network).")
    print()

    print("Pattern 4: Consensus Analysis")
    print("  Multiple specialists analyse the same evidence; flag disagreements.")

    # =========================================================================
    # Part 10: things to keep in mind
    # =========================================================================
    print("\n=== Part 10: Best Practices ===\n")

    print("1. Give specialists focused, non-overlapping evidence domains.")
    print("2. Use clear specialist_type names — they show up in the audit trail.")
    print("3. Write domain-specific system prompts.")
    print("4. Set max_parallel_specialists to match your provider quota.")
    print("5. Include correlation logic in the summariser.")
    print("6. Handle specialist failures gracefully — one bad specialist shouldn't kill the run.")
    print("7. Track per-specialist confidence and duration.")

    print("\n" + "=" * 60)
    print("Next: Notebook 27 — Specialist Agents")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
