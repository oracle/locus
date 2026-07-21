# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 26: STEWARD — one privacy officer, many data-privacy specialists.

STEWARD is the privacy team's named data-protection-officer agent. An
orchestrator routes a privacy request (a data-subject access/erasure
request) to a chosen set of specialist agents, runs them in parallel
under a semaphore, then correlates their outputs into a single privacy
assessment. Compared with a swarm (Notebook 24), the decision of who
investigates what is centralised in the privacy officer instead of
emerging from capability tags — which means the routing step is a
single, auditable choke point. That matters for compliance: a central
router constrains which systems each specialist may touch and makes
over-collection or unauthorized access far easier to evidence than the
same fleet self-organizing — exactly the accountability and
data-minimization posture GDPR and CCPA expect you to demonstrate.

- ``Specialist`` is a domain-focused agent with tools, a system prompt,
  and a confidence threshold. Tulip ships pre-built ones for logs,
  metrics, traces, and code — all useful evidence sources when tracing
  where personal data flows.
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


async def _llm_call(
    prompt: str, *, system: str = "Reply in one short sentence.", max_tokens: int = 80
) -> str:
    """One model call with a timing/token banner — used for commentary."""
    # Slot B (a cheaper/faster model) is enough for short commentary;
    # falls back to slot A when TULIP_MODEL_ID_B is unset.
    agent = Agent(model=get_model_b(max_tokens=max_tokens), system_prompt=system)
    t0 = time.perf_counter()
    res = await agent.arun(prompt)
    dt = time.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · {res.metrics.prompt_tokens}→{res.metrics.completion_tokens} tokens]"
    )
    return res.message.strip()


async def main():
    print("=" * 60)
    print("Notebook 26: STEWARD privacy officer — orchestrator + specialists + fan-out")
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
        task="In one sentence, what does a log analyst contribute to a data-privacy investigation?"
    )
    dt = time.perf_counter() - t0
    print(f"  [model call: {dt:.2f}s · log_analyst.execute()]")
    if p1.output:
        print(f"  Output: {p1.output[:160]}")

    # =========================================================================
    # Part 2: build a custom specialist with its own tools
    # =========================================================================
    print("\n=== Part 2: Custom Specialists ===\n")

    @tool(
        name="query_data_catalog",
        description="Query the data catalog for stores holding a subject's records",
    )
    async def query_data_catalog() -> str:
        return "Catalog: 3 stores hold records for subject S-8842, latest: crm.customers with email, phone, home address"

    @tool(
        name="check_consent_ledger",
        description="Check the consent ledger for a subject's current permissions",
    )
    async def check_consent_ledger() -> str:
        return "Consent ledger: marketing consent withdrawn 2026-03-12 for S-8842; 2 systems still receiving feeds"

    inventory_specialist = Specialist(
        name="Data Inventory Specialist",
        specialist_type="data_inventory_analyst",
        description="Maps where personal data for a subject lives and which purposes it serves",
        system_prompt="""You are a data-inventory privacy specialist. Your expertise includes:
- Locating personal data for a subject across catalogued stores
- Mapping each data element to a processing purpose and lawful basis
- Correlating storage with the subject's current consent status
- Recommending minimization, restriction, or erasure steps

When analyzing, look for processing without a lawful basis, stale retention, and untracked copies.""",
        tools=[query_data_catalog, check_consent_ledger],
        max_iterations=5,
        confidence_threshold=0.8,
        model=model,
    )

    print(f"Custom Specialist: {inventory_specialist.name}")
    print(f"  Tools: {[t.name for t in inventory_specialist.tools]}")
    _ai_note = await _llm_call("In one sentence, why is a custom Specialist with data-catalog tools better than a generic Agent for personal-data discovery?")
    print(f"AI commentary: {_ai_note}")

    # =========================================================================
    # Part 3: run one specialist on its own
    # =========================================================================
    print("\n=== Part 3: Executing a Specialist ===\n")

    result = await inventory_specialist.execute(
        task="Map every store holding personal data for subject S-8842 and flag any processing without a lawful basis",
        context={"request_id": "DSAR-2026-042", "request_type": "Access + Erasure"},
    )

    print("Specialist Result:")
    print(f"  Success: {result.success}")
    print(f"  Confidence: {result.confidence:.0%}")
    print(f"  Duration: {result.duration_ms:.0f}ms")
    if result.output:
        print(f"  Output: {result.output[:300]}...")

    # =========================================================================
    # Part 4: wire the privacy officer
    # =========================================================================
    print("\n=== Part 4: Creating an Orchestrator ===\n")

    orchestrator = create_orchestrator(
        name="STEWARD Privacy Officer",
        specialists=[log_analyst, metrics_analyst, inventory_specialist],
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
        name="Privacy Response Coordinator",
        description="Coordinates privacy specialists for data-subject request handling",
        system_prompt="""You coordinate specialist agents for data-subject request handling.

When routing:
1. For access/erasure requests -> data_inventory + log specialists
2. For consent disputes -> log + metrics specialists
3. For unknown requests -> all specialists

Prioritize based on the urgency and statutory deadline indicated in the task.""",
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
        specialists=["log_analyst", "data_inventory_analyst"],
        reasoning="Erasure request needs a full data map plus access-log corroboration",
        context={
            "subtasks": {
                "log_analyst": "Search access logs for reads of subject S-8842's records in the last 90 days",
                "data_inventory_analyst": "Enumerate every store holding personal data for subject S-8842",
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
        task="A data subject filed a DSAR (access + erasure) for subject S-8842: locate all "
        "personal data and flag any processing without a lawful basis",
        context={
            "request_type": "access+erasure",
            "affected_systems": ["crm.customers", "marketing.feeds"],
        },
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

    retention_specialist = Specialist(
        name="Retention Policy Analyst",
        specialist_type="retention_analyst",
        description="Reviews retention schedules, minimization gaps, and lawful-basis records",
        system_prompt="You analyze retention evidence: retention schedules, minimization gaps, lawful-basis records.",
        model=model,
    )

    orchestrator.register_specialist(retention_specialist)
    print(f"Added specialist: {retention_specialist.name}")
    print(f"Total specialists: {len(orchestrator.specialists)}")

    t0 = time.perf_counter()
    p8 = await retention_specialist.execute(
        task="In one short sentence, what would you check first if a dataset has no documented retention period?",
    )
    dt = time.perf_counter() - t0
    print(f"  [model call: {dt:.2f}s · retention_specialist.execute()]")
    if p8.output:
        print(f"  Output: {p8.output[:160]}")

    # =========================================================================
    # Part 9: common orchestration shapes
    # =========================================================================
    print("\n=== Part 9: Common Patterns ===\n")

    print("Pattern 1: Parallel Analysis")
    print("  Invoke multiple specialists at once, correlate, produce one privacy assessment.")
    print()

    print("Pattern 2: Sequential Refinement")
    print("  Broad data discovery first, then route to a specific specialist based on findings.")
    print()

    print("Pattern 3: Hierarchical Routing")
    print(
        "  Top-level officer routes to sub-orchestrators per domain (customer data, employee data)."
    )
    print()

    print("Pattern 4: Consensus Analysis")
    print("  Multiple specialists analyse the same records; flag disagreements.")

    # =========================================================================
    # Part 10: things to keep in mind
    # =========================================================================
    print("\n=== Part 10: Best Practices ===\n")

    print("1. Give specialists focused, non-overlapping data domains.")
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
