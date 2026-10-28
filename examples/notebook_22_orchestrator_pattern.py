# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""
Notebook 22: STEWARD — one privacy officer, many data-privacy specialists.

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

Key ideas:
- ``Specialist`` is a domain-focused agent with tools, a system prompt,
  and a confidence threshold. Locus ships pre-built ones for logs,
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
    .venv/bin/python examples/notebook_22_orchestrator_pattern.py

The default provider is OCI GenAI when ``~/.oci/config`` holds a usable profile, otherwise the bundled mock model; set ``LOCUS_MODEL_PROVIDER`` (``oci``, ``openai``, ``anthropic``, ``mock``) to choose. Set
``LOCUS_MODEL_PROVIDER=mock`` for offline runs.

Prerequisites:
- Notebook 02 (Agent basics).
- Notebook 24 (Swarm) for the unsupervised counterpoint.
"""

import asyncio
import time

from config import get_model, print_config

from locus.multiagent import (
    Orchestrator,
    RoutingDecision,
    Specialist,
    create_code_analyst,
    create_log_analyst,
    create_metrics_analyst,
    create_orchestrator,
    create_trace_analyst,
)
from locus.tools.decorator import tool


async def main():
    print("=" * 60)
    print("Notebook 22: STEWARD privacy officer — orchestrator + specialists + fan-out")
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
    print(f"  Parallel cap: max_parallel_specialists={orchestrator.max_parallel_specialists}")
    print(f"  Decisions made: {len(orch_result.decisions)}")

    for i, decision in enumerate(orch_result.decisions):
        print(f"\n  Decision {i + 1}: {decision.decision_type}")
        if decision.specialists:
            print(f"    Specialists: {decision.specialists}")

    # The specialists ran concurrently behind the semaphore, so the
    # orchestration total above should sit near the slowest specialist
    # below rather than near their sum.
    print("\nSpecialist Results:")
    for spec_id, spec_result in orch_result.specialist_results.items():
        status = "OK" if spec_result.success else f"ERROR: {spec_result.error}"
        print(f"  {spec_id}: {status} ({spec_result.duration_ms:.0f}ms)")
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

    print("\n" + "=" * 60)
    print("Next: Notebook 23 — RIGHTSIZER — the cloud-rightsizing specialist")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
