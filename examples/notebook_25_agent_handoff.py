# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""
Notebook 25: Support tier escalation — L1 → L2 → L3 handoff with typed context.

A handoff is one support agent saying "I'm done, please take this further."
The source agent packages the ticket, its findings, and an explicit reason
into a typed ``HandoffContext`` so the next tier inherits the full case
state — not just a string. No re-asking the customer, no lost context.

The running case is a classic support tier escalation: a customer reports
that a paid subscription was charged twice but only one plan is active. The
ticket walks L1 → L2 → L3, with billing-system lookups and account history
gathered along the way. The typed chain is also the audit trail — every
transfer, reason, and confidence value is recorded for the QA review.

Key ideas:
- ``HandoffContext`` carries the source/target ids, original task,
  conversation summary, findings dict, confidence, instructions, and the
  full handoff chain.
- ``HandoffReason`` enumerates why the handoff happened — SPECIALIZATION,
  ESCALATION, DELEGATION, COMPLETION, FAILURE. The reason drives prompt
  templating and audit trails.
- ``HandoffManager`` registers a pool of agents, enforces a
  ``max_handoff_chain`` cap (prevents loops), and records every transfer
  for inspection or replay.
- ``manager.chain_handoff([a, b, c], task)`` walks the chain end-to-end
  with each tier inheriting the previous one's findings.

Run it:
    .venv/bin/python examples/notebook_25_agent_handoff.py

The default provider is OCI GenAI when ``~/.oci/config`` holds a usable profile, otherwise the bundled mock model; set ``LOCUS_MODEL_PROVIDER`` (``oci``, ``openai``, ``anthropic``, ``mock``) to choose. Set
``LOCUS_MODEL_PROVIDER=mock`` for offline runs.

This notebook fires five handoffs serially, so it can use Locus's "Model B"
slot — a second, typically cheaper model id read from ``LOCUS_MODEL_ID_B``
— for the L1 front-line seat. With Model B unset, the slot collapses to Model A.

Prerequisites:
- Notebook 02 (Agent basics).
- Notebook 24 (Swarm) for the unsupervised counterpoint.
"""

import asyncio
import time

from config import get_model, get_model_b, print_config

from locus.core.messages import Message
from locus.core.state import AgentState
from locus.multiagent.handoff import (
    HandoffContext,
    HandoffReason,
    create_handoff_agent,
    create_handoff_manager,
)


# What each HandoffReason means for a support desk. The reason is part of
# the typed context, so it also shows up in the prompt the target receives
# and in the audit trail.
REASON_DESCRIPTIONS = {
    HandoffReason.SPECIALIZATION: "Target has better capabilities for this ticket",
    HandoffReason.ESCALATION: "Issue needs a higher tier or more authority",
    HandoffReason.DELEGATION: "Sub-task delegated to another support agent",
    HandoffReason.COMPLETION: "Work completed, returning to parent",
    HandoffReason.FAILURE: "Agent failed, trying another approach",
}


def _banner(label: str, dt: float) -> None:
    """Print a uniform [model call …] banner so each Part shows it hit the model."""
    print(f"  [model call · {label}: {dt:.2f}s]")


async def main():
    print("=" * 60)
    print("Notebook 25: Support tier escalation — typed L1 → L2 → L3 handoffs")
    print("=" * 60)
    print()
    print_config()

    # =========================================================================
    # Part 1: build the support tiers and wire allowed escalation paths
    # =========================================================================
    print("\n=== Part 1: Creating Handoff Agents ===\n")

    # The L1 front-line seat reads from Locus's "Model B" slot
    # (env: LOCUS_MODEL_ID_B). Set a cheaper/faster model there to cut
    # round-trip latency; falls back to Model A when unset. Models are
    # attached here, once, so every later part uses the same agents.
    triage_model = get_model_b(max_tokens=2000)
    model = get_model(max_tokens=2000)

    l1_agent = create_handoff_agent(
        name="L1 Support Agent",
        description="First-line ticket triage and routing",
        system_prompt="You are an L1 support agent. Triage tickets and route them to specialists.",
        model=triage_model,
    )

    l2_specialist = create_handoff_agent(
        name="L2 Support Specialist",
        description="Deep investigation of escalated tickets",
        system_prompt="You are an L2 support specialist. Perform detailed analysis of escalations.",
        model=model,
    )

    l3_manager = create_handoff_agent(
        name="L3 Escalation Manager",
        description="Handles confirmed issues requiring senior authority",
        system_prompt="You are an L3 escalation manager. Direct the resolution of confirmed issues.",
        model=model,
    )

    print("Created agents:")
    print(f"  - {l1_agent.name} (id: {l1_agent.id})")
    print(f"  - {l2_specialist.name} (id: {l2_specialist.id})")
    print(f"  - {l3_manager.name} (id: {l3_manager.id})")

    l1_agent.can_delegate_to = [l2_specialist.id]
    l1_agent.can_escalate_to = [l3_manager.id]
    l2_specialist.can_escalate_to = [l3_manager.id]

    print("\nHandoff paths:")
    print("  L1 -> L2 (delegation)")
    print("  L1 -> L3 (escalation)")
    print("  L2 -> L3 (escalation)")

    smoke_ctx = HandoffContext(
        source_agent_id="user",
        target_agent_id=l1_agent.id,
        reason=HandoffReason.SPECIALIZATION,
        original_task="Smoke test the L1 front-line seat",
        conversation_summary="Need a one-line confirmation the agent is alive.",
        confidence=0.5,
        instructions="Reply 'L1 support online'.",
    )
    t0 = time.perf_counter()
    smoke_result = await l1_agent.receive_handoff(smoke_ctx)
    _banner("Part 1", time.perf_counter() - t0)
    print(f"  Smoke output: {(smoke_result.output or '')[:120]}")

    # =========================================================================
    # Part 2: anatomy of a HandoffContext
    # =========================================================================
    print("\n=== Part 2: Handoff Context ===\n")

    context = HandoffContext(
        source_agent_id=l1_agent.id,
        target_agent_id=l2_specialist.id,
        reason=HandoffReason.SPECIALIZATION,
        original_task="Investigate a duplicate subscription charge reported by a customer",
        conversation_summary="Customer was billed twice this month but only one plan is active.",
        findings={
            "customer_id": "CUST-48213",
            "charges_this_cycle": "2",
            "active_subscriptions": "1",
        },
        confidence=0.4,
        instructions="Determine whether the second charge was a genuine duplicate or a retry",
        handoff_chain=[l1_agent.id],
    )

    print("Handoff Context:")
    print(f"  From: {context.source_agent_id}")
    print(f"  To: {context.target_agent_id}")
    print(f"  Reason: {context.reason.value}")
    print(f"  Confidence: {context.confidence:.0%}")

    # ``to_prompt()`` turns the typed context into the prompt body the
    # target agent will receive — handy for inspection and tests.
    prompt = context.to_prompt()
    print("\nGenerated prompt for target agent:")
    print("-" * 40)
    print(prompt[:500] + "...")

    # =========================================================================
    # Part 3: the five handoff reasons
    # =========================================================================
    print("\n=== Part 3: Handoff Reasons ===\n")

    for reason in HandoffReason:
        print(f"  {reason.value}: {REASON_DESCRIPTIONS[reason]}")

    # =========================================================================
    # Part 4: the HandoffManager — registry + chain cap + history
    # =========================================================================
    print("\n=== Part 4: Handoff Manager ===\n")

    manager = create_handoff_manager(
        agents=[l1_agent, l2_specialist, l3_manager],
        max_chain=5,
    )

    print("Handoff Manager:")
    print(f"  Registered agents: {len(manager.agents)}")
    print(f"  Max chain length: {manager.max_handoff_chain}")

    state_smoke = AgentState(agent_id=l1_agent.id).with_message(
        Message.user("Customer says they were charged twice but only have one active plan.")
    )
    t0 = time.perf_counter()
    mgr_result = await manager.execute_handoff(
        source_agent=l1_agent,
        target_agent_id=l2_specialist.id,
        task="Investigate the duplicate subscription charge",
        reason=HandoffReason.SPECIALIZATION,
        state=state_smoke,
        findings={"related_tickets": 3},
    )
    _banner("Part 4", time.perf_counter() - t0)
    print(f"  Manager handoff output: {(mgr_result.output or '')[:160]}")

    # =========================================================================
    # Part 5: build a HandoffContext from real AgentState
    # =========================================================================
    print("\n=== Part 5: Creating Handoffs ===\n")

    state = AgentState(
        agent_id=l1_agent.id,
        tool_history=("lookup_account", "query_billing"),
    )
    state = state.with_message(Message.user("I was charged twice for my subscription this month"))
    state = state.with_message(
        Message.assistant("I'll triage the ticket and pull your billing history.")
    )

    handoff_context = await manager.create_handoff(
        source_agent=l1_agent,
        target_agent_id=l2_specialist.id,
        task="Investigate duplicate subscription charge for CUST-48213",
        reason=HandoffReason.SPECIALIZATION,
        state=state,
        findings={
            "initial_triage": "Two charges on the same plan; second has a different invoice id"
        },
        instructions="Determine whether a refund is warranted and which charge to reverse",
    )

    print("Created handoff:")
    print(f"  ID: {handoff_context.handoff_id}")
    print(f"  Chain: {' -> '.join(handoff_context.handoff_chain)}")
    print(f"  State snapshot: {handoff_context.state_snapshot}")

    # =========================================================================
    # Part 6: chain_handoff — walk L1 -> L2 -> L3
    # =========================================================================
    print("\n=== Part 6: Chain Handoffs ===\n")

    t0 = time.perf_counter()
    chain_results = await manager.chain_handoff(
        agent_chain=[l1_agent.id, l2_specialist.id, l3_manager.id],
        task="Enterprise customer threatening to churn over repeated billing errors — needs a goodwill credit",
        initial_state=state,
    )
    _banner("Part 6", time.perf_counter() - t0)

    print("Chain handoff completed:")
    for i, result in enumerate(chain_results):
        status = "OK" if result.success else f"FAILED: {result.error}"
        print(f"  Step {i + 1}: {result.source_agent_id} -> {result.target_agent_id}: {status}")

    # =========================================================================
    # Part 7: inspect the audit log
    # =========================================================================
    print("\n=== Part 7: Handoff History ===\n")

    print(f"Total handoffs in history: {len(manager.history)}")
    for ctx in manager.history[-3:]:
        print(f"  {ctx.handoff_id}: {ctx.source_agent_id} -> {ctx.target_agent_id}")
        print(f"    Reason: {ctx.reason.value}")
        print(f"    Created: {ctx.created_at.isoformat()}")

    print("\n" + "=" * 60)
    print("Next: Notebook 26 — incident adjudication debate — real incident vs noise")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
