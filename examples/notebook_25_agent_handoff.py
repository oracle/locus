# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL
"""
Notebook 26: agent-to-agent handoff with a structured context payload.

A handoff is one agent saying "I'm done, please take this further." The
source agent packages the task, its findings, and an explicit reason
into a typed ``HandoffContext`` so the target inherits the full work
state — not just a string.

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
  with each agent inheriting the previous one's findings.

Run it:
    .venv/bin/python examples/notebook_31_agent_handoff.py

The default provider is the bundled mock model. Set TULIP_MODEL_PROVIDER=openai
(or anthropic) and the matching credentials to use a live model. Set
``TULIP_MODEL_PROVIDER=mock`` for offline runs.

This notebook fires ~9 handoffs serially, so it can use Tulip's "Model B"
slot — a second, typically cheaper model id read from ``TULIP_MODEL_ID_B``
— for the triage seat. With Model B unset, the slot collapses to Model A.

Prerequisites:
- Notebook 08 (Agent basics).
- Notebook 25 (Swarm) for the unsupervised counterpoint.
"""

import asyncio
import time

from config import get_model, get_model_b, print_config

from tulip.core.messages import Message
from tulip.core.state import AgentState
from tulip.multiagent.handoff import (
    HandoffContext,
    HandoffReason,
    create_handoff_agent,
    create_handoff_manager,
)


def _banner(label: str, dt: float, prompt_tok: int = 0, completion_tok: int = 0) -> None:
    """Print a uniform [model call …] banner so each Part shows it hit the model."""
    print(f"  [model call · {label}: {dt:.2f}s · {prompt_tok}→{completion_tok} tokens]")


async def main():
    print("=" * 60)
    print("Notebook 26: agent-to-agent handoff with structured context")
    print("=" * 60)
    print()
    print_config()

    # =========================================================================
    # Part 1: build the agent pool and wire allowed handoff paths
    # =========================================================================
    print("\n=== Part 1: Creating Handoff Agents ===\n")

    triage_agent = create_handoff_agent(
        name="Triage Agent",
        description="Initial assessment and routing of issues",
        system_prompt="You are a triage agent. Assess issues and route to specialists.",
    )

    technical_agent = create_handoff_agent(
        name="Technical Specialist",
        description="Deep technical analysis and debugging",
        system_prompt="You are a technical specialist. Perform detailed analysis.",
    )

    escalation_agent = create_handoff_agent(
        name="Escalation Manager",
        description="Handles critical issues requiring senior attention",
        system_prompt="You are an escalation manager. Handle critical issues.",
    )

    print("Created agents:")
    print(f"  - {triage_agent.name} (id: {triage_agent.id})")
    print(f"  - {technical_agent.name} (id: {technical_agent.id})")
    print(f"  - {escalation_agent.name} (id: {escalation_agent.id})")

    triage_agent.can_delegate_to = [technical_agent.id]
    triage_agent.can_escalate_to = [escalation_agent.id]
    technical_agent.can_escalate_to = [escalation_agent.id]

    print("\nHandoff paths:")
    print("  Triage -> Technical (delegation)")
    print("  Triage -> Escalation (escalation)")
    print("  Technical -> Escalation (escalation)")

    # The triage seat reads from Tulip's "Model B" slot
    # (env: TULIP_MODEL_ID_B). Set a cheaper/faster model there to cut
    # round-trip latency; falls back to Model A when unset.
    triage_model = get_model_b(max_tokens=2000)
    model = get_model(max_tokens=2000)
    triage_with_model = triage_agent.with_model(triage_model)
    smoke_ctx = HandoffContext(
        source_agent_id="user",
        target_agent_id=triage_agent.id,
        reason=HandoffReason.SPECIALIZATION,
        original_task="Smoke test the triage agent",
        conversation_summary="Need a one-line confirmation the agent is alive.",
        confidence=0.5,
        instructions="Reply 'triage agent online'.",
    )
    t0 = time.perf_counter()
    smoke_result = await triage_with_model.receive_handoff(smoke_ctx)
    _banner("Part 1", time.perf_counter() - t0)
    print(f"  Smoke output: {(smoke_result.output or '')[:120]}")

    # =========================================================================
    # Part 2: anatomy of a HandoffContext
    # =========================================================================
    print("\n=== Part 2: Handoff Context ===\n")

    context = HandoffContext(
        source_agent_id=triage_agent.id,
        target_agent_id=technical_agent.id,
        reason=HandoffReason.SPECIALIZATION,
        original_task="Investigate slow API response times",
        conversation_summary="User reported 5s response times. Initial check shows normal CPU.",
        findings={
            "api_latency_p99": "5200ms",
            "cpu_usage": "45%",
            "memory_usage": "62%",
        },
        confidence=0.4,
        instructions="Focus on database query performance",
        handoff_chain=[triage_agent.id],
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
        descriptions = {
            HandoffReason.SPECIALIZATION: "Target has better capabilities for this task",
            HandoffReason.ESCALATION: "Issue needs higher authority or expertise",
            HandoffReason.DELEGATION: "Sub-task delegation to another agent",
            HandoffReason.COMPLETION: "Task completed, returning to parent",
            HandoffReason.FAILURE: "Agent failed, trying another approach",
        }
        print(f"  {reason.value}: {descriptions[reason]}")

    # =========================================================================
    # Part 4: the HandoffManager — registry + chain cap + history
    # =========================================================================
    print("\n=== Part 4: Handoff Manager ===\n")

    manager = create_handoff_manager(
        agents=[triage_agent, technical_agent, escalation_agent],
        max_chain=5,
    )

    print("Handoff Manager:")
    print(f"  Registered agents: {len(manager.agents)}")
    print(f"  Max chain length: {manager.max_handoff_chain}")

    for agent_id in list(manager.agents):
        manager.agents[agent_id] = manager.agents[agent_id].with_model(model)
    state_smoke = AgentState(agent_id=triage_agent.id).with_message(
        Message.user("DB latency spiked to 5s, cpu normal.")
    )
    t0 = time.perf_counter()
    mgr_result = await manager.execute_handoff(
        source_agent=triage_agent,
        target_agent_id=technical_agent.id,
        task="Diagnose the latency spike",
        reason=HandoffReason.SPECIALIZATION,
        state=state_smoke,
        findings={"p99_ms": 5000},
    )
    _banner("Part 4", time.perf_counter() - t0)
    print(f"  Manager handoff output: {(mgr_result.output or '')[:160]}")

    # =========================================================================
    # Part 5: build a HandoffContext from real AgentState
    # =========================================================================
    print("\n=== Part 5: Creating Handoffs ===\n")

    state = AgentState(
        agent_id=triage_agent.id,
        tool_history=("check_metrics", "query_logs"),
    )
    state = state.with_message(Message.user("API is slow"))
    state = state.with_message(Message.assistant("I'll investigate the API performance."))

    handoff_context = await manager.create_handoff(
        source_agent=triage_agent,
        target_agent_id=technical_agent.id,
        task="Investigate slow API response times",
        reason=HandoffReason.SPECIALIZATION,
        state=state,
        findings={"initial_metrics": "Normal CPU, high DB latency"},
        instructions="Focus on database performance",
    )

    print("Created handoff:")
    print(f"  ID: {handoff_context.handoff_id}")
    print(f"  Chain: {' -> '.join(handoff_context.handoff_chain)}")
    print(f"  State snapshot: {handoff_context.state_snapshot}")

    # =========================================================================
    # Part 6: where execute_handoff fits
    # =========================================================================
    print("\n=== Part 6: Executing Handoffs ===\n")
    print("`manager.execute_handoff(...)` was exercised in Part 4. The same")
    print("call shape works for any (source -> target, reason) pair — see")
    print("the chain demo in Part 7 for back-to-back execution.")

    # =========================================================================
    # Part 7: chain_handoff — walk Triage -> Technical -> Escalation
    # =========================================================================
    print("\n=== Part 7: Chain Handoffs ===\n")

    manager.agents[triage_agent.id] = triage_agent.with_model(triage_model)
    manager.agents[escalation_agent.id] = escalation_agent.with_model(model)

    chain_results = await manager.chain_handoff(
        agent_chain=[triage_agent.id, technical_agent.id, escalation_agent.id],
        task="Critical production outage affecting all users",
        initial_state=state,
    )

    print("Chain handoff completed:")
    for i, result in enumerate(chain_results):
        status = "OK" if result.success else f"FAILED: {result.error}"
        print(f"  Step {i + 1}: {result.source_agent_id} -> {result.target_agent_id}: {status}")

    # =========================================================================
    # Part 8: inspect the audit log
    # =========================================================================
    print("\n=== Part 8: Handoff History ===\n")

    print(f"Total handoffs in history: {len(manager.history)}")
    for ctx in manager.history[-3:]:
        print(f"  {ctx.handoff_id}: {ctx.source_agent_id} -> {ctx.target_agent_id}")
        print(f"    Reason: {ctx.reason.value}")
        print(f"    Created: {ctx.created_at.isoformat()}")

    # =========================================================================
    # Part 9: common handoff shapes
    # =========================================================================
    print("\n=== Part 9: Common Handoff Patterns ===\n")

    print("Pattern 1: Triage -> Specialist")
    print("  A generalist agent assesses and routes to domain experts")
    print()

    print("Pattern 2: Hierarchical Escalation")
    print("  L1 -> L2 -> L3 support escalation chain")
    print()

    print("Pattern 3: Parallel Specialists")
    print("  Multiple specialists analyze in parallel, results aggregated")
    print()

    print("Pattern 4: Return with Findings")
    print("  Specialist completes work and returns to coordinator")
    print()

    print("Pattern 5: Failover")
    print("  If one agent fails, handoff to backup agent")

    # =========================================================================
    # Part 10: things to keep in mind
    # =========================================================================
    print("\n=== Part 10: Best Practices ===\n")

    print("1. Keep handoff contexts focused — transfer only relevant info.")
    print("2. Set max_chain to prevent infinite loops.")
    print("3. Give the target agent explicit instructions, not just the task.")
    print("4. Track confidence through the chain so you can audit decay.")
    print("5. Pick the right HandoffReason — it drives prompt templating.")
    print("6. Preserve key findings — don't drop them mid-chain.")
    print("7. Watch manager.history during debugging.")

    # =========================================================================
    print("\n" + "=" * 60)
    print("Next: Notebook 27 — Orchestrator Pattern")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
