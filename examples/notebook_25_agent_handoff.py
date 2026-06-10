# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 25: SOC tier escalation — L1 → L2 → L3 handoff with typed context.

A handoff is one analyst agent saying "I'm done, please take this further."
The source agent packages the alert, its findings, and an explicit reason
into a typed ``HandoffContext`` so the next tier inherits the full
investigation state — not just a string. No re-triage, no lost evidence.

The running case is a classic-SOC tier escalation: an EDR alert for
encoded PowerShell spawned by a document handler (ATT&CK T1059.001
PowerShell, T1566 Phishing as the suspected initial access) walks L1 → L2
→ L3. The typed chain is also the audit trail — every transfer, reason,
and confidence value is recorded for the post-incident review.

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

The default provider is the bundled mock model. Set TULIP_MODEL_PROVIDER=openai
(or anthropic) and the matching credentials to use a live model. Set
``TULIP_MODEL_PROVIDER=mock`` for offline runs.

This notebook fires ~9 handoffs serially, so it can use Tulip's "Model B"
slot — a second, typically cheaper model id read from ``TULIP_MODEL_ID_B``
— for the L1 triage seat. With Model B unset, the slot collapses to Model A.

Prerequisites:
- Notebook 06 (Agent basics).
- Notebook 24 (Swarm) for the unsupervised counterpoint.
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
    print("Notebook 25: SOC tier escalation — typed L1 → L2 → L3 handoffs")
    print("=" * 60)
    print()
    print_config()

    # =========================================================================
    # Part 1: build the SOC tiers and wire allowed escalation paths
    # =========================================================================
    print("\n=== Part 1: Creating Handoff Agents ===\n")

    l1_analyst = create_handoff_agent(
        name="L1 Triage Analyst",
        description="First-line alert triage and routing",
        system_prompt="You are an L1 SOC analyst. Triage alerts and route them to specialists.",
    )

    l2_investigator = create_handoff_agent(
        name="L2 Investigator",
        description="Deep investigation of escalated alerts",
        system_prompt="You are an L2 SOC investigator. Perform detailed analysis of escalations.",
    )

    l3_commander = create_handoff_agent(
        name="L3 Incident Commander",
        description="Handles confirmed incidents requiring senior response",
        system_prompt="You are an L3 incident commander. Direct the response to confirmed incidents.",
    )

    print("Created agents:")
    print(f"  - {l1_analyst.name} (id: {l1_analyst.id})")
    print(f"  - {l2_investigator.name} (id: {l2_investigator.id})")
    print(f"  - {l3_commander.name} (id: {l3_commander.id})")

    l1_analyst.can_delegate_to = [l2_investigator.id]
    l1_analyst.can_escalate_to = [l3_commander.id]
    l2_investigator.can_escalate_to = [l3_commander.id]

    print("\nHandoff paths:")
    print("  L1 -> L2 (delegation)")
    print("  L1 -> L3 (escalation)")
    print("  L2 -> L3 (escalation)")

    # The L1 triage seat reads from Tulip's "Model B" slot
    # (env: TULIP_MODEL_ID_B). Set a cheaper/faster model there to cut
    # round-trip latency; falls back to Model A when unset.
    triage_model = get_model_b(max_tokens=2000)
    model = get_model(max_tokens=2000)
    l1_with_model = l1_analyst.with_model(triage_model)
    smoke_ctx = HandoffContext(
        source_agent_id="user",
        target_agent_id=l1_analyst.id,
        reason=HandoffReason.SPECIALIZATION,
        original_task="Smoke test the L1 triage seat",
        conversation_summary="Need a one-line confirmation the agent is alive.",
        confidence=0.5,
        instructions="Reply 'L1 triage online'.",
    )
    t0 = time.perf_counter()
    smoke_result = await l1_with_model.receive_handoff(smoke_ctx)
    _banner("Part 1", time.perf_counter() - t0)
    print(f"  Smoke output: {(smoke_result.output or '')[:120]}")

    # =========================================================================
    # Part 2: anatomy of a HandoffContext
    # =========================================================================
    print("\n=== Part 2: Handoff Context ===\n")

    context = HandoffContext(
        source_agent_id=l1_analyst.id,
        target_agent_id=l2_investigator.id,
        reason=HandoffReason.SPECIALIZATION,
        original_task="Investigate a suspected phishing email reported by finance",
        conversation_summary="User reported a credential-prompt email. Sender domain is 3 days old.",
        findings={
            "sender_domain": "phish.example.net",
            "domain_age_days": "3",
            "url_reputation": "uncategorized",
        },
        confidence=0.4,
        instructions="Determine whether any credentials were actually submitted",
        handoff_chain=[l1_analyst.id],
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
            HandoffReason.SPECIALIZATION: "Target has better capabilities for this alert",
            HandoffReason.ESCALATION: "Incident needs a higher tier or more authority",
            HandoffReason.DELEGATION: "Sub-task delegated to another analyst agent",
            HandoffReason.COMPLETION: "Work completed, returning to parent",
            HandoffReason.FAILURE: "Agent failed, trying another approach",
        }
        print(f"  {reason.value}: {descriptions[reason]}")

    # =========================================================================
    # Part 4: the HandoffManager — registry + chain cap + history
    # =========================================================================
    print("\n=== Part 4: Handoff Manager ===\n")

    manager = create_handoff_manager(
        agents=[l1_analyst, l2_investigator, l3_commander],
        max_chain=5,
    )

    print("Handoff Manager:")
    print(f"  Registered agents: {len(manager.agents)}")
    print(f"  Max chain length: {manager.max_handoff_chain}")

    for agent_id in list(manager.agents):
        manager.agents[agent_id] = manager.agents[agent_id].with_model(model)
    state_smoke = AgentState(agent_id=l1_analyst.id).with_message(
        Message.user("EDR flagged encoded PowerShell on WS-0142; user also reported phishing.")
    )
    t0 = time.perf_counter()
    mgr_result = await manager.execute_handoff(
        source_agent=l1_analyst,
        target_agent_id=l2_investigator.id,
        task="Investigate the suspicious PowerShell execution",
        reason=HandoffReason.SPECIALIZATION,
        state=state_smoke,
        findings={"related_alerts": 3},
    )
    _banner("Part 4", time.perf_counter() - t0)
    print(f"  Manager handoff output: {(mgr_result.output or '')[:160]}")

    # =========================================================================
    # Part 5: build a HandoffContext from real AgentState
    # =========================================================================
    print("\n=== Part 5: Creating Handoffs ===\n")

    state = AgentState(
        agent_id=l1_analyst.id,
        tool_history=("lookup_hash", "query_siem"),
    )
    state = state.with_message(Message.user("EDR alert fired on WS-0142"))
    state = state.with_message(Message.assistant("I'll triage the alert and gather context."))

    handoff_context = await manager.create_handoff(
        source_agent=l1_analyst,
        target_agent_id=l2_investigator.id,
        task="Investigate suspicious PowerShell execution on WS-0142",
        reason=HandoffReason.SPECIALIZATION,
        state=state,
        findings={"initial_triage": "Encoded command line; parent process is winword.exe"},
        instructions="Determine whether the payload executed and what it touched",
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
    # Part 7: chain_handoff — walk L1 -> L2 -> L3
    # =========================================================================
    print("\n=== Part 7: Chain Handoffs ===\n")

    manager.agents[l1_analyst.id] = l1_analyst.with_model(triage_model)
    manager.agents[l3_commander.id] = l3_commander.with_model(model)

    chain_results = await manager.chain_handoff(
        agent_chain=[l1_analyst.id, l2_investigator.id, l3_commander.id],
        task="Suspected ransomware on file server FS-03 — mass file renames in progress",
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

    print("Pattern 1: L1 Triage -> Specialist")
    print("  A generalist analyst assesses alerts and routes to domain experts")
    print()

    print("Pattern 2: Hierarchical Escalation")
    print("  L1 -> L2 -> L3 SOC tier escalation chain")
    print()

    print("Pattern 3: Parallel Specialists")
    print("  Forensics and threat-intel analyze in parallel, results aggregated")
    print()

    print("Pattern 4: Return with Findings")
    print("  Specialist completes work and returns to the incident commander")
    print()

    print("Pattern 5: Failover")
    print("  If one analyst agent fails, handoff to a backup agent")

    # =========================================================================
    # Part 10: things to keep in mind
    # =========================================================================
    print("\n=== Part 10: Best Practices ===\n")

    print("1. Keep handoff contexts focused — transfer only case-relevant evidence.")
    print("2. Set max_chain to prevent infinite escalation loops.")
    print("3. Give the next tier explicit instructions, not just the alert.")
    print("4. Track confidence through the chain so you can audit decay.")
    print("5. Pick the right HandoffReason — it drives prompt templating.")
    print("6. Preserve key findings — don't drop evidence mid-chain.")
    print("7. manager.history is your audit trail for the post-incident review.")

    # =========================================================================
    print("\n" + "=" * 60)
    print("Next: Notebook 26 — Orchestrator Pattern")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
