# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 24: Outage war room — mitigation, diagnostics, and comms agents in a swarm.

A swarm is a pool of agents that pull tasks from a shared queue based on
declared capabilities. No commander decides who does what — each task
finds the responder whose tags fit best, and the swarm runs them in
parallel where it can. Here the swarm is the response side of the PILOT
incident-response function (the named on-call commander built in
Notebook 26): mitigation, diagnostics, and comms agents claiming work off
the same board during a production outage.

Contrast this with Notebook 26: there, PILOT routes work centrally and
can be audited at the routing step. A swarm trades that for emergent
parallelism — useful, but worth noting that a self-organizing agent pool
is also where agentic failure modes concentrate: a mis-tagged or buggy
responder can claim work it should not, and one bad rollback can cascade
across services. Keep capability tags tight and the blackboard auditable.

- Each agent advertises ``capabilities`` (free-form string tags).
- Tasks carry ``required_tags`` (hard filter) and ``preferred_tags``
  (priority boost). Tagless tasks fall back to substring matching against
  the description so older swarms keep working.
- A ``SharedContext`` (findings, blackboard messages, task results) lets
  responders leave notes for each other without a central coordinator.
- ``Swarm.execute`` decomposes the initial incident into capability-matched
  subtasks and runs them.

Run it:
    .venv/bin/python examples/notebook_24_swarm_multiagent.py

The default provider is the bundled mock model. Set TULIP_MODEL_PROVIDER=openai
(or anthropic) and the matching credentials so the swarm talks to a live
model. Set ``TULIP_MODEL_PROVIDER=mock`` for offline runs.

Prerequisites:
- Notebook 06 (Agent basics).
- Notebook 26 (Orchestrator) if you want the supervised counterpoint.
"""

import asyncio
import time

from config import get_model, print_config

from tulip.agent import Agent
from tulip.multiagent.swarm import (
    SharedContext,
    Swarm,
    SwarmTask,
    create_swarm,
    create_swarm_agent,
)


async def _llm_call(
    prompt: str, *, system: str = "Reply in one short sentence.", max_tokens: int = 80
) -> str:
    """One model call with a timing/token banner so each Part shows its work."""
    agent = Agent(model=get_model(max_tokens=max_tokens), system_prompt=system)
    t0 = time.perf_counter()
    res = await agent.arun(prompt)
    dt = time.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · {res.metrics.prompt_tokens}→{res.metrics.completion_tokens} tokens]"
    )
    return res.message.strip()


# =============================================================================
# Part 1: declare the responders
# =============================================================================


async def example_create_agents():
    """Create three specialist incident responders."""
    print("=== Part 1: Creating Swarm Agents ===\n")
    _ai_note = await _llm_call("In one sentence, when is a swarm of small specialised responders a better fit than one generalist SRE agent?")
    print(f"AI rationale: {_ai_note}")

    mitigation = create_swarm_agent(
        name="Mitigation",
        capabilities=["rollback", "scale", "failover"],
        system_prompt="You are a mitigation specialist. Roll back bad deploys and scale services.",
    )

    diagnostics = create_swarm_agent(
        name="Diagnostics",
        capabilities=["diagnose", "analyze", "investigate"],
        system_prompt="You are a diagnostics specialist. Collect and analyze logs and metrics.",
    )

    comms = create_swarm_agent(
        name="Comms",
        capabilities=["notify", "summarize", "document"],
        system_prompt="You are a comms specialist. Keep stakeholders accurately informed.",
    )

    print("Created agents:")
    for agent in [mitigation, diagnostics, comms]:
        print(f"  - {agent.name}: capabilities = {agent.capabilities}")
    print()

    return mitigation, diagnostics, comms


# =============================================================================
# Part 2: the blackboard
# =============================================================================


async def example_shared_context():
    """SharedContext — the blackboard responders use to leave notes for each other."""
    print("=== Part 2: Shared Context ===\n")
    _ai_note = await _llm_call("In one sentence, why does an outage war room need a SharedContext for findings and blackboard notes?")
    print(f"AI rationale: {_ai_note}")

    context = SharedContext()

    await context.add_finding(
        key="initial_symptom",
        value="api-gateway pods OOMKilled and crash-looping every 60 seconds",
        agent_id="diagnostics_1",
    )

    await context.post_to_blackboard(
        key="need_review",
        message="Need someone to review the rollback plan before we revert the deploy",
        agent_id="mitigation_1",
    )

    await context.record_task_result(
        task_id="task_001",
        result="Captured logs and heap dump from api-gateway",
    )

    print("Current context:")
    print(context.get_summary())
    print()


# =============================================================================
# Part 3: priority queue
# =============================================================================


async def example_task_queue():
    """Tasks are pulled in priority order — highest first."""
    print("=== Part 3: Task Queue ===\n")
    _ai_note = await _llm_call("In one sentence, why is priority-queue task routing useful during a production incident?")
    print(f"AI rationale: {_ai_note}")

    swarm = Swarm(name="Outage War Room")

    task1 = swarm.add_task("Collect logs from the api-gateway pods", priority=5)
    task2 = swarm.add_task("Draft the status page update", priority=3)
    task3 = swarm.add_task("Review the proposed rollback plan", priority=2)
    task4 = swarm.add_task("Scale up the overloaded api-gateway service", priority=10)

    print("Task queue (sorted by priority):")
    for task in swarm.task_queue:
        print(f"  [{task.priority}] {task.description} (status: {task.status})")
    print()

    return swarm


# =============================================================================
# Part 4: capability matching
# =============================================================================


async def example_capability_matching():
    """How tasks are scored against agent capabilities."""
    print("=== Part 4: Capability-Based Assignment ===\n")
    _ai_note = await _llm_call("In one sentence, why is capability-based agent selection better than random round-robin in an on-call team?")
    print(f"AI rationale: {_ai_note}")

    diagnostics = create_swarm_agent(
        name="Diagnostics",
        capabilities=["diagnose", "analyze"],
    )

    comms = create_swarm_agent(
        name="Comms",
        capabilities=["notify", "document"],
    )

    # required_tags are a hard filter (set membership). preferred_tags
    # only boost the score. Tasks without any tags fall back to substring
    # matching against the description — so older tagless swarms still
    # route correctly.
    tasks = [
        SwarmTask(
            description="Analyze the elevated error rate on api-gateway",
            required_tags=["diagnose"],
            preferred_tags=["analyze"],
        ),
        SwarmTask(
            description="Notify affected teams and document the incident",
            required_tags=["notify", "document"],
        ),
        SwarmTask(description="Analyze the pod restart history", required_tags=["analyze"]),
        SwarmTask(description="Create a summary document for the postmortem"),
    ]

    print("Task-Agent matching:")
    for task in tasks:
        print(f"\n  Task: {task.description}")
        print(f"    required_tags={task.required_tags} preferred_tags={task.preferred_tags}")
        print(f"    Diagnostics can handle: {diagnostics.can_handle(task)}")
        print(f"    Comms can handle: {comms.can_handle(task)}")
        print(f"    Diagnostics priority: {diagnostics.priority_for_task(task):.2f}")
        print(f"    Comms priority: {comms.priority_for_task(task):.2f}")
    print()


# =============================================================================
# Part 5: a swarm without a model — just the shape
# =============================================================================


async def example_simple_swarm():
    """Stand up a swarm without a model — useful when you only want the shape."""
    print("=== Part 5: Simple Swarm Execution ===\n")
    rationale_prompt = (
        "In one sentence, what does 'simple swarm execution' mean "
        "and when is it enough for an incident drill?"
    )
    _ai_note = await _llm_call(rationale_prompt)
    print(f"AI rationale: {_ai_note}")

    swarm = Swarm(name="Drill Swarm")

    agent1 = create_swarm_agent(
        name="Mitigation Lead",
        capabilities=["rollback"],
        system_prompt="You roll back bad deploys.",
    )

    agent2 = create_swarm_agent(
        name="Scribe",
        capabilities=["report"],
        system_prompt="You write incident reports.",
    )

    swarm.add_agent(agent1)
    swarm.add_agent(agent2)

    swarm.add_task("Roll back the bad deploy on api-gateway", priority=5)
    swarm.add_task("Report on the mitigation status", priority=3)

    print(f"Swarm '{swarm.name}' configured:")
    print(f"  Agents: {[a.name for a in swarm.agents]}")
    print(f"  Tasks: {len(swarm.task_queue)}")
    print()

    print("Note: full execution requires a configured model. See Part 6.")
    print()


# =============================================================================
# Part 6: live run against a real model
# =============================================================================


async def example_full_swarm():
    """Run the war-room swarm end-to-end against a real model."""
    print("=== Part 6: Full Swarm with Model ===\n")

    # Each agent emits a structured `### Findings / ### Analysis /
    # ### Blackboard` block — leave enough completion budget for it.
    model = get_model(max_tokens=2000)

    swarm = create_swarm(
        name="PILOT Outage War Room",
        agents=[
            create_swarm_agent(
                name="Diagnostics",
                capabilities=["investigate", "analyze", "examine"],
                system_prompt="You are a diagnostics responder. Investigate services and collect logs and metrics.",
            ),
            create_swarm_agent(
                name="Mitigation",
                capabilities=["rollback", "scale", "failover"],
                system_prompt="You are a mitigation responder. Propose rollback and scaling steps.",
            ),
            create_swarm_agent(
                name="Comms",
                capabilities=["write", "summarize", "notify"],
                system_prompt="You write clear, concise incident status updates.",
            ),
        ],
        model=model,
    )

    print("Executing swarm on: 'Production outage on the api-gateway service'")
    print("This may take a moment...\n")

    result = await swarm.execute(
        initial_task=(
            "Investigate the api-gateway outage, mitigate the affected service, and "
            "write a brief status update for stakeholders."
        ),
        decompose_tasks=True,
    )

    print("Swarm completed!")
    print(f"  Success: {result.success}")
    print(f"  Completed tasks: {len(result.completed_tasks)}")
    print(f"  Failed tasks: {len(result.failed_tasks)}")
    print(f"  Duration: {result.duration_ms:.0f}ms")

    if result.completed_tasks:
        print("\nCompleted subtasks:")
        for t in result.completed_tasks[:5]:
            assigned = t.claimed_by or "unassigned"
            print(f"  - [{assigned}] {t.description[:80]}")
            preview = (t.result or "<empty>").strip().splitlines()[0:6]
            for line in preview:
                print(f"      {line[:120]}")
            if not t.result:
                print("      (no .result text — model returned empty)")
    if result.failed_tasks:
        print("\nFailed subtasks:")
        for t in result.failed_tasks[:5]:
            print(f"  - {t.description[:80]} (reason: {t.error or 'no agent matched'})")
    if result.summary:
        print(f"\nSummary:\n{result.summary[:500]}...")
    print()


# =============================================================================
# Part 7: three common swarm shapes
# =============================================================================


async def example_swarm_patterns():
    """Three common swarm shapes: specialist team, redundant team, pipeline."""
    print("=== Part 7: Swarm Patterns ===\n")
    _ai_note = await _llm_call("In one sentence, when is a Specialist Team swarm preferable to a Pipeline swarm for incident response?")
    print(f"AI rationale: {_ai_note}")

    print("Pattern 1: Specialist Team")
    print("-" * 40)
    specialist_team = create_swarm(
        name="Specialist Team",
        agents=[
            create_swarm_agent("Mitigation", ["rollback", "scale", "failover"]),
            create_swarm_agent("Diagnostics", ["diagnose", "metrics", "logs"]),
            create_swarm_agent("Comms", ["notify", "report", "document"]),
        ],
    )
    print("  Distinct, non-overlapping capabilities; each task goes to its expert.")
    print()

    print("Pattern 2: Redundant Team")
    print("-" * 40)
    redundant_team = create_swarm(
        name="Redundant Team",
        agents=[
            create_swarm_agent("Triage SRE A", ["triage", "investigate"]),
            create_swarm_agent("Triage SRE B", ["triage", "investigate"]),
            create_swarm_agent("Triage SRE C", ["triage", "investigate"]),
        ],
    )
    print("  Overlapping capabilities; alert backlogs fan out for parallel triage.")
    print()

    print("Pattern 3: Pipeline Team")
    print("-" * 40)
    pipeline_team = create_swarm(
        name="Pipeline Team",
        agents=[
            create_swarm_agent("Collector", ["collect", "acquire", "fetch"]),
            create_swarm_agent("Examiner", ["examine", "correlate", "enrich"]),
            create_swarm_agent("Reporter", ["report", "summarize", "present"]),
        ],
    )
    print("  Signals flow through a chain; output of one feeds the next.")
    print()


# =============================================================================
# Main
# =============================================================================


async def main():
    print("=" * 60)
    print("Notebook 24: Outage war room — capability-matched responders, shared queue")
    print("=" * 60)
    print()

    print_config()
    print()

    await example_create_agents()
    await example_shared_context()
    await example_task_queue()
    await example_capability_matching()
    await example_simple_swarm()
    await example_full_swarm()
    await example_swarm_patterns()

    print("=" * 60)
    print("Next: Notebook 25 — Agent Handoff")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
