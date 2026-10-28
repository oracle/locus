# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""
Notebook 24: Outage war room — mitigation, diagnostics, and comms agents in a swarm.

A swarm is a pool of agents that pull tasks from a shared queue based on
declared capabilities. No commander decides who does what — each task
finds the responder whose tags fit best, and the swarm runs them in
parallel where it can. Here the swarm *is* PILOT, the outage war room:
mitigation, diagnostics, and comms agents claiming work off the same board
during a production outage.

Contrast this with Notebook 22: there, STEWARD routes work centrally and
can be audited at the routing step. A swarm trades that for emergent
parallelism — useful, but worth noting that a self-organizing agent pool
is also where agentic failure modes concentrate: a mis-tagged or buggy
responder can claim work it should not, and one bad rollback can cascade
across services. Keep capability tags tight and the blackboard auditable.

Key ideas:
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

The default provider is OCI GenAI when ``~/.oci/config`` holds a usable profile, otherwise the bundled mock model; set ``LOCUS_MODEL_PROVIDER`` (``oci``, ``openai``, ``anthropic``, ``mock``) to choose. Set ``LOCUS_MODEL_PROVIDER=mock`` for offline runs.

Prerequisites:
- Notebook 02 (Agent basics).
- Notebook 22 (Orchestrator) if you want the supervised counterpoint.
"""

import asyncio

from config import get_model, print_config

from locus.multiagent.swarm import (
    SharedContext,
    Swarm,
    SwarmTask,
    create_swarm,
    create_swarm_agent,
)


# =============================================================================
# Part 1: declare the responders
# =============================================================================


async def example_create_agents():
    """Create three specialist incident responders."""
    print("=== Part 1: Creating Swarm Agents ===\n")

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


# =============================================================================
# Part 2: the blackboard
# =============================================================================


async def example_shared_context():
    """SharedContext — the blackboard responders use to leave notes for each other."""
    print("=== Part 2: Shared Context ===\n")

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

    swarm = Swarm(name="Outage War Room")

    swarm.add_task("Collect logs from the api-gateway pods", priority=5)
    swarm.add_task("Draft the status page update", priority=3)
    swarm.add_task("Review the proposed rollback plan", priority=2)
    swarm.add_task("Scale up the overloaded api-gateway service", priority=10)

    print("Task queue (sorted by priority):")
    for task in swarm.task_queue:
        print(f"  [{task.priority}] {task.description} (status: {task.status})")
    print()


# =============================================================================
# Part 4: capability matching
# =============================================================================


async def example_capability_matching():
    """How tasks are scored against agent capabilities."""
    print("=== Part 4: Capability-Based Assignment ===\n")

    diagnostics = create_swarm_agent(
        name="Diagnostics",
        capabilities=["diagnose", "analyze"],
    )

    comms = create_swarm_agent(
        name="Comms",
        capabilities=["notify", "document"],
    )

    # required_tags are a hard filter (set membership); preferred_tags
    # only boost the score. The last task is tagless on purpose.
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
# Part 5: run the war-room swarm end-to-end
# =============================================================================


async def example_full_swarm():
    """Build the war-room swarm and run it end-to-end against the configured model."""
    print("=== Part 5: Full Swarm with Model ===\n")

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

    print(f"Swarm '{swarm.name}' configured:")
    print(f"  Agents: {[a.name for a in swarm.agents]}")
    print()

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
# Part 6: three common swarm shapes
# =============================================================================


async def example_swarm_patterns():
    """Three common swarm shapes: specialist team, redundant team, pipeline."""
    print("=== Part 6: Swarm Patterns ===\n")

    # The shape is entirely in the capability tags: disjoint tags give each
    # task one expert, identical tags fan a backlog out, and a chain of
    # tags turns the board into a pipeline.
    shapes = [
        (
            "Distinct, non-overlapping capabilities; each task goes to its expert.",
            create_swarm(
                name="Specialist Team",
                agents=[
                    create_swarm_agent("Mitigation", ["rollback", "scale", "failover"]),
                    create_swarm_agent("Diagnostics", ["diagnose", "metrics", "logs"]),
                    create_swarm_agent("Comms", ["notify", "report", "document"]),
                ],
            ),
        ),
        (
            "Overlapping capabilities; alert backlogs fan out for parallel triage.",
            create_swarm(
                name="Redundant Team",
                agents=[
                    create_swarm_agent("Triage SRE A", ["triage", "investigate"]),
                    create_swarm_agent("Triage SRE B", ["triage", "investigate"]),
                    create_swarm_agent("Triage SRE C", ["triage", "investigate"]),
                ],
            ),
        ),
        (
            "Signals flow through a chain; output of one feeds the next.",
            create_swarm(
                name="Pipeline Team",
                agents=[
                    create_swarm_agent("Collector", ["collect", "acquire", "fetch"]),
                    create_swarm_agent("Examiner", ["examine", "correlate", "enrich"]),
                    create_swarm_agent("Reporter", ["report", "summarize", "present"]),
                ],
            ),
        ),
    ]

    for note, team in shapes:
        print(f"{team.name}")
        print("-" * 40)
        for agent in team.agents:
            print(f"  {agent.name}: {agent.capabilities}")
        print(f"  {note}")
        print()


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
    await example_full_swarm()
    await example_swarm_patterns()

    print("=" * 60)
    print("Next: Notebook 25 — Support tier escalation — L1 → L2 → L3 handoff with typed context")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
