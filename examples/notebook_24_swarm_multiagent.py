# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 24: IR war room — containment, forensics, and comms agents in a swarm.

A swarm is a pool of agents that pull tasks from a shared queue based on
declared capabilities. No commander decides who does what — each task
finds the responder whose tags fit best, and the swarm runs them in
parallel where it can. Here the swarm is the response side of the MARSHAL
incident-response function (the named IR commander built in Notebook 26):
containment, forensics, and comms agents claiming work off the same board.

Contrast this with Notebook 26: there, MARSHAL routes work centrally and
can be audited at the routing step. A swarm trades that for emergent
parallelism — useful, but worth noting that a self-organizing agent pool
is also where agentic failure modes concentrate (OWASP ASI08 Cascading
Failures, ASI10 Rogue Agents): a mis-tagged or compromised responder can
claim work it should not. Keep capability tags tight and the blackboard
auditable.

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


def _llm_call(
    prompt: str, *, system: str = "Reply in one short sentence.", max_tokens: int = 80
) -> str:
    """One model call with a timing/token banner so each Part shows its work."""
    agent = Agent(model=get_model(max_tokens=max_tokens), system_prompt=system)
    t0 = time.perf_counter()
    res = agent.run_sync(prompt)
    dt = time.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · {res.metrics.prompt_tokens}→{res.metrics.completion_tokens} tokens]"
    )
    return res.message.strip()


# =============================================================================
# Part 1: declare the responders
# =============================================================================


def example_create_agents():
    """Create three specialist incident responders."""
    print("=== Part 1: Creating Swarm Agents ===\n")
    print(
        f"AI rationale: {_llm_call('In one sentence, when is a swarm of small specialised responders a better fit than one generalist security agent?')}"
    )

    containment = create_swarm_agent(
        name="Containment",
        capabilities=["contain", "isolate", "block"],
        system_prompt="You are a containment specialist. Isolate hosts and block indicators.",
    )

    forensics = create_swarm_agent(
        name="Forensics",
        capabilities=["forensics", "analyze", "investigate"],
        system_prompt="You are a forensics specialist. Collect and analyze evidence.",
    )

    comms = create_swarm_agent(
        name="Comms",
        capabilities=["notify", "summarize", "document"],
        system_prompt="You are a comms specialist. Keep stakeholders accurately informed.",
    )

    print("Created agents:")
    for agent in [containment, forensics, comms]:
        print(f"  - {agent.name}: capabilities = {agent.capabilities}")
    print()

    return containment, forensics, comms


# =============================================================================
# Part 2: the blackboard
# =============================================================================


async def example_shared_context():
    """SharedContext — the blackboard responders use to leave notes for each other."""
    print("=== Part 2: Shared Context ===\n")
    print(
        f"AI rationale: {_llm_call('In one sentence, why does an IR war room need a SharedContext for findings and blackboard notes?')}"
    )

    context = SharedContext()

    await context.add_finding(
        key="initial_ioc",
        value="WS-0142 beaconing to evil.example every 60 seconds",
        agent_id="forensics_1",
    )

    await context.post_to_blackboard(
        key="need_review",
        message="Need someone to review the firewall block list before we push it",
        agent_id="containment_1",
    )

    await context.record_task_result(
        task_id="task_001",
        result="Completed memory acquisition on WS-0142",
    )

    print("Current context:")
    print(context.get_summary())
    print()


# =============================================================================
# Part 3: priority queue
# =============================================================================


def example_task_queue():
    """Tasks are pulled in priority order — highest first."""
    print("=== Part 3: Task Queue ===\n")
    print(
        f"AI rationale: {_llm_call('In one sentence, why is priority-queue task routing useful during incident response?')}"
    )

    swarm = Swarm(name="IR War Room")

    task1 = swarm.add_task("Collect a memory image from WS-0142", priority=5)
    task2 = swarm.add_task("Draft the stakeholder status update", priority=3)
    task3 = swarm.add_task("Review the proposed firewall block list", priority=2)
    task4 = swarm.add_task("Isolate the compromised workstation", priority=10)

    print("Task queue (sorted by priority):")
    for task in swarm.task_queue:
        print(f"  [{task.priority}] {task.description} (status: {task.status})")
    print()

    return swarm


# =============================================================================
# Part 4: capability matching
# =============================================================================


def example_capability_matching():
    """How tasks are scored against agent capabilities."""
    print("=== Part 4: Capability-Based Assignment ===\n")
    print(
        f"AI rationale: {_llm_call('In one sentence, why is capability-based agent selection better than random round-robin in an IR team?')}"
    )

    forensics = create_swarm_agent(
        name="Forensics",
        capabilities=["forensics", "analyze"],
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
            description="Analyze the suspicious login activity on WS-0142",
            required_tags=["forensics"],
            preferred_tags=["analyze"],
        ),
        SwarmTask(
            description="Notify affected teams and document the incident",
            required_tags=["notify", "document"],
        ),
        SwarmTask(description="Analyze the EDR process tree", required_tags=["analyze"]),
        SwarmTask(description="Create a summary document for the postmortem"),
    ]

    print("Task-Agent matching:")
    for task in tasks:
        print(f"\n  Task: {task.description}")
        print(f"    required_tags={task.required_tags} preferred_tags={task.preferred_tags}")
        print(f"    Forensics can handle: {forensics.can_handle(task)}")
        print(f"    Comms can handle: {comms.can_handle(task)}")
        print(f"    Forensics priority: {forensics.priority_for_task(task):.2f}")
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
        "and when is it enough for an IR drill?"
    )
    print(f"AI rationale: {_llm_call(rationale_prompt)}")

    swarm = Swarm(name="Drill Swarm")

    agent1 = create_swarm_agent(
        name="Containment Lead",
        capabilities=["isolate"],
        system_prompt="You isolate compromised hosts.",
    )

    agent2 = create_swarm_agent(
        name="Scribe",
        capabilities=["report"],
        system_prompt="You write incident reports.",
    )

    swarm.add_agent(agent1)
    swarm.add_agent(agent2)

    swarm.add_task("Isolate the compromised workstation WS-0142", priority=5)
    swarm.add_task("Report on the containment status", priority=3)

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
        name="MARSHAL IR War Room",
        agents=[
            create_swarm_agent(
                name="Forensics",
                capabilities=["investigate", "analyze", "examine"],
                system_prompt="You are a forensics responder. Investigate hosts and collect evidence.",
            ),
            create_swarm_agent(
                name="Containment",
                capabilities=["contain", "isolate", "block"],
                system_prompt="You are a containment responder. Propose isolation and blocking steps.",
            ),
            create_swarm_agent(
                name="Comms",
                capabilities=["write", "summarize", "notify"],
                system_prompt="You write clear, concise incident status updates.",
            ),
        ],
        model=model,
    )

    print("Executing swarm on: 'Phishing compromise on workstation WS-0142'")
    print("This may take a moment...\n")

    result = await swarm.execute(
        initial_task=(
            "Investigate the phishing compromise on workstation WS-0142, contain "
            "the affected host, and write a brief status update for stakeholders."
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


def example_swarm_patterns():
    """Three common swarm shapes: specialist team, redundant team, pipeline."""
    print("=== Part 7: Swarm Patterns ===\n")
    print(
        f"AI rationale: {_llm_call('In one sentence, when is a Specialist Team swarm preferable to a Pipeline swarm for incident response?')}"
    )

    print("Pattern 1: Specialist Team")
    print("-" * 40)
    specialist_team = create_swarm(
        name="Specialist Team",
        agents=[
            create_swarm_agent("Containment", ["contain", "isolate", "block"]),
            create_swarm_agent("Forensics", ["forensics", "memory", "disk"]),
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
            create_swarm_agent("Triage Analyst A", ["triage", "investigate"]),
            create_swarm_agent("Triage Analyst B", ["triage", "investigate"]),
            create_swarm_agent("Triage Analyst C", ["triage", "investigate"]),
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
    print("  Evidence flows through a chain; output of one feeds the next.")
    print()


# =============================================================================
# Main
# =============================================================================


async def main():
    print("=" * 60)
    print("Notebook 24: IR war room — capability-matched responders, shared queue")
    print("=" * 60)
    print()

    print_config()
    print()

    example_create_agents()
    await example_shared_context()
    example_task_queue()
    example_capability_matching()
    await example_simple_swarm()
    await example_full_swarm()
    example_swarm_patterns()

    print("=" * 60)
    print("Next: Notebook 25 — Agent Handoff")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
