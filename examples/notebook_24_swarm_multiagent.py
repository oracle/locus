# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 25: swarm — capability-matched workers claiming from a shared queue.

A swarm is a pool of agents that pull tasks from a shared queue based on
declared capabilities. No supervisor decides who does what — each task
finds the worker whose tags fit best, and the swarm runs them in parallel
where it can.

- Each agent advertises ``capabilities`` (free-form string tags).
- Tasks carry ``required_tags`` (hard filter) and ``preferred_tags``
  (priority boost). Tagless tasks fall back to substring matching against
  the description so older swarms keep working.
- A ``SharedContext`` (findings, blackboard messages, task results) lets
  agents leave notes for each other without a central coordinator.
- ``Swarm.execute`` decomposes the initial task into capability-matched
  subtasks and runs them.

Run it:
    .venv/bin/python examples/notebook_30_swarm_multiagent.py

The default provider is the bundled mock model. Set TULIP_MODEL_PROVIDER=openai (or anthropic)
the swarm talks to a live model (e.g.
``openai.gpt-4.1`` or ``meta.llama-3.3-70b-instruct``). Set
``TULIP_MODEL_PROVIDER=mock`` for offline runs.

Prerequisites:
- Notebook 08 (Agent basics).
- Notebook 27 (Orchestrator) if you want the supervised counterpoint.
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
# Part 1: declare the workers
# =============================================================================


def example_create_agents():
    """Create three specialist swarm agents."""
    print("=== Part 1: Creating Swarm Agents ===\n")
    print(
        f"AI rationale: {_llm_call('In one sentence, when is a swarm of small specialised agents a better fit than one generalist agent?')}"
    )

    researcher = create_swarm_agent(
        name="Researcher",
        capabilities=["research", "analyze", "investigate"],
        system_prompt="You are a research specialist. Find and analyze information.",
    )

    writer = create_swarm_agent(
        name="Writer",
        capabilities=["write", "summarize", "document"],
        system_prompt="You are a writing specialist. Create clear documentation.",
    )

    reviewer = create_swarm_agent(
        name="Reviewer",
        capabilities=["review", "validate", "check"],
        system_prompt="You are a quality reviewer. Verify accuracy and completeness.",
    )

    print("Created agents:")
    for agent in [researcher, writer, reviewer]:
        print(f"  - {agent.name}: capabilities = {agent.capabilities}")
    print()

    return researcher, writer, reviewer


# =============================================================================
# Part 2: the blackboard
# =============================================================================


async def example_shared_context():
    """SharedContext — the blackboard agents use to leave notes for each other."""
    print("=== Part 2: Shared Context ===\n")
    print(
        f"AI rationale: {_llm_call('In one sentence, why does a swarm need SharedContext for messages and discoveries?')}"
    )

    context = SharedContext()

    await context.add_finding(
        key="api_docs",
        value="The API uses REST with JSON responses",
        agent_id="agent_1",
    )

    await context.post_to_blackboard(
        key="need_help",
        message="Need someone to review the authentication section",
        agent_id="agent_1",
    )

    await context.record_task_result(
        task_id="task_001",
        result="Completed analysis of the codebase structure",
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
        f"AI rationale: {_llm_call('In one sentence, why is task-queue routing useful for a heterogeneous swarm?')}"
    )

    swarm = Swarm(name="Research Team")

    task1 = swarm.add_task("Research the API documentation", priority=5)
    task2 = swarm.add_task("Write a summary report", priority=3)
    task3 = swarm.add_task("Review the findings for accuracy", priority=2)
    task4 = swarm.add_task("Investigate security concerns", priority=10)

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
        f"AI rationale: {_llm_call('In one sentence, why is capability-based agent selection better than random round-robin?')}"
    )

    researcher = create_swarm_agent(
        name="Researcher",
        capabilities=["research", "analyze"],
    )

    writer = create_swarm_agent(
        name="Writer",
        capabilities=["write", "document"],
    )

    # required_tags are a hard filter (set membership). preferred_tags
    # only boost the score. Tasks without any tags fall back to substring
    # matching against the description — so older tagless swarms still
    # route correctly.
    tasks = [
        SwarmTask(
            description="Research the competitor landscape",
            required_tags=["research"],
            preferred_tags=["analyze"],
        ),
        SwarmTask(
            description="Write documentation for the API",
            required_tags=["write", "document"],
        ),
        SwarmTask(description="Analyze the performance data", required_tags=["analyze"]),
        SwarmTask(description="Create a summary document"),
    ]

    print("Task-Agent matching:")
    for task in tasks:
        print(f"\n  Task: {task.description}")
        print(f"    required_tags={task.required_tags} preferred_tags={task.preferred_tags}")
        print(f"    Researcher can handle: {researcher.can_handle(task)}")
        print(f"    Writer can handle: {writer.can_handle(task)}")
        print(f"    Researcher priority: {researcher.priority_for_task(task):.2f}")
        print(f"    Writer priority: {writer.priority_for_task(task):.2f}")
    print()


# =============================================================================
# Part 5: a swarm without a model — just the shape
# =============================================================================


async def example_simple_swarm():
    """Stand up a swarm without a model — useful when you only want the shape."""
    print("=== Part 5: Simple Swarm Execution ===\n")
    rationale_prompt = (
        "In one sentence, what does 'simple swarm execution' mean and when is it enough?"
    )
    print(f"AI rationale: {_llm_call(rationale_prompt)}")

    swarm = Swarm(name="Demo Swarm")

    agent1 = create_swarm_agent(
        name="Analyst",
        capabilities=["analyze"],
        system_prompt="You analyze data.",
    )

    agent2 = create_swarm_agent(
        name="Reporter",
        capabilities=["report"],
        system_prompt="You create reports.",
    )

    swarm.add_agent(agent1)
    swarm.add_agent(agent2)

    swarm.add_task("Analyze the sales data", priority=5)
    swarm.add_task("Report on the findings", priority=3)

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
    """Run the swarm end-to-end against a real model."""
    print("=== Part 6: Full Swarm with Model ===\n")

    # Each agent emits a structured `### Findings / ### Analysis /
    # ### Blackboard` block — leave enough completion budget for it.
    model = get_model(max_tokens=2000)

    swarm = create_swarm(
        name="Analysis Team",
        agents=[
            create_swarm_agent(
                name="Researcher",
                capabilities=["research", "investigate", "find"],
                system_prompt="You are a research specialist. Find relevant information.",
            ),
            create_swarm_agent(
                name="Analyst",
                capabilities=["analyze", "evaluate", "assess"],
                system_prompt="You analyze and evaluate findings critically.",
            ),
            create_swarm_agent(
                name="Writer",
                capabilities=["write", "summarize", "document"],
                system_prompt="You write clear, concise summaries.",
            ),
        ],
        model=model,
    )

    print("Executing swarm on: 'Analyze the benefits of async programming'")
    print("This may take a moment...\n")

    result = await swarm.execute(
        initial_task=(
            "Research, analyze, and write a brief summary of the benefits "
            "of async programming in Python."
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
        f"AI rationale: {_llm_call('In one sentence, when is a Specialist Team swarm preferable to a Pipeline swarm?')}"
    )

    print("Pattern 1: Specialist Team")
    print("-" * 40)
    specialist_team = create_swarm(
        name="Specialist Team",
        agents=[
            create_swarm_agent("Frontend Dev", ["frontend", "UI", "React"]),
            create_swarm_agent("Backend Dev", ["backend", "API", "database"]),
            create_swarm_agent("DevOps", ["deploy", "infrastructure", "CI/CD"]),
        ],
    )
    print("  Distinct, non-overlapping capabilities; each task goes to its expert.")
    print()

    print("Pattern 2: Redundant Team")
    print("-" * 40)
    redundant_team = create_swarm(
        name="Redundant Team",
        agents=[
            create_swarm_agent("Analyst A", ["analyze", "research"]),
            create_swarm_agent("Analyst B", ["analyze", "research"]),
            create_swarm_agent("Analyst C", ["analyze", "research"]),
        ],
    )
    print("  Overlapping capabilities; tasks fan out for parallel processing.")
    print()

    print("Pattern 3: Pipeline Team")
    print("-" * 40)
    pipeline_team = create_swarm(
        name="Pipeline Team",
        agents=[
            create_swarm_agent("Gatherer", ["gather", "collect", "fetch"]),
            create_swarm_agent("Processor", ["process", "transform", "clean"]),
            create_swarm_agent("Presenter", ["present", "format", "display"]),
        ],
    )
    print("  Agents form a processing chain; output of one feeds the next.")
    print()


# =============================================================================
# Main
# =============================================================================


async def main():
    print("=" * 60)
    print("Notebook 25: swarm — capability-matched workers, shared queue")
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
    print("Next: Notebook 26 — Agent Handoff")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
