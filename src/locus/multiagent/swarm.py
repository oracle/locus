# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Self-organizing swarm of agents with shared context."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, PrivateAttr

from locus.core.messages import Message


class TaskStatus(StrEnum):
    """Status of a task in the swarm."""

    PENDING = "pending"
    CLAIMED = "claimed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class SwarmTask(BaseModel):
    """A task in the swarm task queue.

    ``required_tags`` declares the capability tags an agent must
    advertise to claim this task — set-membership against
    :attr:`SwarmAgent.capabilities`. Empty means any agent may claim
    it; ``preferred_tags`` boost an agent's priority score without
    being a hard requirement.

    Backwards-compat: agents that pre-date this change still work —
    if no tags are set, the prior substring-match path runs as a
    fallback (see ``SwarmAgent.can_handle``).
    """

    id: str = Field(default_factory=lambda: f"task_{uuid4().hex[:8]}")
    description: str
    priority: int = 0  # Higher = more important
    required_tags: list[str] = Field(default_factory=list)
    preferred_tags: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    claimed_by: str | None = None
    result: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    parent_task_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SharedContext(BaseModel):
    """
    Shared context/memory for swarm agents.

    All agents can read from and write to this shared state.
    """

    # Key-value store for findings
    findings: dict[str, Any] = Field(default_factory=dict)

    # Ordered log of discoveries
    discovery_log: list[dict[str, Any]] = Field(default_factory=list)

    # Blackboard for inter-agent communication
    blackboard: dict[str, str] = Field(default_factory=dict)

    # Task results indexed by task ID
    task_results: dict[str, str] = Field(default_factory=dict)

    # Lock for thread-safe updates (not serialized)
    _lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    model_config = {"arbitrary_types_allowed": True}

    async def add_finding(self, key: str, value: Any, agent_id: str) -> None:
        """Add a finding to the shared context."""
        async with self._lock:
            self.findings[key] = value
            self.discovery_log.append(
                {
                    "type": "finding",
                    "key": key,
                    "value": value,
                    "agent_id": agent_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

    async def post_to_blackboard(self, key: str, message: str, agent_id: str) -> None:
        """Post a message to the blackboard."""
        async with self._lock:
            self.blackboard[key] = message
            self.discovery_log.append(
                {
                    "type": "blackboard",
                    "key": key,
                    "message": message,
                    "agent_id": agent_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

    async def record_task_result(self, task_id: str, result: str) -> None:
        """Record a task result."""
        async with self._lock:
            self.task_results[task_id] = result

    def get_summary(self) -> str:
        """Get a summary of the current context state."""
        lines = ["## Shared Context Summary"]

        if self.findings:
            lines.append("\n### Findings:")
            for key, value in self.findings.items():
                lines.append(f"- **{key}**: {value}")

        if self.task_results:
            lines.append("\n### Task Results:")
            for tid, result in self.task_results.items():
                snippet = result if len(result) <= 600 else result[:600].rstrip() + "…"
                lines.append(f"- **{tid}**:\n{snippet}")

        if self.blackboard:
            lines.append("\n### Blackboard Messages:")
            for key, msg in self.blackboard.items():
                lines.append(f"- **{key}**: {msg}")

        if len(self.discovery_log) > 5:
            lines.append(f"\n### Recent Activity: ({len(self.discovery_log)} total entries)")
            for entry in self.discovery_log[-5:]:
                lines.append(f"- [{entry['type']}] {entry.get('key', 'unknown')}")

        return "\n".join(lines)


class SwarmAgent(BaseModel):
    """
    An agent in the swarm.

    Autonomously claims and works on tasks from the shared queue.
    """

    id: str = Field(default_factory=lambda: f"agent_{uuid4().hex[:8]}")
    name: str
    capabilities: list[str] = Field(default_factory=list)
    system_prompt: str = ""
    model: Any = None

    # Agent state
    current_task: SwarmTask | None = None
    tasks_completed: int = 0

    model_config = {"arbitrary_types_allowed": True}

    def can_handle(self, task: SwarmTask) -> bool:
        """Decide whether this agent is eligible to claim ``task``.

        Resolution order:

        1. **Tag set-membership** — if the task declares
           ``required_tags``, every required tag must appear in
           :attr:`capabilities`. This is the deterministic path —
           the swarm's primary discovery mechanism.
        2. **Generalist** — if neither side declares anything,
           the agent claims the task (``capabilities=[]`` ⇒
           generalist).
        3. **Substring fallback** — kept for backwards compatibility
           with pre-tag swarms: if the agent has capabilities but the
           task has no tags, match keywords against the description.
        """
        if task.required_tags:
            agent_tags = {c.lower() for c in self.capabilities}
            return all(req.lower() in agent_tags for req in task.required_tags)
        if not self.capabilities:
            return True
        task_lower = task.description.lower()
        return any(cap.lower() in task_lower for cap in self.capabilities)

    def priority_for_task(self, task: SwarmTask) -> float:
        """Score this agent's fit for ``task`` in the range ``[0, 1]``.

        Tag-driven scoring (when the task declares tags):

        - Each required tag the agent advertises adds 1.0 weight.
        - Each preferred tag the agent advertises adds 0.5 weight.
        - Score = (sum of weights) / (max possible weights), clamped.

        Falls through to the legacy substring score for tag-less
        tasks so pre-tag swarms keep their old behaviour.
        """
        if task.required_tags or task.preferred_tags:
            agent_tags = {c.lower() for c in self.capabilities}
            req_hits = sum(1 for t in task.required_tags if t.lower() in agent_tags)
            pref_hits = sum(1 for t in task.preferred_tags if t.lower() in agent_tags)
            max_weight = float(len(task.required_tags)) + 0.5 * len(task.preferred_tags)
            if max_weight == 0:
                return 0.5
            return min(1.0, (req_hits + 0.5 * pref_hits) / max_weight)
        if not self.capabilities:
            return 0.5
        task_lower = task.description.lower()
        matches = sum(1 for cap in self.capabilities if cap.lower() in task_lower)
        return min(1.0, matches / len(self.capabilities))

    async def work_on_task(
        self,
        task: SwarmTask,
        context: SharedContext,
    ) -> tuple[str | None, str | None]:
        """
        Work on a task using the shared context.

        Args:
            task: The task to work on
            context: Shared context with other agents

        Returns:
            Tuple of (result, error)
        """
        if self.model is None:
            return None, "No model configured for agent"

        # Build prompt with context
        context_summary = context.get_summary()

        prompt = f"""## Your Role
{self.system_prompt}

## Shared Context
{context_summary}

## Task
{task.description}

## Instructions
1. Analyze the task and shared context
2. If you discover new findings, note them clearly
3. If you need information from other agents, post a request to the blackboard
4. Complete the task to the best of your ability
5. Report your findings clearly

Format your response as:
### Findings
(Any new discoveries)

### Analysis
(Your analysis and conclusions)

### Blackboard (optional)
(Any messages for other agents)"""

        messages = [
            Message.system("You are a collaborative agent in a swarm."),
            Message.user(prompt),
        ]

        try:
            response = await self.model.complete(messages=messages)
            content = response.message.content or ""

            # Extract findings and update context
            await self._extract_and_share(content, context, task.id)

            return content, None

        except Exception as e:  # noqa: BLE001
            return None, str(e)

    async def _extract_and_share(
        self,
        response: str,
        context: SharedContext,
        task_id: str,
    ) -> None:
        """Extract findings from response and share to context."""
        # Simple extraction - could be enhanced with structured output
        if "### Findings" in response:
            findings_section = response.split("### Findings")[1]
            if "###" in findings_section:
                findings_section = findings_section.split("###")[0]

            # Record as a finding
            await context.add_finding(
                key=f"task_{task_id}_findings",
                value=findings_section.strip(),
                agent_id=self.id,
            )

        if "### Blackboard" in response:
            blackboard_section = response.split("### Blackboard")[1]
            if "###" in blackboard_section:
                blackboard_section = blackboard_section.split("###")[0]

            await context.post_to_blackboard(
                key=f"agent_{self.id}_message",
                message=blackboard_section.strip(),
                agent_id=self.id,
            )

    def with_model(self, model: Any) -> SwarmAgent:
        """Return a copy with the given model."""
        return self.model_copy(update={"model": model})


class SwarmResult(BaseModel):
    """Result from swarm execution."""

    swarm_id: str
    success: bool
    completed_tasks: list[SwarmTask] = Field(default_factory=list)
    failed_tasks: list[SwarmTask] = Field(default_factory=list)
    context: SharedContext = Field(default_factory=SharedContext)
    summary: str | None = None
    duration_ms: float = 0.0
    error: str | None = None

    model_config = {"arbitrary_types_allowed": True}


class Swarm(BaseModel):
    """
    A self-organizing swarm of agents.

    Features:
    - Agents coordinate autonomously
    - Shared context/memory for communication
    - Dynamic task allocation based on capabilities
    - Parallel execution with coordination
    """

    id: str = Field(default_factory=lambda: f"swarm_{uuid4().hex[:8]}")
    name: str = "Swarm"

    # Agents in the swarm
    agents: list[SwarmAgent] = Field(default_factory=list)

    # Task queue
    task_queue: list[SwarmTask] = Field(default_factory=list)

    # Shared context
    context: SharedContext = Field(default_factory=SharedContext)

    # Configuration
    max_iterations: int = 10
    max_parallel_agents: int = 5
    # Real LLM completions with structured-response budgets routinely exceed
    # 30s on gpt-class models, so the default per-task timeout is 120s.
    task_timeout_ms: float = 120000

    # Model for coordination decisions
    model: Any = None

    # Internal state
    _task_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    model_config = {"arbitrary_types_allowed": True}

    def add_agent(self, agent: SwarmAgent) -> Swarm:
        """Add an agent to the swarm.

        If the swarm has a model configured and the incoming agent does
        not, the agent inherits the swarm's model. Without this, the
        agent's first ``work_on_task`` would fail with
        "No model configured for agent" — which used to be the most
        common silent-failure mode for new users of the swarm API.
        """
        if self.model is not None and agent.model is None:
            agent = agent.with_model(self.model)
        self.agents.append(agent)
        return self

    def add_task(
        self,
        description: str,
        priority: int = 0,
        parent_task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SwarmTask:
        """Add a task to the queue."""
        task = SwarmTask(
            description=description,
            priority=priority,
            parent_task_id=parent_task_id,
            metadata=metadata or {},
        )
        self.task_queue.append(task)
        # Sort by priority (higher first)
        self.task_queue.sort(key=lambda t: -t.priority)
        return task

    async def _claim_task(self, agent: SwarmAgent) -> SwarmTask | None:
        """Have an agent claim a task from the queue."""
        async with self._task_lock:
            # Find unclaimed tasks this agent can handle
            for task in self.task_queue:
                if task.status == TaskStatus.PENDING and agent.can_handle(task):
                    task.status = TaskStatus.CLAIMED
                    task.claimed_by = agent.id
                    return task

            return None

    async def _run_agent_loop(self, agent: SwarmAgent) -> list[SwarmTask]:
        """Run an agent's work loop."""
        completed_tasks: list[SwarmTask] = []

        while True:
            # Try to claim a task
            task = await self._claim_task(agent)
            if task is None:
                break  # No more tasks for this agent

            # Work on the task
            task.status = TaskStatus.IN_PROGRESS

            try:
                result, error = await asyncio.wait_for(
                    agent.work_on_task(task, self.context),
                    timeout=self.task_timeout_ms / 1000,
                )

                if error:
                    task.status = TaskStatus.FAILED
                    task.error = error
                else:
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    task.completed_at = datetime.now(UTC)
                    completed_tasks.append(task)

                    # Record result in context
                    if result:
                        await self.context.record_task_result(task.id, result)

            except asyncio.TimeoutError:
                task.status = TaskStatus.FAILED
                task.error = f"Task timed out after {self.task_timeout_ms}ms"

            agent.tasks_completed += 1

        return completed_tasks

    async def _generate_subtasks(self, task: SwarmTask) -> list[SwarmTask]:
        """Use the model to break down a task into subtasks if needed."""
        if self.model is None:
            return []

        prompt = f"""Analyze this task and determine if it should be broken into subtasks:

Task: {task.description}

If this task is complex and would benefit from being split, respond with a JSON array of subtask descriptions.
If the task is simple enough to handle directly, respond with an empty array [].

Example response:
["Analyze the logs for errors", "Check the metrics for anomalies", "Correlate the findings"]"""

        messages = [
            Message.system("You are a task decomposition assistant."),
            Message.user(prompt),
        ]

        try:
            response = await self.model.complete(messages=messages)
            content = response.message.content or ""

            import json
            import re

            # Try to extract JSON array
            match = re.search(r"\[.*\]", content, re.DOTALL)
            if match:
                subtasks_data = json.loads(match.group())
                subtasks = []
                for desc in subtasks_data:
                    if isinstance(desc, str):
                        subtasks.append(
                            self.add_task(
                                description=desc,
                                priority=task.priority - 1,
                                parent_task_id=task.id,
                            )
                        )
                return subtasks

        except Exception:  # noqa: BLE001
            pass

        return []

    async def _generate_summary(self) -> str:
        """Generate a summary of the swarm's work."""
        if self.model is None:
            return self.context.get_summary()

        completed_lines = "\n".join(
            f"- {t.description}: {(t.result[:200] if t.result else 'No result')}..."
            for t in self.task_queue
            if t.status == TaskStatus.COMPLETED
        )
        prompt = f"""Summarize the work completed by the swarm:

{self.context.get_summary()}

## Completed Tasks
{completed_lines}

Provide a concise summary of the findings and conclusions."""

        messages = [
            Message.system("You are a summarization assistant."),
            Message.user(prompt),
        ]

        try:
            response = await self.model.complete(messages=messages)
            return response.message.content or self.context.get_summary()
        except Exception:  # noqa: BLE001
            return self.context.get_summary()

    async def execute(
        self,
        initial_task: str | None = None,
        decompose_tasks: bool = True,
    ) -> SwarmResult:
        """
        Execute the swarm.

        Args:
            initial_task: Optional initial task to add
            decompose_tasks: Whether to decompose tasks into subtasks

        Returns:
            SwarmResult with all completed work
        """
        start_time = time.perf_counter()

        # Add initial task if provided
        if initial_task:
            main_task = self.add_task(initial_task, priority=10)

            # Optionally decompose into subtasks
            if decompose_tasks:
                await self._generate_subtasks(main_task)

        try:
            semaphore = asyncio.Semaphore(self.max_parallel_agents)

            async def run_with_limit(agent: SwarmAgent) -> list[SwarmTask]:
                async with semaphore:
                    return await self._run_agent_loop(agent)

            # Group tasks by priority and run in waves (high priority first).
            # Tasks within the same priority wave run in parallel.
            # Lower-priority waves wait for higher-priority waves to complete,
            # so they can see the earlier findings in SharedContext.
            iteration = 0
            all_completed: list[SwarmTask] = []

            # Get unique priority levels (sorted descending = highest first)
            priority_levels = sorted({t.priority for t in self.task_queue}, reverse=True)

            for priority in priority_levels:
                if iteration >= self.max_iterations:
                    break

                # Check if there are pending tasks at this priority
                pending_at_level = [
                    t
                    for t in self.task_queue
                    if t.status == TaskStatus.PENDING and t.priority == priority
                ]
                if not pending_at_level:
                    continue

                # Run agents for this priority wave.
                # Agents run sequentially to avoid concurrent API issues
                # (some providers return empty responses under parallel load).
                # Each agent claims one task, completes it, then the next agent goes.
                results = []
                for agent in self.agents:
                    agent_results = await self._run_agent_loop(agent)
                    results.append(agent_results)

                for completed_list in results:
                    all_completed.extend(completed_list)

                iteration += 1

            # Handle any remaining pending tasks (fallback)
            remaining = [t for t in self.task_queue if t.status == TaskStatus.PENDING]
            while remaining and iteration < self.max_iterations:
                tasks = [run_with_limit(agent) for agent in self.agents]
                results = await asyncio.gather(*tasks)
                for completed_list in results:
                    all_completed.extend(completed_list)
                remaining = [t for t in self.task_queue if t.status == TaskStatus.PENDING]
                iteration += 1

            # Collect results
            completed = [t for t in self.task_queue if t.status == TaskStatus.COMPLETED]
            failed = [t for t in self.task_queue if t.status == TaskStatus.FAILED]

            # Generate summary
            summary = await self._generate_summary()

            duration_ms = (time.perf_counter() - start_time) * 1000

            return SwarmResult(
                swarm_id=self.id,
                success=len(failed) == 0,
                completed_tasks=completed,
                failed_tasks=failed,
                context=self.context,
                summary=summary,
                duration_ms=duration_ms,
            )

        except Exception as e:  # noqa: BLE001
            duration_ms = (time.perf_counter() - start_time) * 1000
            return SwarmResult(
                swarm_id=self.id,
                success=False,
                duration_ms=duration_ms,
                error=str(e),
            )

    def with_model(self, model: Any) -> Swarm:
        """Return a copy with the given model for all agents."""
        updated_agents = [agent.with_model(model) for agent in self.agents]
        return self.model_copy(
            update={
                "model": model,
                "agents": updated_agents,
            }
        )


def create_swarm(
    name: str = "Swarm",
    agents: list[SwarmAgent] | None = None,
    model: Any = None,
) -> Swarm:
    """
    Create a swarm with the given agents.

    Args:
        name: Swarm name
        agents: List of agents to add
        model: Model for agents and coordination. When provided, any agent
            in ``agents`` that doesn't already carry its own model inherits
            this one — otherwise ``Swarm.execute`` would later report
            "No model configured for agent" for those agents.

    Returns:
        Configured Swarm instance
    """
    swarm = Swarm(name=name, model=model)

    # ``Swarm.add_agent`` propagates the swarm's model into any agent that
    # doesn't already carry one, so we don't repeat that logic here.
    if agents:
        for agent in agents:
            swarm.add_agent(agent)

    return swarm


def create_swarm_agent(
    name: str,
    capabilities: list[str] | None = None,
    system_prompt: str = "",
    model: Any = None,
) -> SwarmAgent:
    """
    Create a swarm agent.

    Args:
        name: Agent name
        capabilities: List of capability keywords
        system_prompt: System prompt for the agent
        model: Model for the agent

    Returns:
        Configured SwarmAgent instance
    """
    return SwarmAgent(
        name=name,
        capabilities=capabilities or [],
        system_prompt=system_prompt,
        model=model,
    )
