# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for multiagent swarm module."""

from datetime import UTC, datetime

import pytest

from locus.multiagent.swarm import (
    SharedContext,
    SwarmTask,
    TaskStatus,
)


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.CLAIMED == "claimed"
        assert TaskStatus.IN_PROGRESS == "in_progress"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"


class TestSwarmTask:
    """Tests for SwarmTask model."""

    def test_create_minimal_task(self):
        """Test creating task with minimal fields."""
        task = SwarmTask(description="Test task")
        assert task.description == "Test task"
        assert task.status == TaskStatus.PENDING
        assert task.priority == 0
        assert task.claimed_by is None
        assert task.result is None
        assert task.id.startswith("task_")

    def test_create_full_task(self):
        """Test creating task with all fields."""
        now = datetime.now(UTC)
        task = SwarmTask(
            id="custom_id",
            description="Full task",
            priority=10,
            status=TaskStatus.IN_PROGRESS,
            claimed_by="agent1",
            metadata={"key": "value"},
            parent_task_id="parent1",
            created_at=now,
        )
        assert task.id == "custom_id"
        assert task.priority == 10
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.claimed_by == "agent1"
        assert task.metadata == {"key": "value"}
        assert task.parent_task_id == "parent1"

    def test_task_id_auto_generated(self):
        """Test task ID is auto-generated."""
        task1 = SwarmTask(description="Task 1")
        task2 = SwarmTask(description="Task 2")
        assert task1.id != task2.id

    def test_task_default_timestamps(self):
        """Test task has created_at by default."""
        task = SwarmTask(description="Test")
        assert task.created_at is not None
        assert task.completed_at is None


class TestSharedContext:
    """Tests for SharedContext model."""

    def test_create_empty_context(self):
        """Test creating empty shared context."""
        ctx = SharedContext()
        assert ctx.findings == {}
        assert ctx.discovery_log == []
        assert ctx.blackboard == {}
        assert ctx.task_results == {}

    @pytest.mark.asyncio
    async def test_add_finding(self):
        """Test adding a finding to context."""
        ctx = SharedContext()
        await ctx.add_finding("key1", "value1", "agent1")

        assert ctx.findings["key1"] == "value1"
        assert len(ctx.discovery_log) == 1
        assert ctx.discovery_log[0]["type"] == "finding"
        assert ctx.discovery_log[0]["key"] == "key1"
        assert ctx.discovery_log[0]["agent_id"] == "agent1"

    @pytest.mark.asyncio
    async def test_add_multiple_findings(self):
        """Test adding multiple findings."""
        ctx = SharedContext()
        await ctx.add_finding("key1", "value1", "agent1")
        await ctx.add_finding("key2", "value2", "agent2")

        assert len(ctx.findings) == 2
        assert len(ctx.discovery_log) == 2

    @pytest.mark.asyncio
    async def test_post_to_blackboard(self):
        """Test posting to blackboard."""
        ctx = SharedContext()
        await ctx.post_to_blackboard("topic", "message", "agent1")

        assert ctx.blackboard["topic"] == "message"
        # Check discovery log entry
        assert len(ctx.discovery_log) == 1
        assert ctx.discovery_log[0]["type"] == "blackboard"

    @pytest.mark.asyncio
    async def test_record_task_result(self):
        """Test recording task result."""
        ctx = SharedContext()
        await ctx.record_task_result("task1", "result1")

        assert ctx.task_results["task1"] == "result1"

    @pytest.mark.asyncio
    async def test_access_findings_directly(self):
        """Test accessing findings directly."""
        ctx = SharedContext()
        await ctx.add_finding("key1", {"data": "value"}, "agent1")

        assert ctx.findings["key1"] == {"data": "value"}
        assert ctx.findings.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_access_blackboard_directly(self):
        """Test accessing blackboard directly."""
        ctx = SharedContext()
        await ctx.post_to_blackboard("topic", "hello", "agent1")

        assert ctx.blackboard["topic"] == "hello"
        assert ctx.blackboard.get("nonexistent") is None

    def test_get_summary(self):
        """Test getting context summary."""
        ctx = SharedContext()
        ctx.findings["key1"] = "value1"
        ctx.task_results["task1"] = "result1"

        summary = ctx.get_summary()

        # Summary returns a formatted string
        assert isinstance(summary, str)
        assert "Shared Context Summary" in summary
        assert "key1" in summary

    def test_get_summary_with_blackboard(self):
        """Test summary includes blackboard messages."""
        ctx = SharedContext()
        ctx.blackboard["topic1"] = "Message here"

        summary = ctx.get_summary()
        assert "Blackboard" in summary
        assert "topic1" in summary

    def test_get_summary_with_many_log_entries(self):
        """Test summary with more than 5 discovery log entries."""
        ctx = SharedContext()
        for i in range(10):
            ctx.discovery_log.append(
                {
                    "type": "finding",
                    "key": f"key{i}",
                    "value": f"value{i}",
                }
            )

        summary = ctx.get_summary()
        assert "10 total entries" in summary


from unittest.mock import AsyncMock, MagicMock

from locus.multiagent.swarm import (
    Swarm,
    SwarmAgent,
    SwarmResult,
    create_swarm,
    create_swarm_agent,
)


class TestSwarmAgent:
    """Tests for SwarmAgent model."""

    def test_create_minimal_agent(self):
        """Test creating agent with minimal fields."""
        agent = SwarmAgent(name="TestAgent")
        assert agent.name == "TestAgent"
        assert agent.capabilities == []
        assert agent.system_prompt == ""
        assert agent.model is None
        assert agent.current_task is None
        assert agent.tasks_completed == 0

    def test_create_full_agent(self):
        """Test creating agent with all fields."""
        mock_model = MagicMock()
        agent = SwarmAgent(
            name="FullAgent",
            capabilities=["research", "analysis"],
            system_prompt="You are an expert.",
            model=mock_model,
        )
        assert agent.capabilities == ["research", "analysis"]
        assert agent.system_prompt == "You are an expert."
        assert agent.model is mock_model

    def test_can_handle_generalist(self):
        """Test generalist agent can handle any task."""
        agent = SwarmAgent(name="Generalist")
        task = SwarmTask(description="Any random task")
        assert agent.can_handle(task) is True

    def test_can_handle_specialist_match(self):
        """Test specialist can handle matching task."""
        agent = SwarmAgent(name="Researcher", capabilities=["research"])
        task = SwarmTask(description="Research the market trends")
        assert agent.can_handle(task) is True

    def test_can_handle_specialist_no_match(self):
        """Test specialist cannot handle non-matching task."""
        agent = SwarmAgent(name="Researcher", capabilities=["research"])
        task = SwarmTask(description="Write code for feature")
        assert agent.can_handle(task) is False

    def test_priority_for_task_generalist(self):
        """Test priority for generalist is neutral."""
        agent = SwarmAgent(name="Generalist")
        task = SwarmTask(description="Any task")
        assert agent.priority_for_task(task) == 0.5

    def test_priority_for_task_full_match(self):
        """Test priority for full capability match."""
        agent = SwarmAgent(name="Specialist", capabilities=["research"])
        task = SwarmTask(description="Research the topic")
        assert agent.priority_for_task(task) == 1.0

    def test_priority_for_task_partial_match(self):
        """Test priority for partial capability match."""
        agent = SwarmAgent(
            name="MultiSkill",
            capabilities=["research", "analysis", "writing"],
        )
        task = SwarmTask(description="Research something")
        priority = agent.priority_for_task(task)
        assert priority == pytest.approx(1 / 3, rel=0.01)

    @pytest.mark.asyncio
    async def test_work_on_task_no_model(self):
        """Test work on task without model returns error."""
        agent = SwarmAgent(name="NoModel")
        task = SwarmTask(description="Do something")
        context = SharedContext()

        result, error = await agent.work_on_task(task, context)

        assert result is None
        assert error == "No model configured for agent"

    @pytest.mark.asyncio
    async def test_work_on_task_with_model(self):
        """Test work on task with mocked model."""
        mock_model = AsyncMock()
        mock_response = MagicMock()
        mock_response.message.content = """### Findings
Important discovery

### Analysis
Analysis here"""
        mock_model.complete.return_value = mock_response

        agent = SwarmAgent(name="WithModel", model=mock_model)
        task = SwarmTask(id="task1", description="Research topic")
        context = SharedContext()

        result, error = await agent.work_on_task(task, context)

        assert result is not None
        assert error is None
        mock_model.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_work_on_task_extracts_findings(self):
        """Test findings are extracted and shared."""
        mock_model = AsyncMock()
        mock_response = MagicMock()
        mock_response.message.content = """### Findings
Important discovery here

### Analysis
Done"""
        mock_model.complete.return_value = mock_response

        agent = SwarmAgent(name="Agent", model=mock_model)
        task = SwarmTask(id="task123", description="Research")
        context = SharedContext()

        await agent.work_on_task(task, context)

        assert "task_task123_findings" in context.findings

    @pytest.mark.asyncio
    async def test_work_on_task_extracts_blackboard(self):
        """Test blackboard messages are extracted."""
        mock_model = AsyncMock()
        mock_response = MagicMock()
        mock_response.message.content = """### Findings
Discovery

### Blackboard
Need help with X"""
        mock_model.complete.return_value = mock_response

        agent = SwarmAgent(id="agent_123", name="Agent", model=mock_model)
        task = SwarmTask(description="Research")
        context = SharedContext()

        await agent.work_on_task(task, context)

        assert "agent_agent_123_message" in context.blackboard

    @pytest.mark.asyncio
    async def test_work_on_task_model_error(self):
        """Test handling model errors."""
        mock_model = AsyncMock()
        mock_model.complete.side_effect = RuntimeError("API Error")

        agent = SwarmAgent(name="FailingModel", model=mock_model)
        task = SwarmTask(description="Do something")
        context = SharedContext()

        result, error = await agent.work_on_task(task, context)

        assert result is None
        assert "API Error" in error

    def test_with_model(self):
        """Test with_model creates copy with model."""
        agent = SwarmAgent(name="Agent")
        mock_model = MagicMock()

        new_agent = agent.with_model(mock_model)

        assert new_agent.model is mock_model
        assert agent.model is None  # Original unchanged


class TestSwarmResult:
    """Tests for SwarmResult model."""

    def test_create_success_result(self):
        """Test creating successful result."""
        result = SwarmResult(
            swarm_id="swarm_123",
            success=True,
            duration_ms=1500.0,
        )
        assert result.swarm_id == "swarm_123"
        assert result.success is True
        assert result.duration_ms == 1500.0
        assert result.completed_tasks == []
        assert result.failed_tasks == []
        assert result.error is None

    def test_create_failure_result(self):
        """Test creating failure result with error."""
        result = SwarmResult(
            swarm_id="swarm_123",
            success=False,
            error="Something went wrong",
        )
        assert result.success is False
        assert result.error == "Something went wrong"


class TestSwarm:
    """Tests for Swarm model."""

    def test_create_default_swarm(self):
        """Test creating swarm with defaults."""
        swarm = Swarm()
        assert swarm.name == "Swarm"
        assert swarm.agents == []
        assert swarm.task_queue == []
        assert swarm.max_iterations == 10
        assert swarm.max_parallel_agents == 5
        assert swarm.id.startswith("swarm_")

    def test_add_agent(self):
        """Test adding agent to swarm."""
        swarm = Swarm()
        agent = SwarmAgent(name="Agent1")

        result = swarm.add_agent(agent)

        assert result is swarm  # Returns self for chaining
        assert len(swarm.agents) == 1

    def test_add_task(self):
        """Test adding task to queue."""
        swarm = Swarm()
        task = swarm.add_task("Do something", priority=5)

        assert len(swarm.task_queue) == 1
        assert task.description == "Do something"
        assert task.priority == 5

    def test_add_task_sorted_by_priority(self):
        """Test tasks sorted by priority (high first)."""
        swarm = Swarm()
        swarm.add_task("Low", priority=1)
        swarm.add_task("High", priority=10)
        swarm.add_task("Medium", priority=5)

        assert swarm.task_queue[0].priority == 10
        assert swarm.task_queue[1].priority == 5
        assert swarm.task_queue[2].priority == 1

    def test_add_task_with_parent(self):
        """Test adding subtask with parent."""
        swarm = Swarm()
        task = swarm.add_task(
            "Subtask",
            parent_task_id="parent123",
            metadata={"key": "value"},
        )

        assert task.parent_task_id == "parent123"
        assert task.metadata == {"key": "value"}

    @pytest.mark.asyncio
    async def test_claim_task(self):
        """Test agent claiming a task."""
        swarm = Swarm()
        swarm.add_task("Task 1")
        agent = SwarmAgent(name="Agent1")

        task = await swarm._claim_task(agent)

        assert task is not None
        assert task.status == TaskStatus.CLAIMED
        assert task.claimed_by == agent.id

    @pytest.mark.asyncio
    async def test_claim_task_empty_queue(self):
        """Test claiming from empty queue."""
        swarm = Swarm()
        agent = SwarmAgent(name="Agent1")

        task = await swarm._claim_task(agent)
        assert task is None

    @pytest.mark.asyncio
    async def test_claim_task_respects_capabilities(self):
        """Test agent can only claim matching tasks."""
        swarm = Swarm()
        swarm.add_task("Write some code")
        agent = SwarmAgent(name="Researcher", capabilities=["research"])

        task = await swarm._claim_task(agent)
        assert task is None

    @pytest.mark.asyncio
    async def test_execute_no_tasks(self):
        """Test execute with no tasks."""
        swarm = Swarm()
        swarm.add_agent(SwarmAgent(name="Agent1"))

        result = await swarm.execute()

        assert result.success is True
        assert len(result.completed_tasks) == 0

    @pytest.mark.asyncio
    async def test_execute_with_initial_task(self):
        """Test execute with initial task."""
        mock_model = AsyncMock()
        mock_response = MagicMock()
        mock_response.message.content = "Done"
        mock_model.complete.return_value = mock_response

        swarm = Swarm()
        agent = SwarmAgent(name="Agent1", model=mock_model)
        swarm.add_agent(agent)

        result = await swarm.execute(initial_task="Do something", decompose_tasks=False)

        assert result.swarm_id == swarm.id
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_execute_with_decomposition(self):
        """Test execute with task decomposition."""
        mock_model = AsyncMock()
        mock_response = MagicMock()
        mock_response.message.content = '["Subtask 1", "Subtask 2"]'
        mock_model.complete.return_value = mock_response

        swarm = Swarm(model=mock_model)
        agent = SwarmAgent(name="Agent1", model=mock_model)
        swarm.add_agent(agent)

        result = await swarm.execute(initial_task="Complex task", decompose_tasks=True)

        assert result.swarm_id == swarm.id
        # Main task + subtasks
        assert len(swarm.task_queue) >= 1

    @pytest.mark.asyncio
    async def test_execute_task_failure(self):
        """Test execute with failing task."""
        mock_model = AsyncMock()
        mock_model.complete.side_effect = RuntimeError("API Error")

        swarm = Swarm()
        agent = SwarmAgent(name="Agent1", model=mock_model)
        swarm.add_agent(agent)
        swarm.add_task("Failing task")

        result = await swarm.execute()

        assert len(result.failed_tasks) >= 1

    @pytest.mark.asyncio
    async def test_execute_task_timeout(self):
        """Test task timeout handling."""
        import asyncio

        async def slow_complete(*args, **kwargs):
            await asyncio.sleep(10)
            return MagicMock()

        mock_model = MagicMock()
        mock_model.complete = slow_complete

        swarm = Swarm(task_timeout_ms=50)
        agent = SwarmAgent(name="SlowAgent", model=mock_model)
        swarm.add_agent(agent)
        swarm.add_task("Slow task")

        result = await swarm.execute()

        assert len(result.failed_tasks) == 1
        assert "timed out" in result.failed_tasks[0].error

    def test_with_model(self):
        """Test with_model updates swarm and agents."""
        swarm = Swarm()
        agent1 = SwarmAgent(name="Agent1")
        swarm.add_agent(agent1)

        mock_model = MagicMock()
        new_swarm = swarm.with_model(mock_model)

        assert new_swarm.model is mock_model
        assert new_swarm.agents[0].model is mock_model
        assert swarm.model is None  # Original unchanged

    @pytest.mark.asyncio
    async def test_generate_summary_no_model(self):
        """Test summary without model."""
        swarm = Swarm()
        summary = await swarm._generate_summary()
        assert "Shared Context Summary" in summary

    @pytest.mark.asyncio
    async def test_generate_summary_with_model(self):
        """Test summary with model."""
        mock_model = AsyncMock()
        mock_response = MagicMock()
        mock_response.message.content = "Executive summary"
        mock_model.complete.return_value = mock_response

        swarm = Swarm(model=mock_model)
        summary = await swarm._generate_summary()

        assert summary == "Executive summary"

    @pytest.mark.asyncio
    async def test_generate_summary_model_error(self):
        """Test summary fallback on model error."""
        mock_model = AsyncMock()
        mock_model.complete.side_effect = RuntimeError("Error")

        swarm = Swarm(model=mock_model)
        summary = await swarm._generate_summary()

        assert "Shared Context Summary" in summary

    @pytest.mark.asyncio
    async def test_generate_subtasks(self):
        """Test subtask generation."""
        mock_model = AsyncMock()
        mock_response = MagicMock()
        mock_response.message.content = '["Subtask A", "Subtask B"]'
        mock_model.complete.return_value = mock_response

        swarm = Swarm(model=mock_model)
        task = SwarmTask(description="Main task", priority=5)

        subtasks = await swarm._generate_subtasks(task)

        assert len(subtasks) == 2
        assert subtasks[0].parent_task_id == task.id
        assert subtasks[0].priority == 4  # Lower than parent

    @pytest.mark.asyncio
    async def test_generate_subtasks_no_model(self):
        """Test subtask generation without model."""
        swarm = Swarm()
        task = SwarmTask(description="Main task")

        subtasks = await swarm._generate_subtasks(task)
        assert subtasks == []


class TestCreateSwarm:
    """Tests for create_swarm factory."""

    def test_create_empty_swarm(self):
        """Test creating empty swarm."""
        swarm = create_swarm(name="TestSwarm")
        assert swarm.name == "TestSwarm"
        assert swarm.agents == []

    def test_create_swarm_with_agents(self):
        """Test creating swarm with agents."""
        agents = [SwarmAgent(name="A1"), SwarmAgent(name="A2")]
        swarm = create_swarm(name="Team", agents=agents)

        assert len(swarm.agents) == 2

    def test_create_swarm_with_model(self):
        """Test creating swarm with model."""
        mock_model = MagicMock()
        swarm = create_swarm(name="ModelSwarm", model=mock_model)

        assert swarm.model is mock_model


class TestCreateSwarmAgent:
    """Tests for create_swarm_agent factory."""

    def test_create_basic_agent(self):
        """Test creating basic agent."""
        agent = create_swarm_agent(name="Basic")
        assert agent.name == "Basic"
        assert agent.capabilities == []
        assert agent.model is None

    def test_create_full_agent(self):
        """Test creating fully configured agent."""
        mock_model = MagicMock()
        agent = create_swarm_agent(
            name="Full",
            capabilities=["research"],
            system_prompt="Expert",
            model=mock_model,
        )

        assert agent.capabilities == ["research"]
        assert agent.system_prompt == "Expert"
        assert agent.model is mock_model
