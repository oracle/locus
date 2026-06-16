# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Tests for multi-agent orchestration."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from locus.core.messages import Message
from locus.core.protocols import ModelResponse
from locus.core.state import AgentState
from locus.multiagent import (
    Edge,
    HandoffContext,
    HandoffReason,
    Node,
    NodeStatus,
    Playbook,
    PlaybookStep,
    RoutingDecision,
    SharedContext,
    Specialist,
    SwarmAgent,
    SwarmTask,
    TaskStatus,
    create_graph,
    create_handoff_agent,
    create_handoff_manager,
    create_orchestrator,
    create_swarm,
    create_swarm_agent,
    node,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_model() -> MagicMock:
    """Create a mock model for testing."""
    model = MagicMock()
    model.complete = AsyncMock(
        return_value=ModelResponse(
            message=Message.assistant("Test response"),
            usage={"input_tokens": 100, "output_tokens": 50},
        )
    )
    return model


@pytest.fixture
def sample_state() -> AgentState:
    """Create a sample agent state."""
    state = AgentState(agent_id="test_agent")
    state = state.with_message(Message.system("You are a test agent."))
    state = state.with_message(Message.user("Test task"))
    state = state.with_confidence(0.5)
    return state


# =============================================================================
# Graph Tests
# =============================================================================


class TestNode:
    """Tests for Node class."""

    @pytest.mark.asyncio
    async def test_execute_simple_function(self):
        """Node executes a simple function."""

        async def simple_fn(inputs: dict[str, Any]) -> str:
            return f"Processed: {inputs.get('data', 'none')}"

        n = Node(name="simple", executor=simple_fn)
        result = await n.execute({"data": "hello"})

        assert result.success
        assert result.output == "Processed: hello"
        assert result.status == NodeStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_sync_function(self):
        """Node executes a sync function."""

        def sync_fn(inputs: dict[str, Any]) -> int:
            return inputs.get("x", 0) * 2

        n = Node(name="sync", executor=sync_fn)
        result = await n.execute({"x": 5})

        assert result.success
        assert result.output == 10

    @pytest.mark.asyncio
    async def test_execute_with_error(self):
        """Node handles execution errors."""

        async def failing_fn(inputs: dict[str, Any]) -> str:
            raise ValueError("Intentional error")

        n = Node(name="failing", executor=failing_fn)
        result = await n.execute({})

        assert not result.success
        assert result.status == NodeStatus.FAILED
        assert "Intentional error" in result.error

    @pytest.mark.asyncio
    async def test_execute_with_condition_true(self):
        """Node executes when condition is True."""

        async def fn(inputs: dict[str, Any]) -> str:
            return "executed"

        def condition(inputs: dict[str, Any]) -> bool:
            return inputs.get("run", False)

        n = Node(name="conditional", executor=fn, condition=condition)
        result = await n.execute({"run": True})

        assert result.success
        assert result.output == "executed"

    @pytest.mark.asyncio
    async def test_execute_with_condition_false(self):
        """Node is skipped when condition is False."""

        async def fn(inputs: dict[str, Any]) -> str:
            return "executed"

        def condition(inputs: dict[str, Any]) -> bool:
            return inputs.get("run", False)

        n = Node(name="conditional", executor=fn, condition=condition)
        result = await n.execute({"run": False})

        assert result.status == NodeStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_execute_with_retry(self):
        """Node retries on failure."""
        attempt_count = 0

        async def flaky_fn(inputs: dict[str, Any]) -> str:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ValueError("Temporary failure")
            return "success"

        n = Node(name="flaky", executor=flaky_fn, max_retries=2, retry_delay_ms=10)
        result = await n.execute({})

        assert result.success
        assert attempt_count == 2


class TestEdge:
    """Tests for Edge class."""

    def test_apply_simple(self):
        """Edge applies simple output."""
        edge = Edge(source_id="node1", target_id="node2")
        result = edge.apply({"key": "value"})

        assert result == {"node1": {"key": "value"}}

    def test_apply_with_key_mapping(self):
        """Edge applies key mapping."""
        edge = Edge(
            source_id="node1",
            target_id="node2",
            key_mapping={"out_key": "in_key"},
        )
        result = edge.apply({"out_key": "value", "other": "ignored"})

        assert result == {"in_key": "value"}

    def test_apply_with_transform(self):
        """Edge applies transformation."""
        edge = Edge(
            source_id="node1",
            target_id="node2",
            transform=lambda x: x.upper() if isinstance(x, str) else x,
        )
        result = edge.apply("hello")

        assert result == {"node1": "HELLO"}


class TestGraph:
    """Tests for Graph class."""

    def test_add_node(self):
        """Add nodes to graph."""
        graph = create_graph(name="test")

        n1 = node("node1", executor=lambda x: x)
        n2 = node("node2", executor=lambda x: x)

        graph.add_node(n1)
        graph.add_node(n2)

        # Graph now includes START and END nodes by default
        assert n1.id in graph.nodes
        assert n2.id in graph.nodes
        assert "__START__" in graph.nodes
        assert "__END__" in graph.nodes

    def test_add_duplicate_node_fails(self):
        """Adding duplicate node fails."""
        graph = create_graph()
        n = node("node1", executor=lambda x: x)

        graph.add_node(n)

        with pytest.raises(ValueError, match="already exists"):
            graph.add_node(n)

    def test_add_edge(self):
        """Add edges to graph."""
        graph = create_graph()
        n1 = node("node1", executor=lambda x: x)
        n2 = node("node2", executor=lambda x: x)

        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_edge(n1, n2)

        assert len(graph.edges) == 1
        assert graph.edges[0].source_id == n1.id
        assert graph.edges[0].target_id == n2.id

    def test_add_edge_creates_cycle_fails(self):
        """Adding edge that creates cycle fails."""
        graph = create_graph()
        n1 = node("node1", executor=lambda x: x)
        n2 = node("node2", executor=lambda x: x)

        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_edge(n1, n2)

        with pytest.raises(ValueError, match="cycle"):
            graph.add_edge(n2, n1)

    @pytest.mark.asyncio
    async def test_execute_linear_graph(self):
        """Execute linear graph in order."""
        from locus.multiagent import END, START

        execution_order = []

        async def make_fn(name: str):
            async def fn(inputs: dict[str, Any]) -> str:
                execution_order.append(name)
                return {"output": f"{name}_output"}

            return fn

        graph = create_graph()

        n1 = Node(name="first", executor=await make_fn("first"))
        n2 = Node(name="second", executor=await make_fn("second"))
        n3 = Node(name="third", executor=await make_fn("third"))

        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)
        graph.add_edge(START, n1)
        graph.add_edge(n1, n2)
        graph.add_edge(n2, n3)
        graph.add_edge(n3, END)

        graph.config.parallel = False
        result = await graph.execute({})

        assert result.success
        assert execution_order == ["first", "second", "third"]
        # Check node executed (final_outputs keyed by node id)
        assert n3.id in result.node_results

    @pytest.mark.asyncio
    async def test_execute_parallel_graph(self):
        """Execute independent nodes in parallel."""
        from locus.multiagent import END, START

        graph = create_graph()

        async def slow_fn(inputs: dict[str, Any]) -> dict[str, str]:
            await asyncio.sleep(0.1)
            return {"result": "done"}

        n1 = Node(name="parallel1", executor=slow_fn)
        n2 = Node(name="parallel2", executor=slow_fn)
        n3 = Node(name="final", executor=lambda x: {"final": "yes"})

        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)
        graph.add_edge(START, n1)
        graph.add_edge(START, n2)
        graph.add_edge(n1, n3)
        graph.add_edge(n2, n3)
        graph.add_edge(n3, END)

        graph.config.parallel = True
        result = await graph.execute({})

        assert result.success
        # With parallel execution, n1 and n2 should run concurrently
        # So total time should be ~100ms, not ~200ms
        assert result.duration_ms < 250  # Allow some overhead

    @pytest.mark.asyncio
    async def test_execute_passes_data_between_nodes(self):
        """Data flows between connected nodes."""
        from locus.multiagent import END, START

        graph = create_graph()

        async def producer(inputs: dict[str, Any]) -> dict[str, int]:
            return {"value": 42}

        async def consumer(inputs: dict[str, Any]) -> dict[str, str]:
            # With new API, producer's output keys are merged into state
            value = inputs.get("value", 0)
            return {"result": f"Received: {value}"}

        producer_node = Node(name="producer", executor=producer)
        consumer_node = Node(name="consumer", executor=consumer)

        graph.add_node(producer_node)
        graph.add_node(consumer_node)
        graph.add_edge(START, producer_node)
        graph.add_edge(producer_node, consumer_node)
        graph.add_edge(consumer_node, END)

        result = await graph.execute({})

        assert result.success
        assert result.final_state.get("result") == "Received: 42"


# =============================================================================
# Specialist Tests
# =============================================================================


class TestSpecialist:
    """Tests for Specialist class."""

    def test_create_specialist(self):
        """Create a specialist with configuration."""
        spec = Specialist(
            name="Test Specialist",
            specialist_type="test",
            description="A test specialist",
            system_prompt="You are a test specialist.",
        )

        assert spec.name == "Test Specialist"
        assert spec.specialist_type == "test"

    def test_select_playbook(self):
        """Specialist selects appropriate playbook."""
        playbook1 = Playbook(
            name="Log Analysis",
            description="Analyze application logs",
            steps=[PlaybookStep(instruction="Check error logs")],
        )
        playbook2 = Playbook(
            name="Metrics Analysis",
            description="Analyze system metrics",
            steps=[PlaybookStep(instruction="Check CPU usage")],
        )

        spec = Specialist(
            name="Test",
            specialist_type="test",
            description="Test",
            system_prompt="Test",
            playbooks=[playbook1, playbook2],
        )

        selected = spec.select_playbook("I need to analyze the error logs")
        assert selected is not None
        assert selected.name == "Log Analysis"

    @pytest.mark.asyncio
    async def test_execute_without_model(self):
        """Specialist returns error without model."""
        spec = Specialist(
            name="Test",
            specialist_type="test",
            description="Test",
            system_prompt="Test",
        )

        result = await spec.execute("Analyze this")

        assert not result.success
        assert "No model" in result.error

    @pytest.mark.asyncio
    async def test_execute_with_model(self, mock_model):
        """Specialist executes with model."""
        spec = Specialist(
            name="Test",
            specialist_type="test",
            description="Test",
            system_prompt="Test",
            model=mock_model,
        )

        result = await spec.execute("Analyze this")

        assert result.success
        assert result.output == "Test response"
        mock_model.complete.assert_called_once()

    def test_specialist_with_model(self, mock_model):
        """Test Specialist.with_model returns copy with model."""
        spec = Specialist(
            name="Test",
            specialist_type="test",
            description="Test",
            system_prompt="Test",
        )

        new_spec = spec.with_model(mock_model)

        assert new_spec is not spec
        assert new_spec.model is mock_model
        assert spec.model is None

    def test_playbook_to_prompt(self):
        """Playbook converts to prompt correctly."""
        playbook = Playbook(
            name="Debug Procedure",
            description="Steps to debug an issue",
            preconditions=["System is accessible"],
            steps=[
                PlaybookStep(
                    instruction="Check logs",
                    required_tools=["log_search"],
                    expected_output="Error entries",
                ),
                PlaybookStep(
                    instruction="Analyze errors",
                    on_failure="Escalate to senior",
                ),
            ],
            success_criteria="Root cause identified",
        )

        prompt = playbook.to_prompt()

        assert "Debug Procedure" in prompt
        assert "Steps to debug" in prompt
        assert "System is accessible" in prompt
        assert "Check logs" in prompt
        assert "log_search" in prompt
        assert "Root cause identified" in prompt


# =============================================================================
# Orchestrator Tests
# =============================================================================


class TestOrchestrator:
    """Tests for Orchestrator class."""

    def test_create_orchestrator(self):
        """Create an orchestrator."""
        orch = create_orchestrator(name="Test Orchestrator")

        assert orch.name == "Test Orchestrator"
        assert len(orch.specialists) == 0

    def test_register_specialists(self, mock_model):
        """Register specialists with orchestrator."""
        spec1 = Specialist(
            name="Spec1",
            specialist_type="type1",
            description="First",
            system_prompt="Test",
            model=mock_model,
        )
        spec2 = Specialist(
            name="Spec2",
            specialist_type="type2",
            description="Second",
            system_prompt="Test",
            model=mock_model,
        )

        orch = create_orchestrator(specialists=[spec1, spec2])

        assert len(orch.specialists) == 2
        assert spec1.id in orch.specialists
        assert spec2.id in orch.specialists

    def test_with_model(self, mock_model):
        """Test with_model returns orchestrator copy with model."""
        spec = Specialist(
            name="Spec1",
            specialist_type="type1",
            description="Test",
            system_prompt="Test",
        )
        orch = create_orchestrator(specialists=[spec])

        new_orch = orch.with_model(mock_model)

        assert new_orch is not orch
        assert new_orch.model is mock_model
        # Specialists should also have the model
        for spec in new_orch.specialists.values():
            assert spec.model is mock_model

    @pytest.mark.asyncio
    async def test_execute_invokes_specialists(self, mock_model):
        """Orchestrator invokes specialists."""
        # Configure mock to return routing decision
        mock_model.complete = AsyncMock(
            side_effect=[
                # First call: routing decision
                ModelResponse(
                    message=Message.assistant("""```json
{
    "specialists": ["spec1"],
    "reasoning": "Need log analysis",
    "subtasks": {"spec1": "Analyze logs"}
}
```"""),
                ),
                # Second call: specialist execution
                ModelResponse(
                    message=Message.assistant("Found error in logs"),
                ),
                # Third call: correlation
                ModelResponse(
                    message=Message.assistant("Correlation analysis"),
                ),
                # Fourth call: summary
                ModelResponse(
                    message=Message.assistant("Summary of findings"),
                ),
            ]
        )

        spec = Specialist(
            id="spec1",
            name="Log Analyst",
            specialist_type="log",
            description="Analyzes logs",
            system_prompt="Analyze logs",
            model=mock_model,
        )

        orch = create_orchestrator(
            specialists=[spec],
            model=mock_model,
        )

        result = await orch.execute("Investigate the error")

        assert result.success
        assert "Summary" in result.summary or result.summary is not None

    @pytest.mark.asyncio
    async def test_execute_without_model(self):
        """Orchestrator works without model (invokes all specialists)."""
        mock_model = MagicMock()
        mock_model.complete = AsyncMock(
            return_value=ModelResponse(
                message=Message.assistant("Analysis complete"),
            )
        )

        spec = Specialist(
            id="spec1",
            name="Analyst",
            specialist_type="general",
            description="General analyst",
            system_prompt="Analyze",
            model=mock_model,
        )

        orch = create_orchestrator(specialists=[spec])

        result = await orch.execute("Analyze this")

        # Should invoke all specialists without routing decision
        assert "spec1" in result.specialist_results


class TestRoutingDecision:
    """Tests for RoutingDecision class."""

    def test_create_routing_decision(self):
        """Create a routing decision."""
        decision = RoutingDecision(
            decision_type="invoke",
            specialists=["spec1", "spec2"],
            reasoning="Both specialists needed",
        )

        assert decision.decision_type == "invoke"
        assert len(decision.specialists) == 2


# =============================================================================
# Swarm Tests
# =============================================================================


class TestSwarmTask:
    """Tests for SwarmTask class."""

    def test_create_task(self):
        """Create a swarm task."""
        task = SwarmTask(
            description="Analyze logs",
            priority=5,
        )

        assert task.description == "Analyze logs"
        assert task.priority == 5
        assert task.status == TaskStatus.PENDING


class TestSharedContext:
    """Tests for SharedContext class."""

    @pytest.mark.asyncio
    async def test_add_finding(self):
        """Add finding to shared context."""
        ctx = SharedContext()

        await ctx.add_finding("key1", "value1", "agent1")

        assert ctx.findings["key1"] == "value1"
        assert len(ctx.discovery_log) == 1

    @pytest.mark.asyncio
    async def test_post_to_blackboard(self):
        """Post message to blackboard."""
        ctx = SharedContext()

        await ctx.post_to_blackboard("topic", "message", "agent1")

        assert ctx.blackboard["topic"] == "message"

    def test_get_summary(self):
        """Get context summary."""
        ctx = SharedContext()
        ctx.findings = {"error": "Found issue"}
        ctx.blackboard = {"status": "In progress"}

        summary = ctx.get_summary()

        assert "error" in summary
        assert "Found issue" in summary
        assert "status" in summary


class TestSwarmAgent:
    """Tests for SwarmAgent class."""

    def test_can_handle_with_capabilities(self):
        """Agent can handle task matching capabilities."""
        agent = SwarmAgent(
            name="Log Agent",
            capabilities=["log", "error"],
        )

        task1 = SwarmTask(description="Analyze the error logs")
        task2 = SwarmTask(description="Check metrics")

        assert agent.can_handle(task1)
        assert not agent.can_handle(task2)

    def test_can_handle_generalist(self):
        """Agent without capabilities can handle any task."""
        agent = SwarmAgent(name="Generalist")

        task = SwarmTask(description="Any task")

        assert agent.can_handle(task)

    def test_priority_for_task(self):
        """Calculate priority for task handling."""
        agent = SwarmAgent(
            name="Specialist",
            capabilities=["log", "error", "trace"],
        )

        task1 = SwarmTask(description="Analyze log errors")
        task2 = SwarmTask(description="Check something")

        priority1 = agent.priority_for_task(task1)
        priority2 = agent.priority_for_task(task2)

        assert priority1 > priority2


class TestSwarm:
    """Tests for Swarm class."""

    def test_add_task(self):
        """Add task to swarm queue."""
        swarm = create_swarm()

        task = swarm.add_task("Test task", priority=5)

        assert len(swarm.task_queue) == 1
        assert task.description == "Test task"

    def test_tasks_sorted_by_priority(self):
        """Tasks are sorted by priority."""
        swarm = create_swarm()

        swarm.add_task("Low priority", priority=1)
        swarm.add_task("High priority", priority=10)
        swarm.add_task("Medium priority", priority=5)

        assert swarm.task_queue[0].priority == 10
        assert swarm.task_queue[1].priority == 5
        assert swarm.task_queue[2].priority == 1

    @pytest.mark.asyncio
    async def test_execute_with_agents(self, mock_model):
        """Swarm executes tasks with agents."""
        agent1 = create_swarm_agent(
            name="Agent1",
            capabilities=["analyze"],
            model=mock_model,
        )

        swarm = create_swarm(agents=[agent1], model=mock_model)
        swarm.add_task("Analyze the data")

        result = await swarm.execute()

        assert len(result.completed_tasks) > 0 or len(result.failed_tasks) > 0

    @pytest.mark.asyncio
    async def test_execute_empty_queue(self, mock_model):
        """Swarm handles empty task queue."""
        swarm = create_swarm(model=mock_model)

        result = await swarm.execute()

        assert result.success
        assert len(result.completed_tasks) == 0

    def test_create_swarm_propagates_model_to_agents(self, mock_model):
        """Agents passed to ``create_swarm`` inherit the swarm's model.

        Without this, a caller who builds agents first and then passes them
        and the model into ``create_swarm`` would later get
        "No model configured for agent" at execute time — the original most
        common silent-failure mode for swarm users.
        """
        agent_no_model = create_swarm_agent(name="Worker", capabilities=["analyze"])
        assert agent_no_model.model is None

        swarm = create_swarm(agents=[agent_no_model], model=mock_model)

        assert swarm.agents[0].model is mock_model

    def test_add_agent_inherits_swarm_model(self, mock_model):
        """``swarm.add_agent`` after ``with_model``-style setup propagates
        the swarm's model into the new agent if it doesn't carry one.
        """
        swarm = create_swarm(model=mock_model)
        agent = create_swarm_agent(name="LateJoiner", capabilities=["x"])

        swarm.add_agent(agent)

        assert swarm.agents[0].model is mock_model

    def test_add_agent_preserves_explicit_agent_model(self, mock_model):
        """An agent that already carries a model keeps its own."""
        own_model = MagicMock()
        swarm = create_swarm(model=mock_model)
        agent = create_swarm_agent(name="SelfModeled", capabilities=["x"], model=own_model)

        swarm.add_agent(agent)

        assert swarm.agents[0].model is own_model


# =============================================================================
# Handoff Tests
# =============================================================================


class TestHandoffContext:
    """Tests for HandoffContext class."""

    def test_create_context(self):
        """Create handoff context."""
        ctx = HandoffContext(
            source_agent_id="agent1",
            target_agent_id="agent2",
            reason=HandoffReason.SPECIALIZATION,
            original_task="Investigate issue",
            progress_summary="Found initial symptoms",
            confidence=0.4,
        )

        assert ctx.source_agent_id == "agent1"
        assert ctx.target_agent_id == "agent2"
        assert ctx.reason == HandoffReason.SPECIALIZATION

    def test_to_prompt(self):
        """Convert context to prompt."""
        ctx = HandoffContext(
            source_agent_id="agent1",
            target_agent_id="agent2",
            reason=HandoffReason.ESCALATION,
            original_task="Debug the error",
            progress_summary="Checked logs",
            findings={"error_type": "NullPointer"},
            instructions="Focus on the database layer",
            confidence=0.5,
        )

        prompt = ctx.to_prompt()

        assert "Handoff Context" in prompt
        assert "escalation" in prompt
        assert "Debug the error" in prompt
        assert "error_type" in prompt
        assert "Focus on the database" in prompt


class TestHandoffAgent:
    """Tests for HandoffAgent class."""

    @pytest.mark.asyncio
    async def test_receive_handoff(self, mock_model):
        """Agent receives and processes handoff."""
        agent = create_handoff_agent(
            name="Target Agent",
            system_prompt="You are a specialist.",
            model=mock_model,
        )

        ctx = HandoffContext(
            source_agent_id="source",
            target_agent_id=agent.id,
            reason=HandoffReason.DELEGATION,
            original_task="Complete the analysis",
        )

        result = await agent.receive_handoff(ctx)

        assert result.success
        assert result.target_agent_id == agent.id
        mock_model.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_receive_handoff_with_key_messages(self, mock_model):
        """Agent receives handoff with key messages from context."""
        agent = create_handoff_agent(
            name="Target Agent",
            system_prompt="You are a specialist.",
            model=mock_model,
        )

        ctx = HandoffContext(
            source_agent_id="source",
            target_agent_id=agent.id,
            reason=HandoffReason.DELEGATION,
            original_task="Complete the analysis",
            key_messages=[Message.user("Important context message")],
        )

        result = await agent.receive_handoff(ctx)

        assert result.success
        # Verify key message was included in the call
        call_args = mock_model.complete.call_args
        messages = call_args.kwargs.get("messages", call_args.args[0] if call_args.args else [])
        assert any(m.content == "Important context message" for m in messages)

    @pytest.mark.asyncio
    async def test_receive_handoff_with_tools(self, mock_model):
        """Agent receives handoff with tools."""
        from locus.tools.decorator import tool

        @tool
        def test_tool(x: int) -> str:
            """A test tool."""
            return str(x)

        agent = create_handoff_agent(
            name="Target Agent",
            system_prompt="You are a specialist.",
            model=mock_model,
            tools=[test_tool],
        )

        ctx = HandoffContext(
            source_agent_id="source",
            target_agent_id=agent.id,
            reason=HandoffReason.DELEGATION,
            original_task="Complete the analysis",
        )

        result = await agent.receive_handoff(ctx)

        assert result.success
        # Verify tools were included in the call
        call_args = mock_model.complete.call_args
        tools = call_args.kwargs.get("tools")
        assert tools is not None

    @pytest.mark.asyncio
    async def test_receive_handoff_without_model(self):
        """Handoff fails without model."""
        agent = create_handoff_agent(name="No Model Agent")

        ctx = HandoffContext(
            source_agent_id="source",
            target_agent_id=agent.id,
            reason=HandoffReason.DELEGATION,
            original_task="Task",
        )

        result = await agent.receive_handoff(ctx)

        assert not result.success
        assert "No model" in result.error

    @pytest.mark.asyncio
    async def test_receive_handoff_retries_on_empty_content(self):
        """Empty model response triggers one retry with a directive nudge.

        Some providers return an empty body when the prompt is mostly
        structured headers. Without the retry, the handoff used to silently
        report ``success=True`` with no output.
        """
        model = MagicMock()
        model.complete = AsyncMock(
            side_effect=[
                ModelResponse(
                    message=Message.assistant(""),  # first call: empty
                    usage={"input_tokens": 10, "output_tokens": 0},
                ),
                ModelResponse(
                    message=Message.assistant("Recovered findings."),
                    usage={"input_tokens": 12, "output_tokens": 4},
                ),
            ]
        )
        agent = create_handoff_agent(
            name="Retry Agent",
            system_prompt="You are a specialist.",
            model=model,
        )

        ctx = HandoffContext(
            source_agent_id="source",
            target_agent_id=agent.id,
            reason=HandoffReason.DELEGATION,
            original_task="Task",
        )

        result = await agent.receive_handoff(ctx)

        assert model.complete.await_count == 2
        assert result.output == "Recovered findings."
        assert result.success is True

    @pytest.mark.asyncio
    async def test_receive_handoff_success_false_when_still_empty(self):
        """If the retry also returns empty, ``success`` is False (was True).

        Regression: previously ``success=True`` was hard-coded, so callers
        had to check ``result.output`` themselves to detect the dud.
        """
        model = MagicMock()
        empty = ModelResponse(
            message=Message.assistant(""),
            usage={"input_tokens": 10, "output_tokens": 0},
        )
        model.complete = AsyncMock(side_effect=[empty, empty])
        agent = create_handoff_agent(
            name="Stuck Agent",
            system_prompt="You are a specialist.",
            model=model,
        )

        ctx = HandoffContext(
            source_agent_id="source",
            target_agent_id=agent.id,
            reason=HandoffReason.DELEGATION,
            original_task="Task",
        )

        result = await agent.receive_handoff(ctx)

        assert result.success is False
        assert result.output == ""
        assert result.error is not None
        assert "empty" in result.error.lower()


class TestHandoff:
    """Tests for Handoff manager class."""

    def test_register_agents(self, mock_model):
        """Register agents with handoff manager."""
        agent1 = create_handoff_agent(name="Agent1", model=mock_model)
        agent2 = create_handoff_agent(name="Agent2", model=mock_model)

        manager = create_handoff_manager(agents=[agent1, agent2])

        assert len(manager.agents) == 2
        assert agent1.id in manager.agents
        assert agent2.id in manager.agents

    @pytest.mark.asyncio
    async def test_create_handoff_context(self, mock_model, sample_state):
        """Create handoff context from state."""
        agent1 = create_handoff_agent(name="Source", model=mock_model)
        agent2 = create_handoff_agent(name="Target", model=mock_model)

        manager = create_handoff_manager(agents=[agent1, agent2])

        ctx = await manager.create_handoff(
            source_agent=agent1,
            target_agent_id=agent2.id,
            task="Test task",
            reason=HandoffReason.DELEGATION,
            state=sample_state,
            findings={"key": "value"},
        )

        assert ctx.source_agent_id == agent1.id
        assert ctx.target_agent_id == agent2.id
        assert ctx.findings["key"] == "value"
        assert ctx.confidence == sample_state.confidence

    @pytest.mark.asyncio
    async def test_execute_handoff(self, mock_model, sample_state):
        """Execute complete handoff."""
        agent1 = create_handoff_agent(name="Source", model=mock_model)
        agent2 = create_handoff_agent(name="Target", model=mock_model)

        manager = create_handoff_manager(agents=[agent1, agent2])

        result = await manager.execute_handoff(
            source_agent=agent1,
            target_agent_id=agent2.id,
            task="Continue analysis",
            reason=HandoffReason.SPECIALIZATION,
            state=sample_state,
        )

        assert result.success
        assert result.source_agent_id == agent1.id
        assert result.target_agent_id == agent2.id

    @pytest.mark.asyncio
    async def test_execute_handoff_unknown_target(self, mock_model):
        """Handoff to unknown target fails."""
        agent1 = create_handoff_agent(name="Source", model=mock_model)

        manager = create_handoff_manager(agents=[agent1])

        result = await manager.execute_handoff(
            source_agent=agent1,
            target_agent_id="unknown_agent",
            task="Task",
            reason=HandoffReason.DELEGATION,
        )

        assert not result.success
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_chain_handoff(self, mock_model):
        """Execute chain of handoffs."""
        agent1 = create_handoff_agent(name="First", model=mock_model)
        agent2 = create_handoff_agent(name="Second", model=mock_model)
        agent3 = create_handoff_agent(name="Third", model=mock_model)

        manager = create_handoff_manager(agents=[agent1, agent2, agent3])

        results = await manager.chain_handoff(
            agent_chain=[agent1.id, agent2.id, agent3.id],
            task="Process through chain",
        )

        assert len(results) == 2  # Two handoffs in a chain of 3 agents
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_max_handoff_chain_limit(self, mock_model, sample_state):
        """Handoff chain respects max limit."""
        agents = [create_handoff_agent(name=f"Agent{i}", model=mock_model) for i in range(10)]

        manager = create_handoff_manager(agents=agents, max_chain=3)

        # Simulate existing handoffs
        for i in range(3):
            await manager.create_handoff(
                source_agent=agents[i],
                target_agent_id=agents[i + 1].id,
                task="Task",
                reason=HandoffReason.DELEGATION,
            )

        # This should fail due to chain limit
        result = await manager.execute_handoff(
            source_agent=agents[3],
            target_agent_id=agents[4].id,
            task="Task",
            reason=HandoffReason.DELEGATION,
        )

        assert not result.success
        assert "exceeded" in result.error


# =============================================================================
# Integration Tests
# =============================================================================


class TestMultiAgentIntegration:
    """Integration tests for multi-agent components."""

    @pytest.mark.asyncio
    async def test_graph_with_specialist_nodes(self, mock_model):
        """Graph nodes can wrap specialist execution."""
        spec1 = Specialist(
            name="Analyzer",
            specialist_type="analyzer",
            description="Analyzes data",
            system_prompt="Analyze",
            model=mock_model,
        )
        spec2 = Specialist(
            name="Summarizer",
            specialist_type="summarizer",
            description="Summarizes findings",
            system_prompt="Summarize",
            model=mock_model,
        )

        async def analyze_node(inputs: dict[str, Any]) -> dict[str, Any]:
            result = await spec1.execute(inputs.get("task", ""))
            return {"analysis": result.output}

        async def summarize_node(inputs: dict[str, Any]) -> dict[str, Any]:
            # With new API, state values are flattened
            analysis = inputs.get("analysis", "")
            result = await spec2.execute(f"Summarize: {analysis}")
            return {"summary": result.output}

        from locus.multiagent import END, START

        graph = create_graph()

        analyzer = Node(id="analyzer", name="Analyze", executor=analyze_node)
        summarizer = Node(id="summarizer", name="Summarize", executor=summarize_node)

        graph.add_node(analyzer)
        graph.add_node(summarizer)
        graph.add_edge(START, analyzer)
        graph.add_edge(analyzer, summarizer)
        graph.add_edge(summarizer, END)

        result = await graph.execute({"task": "Analyze the data"})

        assert result.success
        assert mock_model.complete.call_count == 2

    @pytest.mark.asyncio
    async def test_orchestrator_with_handoff(self, mock_model):
        """Orchestrator can hand off to specialists."""
        # This tests the concept - actual implementation would be more complex
        mock_model.complete = AsyncMock(
            return_value=ModelResponse(
                message=Message.assistant("Analysis complete with high confidence"),
            )
        )

        spec = Specialist(
            name="Expert",
            specialist_type="expert",
            description="Domain expert",
            system_prompt="You are an expert",
            model=mock_model,
        )

        orch = create_orchestrator(specialists=[spec], model=mock_model)
        result = await orch.execute("Complex investigation")

        assert result.success
        assert len(result.specialist_results) > 0
