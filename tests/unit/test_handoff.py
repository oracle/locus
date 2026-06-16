# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for multiagent handoff module."""

from datetime import datetime

from locus.core.messages import Message
from locus.multiagent.handoff import (
    HandoffContext,
    HandoffEvent,
    HandoffReason,
)


class TestHandoffReason:
    """Tests for HandoffReason enum."""

    def test_all_reasons(self):
        """Test all handoff reasons exist."""
        assert HandoffReason.SPECIALIZATION == "specialization"
        assert HandoffReason.ESCALATION == "escalation"
        assert HandoffReason.DELEGATION == "delegation"
        assert HandoffReason.COMPLETION == "completion"
        assert HandoffReason.FAILURE == "failure"


class TestHandoffEvent:
    """Tests for HandoffEvent."""

    def test_create_event(self):
        """Test creating handoff event."""
        event = HandoffEvent(
            source_agent_id="agent1",
            target_agent_id="agent2",
            reason=HandoffReason.DELEGATION,
        )
        assert event.event_type == "handoff"
        assert event.source_agent_id == "agent1"
        assert event.target_agent_id == "agent2"
        assert event.reason == HandoffReason.DELEGATION
        assert event.context_summary is None

    def test_event_with_summary(self):
        """Test event with context summary."""
        event = HandoffEvent(
            source_agent_id="agent1",
            target_agent_id="agent2",
            reason=HandoffReason.ESCALATION,
            context_summary="Need supervisor help",
        )
        assert event.context_summary == "Need supervisor help"


class TestHandoffContext:
    """Tests for HandoffContext."""

    def test_create_minimal_context(self):
        """Test creating context with minimal fields."""
        ctx = HandoffContext(
            source_agent_id="agent1",
            target_agent_id="agent2",
            reason=HandoffReason.DELEGATION,
            original_task="Complete the task",
        )
        assert ctx.source_agent_id == "agent1"
        assert ctx.target_agent_id == "agent2"
        assert ctx.reason == HandoffReason.DELEGATION
        assert ctx.original_task == "Complete the task"
        assert ctx.handoff_id.startswith("handoff_")

    def test_create_full_context(self):
        """Test creating context with all fields."""
        ctx = HandoffContext(
            source_agent_id="agent1",
            target_agent_id="agent2",
            reason=HandoffReason.SPECIALIZATION,
            original_task="Analyze data",
            conversation_summary="Discussed data analysis",
            key_messages=[Message(role="user", content="Help me")],
            state_snapshot={"key": "value"},
            findings={"result": "found"},
            progress_summary="50% done",
            confidence=0.75,
            instructions="Focus on X",
            handoff_chain=["agent0"],
        )
        assert ctx.conversation_summary == "Discussed data analysis"
        assert len(ctx.key_messages) == 1
        assert ctx.state_snapshot == {"key": "value"}
        assert ctx.findings == {"result": "found"}
        assert ctx.confidence == 0.75
        assert ctx.instructions == "Focus on X"

    def test_handoff_id_unique(self):
        """Test handoff IDs are unique."""
        ctx1 = HandoffContext(
            source_agent_id="a1",
            target_agent_id="a2",
            reason=HandoffReason.DELEGATION,
            original_task="Task",
        )
        ctx2 = HandoffContext(
            source_agent_id="a1",
            target_agent_id="a2",
            reason=HandoffReason.DELEGATION,
            original_task="Task",
        )
        assert ctx1.handoff_id != ctx2.handoff_id

    def test_created_at_set(self):
        """Test created_at is set automatically."""
        ctx = HandoffContext(
            source_agent_id="a1",
            target_agent_id="a2",
            reason=HandoffReason.DELEGATION,
            original_task="Task",
        )
        assert ctx.created_at is not None
        assert isinstance(ctx.created_at, datetime)

    def test_to_prompt(self):
        """Test converting context to prompt."""
        ctx = HandoffContext(
            source_agent_id="agent1",
            target_agent_id="agent2",
            reason=HandoffReason.DELEGATION,
            original_task="Analyze the data",
            progress_summary="Started analysis",
            instructions="Focus on trends",
        )
        prompt = ctx.to_prompt()

        assert isinstance(prompt, str)
        assert "Analyze the data" in prompt
        assert "agent1" in prompt or "DELEGATION" in prompt

    def test_defaults(self):
        """Test default values."""
        ctx = HandoffContext(
            source_agent_id="a1",
            target_agent_id="a2",
            reason=HandoffReason.COMPLETION,
            original_task="Task",
        )
        assert ctx.conversation_summary is None
        assert ctx.key_messages == []
        assert ctx.state_snapshot == {}
        assert ctx.findings == {}
        assert ctx.progress_summary is None
        assert ctx.confidence == 0.0
        assert ctx.instructions is None
        assert ctx.handoff_chain == []


class TestHandoffResult:
    """Tests for HandoffResult."""

    def test_create_success_result(self):
        """Test creating successful handoff result."""
        from locus.multiagent.handoff import HandoffResult

        result = HandoffResult(
            handoff_id="handoff_123",
            success=True,
            source_agent_id="agent1",
            target_agent_id="agent2",
            output="Task completed successfully",
            final_confidence=0.95,
            duration_ms=1500.0,
        )
        assert result.success is True
        assert result.output == "Task completed successfully"
        assert result.final_confidence == 0.95
        assert result.error is None

    def test_create_failure_result(self):
        """Test creating failed handoff result."""
        from locus.multiagent.handoff import HandoffResult

        result = HandoffResult(
            handoff_id="handoff_456",
            success=False,
            source_agent_id="agent1",
            target_agent_id="agent2",
            error="Agent not found",
            duration_ms=100.0,
        )
        assert result.success is False
        assert result.error == "Agent not found"
        assert result.output is None


class TestHandoffAgent:
    """Tests for HandoffAgent."""

    def test_create_agent(self):
        """Test creating handoff agent."""
        from locus.multiagent.handoff import HandoffAgent

        agent = HandoffAgent(
            name="Test Agent",
            description="A test agent",
            system_prompt="You are a helpful agent",
        )
        assert agent.name == "Test Agent"
        assert agent.description == "A test agent"
        assert agent.id.startswith("agent_")

    def test_agent_with_model(self):
        """Test with_model returns copy."""
        from unittest.mock import MagicMock

        from locus.multiagent.handoff import HandoffAgent

        agent = HandoffAgent(
            name="Agent",
            description="Test",
            system_prompt="System prompt",
        )
        mock_model = MagicMock()
        new_agent = agent.with_model(mock_model)

        assert new_agent is not agent
        assert new_agent.model is mock_model


class TestHandoff:
    """Tests for Handoff manager."""

    def test_create_manager(self):
        """Test creating handoff manager."""
        from locus.multiagent.handoff import Handoff

        manager = Handoff()
        assert manager.id.startswith("handoff_mgr_")
        assert len(manager.agents) == 0

    def test_register_agent(self):
        """Test registering agents."""
        from locus.multiagent.handoff import Handoff, HandoffAgent

        manager = Handoff()
        agent = HandoffAgent(
            name="Test",
            description="Test agent",
            system_prompt="Prompt",
        )
        manager.register_agent(agent)

        assert agent.id in manager.agents
        assert manager.agents[agent.id] is agent

    def test_register_agents(self):
        """Test registering multiple agents."""
        from locus.multiagent.handoff import Handoff, HandoffAgent

        manager = Handoff()
        agents = [
            HandoffAgent(name="A1", description="Agent 1", system_prompt="P1"),
            HandoffAgent(name="A2", description="Agent 2", system_prompt="P2"),
        ]
        manager.register_agents(agents)

        assert len(manager.agents) == 2


class TestCreateHandoffManager:
    """Tests for create_handoff_manager function."""

    def test_create_empty(self):
        """Test creating empty manager."""
        from locus.multiagent.handoff import create_handoff_manager

        manager = create_handoff_manager()
        assert len(manager.agents) == 0
        assert manager.max_handoff_chain == 5

    def test_create_with_agents(self):
        """Test creating manager with agents."""
        from locus.multiagent.handoff import HandoffAgent, create_handoff_manager

        agents = [
            HandoffAgent(name="A1", description="Test", system_prompt="P"),
        ]
        manager = create_handoff_manager(agents=agents)

        assert len(manager.agents) == 1

    def test_create_with_max_chain(self):
        """Test creating with custom max chain."""
        from locus.multiagent.handoff import create_handoff_manager

        manager = create_handoff_manager(max_chain=10)
        assert manager.max_handoff_chain == 10


class TestCreateHandoffAgent:
    """Tests for create_handoff_agent function."""

    def test_create_minimal(self):
        """Test creating minimal agent."""
        from locus.multiagent.handoff import create_handoff_agent

        agent = create_handoff_agent(name="Test")
        assert agent.name == "Test"
        assert agent.description == ""

    def test_create_full(self):
        """Test creating agent with all options."""
        from locus.multiagent.handoff import create_handoff_agent
        from locus.tools.decorator import tool

        @tool
        def my_tool(x: int) -> str:
            """A tool."""
            return str(x)

        agent = create_handoff_agent(
            name="Full Agent",
            description="A fully configured agent",
            system_prompt="You are helpful",
            tools=[my_tool],
        )
        assert agent.name == "Full Agent"
        assert agent.description == "A fully configured agent"
        assert len(agent.tools) == 1


class TestHandoffAgentConfidenceEstimation:
    """Tests for HandoffAgent confidence estimation."""

    def test_confidence_increase_with_solved(self):
        """Test confidence increases for positive keywords."""
        from locus.multiagent.handoff import HandoffAgent

        agent = HandoffAgent(
            name="Test",
            description="Test agent",
            system_prompt="System",
        )

        result = agent._estimate_confidence("Problem solved successfully", 0.5)
        assert result > 0.5

    def test_confidence_decrease_with_uncertain(self):
        """Test confidence decreases for uncertain keywords."""
        from locus.multiagent.handoff import HandoffAgent

        agent = HandoffAgent(
            name="Test",
            description="Test agent",
            system_prompt="System",
        )

        result = agent._estimate_confidence("I'm unclear about this", 0.5)
        assert result < 0.5

    def test_confidence_slight_increase_default(self):
        """Test default confidence increase."""
        from locus.multiagent.handoff import HandoffAgent

        agent = HandoffAgent(
            name="Test",
            description="Test agent",
            system_prompt="System",
        )

        result = agent._estimate_confidence("Normal response", 0.5)
        assert result == 0.6  # Base + 0.1

    def test_confidence_capped_at_one(self):
        """Test confidence is capped at 1.0."""
        from locus.multiagent.handoff import HandoffAgent

        agent = HandoffAgent(
            name="Test",
            description="Test agent",
            system_prompt="System",
        )

        result = agent._estimate_confidence("Fully resolved and confirmed", 0.95)
        assert result == 1.0


class TestHandoffManagerExtractKeyMessages:
    """Tests for Handoff manager key message extraction."""

    def test_extract_short_conversation(self):
        """Test extracting from short conversation."""
        from locus.core.state import AgentState
        from locus.multiagent.handoff import Handoff

        manager = Handoff()

        state = AgentState(
            run_id="test",
            messages=[Message(role="user", content="Hello")],
        )

        result = manager._extract_key_messages(state, max_messages=5)
        assert len(result) == 1

    def test_extract_preserves_system_message(self):
        """Test that system message is preserved."""
        from locus.core.state import AgentState
        from locus.multiagent.handoff import Handoff

        manager = Handoff()

        state = AgentState(
            run_id="test",
            messages=[
                Message(role="system", content="You are helpful"),
                Message(role="user", content="Hello 1"),
                Message(role="assistant", content="Hi 1"),
                Message(role="user", content="Hello 2"),
                Message(role="assistant", content="Hi 2"),
                Message(role="user", content="Hello 3"),
                Message(role="assistant", content="Hi 3"),
            ],
        )

        result = manager._extract_key_messages(state, max_messages=3)

        # Should have system + last 3
        assert any(m.role.value == "system" for m in result)
        assert len(result) <= 4  # System + 3 messages


class TestHandoffManagerSummarizeConversation:
    """Tests for conversation summarization."""

    def test_summarize_simple_conversation(self):
        """Test summarizing a simple conversation."""
        from locus.multiagent.handoff import Handoff

        manager = Handoff()

        messages = [
            Message(role="user", content="What is 2+2?"),
            Message(role="assistant", content="2+2 equals 4"),
        ]

        result = manager._summarize_conversation(messages)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_summarize_empty_conversation(self):
        """Test summarizing empty conversation."""
        from locus.multiagent.handoff import Handoff

        manager = Handoff()

        result = manager._summarize_conversation([])

        assert result == ""


class TestHandoffContextToPrompt:
    """Tests for HandoffContext.to_prompt method."""

    def test_to_prompt_includes_task(self):
        """Test to_prompt includes original task."""
        ctx = HandoffContext(
            source_agent_id="agent1",
            target_agent_id="agent2",
            reason=HandoffReason.DELEGATION,
            original_task="Complete the analysis",
        )

        prompt = ctx.to_prompt()

        assert "Complete the analysis" in prompt

    def test_to_prompt_includes_instructions(self):
        """Test to_prompt includes instructions."""
        ctx = HandoffContext(
            source_agent_id="agent1",
            target_agent_id="agent2",
            reason=HandoffReason.DELEGATION,
            original_task="Task",
            instructions="Focus on X specifically",
        )

        prompt = ctx.to_prompt()

        assert "Focus on X specifically" in prompt

    def test_to_prompt_includes_progress(self):
        """Test to_prompt includes progress summary."""
        ctx = HandoffContext(
            source_agent_id="agent1",
            target_agent_id="agent2",
            reason=HandoffReason.DELEGATION,
            original_task="Task",
            progress_summary="50% complete",
        )

        prompt = ctx.to_prompt()

        assert "50% complete" in prompt
