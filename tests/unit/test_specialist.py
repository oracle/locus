# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for specialist agents."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from locus.core.state import AgentState
from locus.multiagent.specialist import (
    Playbook,
    PlaybookStep,
    Specialist,
    SpecialistResult,
    create_code_analyst,
    create_log_analyst,
    create_metrics_analyst,
    create_trace_analyst,
)
from locus.tools.decorator import tool


class TestSpecialistResult:
    """Tests for SpecialistResult dataclass."""

    def test_success_when_no_error(self):
        """Test success property returns True when no error."""
        result = SpecialistResult(
            specialist_id="spec_123",
            specialist_type="log_analyst",
            output="Analysis complete",
            confidence=0.9,
        )
        assert result.success is True

    def test_success_false_when_error(self):
        """Test success property returns False when error exists."""
        result = SpecialistResult(
            specialist_id="spec_123",
            specialist_type="log_analyst",
            error="Model failed",
        )
        assert result.success is False

    def test_default_values(self):
        """Test default values are set correctly."""
        result = SpecialistResult(
            specialist_id="spec_123",
            specialist_type="test",
        )
        assert result.output is None
        assert result.confidence == 0.0
        assert result.duration_ms == 0.0
        assert result.state is None
        assert result.error is None

    def test_with_state(self):
        """Test result with AgentState."""
        state = AgentState(run_id="test")
        result = SpecialistResult(
            specialist_id="spec_123",
            specialist_type="test",
            state=state,
        )
        assert result.state is state


class TestPlaybookStep:
    """Tests for PlaybookStep model."""

    def test_create_minimal_step(self):
        """Test creating step with minimal fields."""
        step = PlaybookStep(instruction="Do something")
        assert step.instruction == "Do something"
        assert step.required_tools == []
        assert step.expected_output is None
        assert step.on_failure is None

    def test_create_full_step(self):
        """Test creating step with all fields."""
        step = PlaybookStep(
            instruction="Search logs",
            required_tools=["grep", "search"],
            expected_output="List of matching entries",
            on_failure="Try broader search",
        )
        assert step.instruction == "Search logs"
        assert step.required_tools == ["grep", "search"]
        assert step.expected_output == "List of matching entries"
        assert step.on_failure == "Try broader search"


class TestPlaybook:
    """Tests for Playbook model."""

    def test_create_minimal_playbook(self):
        """Test creating playbook with minimal fields."""
        playbook = Playbook(
            name="Test Playbook",
            description="A test playbook",
        )
        assert playbook.name == "Test Playbook"
        assert playbook.description == "A test playbook"
        assert playbook.steps == []
        assert playbook.preconditions == []
        assert playbook.success_criteria is None

    def test_create_full_playbook(self):
        """Test creating playbook with all fields."""
        steps = [
            PlaybookStep(instruction="Step 1"),
            PlaybookStep(instruction="Step 2"),
        ]
        playbook = Playbook(
            name="Full Playbook",
            description="Complete playbook",
            steps=steps,
            preconditions=["Condition 1", "Condition 2"],
            success_criteria="All tests pass",
        )
        assert playbook.name == "Full Playbook"
        assert len(playbook.steps) == 2
        assert len(playbook.preconditions) == 2
        assert playbook.success_criteria == "All tests pass"

    def test_to_prompt_minimal(self):
        """Test to_prompt with minimal playbook."""
        playbook = Playbook(
            name="Simple",
            description="A simple playbook",
        )
        prompt = playbook.to_prompt()

        assert "## Playbook: Simple" in prompt
        assert "A simple playbook" in prompt
        assert "### Steps:" in prompt

    def test_to_prompt_with_preconditions(self):
        """Test to_prompt includes preconditions."""
        playbook = Playbook(
            name="With Preconditions",
            description="Description",
            preconditions=["System is running", "User authenticated"],
        )
        prompt = playbook.to_prompt()

        assert "### Preconditions:" in prompt
        assert "- System is running" in prompt
        assert "- User authenticated" in prompt

    def test_to_prompt_with_steps(self):
        """Test to_prompt includes step details."""
        playbook = Playbook(
            name="Steps",
            description="Description",
            steps=[
                PlaybookStep(
                    instruction="Search logs",
                    required_tools=["grep"],
                    expected_output="Log entries",
                    on_failure="Widen search",
                ),
            ],
        )
        prompt = playbook.to_prompt()

        assert "1. Search logs" in prompt
        assert "Tools: grep" in prompt
        assert "Expected: Log entries" in prompt
        assert "On failure: Widen search" in prompt

    def test_to_prompt_with_success_criteria(self):
        """Test to_prompt includes success criteria."""
        playbook = Playbook(
            name="Test",
            description="Description",
            success_criteria="All checks pass",
        )
        prompt = playbook.to_prompt()

        assert "### Success Criteria: All checks pass" in prompt

    def test_to_prompt_multiple_steps(self):
        """Test to_prompt with multiple steps."""
        playbook = Playbook(
            name="Multi",
            description="Description",
            steps=[
                PlaybookStep(instruction="Step 1"),
                PlaybookStep(instruction="Step 2"),
                PlaybookStep(instruction="Step 3"),
            ],
        )
        prompt = playbook.to_prompt()

        assert "1. Step 1" in prompt
        assert "2. Step 2" in prompt
        assert "3. Step 3" in prompt


class TestSpecialist:
    """Tests for Specialist model."""

    def test_create_specialist(self):
        """Test creating specialist."""
        specialist = Specialist(
            name="Test Specialist",
            specialist_type="test",
            description="A test specialist",
            system_prompt="You are helpful",
        )
        assert specialist.name == "Test Specialist"
        assert specialist.specialist_type == "test"
        assert specialist.id.startswith("specialist_")
        assert specialist.model is None

    def test_default_values(self):
        """Test default values."""
        specialist = Specialist(
            name="Test",
            specialist_type="test",
            description="Desc",
            system_prompt="Prompt",
        )
        assert specialist.max_iterations == 10
        assert specialist.confidence_threshold == 0.85
        assert specialist.tools == []
        assert specialist.playbooks == []

    def test_with_model(self):
        """Test with_model returns copy."""
        specialist = Specialist(
            name="Test",
            specialist_type="test",
            description="Desc",
            system_prompt="Prompt",
        )
        mock_model = MagicMock()
        new_specialist = specialist.with_model(mock_model)

        assert new_specialist is not specialist
        assert new_specialist.model is mock_model
        assert specialist.model is None

    def test_build_system_prompt_no_playbook(self):
        """Test _build_system_prompt without playbook."""
        specialist = Specialist(
            name="Log Analyst",
            specialist_type="log",
            description="Analyzes logs",
            system_prompt="Follow procedures",
        )
        prompt = specialist._build_system_prompt("Analyze error logs")

        assert "Log Analyst" in prompt
        assert "Analyzes logs" in prompt
        assert "Follow procedures" in prompt
        assert "Analyze error logs" in prompt
        assert "## Current Task:" in prompt

    def test_build_system_prompt_with_playbook(self):
        """Test _build_system_prompt with playbook."""
        specialist = Specialist(
            name="Test",
            specialist_type="test",
            description="Test specialist",
            system_prompt="Be helpful",
        )
        playbook = Playbook(
            name="Error Analysis",
            description="Find errors in logs",
        )
        prompt = specialist._build_system_prompt("Find errors", playbook)

        assert "## Playbook: Error Analysis" in prompt
        assert "Find errors in logs" in prompt

    def test_select_playbook_no_playbooks(self):
        """Test select_playbook with no playbooks."""
        specialist = Specialist(
            name="Test",
            specialist_type="test",
            description="Desc",
            system_prompt="Prompt",
        )
        result = specialist.select_playbook("some task")
        assert result is None

    def test_select_playbook_single_match(self):
        """Test select_playbook with single matching playbook."""
        playbook = Playbook(
            name="Error Analysis",
            description="Analyze error logs",
        )
        specialist = Specialist(
            name="Test",
            specialist_type="test",
            description="Desc",
            system_prompt="Prompt",
            playbooks=[playbook],
        )
        result = specialist.select_playbook("analyze error logs")
        assert result is playbook

    def test_select_playbook_best_match(self):
        """Test select_playbook selects best match."""
        playbook1 = Playbook(name="Error Analysis", description="Find errors")
        playbook2 = Playbook(name="Performance Check", description="Check performance")
        specialist = Specialist(
            name="Test",
            specialist_type="test",
            description="Desc",
            system_prompt="Prompt",
            playbooks=[playbook1, playbook2],
        )
        # Should match playbook2 better
        result = specialist.select_playbook("check system performance")
        assert result is playbook2

    def test_format_context(self):
        """Test _format_context method."""
        specialist = Specialist(
            name="Test",
            specialist_type="test",
            description="Desc",
            system_prompt="Prompt",
        )
        context = {
            "logs": "Error at line 42",
            "metrics": "CPU 95%",
        }
        result = specialist._format_context(context)

        assert "## Context from previous analysis:" in result
        assert "### logs:" in result
        assert "Error at line 42" in result
        assert "### metrics:" in result
        assert "CPU 95%" in result

    def test_estimate_confidence_high(self):
        """Test _estimate_confidence with high confidence markers."""
        specialist = Specialist(
            name="Test",
            specialist_type="test",
            description="Desc",
            system_prompt="Prompt",
        )
        result = specialist._estimate_confidence(
            "I definitely found the issue. It's clearly a memory leak."
        )
        assert result > 0.5

    def test_estimate_confidence_low(self):
        """Test _estimate_confidence with low confidence markers."""
        specialist = Specialist(
            name="Test",
            specialist_type="test",
            description="Desc",
            system_prompt="Prompt",
        )
        result = specialist._estimate_confidence(
            "I'm uncertain about this. It might be a bug but I'm unsure."
        )
        assert result < 0.5

    def test_estimate_confidence_neutral(self):
        """Test _estimate_confidence with neutral text."""
        specialist = Specialist(
            name="Test",
            specialist_type="test",
            description="Desc",
            system_prompt="Prompt",
        )
        result = specialist._estimate_confidence("The analysis is complete.")
        assert result == 0.5

    def test_estimate_confidence_clamped_max(self):
        """Test confidence is clamped to 1.0."""
        specialist = Specialist(
            name="Test",
            specialist_type="test",
            description="Desc",
            system_prompt="Prompt",
        )
        result = specialist._estimate_confidence(
            "Definitely certainly clearly confirmed verified established"
        )
        assert result == 1.0

    def test_estimate_confidence_clamped_min(self):
        """Test confidence is clamped to 0.0."""
        specialist = Specialist(
            name="Test",
            specialist_type="test",
            description="Desc",
            system_prompt="Prompt",
        )
        result = specialist._estimate_confidence(
            "Might possibly perhaps unclear uncertain unsure need more requires further"
        )
        assert result == 0.0


class TestSpecialistExecution:
    """Tests for Specialist.execute method."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = MagicMock()
        model.complete = AsyncMock()
        return model

    @pytest.mark.asyncio
    async def test_execute_without_model(self):
        """Test execute returns error when no model."""
        specialist = Specialist(
            name="Test",
            specialist_type="test",
            description="Desc",
            system_prompt="Prompt",
        )
        result = await specialist.execute("Do something")

        assert result.success is False
        assert result.error == "No model configured for specialist"
        assert result.specialist_id == specialist.id
        assert result.specialist_type == "test"

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_model):
        """Test successful execution."""
        from locus.core.messages import Message

        mock_response = MagicMock()
        mock_response.message = Message.assistant("Analysis complete. Issue confirmed.")
        mock_model.complete.return_value = mock_response

        specialist = Specialist(
            name="Test",
            specialist_type="test",
            description="Desc",
            system_prompt="Prompt",
            model=mock_model,
        )
        result = await specialist.execute("Analyze this")

        assert result.success is True
        assert result.output == "Analysis complete. Issue confirmed."
        assert result.confidence > 0.5  # "confirmed" increases confidence
        assert result.duration_ms > 0
        assert result.state is not None

    @pytest.mark.asyncio
    async def test_execute_with_context(self, mock_model):
        """Test execution with context."""
        from locus.core.messages import Message

        mock_response = MagicMock()
        mock_response.message = Message.assistant("Done")
        mock_model.complete.return_value = mock_response

        specialist = Specialist(
            name="Test",
            specialist_type="test",
            description="Desc",
            system_prompt="Prompt",
            model=mock_model,
        )
        result = await specialist.execute(
            "Analyze this",
            context={"previous": "Some prior analysis"},
        )

        assert result.success is True
        # Check that complete was called
        mock_model.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_tools(self, mock_model):
        """Test execution with tools."""
        from locus.core.messages import Message

        mock_response = MagicMock()
        mock_response.message = Message.assistant("Done")
        mock_model.complete.return_value = mock_response

        @tool
        def my_tool(x: int) -> str:
            """A tool."""
            return str(x)

        specialist = Specialist(
            name="Test",
            specialist_type="test",
            description="Desc",
            system_prompt="Prompt",
            model=mock_model,
            tools=[my_tool],
        )
        result = await specialist.execute("Use the tool")

        assert result.success is True
        # Verify tools were passed to complete
        call_kwargs = mock_model.complete.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] is not None

    @pytest.mark.asyncio
    async def test_execute_exception(self, mock_model):
        """Test execution handles exceptions."""
        mock_model.complete.side_effect = Exception("Model error")

        specialist = Specialist(
            name="Test",
            specialist_type="test",
            description="Desc",
            system_prompt="Prompt",
            model=mock_model,
        )
        result = await specialist.execute("Do something")

        assert result.success is False
        assert result.error == "Model error"
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_execute_selects_playbook(self, mock_model):
        """Test execution selects appropriate playbook."""
        from locus.core.messages import Message

        mock_response = MagicMock()
        mock_response.message = Message.assistant("Done")
        mock_model.complete.return_value = mock_response

        playbook = Playbook(
            name="Error Analysis",
            description="Analyze errors",
            steps=[PlaybookStep(instruction="Find errors")],
        )
        specialist = Specialist(
            name="Test",
            specialist_type="test",
            description="Desc",
            system_prompt="Prompt",
            model=mock_model,
            playbooks=[playbook],
        )
        await specialist.execute("analyze errors")

        # Verify system prompt includes playbook
        call_kwargs = mock_model.complete.call_args[1]
        messages = call_kwargs["messages"]
        system_msg = next(m for m in messages if m.role.value == "system")
        assert "Error Analysis" in system_msg.content


class TestSpecialistFactories:
    """Tests for specialist factory functions."""

    def test_create_log_analyst(self):
        """Test create_log_analyst factory."""
        specialist = create_log_analyst()
        assert specialist.name == "Log Analyst"
        assert specialist.specialist_type == "log_analyst"
        assert "log" in specialist.description.lower()
        assert specialist.model is None

    def test_create_log_analyst_with_model(self):
        """Test create_log_analyst with model."""
        mock_model = MagicMock()
        specialist = create_log_analyst(model=mock_model)
        assert specialist.model is mock_model

    def test_create_log_analyst_with_tools(self):
        """Test create_log_analyst with tools."""

        @tool
        def search_logs(query: str) -> str:
            """Search logs."""
            return query

        specialist = create_log_analyst(tools=[search_logs])
        assert len(specialist.tools) == 1

    def test_create_metrics_analyst(self):
        """Test create_metrics_analyst factory."""
        specialist = create_metrics_analyst()
        assert specialist.name == "Metrics Analyst"
        assert specialist.specialist_type == "metrics_analyst"
        assert "metrics" in specialist.description.lower()

    def test_create_metrics_analyst_with_model(self):
        """Test create_metrics_analyst with model."""
        mock_model = MagicMock()
        specialist = create_metrics_analyst(model=mock_model)
        assert specialist.model is mock_model

    def test_create_trace_analyst(self):
        """Test create_trace_analyst factory."""
        specialist = create_trace_analyst()
        assert specialist.name == "Trace Analyst"
        assert specialist.specialist_type == "trace_analyst"
        assert "trace" in specialist.description.lower()

    def test_create_trace_analyst_with_model(self):
        """Test create_trace_analyst with model."""
        mock_model = MagicMock()
        specialist = create_trace_analyst(model=mock_model)
        assert specialist.model is mock_model

    def test_create_code_analyst(self):
        """Test create_code_analyst factory."""
        specialist = create_code_analyst()
        assert specialist.name == "Code Analyst"
        assert specialist.specialist_type == "code_analyst"
        assert "code" in specialist.description.lower()

    def test_create_code_analyst_with_model(self):
        """Test create_code_analyst with model."""
        mock_model = MagicMock()
        specialist = create_code_analyst(model=mock_model)
        assert specialist.model is mock_model


class TestSpecialistWithPlaybooks:
    """Tests for Specialist with playbooks integration."""

    def test_specialist_with_multiple_playbooks(self):
        """Test specialist with multiple playbooks."""
        playbooks = [
            Playbook(name="Error Analysis", description="Analyze errors"),
            Playbook(name="Performance Check", description="Check performance"),
            Playbook(name="Security Audit", description="Audit security"),
        ]
        specialist = Specialist(
            name="Multi",
            specialist_type="multi",
            description="Multi-playbook",
            system_prompt="Prompt",
            playbooks=playbooks,
        )
        assert len(specialist.playbooks) == 3

    def test_select_playbook_no_match(self):
        """Test select_playbook when no keywords match."""
        playbook = Playbook(name="Error Analysis", description="Find errors")
        specialist = Specialist(
            name="Test",
            specialist_type="test",
            description="Desc",
            system_prompt="Prompt",
            playbooks=[playbook],
        )
        # Task with no matching keywords
        result = specialist.select_playbook("completely unrelated task xyz")
        # Returns None when score is 0
        assert result is None

    def test_select_playbook_case_insensitive(self):
        """Test select_playbook is case insensitive."""
        playbook = Playbook(name="ERROR Analysis", description="Find ERRORS")
        specialist = Specialist(
            name="Test",
            specialist_type="test",
            description="Desc",
            system_prompt="Prompt",
            playbooks=[playbook],
        )
        result = specialist.select_playbook("analyze error logs")
        assert result is playbook


class TestSpecialistIdGeneration:
    """Tests for Specialist ID generation."""

    def test_unique_ids(self):
        """Test that each specialist gets a unique ID."""
        specialists = [
            Specialist(
                name="Test",
                specialist_type="test",
                description="Desc",
                system_prompt="Prompt",
            )
            for _ in range(5)
        ]
        ids = [s.id for s in specialists]
        assert len(set(ids)) == 5  # All unique

    def test_id_format(self):
        """Test ID format."""
        specialist = Specialist(
            name="Test",
            specialist_type="test",
            description="Desc",
            system_prompt="Prompt",
        )
        assert specialist.id.startswith("specialist_")
        assert len(specialist.id) == len("specialist_") + 8  # 8 hex chars
