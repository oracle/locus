# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for notebooks 13-21.

Tests validate that all notebook examples work correctly.
"""

from __future__ import annotations

import os
import tempfile

import pytest


# Skip all tests if no model is available
pytestmark = pytest.mark.integration


def has_model_available() -> bool:
    """Check if a model is available for testing."""
    # Check for standard API keys
    if os.environ.get("OPENAI_API_KEY"):
        return True
    if os.environ.get("MODEL_PROVIDER"):
        return True

    return False


# =============================================================================
# Notebook 14: Structured Output Tests
# =============================================================================


class TestNotebook13StructuredOutput:
    """Tests for Notebook 14: Structured Output."""

    def test_json_extraction_plain_text(self):
        """Test extracting JSON from plain text."""
        import json

        from tulip.core.structured import extract_json

        raw_text = '{"name": "Alice", "age": 30}'
        result = extract_json(raw_text)
        # extract_json returns a string, not a dict
        assert json.loads(result) == {"name": "Alice", "age": 30}

    def test_json_extraction_markdown(self):
        """Test extracting JSON from markdown code blocks."""
        import json

        from tulip.core.structured import extract_json

        markdown = """Here's the data:
```json
{"key": "value"}
```
"""
        result = extract_json(markdown)
        # extract_json returns a string, not a dict
        assert json.loads(result) == {"key": "value"}

    def test_parse_pydantic_success(self):
        """Test parsing into Pydantic model."""
        from pydantic import BaseModel

        from tulip.core.structured import parse_structured

        class Person(BaseModel):
            name: str
            age: int

        content = '{"name": "Bob", "age": 25}'
        result = parse_structured(content, Person, strict=False)

        assert result.success is True
        assert result.parsed.name == "Bob"
        assert result.parsed.age == 25

    def test_parse_pydantic_failure(self):
        """Test handling parse failures."""
        from pydantic import BaseModel

        from tulip.core.structured import parse_structured

        class Person(BaseModel):
            name: str
            age: int

        content = "not valid json"
        result = parse_structured(content, Person, strict=False)

        assert result.success is False
        assert result.error is not None

    def test_schema_prompt_generation(self):
        """Test schema prompt generation."""
        from pydantic import BaseModel, Field

        from tulip.core.structured import create_schema_prompt

        class Task(BaseModel):
            name: str = Field(..., description="Task name")
            done: bool = Field(default=False)

        prompt = create_schema_prompt(Task)
        assert "name" in prompt
        assert "Task name" in prompt


# =============================================================================
# Notebook 15: Reasoning Patterns Tests
# =============================================================================


class TestNotebook14ReasoningPatterns:
    """Tests for Notebook 15: Reasoning Patterns."""

    def test_reflector_assessment(self):
        """Test Reflector evaluation."""
        from tulip.core.state import AgentState, ToolExecution
        from tulip.reasoning import AssessmentCategory, Reflector

        reflector = Reflector(loop_threshold=3)
        state = AgentState(agent_id="test")

        # Add a successful execution (tool_call_id is required)
        execution = ToolExecution(
            tool_name="search",
            tool_call_id="call_001",
            arguments={"q": "test"},
            result="Found data",
        )
        state = state.with_tool_execution(execution)

        result = reflector.reflect(state)
        assert result.assessment in list(AssessmentCategory)

    def test_reflector_loop_detection(self):
        """Test loop detection in Reflector."""
        from tulip.core.messages import ToolCall
        from tulip.core.state import AgentState, ReasoningStep, ToolExecution
        from tulip.reasoning import AssessmentCategory, Reflector

        reflector = Reflector(loop_threshold=3)
        state = AgentState(agent_id="test")

        # Add repeated tool calls across iterations (with reasoning_steps)
        for i in range(4):
            step = ReasoningStep(
                iteration=i + 1,
                thought=f"Call {i}",
                tool_calls=[ToolCall(name="same_tool", arguments={})],
            )
            state = state.with_reasoning_step(step)
            execution = ToolExecution(
                tool_name="same_tool",
                tool_call_id=f"call_{i}",
                arguments={},
            )
            state = state.with_tool_execution(execution)
            state = state.next_iteration()

        result = reflector.reflect(state)
        assert result.assessment == AssessmentCategory.LOOP_DETECTED
        assert result.loop_pattern is not None

    def test_grounding_evaluator(self):
        """Test grounding evaluation."""
        from tulip.reasoning import GroundingEvaluator

        evaluator = GroundingEvaluator(replan_threshold=0.5)

        claims = ["The sky is blue", "Water is wet"]
        evidence = ["The sky appears blue due to scattering", "Water makes things wet"]

        result = evaluator.evaluate(claims, evidence)
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.claims, list)

    def test_grounding_convenience_function(self):
        """Test evaluate_grounding convenience function."""
        from tulip.reasoning import evaluate_grounding

        result = evaluate_grounding(
            claims=["Test claim"],
            evidence=["Test evidence about Test claim"],
            threshold=0.5,
        )
        assert hasattr(result, "score")
        assert hasattr(result, "requires_replan")

    def test_causal_chain_creation(self):
        """Test creating causal chains."""
        from tulip.reasoning import CausalChain, NodeType, RelationshipType

        chain = CausalChain()

        node1 = chain.create_node(label="Root cause", node_type=NodeType.ROOT_CAUSE)
        node2 = chain.create_node(label="Effect")

        chain.link(node1.id, node2.id, relationship=RelationshipType.CAUSES)

        assert len(chain.nodes) == 2
        assert len(chain.edges) == 1

    def test_causal_chain_root_cause_detection(self):
        """Test identifying root causes."""
        from tulip.reasoning import CausalChain, RelationshipType

        chain = CausalChain()

        root = chain.create_node(label="Root")
        middle = chain.create_node(label="Middle")
        symptom = chain.create_node(label="Symptom")

        chain.link(root.id, middle.id, relationship=RelationshipType.CAUSES)
        chain.link(middle.id, symptom.id, relationship=RelationshipType.CAUSES)

        root_causes = chain.identify_root_causes()
        symptoms = chain.identify_symptoms()

        assert len(root_causes) == 1
        assert root_causes[0].label == "Root"
        assert len(symptoms) == 1
        assert symptoms[0].label == "Symptom"

    def test_build_causal_chain(self):
        """Test building chain from events."""
        from tulip.reasoning import build_causal_chain

        events = [
            {"label": "Error occurred"},
            {"label": "Service crashed", "causes": ["Error occurred"]},
        ]

        chain = build_causal_chain(events, auto_classify=True)
        assert len(chain.nodes) == 2
        assert len(chain.edges) == 1


# =============================================================================
# Notebook 16: Playbooks Tests
# =============================================================================


class TestNotebook15Playbooks:
    """Tests for Notebook 16: Playbooks."""

    def test_playbook_step_creation(self):
        """Test creating playbook steps."""
        from tulip.playbooks import PlaybookStep

        step = PlaybookStep(
            id="step_1",
            description="Test step",
            expected_tools=["tool_a", "tool_b"],
            required=True,
        )

        assert step.id == "step_1"
        assert len(step.expected_tools) == 2
        assert step.required is True

    def test_playbook_creation(self):
        """Test creating playbooks."""
        from tulip.playbooks import Playbook, PlaybookStep

        steps = [
            PlaybookStep(id="s1", description="Step 1"),
            PlaybookStep(id="s2", description="Step 2"),
        ]

        playbook = Playbook(
            id="test_playbook",
            name="Test Playbook",
            steps=steps,
            strict_sequence=True,
        )

        assert playbook.id == "test_playbook"
        assert len(playbook.steps) == 2
        assert playbook.strict_sequence is True

    def test_playbook_get_step(self):
        """Test getting step by ID."""
        from tulip.playbooks import Playbook, PlaybookStep

        playbook = Playbook(
            id="test",
            name="Test",
            steps=[
                PlaybookStep(id="first", description="First step"),
                PlaybookStep(id="second", description="Second step"),
            ],
        )

        step = playbook.get_step("first")
        assert step is not None
        assert step.description == "First step"

        assert playbook.get_step("nonexistent") is None

    def test_playbook_plan_progress(self):
        """Test playbook plan progress tracking."""
        from tulip.playbooks import Playbook, PlaybookPlan, PlaybookStep, StepExecution, StepStatus

        playbook = Playbook(
            id="test",
            name="Test",
            steps=[
                PlaybookStep(id="s1", description="Step 1"),
                PlaybookStep(id="s2", description="Step 2"),
            ],
        )

        plan = PlaybookPlan(playbook=playbook)
        assert plan.progress == 0.0

        plan.step_executions["s1"] = StepExecution(
            step_id="s1",
            status=StepStatus.COMPLETED,
        )

        assert plan.progress == 0.5
        assert "s1" in plan.completed_steps


# =============================================================================
# Notebook 17: Agent Handoff Tests
# =============================================================================


class TestNotebook16AgentHandoff:
    """Tests for Notebook 17: Agent Handoff."""

    def test_handoff_agent_creation(self):
        """Test creating handoff agents."""
        from tulip.multiagent.handoff import create_handoff_agent

        agent = create_handoff_agent(
            name="Test Agent",
            description="Test description",
            system_prompt="You are a test agent.",
        )

        assert agent.name == "Test Agent"
        assert agent.description == "Test description"

    def test_handoff_context_creation(self):
        """Test creating handoff context."""
        from tulip.multiagent.handoff import HandoffContext, HandoffReason

        context = HandoffContext(
            source_agent_id="agent_1",
            target_agent_id="agent_2",
            reason=HandoffReason.SPECIALIZATION,
            original_task="Test task",
            conversation_summary="Test summary",
        )

        assert context.source_agent_id == "agent_1"
        assert context.reason == HandoffReason.SPECIALIZATION

    def test_handoff_context_to_prompt(self):
        """Test converting context to prompt."""
        from tulip.multiagent.handoff import HandoffContext, HandoffReason

        context = HandoffContext(
            source_agent_id="agent_1",
            target_agent_id="agent_2",
            reason=HandoffReason.ESCALATION,
            original_task="Investigate issue",
            findings={"error_count": 42},
        )

        prompt = context.to_prompt()
        assert "ESCALATION" in prompt or "escalation" in prompt.lower()
        assert "Investigate issue" in prompt

    def test_handoff_reasons(self):
        """Test handoff reason enum."""
        from tulip.multiagent.handoff import HandoffReason

        reasons = list(HandoffReason)
        assert len(reasons) >= 4
        assert HandoffReason.SPECIALIZATION in reasons
        assert HandoffReason.ESCALATION in reasons


# =============================================================================
# Notebook 18: Orchestrator Pattern Tests
# =============================================================================


class TestNotebook17OrchestratorPattern:
    """Tests for Notebook 18: Orchestrator Pattern."""

    def test_specialist_creation(self):
        """Test creating specialists."""
        from tulip.multiagent.specialist import Specialist

        specialist = Specialist(
            name="Test Specialist",
            specialist_type="test",
            description="Test description",
            system_prompt="You are a test specialist.",
        )

        assert specialist.name == "Test Specialist"
        assert specialist.specialist_type == "test"

    def test_prebuilt_specialists(self):
        """Test pre-built specialist factories."""
        from tulip.multiagent.specialist import (
            create_code_analyst,
            create_log_analyst,
            create_metrics_analyst,
            create_trace_analyst,
        )

        log = create_log_analyst()
        metrics = create_metrics_analyst()
        trace = create_trace_analyst()
        code = create_code_analyst()

        assert log.specialist_type == "log_analyst"
        assert metrics.specialist_type == "metrics_analyst"
        assert trace.specialist_type == "trace_analyst"
        assert code.specialist_type == "code_analyst"

    def test_routing_decision(self):
        """Test routing decision creation."""
        from tulip.multiagent import RoutingDecision

        decision = RoutingDecision(
            decision_type="invoke",
            specialists=["log_analyst", "metrics_analyst"],
            reasoning="Need both for comprehensive analysis",
        )

        assert decision.decision_type == "invoke"
        assert len(decision.specialists) == 2


# =============================================================================
# Notebook 19: Specialist Agents Tests
# =============================================================================


class TestNotebook18SpecialistAgents:
    """Tests for Notebook 19: Specialist Agents."""

    def test_specialist_playbook(self):
        """Test specialist playbooks."""
        from tulip.multiagent.specialist import Playbook, PlaybookStep

        playbook = Playbook(
            name="Test Playbook",
            description="Test procedure",
            steps=[
                PlaybookStep(instruction="Step 1"),
                PlaybookStep(instruction="Step 2", required_tools=["tool_a"]),
            ],
            success_criteria="All steps complete",
        )

        assert playbook.name == "Test Playbook"
        assert len(playbook.steps) == 2

    def test_playbook_to_prompt(self):
        """Test playbook prompt generation."""
        from tulip.multiagent.specialist import Playbook, PlaybookStep

        playbook = Playbook(
            name="Debug Procedure",
            description="Standard debugging steps",
            preconditions=["Logs available"],
            steps=[
                PlaybookStep(instruction="Check logs"),
            ],
        )

        prompt = playbook.to_prompt()
        assert "Debug Procedure" in prompt
        assert "Check logs" in prompt

    def test_specialist_with_playbooks(self):
        """Test specialist with playbook selection."""
        from tulip.multiagent.specialist import Playbook, PlaybookStep, Specialist

        playbook1 = Playbook(
            name="Performance Analysis",
            description="Analyze performance issues",
            steps=[PlaybookStep(instruction="Check metrics")],
        )

        playbook2 = Playbook(
            name="Error Investigation",
            description="Investigate errors",
            steps=[PlaybookStep(instruction="Check logs")],
        )

        specialist = Specialist(
            name="Test",
            specialist_type="test",
            description="Test",
            system_prompt="Test",
            playbooks=[playbook1, playbook2],
        )

        # Should select based on keywords
        selected = specialist.select_playbook("Check performance metrics")
        assert selected is not None


# =============================================================================
# Notebook 20: Guardrails & Security Tests
# =============================================================================


class TestNotebook19GuardrailsSecurity:
    """Tests for Notebook 20: Guardrails & Security."""

    def test_guardrail_config(self):
        """Test guardrail configuration."""
        from tulip.hooks.builtin.guardrails import GuardrailAction, GuardrailConfig

        config = GuardrailConfig(
            block_dangerous_tools=frozenset({"exec", "eval"}),
            max_prompt_length=10000,
            default_action=GuardrailAction.BLOCK,
        )

        assert "exec" in config.block_dangerous_tools
        assert config.max_prompt_length == 10000
        assert config.default_action == GuardrailAction.BLOCK

    @pytest.mark.asyncio
    async def test_guardrails_hook_tool_blocking(self):
        """Test tool blocking in guardrails."""
        from tulip.core.events import BeforeToolCallEvent
        from tulip.hooks.builtin.guardrails import GuardrailConfig, GuardrailsHook

        config = GuardrailConfig(
            block_dangerous_tools=frozenset({"dangerous_tool"}),
        )

        hook = GuardrailsHook(config=config)

        # Safe tool should pass
        await hook.on_before_tool_call(BeforeToolCallEvent(tool_name="safe_tool", arguments={}))

        # Dangerous tool should be blocked
        with pytest.raises(ValueError):
            await hook.on_before_tool_call(
                BeforeToolCallEvent(tool_name="dangerous_tool", arguments={})
            )

    def test_guardrail_actions(self):
        """Test guardrail action types."""
        from tulip.hooks.builtin.guardrails import GuardrailAction

        actions = list(GuardrailAction)
        assert GuardrailAction.BLOCK in actions
        assert GuardrailAction.WARN in actions
        assert GuardrailAction.REDACT in actions

    @pytest.mark.asyncio
    async def test_content_filter_hook(self):
        """Test content filter hook."""
        from tulip.core.state import AgentState
        from tulip.hooks.builtin.guardrails import ContentFilterHook

        hook = ContentFilterHook(
            blocked_words=["forbidden"],
            max_input_length=100,
        )

        state = AgentState(agent_id="test")

        # Normal input should pass
        await hook.on_before_invocation("Hello world", state)

        # Blocked word should raise
        with pytest.raises(ValueError):
            await hook.on_before_invocation("This is forbidden", state)


# =============================================================================
# Notebook 21: Checkpoint Backends Tests
# =============================================================================


class TestNotebook20CheckpointBackends:
    """Tests for Notebook 21: Checkpoint Backends."""

    @pytest.mark.asyncio
    async def test_memory_checkpointer(self):
        """Test in-memory checkpointer with AgentState."""
        from tulip.core.state import AgentState
        from tulip.memory.backends.memory import MemoryCheckpointer

        checkpointer = MemoryCheckpointer()

        # Create a state
        state = AgentState(agent_id="test_agent")

        # Save and load
        checkpoint_id = await checkpointer.save(state, "test_thread")
        assert checkpoint_id is not None

        loaded = await checkpointer.load("test_thread")
        assert loaded is not None
        assert loaded.agent_id == "test_agent"

        # List threads
        threads = await checkpointer.list_threads()
        assert "test_thread" in threads

        # Delete
        deleted = await checkpointer.delete("test_thread")
        assert deleted is True

    @pytest.mark.asyncio
    async def test_file_checkpointer(self):
        """Test file-based checkpointer with AgentState."""
        from tulip.core.state import AgentState
        from tulip.memory.backends.file import FileCheckpointer

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpointer = FileCheckpointer(base_dir=temp_dir)

            # Create a state
            state = AgentState(agent_id="file_test")

            # Save and load
            checkpoint_id = await checkpointer.save(state, "file_thread")
            assert checkpoint_id is not None

            loaded = await checkpointer.load("file_thread")
            assert loaded is not None
            assert loaded.agent_id == "file_test"

    @pytest.mark.asyncio
    async def test_checkpointer_capabilities(self):
        """Test checkpointer capability inspection."""
        from tulip.memory.backends.memory import MemoryCheckpointer

        checkpointer = MemoryCheckpointer()

        # Check capabilities
        caps = checkpointer.capabilities
        assert caps.list_threads is True
        assert caps.persistent_checkpoint_ids is True


# =============================================================================
# Notebook 22: SSE Streaming Tests
# =============================================================================


class TestNotebook21SSEStreaming:
    """Tests for Notebook 22: SSE Streaming."""

    def test_sse_message_format(self):
        """Test SSE message formatting."""
        from tulip.streaming.sse import SSEMessage

        msg = SSEMessage(
            event="test",
            data='{"key": "value"}',
            id="1",
        )

        formatted = msg.format()
        assert "event: test" in formatted
        assert 'data: {"key": "value"}' in formatted
        assert "id: 1" in formatted

    def test_sse_multiline_data(self):
        """Test SSE with multi-line data."""
        from tulip.streaming.sse import SSEMessage

        msg = SSEMessage(
            event="code",
            data="line1\nline2\nline3",
        )

        formatted = msg.format()
        assert formatted.count("data:") == 3

    @pytest.mark.asyncio
    async def test_sse_handler(self):
        """Test SSE handler buffering."""
        from tulip.core.events import ThinkEvent
        from tulip.streaming.sse import SSEHandler

        handler = SSEHandler(include_id=True, id_prefix="e_")

        # ThinkEvent requires iteration field
        await handler.on_event(ThinkEvent(iteration=1, reasoning="Test thought"))
        await handler.on_complete()

        messages = handler.get_messages()
        assert len(messages) == 2  # Event + done

        # Check IDs
        assert messages[0].id == "e_1"
        assert messages[1].id == "e_2"

    @pytest.mark.asyncio
    async def test_sse_handler_error(self):
        """Test SSE handler error handling."""
        from tulip.streaming.sse import SSEHandler

        handler = SSEHandler()

        await handler.on_error(ValueError("Test error"))

        assert handler.has_error is True
        assert handler.is_complete is True

        messages = handler.get_messages()
        assert len(messages) == 1
        assert messages[0].event == "error"

    @pytest.mark.asyncio
    async def test_async_sse_handler(self):
        """Test async SSE handler streaming."""
        import asyncio

        from tulip.core.events import ThinkEvent
        from tulip.streaming.sse import AsyncSSEHandler

        handler = AsyncSSEHandler()

        async def producer():
            # ThinkEvent requires iteration field
            await handler.on_event(ThinkEvent(iteration=1, reasoning="Test"))
            await handler.on_complete()

        async def consumer():
            messages = []
            async for msg in handler.stream():
                messages.append(msg)
            return messages

        # Run both
        producer_task = asyncio.create_task(producer())
        messages = await consumer()
        await producer_task

        assert len(messages) == 2  # Event + done

    def test_sse_response_headers(self):
        """Test SSE response headers."""
        from tulip.streaming.sse import create_sse_response_headers

        headers = create_sse_response_headers()

        assert headers["Content-Type"] == "text/event-stream"
        assert headers["Cache-Control"] == "no-cache"


# =============================================================================
# Notebook Execution Tests
# =============================================================================


@pytest.mark.requires_model
class TestNotebookExecution:
    """Tests that run actual notebooks (with mock model)."""

    # Test method names mirror the current ``examples/notebook_NN_*.py``
    # numbering. The notebooks have been renumbered a few times during
    # development; the previous test methods pointed at filenames that
    # no longer exist on disk and were failing with FileNotFoundError.
    # Keep these in sync with the actual ``examples/`` layout.

    @pytest.mark.asyncio
    async def test_notebook_13_runs(self):
        """Smoke: notebook 13 (SSE streaming) executes cleanly."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "examples/notebook_13_sse_streaming.py"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, f"Notebook 13 failed: {result.stderr}"

    @pytest.mark.asyncio
    async def test_notebook_35_runs(self):
        """Smoke: notebook 35 (structured output) executes cleanly."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "examples/notebook_35_structured_output.py"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, f"Notebook 35 failed: {result.stderr}"

    @pytest.mark.asyncio
    async def test_notebook_36_runs(self):
        """Smoke: notebook 36 (reasoning patterns) executes cleanly."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "examples/notebook_36_reasoning_patterns.py"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, f"Notebook 36 failed: {result.stderr}"

    @pytest.mark.asyncio
    async def test_notebook_46_runs(self):
        """Smoke: notebook 46 (playbooks) executes cleanly."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "examples/notebook_46_playbooks.py"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, f"Notebook 46 failed: {result.stderr}"

    @pytest.mark.asyncio
    async def test_notebook_52_runs(self):
        """Smoke: notebook 52 (checkpoint backends) executes cleanly."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "examples/notebook_52_checkpoint_backends.py"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, f"Notebook 52 failed: {result.stderr}"
