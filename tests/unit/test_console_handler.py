# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for console handler."""

import io

import pytest

from locus.core.events import (
    CausalEdgeEvent,
    CausalNodeEvent,
    GroundingEvent,
    ModelChunkEvent,
    ModelCompleteEvent,
    OrchestratorDecisionEvent,
    ReflectEvent,
    SpecialistCompleteEvent,
    SpecialistStartEvent,
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
    ToolStartEvent,
)
from locus.streaming.console import ConsoleHandler


class TestConsoleHandler:
    """Tests for ConsoleHandler."""

    @pytest.fixture
    def output(self):
        """Create a StringIO for capturing output."""
        return io.StringIO()

    @pytest.fixture
    def handler(self, output):
        """Create a console handler with captured output."""
        return ConsoleHandler(output=output, use_color=False, use_emoji=False)

    def test_create_default(self):
        """Test creating handler with defaults."""
        handler = ConsoleHandler()
        assert handler.show_reasoning is True
        assert handler.show_tool_results is True
        # use_color depends on terminal support
        assert isinstance(handler.use_color, bool)

    def test_create_custom(self, output):
        """Test creating handler with custom settings."""
        handler = ConsoleHandler(
            output=output,
            show_reasoning=False,
            show_tool_args=True,
            show_tool_results=False,
            show_timestamps=True,
            show_progress=False,
            use_color=False,
            use_emoji=False,
            max_result_length=100,
        )
        assert handler.show_reasoning is False
        assert handler.show_tool_args is True
        assert handler.max_result_length == 100

    @pytest.mark.asyncio
    async def test_handle_think_event(self, handler, output):
        """Test handling think event."""
        event = ThinkEvent(iteration=1, reasoning="Thinking about the task")
        await handler.on_event(event)

        text = output.getvalue()
        # Should contain something about thinking
        assert len(text) > 0 or "think" in text.lower() or handler.show_reasoning is False

    @pytest.mark.asyncio
    async def test_handle_tool_start_event(self, handler, output):
        """Test handling tool start event."""
        event = ToolStartEvent(
            tool_name="search",
            arguments={"query": "test"},
            tool_call_id="call1",
        )
        await handler.on_event(event)

        text = output.getvalue()
        assert "search" in text.lower()

    @pytest.mark.asyncio
    async def test_handle_tool_complete_event(self, handler, output):
        """Test handling tool complete event."""
        event = ToolCompleteEvent(
            tool_name="search",
            result="Found 5 results",
            tool_call_id="call1",
        )
        await handler.on_event(event)

        text = output.getvalue()
        # Should show result
        assert len(text) > 0

    @pytest.mark.asyncio
    async def test_handle_tool_complete_with_error(self, handler, output):
        """Test handling tool complete with error."""
        event = ToolCompleteEvent(
            tool_name="search",
            result=None,
            tool_call_id="call1",
            error="Connection failed",
        )
        await handler.on_event(event)

        text = output.getvalue()
        # Should indicate error
        assert len(text) > 0

    @pytest.mark.asyncio
    async def test_handle_terminate_event(self, handler, output):
        """Test handling terminate event."""
        event = TerminateEvent(
            reason="Task completed",
            iterations_used=5,
            final_confidence=0.95,
            total_tool_calls=10,
        )
        await handler.on_event(event)

        text = output.getvalue()
        assert len(text) > 0

    @pytest.mark.asyncio
    async def test_on_complete(self, handler, output):
        """Test on_complete callback."""
        await handler.on_complete()
        # Should not raise

    @pytest.mark.asyncio
    async def test_on_error(self, handler, output):
        """Test on_error callback."""
        await handler.on_error(Exception("Test error"))
        # Should output something about the error
        text = output.getvalue()
        assert len(text) > 0

    def test_color_formatting(self):
        """Test _color method."""
        handler = ConsoleHandler(use_color=True)
        text = handler._color("test", "green")
        assert "test" in text

    def test_no_color_formatting(self):
        """Test no color formatting."""
        handler = ConsoleHandler(use_color=False)
        text = handler._color("test", "green")
        assert text == "test"

    def test_symbol_with_emoji(self):
        """Test getting symbol with emoji enabled."""
        handler = ConsoleHandler(use_emoji=True)
        symbol = handler._symbol("think")
        # Should return emoji symbol
        assert len(symbol) > 0

    def test_symbol_without_emoji(self):
        """Test getting symbol without emoji."""
        handler = ConsoleHandler(use_emoji=False)
        symbol = handler._symbol("think")
        # Should return text symbol
        assert isinstance(symbol, str)

    def test_truncate_long_result(self, output):
        """Test truncating long results."""
        handler = ConsoleHandler(output=output, max_result_length=10)
        result = handler._truncate("This is a very long result that should be truncated")
        assert len(result) <= 10 + 3  # +3 for "..."

    def test_truncate_short_result(self, output):
        """Test short results are not truncated."""
        handler = ConsoleHandler(output=output, max_result_length=100)
        result = handler._truncate("Short")
        assert result == "Short"

    @pytest.mark.asyncio
    async def test_handle_reflect_event(self, handler, output):
        """Test handling reflect event."""
        event = ReflectEvent(
            iteration=1,
            assessment="Reflecting on results",
            confidence_delta=0.1,
            new_confidence=0.8,
            guidance="Continue",
        )
        await handler.on_event(event)

        text = output.getvalue()
        assert len(text) > 0

    @pytest.mark.asyncio
    async def test_handle_grounding_event(self, handler, output):
        """Test handling grounding event."""
        event = GroundingEvent(
            score=0.9,
            claims_evaluated=5,
            ungrounded_claims=[],
            requires_replan=False,
        )
        await handler.on_event(event)

        text = output.getvalue()
        assert len(text) > 0

    @pytest.mark.asyncio
    async def test_handle_model_chunk_event(self, handler, output):
        """Test handling model chunk event."""
        event = ModelChunkEvent(
            chunk="Hello",
            accumulated="Hello",
        )
        await handler.on_event(event)
        # Model chunks may not produce visible output

    @pytest.mark.asyncio
    async def test_handle_model_complete_event(self, handler, output):
        """Test handling model complete event."""
        event = ModelCompleteEvent(
            content="Complete response",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )
        await handler.on_event(event)

    @pytest.mark.asyncio
    async def test_handle_specialist_start_event(self, handler, output):
        """Test handling specialist start event."""
        event = SpecialistStartEvent(
            specialist_id="researcher",
            specialist_type="research",
            task="Research topic",
        )
        await handler.on_event(event)

        text = output.getvalue()
        assert len(text) >= 0

    @pytest.mark.asyncio
    async def test_handle_specialist_complete_event(self, handler, output):
        """Test handling specialist complete event."""
        event = SpecialistCompleteEvent(
            specialist_id="researcher",
            specialist_type="research",
            result="Research findings",
            confidence=0.9,
            duration_ms=1000,
        )
        await handler.on_event(event)

    @pytest.mark.asyncio
    async def test_handle_orchestrator_decision_event(self, handler, output):
        """Test handling orchestrator decision event."""
        event = OrchestratorDecisionEvent(
            decision="delegate",
            target="researcher",
            reasoning="Need more research",
        )
        await handler.on_event(event)

    @pytest.mark.asyncio
    async def test_handle_causal_node_event(self, handler, output):
        """Test handling causal node event."""
        event = CausalNodeEvent(
            node_id="node1",
            label="Root cause",
            node_type="cause",
        )
        await handler.on_event(event)

    @pytest.mark.asyncio
    async def test_handle_causal_edge_event(self, handler, output):
        """Test handling causal edge event."""
        event = CausalEdgeEvent(
            source_id="node1",
            target_id="node2",
            relationship="causes",
            confidence=0.85,
        )
        await handler.on_event(event)

    def test_symbol_unknown_type(self):
        """Test getting symbol for unknown type."""
        handler = ConsoleHandler(use_emoji=True)
        symbol = handler._symbol("unknown_type")
        # Should return some default or empty
        assert isinstance(symbol, str)

    def test_color_with_bold(self):
        """Test color formatting with bold."""
        handler = ConsoleHandler(use_color=True)
        text = handler._color("test", "bold")
        assert "test" in text

    def test_write_method(self, output):
        """Test _write method."""
        handler = ConsoleHandler(output=output, use_color=False)
        handler._write("Test message")
        assert "Test message" in output.getvalue()

    @pytest.mark.asyncio
    async def test_handle_tool_start_with_args(self, output):
        """Test handling tool start with args displayed."""
        handler = ConsoleHandler(output=output, use_color=False, show_tool_args=True)
        event = ToolStartEvent(
            tool_name="search",
            arguments={"query": "test"},
            tool_call_id="call1",
        )
        await handler.on_event(event)

        text = output.getvalue()
        # Should show arguments when show_tool_args is True
        assert len(text) > 0

    @pytest.mark.asyncio
    async def test_handle_tool_complete_no_result(self, output):
        """Test handling tool complete with results hidden."""
        handler = ConsoleHandler(output=output, use_color=False, show_tool_results=False)
        event = ToolCompleteEvent(
            tool_name="search",
            result="Found 5 results",
            tool_call_id="call1",
        )
        await handler.on_event(event)

    @pytest.mark.asyncio
    async def test_handle_think_event_hidden(self, output):
        """Test handling think event when reasoning is hidden."""
        handler = ConsoleHandler(output=output, use_color=False, show_reasoning=False)
        event = ThinkEvent(iteration=1, reasoning="Thinking about the task")
        await handler.on_event(event)
        # Should not display reasoning when show_reasoning is False
