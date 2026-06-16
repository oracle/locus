# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for console streaming handler."""

from datetime import UTC, datetime
from io import StringIO
from unittest.mock import MagicMock

import pytest

from locus.core.events import (
    CausalEdgeEvent,
    CausalNodeEvent,
    GroundingEvent,
    ModelChunkEvent,
    OrchestratorDecisionEvent,
    ReflectEvent,
    SpecialistCompleteEvent,
    SpecialistStartEvent,
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
    ToolStartEvent,
)
from locus.core.messages import ToolCall
from locus.streaming.console import ConsoleHandler, MinimalConsoleHandler


class TestConsoleHandlerInit:
    """Tests for ConsoleHandler initialization."""

    def test_default_init(self):
        """Test creating handler with defaults."""
        handler = ConsoleHandler()
        assert handler.show_reasoning is True
        assert handler.show_tool_args is False
        assert handler.show_tool_results is True
        assert handler.max_result_length == 500

    def test_custom_output(self):
        """Test creating handler with custom output."""
        output = StringIO()
        handler = ConsoleHandler(output=output)
        assert handler.output is output

    def test_all_options(self):
        """Test creating handler with all options."""
        handler = ConsoleHandler(
            show_reasoning=False,
            show_tool_args=True,
            show_tool_results=False,
            show_timestamps=True,
            show_progress=False,
            use_color=False,
            use_emoji=False,
            max_result_length=100,
            indent="    ",
        )
        assert handler.show_reasoning is False
        assert handler.show_tool_args is True
        assert handler.show_tool_results is False
        assert handler.show_timestamps is True
        assert handler.show_progress is False
        assert handler.use_color is False
        assert handler.use_emoji is False
        assert handler.max_result_length == 100
        assert handler.indent == "    "


class TestConsoleHandlerHelpers:
    """Tests for helper methods."""

    @pytest.fixture
    def handler(self):
        """Create handler with StringIO output."""
        return ConsoleHandler(output=StringIO(), use_color=False, use_emoji=False)

    @pytest.fixture
    def color_handler(self):
        """Create handler with colors enabled."""
        handler = ConsoleHandler(output=StringIO(), use_emoji=False)
        handler.use_color = True
        return handler

    def test_supports_color_with_tty(self):
        """Test color support detection with tty."""
        output = MagicMock()
        output.isatty.return_value = True
        handler = ConsoleHandler(output=output)
        assert handler._supports_color() is True

    def test_supports_color_without_tty(self):
        """Test color support detection without tty."""
        output = StringIO()
        handler = ConsoleHandler(output=output)
        assert handler._supports_color() is False

    def test_supports_color_no_isatty(self):
        """Test color support when isatty not available."""
        output = MagicMock(spec=[])  # No isatty method
        handler = ConsoleHandler(output=output)
        assert handler._supports_color() is False

    def test_color_applies_when_enabled(self, color_handler):
        """Test color application when enabled."""
        result = color_handler._color("test", "red")
        assert "\033[31m" in result
        assert "\033[0m" in result

    def test_color_skipped_when_disabled(self, handler):
        """Test color skipped when disabled."""
        result = handler._color("test", "red")
        assert result == "test"

    def test_symbol_returns_emoji(self):
        """Test symbol returns emoji when enabled."""
        handler = ConsoleHandler(output=StringIO(), use_emoji=True)
        assert handler._symbol("think") == "💭"

    def test_symbol_empty_when_disabled(self, handler):
        """Test symbol returns empty when disabled."""
        assert handler._symbol("think") == ""

    def test_symbol_unknown_type(self, handler):
        """Test symbol for unknown type."""
        assert handler._symbol("unknown") == ""

    def test_truncate_short_text(self, handler):
        """Test truncate with short text."""
        result = handler._truncate("short", max_length=100)
        assert result == "short"

    def test_truncate_long_text(self, handler):
        """Test truncate with long text."""
        long_text = "a" * 100
        result = handler._truncate(long_text, max_length=50)
        assert len(result) == 50
        assert result.endswith("...")

    def test_write_with_newline(self, handler):
        """Test write adds newline."""
        handler._write("test")
        assert handler.output.getvalue() == "test\n"

    def test_write_without_newline(self, handler):
        """Test write without newline."""
        handler._write("test", newline=False)
        assert handler.output.getvalue() == "test"

    def test_format_timestamp_when_disabled(self, handler):
        """Test timestamp formatting when disabled."""
        event = MagicMock()
        result = handler._format_timestamp(event)
        assert result == ""

    def test_format_timestamp_when_enabled(self):
        """Test timestamp formatting when enabled."""
        handler = ConsoleHandler(output=StringIO(), show_timestamps=True)
        event = MagicMock()
        event.timestamp = datetime(2024, 1, 15, 10, 30, 45, 123456, tzinfo=UTC)
        result = handler._format_timestamp(event)
        assert "10:30:45" in result


class TestConsoleHandlerEvents:
    """Tests for event handling."""

    @pytest.fixture
    def handler(self):
        """Create handler with StringIO output."""
        return ConsoleHandler(
            output=StringIO(),
            use_color=False,
            use_emoji=False,
            show_progress=True,
            show_reasoning=True,
        )

    @pytest.mark.asyncio
    async def test_handle_think_event(self, handler):
        """Test handling think event."""
        event = ThinkEvent(
            iteration=1,
            reasoning="Thinking about the problem",
            tool_calls=[],
        )
        await handler.on_event(event)
        output = handler.output.getvalue()
        assert "Iteration 1" in output

    @pytest.mark.asyncio
    async def test_handle_think_event_with_reasoning(self, handler):
        """Test think event shows reasoning."""
        event = ThinkEvent(
            iteration=1,
            reasoning="Line 1\nLine 2",
            tool_calls=[],
        )
        await handler.on_event(event)
        output = handler.output.getvalue()
        assert "Line 1" in output
        assert "Line 2" in output

    @pytest.mark.asyncio
    async def test_handle_think_event_with_tool_calls(self, handler):
        """Test think event with planned tool calls."""
        event = ThinkEvent(
            iteration=1,
            reasoning="",
            tool_calls=[
                ToolCall(id="tc1", name="search", arguments={}),
                ToolCall(id="tc2", name="get_weather", arguments={}),
            ],
        )
        await handler.on_event(event)
        output = handler.output.getvalue()
        assert "2 tool calls" in output

    @pytest.mark.asyncio
    async def test_handle_tool_start(self, handler):
        """Test handling tool start event."""
        event = ToolStartEvent(tool_call_id="tc1", tool_name="search", arguments={})
        await handler.on_event(event)
        output = handler.output.getvalue()
        assert "search" in output

    @pytest.mark.asyncio
    async def test_handle_tool_start_with_args(self):
        """Test tool start with arguments shown."""
        handler = ConsoleHandler(
            output=StringIO(),
            use_color=False,
            use_emoji=False,
            show_tool_args=True,
        )
        event = ToolStartEvent(
            tool_call_id="tc1",
            tool_name="search",
            arguments={"query": "test", "limit": 10},
        )
        await handler.on_event(event)
        output = handler.output.getvalue()
        assert "search" in output
        assert "query" in output

    @pytest.mark.asyncio
    async def test_handle_tool_complete_success(self, handler):
        """Test handling successful tool completion."""
        event = ToolCompleteEvent(
            tool_call_id="tc1",
            tool_name="search",
            result="Found 5 results",
            duration_ms=150.0,
        )
        await handler.on_event(event)
        output = handler.output.getvalue()
        assert "search" in output
        assert "150ms" in output

    @pytest.mark.asyncio
    async def test_handle_tool_complete_error(self, handler):
        """Test handling tool completion with error."""
        event = ToolCompleteEvent(
            tool_call_id="tc1",
            tool_name="search",
            error="Connection timeout",
        )
        await handler.on_event(event)
        output = handler.output.getvalue()
        assert "search" in output
        assert "Connection timeout" in output

    @pytest.mark.asyncio
    async def test_handle_reflect_on_track(self, handler):
        """Test handling reflect event - on track."""
        event = ReflectEvent(
            iteration=1,
            assessment="on_track",
            confidence_delta=0.05,
            guidance="Continue with current approach",
            new_confidence=0.85,
        )
        await handler.on_event(event)
        output = handler.output.getvalue()
        assert "on_track" in output
        assert "0.85" in output

    @pytest.mark.asyncio
    async def test_handle_reflect_stuck(self, handler):
        """Test handling reflect event - stuck."""
        event = ReflectEvent(
            iteration=2,
            assessment="stuck",
            confidence_delta=-0.2,
            new_confidence=0.3,
        )
        await handler.on_event(event)
        output = handler.output.getvalue()
        assert "stuck" in output

    @pytest.mark.asyncio
    async def test_handle_grounding_high_score(self, handler):
        """Test handling grounding event with high score."""
        event = GroundingEvent(
            score=0.9,
            claims_evaluated=5,
            supported_claims=["claim1"],
            ungrounded_claims=[],
        )
        await handler.on_event(event)
        output = handler.output.getvalue()
        assert "0.90" in output
        assert "5 claims" in output

    @pytest.mark.asyncio
    async def test_handle_grounding_with_ungrounded(self, handler):
        """Test handling grounding event with ungrounded claims."""
        event = GroundingEvent(
            score=0.4,
            claims_evaluated=3,
            supported_claims=[],
            ungrounded_claims=["Unsupported claim 1"],
        )
        await handler.on_event(event)
        output = handler.output.getvalue()
        assert "Ungrounded" in output

    @pytest.mark.asyncio
    async def test_handle_terminate(self, handler):
        """Test handling terminate event."""
        event = TerminateEvent(
            reason="complete",
            iterations_used=3,
            total_tool_calls=5,
            final_confidence=0.95,
        )
        await handler.on_event(event)
        output = handler.output.getvalue()
        assert "Terminated" in output
        assert "complete" in output
        assert "3" in output
        assert "5" in output

    @pytest.mark.asyncio
    async def test_handle_specialist_start(self, handler):
        """Test handling specialist start event."""
        event = SpecialistStartEvent(
            specialist_id="spec-1",
            specialist_type="researcher",
            task="Investigate the market",
        )
        await handler.on_event(event)
        output = handler.output.getvalue()
        assert "researcher" in output
        assert "Investigate" in output

    @pytest.mark.asyncio
    async def test_handle_specialist_complete(self, handler):
        """Test handling specialist complete event."""
        event = SpecialistCompleteEvent(
            specialist_id="spec-1",
            specialist_type="researcher",
            result="Analysis complete",
            confidence=0.88,
            duration_ms=500.0,
        )
        await handler.on_event(event)
        output = handler.output.getvalue()
        assert "researcher" in output
        assert "0.88" in output
        assert "500" in output

    @pytest.mark.asyncio
    async def test_handle_orchestrator_decision(self, handler):
        """Test handling orchestrator decision event."""
        event = OrchestratorDecisionEvent(
            decision="delegate",
            specialists_selected=["researcher", "writer"],
        )
        await handler.on_event(event)
        output = handler.output.getvalue()
        assert "Orchestrator" in output
        assert "delegate" in output
        assert "researcher" in output

    @pytest.mark.asyncio
    async def test_handle_model_chunk(self, handler):
        """Test handling model chunk event."""
        event = ModelChunkEvent(content="Hello", done=False)
        await handler.on_event(event)
        output = handler.output.getvalue()
        assert "Hello" in output

    @pytest.mark.asyncio
    async def test_handle_model_chunk_done(self, handler):
        """Test handling model chunk done event."""
        event = ModelChunkEvent(content="", done=True)
        await handler.on_event(event)
        # Should add newline
        assert handler.output.getvalue() == "\n"

    @pytest.mark.asyncio
    async def test_handle_causal_node(self, handler):
        """Test handling causal node event."""
        event = CausalNodeEvent(
            node_id="n1",
            label="High latency",
            node_type="root_cause",
        )
        await handler.on_event(event)
        output = handler.output.getvalue()
        assert "High latency" in output
        assert "root_cause" in output

    @pytest.mark.asyncio
    async def test_handle_causal_edge(self, handler):
        """Test handling causal edge event."""
        event = CausalEdgeEvent(
            source_id="n1",
            target_id="n2",
            relationship="causes",
            confidence=0.9,
        )
        await handler.on_event(event)
        output = handler.output.getvalue()
        assert "n1" in output
        assert "n2" in output
        assert "causes" in output

    @pytest.mark.asyncio
    async def test_handle_unknown_event(self, handler):
        """Test handling unknown event type."""
        event = MagicMock()
        event.event_type = "custom_event"
        await handler.on_event(event)
        output = handler.output.getvalue()
        assert "custom_event" in output


class TestConsoleHandlerCompletion:
    """Tests for completion and error handling."""

    @pytest.fixture
    def handler(self):
        """Create handler with StringIO output."""
        return ConsoleHandler(output=StringIO(), use_color=False, use_emoji=False)

    @pytest.mark.asyncio
    async def test_on_complete(self, handler):
        """Test on_complete method."""
        handler._tool_count = 5
        await handler.on_complete()
        output = handler.output.getvalue()
        assert "complete" in output.lower()
        assert "5 tool calls" in output

    @pytest.mark.asyncio
    async def test_on_error(self, handler):
        """Test on_error method."""
        error = ValueError("Something went wrong")
        await handler.on_error(error)
        output = handler.output.getvalue()
        assert "Error" in output
        assert "Something went wrong" in output


class TestMinimalConsoleHandler:
    """Tests for MinimalConsoleHandler."""

    @pytest.fixture
    def handler(self):
        """Create minimal handler."""
        return MinimalConsoleHandler(output=StringIO())

    @pytest.mark.asyncio
    async def test_tool_start_event(self, handler):
        """Test handling tool start."""
        event = ToolStartEvent(tool_call_id="tc1", tool_name="search", arguments={})
        await handler.on_event(event)
        output = handler.output.getvalue()
        assert "search" in output

    @pytest.mark.asyncio
    async def test_tool_complete_with_error(self, handler):
        """Test handling tool complete with error."""
        event = ToolCompleteEvent(
            tool_call_id="tc1",
            tool_name="search",
            error="Failed",
        )
        await handler.on_event(event)
        output = handler.output.getvalue()
        assert "Error" in output
        assert "Failed" in output

    @pytest.mark.asyncio
    async def test_terminate_event(self, handler):
        """Test handling terminate event."""
        event = TerminateEvent(
            reason="complete",
            iterations_used=3,
            total_tool_calls=5,
            final_confidence=0.9,
        )
        await handler.on_event(event)
        output = handler.output.getvalue()
        assert "3" in output
        assert "iterations" in output

    @pytest.mark.asyncio
    async def test_on_complete(self, handler):
        """Test on_complete (does nothing)."""
        await handler.on_complete()
        assert handler.output.getvalue() == ""

    @pytest.mark.asyncio
    async def test_on_error(self, handler):
        """Test on_error."""
        error = ValueError("Test error")
        await handler.on_error(error)
        output = handler.output.getvalue()
        assert "Error" in output
        assert "Test error" in output
