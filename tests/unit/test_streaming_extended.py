# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Extended unit tests for streaming module."""

from unittest.mock import AsyncMock

import pytest

from locus.core.events import (
    LocusEvent,
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
    ToolStartEvent,
)
from locus.streaming.handler import BaseStreamHandler, CompositeHandler


class TestLocusEvents:
    """Tests for Locus event types."""

    def test_think_event(self):
        """Test ThinkEvent creation."""
        event = ThinkEvent(
            iteration=1,
            reasoning="Thinking about the problem",
            tool_calls=[{"name": "search", "args": {"query": "test"}}],
        )
        assert event.reasoning == "Thinking about the problem"
        assert event.tool_calls is not None
        assert len(event.tool_calls) == 1
        assert event.event_type == "think"
        assert event.iteration == 1

    def test_think_event_no_tools(self):
        """Test ThinkEvent without tool calls."""
        event = ThinkEvent(iteration=2, reasoning="Just thinking")
        assert event.reasoning == "Just thinking"
        assert event.iteration == 2
        assert event.event_type == "think"

    def test_tool_start_event(self):
        """Test ToolStartEvent creation."""
        event = ToolStartEvent(
            tool_name="search",
            arguments={"query": "test"},
            tool_call_id="call_123",
        )
        assert event.tool_name == "search"
        assert event.arguments == {"query": "test"}
        assert event.tool_call_id == "call_123"

    def test_tool_complete_event(self):
        """Test ToolCompleteEvent creation."""
        event = ToolCompleteEvent(
            tool_name="search",
            result="Found 5 results",
            tool_call_id="call_123",
            duration_ms=150.5,
        )
        assert event.tool_name == "search"
        assert event.result == "Found 5 results"
        assert event.duration_ms == 150.5

    def test_tool_complete_event_with_error(self):
        """Test ToolCompleteEvent with error."""
        event = ToolCompleteEvent(
            tool_name="search",
            result=None,
            tool_call_id="call_123",
            error="Connection timeout",
        )
        assert event.error == "Connection timeout"

    def test_terminate_event(self):
        """Test TerminateEvent creation."""
        event = TerminateEvent(
            reason="Task completed successfully",
            iterations_used=5,
            final_confidence=0.95,
            total_tool_calls=10,
        )
        assert event.reason == "Task completed successfully"
        assert event.event_type == "terminate"

    def test_terminate_event_with_iterations(self):
        """Test TerminateEvent with iteration info."""
        event = TerminateEvent(
            reason="Max iterations reached",
            iterations_used=10,
            final_confidence=0.8,
            total_tool_calls=25,
        )
        assert event.iterations_used == 10
        assert event.event_type == "terminate"
        assert event.final_confidence == 0.8
        assert event.total_tool_calls == 25


class TestBaseStreamHandler:
    """Tests for BaseStreamHandler."""

    def test_handler_protocol(self):
        """Test handler follows protocol."""

        # Create concrete implementation
        class TestHandler(BaseStreamHandler):
            async def on_event(self, event: LocusEvent) -> None:
                pass

        handler = TestHandler()
        assert hasattr(handler, "on_event")

    @pytest.mark.asyncio
    async def test_handler_on_event(self):
        """Test on_event method."""
        received_events = []

        class TestHandler(BaseStreamHandler):
            async def on_event(self, event: LocusEvent) -> None:
                received_events.append(event)

        handler = TestHandler()
        event = ThinkEvent(iteration=1, reasoning="Test")

        await handler.on_event(event)

        assert len(received_events) == 1
        assert received_events[0] == event


class TestCompositeHandler:
    """Tests for CompositeHandler."""

    @pytest.fixture
    def mock_handlers(self):
        """Create mock handlers."""
        handler1 = AsyncMock(spec=BaseStreamHandler)
        handler2 = AsyncMock(spec=BaseStreamHandler)
        return handler1, handler2

    def test_create_composite_handler(self, mock_handlers):
        """Test creating composite handler."""
        handler1, handler2 = mock_handlers
        composite = CompositeHandler(handlers=[handler1, handler2])
        assert len(composite.handlers) == 2

    @pytest.mark.asyncio
    async def test_composite_forwards_events(self, mock_handlers):
        """Test composite handler forwards events to all handlers."""
        handler1, handler2 = mock_handlers
        composite = CompositeHandler(handlers=[handler1, handler2])

        event = ThinkEvent(iteration=1, reasoning="Test")
        await composite.on_event(event)

        handler1.on_event.assert_called_once_with(event)
        handler2.on_event.assert_called_once_with(event)

    def test_add_handler(self, mock_handlers):
        """Test adding handler to composite."""
        handler1, handler2 = mock_handlers
        composite = CompositeHandler(handlers=[handler1])

        composite.add_handler(handler2)

        assert len(composite.handlers) == 2
        assert handler2 in composite.handlers

    def test_remove_handler(self, mock_handlers):
        """Test removing handler from composite."""
        handler1, handler2 = mock_handlers
        composite = CompositeHandler(handlers=[handler1, handler2])

        composite.remove_handler(handler1)

        assert len(composite.handlers) == 1
        assert handler1 not in composite.handlers
