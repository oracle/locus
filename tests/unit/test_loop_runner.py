# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for loop runner module."""

from unittest.mock import MagicMock

import pytest

from locus.core.events import (
    ReflectEvent,
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
    ToolStartEvent,
)
from locus.loop.runner import (
    StreamingCollector,
    create_runner,
)


class TestStreamingCollector:
    """Tests for StreamingCollector class."""

    @pytest.fixture
    def collector(self):
        return StreamingCollector()

    def test_initialization(self, collector):
        """Test StreamingCollector initialization."""
        assert collector.events == []
        assert collector.think_events == []
        assert collector.tool_events == []
        assert collector.reflect_events == []
        assert collector.terminate_event is None

    def test_collect_think_event(self, collector):
        """Test collecting think event."""
        event = ThinkEvent(reasoning="Thinking...", iteration=1)
        collector.collect(event)

        assert len(collector.events) == 1
        assert len(collector.think_events) == 1
        assert collector.think_events[0] is event

    def test_collect_tool_start_event(self, collector):
        """Test collecting tool start event."""
        event = ToolStartEvent(
            tool_name="search",
            tool_call_id="call_123",
            arguments={"q": "test"},
        )
        collector.collect(event)

        assert len(collector.events) == 1
        assert len(collector.tool_events) == 1

    def test_collect_tool_complete_event(self, collector):
        """Test collecting tool complete event."""
        event = ToolCompleteEvent(
            tool_name="search",
            tool_call_id="call_123",
            result="found",
        )
        collector.collect(event)

        assert len(collector.events) == 1
        assert len(collector.tool_events) == 1

    def test_collect_reflect_event(self, collector):
        """Test collecting reflect event."""
        event = ReflectEvent(
            assessment="on_track",
            confidence_delta=0.1,
            new_confidence=0.8,
            iteration=1,
        )
        collector.collect(event)

        assert len(collector.events) == 1
        assert len(collector.reflect_events) == 1

    def test_collect_terminate_event(self, collector):
        """Test collecting terminate event."""
        event = TerminateEvent(
            reason="complete", iterations_used=5, final_confidence=0.95, total_tool_calls=3
        )
        collector.collect(event)

        assert len(collector.events) == 1
        assert collector.terminate_event is event

    def test_is_complete_before_terminate(self, collector):
        """Test is_complete returns False before terminate."""
        assert collector.is_complete is False

    def test_is_complete_after_terminate(self, collector):
        """Test is_complete returns True after terminate."""
        event = TerminateEvent(
            reason="done", iterations_used=1, final_confidence=0.9, total_tool_calls=0
        )
        collector.collect(event)

        assert collector.is_complete is True

    def test_iterations_property(self, collector):
        """Test iterations property."""
        assert collector.iterations == 0

        event = TerminateEvent(
            reason="done", iterations_used=10, final_confidence=0.9, total_tool_calls=0
        )
        collector.collect(event)

        assert collector.iterations == 10

    def test_final_confidence_property(self, collector):
        """Test final_confidence property."""
        assert collector.final_confidence == 0.0

        event = TerminateEvent(
            reason="done", iterations_used=5, final_confidence=0.95, total_tool_calls=0
        )
        collector.collect(event)

        assert collector.final_confidence == 0.95

    def test_reset(self, collector):
        """Test resetting the collector."""
        # Add some events
        collector.collect(ThinkEvent(reasoning="Test", iteration=1))
        collector.collect(
            TerminateEvent(
                reason="done", iterations_used=1, final_confidence=0.9, total_tool_calls=0
            )
        )

        assert len(collector.events) == 2
        assert collector.is_complete is True

        # Reset
        collector.reset()

        assert collector.events == []
        assert collector.think_events == []
        assert collector.tool_events == []
        assert collector.reflect_events == []
        assert collector.terminate_event is None
        assert collector.is_complete is False

    def test_collect_multiple_events(self, collector):
        """Test collecting multiple events of different types."""
        events = [
            ThinkEvent(reasoning="First thought", iteration=1),
            ToolStartEvent(tool_name="search", tool_call_id="call_1", arguments={"q": "test"}),
            ToolCompleteEvent(tool_name="search", tool_call_id="call_1", result="found"),
            ThinkEvent(reasoning="Second thought", iteration=2),
            ReflectEvent(
                assessment="on_track", confidence_delta=0.1, new_confidence=0.8, iteration=2
            ),
            TerminateEvent(
                reason="complete", iterations_used=2, final_confidence=0.9, total_tool_calls=1
            ),
        ]

        for event in events:
            collector.collect(event)

        assert len(collector.events) == 6
        assert len(collector.think_events) == 2
        assert len(collector.tool_events) == 2
        assert len(collector.reflect_events) == 1
        assert collector.terminate_event is not None


class TestCreateRunner:
    """Tests for create_runner factory function."""

    def test_create_runner(self):
        """Test creating a runner with factory function."""
        mock_model = MagicMock()
        mock_registry = MagicMock()

        runner = create_runner(
            model=mock_model,
            registry=mock_registry,
        )

        from locus.loop.runner import LoopRunner

        assert isinstance(runner, LoopRunner)
        assert runner.timeout is None
        assert runner.on_event is None

    def test_create_runner_with_options(self):
        """Test creating a runner with custom options."""
        mock_model = MagicMock()
        mock_registry = MagicMock()
        on_event = MagicMock()

        runner = create_runner(
            model=mock_model,
            registry=mock_registry,
            max_iterations=50,
            confidence_threshold=0.9,
            enable_reflection=False,
            system_prompt="You are helpful",
            timeout=60.0,
            on_event=on_event,
        )

        from locus.loop.runner import LoopRunner

        assert isinstance(runner, LoopRunner)
        assert runner.timeout == 60.0
        assert runner.on_event is on_event
        assert runner.loop.config.max_iterations == 50
        assert runner.loop.config.confidence_threshold == 0.9
        assert runner.loop.config.enable_reflection is False
        assert runner.loop.config.system_prompt == "You are helpful"

    def test_create_runner_default_config(self):
        """Test default configuration."""
        mock_model = MagicMock()
        mock_registry = MagicMock()

        runner = create_runner(model=mock_model, registry=mock_registry)

        assert runner.loop.config.max_iterations == 20
        assert runner.loop.config.confidence_threshold == 0.85
        assert runner.loop.config.enable_reflection is True


class TestLoopRunnerProperties:
    """Tests for LoopRunner properties."""

    def test_events_property_empty(self):
        """Test events property when empty."""
        mock_model = MagicMock()
        mock_registry = MagicMock()

        runner = create_runner(model=mock_model, registry=mock_registry)

        assert runner.events == []

    def test_final_state_property_none(self):
        """Test final_state property when not run."""
        mock_model = MagicMock()
        mock_registry = MagicMock()

        runner = create_runner(model=mock_model, registry=mock_registry)

        assert runner.final_state is None
