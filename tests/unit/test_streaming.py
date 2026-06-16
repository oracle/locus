# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Tests for the streaming system."""

import io
import json

import pytest

from locus.core.events import (
    LocusEvent,
    ModelChunkEvent,
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
    ToolStartEvent,
)
from locus.streaming import (
    AsyncSSEHandler,
    BufferingHandler,
    CompositeHandler,
    ConsoleHandler,
    FilteringHandler,
    MinimalConsoleHandler,
    SSEHandler,
    SSEMessage,
    StreamHandler,
    create_sse_response_headers,
)


class TestStreamHandler:
    """Tests for StreamHandler protocol."""

    def test_protocol_definition(self):
        """StreamHandler is a valid protocol."""

        class MyHandler:
            async def on_event(self, event: LocusEvent) -> None:
                pass

            async def on_complete(self) -> None:
                pass

            async def on_error(self, error: Exception) -> None:
                pass

        handler = MyHandler()
        assert isinstance(handler, StreamHandler)


class TestBufferingHandler:
    """Tests for BufferingHandler."""

    @pytest.mark.asyncio
    async def test_buffer_events(self):
        """Buffer events for later processing."""
        handler = BufferingHandler()

        event1 = ThinkEvent(iteration=1, reasoning="First thought")
        event2 = ToolStartEvent(
            tool_name="search",
            tool_call_id="call_1",
            arguments={"query": "test"},
        )

        await handler.on_event(event1)
        await handler.on_event(event2)

        events = handler.get_events()
        assert len(events) == 2
        assert events[0] == event1
        assert events[1] == event2

    @pytest.mark.asyncio
    async def test_buffer_max_size(self):
        """Buffer respects max size."""
        handler = BufferingHandler(max_size=2)

        await handler.on_event(ThinkEvent(iteration=1))
        await handler.on_event(ThinkEvent(iteration=2))
        await handler.on_event(ThinkEvent(iteration=3))

        events = handler.get_events()
        assert len(events) == 2
        assert events[0].iteration == 2  # type: ignore[union-attr]
        assert events[1].iteration == 3  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_complete_status(self):
        """Track completion status."""
        handler = BufferingHandler()

        assert handler.is_complete is False

        await handler.on_complete()

        assert handler.is_complete is True

    @pytest.mark.asyncio
    async def test_error_tracking(self):
        """Track errors."""
        handler = BufferingHandler()

        error = ValueError("Test error")
        await handler.on_error(error)

        errors = handler.get_errors()
        assert len(errors) == 1
        assert errors[0] is error

    @pytest.mark.asyncio
    async def test_clear(self):
        """Clear buffer."""
        handler = BufferingHandler()

        await handler.on_event(ThinkEvent(iteration=1))
        await handler.on_error(ValueError("error"))
        await handler.on_complete()

        handler.clear()

        assert handler.get_events() == []
        assert handler.get_errors() == []
        assert handler.is_complete is False


class TestCompositeHandler:
    """Tests for CompositeHandler."""

    @pytest.mark.asyncio
    async def test_delegate_to_multiple(self):
        """Delegate events to multiple handlers."""
        handler1 = BufferingHandler()
        handler2 = BufferingHandler()
        composite = CompositeHandler([handler1, handler2])

        event = ThinkEvent(iteration=1)
        await composite.on_event(event)

        assert len(handler1.get_events()) == 1
        assert len(handler2.get_events()) == 1

    @pytest.mark.asyncio
    async def test_delegate_complete(self):
        """Delegate completion to all handlers."""
        handler1 = BufferingHandler()
        handler2 = BufferingHandler()
        composite = CompositeHandler([handler1, handler2])

        await composite.on_complete()

        assert handler1.is_complete is True
        assert handler2.is_complete is True

    @pytest.mark.asyncio
    async def test_add_remove_handler(self):
        """Add and remove handlers."""
        handler1 = BufferingHandler()
        handler2 = BufferingHandler()
        composite = CompositeHandler([handler1])

        composite.add_handler(handler2)
        await composite.on_event(ThinkEvent(iteration=1))

        assert len(handler2.get_events()) == 1

        composite.remove_handler(handler2)
        await composite.on_event(ThinkEvent(iteration=2))

        assert len(handler2.get_events()) == 1  # Not updated

    @pytest.mark.asyncio
    async def test_delegate_error(self):
        """Delegate error to all handlers."""
        handler1 = BufferingHandler()
        handler2 = BufferingHandler()
        composite = CompositeHandler([handler1, handler2])

        error = ValueError("Test error")
        await composite.on_error(error)

        assert len(handler1.get_errors()) == 1
        assert handler1.get_errors()[0] is error
        assert len(handler2.get_errors()) == 1
        assert handler2.get_errors()[0] is error


class TestFilteringHandler:
    """Tests for FilteringHandler."""

    @pytest.mark.asyncio
    async def test_include_filter(self):
        """Filter to include only specific event types."""
        buffer = BufferingHandler()
        handler = FilteringHandler(
            delegate=buffer,
            event_types={"think"},
        )

        await handler.on_event(ThinkEvent(iteration=1))
        await handler.on_event(
            ToolStartEvent(
                tool_name="search",
                tool_call_id="call_1",
                arguments={},
            )
        )

        events = buffer.get_events()
        assert len(events) == 1
        assert events[0].event_type == "think"

    @pytest.mark.asyncio
    async def test_exclude_filter(self):
        """Filter to exclude specific event types."""
        buffer = BufferingHandler()
        handler = FilteringHandler(
            delegate=buffer,
            exclude_types={"model_chunk"},
        )

        await handler.on_event(ThinkEvent(iteration=1))
        await handler.on_event(ModelChunkEvent(content="chunk"))

        events = buffer.get_events()
        assert len(events) == 1
        assert events[0].event_type == "think"

    @pytest.mark.asyncio
    async def test_custom_filter_function(self):
        """Use custom filter function."""
        buffer = BufferingHandler()
        handler = FilteringHandler(
            delegate=buffer,
            filter_fn=lambda e: hasattr(e, "iteration") and e.iteration > 1,
        )

        await handler.on_event(ThinkEvent(iteration=1))
        await handler.on_event(ThinkEvent(iteration=2))
        await handler.on_event(ThinkEvent(iteration=3))

        events = buffer.get_events()
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_delegate_complete_error(self):
        """Delegate complete and error."""
        buffer = BufferingHandler()
        handler = FilteringHandler(delegate=buffer, event_types={"think"})

        await handler.on_complete()
        await handler.on_error(ValueError("test"))

        assert buffer.is_complete is True
        assert len(buffer.get_errors()) == 1


class TestConsoleHandler:
    """Tests for ConsoleHandler."""

    @pytest.fixture
    def output_buffer(self):
        """Create an output buffer."""
        return io.StringIO()

    @pytest.mark.asyncio
    async def test_basic_output(self, output_buffer):
        """Basic event output."""
        handler = ConsoleHandler(output=output_buffer, use_color=False, use_emoji=False)

        await handler.on_event(ThinkEvent(iteration=1, reasoning="Testing"))

        output = output_buffer.getvalue()
        assert "Iteration 1" in output

    @pytest.mark.asyncio
    async def test_tool_start_output(self, output_buffer):
        """Tool start event output."""
        handler = ConsoleHandler(output=output_buffer, use_color=False, use_emoji=False)

        await handler.on_event(
            ToolStartEvent(
                tool_name="search",
                tool_call_id="call_1",
                arguments={"query": "test"},
            )
        )

        output = output_buffer.getvalue()
        assert "search" in output

    @pytest.mark.asyncio
    async def test_tool_complete_output(self, output_buffer):
        """Tool complete event output."""
        handler = ConsoleHandler(
            output=output_buffer,
            use_color=False,
            use_emoji=False,
            show_tool_results=True,
        )

        await handler.on_event(
            ToolCompleteEvent(
                tool_name="search",
                tool_call_id="call_1",
                result="Found 10 results",
                duration_ms=150.0,
            )
        )

        output = output_buffer.getvalue()
        assert "search" in output
        assert "150" in output

    @pytest.mark.asyncio
    async def test_tool_error_output(self, output_buffer):
        """Tool error event output."""
        handler = ConsoleHandler(output=output_buffer, use_color=False, use_emoji=False)

        await handler.on_event(
            ToolCompleteEvent(
                tool_name="search",
                tool_call_id="call_1",
                error="Connection timeout",
            )
        )

        output = output_buffer.getvalue()
        assert "Connection timeout" in output

    @pytest.mark.asyncio
    async def test_terminate_output(self, output_buffer):
        """Terminate event output."""
        handler = ConsoleHandler(output=output_buffer, use_color=False, use_emoji=False)

        await handler.on_event(
            TerminateEvent(
                reason="complete",
                iterations_used=5,
                final_confidence=0.95,
                total_tool_calls=10,
            )
        )

        output = output_buffer.getvalue()
        assert "complete" in output
        assert "5" in output

    @pytest.mark.asyncio
    async def test_terminate_error_output(self, output_buffer):
        """Terminate event with error reason output."""
        handler = ConsoleHandler(output=output_buffer, use_color=False, use_emoji=False)

        await handler.on_event(
            TerminateEvent(
                reason="error",
                iterations_used=3,
                final_confidence=0.5,
                total_tool_calls=5,
            )
        )

        output = output_buffer.getvalue()
        assert "error" in output

    @pytest.mark.asyncio
    async def test_on_complete(self, output_buffer):
        """Completion handler output."""
        handler = ConsoleHandler(output=output_buffer, use_color=False, use_emoji=False)

        await handler.on_complete()

        output = output_buffer.getvalue()
        assert "complete" in output.lower()

    @pytest.mark.asyncio
    async def test_on_error(self, output_buffer):
        """Error handler output."""
        handler = ConsoleHandler(output=output_buffer, use_color=False, use_emoji=False)

        await handler.on_error(ValueError("Test error"))

        output = output_buffer.getvalue()
        assert "Error" in output
        assert "Test error" in output

    @pytest.mark.asyncio
    async def test_show_reasoning(self, output_buffer):
        """Show reasoning when enabled."""
        handler = ConsoleHandler(
            output=output_buffer,
            use_color=False,
            use_emoji=False,
            show_reasoning=True,
        )

        await handler.on_event(
            ThinkEvent(
                iteration=1,
                reasoning="I need to search for data first",
            )
        )

        output = output_buffer.getvalue()
        assert "search for data" in output

    @pytest.mark.asyncio
    async def test_hide_reasoning(self, output_buffer):
        """Hide reasoning when disabled."""
        handler = ConsoleHandler(
            output=output_buffer,
            use_color=False,
            use_emoji=False,
            show_reasoning=False,
        )

        await handler.on_event(
            ThinkEvent(
                iteration=1,
                reasoning="Secret reasoning",
            )
        )

        output = output_buffer.getvalue()
        assert "Secret reasoning" not in output

    @pytest.mark.asyncio
    async def test_truncate_long_results(self, output_buffer):
        """Truncate long tool results."""
        handler = ConsoleHandler(
            output=output_buffer,
            use_color=False,
            use_emoji=False,
            max_result_length=20,
            show_tool_results=True,
        )

        await handler.on_event(
            ToolCompleteEvent(
                tool_name="search",
                tool_call_id="call_1",
                result="A" * 100,
            )
        )

        output = output_buffer.getvalue()
        assert "..." in output

    @pytest.mark.asyncio
    async def test_color_disabled(self, output_buffer):
        """Color codes not included when disabled."""
        handler = ConsoleHandler(output=output_buffer, use_color=False)

        await handler.on_event(ThinkEvent(iteration=1))

        output = output_buffer.getvalue()
        assert "\033[" not in output


class TestMinimalConsoleHandler:
    """Tests for MinimalConsoleHandler."""

    @pytest.mark.asyncio
    async def test_minimal_output(self):
        """Minimal output format."""
        output = io.StringIO()
        handler = MinimalConsoleHandler(output=output)

        await handler.on_event(
            ToolStartEvent(
                tool_name="search",
                tool_call_id="call_1",
                arguments={},
            )
        )

        result = output.getvalue()
        assert "search" in result

    @pytest.mark.asyncio
    async def test_error_output(self):
        """Error output format."""
        output = io.StringIO()
        handler = MinimalConsoleHandler(output=output)

        await handler.on_error(ValueError("Test"))

        result = output.getvalue()
        assert "Error" in result


class TestSSEMessage:
    """Tests for SSEMessage."""

    def test_basic_message(self):
        """Basic SSE message formatting."""
        msg = SSEMessage(
            event="think",
            data='{"iteration": 1}',
        )

        formatted = msg.format()

        assert "event: think\n" in formatted
        assert 'data: {"iteration": 1}\n' in formatted
        assert formatted.endswith("\n\n")

    def test_message_with_id(self):
        """SSE message with ID."""
        msg = SSEMessage(
            event="think",
            data="{}",
            id="evt_123",
        )

        formatted = msg.format()

        assert "id: evt_123\n" in formatted

    def test_message_with_retry(self):
        """SSE message with retry."""
        msg = SSEMessage(
            event="think",
            data="{}",
            retry=3000,
        )

        formatted = msg.format()

        assert "retry: 3000\n" in formatted

    def test_multiline_data(self):
        """SSE message with multiline data."""
        msg = SSEMessage(
            event="think",
            data="line1\nline2\nline3",
        )

        formatted = msg.format()

        assert "data: line1\n" in formatted
        assert "data: line2\n" in formatted
        assert "data: line3\n" in formatted


class TestSSEHandler:
    """Tests for SSEHandler."""

    @pytest.mark.asyncio
    async def test_event_to_sse(self):
        """Convert event to SSE message."""
        handler = SSEHandler()

        await handler.on_event(ThinkEvent(iteration=1, reasoning="Testing"))

        messages = handler.get_messages()
        assert len(messages) == 1
        assert messages[0].event == "think"

        data = json.loads(messages[0].data)
        assert data["iteration"] == 1
        assert data["reasoning"] == "Testing"

    @pytest.mark.asyncio
    async def test_event_ids(self):
        """Events get sequential IDs."""
        handler = SSEHandler(include_id=True, id_prefix="locus_")

        await handler.on_event(ThinkEvent(iteration=1))
        await handler.on_event(ThinkEvent(iteration=2))

        messages = handler.get_messages()
        assert messages[0].id == "locus_1"
        assert messages[1].id == "locus_2"

    @pytest.mark.asyncio
    async def test_no_ids_when_disabled(self):
        """No IDs when disabled."""
        handler = SSEHandler(include_id=False)

        await handler.on_event(ThinkEvent(iteration=1))

        messages = handler.get_messages()
        assert messages[0].id is None

    @pytest.mark.asyncio
    async def test_on_complete_adds_done(self):
        """Completion adds done event."""
        handler = SSEHandler()

        await handler.on_event(ThinkEvent(iteration=1))
        await handler.on_complete()

        messages = handler.get_messages()
        assert len(messages) == 2
        assert messages[-1].event == "done"
        assert handler.is_complete is True

    @pytest.mark.asyncio
    async def test_on_error_adds_error(self):
        """Error adds a sanitized error event with a correlation id.

        Under the default (non-debug) settings the exception string is
        not returned on the wire (CWE-209); only a generic 'internal
        error' message and a correlation id that maps to the server log.
        """
        handler = SSEHandler()

        await handler.on_error(ValueError("Test error"))

        messages = handler.get_messages()
        assert len(messages) == 1
        assert messages[0].event == "error"

        data = json.loads(messages[0].data)
        assert data["error"] == "internal error"
        assert "correlation_id" in data
        # Raw exception text must NOT leak to unauthenticated peers.
        assert "Test error" not in data["error"]
        assert "error_type" not in data

    @pytest.mark.asyncio
    async def test_pop_messages(self):
        """Pop messages clears buffer."""
        handler = SSEHandler()

        await handler.on_event(ThinkEvent(iteration=1))

        popped = handler.pop_messages()
        assert len(popped) == 1

        remaining = handler.get_messages()
        assert len(remaining) == 0

    @pytest.mark.asyncio
    async def test_format_all(self):
        """Format all messages as SSE."""
        handler = SSEHandler()

        await handler.on_event(ThinkEvent(iteration=1))
        await handler.on_event(ThinkEvent(iteration=2))

        formatted = handler.format_all()

        assert "event: think" in formatted
        assert formatted.count("event: think") == 2

    @pytest.mark.asyncio
    async def test_clear(self):
        """Clear handler state."""
        handler = SSEHandler()

        await handler.on_event(ThinkEvent(iteration=1))
        await handler.on_complete()

        handler.clear()

        assert handler.get_messages() == []
        assert handler.is_complete is False

    @pytest.mark.asyncio
    async def test_custom_serializer(self):
        """Test SSEHandler with custom serializer."""

        def custom_serializer(event):
            return {"custom": True, "event_type": event.event_type}

        handler = SSEHandler(custom_serializer=custom_serializer)

        await handler.on_event(ThinkEvent(iteration=1))

        messages = handler.get_messages()
        assert len(messages) == 1
        data = json.loads(messages[0].data)
        assert data["custom"] is True

    @pytest.mark.asyncio
    async def test_has_error_property(self):
        """Test has_error property after error."""
        handler = SSEHandler()

        assert handler.has_error is False

        await handler.on_error(ValueError("test"))

        assert handler.has_error is True


class TestAsyncSSEHandler:
    """Tests for AsyncSSEHandler."""

    @pytest.mark.asyncio
    async def test_stream_messages(self):
        """Stream messages via async generator."""
        handler = AsyncSSEHandler()

        # Send events in background
        await handler.on_event(ThinkEvent(iteration=1))
        await handler.on_event(ThinkEvent(iteration=2))
        await handler.on_complete()

        messages = []
        async for msg in handler.stream():
            messages.append(msg)

        assert len(messages) == 3
        assert "event: think" in messages[0]
        assert "event: done" in messages[2]

    @pytest.mark.asyncio
    async def test_stream_message_objects(self):
        """Stream SSEMessage objects."""
        handler = AsyncSSEHandler()

        await handler.on_event(ThinkEvent(iteration=1))
        await handler.on_complete()

        messages = []
        async for msg in handler.stream_messages():
            messages.append(msg)

        assert len(messages) == 2
        assert isinstance(messages[0], SSEMessage)

    @pytest.mark.asyncio
    async def test_is_complete(self):
        """Track completion status."""
        handler = AsyncSSEHandler()

        assert handler.is_complete is False

        await handler.on_complete()

        assert handler.is_complete is True

    @pytest.mark.asyncio
    async def test_on_error(self):
        """Error emits a sanitized event and signals end of stream.

        The raw exception message is intentionally stripped from the
        outgoing payload (CWE-209); a correlation id is emitted so the
        operator can still match the client-visible event to the log.
        """
        handler = AsyncSSEHandler()

        await handler.on_error(ValueError("Test error"))

        messages = []
        async for msg in handler.stream():
            messages.append(msg)

        assert len(messages) == 1
        assert "event: error" in messages[0]
        assert "internal error" in messages[0]
        assert "correlation_id" in messages[0]
        assert "Test error" not in messages[0]
        assert handler.is_complete is True


class TestSSEResponseHeaders:
    """Tests for SSE response headers."""

    def test_create_headers(self):
        """Create standard SSE headers."""
        headers = create_sse_response_headers()

        assert headers["Content-Type"] == "text/event-stream"
        assert headers["Cache-Control"] == "no-cache"
        assert headers["Connection"] == "keep-alive"


class TestToolCompleteEvent:
    """Tests for ToolCompleteEvent properties."""

    def test_success_true(self):
        """Success property returns True when no error."""
        event = ToolCompleteEvent(
            tool_name="search",
            tool_call_id="call_1",
            result="found results",
        )
        assert event.success is True

    def test_success_false(self):
        """Success property returns False when error present."""
        event = ToolCompleteEvent(
            tool_name="search",
            tool_call_id="call_1",
            error="Connection failed",
        )
        assert event.success is False


class TestEventTypeSupport:
    """Test all event types are supported."""

    @pytest.mark.asyncio
    async def test_all_loop_events(self):
        """SSE handler supports all loop events."""
        from locus.core.events import (
            GroundingEvent,
            ReflectEvent,
            TerminateEvent,
            ThinkEvent,
            ToolCompleteEvent,
            ToolStartEvent,
        )

        handler = SSEHandler()

        events = [
            ThinkEvent(iteration=1),
            ToolStartEvent(tool_name="test", tool_call_id="1", arguments={}),
            ToolCompleteEvent(tool_name="test", tool_call_id="1", result="ok"),
            ReflectEvent(
                iteration=1,
                assessment="on_track",
                confidence_delta=0.1,
                new_confidence=0.8,
            ),
            GroundingEvent(score=0.9, claims_evaluated=5),
            TerminateEvent(
                reason="complete",
                iterations_used=1,
                final_confidence=0.9,
                total_tool_calls=1,
            ),
        ]

        for event in events:
            await handler.on_event(event)

        messages = handler.get_messages()
        assert len(messages) == 6

    @pytest.mark.asyncio
    async def test_model_events(self):
        """SSE handler supports model events."""
        from locus.core.events import ModelChunkEvent, ModelCompleteEvent

        handler = SSEHandler()

        await handler.on_event(ModelChunkEvent(content="Hello"))
        await handler.on_event(ModelCompleteEvent(content="Hello world"))

        messages = handler.get_messages()
        assert len(messages) == 2

    @pytest.mark.asyncio
    async def test_multi_agent_events(self):
        """SSE handler supports multi-agent events."""
        from locus.core.events import (
            OrchestratorDecisionEvent,
            SpecialistCompleteEvent,
            SpecialistStartEvent,
        )

        handler = SSEHandler()

        await handler.on_event(
            SpecialistStartEvent(
                specialist_id="spec_1",
                specialist_type="analyzer",
                task="Analyze data",
            )
        )
        await handler.on_event(
            SpecialistCompleteEvent(
                specialist_id="spec_1",
                specialist_type="analyzer",
                confidence=0.9,
                duration_ms=100.0,
            )
        )
        await handler.on_event(
            OrchestratorDecisionEvent(
                decision="invoke_specialist",
                specialists_selected=["analyzer"],
            )
        )

        messages = handler.get_messages()
        assert len(messages) == 3

    @pytest.mark.asyncio
    async def test_causal_events(self):
        """SSE handler supports causal events."""
        from locus.core.events import CausalEdgeEvent, CausalNodeEvent

        handler = SSEHandler()

        await handler.on_event(
            CausalNodeEvent(
                node_id="node_1",
                label="Root cause",
                node_type="root_cause",
            )
        )
        await handler.on_event(
            CausalEdgeEvent(
                source_id="node_1",
                target_id="node_2",
                relationship="causes",
                confidence=0.85,
            )
        )

        messages = handler.get_messages()
        assert len(messages) == 2
