# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Server-Sent Events (SSE) streaming handler."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from locus.core.events import LocusEvent
from locus.streaming.handler import BaseStreamHandler


_logger = logging.getLogger(__name__)


def _build_error_payload(error: Exception) -> dict[str, Any]:
    """Return a generic error payload safe to stream to unauthenticated peers.

    The raw exception string routinely contains DSN fragments, file paths,
    SQL snippets, or bucket names that are not appropriate to leak over a
    public SSE stream (CWE-209). We surface a stable correlation id so an
    operator can match the client-visible event back to the server log.

    The detail is only returned when LocusSettings.debug is True; otherwise
    we log the exception server-side at ERROR with the same correlation id.
    """
    correlation_id = uuid.uuid4().hex
    payload: dict[str, Any] = {
        "error": "internal error",
        "correlation_id": correlation_id,
    }

    try:
        from locus.core.config import get_settings

        debug = bool(get_settings().debug)
    except Exception:  # noqa: BLE001 — settings access must never break streaming
        debug = False

    if debug:
        payload["error"] = str(error)
        payload["error_type"] = type(error).__name__
    else:
        # ``exc_info`` takes the actual exception triple here — the
        # handler is called outside the ``except`` block, so ``True``
        # would resolve to ``sys.exc_info()`` which may be stale.
        _logger.error(
            "SSE stream error (correlation_id=%s): %s: %s",
            correlation_id,
            type(error).__name__,
            error,
            exc_info=(type(error), error, error.__traceback__),
        )
    return payload


class SSEMessage(BaseModel):
    """A Server-Sent Event message.

    Format follows the SSE specification:
    - event: Event type (optional)
    - data: Event data (required)
    - id: Event ID (optional)
    - retry: Reconnection time in ms (optional)
    """

    event: str | None = Field(default=None, description="Event type")
    data: str = Field(..., description="Event data (JSON string)")
    id: str | None = Field(default=None, description="Event ID")
    retry: int | None = Field(default=None, description="Retry interval in ms")

    def format(self) -> str:
        """Format as SSE wire protocol.

        Returns:
            SSE-formatted string ready for transmission
        """
        lines: list[str] = []

        if self.event is not None:
            lines.append(f"event: {self.event}")

        if self.id is not None:
            lines.append(f"id: {self.id}")

        if self.retry is not None:
            lines.append(f"retry: {self.retry}")

        # Data can be multi-line, each line needs "data: " prefix
        for line in self.data.split("\n"):
            lines.append(f"data: {line}")

        # End with empty line
        lines.append("")
        lines.append("")

        return "\n".join(lines)


class SSEHandler(BaseStreamHandler):
    """Stream handler that formats events as Server-Sent Events.

    Supports all 37+ event types from the Locus event system,
    formatting them as SSE messages for HTTP streaming.

    Example:
        >>> handler = SSEHandler()
        >>> await handler.on_event(think_event)
        >>> async for message in handler.messages():
        ...     yield message.format()

    The handler can be used in two modes:
    1. Pull mode: Collect messages via messages() async generator
    2. Push mode: Provide a callback for immediate message handling
    """

    # Supported event types (all from events.py)
    SUPPORTED_EVENTS = {
        # Loop events
        "think",
        "tool_start",
        "tool_complete",
        "reflect",
        "grounding",
        "terminate",
        # Model events
        "model_chunk",
        "model_complete",
        # Multi-agent events
        "specialist_start",
        "specialist_complete",
        "orchestrator_decision",
        # Causal events
        "causal_node",
        "causal_edge",
        # Hook events
        "before_invocation",
        "after_invocation",
        "before_tool_call",
        "after_tool_call",
    }

    def __init__(
        self,
        include_timestamp: bool = True,
        include_id: bool = True,
        id_prefix: str = "",
        custom_serializer: Any | None = None,  # Callable[[LocusEvent], dict]
    ):
        """Initialize the SSE handler.

        Args:
            include_timestamp: Whether to include timestamp in data
            include_id: Whether to include event IDs
            id_prefix: Prefix for event IDs
            custom_serializer: Optional custom event serializer
        """
        self.include_timestamp = include_timestamp
        self.include_id = include_id
        self.id_prefix = id_prefix
        self.custom_serializer = custom_serializer

        self._messages: list[SSEMessage] = []
        self._event_counter = 0
        self._complete = False
        self._error: Exception | None = None

    def _serialize_event(self, event: LocusEvent) -> dict[str, Any]:
        """Serialize an event to a dictionary.

        Args:
            event: Event to serialize

        Returns:
            Dictionary representation of the event
        """
        if self.custom_serializer:
            # User-supplied callable; mypy can't narrow its return.
            return self.custom_serializer(event)  # type: ignore[no-any-return]

        # Use Pydantic's model_dump
        data = event.model_dump()

        # Convert datetime to ISO format
        if "timestamp" in data and isinstance(data["timestamp"], datetime):
            data["timestamp"] = data["timestamp"].isoformat()

        return data

    def _create_message(
        self,
        event_type: str,
        data: dict[str, Any],
    ) -> SSEMessage:
        """Create an SSE message.

        Args:
            event_type: Event type
            data: Event data dictionary

        Returns:
            Formatted SSE message
        """
        self._event_counter += 1

        event_id = None
        if self.include_id:
            event_id = f"{self.id_prefix}{self._event_counter}"

        return SSEMessage(
            event=event_type,
            data=json.dumps(data, default=str),
            id=event_id,
        )

    async def on_event(self, event: LocusEvent) -> None:
        """Handle a streaming event.

        Converts the event to an SSE message and buffers it.

        Args:
            event: The event to process
        """
        data = self._serialize_event(event)
        message = self._create_message(event.event_type, data)
        self._messages.append(message)

    async def on_complete(self) -> None:
        """Handle stream completion.

        Adds a special "done" event to signal completion.
        """
        self._complete = True
        message = self._create_message(
            "done",
            {"status": "complete", "total_events": self._event_counter},
        )
        self._messages.append(message)

    async def on_error(self, error: Exception) -> None:
        """Handle a streaming error.

        Adds an error event and marks stream as complete.

        Args:
            error: The error that occurred
        """
        self._error = error
        self._complete = True
        message = self._create_message("error", _build_error_payload(error))
        self._messages.append(message)

    def get_messages(self) -> list[SSEMessage]:
        """Get all buffered messages.

        Returns:
            List of SSE messages
        """
        return list(self._messages)

    def pop_messages(self) -> list[SSEMessage]:
        """Get and clear all buffered messages.

        Returns:
            List of SSE messages (buffer is cleared)
        """
        messages = self._messages
        self._messages = []
        return messages

    @property
    def is_complete(self) -> bool:
        """Check if streaming is complete."""
        return self._complete

    @property
    def has_error(self) -> bool:
        """Check if an error occurred."""
        return self._error is not None

    def format_all(self) -> str:
        """Format all buffered messages as SSE.

        Returns:
            Complete SSE-formatted string
        """
        return "".join(msg.format() for msg in self._messages)

    def clear(self) -> None:
        """Clear all buffered messages and reset state."""
        self._messages.clear()
        self._event_counter = 0
        self._complete = False
        self._error = None


class AsyncSSEHandler(BaseStreamHandler):
    """Async SSE handler with async generator output.

    Provides an async generator interface for streaming SSE messages,
    suitable for use with async web frameworks.

    Example:
        >>> handler = AsyncSSEHandler()
        >>> # In a background task, events are sent to handler
        >>> async for message in handler.stream():
        ...     await response.write(message)
    """

    def __init__(
        self,
        include_timestamp: bool = True,
        include_id: bool = True,
        id_prefix: str = "",
    ):
        """Initialize the async SSE handler.

        Args:
            include_timestamp: Whether to include timestamp in data
            include_id: Whether to include event IDs
            id_prefix: Prefix for event IDs
        """
        self.include_timestamp = include_timestamp
        self.include_id = include_id
        self.id_prefix = id_prefix

        self._event_counter = 0
        self._complete = False
        self._error: Exception | None = None

        # Use asyncio.Queue for async message passing
        import asyncio

        self._queue: asyncio.Queue[SSEMessage | None] = asyncio.Queue()

    def _serialize_event(self, event: LocusEvent) -> dict[str, Any]:
        """Serialize an event to a dictionary."""
        data = event.model_dump()
        if "timestamp" in data and isinstance(data["timestamp"], datetime):
            data["timestamp"] = data["timestamp"].isoformat()
        return data

    def _create_message(
        self,
        event_type: str,
        data: dict[str, Any],
    ) -> SSEMessage:
        """Create an SSE message."""
        self._event_counter += 1

        event_id = None
        if self.include_id:
            event_id = f"{self.id_prefix}{self._event_counter}"

        return SSEMessage(
            event=event_type,
            data=json.dumps(data, default=str),
            id=event_id,
        )

    async def on_event(self, event: LocusEvent) -> None:
        """Handle a streaming event.

        Args:
            event: The event to process
        """
        data = self._serialize_event(event)
        message = self._create_message(event.event_type, data)
        await self._queue.put(message)

    async def on_complete(self) -> None:
        """Handle stream completion."""
        self._complete = True
        message = self._create_message(
            "done",
            {"status": "complete", "total_events": self._event_counter},
        )
        await self._queue.put(message)
        await self._queue.put(None)  # Signal end

    async def on_error(self, error: Exception) -> None:
        """Handle a streaming error.

        Args:
            error: The error that occurred
        """
        self._error = error
        self._complete = True
        message = self._create_message("error", _build_error_payload(error))
        await self._queue.put(message)
        await self._queue.put(None)  # Signal end

    async def stream(self) -> AsyncIterator[str]:
        """Stream SSE-formatted messages.

        Yields:
            SSE-formatted strings ready for HTTP response
        """
        while True:
            message = await self._queue.get()
            if message is None:
                break
            yield message.format()

    async def stream_messages(self) -> AsyncIterator[SSEMessage]:
        """Stream SSE message objects.

        Yields:
            SSEMessage objects
        """
        while True:
            message = await self._queue.get()
            if message is None:
                break
            yield message

    @property
    def is_complete(self) -> bool:
        """Check if streaming is complete."""
        return self._complete


def create_sse_response_headers() -> dict[str, str]:
    """Create standard SSE HTTP response headers.

    Returns:
        Dictionary of HTTP headers for SSE response
    """
    return {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # Disable nginx buffering
    }
