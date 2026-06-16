# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Stream handler protocol and base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from locus.core.events import LocusEvent


@runtime_checkable
class StreamHandler(Protocol):
    """Protocol for stream event handlers.

    Implementations receive events as they occur during agent execution
    and can process them for display, logging, or forwarding.
    """

    async def on_event(self, event: LocusEvent) -> None:
        """Handle a streaming event.

        Args:
            event: The event to process
        """
        ...

    async def on_complete(self) -> None:
        """Called when streaming is complete.

        Use for cleanup, final output, etc.
        """
        ...

    async def on_error(self, error: Exception) -> None:
        """Handle a streaming error.

        Args:
            error: The error that occurred
        """
        ...


class BaseStreamHandler(ABC):
    """Abstract base class for stream handlers.

    Provides default implementations for on_complete and on_error,
    only requiring subclasses to implement on_event.
    """

    @abstractmethod
    async def on_event(self, event: LocusEvent) -> None:
        """Handle a streaming event.

        Args:
            event: The event to process
        """
        ...

    async def on_complete(self) -> None:
        """Called when streaming is complete.

        Default implementation does nothing.
        Override to add cleanup or final output.
        """

    async def on_error(self, error: Exception) -> None:
        """Handle a streaming error.

        Default implementation does nothing.
        Override to add error handling.

        Args:
            error: The error that occurred
        """


class CompositeHandler(BaseStreamHandler):
    """Handler that delegates to multiple child handlers.

    Useful for sending events to multiple destinations simultaneously.

    Example:
        >>> handler = CompositeHandler(
        ...     [
        ...         ConsoleHandler(),
        ...         SSEHandler(),
        ...         LoggingHandler(),
        ...     ]
        ... )
    """

    def __init__(self, handlers: list[StreamHandler] | None = None):
        """Initialize with optional list of handlers.

        Args:
            handlers: List of handlers to delegate to
        """
        self.handlers: list[StreamHandler] = list(handlers) if handlers else []

    def add_handler(self, handler: StreamHandler) -> None:
        """Add a handler to the composite.

        Args:
            handler: Handler to add
        """
        self.handlers.append(handler)

    def remove_handler(self, handler: StreamHandler) -> None:
        """Remove a handler from the composite.

        Args:
            handler: Handler to remove
        """
        self.handlers.remove(handler)

    async def on_event(self, event: LocusEvent) -> None:
        """Delegate event to all handlers.

        Args:
            event: The event to process
        """
        for handler in self.handlers:
            await handler.on_event(event)

    async def on_complete(self) -> None:
        """Delegate completion to all handlers."""
        for handler in self.handlers:
            await handler.on_complete()

    async def on_error(self, error: Exception) -> None:
        """Delegate error to all handlers.

        Args:
            error: The error that occurred
        """
        for handler in self.handlers:
            await handler.on_error(error)


class BufferingHandler(BaseStreamHandler):
    """Handler that buffers events for later processing.

    Useful for testing or when events need to be processed in batches.

    Example:
        >>> handler = BufferingHandler()
        >>> await handler.on_event(event1)
        >>> await handler.on_event(event2)
        >>> events = handler.get_events()  # [event1, event2]
    """

    def __init__(self, max_size: int | None = None):
        """Initialize the buffer.

        Args:
            max_size: Maximum number of events to buffer (None for unlimited)
        """
        self.max_size = max_size
        self._events: list[LocusEvent] = []
        self._errors: list[Exception] = []
        self._complete: bool = False

    async def on_event(self, event: LocusEvent) -> None:
        """Buffer an event.

        Args:
            event: The event to buffer
        """
        if self.max_size is not None and len(self._events) >= self.max_size:
            self._events.pop(0)
        self._events.append(event)

    async def on_complete(self) -> None:
        """Mark streaming as complete."""
        self._complete = True

    async def on_error(self, error: Exception) -> None:
        """Record an error.

        Args:
            error: The error that occurred
        """
        self._errors.append(error)

    def get_events(self) -> list[LocusEvent]:
        """Get all buffered events.

        Returns:
            List of buffered events
        """
        return list(self._events)

    def get_errors(self) -> list[Exception]:
        """Get all recorded errors.

        Returns:
            List of recorded errors
        """
        return list(self._errors)

    @property
    def is_complete(self) -> bool:
        """Check if streaming completed."""
        return self._complete

    def clear(self) -> None:
        """Clear all buffered events and errors."""
        self._events.clear()
        self._errors.clear()
        self._complete = False


class FilteringHandler(BaseStreamHandler):
    """Handler that filters events before delegating.

    Example:
        >>> handler = FilteringHandler(
        ...     delegate=ConsoleHandler(),
        ...     event_types={"think", "tool_complete"},
        ... )
    """

    def __init__(
        self,
        delegate: StreamHandler,
        event_types: set[str] | None = None,
        exclude_types: set[str] | None = None,
        filter_fn: Any | None = None,  # Callable[[LocusEvent], bool]
    ):
        """Initialize the filtering handler.

        Args:
            delegate: Handler to delegate matching events to
            event_types: If set, only these event types are forwarded
            exclude_types: If set, these event types are excluded
            filter_fn: Optional custom filter function
        """
        self.delegate = delegate
        self.event_types = event_types
        self.exclude_types = exclude_types or set()
        self.filter_fn = filter_fn

    def _should_forward(self, event: LocusEvent) -> bool:
        """Check if event should be forwarded."""
        event_type = event.event_type

        # Check exclude list
        if event_type in self.exclude_types:
            return False

        # Check include list
        if self.event_types is not None and event_type not in self.event_types:
            return False

        # Check custom filter
        if self.filter_fn is not None and not self.filter_fn(event):
            return False

        return True

    async def on_event(self, event: LocusEvent) -> None:
        """Forward event if it passes filters.

        Args:
            event: The event to filter and possibly forward
        """
        if self._should_forward(event):
            await self.delegate.on_event(event)

    async def on_complete(self) -> None:
        """Delegate completion."""
        await self.delegate.on_complete()

    async def on_error(self, error: Exception) -> None:
        """Delegate error.

        Args:
            error: The error that occurred
        """
        await self.delegate.on_error(error)
