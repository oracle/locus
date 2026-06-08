# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Streaming handlers for Tulip.

Provides stream handlers for processing events during agent execution,
including console output, Server-Sent Events (SSE), and extensible base classes.
"""

from tulip.streaming.console import (
    ConsoleHandler,
    MinimalConsoleHandler,
)
from tulip.streaming.handler import (
    BaseStreamHandler,
    BufferingHandler,
    CompositeHandler,
    FilteringHandler,
    StreamHandler,
)
from tulip.streaming.sse import (
    AsyncSSEHandler,
    SSEHandler,
    SSEMessage,
    create_sse_response_headers,
)
from tulip.streaming.structured import StructuredStream, stream_structured


__all__ = [
    # Base handlers
    "StreamHandler",
    "BaseStreamHandler",
    "BufferingHandler",
    "CompositeHandler",
    "FilteringHandler",
    # Console handlers
    "ConsoleHandler",
    "MinimalConsoleHandler",
    # SSE handlers
    "SSEHandler",
    "AsyncSSEHandler",
    "SSEMessage",
    "create_sse_response_headers",
    # Structured streaming
    "StructuredStream",
    "stream_structured",
]
