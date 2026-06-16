# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Streaming handlers for Locus.

Provides stream handlers for processing events during agent execution,
including console output, Server-Sent Events (SSE), and extensible base classes.
"""

from locus.streaming.console import (
    ConsoleHandler,
    MinimalConsoleHandler,
)
from locus.streaming.handler import (
    BaseStreamHandler,
    BufferingHandler,
    CompositeHandler,
    FilteringHandler,
    StreamHandler,
)
from locus.streaming.sse import (
    AsyncSSEHandler,
    SSEHandler,
    SSEMessage,
    create_sse_response_headers,
)
from locus.streaming.structured import StructuredStream, stream_structured


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
