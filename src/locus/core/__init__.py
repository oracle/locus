# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Core primitives for Locus."""

# New primitives for graph control flow
from locus.core.command import (
    Command,
    Continue,
    End,
    end,
    goto,
    is_command,
    normalize_node_output,
    resume_with,
)
from locus.core.config import LocusSettings
from locus.core.errors import (
    CheckpointError,
    CheckpointNotFoundError,
    CheckpointSerializationError,
    ConfigError,
    EmbeddingError,
    LocusError,
    ModelAuthError,
    ModelError,
    ModelResponseError,
    ModelThrottledError,
    RAGError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolValidationError,
    ValidationError,
    VectorStoreError,
)
from locus.core.events import (
    GroundingEvent,
    LocusEvent,
    ModelChunkEvent,
    ReflectEvent,
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
    ToolStartEvent,
)
from locus.core.interrupt import (
    AutoApproveHandler,
    GraphInterrupted,
    InterruptException,
    InterruptHandler,
    InterruptState,
    InterruptValue,
    interrupt,
)
from locus.core.messages import Message, Role, ToolCall, ToolResult
from locus.core.protocols import CheckpointerProtocol, ModelProtocol, ToolProtocol
from locus.core.reducers import (
    Reducer,
    add_messages,
    add_numbers,
    append_list,
    apply_reducers,
    deep_merge_dict,
    first_value,
    get_reducer,
    last_value,
    max_value,
    merge_dict,
    min_value,
    reducer,
    set_union,
    unique_append_list,
)
from locus.core.send import (
    Send,
    SendBatch,
    SendResult,
    aggregate_send_results,
    broadcast,
    extract_send_results,
    is_send,
    is_send_list,
    normalize_sends,
    scatter,
    send,
)
from locus.core.state import AgentState


__all__ = [
    # State
    "AgentState",
    # Protocols
    "CheckpointerProtocol",
    "ModelProtocol",
    "ToolProtocol",
    # Events
    "GroundingEvent",
    "LocusEvent",
    "ModelChunkEvent",
    "ReflectEvent",
    "TerminateEvent",
    "ThinkEvent",
    "ToolCompleteEvent",
    "ToolStartEvent",
    # Config
    "LocusSettings",
    # Errors
    "CheckpointError",
    "CheckpointNotFoundError",
    "CheckpointSerializationError",
    "ConfigError",
    "EmbeddingError",
    "LocusError",
    "ModelAuthError",
    "ModelError",
    "ModelResponseError",
    "ModelThrottledError",
    "RAGError",
    "ToolError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolValidationError",
    "ValidationError",
    "VectorStoreError",
    # Messages
    "Message",
    "Role",
    "ToolCall",
    "ToolResult",
    # Command (control flow)
    "Command",
    "End",
    "Continue",
    "goto",
    "end",
    "resume_with",
    "is_command",
    "normalize_node_output",
    # Interrupt (HITL)
    "interrupt",
    "InterruptException",
    "InterruptValue",
    "InterruptState",
    "GraphInterrupted",
    "InterruptHandler",
    "AutoApproveHandler",
    # Send (map-reduce)
    "Send",
    "SendResult",
    "SendBatch",
    "send",
    "broadcast",
    "scatter",
    "is_send",
    "is_send_list",
    "normalize_sends",
    "extract_send_results",
    "aggregate_send_results",
    # Reducers
    "Reducer",
    "add_messages",
    "merge_dict",
    "deep_merge_dict",
    "append_list",
    "unique_append_list",
    "add_numbers",
    "max_value",
    "min_value",
    "last_value",
    "first_value",
    "set_union",
    "reducer",
    "get_reducer",
    "apply_reducers",
]
