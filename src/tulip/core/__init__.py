# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Core primitives for Tulip."""

# New primitives for graph control flow
from tulip.core.command import (
    Command,
    Continue,
    End,
    end,
    goto,
    is_command,
    normalize_node_output,
    resume_with,
)
from tulip.core.config import TulipSettings
from tulip.core.errors import (
    CheckpointError,
    CheckpointNotFoundError,
    CheckpointSerializationError,
    ConfigError,
    EmbeddingError,
    ModelAuthError,
    ModelError,
    ModelResponseError,
    ModelThrottledError,
    RAGError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolValidationError,
    TulipError,
    ValidationError,
    VectorStoreError,
)
from tulip.core.events import (
    GroundingEvent,
    ModelChunkEvent,
    ReflectEvent,
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
    ToolStartEvent,
    TulipEvent,
)
from tulip.core.interrupt import (
    AutoApproveHandler,
    GraphInterrupted,
    InterruptException,
    InterruptHandler,
    InterruptState,
    InterruptValue,
    interrupt,
)
from tulip.core.messages import Message, Role, ToolCall, ToolResult
from tulip.core.protocols import CheckpointerProtocol, ModelProtocol, ToolProtocol
from tulip.core.reducers import (
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
from tulip.core.send import (
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
from tulip.core.state import AgentState


__all__ = [
    # State
    "AgentState",
    # Protocols
    "CheckpointerProtocol",
    "ModelProtocol",
    "ToolProtocol",
    # Events
    "GroundingEvent",
    "TulipEvent",
    "ModelChunkEvent",
    "ReflectEvent",
    "TerminateEvent",
    "ThinkEvent",
    "ToolCompleteEvent",
    "ToolStartEvent",
    # Config
    "TulipSettings",
    # Errors
    "CheckpointError",
    "CheckpointNotFoundError",
    "CheckpointSerializationError",
    "ConfigError",
    "EmbeddingError",
    "TulipError",
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
