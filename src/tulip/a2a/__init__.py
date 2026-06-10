# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Agent-to-Agent (A2A) protocol — spec-compliant cross-framework interop.

Implements the public A2A protocol (https://a2aproject.github.io/A2A/)
so Tulip agents can talk to peers from other frameworks (Strands, ADK,
Google A2A SDKs) without a translation shim.

Wire surface served by :class:`A2AServer`:

- ``GET  /.well-known/agent-card.json`` — public Agent Card (spec §5.5)
- ``POST /``                            — JSON-RPC 2.0 method dispatch
  for A2A v1.0 methods (``SendMessage``, ``SendStreamingMessage``,
  ``GetTask``, ``ListTasks``, ``CancelTask``, ``SubscribeToTask``)
  and compatibility methods (``message/send``, ``message/stream``,
  ``tasks/get``, ``tasks/cancel``)
- Backwards-compat: ``GET /agent-card``, ``POST /a2a/{invoke,stream}``

Re-exports the spec models so consumers can do ``from tulip.a2a import
AgentCard, AgentSkill, Message, TextPart, Task`` directly. The
``A2AV1*`` types are the canonical A2A v1.0 wire models and are stable
public API for cross-framework interoperability.
"""

from tulip.a2a.protocol import (
    A2AClient,
    A2AMessage,
    A2ARequest,
    A2AResponse,
    A2AServer,
)
from tulip.a2a.spec import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    Artifact,
    DataPart,
    FilePart,
    FileWithBytes,
    FileWithUri,
    JsonRpcError,
    JsonRpcErrorResponse,
    JsonRpcRequest,
    JsonRpcSuccessResponse,
    Message,
    MessageSendConfiguration,
    MessageSendParams,
    Part,
    PushNotificationAuthenticationInfo,
    PushNotificationConfig,
    Task,
    TaskArtifactUpdateEvent,
    TaskIdParams,
    TaskPushNotificationConfig,
    TaskQueryParams,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from tulip.a2a.spec_v1 import (
    A2A_V1_PROTOCOL_VERSION,
    A2AV1Artifact,
    A2AV1CancelTaskRequest,
    A2AV1GetTaskRequest,
    A2AV1ListTasksRequest,
    A2AV1ListTasksResponse,
    A2AV1Message,
    A2AV1Part,
    A2AV1Role,
    A2AV1SendMessageConfiguration,
    A2AV1SendMessageRequest,
    A2AV1SendMessageResponse,
    A2AV1StreamResponse,
    A2AV1Task,
    A2AV1TaskArtifactUpdateEvent,
    A2AV1TaskState,
    A2AV1TaskStatus,
    A2AV1TaskStatusUpdateEvent,
)


__all__ = [
    "A2A_V1_PROTOCOL_VERSION",
    "A2AV1Artifact",
    "A2AV1CancelTaskRequest",
    "A2AV1GetTaskRequest",
    "A2AV1ListTasksRequest",
    "A2AV1ListTasksResponse",
    "A2AV1Message",
    "A2AV1Part",
    "A2AV1Role",
    "A2AV1SendMessageConfiguration",
    "A2AV1SendMessageRequest",
    "A2AV1SendMessageResponse",
    "A2AV1StreamResponse",
    "A2AV1Task",
    "A2AV1TaskArtifactUpdateEvent",
    "A2AV1TaskState",
    "A2AV1TaskStatus",
    "A2AV1TaskStatusUpdateEvent",
    # Server / client.
    "A2AClient",
    "A2AServer",
    # Spec models.
    "AgentCapabilities",
    "AgentCard",
    "AgentInterface",
    "AgentProvider",
    "AgentSkill",
    "Artifact",
    "DataPart",
    "FilePart",
    "FileWithBytes",
    "FileWithUri",
    "JsonRpcError",
    "JsonRpcErrorResponse",
    "JsonRpcRequest",
    "JsonRpcSuccessResponse",
    "Message",
    "MessageSendConfiguration",
    "MessageSendParams",
    "Part",
    "PushNotificationAuthenticationInfo",
    "PushNotificationConfig",
    "Task",
    "TaskArtifactUpdateEvent",
    "TaskIdParams",
    "TaskPushNotificationConfig",
    "TaskQueryParams",
    "TaskState",
    "TaskStatus",
    "TaskStatusUpdateEvent",
    "TextPart",
    # Legacy flat models (kept for the pre-spec wire surface).
    "A2AMessage",
    "A2ARequest",
    "A2AResponse",
]
