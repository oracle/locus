# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: N815 — every camelCase field name here is the public A2A
# wire format. Renaming to snake_case would either break interop or
# require a per-field ``Field(alias=...)`` on every line; the file-level
# directive keeps the spec definitions readable and 1:1 with the
# published JSON schema.

"""Public A2A protocol type system — spec-compliant.

Pydantic models for the Agent-to-Agent protocol as published at
https://a2aproject.github.io/A2A/. Covers:

- Agent Card (capabilities, skills, provider, modes)
- Message parts: TextPart, FilePart, DataPart (discriminated on ``kind``)
- Message + Task + Artifact + TaskStatus
- TaskState lifecycle (8 states)
- Streaming events: TaskStatusUpdateEvent, TaskArtifactUpdateEvent
- JSON-RPC 2.0 envelopes + standard + A2A-specific error codes
- Method param shapes: MessageSendParams, TaskQueryParams,
  TaskIdParams, TaskPushNotificationConfig

These types are wire-format models — every field name matches the spec
verbatim. Pydantic ``alias_generator`` is **not** used because the spec
uses ``camelCase`` and we keep that on the wire (matching peers from
other frameworks). Internal Python code can either use the same names
or set up its own aliasing layer.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 — envelopes + error codes
# ---------------------------------------------------------------------------


# Standard JSON-RPC errors.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# A2A-specific errors (per spec §10.2).
TASK_NOT_FOUND = -32001
TASK_NOT_CANCELABLE = -32002
PUSH_NOTIFICATION_NOT_SUPPORTED = -32003
UNSUPPORTED_OPERATION = -32004
CONTENT_TYPE_NOT_SUPPORTED = -32005
INVALID_AGENT_RESPONSE = -32006
AUTHENTICATED_EXTENDED_CARD_NOT_CONFIGURED = -32007
EXTENSION_SUPPORT_REQUIRED = -32008
VERSION_NOT_SUPPORTED = -32009


class JsonRpcRequest(BaseModel):
    """A JSON-RPC 2.0 request envelope.

    The id MAY be omitted for notifications, but A2A's request methods
    always require a response so callers should always send one.
    """

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int | None = None
    method: str
    params: dict[str, Any] | list[Any] | None = None


class JsonRpcError(BaseModel):
    """JSON-RPC 2.0 error object."""

    code: int
    message: str
    data: Any | None = None


class JsonRpcSuccessResponse(BaseModel):
    """JSON-RPC 2.0 successful response."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int | None
    result: Any


class JsonRpcErrorResponse(BaseModel):
    """JSON-RPC 2.0 error response."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int | None
    error: JsonRpcError


# ---------------------------------------------------------------------------
# Agent Card
# ---------------------------------------------------------------------------


class AgentProvider(BaseModel):
    """The organization / publisher behind the agent."""

    organization: str
    url: str | None = None


class AgentCapabilities(BaseModel):
    """Optional protocol-level capabilities the agent supports.

    Per spec, these are **declarations**: a peer queries the agent card
    to know whether to attempt streaming or push-notification flows.
    """

    streaming: bool = False
    pushNotifications: bool = False
    stateTransitionHistory: bool = False
    extensions: list[str] = Field(default_factory=list)


class AgentSkill(BaseModel):
    """A discrete capability the agent advertises in its card."""

    id: str
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    inputModes: list[str] = Field(default_factory=list)
    outputModes: list[str] = Field(default_factory=list)


class AgentInterface(BaseModel):
    """A concrete protocol binding exposed by an A2A v1.0 agent."""

    protocolBinding: str = "JSONRPC"
    url: str
    tenant: str | None = None
    protocolVersion: str = "1.0"


class AgentCard(BaseModel):
    """Public Agent Card (spec §5.5).

    Published at ``/.well-known/agent-card.json``. The legacy
    ``/agent-card`` endpoint serves the same payload for backwards
    compatibility with peers that haven't picked up the well-known URL.
    """

    name: str
    description: str
    url: str
    provider: AgentProvider | None = None
    iconUrl: str | None = None
    version: str = "0.1.0"
    protocolVersion: str = "1.0"
    documentationUrl: str | None = None
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    securitySchemes: dict[str, Any] = Field(default_factory=dict)
    security: list[dict[str, list[str]]] | None = None
    securityRequirements: list[dict[str, list[str]]] = Field(default_factory=list)
    supportedInterfaces: list[AgentInterface] = Field(default_factory=list)
    defaultInputModes: list[str] = Field(default_factory=lambda: ["text/plain"])
    defaultOutputModes: list[str] = Field(default_factory=lambda: ["text/plain"])
    skills: list[AgentSkill] = Field(default_factory=list)
    supportsAuthenticatedExtendedCard: bool = False


# ---------------------------------------------------------------------------
# Message parts (discriminated on ``kind``)
# ---------------------------------------------------------------------------


class TextPart(BaseModel):
    """A plain-text message part."""

    kind: Literal["text"] = "text"
    text: str
    metadata: dict[str, Any] | None = None


class FileWithBytes(BaseModel):
    """A file referenced by inline base64 bytes."""

    bytes: str  # base64
    mimeType: str | None = None
    name: str | None = None


class FileWithUri(BaseModel):
    """A file referenced by URI."""

    uri: str
    mimeType: str | None = None
    name: str | None = None


class FilePart(BaseModel):
    """A file message part — either inline bytes or a URI reference."""

    kind: Literal["file"] = "file"
    file: FileWithBytes | FileWithUri
    metadata: dict[str, Any] | None = None


class DataPart(BaseModel):
    """A structured-data (e.g. JSON) message part."""

    kind: Literal["data"] = "data"
    data: dict[str, Any]
    metadata: dict[str, Any] | None = None


Part = Annotated[TextPart | FilePart | DataPart, Field(discriminator="kind")]


# ---------------------------------------------------------------------------
# Message + Task + Artifact + status
# ---------------------------------------------------------------------------


class TaskState(StrEnum):
    """Task lifecycle states (spec §6.3)."""

    submitted = "submitted"
    working = "working"
    input_required = "input-required"
    completed = "completed"
    canceled = "canceled"
    failed = "failed"
    rejected = "rejected"
    auth_required = "auth-required"


class Message(BaseModel):
    """A user/agent message with one or more typed parts (spec §6.4)."""

    model_config = ConfigDict(populate_by_name=True)

    role: Literal["user", "agent"]
    parts: list[Part]
    messageId: str
    contextId: str | None = None
    taskId: str | None = None
    referenceTaskIds: list[str] | None = None
    extensions: list[str] | None = None
    metadata: dict[str, Any] | None = None
    kind: Literal["message"] = "message"


class TaskStatus(BaseModel):
    """Status block on a Task (spec §6.2)."""

    state: TaskState
    message: Message | None = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )


class Artifact(BaseModel):
    """A typed result attached to a Task (spec §6.7)."""

    artifactId: str
    name: str | None = None
    description: str | None = None
    parts: list[Part]
    metadata: dict[str, Any] | None = None
    extensions: list[str] | None = None


class Task(BaseModel):
    """A unit of work tracked through the lifecycle (spec §6.1)."""

    id: str
    contextId: str
    status: TaskStatus
    history: list[Message] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None
    kind: Literal["task"] = "task"


# ---------------------------------------------------------------------------
# Streaming events (sent via SSE on message/stream + tasks/resubscribe)
# ---------------------------------------------------------------------------


class TaskStatusUpdateEvent(BaseModel):
    """Status transition event in the SSE stream (spec §7.2.2)."""

    taskId: str
    contextId: str
    kind: Literal["status-update"] = "status-update"
    status: TaskStatus
    final: bool = False
    metadata: dict[str, Any] | None = None


class TaskArtifactUpdateEvent(BaseModel):
    """Artifact-attach event in the SSE stream (spec §7.2.3)."""

    taskId: str
    contextId: str
    kind: Literal["artifact-update"] = "artifact-update"
    artifact: Artifact
    append: bool = False
    lastChunk: bool = False
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Method params
# ---------------------------------------------------------------------------


class MessageSendConfiguration(BaseModel):
    """Optional per-call configuration (spec §7.1.4)."""

    acceptedOutputModes: list[str] | None = None
    historyLength: int | None = None
    blocking: bool | None = None


class MessageSendParams(BaseModel):
    """Parameters for ``message/send`` and ``message/stream`` (spec §7.1)."""

    message: Message
    configuration: MessageSendConfiguration | None = None
    metadata: dict[str, Any] | None = None


class TaskIdParams(BaseModel):
    """Identifier-only params (cancel, push-config get/list/delete)."""

    id: str
    metadata: dict[str, Any] | None = None


class TaskQueryParams(BaseModel):
    """Parameters for ``tasks/get`` (spec §7.3)."""

    id: str
    historyLength: int | None = None
    metadata: dict[str, Any] | None = None


class PushNotificationAuthenticationInfo(BaseModel):
    """Auth shape for push-notification webhook delivery."""

    schemes: list[str]
    credentials: str | None = None


class PushNotificationConfig(BaseModel):
    """Webhook config attached to a Task for async updates."""

    url: str
    token: str | None = None
    authentication: PushNotificationAuthenticationInfo | None = None


class TaskPushNotificationConfig(BaseModel):
    """Bundle: which task + the webhook config (spec §7.5)."""

    taskId: str
    pushNotificationConfig: PushNotificationConfig


__all__ = [
    "AUTHENTICATED_EXTENDED_CARD_NOT_CONFIGURED",
    "CONTENT_TYPE_NOT_SUPPORTED",
    "INTERNAL_ERROR",
    "INVALID_AGENT_RESPONSE",
    "INVALID_PARAMS",
    "INVALID_REQUEST",
    "METHOD_NOT_FOUND",
    "PARSE_ERROR",
    "PUSH_NOTIFICATION_NOT_SUPPORTED",
    "TASK_NOT_CANCELABLE",
    "TASK_NOT_FOUND",
    "UNSUPPORTED_OPERATION",
    "EXTENSION_SUPPORT_REQUIRED",
    "VERSION_NOT_SUPPORTED",
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
]
