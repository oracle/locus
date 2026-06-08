# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

# ruff: noqa: N815 — every camelCase field name here is the public A2A
# v1.0 wire format.

"""A2A v1.0 canonical wire type system.

Pydantic models for the Agent-to-Agent protocol v1.0 wire format as
implemented by the reference A2A SDK. Covers:

- Protocol version constant used for headers and Agent Card declarations
- Role and task-state enum names as published on the v1.0 wire
- Message parts: text, data, raw bytes payloads, and URL references
- Message + Task + Artifact + TaskStatus
- Streaming events: TaskStatusUpdateEvent and TaskArtifactUpdateEvent
- Method param and response shapes for SendMessage, GetTask, CancelTask,
  SendMessageResponse, and StreamResponse

These types intentionally mirror the v1.0 JSON/protobuf wire names so Tulip
can interoperate with reference A2A clients and servers. They live separately
from ``tulip.a2a.spec`` because the earlier Tulip A2A models preserve the
pre-v1/legacy-compatible shape used by existing callers.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


A2A_V1_PROTOCOL_VERSION = "1.0"


class A2AV1Role(StrEnum):
    """A2A v1.0 message role enum names."""

    unspecified = "ROLE_UNSPECIFIED"
    user = "ROLE_USER"
    agent = "ROLE_AGENT"


class A2AV1TaskState(StrEnum):
    """A2A v1.0 task state enum names."""

    unspecified = "TASK_STATE_UNSPECIFIED"
    submitted = "TASK_STATE_SUBMITTED"
    working = "TASK_STATE_WORKING"
    input_required = "TASK_STATE_INPUT_REQUIRED"
    completed = "TASK_STATE_COMPLETED"
    canceled = "TASK_STATE_CANCELED"
    failed = "TASK_STATE_FAILED"
    rejected = "TASK_STATE_REJECTED"
    auth_required = "TASK_STATE_AUTH_REQUIRED"


class A2AV1Part(BaseModel):
    """A2A v1.0 Part oneof.

    Exactly one of ``text``, ``data``, ``raw`` or ``url`` should be set by
    callers. The model is intentionally permissive enough to round-trip
    extension fields that are outside Tulip's plain-text default.
    """

    text: str | None = None
    data: Any | None = None
    raw: str | None = None
    url: str | None = None
    filename: str | None = None
    mediaType: str | None = None
    metadata: dict[str, Any] | None = None


class A2AV1Message(BaseModel):
    """A2A v1.0 Message."""

    messageId: str
    role: A2AV1Role
    parts: list[A2AV1Part]
    contextId: str | None = None
    taskId: str | None = None
    referenceTaskIds: list[str] | None = None
    extensions: list[str] | None = None
    metadata: dict[str, Any] | None = None


class A2AV1TaskStatus(BaseModel):
    """A2A v1.0 TaskStatus."""

    state: A2AV1TaskState
    message: A2AV1Message | None = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )


class A2AV1Artifact(BaseModel):
    """A2A v1.0 Artifact."""

    artifactId: str
    name: str | None = None
    description: str | None = None
    parts: list[A2AV1Part]
    metadata: dict[str, Any] | None = None
    extensions: list[str] | None = None


class A2AV1Task(BaseModel):
    """A2A v1.0 Task."""

    id: str
    contextId: str
    status: A2AV1TaskStatus
    history: list[A2AV1Message] = Field(default_factory=list)
    artifacts: list[A2AV1Artifact] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None


class A2AV1TaskStatusUpdateEvent(BaseModel):
    """A2A v1.0 TaskStatusUpdateEvent.

    v1.0 stream completion is inferred from task state and stream closure;
    the pre-v1 ``final`` field is intentionally absent.
    """

    taskId: str
    contextId: str
    status: A2AV1TaskStatus
    metadata: dict[str, Any] | None = None


class A2AV1TaskArtifactUpdateEvent(BaseModel):
    """A2A v1.0 TaskArtifactUpdateEvent."""

    taskId: str
    contextId: str
    artifact: A2AV1Artifact
    append: bool = False
    lastChunk: bool = False
    metadata: dict[str, Any] | None = None


class A2AV1SendMessageConfiguration(BaseModel):
    """A2A v1.0 SendMessage configuration."""

    model_config = ConfigDict(populate_by_name=True)

    acceptedOutputModes: list[str] | None = None
    historyLength: int | None = Field(
        default=None,
        validation_alias=AliasChoices("historyLength", "history_length"),
    )
    blocking: bool | None = None
    returnImmediately: bool | None = None


class A2AV1SendMessageRequest(BaseModel):
    """A2A v1.0 SendMessage request params."""

    tenant: str | None = None
    message: A2AV1Message
    configuration: A2AV1SendMessageConfiguration | None = None
    metadata: dict[str, Any] | None = None


class A2AV1GetTaskRequest(BaseModel):
    """A2A v1.0 GetTask request params."""

    model_config = ConfigDict(populate_by_name=True)

    tenant: str | None = None
    id: str
    historyLength: int | None = Field(
        default=None,
        validation_alias=AliasChoices("historyLength", "history_length"),
    )
    metadata: dict[str, Any] | None = None


class A2AV1ListTasksRequest(BaseModel):
    """A2A v1.0 ListTasks request params."""

    model_config = ConfigDict(populate_by_name=True)

    tenant: str | None = None
    contextId: str | None = Field(
        default=None,
        validation_alias=AliasChoices("contextId", "context_id"),
    )
    status: A2AV1TaskState | None = None
    pageSize: int | None = Field(
        default=None,
        validation_alias=AliasChoices("pageSize", "page_size"),
    )
    pageToken: str | None = Field(
        default=None,
        validation_alias=AliasChoices("pageToken", "page_token"),
    )
    historyLength: int | None = Field(
        default=None,
        validation_alias=AliasChoices("historyLength", "history_length"),
    )
    statusTimestampAfter: str | None = Field(
        default=None,
        validation_alias=AliasChoices("statusTimestampAfter", "status_timestamp_after"),
    )
    includeArtifacts: bool | None = Field(
        default=None,
        validation_alias=AliasChoices("includeArtifacts", "include_artifacts"),
    )
    metadata: dict[str, Any] | None = None


class A2AV1ListTasksResponse(BaseModel):
    """A2A v1.0 ListTasks response."""

    tasks: list[A2AV1Task]
    nextPageToken: str = ""
    pageSize: int
    totalSize: int


class A2AV1CancelTaskRequest(BaseModel):
    """A2A v1.0 CancelTask request params."""

    tenant: str | None = None
    id: str
    metadata: dict[str, Any] | None = None


class A2AV1SubscribeToTaskRequest(BaseModel):
    """A2A v1.0 SubscribeToTask request params."""

    tenant: str | None = None
    id: str
    metadata: dict[str, Any] | None = None


class A2AV1SendMessageResponse(BaseModel):
    """A2A v1.0 SendMessageResponse oneof."""

    task: A2AV1Task | None = None
    message: A2AV1Message | None = None


class A2AV1StreamResponse(BaseModel):
    """A2A v1.0 StreamResponse oneof."""

    task: A2AV1Task | None = None
    message: A2AV1Message | None = None
    statusUpdate: A2AV1TaskStatusUpdateEvent | None = None
    artifactUpdate: A2AV1TaskArtifactUpdateEvent | None = None


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
    "A2AV1SubscribeToTaskRequest",
    "A2AV1StreamResponse",
    "A2AV1Task",
    "A2AV1TaskArtifactUpdateEvent",
    "A2AV1TaskState",
    "A2AV1TaskStatus",
    "A2AV1TaskStatusUpdateEvent",
]
