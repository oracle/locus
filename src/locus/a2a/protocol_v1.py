# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

# ruff: noqa: SLF001

"""A2A v1.0 protocol adapters.

This module contains the v1.0-specific JSON-RPC method handlers and
conversion helpers that bridge the canonical A2A v1.0 wire models to the
legacy-compatible Locus A2A models implemented in ``protocol.py``.
Keeping this code separate lets the main transport module focus on app
construction, auth, legacy routes, and shared client/server plumbing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, Protocol, cast

from locus.a2a.push import PushNotificationError
from locus.a2a.spec import (
    AUTHENTICATED_EXTENDED_CARD_NOT_CONFIGURED,
    CONTENT_TYPE_NOT_SUPPORTED,
    EXTENSION_SUPPORT_REQUIRED,
    INTERNAL_ERROR,
    INVALID_AGENT_RESPONSE,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    PUSH_NOTIFICATION_NOT_SUPPORTED,
    TASK_NOT_CANCELABLE,
    TASK_NOT_FOUND,
    UNSUPPORTED_OPERATION,
    VERSION_NOT_SUPPORTED,
    Artifact,
    DataPart,
    FilePart,
    FileWithBytes,
    FileWithUri,
    Message,
    Part,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from locus.a2a.spec_v1 import (
    A2A_V1_PROTOCOL_VERSION,
    A2AV1Artifact,
    A2AV1CancelTaskRequest,
    A2AV1DeleteTaskPushNotificationConfigRequest,
    A2AV1GetTaskPushNotificationConfigRequest,
    A2AV1GetTaskRequest,
    A2AV1ListTaskPushNotificationConfigsRequest,
    A2AV1ListTaskPushNotificationConfigsResponse,
    A2AV1ListTasksRequest,
    A2AV1ListTasksResponse,
    A2AV1Message,
    A2AV1Part,
    A2AV1Role,
    A2AV1SendMessageRequest,
    A2AV1StreamResponse,
    A2AV1SubscribeToTaskRequest,
    A2AV1Task,
    A2AV1TaskArtifactUpdateEvent,
    A2AV1TaskPushNotificationConfig,
    A2AV1TaskState,
    A2AV1TaskStatus,
    A2AV1TaskStatusUpdateEvent,
)


A2A_VERSION_HEADER = "A2A-Version"

#: Error code → (HTTP status for the HTTP+JSON binding, ``google.rpc.Code``
#: name for the gRPC binding, ``ErrorInfo.reason``). The reasons are the
#: specification's error names in UPPER_SNAKE_CASE without the ``Error``
#: suffix; every binding puts the same ``ErrorInfo`` detail on a failure so a
#: peer maps it back to the same error whichever binding it used.
A2A_ERROR_MAPPING: dict[int, tuple[int, str, str]] = {
    TASK_NOT_FOUND: (404, "NOT_FOUND", "TASK_NOT_FOUND"),
    TASK_NOT_CANCELABLE: (400, "FAILED_PRECONDITION", "TASK_NOT_CANCELABLE"),
    PUSH_NOTIFICATION_NOT_SUPPORTED: (
        400,
        "FAILED_PRECONDITION",
        "PUSH_NOTIFICATION_NOT_SUPPORTED",
    ),
    UNSUPPORTED_OPERATION: (400, "FAILED_PRECONDITION", "UNSUPPORTED_OPERATION"),
    CONTENT_TYPE_NOT_SUPPORTED: (400, "INVALID_ARGUMENT", "CONTENT_TYPE_NOT_SUPPORTED"),
    INVALID_AGENT_RESPONSE: (500, "INTERNAL", "INVALID_AGENT_RESPONSE"),
    AUTHENTICATED_EXTENDED_CARD_NOT_CONFIGURED: (
        400,
        "FAILED_PRECONDITION",
        "EXTENDED_AGENT_CARD_NOT_CONFIGURED",
    ),
    EXTENSION_SUPPORT_REQUIRED: (400, "FAILED_PRECONDITION", "EXTENSION_SUPPORT_REQUIRED"),
    VERSION_NOT_SUPPORTED: (400, "FAILED_PRECONDITION", "VERSION_NOT_SUPPORTED"),
    INVALID_PARAMS: (400, "INVALID_ARGUMENT", "INVALID_PARAMS"),
    INVALID_REQUEST: (400, "INVALID_ARGUMENT", "INVALID_REQUEST"),
    METHOD_NOT_FOUND: (404, "NOT_FOUND", "METHOD_NOT_FOUND"),
    INTERNAL_ERROR: (500, "INTERNAL", "INTERNAL_ERROR"),
    PARSE_ERROR: (400, "INVALID_ARGUMENT", "JSON_PARSE_ERROR"),
}

#: ``ErrorInfo.reason`` → JSON-RPC error code, the inverse of :data:`A2A_ERROR_MAPPING`.
A2A_REASON_TO_CODE: dict[str, int] = {
    reason: code for code, (_, _, reason) in A2A_ERROR_MAPPING.items()
}

#: ``google.rpc.Code`` names → numbers, for the HTTP+JSON error body.
GRPC_CODE_NUMBERS: dict[str, int] = {
    "OK": 0,
    "CANCELLED": 1,
    "UNKNOWN": 2,
    "INVALID_ARGUMENT": 3,
    "DEADLINE_EXCEEDED": 4,
    "NOT_FOUND": 5,
    "ALREADY_EXISTS": 6,
    "PERMISSION_DENIED": 7,
    "RESOURCE_EXHAUSTED": 8,
    "FAILED_PRECONDITION": 9,
    "ABORTED": 10,
    "OUT_OF_RANGE": 11,
    "UNIMPLEMENTED": 12,
    "INTERNAL": 13,
    "UNAVAILABLE": 14,
    "DATA_LOSS": 15,
    "UNAUTHENTICATED": 16,
}

A2A_ERROR_DOMAIN = "a2a-protocol.org"


def error_mapping(code: int) -> tuple[int, str, str]:
    """``(http_status, grpc_code_name, reason)`` for an A2A or JSON-RPC error code."""
    return A2A_ERROR_MAPPING.get(code, (500, "INTERNAL", "INTERNAL_ERROR"))


def error_details(code: int, message: str) -> list[dict[str, Any]]:
    """The ``google.rpc.ErrorInfo`` detail every binding attaches to an error."""
    _, _, reason = error_mapping(code)
    return [
        {
            "@type": "type.googleapis.com/google.rpc.ErrorInfo",
            "reason": reason,
            "domain": A2A_ERROR_DOMAIN,
            "metadata": {
                "message": message,
                "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            },
        }
    ]


def http_error_body(code: int, message: str) -> dict[str, Any]:
    """The HTTP+JSON binding's error body: ``google.rpc.Status`` in JSON form."""
    _, grpc_name, _ = error_mapping(code)
    return {
        "error": {
            "code": GRPC_CODE_NUMBERS[grpc_name],
            "status": grpc_name,
            "message": message,
            "details": error_details(code, message),
        }
    }


def code_from_error_details(details: Any, default: int = INTERNAL_ERROR) -> int:
    """The A2A error code named by an ``ErrorInfo`` detail list, or ``default``."""
    if isinstance(details, list):
        for detail in details:
            if isinstance(detail, dict) and detail.get("domain") == A2A_ERROR_DOMAIN:
                reason = detail.get("reason")
                if isinstance(reason, str) and reason in A2A_REASON_TO_CODE:
                    return A2A_REASON_TO_CODE[reason]
    return default


A2AV1_JSONRPC_METHODS = frozenset(
    {
        "SendMessage",
        "SendStreamingMessage",
        "GetTask",
        "ListTasks",
        "CancelTask",
        "SubscribeToTask",
        "CreateTaskPushNotificationConfig",
        "GetTaskPushNotificationConfig",
        "ListTaskPushNotificationConfigs",
        "DeleteTaskPushNotificationConfig",
        "GetExtendedAgentCard",
    }
)


class A2AV1ProtocolError(Exception):
    """Structured v1.0 JSON-RPC handler error."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _parse_iso_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as e:
        raise A2AV1ProtocolError(INVALID_PARAMS, f"invalid timestamp: {value!r}") from e
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


class _TaskStoreProtocol(Protocol):
    def list(self) -> list[Task]: ...

    def get(self, task_id: str) -> Task | None: ...


class _PushRegistryProtocol(Protocol):
    """What the v1 push-config handlers need from the host's push layer."""

    def register(
        self, task_id: str, config: A2AV1TaskPushNotificationConfig
    ) -> A2AV1TaskPushNotificationConfig: ...

    def get(self, task_id: str, config_id: str) -> A2AV1TaskPushNotificationConfig | None: ...

    def list(self, task_id: str) -> list[A2AV1TaskPushNotificationConfig]: ...

    def delete(self, task_id: str, config_id: str) -> bool: ...


class _A2AV1ServerHost(Protocol):
    _store: _TaskStoreProtocol

    def _build_card(self) -> Any: ...

    async def _handle_message_send(self, params: dict[str, Any]) -> dict[str, Any]: ...

    async def _handle_tasks_get(self, params: dict[str, Any]) -> dict[str, Any]: ...

    async def _handle_tasks_cancel(self, params: dict[str, Any]) -> dict[str, Any]: ...

    def _stream_message(self, params: dict[str, Any]) -> AsyncIterator[dict[str, Any]]: ...

    def _task_events(self, task_id: str) -> AsyncIterator[dict[str, Any]]: ...

    def _push_registry(self) -> _PushRegistryProtocol | None: ...


def _v1_role_to_legacy(role: A2AV1Role) -> str:
    if role == A2AV1Role.agent:
        return "agent"
    return "user"


def _legacy_role_to_v1(role: str) -> A2AV1Role:
    if role == "agent":
        return A2AV1Role.agent
    return A2AV1Role.user


def _v1_state_to_legacy(state: A2AV1TaskState) -> TaskState:
    mapping = {
        A2AV1TaskState.submitted: TaskState.submitted,
        A2AV1TaskState.working: TaskState.working,
        A2AV1TaskState.input_required: TaskState.input_required,
        A2AV1TaskState.completed: TaskState.completed,
        A2AV1TaskState.canceled: TaskState.canceled,
        A2AV1TaskState.failed: TaskState.failed,
        A2AV1TaskState.rejected: TaskState.rejected,
        A2AV1TaskState.auth_required: TaskState.auth_required,
    }
    return mapping.get(state, TaskState.submitted)


def _legacy_state_to_v1(state: TaskState) -> A2AV1TaskState:
    mapping = {
        TaskState.submitted: A2AV1TaskState.submitted,
        TaskState.working: A2AV1TaskState.working,
        TaskState.input_required: A2AV1TaskState.input_required,
        TaskState.completed: A2AV1TaskState.completed,
        TaskState.canceled: A2AV1TaskState.canceled,
        TaskState.failed: A2AV1TaskState.failed,
        TaskState.rejected: A2AV1TaskState.rejected,
        TaskState.auth_required: A2AV1TaskState.auth_required,
    }
    return mapping[state]


def v1_part_to_legacy(part: A2AV1Part) -> Part:
    if part.text is not None:
        return TextPart(text=part.text, metadata=part.metadata)
    if part.data is not None:
        return DataPart(data=part.data, metadata=part.metadata)
    if part.url is not None:
        return FilePart(
            file=FileWithUri(uri=part.url, mimeType=part.mediaType, name=part.filename),
            metadata=part.metadata,
        )
    if part.raw is not None:
        return FilePart(
            file=FileWithBytes(
                bytes=part.raw,
                mimeType=part.mediaType,
                name=part.filename,
            ),
            metadata=part.metadata,
        )
    return TextPart(text="", metadata=part.metadata)


def legacy_part_to_v1(part: Part) -> A2AV1Part:
    if isinstance(part, TextPart):
        return A2AV1Part(text=part.text, metadata=part.metadata)
    if isinstance(part, DataPart):
        return A2AV1Part(data=part.data, metadata=part.metadata)
    if isinstance(part, FilePart):
        file = part.file
        uri = getattr(file, "uri", None)
        if uri:
            return A2AV1Part(
                url=uri,
                filename=getattr(file, "name", None),
                mediaType=getattr(file, "mimeType", None),
                metadata=part.metadata,
            )
        return A2AV1Part(
            raw=getattr(file, "bytes", None),
            filename=getattr(file, "name", None),
            mediaType=getattr(file, "mimeType", None),
            metadata=part.metadata,
        )
    return A2AV1Part(text="")


def v1_message_to_legacy(message: A2AV1Message) -> Message:
    return Message(
        role=cast("Any", _v1_role_to_legacy(message.role)),
        parts=[v1_part_to_legacy(part) for part in message.parts],
        messageId=message.messageId,
        contextId=message.contextId,
        taskId=message.taskId,
        referenceTaskIds=message.referenceTaskIds,
        extensions=message.extensions,
        metadata=message.metadata,
    )


def legacy_message_to_v1(message: Message) -> A2AV1Message:
    """Convert a legacy A2A ``Message`` to its v1 equivalent."""
    return A2AV1Message(
        role=_legacy_role_to_v1(message.role),
        parts=[legacy_part_to_v1(part) for part in message.parts],
        messageId=message.messageId,
        contextId=message.contextId,
        taskId=message.taskId,
        referenceTaskIds=message.referenceTaskIds,
        extensions=message.extensions,
        metadata=message.metadata,
    )


def legacy_status_to_v1(status: TaskStatus) -> A2AV1TaskStatus:
    return A2AV1TaskStatus(
        state=_legacy_state_to_v1(status.state),
        message=legacy_message_to_v1(status.message) if status.message else None,
        timestamp=status.timestamp,
    )


def v1_status_to_legacy(status: A2AV1TaskStatus) -> TaskStatus:
    return TaskStatus(
        state=_v1_state_to_legacy(status.state),
        message=v1_message_to_legacy(status.message) if status.message else None,
        timestamp=status.timestamp,
    )


def legacy_artifact_to_v1(artifact: Artifact) -> A2AV1Artifact:
    return A2AV1Artifact(
        artifactId=artifact.artifactId,
        name=artifact.name,
        description=artifact.description,
        parts=[legacy_part_to_v1(part) for part in artifact.parts],
        metadata=artifact.metadata,
        extensions=artifact.extensions,
    )


def legacy_task_to_v1(task: Task) -> dict[str, Any]:
    return A2AV1Task(
        id=task.id,
        contextId=task.contextId,
        status=legacy_status_to_v1(task.status),
        history=[legacy_message_to_v1(message) for message in task.history],
        artifacts=[legacy_artifact_to_v1(artifact) for artifact in task.artifacts],
        metadata=task.metadata,
    ).model_dump(exclude_none=True)


def v1_artifact_to_legacy_payload(artifact: A2AV1Artifact | dict[str, Any]) -> dict[str, Any]:
    data = artifact if isinstance(artifact, dict) else artifact.model_dump(exclude_none=True)
    return {
        "artifactId": data["artifactId"],
        "name": data.get("name"),
        "description": data.get("description"),
        "parts": [
            v1_part_to_legacy(A2AV1Part.model_validate(part)).model_dump(exclude_none=True)
            for part in data.get("parts", [])
        ],
        "metadata": data.get("metadata"),
        "extensions": data.get("extensions"),
    }


def v1_task_to_legacy_payload(payload: dict[str, Any]) -> dict[str, Any]:
    task = A2AV1Task.model_validate(payload)
    return {
        "id": task.id,
        "contextId": task.contextId,
        "status": v1_status_to_legacy(task.status).model_dump(exclude_none=True),
        "history": [
            v1_message_to_legacy(message).model_dump(exclude_none=True) for message in task.history
        ],
        "artifacts": [v1_artifact_to_legacy_payload(artifact) for artifact in task.artifacts],
        "metadata": task.metadata,
        "kind": "task",
    }


def task_result_to_legacy_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Normalise a task result to the legacy payload shape.

    Accepts either wire format: a v1 task (whose ``state`` is prefixed
    ``TASK_STATE_``) is downconverted, and a payload already in legacy shape
    is returned untouched, so a caller need not know which peer it drew."""
    payload = result.get("task", result)
    state = payload.get("status", {}).get("state") if isinstance(payload, dict) else None
    if isinstance(state, str) and not state.startswith("TASK_STATE_"):
        return cast("dict[str, Any]", payload)
    return v1_task_to_legacy_payload(payload)


def v1_stream_response_to_legacy_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert one v1 streaming response frame to the legacy payload shape."""
    stream = A2AV1StreamResponse.model_validate(payload)
    if stream.task is not None:
        return v1_task_to_legacy_payload(stream.task.model_dump(exclude_none=True))
    if stream.statusUpdate is not None:
        status_update = stream.statusUpdate
        return {
            "taskId": status_update.taskId,
            "contextId": status_update.contextId,
            "kind": "status-update",
            "status": v1_status_to_legacy(status_update.status).model_dump(exclude_none=True),
            "metadata": status_update.metadata,
        }
    if stream.artifactUpdate is not None:
        artifact_update = stream.artifactUpdate
        return {
            "taskId": artifact_update.taskId,
            "contextId": artifact_update.contextId,
            "kind": "artifact-update",
            "artifact": v1_artifact_to_legacy_payload(artifact_update.artifact),
            "append": artifact_update.append,
            "lastChunk": artifact_update.lastChunk,
            "metadata": artifact_update.metadata,
        }
    if stream.message is not None:
        return v1_message_to_legacy(stream.message).model_dump(exclude_none=True)
    return payload


def legacy_event_to_v1_stream_response(event: dict[str, Any]) -> dict[str, Any]:
    if event.get("kind") == "task":
        task = Task.model_validate(event)
        return A2AV1StreamResponse(
            task=A2AV1Task.model_validate(legacy_task_to_v1(task))
        ).model_dump(exclude_none=True)
    if event.get("kind") == "status-update":
        legacy_status = TaskStatusUpdateEvent.model_validate(event)
        status_update = A2AV1TaskStatusUpdateEvent(
            taskId=legacy_status.taskId,
            contextId=legacy_status.contextId,
            status=legacy_status_to_v1(legacy_status.status),
            metadata=legacy_status.metadata,
        )
        return A2AV1StreamResponse(statusUpdate=status_update).model_dump(exclude_none=True)
    if event.get("kind") == "artifact-update":
        legacy_artifact = TaskArtifactUpdateEvent.model_validate(event)
        artifact_update = A2AV1TaskArtifactUpdateEvent(
            taskId=legacy_artifact.taskId,
            contextId=legacy_artifact.contextId,
            artifact=legacy_artifact_to_v1(legacy_artifact.artifact),
            append=legacy_artifact.append,
            lastChunk=legacy_artifact.lastChunk,
            metadata=legacy_artifact.metadata,
        )
        return A2AV1StreamResponse(artifactUpdate=artifact_update).model_dump(exclude_none=True)
    return event


class A2AV1ServerMixin:
    """Server-side A2A v1.0 method handlers."""

    def _v1_host(self) -> _A2AV1ServerHost:
        return cast("_A2AV1ServerHost", self)

    def _validate_v1_input_modes(self, message: A2AV1Message) -> None:
        allowed = set(self._v1_host()._build_card().defaultInputModes)
        for part in message.parts:
            media_type = part.mediaType or ("text/plain" if part.text is not None else None)
            if media_type is not None and media_type not in allowed:
                raise A2AV1ProtocolError(
                    CONTENT_TYPE_NOT_SUPPORTED,
                    f"content type {media_type!r} is not supported",
                )
            if part.raw is not None or part.url is not None:
                media_type = part.mediaType or "application/octet-stream"
                if media_type not in allowed:
                    raise A2AV1ProtocolError(
                        CONTENT_TYPE_NOT_SUPPORTED,
                        f"content type {media_type!r} is not supported",
                    )

    async def _handle_v1_send_message(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            send = A2AV1SendMessageRequest.model_validate(params)
        except Exception as e:  # noqa: BLE001
            raise A2AV1ProtocolError(INVALID_PARAMS, f"invalid params: {e}") from e
        self._validate_v1_input_modes(send.message)
        legacy_params = {
            "message": v1_message_to_legacy(send.message).model_dump(exclude_none=True),
            "configuration": (
                send.configuration.model_dump(exclude_none=True) if send.configuration else None
            ),
            "metadata": send.metadata,
        }
        try:
            result = await self._v1_host()._handle_message_send(legacy_params)
        except A2AV1ProtocolError as e:
            raise self._v1_send_error(e, send) from e
        return {"task": legacy_task_to_v1(Task.model_validate(result))}

    @staticmethod
    def _v1_send_error(
        error: A2AV1ProtocolError, send: A2AV1SendMessageRequest
    ) -> A2AV1ProtocolError:
        """The v1.0 code for a refused send.

        The legacy dispatcher answers a follow-up on a finished task with
        ``TaskNotCancelable``; the 1.0 specification names that case
        ``UnsupportedOperation``.
        """
        if error.code == TASK_NOT_CANCELABLE and send.message.taskId:
            return A2AV1ProtocolError(UNSUPPORTED_OPERATION, error.message)
        return error

    async def _handle_v1_get_task(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            q = A2AV1GetTaskRequest.model_validate(params)
        except Exception as e:  # noqa: BLE001
            raise A2AV1ProtocolError(INVALID_PARAMS, f"invalid params: {e}") from e
        task = Task.model_validate(
            await self._v1_host()._handle_tasks_get(
                {
                    "id": q.id,
                    "historyLength": q.historyLength,
                    "metadata": q.metadata,
                }
            )
        )
        # v1 GetTask returns the Task directly; SendMessage stays wrapped
        # because its response type is a oneof SendMessageResponse.
        return legacy_task_to_v1(task)

    async def _handle_v1_list_tasks(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            q = A2AV1ListTasksRequest.model_validate(params)
        except Exception as e:  # noqa: BLE001
            raise A2AV1ProtocolError(INVALID_PARAMS, f"invalid params: {e}") from e

        tasks = list(self._v1_host()._store.list())
        if q.contextId is not None:
            tasks = [task for task in tasks if task.contextId == q.contextId]
        if q.status is not None:
            desired = _v1_state_to_legacy(q.status)
            tasks = [task for task in tasks if task.status.state == desired]
        if q.statusTimestampAfter is not None:
            threshold = _parse_iso_timestamp(q.statusTimestampAfter)
            tasks = [
                task for task in tasks if _parse_iso_timestamp(task.status.timestamp) > threshold
            ]

        tasks.sort(key=lambda task: _parse_iso_timestamp(task.status.timestamp), reverse=True)
        total_size = len(tasks)
        try:
            # MVP pagination keeps pageToken as a stringified list offset; it
            # is intentionally not opaque so callers can resume deterministic
            # snapshots without server-side cursor state.
            offset = int(q.pageToken or "0")
        except ValueError as e:
            raise A2AV1ProtocolError(INVALID_PARAMS, "invalid pageToken") from e
        page_size = q.pageSize if q.pageSize and q.pageSize > 0 else total_size
        page = tasks[offset : offset + page_size]
        next_offset = offset + len(page)
        next_page_token = str(next_offset) if next_offset < total_size else ""

        converted = []
        for task in page:
            item = task.model_copy(deep=True)
            if q.historyLength is not None and q.historyLength >= 0:
                item.history = item.history[-q.historyLength :]
            if q.includeArtifacts is False:
                item.artifacts = []
            converted.append(A2AV1Task.model_validate(legacy_task_to_v1(item)))

        return A2AV1ListTasksResponse(
            tasks=converted,
            nextPageToken=next_page_token,
            pageSize=page_size,
            totalSize=total_size,
        ).model_dump(exclude_none=True)

    async def _handle_v1_cancel_task(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            q = A2AV1CancelTaskRequest.model_validate(params)
        except Exception as e:  # noqa: BLE001
            raise A2AV1ProtocolError(INVALID_PARAMS, f"invalid params: {e}") from e
        task = Task.model_validate(
            await self._v1_host()._handle_tasks_cancel({"id": q.id, "metadata": q.metadata})
        )
        # v1 CancelTask returns the Task directly, matching GetTask.
        return legacy_task_to_v1(task)

    async def _stream_v1_task_subscription(
        self, params: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        try:
            q = A2AV1SubscribeToTaskRequest.model_validate(params)
        except Exception as e:  # noqa: BLE001
            raise A2AV1ProtocolError(INVALID_PARAMS, f"invalid params: {e}") from e

        host = self._v1_host()
        task = host._store.get(q.id)
        if task is None:
            raise A2AV1ProtocolError(TASK_NOT_FOUND, f"task {q.id} not found")
        if task.status.state in {
            TaskState.completed,
            TaskState.canceled,
            TaskState.failed,
            TaskState.rejected,
        }:
            raise A2AV1ProtocolError(
                UNSUPPORTED_OPERATION,
                "cannot subscribe to a task in a terminal state",
            )
        # The snapshot first, then the task's live events until it reaches a
        # terminal or interrupted state. Subscribing before the snapshot is
        # taken means no event between the two is lost. A host without a
        # live feed serves the snapshot alone.
        feed = getattr(host, "_task_events", None)
        events = feed(q.id) if feed is not None else None
        yield A2AV1StreamResponse(
            task=A2AV1Task.model_validate(legacy_task_to_v1(task))
        ).model_dump(exclude_none=True)
        if events is None:
            return
        if task.status.state in {TaskState.input_required, TaskState.auth_required}:
            aclose = getattr(events, "aclose", None)
            if aclose is not None:
                await aclose()
            return
        async for event in events:
            yield legacy_event_to_v1_stream_response(event)

    def _preflight_v1_task_subscription(self, params: dict[str, Any]) -> None:
        try:
            q = A2AV1SubscribeToTaskRequest.model_validate(params)
        except Exception as e:  # noqa: BLE001
            raise A2AV1ProtocolError(INVALID_PARAMS, f"invalid params: {e}") from e
        task = self._v1_host()._store.get(q.id)
        if task is None:
            raise A2AV1ProtocolError(TASK_NOT_FOUND, f"task {q.id} not found")
        if task.status.state in {
            TaskState.completed,
            TaskState.canceled,
            TaskState.failed,
            TaskState.rejected,
        }:
            raise A2AV1ProtocolError(
                UNSUPPORTED_OPERATION,
                "cannot subscribe to a task in a terminal state",
            )

    # ----- push-notification configs ---------------------------------------

    def _push_or_raise(self) -> _PushRegistryProtocol:
        registry = self._v1_host()._push_registry()
        if registry is None:
            raise A2AV1ProtocolError(
                PUSH_NOTIFICATION_NOT_SUPPORTED,
                "push notifications are not enabled on this agent",
            )
        return registry

    def _task_or_raise(self, task_id: str) -> Task:
        task = self._v1_host()._store.get(task_id)
        if task is None:
            raise A2AV1ProtocolError(TASK_NOT_FOUND, f"task {task_id} not found")
        return task

    async def _handle_v1_create_push_config(self, params: dict[str, Any]) -> dict[str, Any]:
        registry = self._push_or_raise()
        try:
            config = A2AV1TaskPushNotificationConfig.model_validate(params)
        except Exception as e:  # noqa: BLE001
            raise A2AV1ProtocolError(INVALID_PARAMS, f"invalid params: {e}") from e
        if not config.taskId:
            raise A2AV1ProtocolError(INVALID_PARAMS, "taskId is required")
        self._task_or_raise(config.taskId)
        try:
            stored = registry.register(config.taskId, config)
        except (ValueError, PushNotificationError) as e:
            raise A2AV1ProtocolError(INVALID_PARAMS, str(e)) from e
        return stored.model_dump(exclude_none=True)

    async def _handle_v1_get_push_config(self, params: dict[str, Any]) -> dict[str, Any]:
        registry = self._push_or_raise()
        try:
            q = A2AV1GetTaskPushNotificationConfigRequest.model_validate(params)
        except Exception as e:  # noqa: BLE001
            raise A2AV1ProtocolError(INVALID_PARAMS, f"invalid params: {e}") from e
        self._task_or_raise(q.taskId)
        config = registry.get(q.taskId, q.id)
        if config is None:
            raise A2AV1ProtocolError(
                TASK_NOT_FOUND, f"push notification config {q.id} not found on task {q.taskId}"
            )
        return config.model_dump(exclude_none=True)

    async def _handle_v1_list_push_configs(self, params: dict[str, Any]) -> dict[str, Any]:
        registry = self._push_or_raise()
        try:
            q = A2AV1ListTaskPushNotificationConfigsRequest.model_validate(params)
        except Exception as e:  # noqa: BLE001
            raise A2AV1ProtocolError(INVALID_PARAMS, f"invalid params: {e}") from e
        self._task_or_raise(q.taskId)
        configs = registry.list(q.taskId)
        try:
            offset = int(q.pageToken or "0")
        except ValueError as e:
            raise A2AV1ProtocolError(INVALID_PARAMS, "invalid pageToken") from e
        page_size = q.pageSize if q.pageSize and q.pageSize > 0 else len(configs)
        page = configs[offset : offset + page_size]
        next_offset = offset + len(page)
        return A2AV1ListTaskPushNotificationConfigsResponse(
            configs=page,
            nextPageToken=str(next_offset) if next_offset < len(configs) else "",
        ).model_dump(exclude_none=True)

    async def _handle_v1_delete_push_config(self, params: dict[str, Any]) -> dict[str, Any]:
        registry = self._push_or_raise()
        try:
            q = A2AV1DeleteTaskPushNotificationConfigRequest.model_validate(params)
        except Exception as e:  # noqa: BLE001
            raise A2AV1ProtocolError(INVALID_PARAMS, f"invalid params: {e}") from e
        self._task_or_raise(q.taskId)
        # Deleting a config that is already gone is a success: the
        # specification makes deletion idempotent.
        registry.delete(q.taskId, q.id)
        return {}

    async def _handle_v1_extended_card(self, params: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        raise A2AV1ProtocolError(
            UNSUPPORTED_OPERATION,
            "authenticated extended agent cards are not enabled on this agent",
        )

    # ----- binding-neutral dispatch ------------------------------------------

    async def handle_v1(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Run one unary v1.0 method by its specification name.

        Every binding (JSON-RPC, HTTP+JSON, gRPC) lands here, so the task
        store, the push layer and the interrupt mapping behave the same
        whichever wire the peer used. Raises :class:`A2AV1ProtocolError`.
        """
        handlers = {
            "SendMessage": self._handle_v1_send_message,
            "GetTask": self._handle_v1_get_task,
            "ListTasks": self._handle_v1_list_tasks,
            "CancelTask": self._handle_v1_cancel_task,
            "CreateTaskPushNotificationConfig": self._handle_v1_create_push_config,
            "GetTaskPushNotificationConfig": self._handle_v1_get_push_config,
            "ListTaskPushNotificationConfigs": self._handle_v1_list_push_configs,
            "DeleteTaskPushNotificationConfig": self._handle_v1_delete_push_config,
            "GetExtendedAgentCard": self._handle_v1_extended_card,
        }
        handler = handlers.get(method)
        if handler is None:
            raise A2AV1ProtocolError(METHOD_NOT_FOUND, f"unknown method {method!r}")
        return await handler(params)

    def handle_v1_stream(
        self, method: str, params: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Run one streaming v1.0 method; yields ``StreamResponse`` payloads."""
        if method == "SendStreamingMessage":
            return self._stream_v1_message(params)
        if method == "SubscribeToTask":
            return self._stream_v1_task_subscription(params)

        async def _unknown() -> AsyncIterator[dict[str, Any]]:
            raise A2AV1ProtocolError(METHOD_NOT_FOUND, f"unknown method {method!r}")
            yield {}  # pragma: no cover

        return _unknown()

    async def _stream_v1_message(self, params: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Stream a v1.0 message run as v1.0 StreamResponse payloads."""
        try:
            send = A2AV1SendMessageRequest.model_validate(params)
        except Exception as e:  # noqa: BLE001
            raise A2AV1ProtocolError(INVALID_PARAMS, f"invalid params: {e}") from e
        self._validate_v1_input_modes(send.message)
        legacy_params = {
            "message": v1_message_to_legacy(send.message).model_dump(exclude_none=True),
            "configuration": (
                send.configuration.model_dump(exclude_none=True) if send.configuration else None
            ),
            "metadata": send.metadata,
        }
        try:
            async for event in self._v1_host()._stream_message(legacy_params):
                yield legacy_event_to_v1_stream_response(event)
        except A2AV1ProtocolError as e:
            raise self._v1_send_error(e, send) from e


A2AV1_STREAMING_METHODS = frozenset({"SendStreamingMessage", "SubscribeToTask"})


__all__ = [
    "A2A_ERROR_DOMAIN",
    "A2A_ERROR_MAPPING",
    "A2A_REASON_TO_CODE",
    "A2A_V1_PROTOCOL_VERSION",
    "A2A_VERSION_HEADER",
    "A2AV1_JSONRPC_METHODS",
    "A2AV1_STREAMING_METHODS",
    "GRPC_CODE_NUMBERS",
    "A2AV1ProtocolError",
    "A2AV1ServerMixin",
    "code_from_error_details",
    "error_details",
    "error_mapping",
    "http_error_body",
    "legacy_event_to_v1_stream_response",
    "legacy_message_to_v1",
    "task_result_to_legacy_payload",
    "v1_stream_response_to_legacy_payload",
]
