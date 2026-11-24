# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The A2A v1.0 HTTP+JSON (REST) binding, mounted on the A2AServer's FastAPI app.

Each specification method is a resource path with a verb; bodies and
responses are the v1.0 JSON shapes with no envelope, served as
``application/a2a+json``. Errors are ``google.rpc.Status`` in JSON form,
with the HTTP status the specification assigns and an ``ErrorInfo`` detail
carrying the same reason the JSON-RPC and gRPC bindings send:

| Method | Route |
|---|---|
| ``SendMessage`` | ``POST /message:send`` |
| ``SendStreamingMessage`` | ``POST /message:stream`` (server-sent events) |
| ``GetTask`` | ``GET /tasks/{id}`` |
| ``ListTasks`` | ``GET /tasks`` |
| ``CancelTask`` | ``POST /tasks/{id}:cancel`` |
| ``SubscribeToTask`` | ``GET`` or ``POST /tasks/{id}:subscribe`` (server-sent events) |
| ``CreateTaskPushNotificationConfig`` | ``POST /tasks/{id}/pushNotificationConfigs`` |
| ``GetTaskPushNotificationConfig`` | ``GET /tasks/{id}/pushNotificationConfigs/{configId}`` |
| ``ListTaskPushNotificationConfigs`` | ``GET /tasks/{id}/pushNotificationConfigs`` |
| ``DeleteTaskPushNotificationConfig`` | ``DELETE /tasks/{id}/pushNotificationConfigs/{configId}`` |
| ``GetExtendedAgentCard`` | ``GET /extendedAgentCard`` |

Streams send one ``data: <StreamResponse>`` event per update; an error part
way through is sent as a final ``event: error`` event whose ``data`` is the
same ``google.rpc.Status`` body.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any

# ``Request`` must be a module-level name: with postponed annotations FastAPI
# resolves ``request: Request`` through this module's globals, and a name local
# to ``mount_http_json`` would be read as a query parameter.
from fastapi import Request

from locus.a2a.protocol_v1 import (
    A2A_VERSION_HEADER,
    A2AV1ProtocolError,
    error_mapping,
    http_error_body,
)
from locus.a2a.spec import INTERNAL_ERROR, INVALID_REQUEST, PARSE_ERROR, VERSION_NOT_SUPPORTED
from locus.a2a.spec_v1 import A2A_JSON_MEDIA_TYPE, A2A_V1_PROTOCOL_VERSION
from locus.observability.propagation import trace_scope


if TYPE_CHECKING:
    from fastapi import FastAPI

    from locus.a2a.protocol import A2AServer

_logger = logging.getLogger(__name__)

_INT_QUERY = frozenset({"historyLength", "pageSize"})
_BOOL_QUERY = frozenset({"includeArtifacts"})


def query_to_params(query: Mapping[str, str]) -> dict[str, Any]:
    """Typed v1.0 params from a request's query string (camelCase keys)."""
    params: dict[str, Any] = {}
    for key, value in query.items():
        if key in _INT_QUERY:
            try:
                params[key] = int(value)
            except ValueError as e:
                raise A2AV1ProtocolError(INVALID_REQUEST, f"{key} must be an integer") from e
        elif key in _BOOL_QUERY:
            params[key] = value.lower() in ("true", "1", "yes")
        else:
            params[key] = value
    return params


def mount_http_json(app: FastAPI, server: A2AServer, auth_dep: Any) -> None:
    """Add the HTTP+JSON routes for ``server`` to ``app``.

    ``auth_dep`` is the bearer dependency the JSON-RPC routes use, so the
    two bindings share one credential.
    """
    from fastapi import Header
    from fastapi.responses import JSONResponse, StreamingResponse

    def error_response(code: int, message: str) -> JSONResponse:
        status, _, _ = error_mapping(code)
        return JSONResponse(
            status_code=status,
            content=http_error_body(code, message),
            media_type=A2A_JSON_MEDIA_TYPE,
        )

    def ok(payload: dict[str, Any]) -> JSONResponse:
        return JSONResponse(content=payload, media_type=A2A_JSON_MEDIA_TYPE)

    def check_version(version: str | None) -> None:
        if version not in (None, A2A_V1_PROTOCOL_VERSION):
            raise A2AV1ProtocolError(
                VERSION_NOT_SUPPORTED, f"A2A version {version!r} is not supported"
            )

    async def body_of(request: Request) -> dict[str, Any]:
        raw = await request.body()
        if not raw.strip():
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise A2AV1ProtocolError(PARSE_ERROR, "request body is not valid JSON") from e
        if not isinstance(data, dict):
            raise A2AV1ProtocolError(INVALID_REQUEST, "request body must be a JSON object")
        return data

    async def unary(
        method: str, params: dict[str, Any], version: str | None, request: Request
    ) -> JSONResponse:
        # Lazy import — observability is opt-in.
        from locus.observability.emit import (  # noqa: PLC0415
            EV_A2A_TASK_COMPLETED,
            EV_A2A_TASK_RECEIVED,
            emit,
        )

        # Run under the caller's trace, if its headers carry one.
        with trace_scope(request.headers):
            # The same bracket the JSON-RPC binding publishes; this binding
            # has no request id, so ``rpc_id`` is null.
            await emit(EV_A2A_TASK_RECEIVED, method=method, rpc_id=None)
            started = time.perf_counter()
            # ``None`` until the request ends on its own; a cancelled request
            # reports ``success`` false with no error code.
            outcome: tuple[bool, int | None] | None = None
            try:
                check_version(version)
                response = ok(await server.handle_v1(method, params))
                outcome = (True, None)
                return response
            except A2AV1ProtocolError as e:
                outcome = (False, e.code)
                return error_response(e.code, e.message)
            except Exception:
                outcome = (False, INTERNAL_ERROR)
                correlation_id = uuid.uuid4().hex
                _logger.exception(
                    "A2A HTTP+JSON %s failed (correlation_id=%s)", method, correlation_id
                )
                return error_response(
                    INTERNAL_ERROR, f"internal error (correlation_id={correlation_id})"
                )
            finally:
                success, error_code = outcome if outcome is not None else (False, None)
                data: dict[str, Any] = {
                    "method": method,
                    "success": success,
                    "duration_ms": (time.perf_counter() - started) * 1000,
                }
                if error_code is not None:
                    data["error_code"] = error_code
                await emit(EV_A2A_TASK_COMPLETED, **data)

    async def sse(
        method: str, params: dict[str, Any], version: str | None, request: Request
    ) -> Any:
        # Lazy import — observability is opt-in.
        from locus.observability.emit import EV_A2A_TASK_RECEIVED, emit  # noqa: PLC0415

        carrier = dict(request.headers)
        with trace_scope(carrier):
            await emit(EV_A2A_TASK_RECEIVED, method=method, rpc_id=None)
            # The handler returns the response before the stream runs, so the
            # stream publishes ``a2a.task.completed`` when its generator ends,
            # in the context ``received`` was emitted under.
            completed = server.completion_emitter(method, time.perf_counter())

        async def gen() -> AsyncIterator[str]:
            # The generator runs after the handler returned, so the caller's
            # trace is re-attached here for the run it drives.
            with trace_scope(carrier):
                stream = produce()
                try:
                    async for chunk in stream:
                        yield chunk
                finally:
                    # Closed here, not by the loop's finaliser, so the
                    # completion lands as soon as the response ends.
                    await stream.aclose()

        async def produce() -> AsyncGenerator[str, None]:
            # ``None`` until the stream ends on its own; a client that goes
            # away mid-stream closes the generator before it is set.
            outcome: tuple[bool, int | None] | None = None
            try:
                check_version(version)
                async for frame in server.handle_v1_stream(method, params):
                    yield f"data: {json.dumps(frame)}\n\n"
                outcome = (True, None)
            except A2AV1ProtocolError as e:
                outcome = (False, e.code)
                # An ``error`` event, as the specification's reference client
                # expects for a failure inside a stream.
                yield f"event: error\ndata: {json.dumps(http_error_body(e.code, e.message))}\n\n"
            except Exception:
                outcome = (False, INTERNAL_ERROR)
                correlation_id = uuid.uuid4().hex
                _logger.exception(
                    "A2A HTTP+JSON %s stream failed (correlation_id=%s)", method, correlation_id
                )
                yield (
                    "event: error\ndata: "
                    + json.dumps(
                        http_error_body(
                            INTERNAL_ERROR, f"internal error (correlation_id={correlation_id})"
                        )
                    )
                    + "\n\n"
                )
            finally:
                success, error_code = outcome if outcome is not None else (False, None)
                completed(success, error_code)

        return StreamingResponse(gen(), media_type="text/event-stream")

    # Routes whose path ends in ``:verb`` are registered before ``/tasks/{id}``
    # so a request for ``/tasks/abc:cancel`` is not read as the task ``abc:cancel``.

    @app.post("/message:send")
    async def rest_send_message(
        request: Request,
        a2a_version: str | None = Header(default=None, alias=A2A_VERSION_HEADER),
        _: str = auth_dep,
    ) -> Any:
        try:
            params = await body_of(request)
        except A2AV1ProtocolError as e:
            return error_response(e.code, e.message)
        return await unary("SendMessage", params, a2a_version, request)

    @app.post("/message:stream")
    async def rest_stream_message(
        request: Request,
        a2a_version: str | None = Header(default=None, alias=A2A_VERSION_HEADER),
        _: str = auth_dep,
    ) -> Any:
        try:
            params = await body_of(request)
        except A2AV1ProtocolError as e:
            return error_response(e.code, e.message)
        return await sse("SendStreamingMessage", params, a2a_version, request)

    @app.post("/tasks/{task_id}:cancel")
    async def rest_cancel_task(
        task_id: str,
        request: Request,
        a2a_version: str | None = Header(default=None, alias=A2A_VERSION_HEADER),
        _: str = auth_dep,
    ) -> Any:
        try:
            params = await body_of(request)
        except A2AV1ProtocolError as e:
            return error_response(e.code, e.message)
        params["id"] = task_id
        return await unary("CancelTask", params, a2a_version, request)

    @app.get("/tasks/{task_id}:subscribe")
    @app.post("/tasks/{task_id}:subscribe")
    async def rest_subscribe_task(
        task_id: str,
        request: Request,
        a2a_version: str | None = Header(default=None, alias=A2A_VERSION_HEADER),
        _: str = auth_dep,
    ) -> Any:
        return await sse("SubscribeToTask", {"id": task_id}, a2a_version, request)

    @app.post("/tasks/{task_id}/pushNotificationConfigs")
    async def rest_create_push_config(
        task_id: str,
        request: Request,
        a2a_version: str | None = Header(default=None, alias=A2A_VERSION_HEADER),
        _: str = auth_dep,
    ) -> Any:
        try:
            params = await body_of(request)
        except A2AV1ProtocolError as e:
            return error_response(e.code, e.message)
        params["taskId"] = task_id
        return await unary("CreateTaskPushNotificationConfig", params, a2a_version, request)

    @app.get("/tasks/{task_id}/pushNotificationConfigs/{config_id}")
    async def rest_get_push_config(
        task_id: str,
        config_id: str,
        request: Request,
        a2a_version: str | None = Header(default=None, alias=A2A_VERSION_HEADER),
        _: str = auth_dep,
    ) -> Any:
        return await unary(
            "GetTaskPushNotificationConfig",
            {"taskId": task_id, "id": config_id},
            a2a_version,
            request,
        )

    @app.get("/tasks/{task_id}/pushNotificationConfigs")
    async def rest_list_push_configs(
        task_id: str,
        request: Request,
        a2a_version: str | None = Header(default=None, alias=A2A_VERSION_HEADER),
        _: str = auth_dep,
    ) -> Any:
        try:
            params = query_to_params(dict(request.query_params))
        except A2AV1ProtocolError as e:
            return error_response(e.code, e.message)
        params["taskId"] = task_id
        return await unary("ListTaskPushNotificationConfigs", params, a2a_version, request)

    @app.delete("/tasks/{task_id}/pushNotificationConfigs/{config_id}")
    async def rest_delete_push_config(
        task_id: str,
        config_id: str,
        request: Request,
        a2a_version: str | None = Header(default=None, alias=A2A_VERSION_HEADER),
        _: str = auth_dep,
    ) -> Any:
        return await unary(
            "DeleteTaskPushNotificationConfig",
            {"taskId": task_id, "id": config_id},
            a2a_version,
            request,
        )

    @app.get("/tasks/{task_id}")
    async def rest_get_task(
        task_id: str,
        request: Request,
        a2a_version: str | None = Header(default=None, alias=A2A_VERSION_HEADER),
        _: str = auth_dep,
    ) -> Any:
        try:
            params = query_to_params(dict(request.query_params))
        except A2AV1ProtocolError as e:
            return error_response(e.code, e.message)
        params["id"] = task_id
        return await unary("GetTask", params, a2a_version, request)

    @app.get("/tasks")
    async def rest_list_tasks(
        request: Request,
        a2a_version: str | None = Header(default=None, alias=A2A_VERSION_HEADER),
        _: str = auth_dep,
    ) -> Any:
        try:
            params = query_to_params(dict(request.query_params))
        except A2AV1ProtocolError as e:
            return error_response(e.code, e.message)
        return await unary("ListTasks", params, a2a_version, request)

    @app.get("/extendedAgentCard")
    async def rest_extended_card(
        request: Request,
        a2a_version: str | None = Header(default=None, alias=A2A_VERSION_HEADER),
        _: str = auth_dep,
    ) -> Any:
        return await unary("GetExtendedAgentCard", {}, a2a_version, request)


__all__ = ["mount_http_json", "query_to_params"]
