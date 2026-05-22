# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""A2A protocol implementation — spec-compliant agent-to-agent transport.

This module exposes a Locus Agent over the public A2A protocol
(https://a2aproject.github.io/A2A/), so peers from other frameworks
(Strands, ADK, Google A2A SDKs) can call the agent without an adapter.

Wire surface served by :class:`A2AServer`:

- ``GET  /.well-known/agent-card.json`` — public Agent Card (spec §5.5).
- ``POST /``                            — JSON-RPC 2.0 method dispatch
  with the method names ``message/send``, ``message/stream``,
  ``tasks/get``, ``tasks/cancel``. ``message/stream`` returns SSE.
- Backwards-compat aliases preserved from the pre-spec implementation:
  ``GET /agent-card``, ``POST /a2a/invoke``, ``POST /a2a/stream``.

Security model
--------------
Every route — including the well-known card — requires a bearer
token when ``api_key`` / ``LOCUS_A2A_API_KEY`` is set. With no key,
the server refuses to bind to anything other than loopback unless
``allow_unauthenticated=True`` is passed explicitly. The agent's tool
inventory is exposed only via skills that the operator declared at
construction time, never via tool reflection (CWE-306).
"""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import httpx

from pydantic import BaseModel, Field

from locus.a2a.spec import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PUSH_NOTIFICATION_NOT_SUPPORTED,
    TASK_NOT_CANCELABLE,
    TASK_NOT_FOUND,
    AgentCapabilities,
    AgentCard,
    AgentProvider,
    AgentSkill,
    Artifact,
    DataPart,
    FilePart,
    JsonRpcError,
    JsonRpcErrorResponse,
    JsonRpcRequest,
    JsonRpcSuccessResponse,
    Message,
    MessageSendParams,
    Part,
    Task,
    TaskArtifactUpdateEvent,
    TaskIdParams,
    TaskQueryParams,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)


_logger = logging.getLogger(__name__)

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _is_loopback(host: str) -> bool:
    if host in _LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _now_iso() -> str:
    """Spec timestamps use RFC 3339 / ISO-8601 with a ``Z`` suffix."""
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Backward-compat models from the pre-spec implementation.
# ---------------------------------------------------------------------------


class A2AMessage(BaseModel):
    """Legacy flat message — preserved so peers + tests that still call
    ``/a2a/invoke`` keep working. Spec-aware peers should use
    :class:`locus.a2a.spec.Message`."""

    role: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class A2ARequest(BaseModel):
    """Legacy request envelope for ``POST /a2a/invoke``."""

    messages: list[A2AMessage]
    metadata: dict[str, Any] = Field(default_factory=dict)


class A2AResponse(BaseModel):
    """Legacy response envelope from ``POST /a2a/invoke``."""

    messages: list[A2AMessage]
    status: str = "completed"
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers — Locus Agent ↔ A2A spec types
# ---------------------------------------------------------------------------


def _extract_user_text(parts: list[Part]) -> str:
    """Concatenate text parts into a single prompt string.

    File parts are reported as ``[file: name]`` and data parts are
    serialised inline — the conservative default for an Agent that
    speaks Python/text. Specialised agents can override the server's
    ``_run_agent`` hook to handle parts directly.
    """
    chunks: list[str] = []
    for part in parts:
        if isinstance(part, TextPart):
            chunks.append(part.text)
        elif isinstance(part, FilePart):
            file = part.file
            label = getattr(file, "name", None) or getattr(file, "uri", "file")
            chunks.append(f"[file: {label}]")
        elif isinstance(part, DataPart):
            chunks.append(json.dumps(part.data, ensure_ascii=False))
    return "\n".join(chunks)


def _agent_text_message(text: str, *, context_id: str, task_id: str | None) -> Message:
    """Build a spec ``Message`` from an agent's plain-text reply."""
    return Message(
        role="agent",
        parts=[TextPart(text=text)],
        messageId=uuid.uuid4().hex,
        contextId=context_id,
        taskId=task_id,
    )


def _agent_artifact(text: str) -> Artifact:
    """Wrap an agent's final reply in an Artifact for ``Task.artifacts``."""
    return Artifact(
        artifactId=uuid.uuid4().hex,
        name="reply",
        parts=[TextPart(text=text)],
    )


# ---------------------------------------------------------------------------
# In-memory task store
# ---------------------------------------------------------------------------


class _TaskStore:
    """In-process task store for non-persistent A2A deployments.

    Production deployments behind a load balancer should swap this out
    for a shared store (Redis / SQL / OCI Object Storage) — the
    ``A2AServer`` accepts a ``store`` parameter that follows this
    duck-typed protocol: ``get``, ``put``, ``cancel``.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._cancel_flags: dict[str, bool] = {}

    def put(self, task: Task) -> None:
        self._tasks[task.id] = task

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def cancel(self, task_id: str) -> bool:
        """Mark a task as cancel-requested.

        Returns ``True`` if the task exists and was in a cancellable
        state, ``False`` otherwise (caller maps this to the proper
        JSON-RPC error code).
        """
        task = self._tasks.get(task_id)
        if task is None:
            return False
        if task.status.state in {
            TaskState.completed,
            TaskState.canceled,
            TaskState.failed,
            TaskState.rejected,
        }:
            return False
        self._cancel_flags[task_id] = True
        # Eagerly flip status so a subsequent tasks/get reflects the
        # request even if the underlying agent hasn't observed the
        # flag yet.
        task.status = TaskStatus(state=TaskState.canceled, timestamp=_now_iso())
        return True

    def is_cancel_requested(self, task_id: str) -> bool:
        return self._cancel_flags.get(task_id, False)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class A2AServer:
    """Expose a Locus Agent as a spec-compliant A2A endpoint.

    Args:
        agent: A Locus ``Agent`` (or anything with ``run(prompt) -> AsyncIterator``).
        name: Display name for the Agent Card.
        description: One-line description for the Agent Card.
        skills: List of :class:`AgentSkill` (preferred) or plain strings
            (legacy — auto-promoted to skills with id == name).
        url: Public URL the agent is reachable at — set this for
            cross-process / cross-host deployments so the card's ``url``
            field is correct. Defaults to a placeholder.
        provider: Optional :class:`AgentProvider` (e.g. ``Oracle``).
        version: Agent semver — useful for capability negotiation.
        api_key: Bearer token required on every route; if ``None``,
            falls back to ``LOCUS_A2A_API_KEY``.
        allow_unauthenticated: Bind to non-loopback without a key.
            Use only behind an upstream proxy that terminates auth.

    Example::

        from locus import Agent
        from locus.a2a import A2AServer
        from locus.a2a.spec import AgentSkill

        server = A2AServer(
            agent=my_agent,
            name="Research Agent",
            description="Open-web research with citations.",
            skills=[
                AgentSkill(
                    id="research",
                    name="Research",
                    description="Answer with cited sources.",
                    tags=["search", "summarise"],
                ),
            ],
            url="https://research.example.com",
            api_key="secret",
        )
        server.run(port=8001)
    """

    def __init__(
        self,
        agent: Any,
        name: str = "Locus Agent",
        description: str = "",
        skills: list[AgentSkill] | list[str] | None = None,
        url: str = "",
        provider: AgentProvider | None = None,
        version: str = "0.1.0",
        api_key: str | None = None,
        allow_unauthenticated: bool = False,
    ) -> None:
        self._agent = agent
        self._name = name
        self._description = description or f"A2A-compatible {name}"
        self._skills = self._normalise_skills(skills)
        self._url = url
        self._provider = provider
        self._version = version
        self._api_key = api_key or os.environ.get("LOCUS_A2A_API_KEY") or None
        self._allow_unauthenticated = allow_unauthenticated
        self._app: Any = None
        self._store = _TaskStore()

    @staticmethod
    def _normalise_skills(
        skills: list[AgentSkill] | list[str] | None,
    ) -> list[AgentSkill]:
        if not skills:
            return []
        out: list[AgentSkill] = []
        for s in skills:
            if isinstance(s, AgentSkill):
                out.append(s)
            else:
                # Legacy: a plain string becomes a minimal skill with
                # id == name == description so the wire shape is valid.
                out.append(AgentSkill(id=s, name=s, description=s))
        return out

    def _build_card(self) -> AgentCard:
        # When the server enforces bearer auth on every route (the default
        # when ``api_key`` / ``LOCUS_A2A_API_KEY`` is configured), advertise
        # that requirement in the AgentCard so peers can discover it from
        # ``/.well-known/agent-card.json`` instead of finding out via a
        # 401 on the first call. See A2A spec §5.5 + §5.6 — closes #214.
        security_schemes: dict[str, Any] | None = None
        security: list[dict[str, list[str]]] | None = None
        if self._api_key is not None:
            security_schemes = {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "description": (
                        "Bearer token required on every route. "
                        "Set via the ``api_key=`` constructor argument or "
                        "the ``LOCUS_A2A_API_KEY`` environment variable."
                    ),
                }
            }
            security = [{"bearerAuth": []}]

        return AgentCard(
            name=self._name,
            description=self._description,
            url=self._url or "http://localhost",
            provider=self._provider,
            version=self._version,
            capabilities=AgentCapabilities(
                streaming=True,
                pushNotifications=False,
                stateTransitionHistory=False,
            ),
            securitySchemes=security_schemes,
            security=security,
            defaultInputModes=["text/plain"],
            defaultOutputModes=["text/plain"],
            skills=self._skills,
        )

    @property
    def app(self) -> Any:
        if self._app is None:
            self._app = self._create_app()
        return self._app

    def _resolve_docs_enabled(self) -> bool:
        try:
            from locus.core.config import get_settings

            return bool(get_settings().debug)
        except Exception:  # noqa: BLE001 — settings failure must not leak docs
            return False

    def _require_auth(self) -> Any:
        from fastapi import Header, HTTPException, status

        expected = self._api_key

        async def dependency(
            authorization: str | None = Header(default=None),
        ) -> str:
            if expected is None:
                return "anon"
            if not authorization or not authorization.lower().startswith("bearer "):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Missing bearer token",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            presented = authorization.split(" ", 1)[1].strip()
            if not hmac.compare_digest(presented, expected):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid bearer token",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return "authed"

        return dependency

    # ----- agent driver -------------------------------------------------

    async def _drive_agent(
        self,
        prompt: str,
        *,
        task: Task,
        on_event: Any | None = None,
    ) -> str:
        """Run the wrapped agent and stream events into the task.

        Calls ``on_event(TaskStatusUpdateEvent | TaskArtifactUpdateEvent)``
        for each transition / final artifact when provided (the
        ``message/stream`` path uses this; ``message/send`` ignores).
        Returns the agent's final text reply.
        """
        from locus.core.events import TerminateEvent, ThinkEvent

        final = ""
        # Mark task as working.
        task.status = TaskStatus(state=TaskState.working, timestamp=_now_iso())
        if on_event is not None:
            on_event(
                TaskStatusUpdateEvent(
                    taskId=task.id,
                    contextId=task.contextId,
                    status=task.status,
                )
            )

        async for event in self._agent.run(prompt):
            if self._store.is_cancel_requested(task.id):
                task.status = TaskStatus(state=TaskState.canceled, timestamp=_now_iso())
                if on_event is not None:
                    on_event(
                        TaskStatusUpdateEvent(
                            taskId=task.id,
                            contextId=task.contextId,
                            status=task.status,
                            final=True,
                        )
                    )
                return final
            if isinstance(event, ThinkEvent) and event.reasoning and on_event is not None:
                # Surface intermediate reasoning as a working-state
                # status update; spec peers can ignore or render it.
                on_event(
                    TaskStatusUpdateEvent(
                        taskId=task.id,
                        contextId=task.contextId,
                        status=TaskStatus(
                            state=TaskState.working,
                            message=_agent_text_message(
                                event.reasoning,
                                context_id=task.contextId,
                                task_id=task.id,
                            ),
                            timestamp=_now_iso(),
                        ),
                    )
                )
            elif isinstance(event, TerminateEvent):
                final = event.final_message or final

        artifact = _agent_artifact(final)
        task.artifacts.append(artifact)
        task.status = TaskStatus(state=TaskState.completed, timestamp=_now_iso())
        if on_event is not None:
            on_event(
                TaskArtifactUpdateEvent(
                    taskId=task.id,
                    contextId=task.contextId,
                    artifact=artifact,
                    lastChunk=True,
                )
            )
            on_event(
                TaskStatusUpdateEvent(
                    taskId=task.id,
                    contextId=task.contextId,
                    status=task.status,
                    final=True,
                )
            )
        return final

    # ----- JSON-RPC method handlers ------------------------------------

    async def _handle_message_send(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            send = MessageSendParams.model_validate(params)
        except Exception as e:  # noqa: BLE001
            raise _RpcError(INVALID_PARAMS, f"invalid params: {e}") from e
        prompt = _extract_user_text(send.message.parts)
        context_id = send.message.contextId or uuid.uuid4().hex
        task = Task(
            id=uuid.uuid4().hex,
            contextId=context_id,
            status=TaskStatus(state=TaskState.submitted, timestamp=_now_iso()),
            history=[send.message],
        )
        self._store.put(task)
        try:
            await self._drive_agent(prompt, task=task)
        except Exception as e:  # noqa: BLE001
            task.status = TaskStatus(
                state=TaskState.failed,
                timestamp=_now_iso(),
                message=_agent_text_message(
                    "agent error", context_id=task.contextId, task_id=task.id
                ),
            )
            _logger.exception("A2A message/send failed for task %s", task.id)
            raise _RpcError(INTERNAL_ERROR, "internal error") from e
        return task.model_dump()

    async def _handle_tasks_get(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            q = TaskQueryParams.model_validate(params)
        except Exception as e:  # noqa: BLE001
            raise _RpcError(INVALID_PARAMS, f"invalid params: {e}") from e
        task = self._store.get(q.id)
        if task is None:
            raise _RpcError(TASK_NOT_FOUND, f"task {q.id} not found")
        if q.historyLength is not None and q.historyLength >= 0:
            trimmed = task.model_copy(deep=True)
            trimmed.history = trimmed.history[-q.historyLength :]
            return trimmed.model_dump()
        return task.model_dump()

    async def _handle_tasks_cancel(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            ident = TaskIdParams.model_validate(params)
        except Exception as e:  # noqa: BLE001
            raise _RpcError(INVALID_PARAMS, f"invalid params: {e}") from e
        if self._store.get(ident.id) is None:
            raise _RpcError(TASK_NOT_FOUND, f"task {ident.id} not found")
        if not self._store.cancel(ident.id):
            raise _RpcError(
                TASK_NOT_CANCELABLE,
                "task is in a terminal state and cannot be cancelled",
            )
        task = self._store.get(ident.id)
        assert task is not None
        return task.model_dump()

    async def _stream_message(self, params: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Stream a message run as JSON-RPC responses (one per SSE event)."""
        try:
            send = MessageSendParams.model_validate(params)
        except Exception as e:  # noqa: BLE001
            raise _RpcError(INVALID_PARAMS, f"invalid params: {e}") from e
        prompt = _extract_user_text(send.message.parts)
        context_id = send.message.contextId or uuid.uuid4().hex
        task = Task(
            id=uuid.uuid4().hex,
            contextId=context_id,
            status=TaskStatus(state=TaskState.submitted, timestamp=_now_iso()),
            history=[send.message],
        )
        self._store.put(task)

        # Initial event: the task as it sits in the submitted state.
        yield task.model_dump()

        # Bridge the synchronous on_event callback to an async queue so
        # SSE consumers see updates as they happen.
        queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()

        def emit(ev: Any) -> None:
            queue.put_nowait(ev)

        async def runner() -> None:
            try:
                await self._drive_agent(prompt, task=task, on_event=emit)
            except Exception:  # noqa: BLE001
                task.status = TaskStatus(state=TaskState.failed, timestamp=_now_iso())
                emit(
                    TaskStatusUpdateEvent(
                        taskId=task.id,
                        contextId=task.contextId,
                        status=task.status,
                        final=True,
                    )
                )
            finally:
                queue.put_nowait(sentinel)

        run_task = asyncio.create_task(runner())
        try:
            while True:
                ev = await queue.get()
                if ev is sentinel:
                    break
                yield ev.model_dump() if hasattr(ev, "model_dump") else dict(ev)
        finally:
            await run_task

    # ----- backward-compat: the legacy A2ARequest invoke shape --------

    async def _legacy_invoke(self, request: A2ARequest) -> A2AResponse:
        from locus.core.events import TerminateEvent

        user_msgs = [m for m in request.messages if m.role == "user"]
        prompt = user_msgs[-1].content if user_msgs else ""
        final = ""
        stop_reason = "complete"
        iterations = 0
        async for event in self._agent.run(prompt):
            if isinstance(event, TerminateEvent):
                final = event.final_message or final
                stop_reason = event.reason or stop_reason
            iterations += 1
        return A2AResponse(
            messages=[A2AMessage(role="agent", content=final)],
            status="completed",
            metadata={"stop_reason": stop_reason, "iterations": iterations},
        )

    # ----- app construction --------------------------------------------

    def _create_app(self) -> Any:
        try:
            from fastapi import Body, Depends, FastAPI
            from fastapi.responses import JSONResponse, StreamingResponse
        except ImportError as e:
            msg = "FastAPI required. Install with: pip install fastapi uvicorn"
            raise ImportError(msg) from e

        if self._api_key is None and not self._allow_unauthenticated:
            _logger.warning(
                "A2AServer: no api_key configured; will require "
                "loopback-only binding. Set LOCUS_A2A_API_KEY or pass "
                "allow_unauthenticated=True to override."
            )

        debug_docs = self._resolve_docs_enabled()
        app = FastAPI(
            title=f"A2A: {self._name}",
            docs_url="/docs" if debug_docs else None,
            redoc_url="/redoc" if debug_docs else None,
            openapi_url="/openapi.json" if debug_docs else None,
        )

        if self._api_key is not None:
            auth_dep = Depends(self._require_auth())
        else:

            async def _anon() -> str:
                return "anon"

            auth_dep = Depends(_anon)

        # ----- Agent Card -------------------------------------------------
        @app.get("/.well-known/agent-card.json")
        async def well_known_card(_: str = auth_dep) -> dict[str, Any]:
            return self._build_card().model_dump()

        @app.get("/agent-card")
        async def legacy_card(_: str = auth_dep) -> dict[str, Any]:
            # Backwards-compat alias. Matches the old flat shape so
            # peers that still expect ``skills: list[str]`` keep
            # parsing — we re-emit just id/name/description per skill.
            card = self._build_card()
            payload = card.model_dump()
            payload["skills"] = [s.name for s in card.skills]
            return payload

        # ----- JSON-RPC root -----------------------------------------------
        @app.post("/")
        async def jsonrpc(
            body: dict[str, Any] = Body(...),  # noqa: B008 — FastAPI dep injection
            _: str = auth_dep,
        ) -> Any:
            try:
                req = JsonRpcRequest.model_validate(body)
            except Exception:  # noqa: BLE001
                return JSONResponse(
                    JsonRpcErrorResponse(
                        id=body.get("id") if isinstance(body, dict) else None,
                        error=JsonRpcError(code=INVALID_REQUEST, message="Invalid Request"),
                    ).model_dump()
                )

            params = req.params if isinstance(req.params, dict) else {}

            # Lazy import — observability is opt-in.
            import time as _time  # noqa: PLC0415

            from locus.observability.emit import (  # noqa: PLC0415
                EV_A2A_TASK_COMPLETED,
                EV_A2A_TASK_RECEIVED,
                emit,
            )

            await emit(
                EV_A2A_TASK_RECEIVED,
                method=req.method,
                rpc_id=str(req.id) if req.id is not None else None,
            )
            _started = _time.perf_counter()

            try:
                if req.method == "message/send":
                    result = await self._handle_message_send(params)
                elif req.method == "tasks/get":
                    result = await self._handle_tasks_get(params)
                elif req.method == "tasks/cancel":
                    result = await self._handle_tasks_cancel(params)
                elif req.method in {
                    "tasks/pushNotificationConfig/set",
                    "tasks/pushNotificationConfig/get",
                    "tasks/pushNotificationConfig/list",
                    "tasks/pushNotificationConfig/delete",
                }:
                    raise _RpcError(
                        PUSH_NOTIFICATION_NOT_SUPPORTED,
                        "push notifications are not enabled on this agent",
                    )
                elif req.method == "message/stream":
                    # SSE — peer must POST with Accept: text/event-stream.
                    return await self._stream_response(req)
                else:
                    raise _RpcError(METHOD_NOT_FOUND, f"unknown method {req.method!r}")
            except _RpcError as e:
                await emit(
                    EV_A2A_TASK_COMPLETED,
                    method=req.method,
                    success=False,
                    error_code=e.code,
                    duration_ms=(_time.perf_counter() - _started) * 1000,
                )
                return JSONResponse(
                    JsonRpcErrorResponse(
                        id=req.id, error=JsonRpcError(code=e.code, message=e.message)
                    ).model_dump()
                )

            await emit(
                EV_A2A_TASK_COMPLETED,
                method=req.method,
                success=True,
                duration_ms=(_time.perf_counter() - _started) * 1000,
            )
            return JSONResponse(JsonRpcSuccessResponse(id=req.id, result=result).model_dump())

        # ----- backwards-compat invoke + stream ----------------------------
        @app.post("/a2a/invoke")
        async def legacy_invoke(req: A2ARequest, _: str = auth_dep) -> dict[str, Any]:
            resp = await self._legacy_invoke(req)
            return resp.model_dump()

        @app.post("/a2a/stream")
        async def legacy_stream(req: A2ARequest, _: str = auth_dep) -> StreamingResponse:
            response: StreamingResponse = await self._legacy_stream(req)
            return response

        return app

    async def _stream_response(self, req: JsonRpcRequest) -> Any:
        from fastapi.responses import StreamingResponse

        params = req.params if isinstance(req.params, dict) else {}

        async def gen() -> AsyncIterator[str]:
            try:
                async for result in self._stream_message(params):
                    payload = JsonRpcSuccessResponse(id=req.id, result=result).model_dump()
                    yield f"data: {json.dumps(payload)}\n\n"
            except _RpcError as e:
                payload = JsonRpcErrorResponse(
                    id=req.id, error=JsonRpcError(code=e.code, message=e.message)
                ).model_dump()
                yield f"data: {json.dumps(payload)}\n\n"
            except Exception:  # noqa: BLE001 — sanitise stream errors
                correlation_id = uuid.uuid4().hex
                _logger.exception("A2A stream error (correlation_id=%s)", correlation_id)
                payload = JsonRpcErrorResponse(
                    id=req.id,
                    error=JsonRpcError(
                        code=INTERNAL_ERROR,
                        message="internal error",
                        data={"correlation_id": correlation_id},
                    ),
                ).model_dump()
                yield f"data: {json.dumps(payload)}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    async def _legacy_stream(self, request: A2ARequest) -> Any:
        from fastapi.responses import StreamingResponse

        from locus.core.events import TerminateEvent, ThinkEvent

        user_msgs = [m for m in request.messages if m.role == "user"]
        prompt = user_msgs[-1].content if user_msgs else ""

        async def event_generator() -> AsyncIterator[str]:
            try:
                async for event in self._agent.run(prompt):
                    if isinstance(event, ThinkEvent):
                        data = {"type": "text", "content": event.reasoning or ""}
                    elif isinstance(event, TerminateEvent):
                        data = {"type": "done", "content": event.final_message or ""}
                    else:
                        data = {"type": event.event_type}
                    yield f"data: {json.dumps(data)}\n\n"
            except Exception:  # noqa: BLE001
                correlation_id = uuid.uuid4().hex
                _logger.exception("A2A legacy stream error (correlation_id=%s)", correlation_id)
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "error",
                            "error": "internal error",
                            "correlation_id": correlation_id,
                        }
                    )
                    + "\n\n"
                )
            finally:
                yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    def run(self, host: str = "127.0.0.1", port: int = 8001, **kwargs: Any) -> None:
        """Run the A2A server.

        Defaults to loopback binding. Non-loopback bindings require
        either ``api_key`` to be set or ``allow_unauthenticated=True``.
        """
        if self._api_key is None and not self._allow_unauthenticated and not _is_loopback(host):
            msg = (
                f"Refusing to bind A2AServer to {host!r} without an API "
                "key. Set LOCUS_A2A_API_KEY, pass api_key=... to "
                "A2AServer, or pass allow_unauthenticated=True if an "
                "upstream proxy terminates auth."
            )
            raise RuntimeError(msg)

        try:
            import uvicorn
        except ImportError as e:
            msg = "uvicorn required. Install with: pip install uvicorn"
            raise ImportError(msg) from e
        uvicorn.run(self.app, host=host, port=port, **kwargs)


class _RpcError(Exception):
    """Internal sentinel — a JSON-RPC method handler raises this to
    short-circuit to a structured error response with the right code."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class A2AClient:
    """Call a remote A2A agent from Locus.

    Spec-compliant methods:

    - :meth:`get_agent_card` — fetches ``/.well-known/agent-card.json``,
      falling back to the legacy ``/agent-card`` endpoint.
    - :meth:`send_message` — JSON-RPC ``message/send``; returns a
      :class:`Task` you can poll with :meth:`get_task`.
    - :meth:`send_message_streaming` — JSON-RPC ``message/stream``;
      yields events from the SSE stream.
    - :meth:`get_task`, :meth:`cancel_task` — task lifecycle.

    Plus the legacy convenience APIs preserved from the pre-spec
    implementation:

    - :meth:`invoke` — flat string-in / string-out over ``/a2a/invoke``.
    - :meth:`as_tool` — wrap a remote agent as a Locus ``@tool``.
    """

    #: Default request timeout (seconds) when caller doesn't override.
    DEFAULT_TIMEOUT: float = 120.0

    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> None:
        self._url = url.rstrip("/")
        self._api_key = api_key
        self._timeout: Any = self.DEFAULT_TIMEOUT if timeout is None else timeout

    def _auth_headers(self) -> dict[str, str]:
        if self._api_key:
            return {"Authorization": f"Bearer {self._api_key}"}
        return {}

    async def get_agent_card(self) -> AgentCard:
        """Fetch the remote agent's capability card.

        Tries the spec well-known URL first, falls back to the legacy
        ``/agent-card`` endpoint for older peers.
        """
        import httpx

        async with httpx.AsyncClient() as client:
            for path in ("/.well-known/agent-card.json", "/agent-card"):
                try:
                    resp = await client.get(f"{self._url}{path}", headers=self._auth_headers())
                    if resp.status_code == 200:
                        data = resp.json()
                        # Legacy peers serve flat string skills — promote.
                        if data.get("skills") and isinstance(data["skills"][0], str):
                            data["skills"] = [
                                {"id": s, "name": s, "description": s} for s in data["skills"]
                            ]
                        # Legacy peers also omit url/capabilities.
                        data.setdefault("url", self._url)
                        return AgentCard.model_validate(data)
                except httpx.HTTPError:
                    continue
        msg = f"Could not fetch Agent Card from {self._url}"
        raise RuntimeError(msg)

    # ---- JSON-RPC client helpers --------------------------------------

    async def _rpc(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: float | httpx.Timeout | None = None,  # noqa: ASYNC109 — forwarded to httpx, not an asyncio timer
    ) -> Any:
        import time as _time

        import httpx

        from locus.observability.emit import (  # noqa: PLC0415
            EV_A2A_CLIENT_RECEIVED,
            EV_A2A_CLIENT_SEND,
            emit,
        )

        body = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex,
            "method": method,
            "params": params,
        }
        await emit(EV_A2A_CLIENT_SEND, target_url=self._url, method=method)
        _started = _time.perf_counter()
        async with httpx.AsyncClient(
            timeout=self._timeout if timeout is None else timeout
        ) as client:
            resp = await client.post(f"{self._url}/", json=body, headers=self._auth_headers())
            resp.raise_for_status()
            data = resp.json()
        await emit(
            EV_A2A_CLIENT_RECEIVED,
            target_url=self._url,
            method=method,
            status_code=resp.status_code,
            duration_ms=(_time.perf_counter() - _started) * 1000,
            content_length=len(resp.content),
        )
        if "error" in data:
            err = data["error"]
            msg = f"A2A error {err.get('code')}: {err.get('message')}"
            raise RuntimeError(msg)
        return data["result"]

    async def send_message(
        self,
        message: Message,
        *,
        timeout: float | httpx.Timeout | None = None,  # noqa: ASYNC109 — forwarded to httpx, not an asyncio timer
    ) -> Task:
        """Send a message via JSON-RPC ``message/send`` and return the Task."""
        result = await self._rpc(
            "message/send",
            {"message": message.model_dump(exclude_none=True)},
            timeout=timeout,
        )
        return Task.model_validate(result)

    async def send_message_streaming(
        self,
        message: Message,
        *,
        timeout: float | httpx.Timeout | None = None,  # noqa: ASYNC109 — forwarded to httpx, not an asyncio timer
    ) -> AsyncIterator[dict[str, Any]]:
        """Send a message via JSON-RPC ``message/stream`` and yield events."""
        import httpx

        body = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex,
            "method": "message/stream",
            "params": {"message": message.model_dump(exclude_none=True)},
        }
        headers = self._auth_headers() | {"Accept": "text/event-stream"}
        effective_timeout = self._timeout if timeout is None else timeout
        async with (
            httpx.AsyncClient(timeout=effective_timeout) as client,
            client.stream("POST", f"{self._url}/", json=body, headers=headers) as resp,
        ):
            resp.raise_for_status()
            async for raw in resp.aiter_lines():
                if not raw or not raw.startswith("data: "):
                    continue
                payload = raw[len("data: ") :]
                if payload.strip() == "[DONE]":
                    break
                try:
                    env = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if "error" in env:
                    yield env  # surface error envelope to caller
                    break
                yield env.get("result", env)

    async def get_task(self, task_id: str, history_length: int | None = None) -> Task:
        """Fetch a task by id (JSON-RPC ``tasks/get``)."""
        params: dict[str, Any] = {"id": task_id}
        if history_length is not None:
            params["historyLength"] = history_length
        result = await self._rpc("tasks/get", params)
        return Task.model_validate(result)

    async def cancel_task(self, task_id: str) -> Task:
        """Cancel a task (JSON-RPC ``tasks/cancel``)."""
        result = await self._rpc("tasks/cancel", {"id": task_id})
        return Task.model_validate(result)

    # ---- legacy convenience APIs --------------------------------------

    async def invoke(
        self,
        prompt: str,
        *,
        timeout: float | httpx.Timeout | None = None,  # noqa: ASYNC109 — forwarded to httpx, not an asyncio timer
    ) -> str:
        """Send a flat text prompt over the legacy ``/a2a/invoke``.

        Useful when you control both ends of the wire and want a one-line
        round-trip; spec-compliant peers should prefer
        :meth:`send_message` so they can read the full :class:`Task`.
        """
        import httpx

        request = A2ARequest(messages=[A2AMessage(role="user", content=prompt)])
        async with httpx.AsyncClient(
            timeout=self._timeout if timeout is None else timeout
        ) as client:
            resp = await client.post(
                f"{self._url}/a2a/invoke",
                json=request.model_dump(),
                headers=self._auth_headers(),
            )
            resp.raise_for_status()
            response = A2AResponse.model_validate(resp.json())
        agent_msgs = [m for m in response.messages if m.role == "agent"]
        return agent_msgs[-1].content if agent_msgs else ""

    def as_tool(self, name: str | None = None, description: str | None = None) -> Any:
        """Wrap this remote agent as a Locus ``@tool``."""
        from locus.tools.decorator import tool as tool_decorator

        client = self
        tool_name = name or "remote_agent"
        tool_desc = description or "Call a remote A2A agent"

        @tool_decorator(name=tool_name, description=tool_desc)
        def call_remote(prompt: str) -> str:
            """Send a request to a remote agent."""
            import asyncio

            return asyncio.run(client.invoke(prompt))

        return call_remote


# Re-export the tasked-out spec types so consumers can keep importing
# from ``locus.a2a.protocol`` alone if they prefer (the canonical path
# is ``locus.a2a`` after this change).
__all__ = [
    "A2AClient",
    "A2AMessage",
    "A2ARequest",
    "A2AResponse",
    "A2AServer",
    "AgentCard",
]


# Force time module reference so ruff/mypy don't drop it on cleanup —
# we reference it from the legacy stream path's correlation handling.
_ = time
