# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""A2A protocol implementation — spec-compliant agent-to-agent transport.

This module exposes a Locus Agent over the public A2A protocol
(https://a2aproject.github.io/A2A/), so peers from other frameworks
(Strands, ADK, Google A2A SDKs) can call the agent without an adapter.

Wire surface served by :class:`A2AServer`:

- ``GET  /.well-known/agent-card.json`` — public Agent Card (spec §5.5),
  whose ``supportedInterfaces`` lists every binding below that is on.
- ``POST /``                            — JSON-RPC 2.0 method dispatch
  with A2A v1.0 methods when ``A2A-Version: 1.0`` is requested.
- The HTTP+JSON binding (``POST /message:send``, ``GET /tasks/{id}``, ...;
  see :mod:`locus.a2a.rest`) on the same app.
- The gRPC binding (``lf.a2a.v1.A2AService``; see
  :mod:`locus.a2a.grpc_binding`) on ``grpc_port`` when one is given.
- Push notifications to the webhooks a task registers (see
  :mod:`locus.a2a.push`) when ``push_notifications`` is configured.
- Backwards-compat aliases preserved from the pre-spec implementation:
  ``message/send``, ``message/stream``, ``tasks/get``, ``tasks/cancel``,
  ``GET /agent-card``, ``POST /a2a/invoke``, ``POST /a2a/stream``.

A run that pauses for a person — ``ask_user``, or an approval hold from
:func:`locus.control.gate_tool` — surfaces as the task state
``input-required`` with the question as the status message; the peer's
next message on the same task resumes the run through ``Agent.resume``.

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
import inspect
import ipaddress
import json
import logging
import os
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from locus.a2a.card import interfaces_for
from locus.a2a.protocol_v1 import (
    A2A_V1_PROTOCOL_VERSION,
    A2A_VERSION_HEADER,
    A2AV1_JSONRPC_METHODS,
    A2AV1_STREAMING_METHODS,
    A2AV1ProtocolError,
    A2AV1ServerMixin,
    error_details,
    legacy_event_to_v1_stream_response,
    legacy_message_to_v1,
    task_result_to_legacy_payload,
    v1_stream_response_to_legacy_payload,
)
from locus.a2a.push import (
    JWKS_PATH,
    PushNotificationDispatcher,
    PushNotificationError,
    PushNotificationSettings,
    PushNotificationStore,
)
from locus.a2a.spec import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PUSH_NOTIFICATION_NOT_SUPPORTED,
    TASK_NOT_CANCELABLE,
    TASK_NOT_FOUND,
    UNSUPPORTED_OPERATION,
    VERSION_NOT_SUPPORTED,
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
from locus.a2a.spec_v1 import (
    BINDING_GRPC,
    BINDING_HTTP_JSON,
    BINDING_JSONRPC,
    A2AV1TaskPushNotificationConfig,
)
from locus.a2a.transports import (
    DEFAULT_BINDING_PREFERENCE,
    A2AClientError,
    A2ATransportUnavailableError,
    ClientTransport,
    make_transport,
    normalise_binding,
    select_interfaces,
)
from locus.observability.propagation import inject_trace_headers, trace_scope


# FastAPI resolves a handler's (string) annotations from this module's globals,
# so the request type the trace-context handlers take must live here; the
# server itself stays optional, hence the guard.
try:
    from starlette.requests import Request
except ImportError:  # pragma: no cover - exercised where the server extra is absent
    Request = Any  # type: ignore[assignment,misc]


if TYPE_CHECKING:
    import httpx


_logger = logging.getLogger(__name__)

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

_TERMINAL_STATES = frozenset(
    {TaskState.completed, TaskState.canceled, TaskState.failed, TaskState.rejected}
)
_INTERRUPTED_STATES = frozenset({TaskState.input_required, TaskState.auth_required})

#: Words a peer's reply may open with to decide an approval hold.
_APPROVE_WORDS = frozenset(
    {"approve", "approved", "yes", "y", "allow", "allowed", "ok", "accept", "accepted", "proceed"}
)
_DENY_WORDS = frozenset(
    {"deny", "denied", "reject", "rejected", "no", "n", "refuse", "refused", "decline", "declined"}
)


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


def _verdict_of(text: str) -> str | None:
    """``"approved"`` / ``"denied"`` from the first word of a reply, else ``None``."""
    words = text.strip().split()
    if not words:
        return None
    first = words[0].lower().strip(".,;:!")
    if first in _APPROVE_WORDS:
        return "approved"
    if first in _DENY_WORDS:
        return "denied"
    return None


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


def _agent_text_message(
    text: str,
    *,
    context_id: str,
    task_id: str | None,
    metadata: dict[str, Any] | None = None,
) -> Message:
    """Build a spec ``Message`` from an agent's plain-text reply."""
    return Message(
        role="agent",
        parts=[TextPart(text=text)],
        messageId=uuid.uuid4().hex,
        contextId=context_id,
        taskId=task_id,
        metadata=metadata,
    )


def _agent_artifact(text: str) -> Artifact:
    """Wrap an agent's final reply in an Artifact for ``Task.artifacts``."""
    return Artifact(
        artifactId=uuid.uuid4().hex,
        name="reply",
        parts=[TextPart(text=text)],
    )


def _jsonrpc_error_response(rpc_id: str | int | None, code: int, message: str) -> dict[str, Any]:
    error = JsonRpcError(
        code=code,
        message=message,
        data=error_details(code, message),
    )
    return JsonRpcErrorResponse(id=rpc_id, error=error).model_dump(exclude_none=True)


class _RpcError(A2AV1ProtocolError):
    """Internal sentinel — a JSON-RPC method handler raises this to
    short-circuit to a structured error response with the right code."""


# ---------------------------------------------------------------------------
# In-memory task store
# ---------------------------------------------------------------------------


class _TaskStore:
    """In-process task store for non-persistent A2A deployments.

    Production deployments behind a load balancer should swap this out
    for a shared store (Redis / SQL / S3) that follows the same
    duck-typed protocol: ``get``, ``put``, ``list``, ``cancel``,
    ``is_cancel_requested``, plus ``subscribe`` / ``unsubscribe`` /
    ``publish`` for the live event feed ``SubscribeToTask`` reads.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._cancel_flags: dict[str, bool] = {}
        self._subscribers: dict[str, list[asyncio.Queue[dict[str, Any]]]] = {}

    def put(self, task: Task) -> None:
        self._tasks[task.id] = task

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def list(self) -> list[Task]:
        return list(self._tasks.values())

    def cancel(self, task_id: str) -> bool:
        """Mark a task as cancel-requested.

        Returns ``True`` if the task exists and was in a cancellable
        state, ``False`` otherwise (caller maps this to the proper
        JSON-RPC error code).
        """
        task = self._tasks.get(task_id)
        if task is None:
            return False
        if task.status.state in _TERMINAL_STATES:
            return False
        self._cancel_flags[task_id] = True
        # Eagerly flip status so a subsequent tasks/get reflects the
        # request even if the underlying agent hasn't observed the
        # flag yet.
        task.status = TaskStatus(state=TaskState.canceled, timestamp=_now_iso())
        return True

    def is_cancel_requested(self, task_id: str) -> bool:
        return self._cancel_flags.get(task_id, False)

    def subscribe(self, task_id: str) -> asyncio.Queue[dict[str, Any]]:
        """A queue that receives every event published for ``task_id``."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers.setdefault(task_id, []).append(queue)
        return queue

    def unsubscribe(self, task_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        queues = self._subscribers.get(task_id)
        if queues and queue in queues:
            queues.remove(queue)
        if not queues:
            self._subscribers.pop(task_id, None)

    def publish(self, task_id: str, event: dict[str, Any]) -> None:
        for queue in list(self._subscribers.get(task_id, [])):
            queue.put_nowait(event)


@dataclass
class _PendingInput:
    """A task paused on a question or an approval hold, waiting for the peer."""

    thread_id: str
    interrupt_id: str
    question: str
    approval_id: str | None = None
    status: TaskStatus | None = None


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class A2AServer(A2AV1ServerMixin):
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
        provider: Optional :class:`AgentProvider` (e.g. ``Acme``).
        version: Agent semver — useful for capability negotiation.
        api_key: Bearer token required on every route; if ``None``,
            falls back to ``LOCUS_A2A_API_KEY``.
        allow_unauthenticated: Bind to non-loopback without a key.
            Use only behind an upstream proxy that terminates auth.
        transports: The protocol bindings to serve and advertise:
            ``"JSONRPC"``, ``"HTTP+JSON"`` (both on the FastAPI app) and
            ``"GRPC"`` (on ``grpc_port``; needs the ``a2a-grpc`` extra).
            Defaults to the two HTTP bindings.
        grpc_port: Port for the gRPC binding; setting it adds ``"GRPC"``
            to ``transports``. ``run()`` then serves HTTP and gRPC together.
        grpc_url: The address advertised for the gRPC interface, when
            it differs from the HTTP host plus ``grpc_port``.
        push_notifications: :class:`~locus.a2a.push.PushNotificationSettings`
            (or ``True`` for the defaults) to deliver task updates to the
            webhooks peers register; ``None`` answers push-config methods
            with ``PushNotificationNotSupported``.
        approvals: The agent's approval store (``InMemoryApprovals``,
            ``FileApprovals``, ...). With it, a peer's ``approve`` / ``deny``
            reply to an ``input-required`` task that holds a gated call is
            recorded as the decision before the run resumes.

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
        *,
        transports: Sequence[str] = (BINDING_JSONRPC, BINDING_HTTP_JSON),
        grpc_port: int | None = None,
        grpc_url: str | None = None,
        push_notifications: PushNotificationSettings | bool | None = None,
        approvals: Any | None = None,
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

        bindings = [normalise_binding(t) for t in transports]
        if (grpc_port is not None or grpc_url) and BINDING_GRPC not in bindings:
            bindings.append(BINDING_GRPC)
        if BINDING_GRPC in bindings and grpc_port is None and not grpc_url:
            raise ValueError("the GRPC transport needs grpc_port (or grpc_url)")
        self._transports: tuple[str, ...] = tuple(dict.fromkeys(bindings))
        self._grpc_port = grpc_port
        self._grpc_url = grpc_url

        settings: PushNotificationSettings | None
        if push_notifications is True:
            settings = PushNotificationSettings()
        elif isinstance(push_notifications, PushNotificationSettings):
            settings = push_notifications
        else:
            settings = None
        self._push_store = PushNotificationStore()
        self._push: PushNotificationDispatcher | None = (
            PushNotificationDispatcher(self._push_store, settings) if settings else None
        )
        self._approvals = approvals
        self._pending: dict[str, _PendingInput] = {}
        self._background: set[asyncio.Task[Any]] = set()
        #: Tasks with a run in flight in this process: the ones whose live
        #: event feed ``SubscribeToTask`` can wait on.
        self._active: set[str] = set()
        self._run_accepts_thread_id = self._accepts_keyword(
            getattr(agent, "run", None), "thread_id"
        )
        self._can_resume = callable(getattr(agent, "resume", None))

    @staticmethod
    def _accepts_keyword(fn: Any, keyword: str) -> bool:
        if fn is None:
            return False
        try:
            return keyword in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return False

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

    # ----- public facts ------------------------------------------------

    @property
    def api_key(self) -> str | None:
        """The bearer token every binding requires, or ``None``."""
        return self._api_key

    @property
    def transports(self) -> tuple[str, ...]:
        """The protocol bindings this server serves, as advertised on the card."""
        return self._transports

    @property
    def push_notifications(self) -> PushNotificationDispatcher | None:
        """The push dispatcher, or ``None`` when push notifications are off."""
        return self._push

    def _push_registry(self) -> PushNotificationDispatcher | None:
        return self._push

    def _build_card(self) -> AgentCard:
        # When the server enforces bearer auth on every route (the default
        # when ``api_key`` / ``LOCUS_A2A_API_KEY`` is configured), advertise
        # that requirement in the AgentCard so peers can discover it from
        # ``/.well-known/agent-card.json`` instead of finding out via a
        # 401 on the first call. See A2A spec §5.5 + §5.6 — closes #214.
        security_schemes: dict[str, Any] = {}
        security: list[dict[str, list[str]]] | None = None
        security_requirements: list[dict[str, list[str]]] = []
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
            security_requirements = security

        skills = [
            skill.model_copy(
                update={
                    "inputModes": skill.inputModes or ["text/plain"],
                    "outputModes": skill.outputModes or ["text/plain"],
                }
            )
            for skill in self._skills
        ]

        return AgentCard(
            name=self._name,
            description=self._description,
            url=self._url or "http://localhost",
            provider=self._provider,
            version=self._version,
            protocolVersion=A2A_V1_PROTOCOL_VERSION,
            capabilities=AgentCapabilities(
                streaming=True,
                pushNotifications=self._push is not None,
                stateTransitionHistory=False,
            ),
            securitySchemes=security_schemes,
            security=security,
            securityRequirements=security_requirements,
            supportedInterfaces=interfaces_for(
                self._url or "http://localhost",
                self._transports,
                grpc_port=self._grpc_port,
                grpc_url=self._grpc_url,
            ),
            defaultInputModes=["text/plain"],
            defaultOutputModes=["text/plain"],
            skills=skills,
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

    # ----- event fan-out ---------------------------------------------------

    def _emit(self, task: Task, event: BaseModel, on_event: Callable[[Any], None] | None) -> None:
        """Hand a task event to the streaming consumer, live subscribers and webhooks."""
        if on_event is not None:
            on_event(event)
        payload = event.model_dump()
        self._store.publish(task.id, payload)
        if self._push is not None:
            self._push.notify(task.id, legacy_event_to_v1_stream_response(payload))

    async def _task_events(self, task_id: str) -> AsyncIterator[dict[str, Any]]:
        """Live events for ``task_id`` until it reaches a terminal or interrupted state.

        A task with no run in flight in this process (finished, paused, or
        stored by someone else) yields nothing: there is no one to wait on.
        """
        queue = self._store.subscribe(task_id)
        try:
            task = self._store.get(task_id)
            if (
                task is None
                or task_id not in self._active
                or task.status.state in _TERMINAL_STATES | _INTERRUPTED_STATES
            ):
                return
            while True:
                event = await queue.get()
                yield event
                if event.get("kind") == "status-update":
                    state = event.get("status", {}).get("state")
                    if state in {s.value for s in _TERMINAL_STATES | _INTERRUPTED_STATES}:
                        return
        finally:
            self._store.unsubscribe(task_id, queue)

    # ----- agent driver -------------------------------------------------

    def _start_run(self, prompt: str, task: Task) -> AsyncIterator[Any]:
        kwargs: dict[str, Any] = {"thread_id": task.id} if self._run_accepts_thread_id else {}
        events: AsyncIterator[Any] = self._agent.run(prompt, **kwargs)
        return events

    async def _drive_agent(
        self,
        prompt: str,
        *,
        task: Task,
        on_event: Callable[[Any], None] | None = None,
    ) -> str:
        """Run the wrapped agent and stream events into the task.

        Calls ``on_event(TaskStatusUpdateEvent | TaskArtifactUpdateEvent)``
        for each transition / final artifact when provided (the
        ``message/stream`` path uses this; ``message/send`` ignores).
        Returns the agent's final text reply.
        """
        return await self._consume(self._start_run(prompt, task), task=task, on_event=on_event)

    async def _consume(
        self,
        events: AsyncIterator[Any],
        *,
        task: Task,
        on_event: Callable[[Any], None] | None,
        completion: tuple[TaskState, str | None] = (TaskState.completed, None),
    ) -> str:
        """Drive ``events`` into ``task`` until the run ends, pauses or is cancelled."""
        from locus.core.events import InterruptEvent, TerminateEvent, ThinkEvent

        final = ""
        task.status = TaskStatus(state=TaskState.working, timestamp=_now_iso())
        self._emit(
            task,
            TaskStatusUpdateEvent(taskId=task.id, contextId=task.contextId, status=task.status),
            on_event,
        )

        try:
            async for event in events:
                if self._store.is_cancel_requested(task.id):
                    task.status = TaskStatus(state=TaskState.canceled, timestamp=_now_iso())
                    self._emit(
                        task,
                        TaskStatusUpdateEvent(
                            taskId=task.id, contextId=task.contextId, status=task.status, final=True
                        ),
                        on_event,
                    )
                    aclose = getattr(events, "aclose", None)
                    if aclose is not None:
                        await aclose()
                    return final
                if isinstance(event, InterruptEvent):
                    self._pause(task, event, on_event)
                    return final
                if isinstance(event, ThinkEvent) and event.reasoning:
                    # Surface intermediate reasoning as a working-state
                    # status update; spec peers can ignore or render it.
                    self._emit(
                        task,
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
                        ),
                        on_event,
                    )
                elif isinstance(event, TerminateEvent):
                    final = event.final_message or final
        except Exception:
            # Subscribers and webhooks hear about the failure; the caller's
            # own handling (the error response) follows from the re-raise.
            task.status = TaskStatus(state=TaskState.failed, timestamp=_now_iso())
            self._emit(
                task,
                TaskStatusUpdateEvent(
                    taskId=task.id, contextId=task.contextId, status=task.status, final=True
                ),
                None,
            )
            raise

        artifact = _agent_artifact(final)
        task.artifacts.append(artifact)
        state, note = completion
        task.status = TaskStatus(
            state=state,
            timestamp=_now_iso(),
            message=(
                _agent_text_message(note, context_id=task.contextId, task_id=task.id)
                if note
                else None
            ),
        )
        self._emit(
            task,
            TaskArtifactUpdateEvent(
                taskId=task.id, contextId=task.contextId, artifact=artifact, lastChunk=True
            ),
            on_event,
        )
        self._emit(
            task,
            TaskStatusUpdateEvent(
                taskId=task.id, contextId=task.contextId, status=task.status, final=True
            ),
            on_event,
        )
        return final

    def _pause(self, task: Task, event: Any, on_event: Callable[[Any], None] | None) -> None:
        """Map a paused run onto ``input-required`` with the question as the status message."""
        event_metadata = event.metadata if isinstance(event.metadata, dict) else {}
        approval_id = event_metadata.get("approval_id")
        metadata: dict[str, Any] = {
            "interruptId": event.interrupt_id,
            "kind": "approval" if approval_id else "question",
        }
        if event.options:
            metadata["options"] = list(event.options)
        if event.fields:
            metadata["fields"] = list(event.fields)
        if approval_id:
            metadata["approvalId"] = approval_id
            metadata["hold"] = {
                key: event_metadata[key]
                for key in ("action", "asset", "principal", "reason", "arguments", "approvers")
                if key in event_metadata
            }
            metadata["replies"] = ["approve", "deny"]
        question = _agent_text_message(
            event.question or "The agent needs your input to continue.",
            context_id=task.contextId,
            task_id=task.id,
            metadata=metadata,
        )
        task.history.append(question)
        task.status = TaskStatus(
            state=TaskState.input_required, message=question, timestamp=_now_iso()
        )
        self._pending[task.id] = _PendingInput(
            thread_id=task.id,
            interrupt_id=event.interrupt_id,
            question=event.question or "",
            approval_id=str(approval_id) if approval_id else None,
            status=task.status,
        )
        self._emit(
            task,
            TaskStatusUpdateEvent(
                taskId=task.id, contextId=task.contextId, status=task.status, final=True
            ),
            on_event,
        )

    async def _resume_run(
        self, task: Task, message: Message, *, on_event: Callable[[Any], None] | None
    ) -> str:
        """Continue a paused run with the peer's reply."""
        pending = self._pending.pop(task.id)
        answer = _extract_user_text(message.parts)
        perform = False
        completion: tuple[TaskState, str | None] = (TaskState.completed, None)
        if pending.approval_id:
            verdict = _verdict_of(answer)
            if verdict is None:
                self._pending[task.id] = pending
                raise _RpcError(
                    INVALID_PARAMS,
                    "the reply to an approval hold must start with approve or deny",
                )
            decided_by = str((message.metadata or {}).get("decidedBy") or f"a2a:{task.contextId}")
            if self._approvals is not None:
                try:
                    self._approvals.decide(pending.approval_id, verdict, by=decided_by)
                except Exception as e:  # noqa: BLE001 — authority refusals, unknown ids
                    self._pending[task.id] = pending
                    raise _RpcError(
                        UNSUPPORTED_OPERATION,
                        f"approval {pending.approval_id} could not be decided: {e}",
                    ) from e
            # The gated call is re-issued under the recorded decision: approve
            # performs it, deny refuses it, and the model sees what happened.
            perform = True
            if verdict == "denied":
                completion = (TaskState.rejected, f"Denied by {decided_by}: {pending.question}")
        from locus.core.errors import ApprovalPendingError

        events: AsyncIterator[Any] = self._agent.resume(
            answer, thread_id=pending.thread_id, perform_dangling=perform
        )
        try:
            return await self._consume(events, task=task, on_event=on_event, completion=completion)
        except ApprovalPendingError as e:
            # The run paused again on the same undecided hold: put the task
            # back where it was so the next reply can decide it.
            self._pending[task.id] = pending
            if pending.status is not None:
                task.status = pending.status
            raise _RpcError(
                UNSUPPORTED_OPERATION,
                f"approval {pending.approval_id} has not been decided; reply approve or deny",
            ) from e

    async def _run_or_resume(
        self,
        task: Task,
        message: Message,
        *,
        resume: bool,
        on_event: Callable[[Any], None] | None = None,
    ) -> str:
        self._active.add(task.id)
        try:
            if resume:
                return await self._resume_run(task, message, on_event=on_event)
            return await self._drive_agent(
                _extract_user_text(message.parts), task=task, on_event=on_event
            )
        finally:
            self._active.discard(task.id)

    def _open_task(self, send: MessageSendParams, *, terminal_error: int) -> tuple[Task, bool]:
        """The task a message addresses (new, or existing) and whether it resumes a pause."""
        if send.message.taskId:
            task = self._store.get(send.message.taskId)
            if task is None:
                raise _RpcError(TASK_NOT_FOUND, f"task {send.message.taskId} not found")
            if task.status.state in _TERMINAL_STATES:
                raise _RpcError(terminal_error, "task is in a terminal state")
            send.message.contextId = send.message.contextId or task.contextId
            task.history.append(send.message)
            resume = (
                task.status.state == TaskState.input_required
                and task.id in self._pending
                and self._can_resume
            )
            return task, resume
        context_id = send.message.contextId or uuid.uuid4().hex
        task = Task(
            id=uuid.uuid4().hex,
            contextId=context_id,
            status=TaskStatus(state=TaskState.submitted, timestamp=_now_iso()),
            history=[send.message],
        )
        self._store.put(task)
        return task, False

    def _register_push_from_configuration(self, task: Task, params: dict[str, Any]) -> None:
        configuration = params.get("configuration") or {}
        raw = configuration.get("taskPushNotificationConfig") or configuration.get(
            "pushNotificationConfig"
        )
        if not raw:
            return
        if self._push is None:
            raise _RpcError(
                PUSH_NOTIFICATION_NOT_SUPPORTED, "push notifications are not enabled on this agent"
            )
        try:
            config = A2AV1TaskPushNotificationConfig.model_validate(raw)
            self._push.register(task.id, config)
        except PushNotificationError as e:
            raise _RpcError(INVALID_PARAMS, str(e)) from e
        except Exception as e:  # noqa: BLE001
            raise _RpcError(INVALID_PARAMS, f"invalid push notification config: {e}") from e

    def _in_background(self, coro: Any, task: Task) -> None:
        async def runner() -> None:
            try:
                await coro
            except A2AV1ProtocolError as e:
                _logger.warning("A2A background run for task %s refused: %s", task.id, e.message)
            except Exception:
                _logger.exception("A2A background run failed for task %s", task.id)
            finally:
                self._active.discard(task.id)

        handle = asyncio.ensure_future(runner())
        self._background.add(handle)
        handle.add_done_callback(self._background.discard)

    async def drain(self) -> None:
        """Wait for background runs and push deliveries to finish (tests, shutdown)."""
        while True:
            self._background.difference_update([t for t in self._background if t.done()])
            pending = list(self._background)
            if not pending:
                break
            await asyncio.gather(*pending, return_exceptions=True)
        if self._push is not None:
            await self._push.drain()

    async def aclose(self) -> None:
        """Finish background work and release the push dispatcher's connections."""
        await self.drain()
        if self._push is not None:
            await self._push.aclose()

    # ----- JSON-RPC method handlers ------------------------------------

    async def _handle_message_send(
        self, params: dict[str, Any], *, terminal_error: int = TASK_NOT_CANCELABLE
    ) -> dict[str, Any]:
        try:
            send = MessageSendParams.model_validate(params)
        except Exception as e:  # noqa: BLE001
            raise _RpcError(INVALID_PARAMS, f"invalid params: {e}") from e
        task, resume = self._open_task(send, terminal_error=terminal_error)
        self._register_push_from_configuration(task, params)
        configuration = params.get("configuration") or {}
        detached = configuration.get("returnImmediately") is True or (
            configuration.get("blocking") is False
        )
        work = self._run_or_resume(task, send.message, resume=resume)
        if detached:
            self._active.add(task.id)
            self._in_background(work, task)
            return task.model_dump()
        try:
            await work
        except A2AV1ProtocolError:
            raise
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
        self._pending.pop(task.id, None)
        self._emit(
            task,
            TaskStatusUpdateEvent(
                taskId=task.id, contextId=task.contextId, status=task.status, final=True
            ),
            None,
        )
        return task.model_dump()

    async def _stream_message(
        self, params: dict[str, Any], *, terminal_error: int = TASK_NOT_CANCELABLE
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream a message run as JSON-RPC responses (one per SSE event)."""
        try:
            send = MessageSendParams.model_validate(params)
        except Exception as e:  # noqa: BLE001
            raise _RpcError(INVALID_PARAMS, f"invalid params: {e}") from e
        task, resume = self._open_task(send, terminal_error=terminal_error)
        self._register_push_from_configuration(task, params)

        # Initial event: the task as it sits now (submitted, or paused).
        yield task.model_dump()

        # Bridge the synchronous on_event callback to an async queue so
        # SSE consumers see updates as they happen.
        queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()
        refused: list[A2AV1ProtocolError] = []

        def emit(ev: Any) -> None:
            queue.put_nowait(ev)

        async def runner() -> None:
            try:
                await self._run_or_resume(task, send.message, resume=resume, on_event=emit)
            except A2AV1ProtocolError as e:
                refused.append(e)
            except Exception:  # noqa: BLE001
                _logger.exception("A2A message/stream failed for task %s", task.id)
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
        if refused:
            raise refused[0]

    # ----- backward-compat: the legacy A2ARequest invoke shape --------

    async def _legacy_invoke(self, request: A2ARequest) -> A2AResponse:
        import time as _time  # noqa: PLC0415

        from locus.core.events import TerminateEvent
        from locus.observability.emit import (  # noqa: PLC0415
            EV_A2A_TASK_COMPLETED,
            EV_A2A_TASK_RECEIVED,
            emit,
        )

        # The flat routes carry no JSON-RPC method; the route name stands in.
        await emit(EV_A2A_TASK_RECEIVED, method="a2a/invoke", rpc_id=None)
        _started = _time.perf_counter()
        user_msgs = [m for m in request.messages if m.role == "user"]
        prompt = user_msgs[-1].content if user_msgs else ""
        final = ""
        stop_reason = "complete"
        iterations = 0
        try:
            async for event in self._agent.run(prompt):
                if isinstance(event, TerminateEvent):
                    final = event.final_message or final
                    stop_reason = event.reason or stop_reason
                iterations += 1
        except Exception:
            await emit(
                EV_A2A_TASK_COMPLETED,
                method="a2a/invoke",
                success=False,
                error_code=INTERNAL_ERROR,
                duration_ms=(_time.perf_counter() - _started) * 1000,
            )
            raise
        await emit(
            EV_A2A_TASK_COMPLETED,
            method="a2a/invoke",
            success=True,
            duration_ms=(_time.perf_counter() - _started) * 1000,
        )
        return A2AResponse(
            messages=[A2AMessage(role="agent", content=final)],
            status="completed",
            metadata={"stop_reason": stop_reason, "iterations": iterations},
        )

    # ----- app construction --------------------------------------------

    def _create_app(self) -> Any:
        try:
            from fastapi import Body, Depends, FastAPI, Header
            from fastapi.responses import StreamingResponse
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
            return self._build_card().model_dump(exclude_none=True)

        @app.get("/agent-card")
        async def legacy_card(_: str = auth_dep) -> dict[str, Any]:
            # Backwards-compat alias. Matches the old flat shape so
            # peers that still expect ``skills: list[str]`` keep
            # parsing — we re-emit just id/name/description per skill.
            card = self._build_card()
            payload = card.model_dump(exclude_none=True)
            payload["skills"] = [s.name for s in card.skills]
            return payload

        # The signing keys for push notifications are public by design: a
        # receiver fetches them to verify a notification, before it has any
        # credential of ours.
        if self._push is not None and self._push.settings.signer is not None:
            signer = self._push.settings.signer

            @app.get(JWKS_PATH)
            async def jwks() -> dict[str, Any]:
                return signer.jwks()

        # ----- JSON-RPC root -----------------------------------------------
        if BINDING_JSONRPC in self._transports:

            @app.post("/")
            async def jsonrpc(
                request: Request,
                body: dict[str, Any] = Body(...),  # noqa: B008 — FastAPI dep injection
                a2a_version: str | None = Header(default=None, alias=A2A_VERSION_HEADER),
                _: str = auth_dep,
            ) -> Any:
                return await self._jsonrpc(body, a2a_version, carrier=request.headers)

        # ----- HTTP+JSON binding ------------------------------------------
        if BINDING_HTTP_JSON in self._transports:
            from locus.a2a.rest import mount_http_json

            mount_http_json(app, self, auth_dep)

        # ----- backwards-compat invoke + stream ----------------------------
        @app.post("/a2a/invoke")
        async def legacy_invoke(
            request: Request, req: A2ARequest, _: str = auth_dep
        ) -> dict[str, Any]:
            # Run under the caller's trace, if its headers carry one.
            with trace_scope(request.headers):
                resp = await self._legacy_invoke(req)
            return resp.model_dump()

        @app.post("/a2a/stream")
        async def legacy_stream(
            request: Request, req: A2ARequest, _: str = auth_dep
        ) -> StreamingResponse:
            response: StreamingResponse = await self._legacy_stream(
                req, carrier=dict(request.headers)
            )
            return response

        return app

    async def _jsonrpc(
        self,
        body: dict[str, Any],
        a2a_version: str | None,
        *,
        carrier: Mapping[str, Any] | None = None,
    ) -> Any:
        """Dispatch one JSON-RPC request body (``POST /``).

        ``carrier`` is the request's headers; a ``traceparent`` in them puts
        the dispatch — and the agent run it starts — under the caller's trace.
        """
        with trace_scope(carrier):
            return await self._dispatch_jsonrpc(body, a2a_version, carrier)

    async def _dispatch_jsonrpc(
        self,
        body: dict[str, Any],
        a2a_version: str | None,
        carrier: Mapping[str, Any] | None,
    ) -> Any:
        from fastapi.responses import JSONResponse

        try:
            req = JsonRpcRequest.model_validate(body)
        except Exception:  # noqa: BLE001
            return JSONResponse(
                _jsonrpc_error_response(
                    body.get("id") if isinstance(body, dict) else None,
                    INVALID_REQUEST,
                    "Invalid Request",
                )
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
            if a2a_version not in (None, A2A_V1_PROTOCOL_VERSION):
                raise _RpcError(
                    VERSION_NOT_SUPPORTED,
                    f"A2A version {a2a_version!r} is not supported",
                )
            # Some v1 clients omit the version header; route known v1
            # method names to the v1 handlers to preserve interoperability.
            use_v1 = a2a_version == A2A_V1_PROTOCOL_VERSION or (
                a2a_version is None and req.method in A2AV1_JSONRPC_METHODS
            )
            if use_v1 and req.method in A2AV1_STREAMING_METHODS:
                if req.method == "SubscribeToTask":
                    self._preflight_v1_task_subscription(params)
                return await self._stream_response(req, v1=True, carrier=carrier, started=_started)
            if use_v1 and req.method in A2AV1_JSONRPC_METHODS:
                result = await self.handle_v1(req.method, params)
            elif req.method == "message/send":
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
                # The pre-1.0 method names; peers on 1.0 use the
                # ``*TaskPushNotificationConfig`` methods above.
                raise _RpcError(
                    PUSH_NOTIFICATION_NOT_SUPPORTED,
                    "push notifications use the A2A 1.0 methods on this agent",
                )
            elif req.method == "message/stream":
                # SSE — peer must POST with Accept: text/event-stream.
                return await self._stream_response(req, carrier=carrier, started=_started)
            else:
                raise _RpcError(METHOD_NOT_FOUND, f"unknown method {req.method!r}")
        except A2AV1ProtocolError as e:
            await emit(
                EV_A2A_TASK_COMPLETED,
                method=req.method,
                success=False,
                error_code=e.code,
                duration_ms=(_time.perf_counter() - _started) * 1000,
            )
            return JSONResponse(_jsonrpc_error_response(req.id, e.code, e.message))

        await emit(
            EV_A2A_TASK_COMPLETED,
            method=req.method,
            success=True,
            duration_ms=(_time.perf_counter() - _started) * 1000,
        )
        return JSONResponse(
            JsonRpcSuccessResponse(id=req.id, result=result).model_dump(exclude_none=True)
        )

    def _stream_completion(self, method: str, started: float) -> Callable[[bool, int | None], None]:
        """A callable that publishes ``a2a.task.completed`` for a streaming response.

        The generator may be finalised by the event loop in another context
        once the client is gone; the completion is published in a copy of the
        context the request was served in, so it carries the run id
        ``received`` was emitted under. ``started`` is the clock the request
        arrived on, so ``duration_ms`` covers the whole request.
        """
        import contextvars  # noqa: PLC0415
        import time as _time  # noqa: PLC0415

        # Lazy import — observability is opt-in.
        from locus.observability.emit import EV_A2A_TASK_COMPLETED, emit_sync  # noqa: PLC0415

        request_context = contextvars.copy_context()

        def completed(success: bool, error_code: int | None) -> None:
            data: dict[str, Any] = {
                "method": method,
                "success": success,
                "duration_ms": (_time.perf_counter() - started) * 1000,
            }
            if error_code is not None:
                data["error_code"] = error_code
            request_context.run(emit_sync, EV_A2A_TASK_COMPLETED, **data)

        return completed

    async def _stream_response(
        self,
        req: JsonRpcRequest,
        *,
        v1: bool = False,
        carrier: Mapping[str, Any] | None = None,
        started: float | None = None,
    ) -> Any:
        """The SSE response for a streaming method, which also closes the
        ``a2a.task.received`` / ``a2a.task.completed`` bracket.

        The dispatcher emitted ``received`` and returns this response before
        its own ``completed`` emit, so the stream publishes ``completed`` when
        its generator ends: on success, on a protocol or internal error (with
        the error code the client was sent) and when the client leaves
        mid-stream (``success`` false, no error code). An agent failure
        inside the stream reaches the client as a final ``failed`` status
        update, so that stream counts as served. ``started`` is the
        dispatcher's clock, so ``duration_ms`` covers the whole request.
        """
        import time as _time  # noqa: PLC0415

        from fastapi.responses import StreamingResponse

        params = req.params if isinstance(req.params, dict) else {}
        _started = _time.perf_counter() if started is None else started
        completed = self._stream_completion(req.method, _started)

        async def gen() -> AsyncIterator[str]:
            # The generator runs after the handler returned, so the caller's
            # trace is re-attached here for the run it drives.
            with trace_scope(carrier):
                stream = produce()
                try:
                    async for chunk in stream:
                        yield chunk
                finally:
                    # Close the inner generator here rather than leaving it
                    # to the loop's finaliser, so the completion lands as
                    # soon as this response ends, however it ends.
                    await stream.aclose()

        async def produce() -> AsyncGenerator[str, None]:
            # ``None`` until the stream ends on its own; a client that goes
            # away mid-stream closes the generator before it is set.
            outcome: tuple[bool, int | None] | None = None
            try:
                stream = (
                    self.handle_v1_stream(req.method, params)
                    if v1
                    else self._stream_message(params)
                )
                async for result in stream:
                    payload = JsonRpcSuccessResponse(id=req.id, result=result).model_dump(
                        exclude_none=True
                    )
                    yield f"data: {json.dumps(payload)}\n\n"
                outcome = (True, None)
            except A2AV1ProtocolError as e:
                outcome = (False, e.code)
                payload = _jsonrpc_error_response(req.id, e.code, e.message)
                yield f"data: {json.dumps(payload)}\n\n"
            except Exception:  # noqa: BLE001 — sanitise stream errors
                outcome = (False, INTERNAL_ERROR)
                correlation_id = uuid.uuid4().hex
                _logger.exception("A2A stream error (correlation_id=%s)", correlation_id)
                payload = JsonRpcErrorResponse(
                    id=req.id,
                    error=JsonRpcError(
                        code=INTERNAL_ERROR,
                        message="internal error",
                        data={"correlation_id": correlation_id},
                    ),
                ).model_dump(exclude_none=True)
                yield f"data: {json.dumps(payload)}\n\n"
            finally:
                success, error_code = outcome if outcome is not None else (False, None)
                completed(success, error_code)

        return StreamingResponse(gen(), media_type="text/event-stream")

    async def _legacy_stream(
        self, request: A2ARequest, *, carrier: Mapping[str, Any] | None = None
    ) -> Any:
        import time as _time  # noqa: PLC0415

        from fastapi.responses import StreamingResponse

        from locus.core.events import TerminateEvent, ThinkEvent
        from locus.observability.emit import EV_A2A_TASK_RECEIVED, emit  # noqa: PLC0415

        await emit(EV_A2A_TASK_RECEIVED, method="a2a/stream", rpc_id=None)
        completed = self._stream_completion("a2a/stream", _time.perf_counter())

        user_msgs = [m for m in request.messages if m.role == "user"]
        prompt = user_msgs[-1].content if user_msgs else ""

        async def event_generator() -> AsyncIterator[str]:
            with trace_scope(carrier):
                stream = produce()
                try:
                    async for chunk in stream:
                        yield chunk
                finally:
                    await stream.aclose()

        async def produce() -> AsyncGenerator[str, None]:
            outcome: tuple[bool, int | None] | None = None
            try:
                async for event in self._agent.run(prompt):
                    if isinstance(event, ThinkEvent):
                        data = {"type": "text", "content": event.reasoning or ""}
                    elif isinstance(event, TerminateEvent):
                        data = {"type": "done", "content": event.final_message or ""}
                    else:
                        data = {"type": event.event_type}
                    yield f"data: {json.dumps(data)}\n\n"
                outcome = (True, None)
            except Exception:  # noqa: BLE001
                outcome = (False, INTERNAL_ERROR)
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
                success, error_code = outcome if outcome is not None else (False, None)
                completed(success, error_code)
            # Outside the ``finally``: a generator the client closed must not
            # yield again, and the terminator belongs to a stream that ran to
            # its end, on the agent's answer or on the error frame.
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # ----- gRPC ----------------------------------------------------------

    def grpc_server(
        self, host: str = "127.0.0.1", port: int | None = None, *, credentials: Any | None = None
    ) -> tuple[Any, int]:
        """A ``grpc.aio`` server with ``lf.a2a.v1.A2AService`` bound; ``(server, port)``.

        Not started: call ``await server.start()``. ``port`` defaults to
        ``grpc_port``; ``0`` picks a free port (the bound one is returned).
        Needs the ``a2a-grpc`` extra.
        """
        from locus.a2a.grpc_binding import build_grpc_server

        chosen = self._grpc_port if port is None else port
        if chosen is None:
            raise ValueError("no gRPC port: pass port= or construct A2AServer(grpc_port=...)")
        return build_grpc_server(self, host, chosen, credentials=credentials)

    async def _serve_with_grpc(self, host: str, port: int, **kwargs: Any) -> None:
        import uvicorn

        http = uvicorn.Server(uvicorn.Config(self.app, host=host, port=port, **kwargs))
        grpc_server, bound = self.grpc_server(host)
        await grpc_server.start()
        _logger.info("A2A gRPC binding listening on %s:%d", host, bound)
        try:
            await http.serve()
        finally:
            await grpc_server.stop(grace=1.0)
            await self.aclose()

    def run(self, host: str = "127.0.0.1", port: int = 8001, **kwargs: Any) -> None:
        """Run the A2A server.

        Defaults to loopback binding. Non-loopback bindings require
        either ``api_key`` to be set or ``allow_unauthenticated=True``.
        With ``grpc_port`` set, the gRPC binding is served alongside HTTP.
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
        if BINDING_GRPC in self._transports and self._grpc_port is not None:
            asyncio.run(self._serve_with_grpc(host, port, **kwargs))
            return
        uvicorn.run(self.app, host=host, port=port, **kwargs)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class A2AClient:
    """Call a remote A2A agent from Locus.

    Spec-compliant methods:

    - :meth:`get_agent_card` — fetches ``/.well-known/agent-card.json``,
      falling back to the legacy ``/agent-card`` endpoint.
    - :meth:`send_message` — ``SendMessage``; returns a :class:`Task` you
      can poll with :meth:`get_task` or follow with :meth:`subscribe`.
    - :meth:`send_message_streaming` — ``SendStreamingMessage``; yields
      events from the stream.
    - :meth:`get_task`, :meth:`list_tasks`, :meth:`cancel_task` — task
      lifecycle.
    - :meth:`set_push_notification_config` and friends — webhooks.

    Transports: a client built with the constructor speaks one binding
    (``binding=``, JSON-RPC by default) at ``url``. :meth:`from_card` and
    :meth:`connect` read the peer's ``supportedInterfaces`` and order them
    by ``prefer``; when an interface proves unreachable the client moves on
    to the next one and stays there.

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
        protocol_version: str | None = A2A_V1_PROTOCOL_VERSION,
        *,
        binding: str | None = None,
        prefer: Sequence[str] | None = None,
        http_client: httpx.AsyncClient | None = None,
        grpc_channel: Any | None = None,
    ) -> None:
        self._url = url.rstrip("/")
        self._api_key = api_key
        self._timeout: Any = self.DEFAULT_TIMEOUT if timeout is None else timeout
        self._protocol_version = protocol_version
        self._prefer: tuple[str, ...] = tuple(prefer) if prefer else DEFAULT_BINDING_PREFERENCE
        self._http_client = http_client
        self._grpc_channel = grpc_channel
        self._card: AgentCard | None = None
        self._transports: list[ClientTransport] = [
            self._make_transport(normalise_binding(binding) if binding else BINDING_JSONRPC, url)
        ]

    # ---- transports -----------------------------------------------------

    def _make_transport(self, binding: str, url: str) -> ClientTransport:
        return make_transport(
            binding,
            url,
            api_key=self._api_key,
            timeout=self._timeout,
            http_client=self._http_client,
            grpc_channel=self._grpc_channel if binding == BINDING_GRPC else None,
        )

    def _use_card(self, card: AgentCard) -> None:
        self._card = card
        transports: list[ClientTransport] = []
        for binding, url in select_interfaces(card, prefer=self._prefer, fallback_url=self._url):
            try:
                transports.append(self._make_transport(binding, url))
            except ImportError as e:
                _logger.info("A2A %s interface at %s skipped: %s", binding, url, e)
        self._transports = transports or [self._make_transport(BINDING_JSONRPC, self._url)]

    @classmethod
    def from_card(
        cls,
        card: AgentCard,
        *,
        api_key: str | None = None,
        timeout: float | httpx.Timeout | None = None,
        prefer: Sequence[str] | None = None,
        http_client: httpx.AsyncClient | None = None,
        grpc_channel: Any | None = None,
    ) -> A2AClient:
        """A client on the binding ``card`` advertises first in ``prefer`` order."""
        client = cls(
            card.url,
            api_key,
            timeout,
            prefer=prefer,
            http_client=http_client,
            grpc_channel=grpc_channel,
        )
        client._use_card(card)
        return client

    @classmethod
    async def connect(
        cls,
        url: str,
        *,
        api_key: str | None = None,
        timeout: float | httpx.Timeout | None = None,  # noqa: ASYNC109 — the transports' httpx timeout
        prefer: Sequence[str] | None = None,
        http_client: httpx.AsyncClient | None = None,
        grpc_channel: Any | None = None,
    ) -> A2AClient:
        """Fetch the card at ``url`` and pick a binding from it."""
        client = cls(
            url,
            api_key,
            timeout,
            prefer=prefer,
            http_client=http_client,
            grpc_channel=grpc_channel,
        )
        client._use_card(await client.get_agent_card())
        return client

    @property
    def binding(self) -> str:
        """The protocol binding the next call uses."""
        return self._transports[0].binding

    @property
    def transports(self) -> tuple[ClientTransport, ...]:
        """The transports left to try, preferred first."""
        return tuple(self._transports)

    @property
    def card(self) -> AgentCard | None:
        """The card the transports were chosen from, when there was one."""
        return self._card

    def _advance(self, failed: ClientTransport, error: A2ATransportUnavailableError) -> None:
        if len(self._transports) == 1:
            raise error
        self._transports.pop(0)
        _logger.warning(
            "A2A %s interface at %s unavailable (%s); using %s at %s",
            failed.binding,
            failed.url,
            error.message,
            self._transports[0].binding,
            self._transports[0].url,
        )

    async def _call(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: Any = None,  # noqa: ASYNC109 — forwarded to the transport, not an asyncio timer
    ) -> dict[str, Any]:
        while True:
            transport = self._transports[0]
            try:
                return await transport.call(method, params, timeout=timeout)
            except A2ATransportUnavailableError as e:
                self._advance(transport, e)

    async def _stream(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: Any = None,  # noqa: ASYNC109 — forwarded to the transport, not an asyncio timer
    ) -> AsyncIterator[dict[str, Any]]:
        while True:
            transport = self._transports[0]
            try:
                async for frame in transport.stream(method, params, timeout=timeout):
                    yield frame
                return
            except A2ATransportUnavailableError as e:
                self._advance(transport, e)

    async def aclose(self) -> None:
        """Release every transport's connection."""
        for transport in self._transports:
            await transport.aclose()

    def _auth_headers(self) -> dict[str, str]:
        if self._api_key:
            return {"Authorization": f"Bearer {self._api_key}"}
        return {}

    def _headers(self, *, include_version: bool = True) -> dict[str, str]:
        headers = self._auth_headers()
        if include_version and self._protocol_version:
            headers[A2A_VERSION_HEADER] = self._protocol_version
        return headers

    async def _http(self) -> httpx.AsyncClient:
        import httpx

        return self._http_client or httpx.AsyncClient(timeout=self._timeout)

    async def get_agent_card(self) -> AgentCard:
        """Fetch the remote agent's capability card.

        Tries the spec well-known URL first, falls back to the legacy
        ``/agent-card`` endpoint for older peers.
        """
        import httpx

        client = await self._http()
        try:
            for path in ("/.well-known/agent-card.json", "/agent-card"):
                try:
                    resp = await client.get(
                        f"{self._url}{path}", headers=inject_trace_headers(self._headers())
                    )
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
        finally:
            if client is not self._http_client:
                await client.aclose()
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
        """One method call; on the v1.0 path through the chosen transport."""
        if self._protocol_version == A2A_V1_PROTOCOL_VERSION:
            return await self._call(method, params, timeout=timeout)

        import time as _time

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
        client = await self._http()
        try:
            resp = await client.post(
                f"{self._url}/",
                json=body,
                headers=inject_trace_headers(self._headers()),
                timeout=self._timeout if timeout is None else timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            # Close the bracket with what arrived, if anything did.
            response = getattr(e, "response", None)
            await emit(
                EV_A2A_CLIENT_RECEIVED,
                target_url=self._url,
                method=method,
                status_code=getattr(response, "status_code", None),
                duration_ms=(_time.perf_counter() - _started) * 1000,
                content_length=len(response.content) if response is not None else 0,
                error=type(e).__name__,
            )
            raise
        finally:
            if client is not self._http_client:
                await client.aclose()
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
            raise A2AClientError(
                int(err.get("code", INTERNAL_ERROR)),
                str(err.get("message", "")),
                data=err.get("data"),
            )
        return data["result"]

    @staticmethod
    def _configuration(
        push_notification_config: A2AV1TaskPushNotificationConfig | dict[str, Any] | None,
        blocking: bool | None,
    ) -> dict[str, Any] | None:
        configuration: dict[str, Any] = {}
        if push_notification_config is not None:
            config = (
                push_notification_config
                if isinstance(push_notification_config, A2AV1TaskPushNotificationConfig)
                else A2AV1TaskPushNotificationConfig.model_validate(push_notification_config)
            )
            configuration["taskPushNotificationConfig"] = config.model_dump(exclude_none=True)
        if blocking is not None:
            configuration["blocking"] = blocking
            if not blocking:
                configuration["returnImmediately"] = True
        return configuration or None

    async def send_message(
        self,
        message: Message,
        *,
        timeout: float | httpx.Timeout | None = None,  # noqa: ASYNC109 — forwarded to httpx, not an asyncio timer
        push_notification_config: A2AV1TaskPushNotificationConfig | dict[str, Any] | None = None,
        blocking: bool | None = None,
    ) -> Task:
        """Send a message (``SendMessage``) and return the Task.

        ``push_notification_config`` registers a webhook for the new task in
        the same call. ``blocking=False`` asks the peer to return the task as
        soon as it is created and run it in the background; follow it with
        :meth:`get_task`, :meth:`subscribe` or push notifications.
        """
        configuration = self._configuration(push_notification_config, blocking)
        if self._protocol_version == A2A_V1_PROTOCOL_VERSION:
            params: dict[str, Any] = {
                "message": legacy_message_to_v1(message).model_dump(exclude_none=True)
            }
            if configuration:
                params["configuration"] = configuration
            result = await self._rpc("SendMessage", params, timeout=timeout)
            return Task.model_validate(task_result_to_legacy_payload(result))
        legacy_params: dict[str, Any] = {"message": message.model_dump(exclude_none=True)}
        if configuration:
            legacy_params["configuration"] = configuration
        result = await self._rpc("message/send", legacy_params, timeout=timeout)
        return Task.model_validate(result)

    async def send_message_streaming(
        self,
        message: Message,
        *,
        timeout: float | httpx.Timeout | None = None,  # noqa: ASYNC109 — forwarded to httpx, not an asyncio timer
        push_notification_config: A2AV1TaskPushNotificationConfig | dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Send a message on the streaming method and yield events."""
        configuration = self._configuration(push_notification_config, None)
        if self._protocol_version == A2A_V1_PROTOCOL_VERSION:
            params: dict[str, Any] = {
                "message": legacy_message_to_v1(message).model_dump(exclude_none=True)
            }
            if configuration:
                params["configuration"] = configuration
            async for frame in self._stream("SendStreamingMessage", params, timeout=timeout):
                if "error" in frame:
                    yield frame  # surface error envelope to caller
                    break
                yield v1_stream_response_to_legacy_payload(frame)
            return

        import time as _time  # noqa: PLC0415

        from locus.observability.emit import (  # noqa: PLC0415
            EV_A2A_CLIENT_RECEIVED,
            EV_A2A_CLIENT_SEND,
            emit,
        )

        body = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex,
            "method": "message/stream",
            "params": {"message": message.model_dump(exclude_none=True)},
        }
        headers = inject_trace_headers(self._headers() | {"Accept": "text/event-stream"})
        effective_timeout = self._timeout if timeout is None else timeout
        await emit(EV_A2A_CLIENT_SEND, target_url=self._url, method="message/stream")
        _started = _time.perf_counter()
        status: int | None = None
        frames = 0
        error: str | None = None
        client = await self._http()
        try:
            async with client.stream(
                "POST", f"{self._url}/", json=body, headers=headers, timeout=effective_timeout
            ) as resp:
                status = resp.status_code
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
                        frames += 1
                        yield env  # surface error envelope to caller
                        break
                    frames += 1
                    yield env.get("result", env)
        except Exception as e:
            error = "HTTPStatusError" if status is not None and status >= 400 else type(e).__name__
            raise
        finally:
            if client is not self._http_client:
                await client.aclose()
            data: dict[str, Any] = {
                "target_url": self._url,
                "method": "message/stream",
                "status_code": status,
                "duration_ms": (_time.perf_counter() - _started) * 1000,
                "frames": frames,
            }
            if error is not None:
                data["error"] = error
            await emit(EV_A2A_CLIENT_RECEIVED, **data)

    async def subscribe(self, task_id: str) -> AsyncIterator[dict[str, Any]]:
        """Follow a running task (``SubscribeToTask``): the snapshot, then live events."""
        if self._protocol_version != A2A_V1_PROTOCOL_VERSION:
            raise RuntimeError("subscribe requires A2A v1.0 protocol_version")
        async for frame in self._stream("SubscribeToTask", {"id": task_id}):
            if "error" in frame:
                yield frame
                break
            yield v1_stream_response_to_legacy_payload(frame)

    async def get_task(self, task_id: str, history_length: int | None = None) -> Task:
        """Fetch a task by id (``GetTask``)."""
        params: dict[str, Any] = {"id": task_id}
        if history_length is not None:
            params["historyLength"] = history_length
        if self._protocol_version == A2A_V1_PROTOCOL_VERSION:
            result = await self._rpc("GetTask", params)
            return Task.model_validate(task_result_to_legacy_payload(result))
        result = await self._rpc("tasks/get", params)
        return Task.model_validate(result)

    async def list_tasks(
        self,
        *,
        context_id: str | None = None,
        status: TaskState | str | None = None,
        page_size: int | None = None,
        page_token: str | None = None,
        history_length: int | None = None,
        include_artifacts: bool | None = None,
    ) -> tuple[list[Task], str]:
        """List known tasks.

        Returns ``(tasks, next_page_token)``. The client maps v1.0 wire
        enum states back into SDK-shaped :class:`Task` objects.
        """
        params: dict[str, Any] = {}
        if context_id is not None:
            params["contextId"] = context_id
        if status is not None:
            state = status.value if isinstance(status, TaskState) else str(status)
            if self._protocol_version == A2A_V1_PROTOCOL_VERSION and not state.startswith(
                "TASK_STATE_"
            ):
                state = "TASK_STATE_" + state.replace("-", "_").upper()
            params["status"] = state
        if page_size is not None:
            params["pageSize"] = page_size
        if page_token is not None:
            params["pageToken"] = page_token
        if history_length is not None:
            params["historyLength"] = history_length
        if include_artifacts is not None:
            params["includeArtifacts"] = include_artifacts

        if self._protocol_version == A2A_V1_PROTOCOL_VERSION:
            result = await self._rpc("ListTasks", params)
            tasks = [
                Task.model_validate(task_result_to_legacy_payload(task))
                for task in result.get("tasks", [])
            ]
            return tasks, str(result.get("nextPageToken", ""))

        raise RuntimeError("list_tasks requires A2A v1.0 protocol_version")

    async def cancel_task(self, task_id: str) -> Task:
        """Cancel a task (``CancelTask``)."""
        if self._protocol_version == A2A_V1_PROTOCOL_VERSION:
            result = await self._rpc("CancelTask", {"id": task_id})
            return Task.model_validate(task_result_to_legacy_payload(result))
        result = await self._rpc("tasks/cancel", {"id": task_id})
        return Task.model_validate(result)

    # ---- push notifications -------------------------------------------

    def _require_v1(self, what: str) -> None:
        if self._protocol_version != A2A_V1_PROTOCOL_VERSION:
            raise RuntimeError(f"{what} requires A2A v1.0 protocol_version")

    async def set_push_notification_config(
        self,
        task_id: str,
        config: A2AV1TaskPushNotificationConfig | dict[str, Any],
    ) -> A2AV1TaskPushNotificationConfig:
        """Register a webhook for ``task_id`` (``CreateTaskPushNotificationConfig``)."""
        self._require_v1("push notification configs")
        model = (
            config
            if isinstance(config, A2AV1TaskPushNotificationConfig)
            else A2AV1TaskPushNotificationConfig.model_validate(config)
        )
        params = model.model_dump(exclude_none=True)
        params["taskId"] = task_id
        result = await self._rpc("CreateTaskPushNotificationConfig", params)
        return A2AV1TaskPushNotificationConfig.model_validate(result)

    async def get_push_notification_config(
        self, task_id: str, config_id: str
    ) -> A2AV1TaskPushNotificationConfig:
        """One registered webhook (``GetTaskPushNotificationConfig``)."""
        self._require_v1("push notification configs")
        result = await self._rpc(
            "GetTaskPushNotificationConfig", {"taskId": task_id, "id": config_id}
        )
        return A2AV1TaskPushNotificationConfig.model_validate(result)

    async def list_push_notification_configs(
        self, task_id: str
    ) -> list[A2AV1TaskPushNotificationConfig]:
        """Every webhook registered on ``task_id`` (``ListTaskPushNotificationConfigs``)."""
        self._require_v1("push notification configs")
        result = await self._rpc("ListTaskPushNotificationConfigs", {"taskId": task_id})
        return [
            A2AV1TaskPushNotificationConfig.model_validate(item)
            for item in result.get("configs", [])
        ]

    async def delete_push_notification_config(self, task_id: str, config_id: str) -> None:
        """Remove a webhook (``DeleteTaskPushNotificationConfig``); a repeat is a no-op."""
        self._require_v1("push notification configs")
        await self._rpc("DeleteTaskPushNotificationConfig", {"taskId": task_id, "id": config_id})

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
        request = A2ARequest(messages=[A2AMessage(role="user", content=prompt)])
        client = await self._http()
        try:
            resp = await client.post(
                f"{self._url}/a2a/invoke",
                json=request.model_dump(),
                headers=inject_trace_headers(self._auth_headers()),
                timeout=self._timeout if timeout is None else timeout,
            )
            resp.raise_for_status()
            response = A2AResponse.model_validate(resp.json())
        finally:
            if client is not self._http_client:
                await client.aclose()
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
    "A2AClientError",
    "A2AMessage",
    "A2ARequest",
    "A2AResponse",
    "A2AServer",
    "AgentCard",
]
