# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Threads, runs, resumable SSE, the live WebSocket and the UI protocols.

Mounted on the :class:`~locus.server.app.AgentServer` application by
:func:`mount_run_routes`, next to ``/invoke`` and ``/stream``:

- ``POST /threads`` — create a thread (id chosen by the client or minted).
- ``POST /threads/{id}/runs`` — start a turn, or continue a paused one;
  answers at once with the run id while the agent runs in the background.
- ``GET /runs/{id}`` — the run's status and outcome.
- ``GET /runs/{id}/events`` — the run's events as SSE, resumable with
  ``Last-Event-ID``; ``?protocol=ag-ui`` or ``?protocol=vercel-ai`` for the
  UI wire protocols.
- ``POST /runs/{id}/cancel`` — stop a run.
- ``WS /runs/live`` — one socket for turns, approval answers, cancellation
  and the events of every run started or subscribed on it.
- ``POST /ag-ui`` and ``POST /vercel-ai`` — the endpoints the two
  protocols' clients post to; each starts a run and streams it.

Every route is bearer-checked with the application's dependency and
thread-scoped by principal, as the historical routes are.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import inspect
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from locus.server.protocols import (
    PROTOCOL_AG_UI,
    PROTOCOL_LOCUS,
    PROTOCOL_NAMES,
    PROTOCOL_VERCEL_AI,
    StreamAdapter,
    ag_ui_run_input,
    encode_sse,
    make_adapter,
    ui_message_input,
)
from locus.server.runs import (
    RegistryError,
    RunInProgressError,
    RunRecord,
    RunRegistry,
)


# Route signatures are resolved against this module's globals by FastAPI,
# so the Starlette types must be importable here without making the
# ``server`` extra a hard dependency.
try:
    from starlette.requests import Request
    from starlette.websockets import WebSocket
except ImportError:  # pragma: no cover — server extra not installed
    Request = Any  # type: ignore[misc,assignment]
    WebSocket = Any  # type: ignore[misc,assignment]

if TYPE_CHECKING:
    from fastapi import FastAPI

_logger = logging.getLogger(__name__)

DONE_FRAME = "data: [DONE]\n\n"


# ----------------------------------------------------------------------
# Request bodies
# ----------------------------------------------------------------------


class ThreadCreateRequest(BaseModel):
    """Body of ``POST /threads``."""

    thread_id: str | None = Field(default=None, min_length=1, max_length=200)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResumePayload(BaseModel):
    """How to continue a paused thread (see ``ResumeRequest`` on ``/resume``)."""

    response: str = ""
    decision: dict[str, Any] | None = None
    perform_dangling: bool = True


class RunRequest(BaseModel):
    """Body of ``POST /threads/{id}/runs``: a new turn, or a continuation.

    Exactly one of ``prompt`` (a new user turn) and ``resume`` (the answer to
    the thread's pause: a response to ``ask_user`` or a recorded approval
    decision) is given.
    """

    prompt: str | None = None
    resume: ResumePayload | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _one_of(self) -> RunRequest:
        if (self.prompt is None) == (self.resume is None):
            raise ValueError("give exactly one of 'prompt' or 'resume'")
        return self


class _WsRun(BaseModel):
    type: Literal["run"]
    thread_id: str | None = None
    prompt: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    ref: Any = None


class _WsResume(BaseModel):
    type: Literal["resume"]
    thread_id: str
    response: str = ""
    decision: dict[str, Any] | None = None
    perform_dangling: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    ref: Any = None


class _WsCancel(BaseModel):
    type: Literal["cancel"]
    run_id: str
    ref: Any = None


class _WsSubscribe(BaseModel):
    type: Literal["subscribe"]
    run_id: str
    after: int = Field(default=0, ge=0)
    ref: Any = None


# ----------------------------------------------------------------------
# Wiring
# ----------------------------------------------------------------------

#: ``serialize(event, scoped_thread_id, thread_id) -> JSON-ready payload``.
EventSerializer = Callable[[Any, str | None, str | None], Any]


@dataclass
class RouteContext:
    """What the routes need from the application that mounts them."""

    agent: Any
    registry: RunRegistry
    auth_dep: Any
    api_key: str | None
    principal_key: Callable[[str | None], str]
    scope_thread: Callable[[str, str | None], str | None]
    resolve_metadata: Callable[[Any, str, dict[str, Any]], Awaitable[dict[str, Any]]]
    decision_handler: Callable[..., Any] | None
    #: Whether a ``metadata_resolver`` is configured (a resume then resolves
    #: metadata even when the client sent none, as ``/resume`` does).
    has_resolver: bool
    stream_tokens: bool
    serialize: EventSerializer
    accepts_kwarg: Callable[[Any, str], bool]


class RouteError(Exception):
    """A request refused with an HTTP status; also the WebSocket error frame."""

    def __init__(
        self, status: int, code: str, detail: str, extra: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail
        self.extra = dict(extra or {})

    def body(self) -> dict[str, Any]:
        return {"detail": self.detail, "code": self.code, **self.extra}


def _registry_error(exc: RegistryError) -> RouteError:
    status = 409
    detail = {
        "thread_exists": "Thread already exists",
        "run_exists": "Run id already exists",
        "run_in_progress": "A run is already in progress on this thread",
    }.get(exc.code, str(exc))
    return RouteError(status, exc.code, detail, {"conflict_with": str(exc)})


def _turn_source(
    ctx: RouteContext, principal: str, thread_id: str, prompt: str, metadata: dict[str, Any]
) -> AsyncIterator[dict[str, Any]]:
    """Payloads of a new turn; ``agent.run`` is called when iteration starts."""
    scoped = ctx.scope_thread(principal, thread_id)
    run_kwargs: dict[str, Any] = {"thread_id": scoped, "metadata": metadata}
    if ctx.stream_tokens and ctx.accepts_kwarg(ctx.agent.run, "stream_tokens"):
        run_kwargs["stream_tokens"] = True

    async def source() -> AsyncIterator[dict[str, Any]]:
        events = ctx.agent.run(prompt, **run_kwargs)
        async with _closing(events):
            async for event in events:
                yield ctx.serialize(event, scoped, thread_id)

    return source()


def _closing(iterator: Any) -> contextlib.AbstractAsyncContextManager[Any]:
    """``aclosing`` for async generators; a no-op for other iterators."""
    if hasattr(iterator, "aclose"):
        return contextlib.aclosing(iterator)
    return contextlib.nullcontext(iterator)


async def _probe_resume(
    ctx: RouteContext,
    connection: Any,
    principal: str,
    thread_id: str,
    payload: ResumePayload,
    client_metadata: dict[str, Any],
) -> AsyncIterator[dict[str, Any]]:
    """Start ``agent.resume`` and pull its first event, so that "nothing to
    resume" and "approval still pending" are HTTP statuses, not frames."""
    from locus.core.errors import ApprovalPendingError

    resume_fn = getattr(ctx.agent, "resume", None)
    if resume_fn is None:
        raise RouteError(404, "cannot_resume", "This agent cannot be resumed")
    if payload.decision is not None:
        if ctx.decision_handler is None:
            raise RouteError(400, "decisions_not_accepted", "This server does not accept decisions")
        recorded = ctx.decision_handler(connection, principal, thread_id, dict(payload.decision))
        if inspect.isawaitable(recorded):
            await recorded

    scoped = ctx.scope_thread(principal, thread_id)
    resume_kwargs: dict[str, Any] = {
        "thread_id": scoped,
        "perform_dangling": payload.perform_dangling,
    }
    if client_metadata or ctx.has_resolver:
        resume_kwargs["metadata"] = await ctx.resolve_metadata(
            connection, principal, client_metadata
        )
    events = resume_fn(payload.response, **resume_kwargs)
    first: list[Any] = []
    try:
        first.append(await events.__anext__())
    except StopAsyncIteration:
        pass
    except ApprovalPendingError as pending:
        raise RouteError(
            409,
            "approval_pending",
            "The held approval is still undecided",
            {
                "type": "approval_pending",
                "thread_id": thread_id,
                "interrupt_id": pending.interrupt_id,
                "question": pending.question,
                "metadata": json.loads(json.dumps(pending.metadata, default=str)),
            },
        ) from None
    except RuntimeError as missing:
        if "resume" in str(missing) or "No checkpoint" in str(missing):
            raise RouteError(
                404, "no_paused_run", f"No paused run for thread {thread_id!r}"
            ) from None
        raise

    async def source() -> AsyncIterator[dict[str, Any]]:
        async with _closing(events):
            for event in first:
                yield ctx.serialize(event, scoped, thread_id)
            async for event in events:
                yield ctx.serialize(event, scoped, thread_id)

    return source()


def _on_cancel(ctx: RouteContext, principal: str, thread_id: str) -> Callable[[], Any] | None:
    """``agent.cancel(thread_id=...)`` for agents that have it, else ``None``."""
    cancel = getattr(ctx.agent, "cancel", None)
    if not callable(cancel) or not ctx.accepts_kwarg(cancel, "thread_id"):
        return None
    scoped = ctx.scope_thread(principal, thread_id)
    return lambda: cancel(thread_id=scoped)


async def start_turn(
    ctx: RouteContext,
    connection: Any,
    principal: str,
    thread_id: str,
    body: RunRequest,
    *,
    run_id: str | None = None,
) -> RunRecord:
    """Register and start the run ``body`` describes on ``thread_id``."""
    registry = ctx.registry
    thread = registry.get_thread(principal, thread_id)
    if thread is not None and any(not r.finished for r in registry.runs_for(thread)):
        raise _registry_error(RunInProgressError(thread_id))
    if body.prompt is not None:
        metadata = await ctx.resolve_metadata(connection, principal, body.metadata)
        source = _turn_source(ctx, principal, thread_id, body.prompt, metadata)
        kind: Literal["run", "resume"] = "run"
    else:
        assert body.resume is not None  # noqa: S101 — validated by RunRequest
        source = await _probe_resume(
            ctx, connection, principal, thread_id, body.resume, body.metadata
        )
        kind = "resume"
    try:
        record = registry.new_run(principal, thread_id, kind=kind, run_id=run_id)
    except RegistryError as exc:
        with contextlib.suppress(Exception):
            await source.aclose()  # type: ignore[attr-defined]
        raise _registry_error(exc) from None
    registry.start(record, source, on_cancel=_on_cancel(ctx, principal, thread_id))
    return record


def _parse_after(value: str | None) -> int:
    if value is None or value == "":
        return 0
    try:
        after = int(value)
    except ValueError:
        raise RouteError(400, "bad_last_event_id", "Last-Event-ID must be an integer") from None
    if after < 0:
        raise RouteError(400, "bad_last_event_id", "Last-Event-ID must not be negative")
    return after


async def sse_frames(
    registry: RunRegistry, record: RunRecord, adapter: StreamAdapter, *, after: int = 0
) -> AsyncIterator[str]:
    """The SSE text of ``record`` through ``adapter``, from sequence ``after + 1``.

    The adapter sees every retained event so its state (open messages,
    counters) matches what the client already has; only frames for events
    after ``after`` are sent. The ``id:`` line goes on the last frame of each
    event's group, so a client that drops mid-group resumes from the event
    before it and sees the whole group again. Ends with ``data: [DONE]``.
    """

    def group(frames: list[dict[str, Any]], seq: int) -> list[str]:
        last = len(frames) - 1
        return [
            encode_sse(frame, event_id=str(seq) if index == last else None)
            for index, frame in enumerate(frames)
        ]

    if after == 0:
        for text in group(adapter.start(), 0):
            yield text
    else:
        adapter.start()
    async for seq, payload in registry.events(record, after=0):
        frames = adapter.feed(payload)
        if seq <= after or not frames:
            continue
        for text in group(frames, seq):
            yield text
    for frame in adapter.finish():
        yield encode_sse(frame)
    yield DONE_FRAME


def _ws_principal(ctx: RouteContext, websocket: Any) -> str | None:
    """The socket's principal: the bearer header, or ``token`` in the query."""
    if ctx.api_key is None:
        return "anon"
    header = websocket.headers.get("authorization") or ""
    presented = header.split(" ", 1)[1].strip() if header.lower().startswith("bearer ") else ""
    if not presented:
        presented = websocket.query_params.get("token") or ""
    if not presented or not hmac.compare_digest(presented, ctx.api_key):
        return None
    return ctx.principal_key(presented)


def mount_run_routes(app: FastAPI, ctx: RouteContext) -> None:  # noqa: C901, PLR0915
    """Add the thread, run, live and protocol routes to ``app``."""
    from fastapi import HTTPException, Query
    from fastapi.responses import JSONResponse, StreamingResponse
    from starlette.websockets import WebSocketDisconnect

    registry = ctx.registry
    auth_dep = ctx.auth_dep

    def refused(exc: RouteError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content=exc.body())

    def run_or_404(principal: str, run_id: str) -> RunRecord:
        record = registry.get_run(principal, run_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
        return record

    @app.post("/threads", status_code=201)
    async def create_thread(
        request: ThreadCreateRequest, principal: str = auth_dep
    ) -> dict[str, Any]:
        try:
            thread = registry.create_thread(principal, request.thread_id, request.metadata)
        except RegistryError as exc:
            return refused(_registry_error(exc))  # type: ignore[return-value]
        return thread.to_dict()

    @app.post("/threads/{thread_id}/runs", status_code=202)
    async def create_run(
        thread_id: str,
        request: RunRequest,
        http_request: Request,
        principal: str = auth_dep,
    ) -> Any:
        try:
            record = await start_turn(ctx, http_request, principal, thread_id, request)
        except RouteError as exc:
            return refused(exc)
        return {"run_id": record.run_id, "thread_id": thread_id, "status": record.status}

    @app.get("/runs/{run_id}")
    async def get_run(run_id: str, principal: str = auth_dep) -> dict[str, Any]:
        return run_or_404(principal, run_id).to_dict()

    @app.post("/runs/{run_id}/cancel")
    async def cancel_run(run_id: str, principal: str = auth_dep) -> dict[str, Any]:
        record = run_or_404(principal, run_id)
        accepted = registry.cancel(record)
        return {"run_id": run_id, "status": record.status, "accepted": accepted}

    @app.get("/runs/{run_id}/events")
    async def run_events(
        run_id: str,
        http_request: Request,
        principal: str = auth_dep,
        protocol: str = Query(default=PROTOCOL_LOCUS),
    ) -> Any:
        record = run_or_404(principal, run_id)
        if protocol not in PROTOCOL_NAMES:
            raise HTTPException(
                status_code=400,
                detail=f"unknown protocol {protocol!r}; one of {sorted(PROTOCOL_NAMES)}",
            )
        try:
            after = _parse_after(http_request.headers.get("last-event-id"))
        except RouteError as exc:
            return refused(exc)
        adapter = make_adapter(protocol, thread_id=record.thread_id, run_id=record.run_id)
        return StreamingResponse(
            sse_frames(registry, record, adapter, after=after),
            media_type=adapter.media_type,
            headers=dict(adapter.headers),
        )

    async def protocol_run(
        http_request: Request,
        principal: str,
        protocol: str,
        thread_id: str | None,
        run_id: str | None,
        prompt: str,
        metadata: dict[str, Any],
    ) -> Any:
        if thread_id is None:
            thread_id = registry.create_thread(principal).thread_id
        body = RunRequest(prompt=prompt, metadata=metadata)
        try:
            record = await start_turn(ctx, http_request, principal, thread_id, body, run_id=run_id)
        except RouteError as exc:
            return refused(exc)
        adapter = make_adapter(protocol, thread_id=thread_id, run_id=record.run_id)
        return StreamingResponse(
            sse_frames(registry, record, adapter),
            media_type=adapter.media_type,
            headers=dict(adapter.headers),
        )

    async def json_body(http_request: Request) -> dict[str, Any]:
        try:
            body = await http_request.json()
        except ValueError:
            raise HTTPException(status_code=400, detail="body must be a JSON object") from None
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="body must be a JSON object")
        return body

    @app.post("/ag-ui")
    async def ag_ui(http_request: Request, principal: str = auth_dep) -> Any:
        """Run the agent for an AG-UI ``RunAgentInput``; the response is AG-UI events."""
        body = await json_body(http_request)
        try:
            thread_id, run_id, prompt, metadata = ag_ui_run_input(body)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        return await protocol_run(
            http_request, principal, PROTOCOL_AG_UI, thread_id, run_id, prompt, metadata
        )

    @app.post("/vercel-ai")
    async def vercel_ai(http_request: Request, principal: str = auth_dep) -> Any:
        """Run the agent for a UI message request; the response is a UI message stream."""
        body = await json_body(http_request)
        try:
            thread_id, prompt, metadata = ui_message_input(body)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        return await protocol_run(
            http_request, principal, PROTOCOL_VERCEL_AI, thread_id, None, prompt, metadata
        )

    @app.websocket("/runs/live")
    async def live(websocket: WebSocket) -> None:  # noqa: C901, PLR0915
        principal = _ws_principal(ctx, websocket)
        if principal is None:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        send_lock = asyncio.Lock()
        forwarders: set[asyncio.Task[None]] = set()

        async def send(frame: dict[str, Any]) -> None:
            async with send_lock:
                await websocket.send_text(json.dumps(frame, default=str))

        async def forward(record: RunRecord, after: int, ref: Any) -> None:
            try:
                async for seq, payload in registry.events(record, after=after):
                    await send(
                        {"type": "event", "run_id": record.run_id, "seq": seq, "event": payload}
                    )
                await send(
                    {
                        "type": "run_finished",
                        "run_id": record.run_id,
                        "thread_id": record.thread_id,
                        "status": record.status,
                        "ref": ref,
                    }
                )
            except (WebSocketDisconnect, RuntimeError):
                pass

        def follow(record: RunRecord, after: int, ref: Any) -> None:
            task = asyncio.create_task(forward(record, after, ref))
            forwarders.add(task)
            task.add_done_callback(forwarders.discard)

        async def handle(message: dict[str, Any]) -> None:
            kind = message.get("type")
            ref = message.get("ref")
            if kind == "ping":
                await send({"type": "pong", "ref": ref})
            elif kind == "run":
                run_msg = _WsRun.model_validate(message)
                thread_id = run_msg.thread_id or registry.create_thread(principal).thread_id
                body = RunRequest(prompt=run_msg.prompt, metadata=run_msg.metadata)
                record = await start_turn(ctx, websocket, principal, thread_id, body)
                await send(
                    {
                        "type": "run_started",
                        "run_id": record.run_id,
                        "thread_id": thread_id,
                        "ref": ref,
                    }
                )
                follow(record, 0, ref)
            elif kind == "resume":
                resume_msg = _WsResume.model_validate(message)
                body = RunRequest(
                    resume=ResumePayload(
                        response=resume_msg.response,
                        decision=resume_msg.decision,
                        perform_dangling=resume_msg.perform_dangling,
                    ),
                    metadata=resume_msg.metadata,
                )
                record = await start_turn(ctx, websocket, principal, resume_msg.thread_id, body)
                await send(
                    {
                        "type": "run_started",
                        "run_id": record.run_id,
                        "thread_id": resume_msg.thread_id,
                        "ref": ref,
                    }
                )
                follow(record, 0, ref)
            elif kind == "cancel":
                cancel_msg = _WsCancel.model_validate(message)
                target = registry.get_run(principal, cancel_msg.run_id)
                if target is None:
                    raise RouteError(404, "not_found", f"Run {cancel_msg.run_id!r} not found")
                accepted = registry.cancel(target)
                await send(
                    {
                        "type": "cancelled",
                        "run_id": target.run_id,
                        "status": target.status,
                        "accepted": accepted,
                        "ref": ref,
                    }
                )
            elif kind == "subscribe":
                sub_msg = _WsSubscribe.model_validate(message)
                existing = registry.get_run(principal, sub_msg.run_id)
                if existing is None:
                    raise RouteError(404, "not_found", f"Run {sub_msg.run_id!r} not found")
                follow(existing, sub_msg.after, ref)
            else:
                raise RouteError(400, "unknown_type", f"unknown message type {kind!r}")

        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    message = json.loads(raw)
                    if not isinstance(message, dict):
                        raise RouteError(400, "bad_request", "messages are JSON objects")
                    await handle(message)
                except RouteError as exc:
                    await send({**exc.body(), "type": "error", "ref": None})
                except ValidationError as exc:
                    await send(
                        {
                            "type": "error",
                            "code": "bad_request",
                            "detail": json.loads(exc.json()),
                            "ref": None,
                        }
                    )
                except ValueError as exc:
                    await send(
                        {"type": "error", "code": "bad_request", "detail": str(exc), "ref": None}
                    )
        except WebSocketDisconnect:
            pass
        finally:
            for task in forwarders:
                task.cancel()


__all__ = [
    "DONE_FRAME",
    "ResumePayload",
    "RouteContext",
    "RouteError",
    "RunRequest",
    "ThreadCreateRequest",
    "mount_run_routes",
    "sse_frames",
    "start_turn",
]
