# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""FastAPI-based agent server.

Exposes a Locus Agent as HTTP endpoints:
- POST   /invoke         — synchronous invocation, returns final result
- POST   /stream         — SSE streaming of agent events (token deltas included)
- POST   /resume         — continue a paused thread (SSE), strictly by thread id
- GET    /threads/{tid}  — a thread's persisted state (checkpointer) and its runs
- DELETE /threads/{tid}  — drop a thread's persisted state and forget its runs
- GET    /health         — health check

Threads and runs (see :mod:`locus.server.routes`):
- POST   /threads               — create a thread
- POST   /threads/{tid}/runs    — start a turn (or continue a pause) in the background
- GET    /runs/{rid}            — a run's status and outcome
- GET    /runs/{rid}/events     — the run's events as SSE, resumable with Last-Event-ID;
                                  ``?protocol=ag-ui`` / ``?protocol=vercel-ai``
- POST   /runs/{rid}/cancel     — stop a run
- WS     /runs/live             — turns, approval answers, cancellation and events on one socket
- POST   /ag-ui, /vercel-ai     — the two UI wire protocols' request endpoints

Security model
--------------
When ``api_key`` (constructor arg) or the ``LOCUS_SERVER_API_KEY``
environment variable is set, every route other than ``/health`` requires
an ``Authorization: Bearer <key>`` header. The API key is also used to
derive the per-principal checkpoint namespace, so two clients that share
one agent instance cannot resume each other's threads.

If no API key is configured and the server is bound to anything other
than ``127.0.0.1`` / ``::1`` / ``localhost``, the server refuses to
start — an unauthenticated network-reachable agent is remote code
execution waiting to happen. Disable this check only via the
``allow_unauthenticated`` constructor arg (documented footgun; for
local development or when an upstream proxy handles auth).
"""

from __future__ import annotations

import contextlib
import hmac
import inspect
import ipaddress
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any

from pydantic import BaseModel, Field

from locus.server.routes import RouteContext, mount_run_routes
from locus.server.runs import RunRegistry


# Route signatures are resolved by FastAPI against this module's globals
# (``from __future__ import annotations`` makes them strings), so ``Request``
# must be importable here — without making fastapi a hard dependency.
try:
    from starlette.requests import Request
except ImportError:  # pragma: no cover — server extra not installed
    Request = Any  # type: ignore[misc,assignment]

_logger = logging.getLogger(__name__)

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _is_loopback(host: str) -> bool:
    """Return True if ``host`` resolves to a loopback address."""
    if host in _LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _principal_key(api_key: str | None) -> str:
    """Derive a stable, non-reversible principal id from the presented key.

    Only 12 hex chars of a SHA-256 digest land in checkpoint keys — enough
    to namespace threads, not enough to be a secret-recovery channel for
    anyone who gains read access to the checkpointer.
    """
    if not api_key:
        return "anon"
    import hashlib

    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]


class InvokeRequest(BaseModel):
    """Request body for /invoke endpoint."""

    prompt: str
    thread_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResumeRequest(BaseModel):
    """Request body for /resume: continue ONE paused thread.

    ``thread_id`` is required — a resume is never matched to "whichever run is
    paused". ``response`` answers an ``ask_user`` question. ``decision`` is an
    opaque payload handed to the server's ``decision_handler`` (typically
    ``{"approval_id": ..., "verdict": "approved", "arguments": {...}}``) so the
    host can record an approval before the held call is re-invoked.
    """

    thread_id: str
    response: str = ""
    decision: dict[str, Any] | None = None
    #: Re-invoke the held (gated) call so an approval actually performs it.
    #: ``ask_user`` pauses are unaffected and fold ``response`` as before.
    perform_dangling: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


#: ``metadata_resolver(request, principal) -> trusted metadata``. Whatever it
#: returns is merged OVER the client's ``metadata``, so a client can never
#: spoof a key the server vouches for (the user id a tool bills, say).
MetadataResolver = Callable[[Any, str], "Mapping[str, Any] | Awaitable[Mapping[str, Any]]"]

#: ``decision_handler(request, principal, thread_id, decision)`` records a
#: client-supplied decision (e.g. in an approval store) before /resume
#: continues the thread. It MUST authorise the decision against the caller —
#: the thread is already principal-scoped, the approval id inside the payload
#: is not.
DecisionHandler = Callable[[Any, str, str, dict[str, Any]], "Awaitable[None] | None"]


def _event_payload(event: Any, *, scoped_thread_id: str | None, thread_id: str | None) -> Any:
    """JSON-ready SSE payload for one event.

    The four historical shapes (``think``, ``tool_start``, ``tool_complete``,
    ``done``) are kept key-for-key so existing clients do not break; every
    other event is its full ``model_dump(mode="json")`` under ``type`` —
    never a Python ``repr``, which is what interrupts used to be sent as.
    """
    from locus.core.events import (
        TerminateEvent,
        ThinkEvent,
        ToolCompleteEvent,
        ToolStartEvent,
    )

    data: dict[str, Any]
    if isinstance(event, ThinkEvent):
        data = {"type": "think", "content": event.reasoning or ""}
    elif isinstance(event, ToolStartEvent):
        data = {
            "type": "tool_start",
            "tool": event.tool_name,
            "tool_call_id": event.tool_call_id,
            # arguments are echoed back to the client exactly as the model
            # produced them; if your deployment considers tool args
            # sensitive, wrap the agent to redact.
            "arguments": event.arguments,
        }
    elif isinstance(event, ToolCompleteEvent):
        data = {
            "type": "tool_complete",
            "tool": event.tool_name,
            "tool_call_id": event.tool_call_id,
            "result": event.result,
            "error": event.error,
        }
    elif isinstance(event, TerminateEvent):
        data = {
            "type": "done",
            "message": event.final_message or "",
            "reason": event.reason,
        }
    else:
        dump = getattr(event, "model_dump", None)
        body: dict[str, Any] = dump(mode="json") if callable(dump) else {"data": str(event)}
        event_type = body.pop("event_type", None) or getattr(event, "event_type", "event")
        body.pop("timestamp", None)
        # Never leak the principal-scoped storage key; clients know their
        # thread by the id they sent.
        if scoped_thread_id is not None and body.get("thread_id") == scoped_thread_id:
            body["thread_id"] = thread_id
        data = {"type": event_type, **body}
    return data


def _accepts_kwarg(fn: Any, name: str) -> bool:
    """Whether ``fn`` can be called with keyword ``name``."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


class InvokeResponse(BaseModel):
    """Response body for /invoke endpoint."""

    message: str
    success: bool
    stop_reason: str
    iterations: int = 0
    tool_calls: int = 0
    duration_ms: float = 0.0


# Stop reasons that count as a successful agent run. ``confidence_met``
# and ``terminal_tool`` are deliberate, intended stops (the agent finished
# because grounding cleared or a terminal tool fired); ``complete`` is the
# normal model-driven completion. ``max_iterations``, ``tool_loop`` and
# ``error`` indicate the run hit a guard or failed outright — those map
# to ``success=False`` so callers can branch without parsing stop_reason.
_INVOKE_SUCCESS_REASONS = frozenset({"complete", "confidence_met", "terminal_tool"})


def _invoke_success(stop_reason: str) -> bool:
    """Return whether a terminal stop_reason should be reported as success.

    Exposed as a module-level helper so the mapping is unit-testable and
    consumers can mirror the same semantics in their own code.
    """
    return stop_reason in _INVOKE_SUCCESS_REASONS


class AgentServer:
    """Wrap a Locus Agent as a FastAPI application.

    Example:
        >>> from locus.agent import Agent, AgentConfig
        >>> from locus.server import AgentServer
        >>>
        >>> agent = Agent(config=AgentConfig(system_prompt="Hello", model=model))
        >>> server = AgentServer(agent=agent, api_key="secret")
        >>> server.run(host="127.0.0.1", port=8000)
    """

    def __init__(
        self,
        agent: Any,
        title: str = "Locus Agent Server",
        description: str = "HTTP API for a Locus AI Agent",
        api_key: str | None = None,
        allow_unauthenticated: bool = False,
        *,
        stream_tokens: bool = True,
        metadata_resolver: MetadataResolver | None = None,
        decision_handler: DecisionHandler | None = None,
        run_registry: RunRegistry | None = None,
    ) -> None:
        """Wrap ``agent``.

        Args:
            stream_tokens: Stream model token deltas on ``/stream``,
                ``/resume`` and background runs (as ``{"type": "model_chunk",
                ...}`` events) when the agent supports it.
            run_registry: The in-memory threads-and-runs registry behind
                ``/threads``, ``/runs`` and ``/runs/live`` (see
                :class:`~locus.server.runs.RunRegistry` for its bounds: event
                history per run, runs kept, TTL). A default one is created.
            metadata_resolver: Server-side source of trusted run metadata,
                called per request with ``(request, principal)``. Its keys
                override the client's ``metadata`` so a client cannot spoof
                them. Without it, client metadata is passed through as before.
            decision_handler: Records the ``decision`` payload of a
                ``/resume`` request (see :data:`DecisionHandler`). Without it,
                ``/resume`` rejects requests that carry a decision.
        """
        self.agent = agent
        self._stream_tokens = stream_tokens
        self._metadata_resolver = metadata_resolver
        self._decision_handler = decision_handler
        self._title = title
        self._description = description
        # Prefer the explicit arg; fall back to the environment so that
        # deployments don't have to thread the secret through code.
        self._api_key = api_key or os.environ.get("LOCUS_SERVER_API_KEY") or None
        self._allow_unauthenticated = allow_unauthenticated
        self._registry = run_registry or RunRegistry()
        self._app = None

    @property
    def app(self) -> Any:
        """Get or create the FastAPI application."""
        if self._app is None:
            self._app = self._create_app()
        return self._app

    @property
    def runs(self) -> RunRegistry:
        """The registry of threads and background runs this server drives."""
        return self._registry

    def _resolve_docs_enabled(self) -> bool:
        """Expose /docs, /redoc, /openapi.json only when debug is on.

        FastAPI turns these on by default. On an unauthenticated or partly
        authenticated deployment they are a schema-disclosure surface + "try it" UI
        (CWE-1295 / CWE-200), so we flip them off unless the operator is
        running in an explicit development configuration.
        """
        try:
            from locus.core.config import get_settings

            return bool(get_settings().debug)
        except Exception:  # noqa: BLE001 — settings failure must not leak docs
            return False

    def _require_auth(self) -> Any:
        """Build the FastAPI dependency that enforces the API key."""
        from fastapi import Header, HTTPException, status

        expected = self._api_key

        async def dependency(
            authorization: str | None = Header(default=None),
        ) -> str:
            if expected is None:
                # _create_app() guarantees we never reach here without
                # api_key configured or allow_unauthenticated=True; but
                # we defend in depth in case someone instantiates the
                # dependency directly.
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
            return _principal_key(presented)

        return dependency

    def _scoped_thread_id(self, principal: str, thread_id: str | None) -> str | None:
        """Prefix ``thread_id`` with the caller principal.

        Prevents one authenticated client from resuming another client's
        conversation by guessing / observing a thread id (CWE-639).
        """
        if thread_id is None:
            return None
        return f"{principal}:{thread_id}"

    def _create_app(self) -> Any:
        """Create the FastAPI application with routes."""
        try:
            from fastapi import Depends, FastAPI
            from fastapi.responses import StreamingResponse
        except ImportError as e:
            msg = "FastAPI is required for AgentServer. Install with: pip install fastapi uvicorn"
            raise ImportError(msg) from e

        if self._api_key is None and not self._allow_unauthenticated:
            # Force the operator to make an explicit choice. Without this
            # check, the historical default was an unauthenticated 0.0.0.0
            # listener driving arbitrary LLM / tool execution (CWE-306).
            _logger.warning(
                "AgentServer: no api_key configured; will require "
                "loopback-only binding. Set LOCUS_SERVER_API_KEY or pass "
                "allow_unauthenticated=True to override."
            )

        debug_docs = self._resolve_docs_enabled()
        registry = self._registry

        @contextlib.asynccontextmanager
        async def lifespan(_app: Any) -> AsyncIterator[None]:
            # Background runs outlive the request that started them; stop
            # the ones still going when the application shuts down.
            try:
                yield
            finally:
                await registry.aclose()

        app = FastAPI(
            title=self._title,
            description=self._description,
            docs_url="/docs" if debug_docs else None,
            redoc_url="/redoc" if debug_docs else None,
            openapi_url="/openapi.json" if debug_docs else None,
            lifespan=lifespan,
        )
        agent = self.agent
        scope_thread = self._scoped_thread_id

        if self._api_key is not None:
            auth_dep = Depends(self._require_auth())
        else:
            # Loopback-bound server with allow_unauthenticated=True or
            # the explicit warning path above: dependency returns a
            # fixed "anon" principal so every caller shares one
            # namespace, which matches the previous behaviour.
            async def _anon() -> str:
                return "anon"

            auth_dep = Depends(_anon)

        resolver = self._metadata_resolver
        decision_handler = self._decision_handler
        want_tokens = self._stream_tokens

        async def resolve_metadata(
            http_request: Request, principal: str, client: dict[str, Any]
        ) -> dict[str, Any]:
            """Client metadata with the server's trusted keys laid over it."""
            if resolver is None:
                return dict(client)
            trusted = resolver(http_request, principal)
            if inspect.isawaitable(trusted):
                trusted = await trusted
            return {**client, **dict(trusted or {})}

        def sse(payload: Any) -> str:
            return f"data: {json.dumps(payload, default=str)}\n\n"

        def error_frame() -> str:
            correlation_id = uuid.uuid4().hex
            _logger.exception("agent stream error (correlation_id=%s)", correlation_id)
            # Emit a generic error event so unauthenticated peers don't get
            # str(exc) (CWE-209). Details live in logs keyed to the id.
            return sse(
                {
                    "type": "error",
                    "error": "internal error",
                    "correlation_id": correlation_id,
                }
            )

        async def sse_stream(
            events: AsyncIterator[Any],
            *,
            scoped_id: str | None,
            thread_id: str | None,
            first: list[Any] | None = None,
        ) -> AsyncIterator[str]:
            try:
                for event in first or []:
                    yield sse(
                        _event_payload(event, scoped_thread_id=scoped_id, thread_id=thread_id)
                    )
                async for event in events:
                    yield sse(
                        _event_payload(event, scoped_thread_id=scoped_id, thread_id=thread_id)
                    )
            except Exception:  # noqa: BLE001 — all agent errors get sanitized
                yield error_frame()
            finally:
                yield "data: [DONE]\n\n"

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        @app.post("/invoke", response_model=InvokeResponse)
        async def invoke(
            request: InvokeRequest,
            http_request: Request,
            principal: str = auth_dep,
        ) -> InvokeResponse:
            # Native async path: iterating agent.run() on the event loop
            # avoids the run_sync/future.result() trap that would block
            # uvicorn for the duration of the agent run (CWE-1088).
            from locus.core.events import TerminateEvent, ToolCompleteEvent

            final = ""
            iterations = 0
            tool_calls = 0
            stop_reason = "complete"

            metadata = await resolve_metadata(http_request, principal, request.metadata)
            t0 = time.perf_counter()
            async for event in agent.run(
                request.prompt,
                thread_id=scope_thread(principal, request.thread_id),
                metadata=metadata,
            ):
                if isinstance(event, TerminateEvent):
                    final = event.final_message or final
                    stop_reason = event.reason or stop_reason
                elif isinstance(event, ToolCompleteEvent):
                    tool_calls += 1
                iterations += 1
            duration_ms = (time.perf_counter() - t0) * 1000.0

            return InvokeResponse(
                message=final,
                success=_invoke_success(stop_reason),
                stop_reason=stop_reason,
                iterations=iterations,
                tool_calls=tool_calls,
                duration_ms=duration_ms,
            )

        @app.post("/stream")
        async def stream(
            request: InvokeRequest,
            http_request: Request,
            principal: str = auth_dep,
        ) -> StreamingResponse:
            scoped_id = scope_thread(principal, request.thread_id)
            metadata = await resolve_metadata(http_request, principal, request.metadata)
            run_kwargs: dict[str, Any] = {"thread_id": scoped_id, "metadata": metadata}
            # Token deltas: only for runnables that take the flag (a
            # GraphRunnable, say, does not).
            if want_tokens and _accepts_kwarg(agent.run, "stream_tokens"):
                run_kwargs["stream_tokens"] = True

            async def event_generator() -> AsyncIterator[str]:
                try:
                    events = agent.run(request.prompt, **run_kwargs)
                except Exception:  # noqa: BLE001 — sanitized like any stream error
                    yield error_frame()
                    yield "data: [DONE]\n\n"
                    return
                async for frame in sse_stream(
                    events, scoped_id=scoped_id, thread_id=request.thread_id
                ):
                    yield frame

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
            )

        @app.post("/resume", response_model=None)
        async def resume(
            request: ResumeRequest,
            http_request: Request,
            principal: str = auth_dep,
        ) -> Any:
            """Continue a paused thread, strictly by ``thread_id`` (SSE).

            409 when the held approval is still undecided (the thread stays
            paused), 404 when the caller has no paused thread by that id.
            """
            from fastapi import HTTPException
            from fastapi.responses import JSONResponse

            from locus.core.errors import ApprovalPendingError

            resume_fn = getattr(agent, "resume", None)
            if resume_fn is None:
                raise HTTPException(status_code=404, detail="This agent cannot be resumed")

            scoped_id = scope_thread(principal, request.thread_id)
            if request.decision is not None:
                if decision_handler is None:
                    raise HTTPException(
                        status_code=400,
                        detail="This server does not accept decisions on /resume",
                    )
                recorded = decision_handler(
                    http_request, principal, request.thread_id, dict(request.decision)
                )
                if inspect.isawaitable(recorded):
                    await recorded

            metadata: dict[str, Any] | None = None
            if request.metadata or resolver is not None:
                metadata = await resolve_metadata(http_request, principal, request.metadata)
            resume_kwargs: dict[str, Any] = {
                "thread_id": scoped_id,
                "perform_dangling": request.perform_dangling,
            }
            if metadata is not None:
                resume_kwargs["metadata"] = metadata

            events = resume_fn(request.response, **resume_kwargs)
            # Pull the first event before answering, so "nothing to resume" and
            # "still pending" become real HTTP statuses instead of an error
            # frame inside a 200 stream.
            first: list[Any] = []
            try:
                first.append(await events.__anext__())
            except StopAsyncIteration:
                pass
            except ApprovalPendingError as pending:
                return JSONResponse(
                    status_code=409,
                    content={
                        "type": "approval_pending",
                        "thread_id": request.thread_id,
                        "interrupt_id": pending.interrupt_id,
                        "question": pending.question,
                        "metadata": json.loads(json.dumps(pending.metadata, default=str)),
                    },
                )
            except RuntimeError as missing:
                if "resume" in str(missing) or "No checkpoint" in str(missing):
                    raise HTTPException(
                        status_code=404,
                        detail=f"No paused run for thread {request.thread_id!r}",
                    ) from None
                raise

            return StreamingResponse(
                sse_stream(
                    events,
                    scoped_id=scoped_id,
                    thread_id=request.thread_id,
                    first=first,
                ),
                media_type="text/event-stream",
            )

        @app.get("/threads/{thread_id}")
        async def get_thread(
            thread_id: str,
            principal: str = auth_dep,
        ) -> dict[str, Any]:
            """Return the thread's state, scoped to the caller principal.

            The transcript comes from the agent's checkpointer when one is
            configured; the thread's runs come from the server's registry
            (``source`` says which answered). 404 when neither knows the
            thread. The principal-scoping prevents thread enumeration across
            API keys when an upstream proxy multiplexes clients.
            """
            from fastapi import HTTPException

            checkpointer = getattr(getattr(agent, "config", None), "checkpointer", None)
            scoped_id = scope_thread(principal, thread_id)
            known = registry.get_thread(principal, thread_id)
            state = await checkpointer.load(scoped_id) if checkpointer is not None else None
            if state is None and known is None:
                detail = f"Thread {thread_id!r} not found"
                if checkpointer is None:
                    detail += (
                        " (no checkpointer configured on this AgentServer; only threads "
                        "created or run through it are known)"
                    )
                raise HTTPException(status_code=404, detail=detail)
            # Hand back the public Pydantic projection. AgentState is already
            # JSON-serializable; the principal scope is intentionally hidden
            # from the response (callers see their unprefixed id).
            metadata: Any = state.metadata if state is not None else {}
            if state is None and known is not None:
                metadata = known.metadata
            return {
                "thread_id": thread_id,
                "iteration": state.iteration if state is not None else 0,
                "messages": [m.model_dump(mode="json") for m in state.messages] if state else [],
                "tool_executions": (
                    [te.model_dump(mode="json") for te in state.tool_executions] if state else []
                ),
                "metadata": metadata,
                "source": "checkpointer" if state is not None else "registry",
                "runs": [r.to_dict() for r in registry.runs_for(known)] if known else [],
            }

        @app.delete("/threads/{thread_id}")
        async def delete_thread(
            thread_id: str,
            principal: str = auth_dep,
        ) -> dict[str, Any]:
            """Drop a thread's persisted state and forget its finished runs.

            Idempotent: deleting a non-existent thread returns ``deleted=False``
            with a 200, matching ``BaseCheckpointer.delete()``'s contract. 404
            when no checkpointer is configured and the registry does not know
            the thread either.
            """
            from fastapi import HTTPException

            checkpointer = getattr(getattr(agent, "config", None), "checkpointer", None)
            dropped = registry.drop_thread(principal, thread_id)
            if checkpointer is None:
                if not dropped:
                    raise HTTPException(
                        status_code=404,
                        detail="No checkpointer configured on this AgentServer",
                    )
                return {"thread_id": thread_id, "deleted": True}
            scoped_id = scope_thread(principal, thread_id)
            deleted = await checkpointer.delete(scoped_id)
            return {"thread_id": thread_id, "deleted": bool(deleted) or dropped}

        mount_run_routes(
            app,
            RouteContext(
                agent=agent,
                registry=registry,
                auth_dep=auth_dep,
                api_key=self._api_key,
                principal_key=_principal_key,
                scope_thread=scope_thread,
                resolve_metadata=resolve_metadata,
                decision_handler=decision_handler,
                has_resolver=resolver is not None,
                stream_tokens=want_tokens,
                serialize=lambda event, scoped, tid: _event_payload(
                    event, scoped_thread_id=scoped, thread_id=tid
                ),
                accepts_kwarg=_accepts_kwarg,
            ),
        )

        return app

    def run(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        **kwargs: Any,
    ) -> None:
        """Run the server with uvicorn.

        Args:
            host: Bind address. Defaults to loopback — using a
                non-loopback host requires either ``api_key`` to be set
                or ``allow_unauthenticated=True`` on this server.
            port: Bind port.
            **kwargs: Additional uvicorn.run() arguments.
        """
        if self._api_key is None and not self._allow_unauthenticated and not _is_loopback(host):
            msg = (
                f"Refusing to bind AgentServer to {host!r} without an API "
                "key. Set LOCUS_SERVER_API_KEY, pass api_key=... to "
                "AgentServer, or pass allow_unauthenticated=True if an "
                "upstream proxy terminates auth."
            )
            raise RuntimeError(msg)

        try:
            import uvicorn
        except ImportError as e:
            msg = "uvicorn is required for AgentServer.run(). Install with: pip install uvicorn"
            raise ImportError(msg) from e

        uvicorn.run(self.app, host=host, port=port, **kwargs)
