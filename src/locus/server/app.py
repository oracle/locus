# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""FastAPI-based agent server.

Exposes a Locus Agent as HTTP endpoints:
- POST   /invoke         — synchronous invocation, returns final result
- POST   /stream         — SSE streaming of agent events
- GET    /threads/{tid}  — load a thread's persisted state (requires checkpointer)
- DELETE /threads/{tid}  — drop a thread's persisted state (requires checkpointer)
- GET    /health         — health check

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

import hmac
import ipaddress
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field


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
    to namespace threads, not enough to be a secret-recovery oracle for
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
    ) -> None:
        self.agent = agent
        self._title = title
        self._description = description
        # Prefer the explicit arg; fall back to the environment so that
        # deployments don't have to thread the secret through code.
        self._api_key = api_key or os.environ.get("LOCUS_SERVER_API_KEY") or None
        self._allow_unauthenticated = allow_unauthenticated
        self._app = None

    @property
    def app(self) -> Any:
        """Get or create the FastAPI application."""
        if self._app is None:
            self._app = self._create_app()
        return self._app

    def _resolve_docs_enabled(self) -> bool:
        """Expose /docs, /redoc, /openapi.json only when debug is on.

        FastAPI turns these on by default. On an unauthenticated or partly
        authenticated deployment they are a schema oracle + "try it" UI
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
        app = FastAPI(
            title=self._title,
            description=self._description,
            docs_url="/docs" if debug_docs else None,
            redoc_url="/redoc" if debug_docs else None,
            openapi_url="/openapi.json" if debug_docs else None,
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

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        @app.post("/invoke", response_model=InvokeResponse)
        async def invoke(
            request: InvokeRequest,
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

            t0 = time.perf_counter()
            async for event in agent.run(
                request.prompt,
                thread_id=scope_thread(principal, request.thread_id),
                metadata=request.metadata,
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
            principal: str = auth_dep,
        ) -> StreamingResponse:
            from locus.core.events import (
                TerminateEvent,
                ThinkEvent,
                ToolCompleteEvent,
                ToolStartEvent,
            )

            scoped_id = scope_thread(principal, request.thread_id)

            async def event_generator() -> AsyncIterator[str]:
                correlation_id: str | None = None
                try:
                    async for event in agent.run(
                        request.prompt,
                        thread_id=scoped_id,
                        metadata=request.metadata,
                    ):
                        # Each branch builds a JSON-serialisable payload.
                        # Mixed value types (str, dict, list, None) so the
                        # dict is annotated as ``dict[str, Any]``.
                        data: dict[str, Any]
                        if isinstance(event, ThinkEvent):
                            data = {"type": "think", "content": event.reasoning or ""}
                        elif isinstance(event, ToolStartEvent):
                            data = {
                                "type": "tool_start",
                                "tool": event.tool_name,
                                # arguments are echoed back to the client
                                # exactly as the model produced them; if
                                # your deployment considers tool args
                                # sensitive, wrap the agent to redact.
                                "arguments": event.arguments,
                            }
                        elif isinstance(event, ToolCompleteEvent):
                            data = {
                                "type": "tool_complete",
                                "tool": event.tool_name,
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
                            data = {"type": event.event_type, "data": str(event)}

                        yield f"data: {json.dumps(data)}\n\n"
                except Exception:  # noqa: BLE001 — all agent errors get sanitized
                    correlation_id = uuid.uuid4().hex
                    _logger.exception("agent stream error (correlation_id=%s)", correlation_id)
                    # Emit a generic error event so unauthenticated peers
                    # don't get str(exc) (CWE-209). Details live in logs
                    # keyed to ``correlation_id``.
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

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
            )

        @app.get("/threads/{thread_id}")
        async def get_thread(
            thread_id: str,
            principal: str = auth_dep,
        ) -> dict[str, Any]:
            """Return the persisted thread state, scoped to the caller principal.

            Returns 404 if no checkpointer is configured or the thread isn't
            found. The principal-scoping prevents thread enumeration across
            API keys when an upstream proxy multiplexes clients.
            """
            from fastapi import HTTPException

            checkpointer = agent.config.checkpointer
            if checkpointer is None:
                raise HTTPException(
                    status_code=404,
                    detail="No checkpointer configured on this AgentServer",
                )
            scoped_id = scope_thread(principal, thread_id)
            state = await checkpointer.load(scoped_id)
            if state is None:
                raise HTTPException(status_code=404, detail=f"Thread {thread_id!r} not found")
            # Hand back the public Pydantic projection. AgentState is already
            # JSON-serializable; the principal scope is intentionally hidden
            # from the response (callers see their unprefixed id).
            return {
                "thread_id": thread_id,
                "iteration": state.iteration,
                "messages": [m.model_dump(mode="json") for m in state.messages],
                "tool_executions": [te.model_dump(mode="json") for te in state.tool_executions],
                "metadata": state.metadata,
            }

        @app.delete("/threads/{thread_id}")
        async def delete_thread(
            thread_id: str,
            principal: str = auth_dep,
        ) -> dict[str, Any]:
            """Drop a thread's persisted state. 404 when no checkpointer.

            Idempotent: deleting a non-existent thread returns ``deleted=False``
            with a 200, matching ``BaseCheckpointer.delete()``'s contract.
            """
            from fastapi import HTTPException

            checkpointer = agent.config.checkpointer
            if checkpointer is None:
                raise HTTPException(
                    status_code=404,
                    detail="No checkpointer configured on this AgentServer",
                )
            scoped_id = scope_thread(principal, thread_id)
            deleted = await checkpointer.delete(scoped_id)
            return {"thread_id": thread_id, "deleted": bool(deleted)}

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
