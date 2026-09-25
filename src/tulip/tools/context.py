# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Tool execution context - 100% Pydantic."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tulip.core.events import ToolProgressEvent


class ToolContext(BaseModel):
    """
    Context passed to tools during execution.

    Provides access to agent state, metadata, and utilities.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Identifiers
    tool_call_id: str = Field(..., description="Unique ID of this tool call")
    tool_name: str = Field(..., description="Name of the tool being called")

    # Agent context
    agent_id: str | None = Field(default=None, description="ID of the calling agent")
    run_id: str = Field(..., description="ID of the current agent run")
    iteration: int = Field(..., description="Current iteration number")

    # State access (read-only view)
    state: Any = Field(default=None, description="Current agent state")

    # User-provided metadata
    invocation_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata passed at invocation time",
    )

    # Run metadata that is never persisted (``mcp_headers``: per-run MCP
    # credentials). Excluded from ``repr`` and serialization.
    ephemeral_metadata: dict[str, Any] = Field(
        default_factory=dict,
        exclude=True,
        repr=False,
        description="Per-run metadata carried in memory only (e.g. mcp_headers)",
    )

    # Tool-specific config
    tool_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool-specific configuration",
    )

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get a metadata value (ephemeral metadata included)."""
        if key in self.invocation_metadata:
            return self.invocation_metadata[key]
        return self.ephemeral_metadata.get(key, default)

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a tool config value."""
        return self.tool_config.get(key, default)

    @property
    def messages(self) -> list[Any]:
        """Get conversation messages (if state available)."""
        if self.state is None:
            return []
        return list(self.state.messages)

    @property
    def confidence(self) -> float:
        """Get current confidence score (if state available)."""
        if self.state is None:
            return 0.0
        # ``self.state`` is typed as ``Any`` upstream — narrow the return
        # to ``float`` to satisfy strict mypy.
        return float(self.state.confidence)


# =============================================================================
# The tool call in flight
# =============================================================================
#
# A tool body — and anything it calls, like an MCP client — sometimes needs to
# know *which* call it is serving: to attribute a progress notification to the
# right ``tool_call_id``, or to read the run's metadata (a per-user token) when
# its signature has no ``ctx`` parameter. The executor binds the context here
# for the duration of each call. Context variables are task-local, so
# concurrent calls in one batch each see their own.

_current_tool_context: ContextVar[ToolContext | None] = ContextVar(
    "tulip_current_tool_context", default=None
)

#: Where progress for the current run goes. Set by the agent runtime around a
#: tool batch that contains a progress-emitting tool; ``None`` everywhere else,
#: which makes :func:`report_progress` a no-op.
_progress_sink: ContextVar[Callable[[ToolProgressEvent], None] | None] = ContextVar(
    "tulip_progress_sink", default=None
)


def current_tool_context() -> ToolContext | None:
    """The :class:`ToolContext` of the tool call running in this task, if any."""
    return _current_tool_context.get()


@contextmanager
def bind_tool_context(ctx: ToolContext | None) -> Iterator[None]:
    """Make ``ctx`` the current tool context for the enclosed block."""
    token = _current_tool_context.set(ctx)
    try:
        yield
    finally:
        _current_tool_context.reset(token)


ProgressReporter = Callable[[float, float | None, str | None], None]


def progress_reporter() -> ProgressReporter | None:
    """Capture the current call's progress channel for use from elsewhere.

    :func:`report_progress` reads task-local state, so it only works from the
    task running the tool. Protocol callbacks — an MCP session's
    ``notifications/progress`` handler — run on the connection's own task.
    Capture a reporter in the tool's task and hand it to the callback.

    Returns:
        A ``(progress, total, message)`` callable, or ``None`` when nobody is
        listening (no progress sink, or no tool call in flight).
    """
    sink = _progress_sink.get()
    ctx = _current_tool_context.get()
    if sink is None or ctx is None:
        return None
    call_id, name = ctx.tool_call_id, ctx.tool_name

    def _report(progress: float, total: float | None = None, message: str | None = None) -> None:
        sink(
            ToolProgressEvent(
                tool_name=name,
                tool_call_id=call_id,
                progress=float(progress),
                total=None if total is None else float(total),
                message=message,
            )
        )

    return _report


def report_progress(
    progress: float,
    total: float | None = None,
    message: str | None = None,
) -> bool:
    """Report progress for the tool call running in this task.

    Surfaces as a :class:`~tulip.core.events.ToolProgressEvent` on the agent's
    event stream — live, for a tool declared with ``emits_progress=True``.

    Args:
        progress: Units done so far.
        total: Units in all, when known.
        message: A short human-readable status.

    Returns:
        Whether the report was delivered; ``False`` when nothing is listening.
    """
    reporter = progress_reporter()
    if reporter is None:
        return False
    reporter(progress, total, message)
    return True


@contextmanager
def progress_sink(sink: Callable[[ToolProgressEvent], None] | None) -> Iterator[None]:
    """Route :func:`report_progress` calls in the enclosed block to ``sink``."""
    token = _progress_sink.set(sink)
    try:
        yield
    finally:
        _progress_sink.reset(token)
