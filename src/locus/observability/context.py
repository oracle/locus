# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Context-local correlation id for opt-in telemetry.

Every emission point in the SDK reads :func:`current_run_id` and
no-ops when it returns ``None``. That gives us two properties at once:

* **Telemetry is opt-in.** Users who never call :func:`run_context`
  (or pass ``run_id`` to :meth:`Router.dispatch`) pay zero cost — no
  bus instantiated, no events constructed, no imports beyond this
  module.
* **Telemetry composes through the call tree.** Once a ``run_id`` is
  set on the current asyncio context, every nested call — orchestrators
  invoking specialists, pipelines spawning agents, agents firing tool
  hooks — sees the same id and tags every event with it.

Two ways to enter a run context:

* :func:`run_context` — async context manager with explicit
  ``__aenter__`` / ``__aexit__`` semantics. Preferred for code paths
  that want to control the close (and, e.g., trigger
  :meth:`EventBus.close_stream`).
* :meth:`Router.dispatch` — sets the contextvar implicitly for the
  duration of the dispatch, generates a fresh ``uuid4`` if none was
  provided.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from uuid import uuid4


# ``ContextVar`` defaults to ``None`` so non-telemetry callers see the
# absence-of-correlation-id state immediately. ``contextvars.copy_context``
# in the asyncio event loop preserves the value across ``await`` and
# ``asyncio.gather``; ``asyncio.to_thread`` also propagates it on
# Python 3.7+. Subprocess boundaries do NOT — the workbench's bootstrap
# stamps ``LOCUS_WORKBENCH_RUN_ID`` into the environment for that case.
_run_id_var: ContextVar[str | None] = ContextVar("locus_run_id", default=None)


# Loop reference recorded inside ``run_context`` so worker threads
# (the @tool decorator's executor, ``asyncio.to_thread`` callers) can
# find the right loop to hop publishes back to via
# ``run_coroutine_threadsafe``. Stored as a contextvar (not a global)
# so it survives ``contextvars.copy_context()`` propagation into the
# worker thread but doesn't leak between unrelated dispatches.
import asyncio as _asyncio  # noqa: E402, PLC0415


_owner_loop_var: ContextVar[_asyncio.AbstractEventLoop | None] = ContextVar(
    "locus_owner_loop", default=None
)


def current_owner_loop() -> _asyncio.AbstractEventLoop | None:
    """Return the asyncio loop bound by ``run_context``, or ``None``.

    ``emit_sync`` reads this when called from a worker thread so it
    can hop publishes back to the right loop. Bootstrapping the loop
    at ``run_context`` entry costs nothing for callers who don't
    publish — it's a single ``ContextVar.set`` per dispatch.
    """
    return _owner_loop_var.get()


def current_run_id() -> str | None:
    """Return the run_id active on the current context, or ``None``.

    Every emission helper in :mod:`locus.observability.emit` calls this
    and bails when it returns ``None``. SDK users who never enter a
    run context pay nothing.
    """
    return _run_id_var.get()


def set_run_id(run_id: str | None) -> object:
    """Set the run_id on the current context, returning the reset token.

    Lower-level than :func:`run_context` — exposed so the workbench
    bootstrap can pin the contextvar from an env variable inside the
    notebook subprocess. Most code should use the context manager.
    """
    return _run_id_var.set(run_id)


def reset_run_id(token: object) -> None:
    """Restore the contextvar to whatever it was before
    :func:`set_run_id`. Mirrors the standard ``ContextVar.reset``."""
    _run_id_var.reset(token)  # type: ignore[arg-type]


@asynccontextmanager
async def run_context(run_id: str | None = None) -> AsyncIterator[str]:
    """Bind a ``run_id`` to the current asyncio context.

    Yields the bound id (generates a fresh uuid4 hex if ``run_id`` is
    ``None``) and restores the previous contextvar value on exit.

    Example::

        from locus.observability import run_context, get_event_bus

        async with run_context() as rid:
            # any emission inside this block tags events with `rid`
            await my_pipeline.run(task)
            # downstream subscribers can now `bus.subscribe(rid)`

    Note:
        The context manager does **not** call
        :meth:`EventBus.close_stream` — callers decide when to close
        based on their own lifecycle (the router does it at end of
        dispatch; notebook subprocesses do it on exit).
    """
    rid = run_id or uuid4().hex
    token = _run_id_var.set(rid)
    # Capture the running loop so worker-thread emit_sync calls can
    # hop publishes back via run_coroutine_threadsafe. Cheap — one
    # ``ContextVar.set`` per dispatch, even if no one ever subscribes.
    loop_token = None
    try:
        loop_token = _owner_loop_var.set(_asyncio.get_running_loop())
    except RuntimeError:
        # Caller is using ``run_context`` from sync code — no loop to
        # capture, but ``emit_sync`` will simply drop in that case.
        pass
    try:
        yield rid
    finally:
        if loop_token is not None:
            _owner_loop_var.reset(loop_token)
        _run_id_var.reset(token)
