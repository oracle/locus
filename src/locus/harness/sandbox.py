# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The sandbox protocol: a workspace with a lifecycle, limits and evidence hooks.

:class:`~locus.harness.backend.WorkspaceBackend` is what the tools need from
a workspace: files and a shell. :class:`SandboxBackend` is what the harness
needs from a *sandbox*: the same files and shell, plus the three things a
remote workspace has that a directory on the host does not.

- **A lifecycle.** A sandbox is created, runs, may be stopped and is
  destroyed. :attr:`SandboxBackend.state` says where it is;
  :meth:`~SandboxBackend.start`, :meth:`~SandboxBackend.stop` and
  :meth:`~SandboxBackend.close` move it on, and every one of them is
  idempotent, so a caller that is not sure what happened calls again. The
  creation step is a classmethod on each backend (``create``, ``connect``),
  because what it takes — a project, a host, a compartment — is the one
  thing the backends do not share; :mod:`locus.harness.spec` turns a
  declarative :class:`~locus.harness.spec.SandboxSpec` into that call.
- **Limits.** :attr:`SandboxBackend.limits` declares what the sandbox was
  given: CPUs, memory, how long one command may run, how much output one
  command returns. ``None`` means unbounded or unknown. A limit on command
  time is enforced here, by clamping every ``exec`` timeout to it; the others
  are enforced by whatever runs the sandbox and are declared so a policy can
  read them.
- **Evidence hooks.** :meth:`SandboxBackend.add_exec_hook` registers a
  callable that receives an :class:`~locus.harness.evidence.ExecRecord` for
  every command the backend runs for a caller — digests, not text — whether
  or not the command came through a harness tool. The tools still publish
  their own records on the event bus; a hook is for the caller that holds the
  backend: a runner's evidence ledger, a test, a gateway. A hook runs after
  the command; one that raises is logged and the result still reaches the
  caller, who must not lose what the command did because a listener failed.

:class:`SandboxMixin` implements all of this once. Every backend in
:mod:`locus.harness` mixes it in, so a backend written over
:class:`~locus.harness.session.SessionBackend` gets the protocol by
overriding at most three hooks: ``_start``, ``_stop`` and ``_destroy``.

:class:`SandboxPool` is the client side of a *warm pool*: sandboxes created
ahead of need and handed out one at a time, for a backend whose creation
takes a minute. A service that keeps its own warm pool (OCI GenAI Sandboxes)
is told about it through :class:`WarmPoolHint.name` instead; the spec carries
both and each backend honours the one it can.

``locus.harness.testing.assert_backend_conformance`` checks a backend against
this protocol and the workspace contract behind it.
"""

from __future__ import annotations

import atexit
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from locus.harness.backend import BackendCapabilities, ExecResult, WorkspaceBackend
from locus.harness.evidence import ExecRecord, exec_record


__all__ = [
    "ExecHook",
    "LifecycleState",
    "ResourceLimits",
    "SandboxBackend",
    "SandboxMixin",
    "SandboxPool",
    "WarmPoolHint",
]

logger = logging.getLogger(__name__)

#: Seconds a pool still open at interpreter exit waits for a sandbox being
#: created, so that it can be closed rather than left running.
EXIT_GRACE = 60.0

#: Where a sandbox is in its life. ``starting`` is between creation and the
#: first command; ``stopped`` keeps the sandbox but not its compute; ``closed``
#: is destroyed, or detached for a sandbox the backend did not create.
LifecycleState = Literal["starting", "running", "stopped", "closed"]

#: Receives a record of every command the backend ran for a caller.
ExecHook = Callable[[ExecRecord], None]


@dataclass(frozen=True)
class ResourceLimits:
    """What a sandbox was given. ``None`` is unbounded, or not known.

    Attributes:
        cpus: Processor cores, possibly fractional.
        memory_mb: Memory in megabytes.
        disk_mb: Writable storage in megabytes.
        max_exec_s: The longest one command may run. Every ``exec`` timeout is
            clamped to it.
        max_output_bytes: The most output one command returns to the caller.
    """

    cpus: float | None = None
    memory_mb: int | None = None
    disk_mb: int | None = None
    max_exec_s: float | None = None
    max_output_bytes: int | None = None

    def bound(self, timeout: float) -> float:
        """``timeout``, or the command-time limit when that is shorter."""
        if self.max_exec_s is None:
            return timeout
        return min(timeout, self.max_exec_s)


@dataclass(frozen=True)
class WarmPoolHint:
    """How many sandboxes to have ready before they are asked for.

    Attributes:
        name: A pool the service itself keeps, for a backend whose service
            has one (an OCI GenAI Sandbox project's warm pool). The backend
            passes it on at creation.
        ready: How many sandboxes a :class:`SandboxPool` keeps created on
            this side, for a backend whose creation is slow and whose service
            has no pool of its own. Zero means create on demand.
    """

    name: str | None = None
    ready: int = 0


@runtime_checkable
class SandboxBackend(WorkspaceBackend, Protocol):
    """A workspace with a lifecycle, declared limits and evidence hooks.

    Everything :class:`~locus.harness.backend.WorkspaceBackend` has, and:
    """

    @property
    def state(self) -> LifecycleState:
        """Where the sandbox is in its life."""
        ...

    @property
    def limits(self) -> ResourceLimits:
        """What the sandbox was given; ``None`` fields are unbounded or unknown."""
        ...

    def start(self) -> None:
        """Make the sandbox ready for commands. A running sandbox is left alone."""
        ...

    def stop(self) -> None:
        """End every command and release the compute; keep the sandbox.

        What survives a stop is the backend's to say: a container's
        filesystem usually does not, a host's does. Idempotent.
        """
        ...

    def close(self) -> None:
        """Stop, then destroy whatever this backend created. Idempotent.

        A backend attached to a sandbox it did not create ends its own
        commands and detaches; the sandbox stays.
        """
        ...

    def add_exec_hook(self, hook: ExecHook) -> Callable[[], None]:
        """Call ``hook`` with a record of every command run for a caller.

        The hook runs after the command. One that raises is logged and
        skipped; the command's result reaches the caller regardless.
        Returns a callable that removes the hook again.
        """
        ...


class SandboxMixin:
    """The sandbox protocol, written once over a backend's own operations.

    A backend mixes this in before its other bases and overrides the hooks
    it needs: ``_start`` to resume compute, ``_stop`` to end commands and
    release it, ``_destroy`` to delete what ``create`` made. The default of
    each is to do nothing. The backend's ``exec`` and ``start_background``
    call :meth:`_record` so registered hooks see every command, and clamp
    their timeout with :meth:`_bounded`.

    The state machine is forgiving on purpose. ``start`` on a running sandbox,
    ``stop`` on a stopped one and ``close`` on a closed one are no-ops. A hook
    that raises leaves the state where it was, so the next call retries it and
    repeats no step that succeeded. ``close`` attempts ``_destroy`` even when
    ``_stop`` raised, because a sandbox that outlives the process is the worse
    outcome; once ``_destroy`` has succeeded the sandbox is closed whatever
    ``_stop`` did, and when it raises the next ``close`` tries again.
    """

    _sandbox_state: LifecycleState = "running"
    _sandbox_limits: ResourceLimits = ResourceLimits()
    _sandbox_hooks: tuple[ExecHook, ...] = ()
    _sandbox_hook_lock: threading.Lock | None = None

    @property
    def capabilities(self) -> BackendCapabilities:  # pragma: no cover - provided by the backend
        raise NotImplementedError

    # -------------------------------------------------------------- state --

    @property
    def state(self) -> LifecycleState:
        return self._sandbox_state

    @property
    def limits(self) -> ResourceLimits:
        return self._sandbox_limits

    @limits.setter
    def limits(self, limits: ResourceLimits) -> None:
        self._sandbox_limits = limits

    def start(self) -> None:
        if self._sandbox_state == "closed":
            raise RuntimeError(f"the {self.capabilities.label} sandbox is closed")
        if self._sandbox_state == "running":
            return
        self._start()
        self._sandbox_state = "running"

    def stop(self) -> None:
        if self._sandbox_state in ("stopped", "closed"):
            return
        self._stop()
        self._sandbox_state = "stopped"

    def close(self) -> None:
        if self._sandbox_state == "closed":
            return
        try:
            if self._sandbox_state != "stopped":
                self._stop()
                self._sandbox_state = "stopped"
        finally:
            self._destroy()
            self._sandbox_state = "closed"

    def _start(self) -> None:
        """Resume the sandbox's compute. Nothing to do for most backends."""

    def _stop(self) -> None:
        """End every command and release the compute."""

    def _destroy(self) -> None:
        """Delete what ``create`` made; detach from what ``connect`` found."""

    # -------------------------------------------------------------- hooks --

    def add_exec_hook(self, hook: ExecHook) -> Callable[[], None]:
        with self._hook_lock():
            self._sandbox_hooks = (*self._sandbox_hooks, hook)

        def remove() -> None:
            with self._hook_lock():
                self._sandbox_hooks = tuple(h for h in self._sandbox_hooks if h is not hook)

        return remove

    def _hook_lock(self) -> threading.Lock:
        # Created on first use: a mixin has no ``__init__`` of its own to run.
        lock = self._sandbox_hook_lock
        if lock is None:
            lock = self._sandbox_hook_lock = threading.Lock()
        return lock

    def _record(self, command: str, result: ExecResult, *, background: bool = False) -> None:
        """Hand every registered hook a record of ``command``.

        The command has run by now, so a hook that raises must not take the
        result from the caller — who might run the command again. The error
        is logged and the other hooks still see the record.
        """
        hooks = self._sandbox_hooks
        if not hooks:
            return
        record = exec_record(
            command, result, backend_label=self.capabilities.label, background=background
        )
        for hook in hooks:
            try:
                hook(record)
            except Exception:
                logger.exception(
                    "an exec hook on %s failed; the result stands", record.backend_label
                )

    def _bounded(self, timeout: float) -> float:
        """``timeout`` clamped to the declared command-time limit."""
        return self._sandbox_limits.bound(timeout)


class SandboxPool:
    """Sandboxes created ahead of need, handed out one at a time.

    A backend whose creation takes a minute makes a poor interactive tool.
    The pool runs ``factory`` on worker threads so that ``ready`` sandboxes
    are waiting; :meth:`acquire` hands one over and starts its replacement.
    A sandbox that comes back through :meth:`release` is closed, not reused:
    a workspace an agent has worked in is not clean, and a fresh one is what
    the next run expects.

    A factory error is kept for :meth:`acquire` to raise when nothing else
    is to be had: not while a replacement is still being created, and not
    once a later creation has succeeded. A pool still open when the
    interpreter exits closes what it holds, waiting up to :data:`EXIT_GRACE`
    seconds for a sandbox being created so that it is closed too, not left
    running.

    Args:
        factory: Creates one sandbox, synchronously. Called on worker
            threads, so it must be safe to call concurrently.
        ready: How many sandboxes to keep created. Zero means create inline
            in :meth:`acquire`.
    """

    def __init__(self, factory: Callable[[], SandboxBackend], *, ready: int = 1) -> None:
        if ready < 0:
            raise ValueError(f"ready must be zero or more, got {ready}")
        self._factory = factory
        self._target = ready
        self._idle: list[SandboxBackend] = []
        self._warming = 0
        self._errors: list[BaseException] = []
        self._closed = False
        self._cond = threading.Condition()
        atexit.register(self._at_exit)
        self._fill()

    @property
    def ready(self) -> int:
        """How many sandboxes are created and waiting."""
        with self._cond:
            return len(self._idle)

    def _fill(self) -> None:
        with self._cond:
            while not self._closed and len(self._idle) + self._warming < self._target:
                self._start_warm()

    def _start_warm(self) -> None:
        # Under the lock. The thread is a daemon so a factory that hangs does
        # not hold the interpreter; ``_at_exit`` waits for it within reason.
        self._warming += 1
        threading.Thread(target=self._warm, name="locus-sandbox-pool", daemon=True).start()

    def _warm(self) -> None:
        try:
            backend = self._factory()
        except BaseException as exc:  # noqa: BLE001 - reported to the caller of acquire()
            with self._cond:
                self._warming -= 1
                self._errors.append(exc)
                self._cond.notify_all()
            return
        with self._cond:
            stale = self._closed
            if not stale:
                self._idle.append(backend)
                self._errors.clear()
                self._warming -= 1
                self._cond.notify_all()
        if stale:
            # Counted as warming until it is closed, so a close or an exit
            # waiting for the pool to drain waits for this too.
            try:
                backend.close()
            finally:
                with self._cond:
                    self._warming -= 1
                    self._cond.notify_all()

    def acquire(self, timeout: float | None = None) -> SandboxBackend:
        """A created sandbox, waiting at most ``timeout`` seconds for one.

        Raises:
            RuntimeError: The pool is closed, or no sandbox arrived in time.
            Exception: The factory failed and there is no sandbox to give
                instead, nor one on its way; the factory's own error.
        """
        with self._cond:
            if self._closed:
                raise RuntimeError("the sandbox pool is closed")
            if not self._idle and self._warming == 0:
                self._start_warm()
            ready = self._cond.wait_for(
                lambda: bool(self._idle) or (bool(self._errors) and self._warming == 0),
                timeout=timeout,
            )
            if not ready:
                raise RuntimeError(f"no sandbox was ready within {timeout:g}s")
            if not self._idle:
                error, self._errors = self._errors[-1], []
                raise error
            backend = self._idle.pop(0)
        self._fill()
        return backend

    def release(self, backend: SandboxBackend) -> None:
        """Close ``backend``; its replacement is already being created."""
        backend.close()
        self._fill()

    def close(self) -> None:
        """Close every waiting sandbox and stop creating more."""
        with self._cond:
            self._closed = True
            idle, self._idle = self._idle, []
            self._cond.notify_all()
        atexit.unregister(self._at_exit)
        for backend in idle:
            backend.close()

    def _at_exit(self) -> None:
        """Close the pool at interpreter exit, waiting for sandboxes being created."""
        with self._cond:
            self._closed = True
            self._cond.wait_for(lambda: self._warming == 0, timeout=EXIT_GRACE)
        self.close()
