# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Threads and runs for :class:`~locus.server.app.AgentServer`.

A **thread** is a conversation; a **run** is one turn on it (or the
continuation of a paused turn). The registry keeps both in memory, drives
each run as a background task and publishes every event the run yields on
a server-owned :class:`~locus.observability.event_bus.EventBus`, numbered
from 1. The bus's per-run history is what makes ``GET /runs/{id}/events``
resumable: a client that reconnects with ``Last-Event-ID: N`` is replayed
every event after ``N`` and then follows the run live.

Bounds, all constructor arguments of :class:`RunRegistry`:

* ``history_per_run`` (500) — events kept per run for replay. A resume that
  reaches past the retained window gets one ``replay_gap`` payload naming
  the sequence numbers it missed, then whatever is still held.
* ``max_runs`` (200) — finished runs kept. The oldest finished run is
  forgotten first once the cap is reached; a running run is never dropped.
* ``ttl_seconds`` (3600) — a finished run, and a thread with no run in
  flight, are forgotten this long after their last activity. Thread state
  itself lives in the agent's checkpointer when one is configured; the
  registry only knows which threads exist and which runs they had.

The registry is in-process: a run started on one worker is only visible on
that worker. Put a sticky route in front of multi-worker deployments, or
pin threads to a worker.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from locus.observability.event_bus import EventBus, StreamEvent


_logger = logging.getLogger(__name__)

RunStatus = Literal["queued", "running", "completed", "interrupted", "cancelled", "failed"]
RunKind = Literal["run", "resume"]

#: Statuses a run never leaves.
FINISHED_STATUSES: frozenset[str] = frozenset({"completed", "interrupted", "cancelled", "failed"})

#: Payload ``type`` of the frame that stands in for evicted events.
REPLAY_GAP = "replay_gap"


class RegistryError(Exception):
    """A request the registry cannot honour; ``code`` is the wire-level reason."""

    code = "registry_error"


class ThreadExistsError(RegistryError):
    """``POST /threads`` named a thread the caller already has."""

    code = "thread_exists"


class RunExistsError(RegistryError):
    """A client-supplied run id is already taken."""

    code = "run_exists"


class RunInProgressError(RegistryError):
    """A run was requested on a thread that already has one in flight."""

    code = "run_in_progress"


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=UTC).isoformat()


@dataclass
class ThreadRecord:
    """A conversation the server knows about."""

    thread_id: str
    principal: str
    created_at: float
    metadata: dict[str, Any] = field(default_factory=dict)
    run_ids: list[str] = field(default_factory=list)
    last_activity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "created_at": _iso(self.created_at),
            "metadata": self.metadata,
            "run_ids": list(self.run_ids),
        }


@dataclass
class RunRecord:
    """One turn (or resumed segment) on a thread, and what became of it."""

    run_id: str
    thread_id: str
    principal: str
    kind: RunKind
    created_at: float
    status: RunStatus = "queued"
    started_at: float | None = None
    finished_at: float | None = None
    #: Sequence number of the last event published; events are numbered from 1.
    event_count: int = 0
    stop_reason: str | None = None
    message: str | None = None
    #: The ``interrupt`` payload the run paused on, while it is paused.
    interrupt: dict[str, Any] | None = None
    error_correlation_id: str | None = None
    cancel_requested: bool = False
    task: asyncio.Task[None] | None = field(default=None, repr=False)
    on_cancel: Callable[[], Any] | None = field(default=None, repr=False)

    @property
    def finished(self) -> bool:
        return self.status in FINISHED_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "kind": self.kind,
            "status": self.status,
            "created_at": _iso(self.created_at),
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
            "event_count": self.event_count,
            "stop_reason": self.stop_reason,
            "message": self.message,
            "interrupt": self.interrupt,
            "error_correlation_id": self.error_correlation_id,
        }


def sanitized_error_payload(exc: BaseException) -> dict[str, Any]:
    """Log ``exc`` under a correlation id and return the generic error frame.

    The same shape ``/stream`` sends: a client never sees ``str(exc)``, which
    routinely carries DSNs, paths and bucket names (CWE-209).
    """
    correlation_id = uuid.uuid4().hex
    _logger.error(
        "agent run error (correlation_id=%s)",
        correlation_id,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return {"type": "error", "error": "internal error", "correlation_id": correlation_id}


class RunRegistry:
    """In-memory threads, runs and the event history behind resumable streams.

    Args:
        ttl_seconds: How long a finished run and an idle thread are kept.
        max_runs: Finished runs kept before the oldest is forgotten.
        history_per_run: Events retained per run for ``Last-Event-ID`` replay.
        cancel_grace_seconds: After a cooperative cancel (``agent.cancel``),
            how long to wait for the run to stop on its own before its task
            is cancelled outright.
        clock: Wall-clock source; tests inject one to drive the TTL.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = 3600.0,
        max_runs: int = 200,
        history_per_run: int = 500,
        cancel_grace_seconds: float = 5.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_runs < 1:
            raise ValueError("max_runs must be at least 1")
        if history_per_run < 1:
            raise ValueError("history_per_run must be at least 1")
        self.ttl_seconds = ttl_seconds
        self.max_runs = max_runs
        self.history_per_run = history_per_run
        self.cancel_grace_seconds = cancel_grace_seconds
        self._clock = clock
        # The bus never evicts on its own: the registry drops a run's history
        # when it forgets the run, so a run the registry still knows about
        # always has its history (or its closed-stream sentinel) on the bus.
        self._bus = EventBus(history_per_run=history_per_run, max_runs_retained=max_runs * 4)
        self._threads: dict[tuple[str, str], ThreadRecord] = {}
        self._runs: dict[str, RunRecord] = {}
        self._helpers: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------------
    # Threads
    # ------------------------------------------------------------------

    def create_thread(
        self,
        principal: str,
        thread_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ThreadRecord:
        """Register a thread; raises :class:`ThreadExistsError` for a known id."""
        self._sweep()
        tid = thread_id or uuid.uuid4().hex
        key = (principal, tid)
        if key in self._threads:
            raise ThreadExistsError(tid)
        now = self._clock()
        record = ThreadRecord(
            thread_id=tid,
            principal=principal,
            created_at=now,
            metadata=dict(metadata or {}),
            last_activity=now,
        )
        self._threads[key] = record
        return record

    def get_thread(self, principal: str, thread_id: str) -> ThreadRecord | None:
        self._sweep()
        return self._threads.get((principal, thread_id))

    def ensure_thread(self, principal: str, thread_id: str) -> ThreadRecord:
        """The caller's thread, created on first use."""
        record = self.get_thread(principal, thread_id)
        if record is None:
            record = self.create_thread(principal, thread_id)
        return record

    def drop_thread(self, principal: str, thread_id: str) -> bool:
        """Forget a thread and its finished runs; a running run is kept."""
        record = self._threads.pop((principal, thread_id), None)
        if record is None:
            return False
        for run_id in list(record.run_ids):
            finished_run = self._runs.get(run_id)
            if finished_run is not None and finished_run.finished:
                self._forget_run(finished_run)
        return True

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def new_run(
        self,
        principal: str,
        thread_id: str,
        *,
        kind: RunKind = "run",
        run_id: str | None = None,
    ) -> RunRecord:
        """Register a run on ``thread_id`` (created on first use).

        Raises :class:`RunInProgressError` when the thread already has a run that
        has not finished, and :class:`RunExistsError` when ``run_id`` is taken.
        """
        thread = self.ensure_thread(principal, thread_id)
        for existing_id in thread.run_ids:
            existing = self._runs.get(existing_id)
            if existing is not None and not existing.finished:
                raise RunInProgressError(existing_id)
        rid = run_id or uuid.uuid4().hex
        if rid in self._runs:
            raise RunExistsError(rid)
        now = self._clock()
        record = RunRecord(
            run_id=rid, thread_id=thread_id, principal=principal, kind=kind, created_at=now
        )
        self._runs[rid] = record
        thread.run_ids.append(rid)
        thread.last_activity = now
        return record

    def get_run(self, principal: str, run_id: str) -> RunRecord | None:
        """The caller's run, or ``None`` (another principal's run is ``None`` too)."""
        self._sweep()
        record = self._runs.get(run_id)
        if record is None or record.principal != principal:
            return None
        return record

    def runs_for(self, thread: ThreadRecord) -> list[RunRecord]:
        return [self._runs[rid] for rid in thread.run_ids if rid in self._runs]

    def start(
        self,
        record: RunRecord,
        source: AsyncIterator[dict[str, Any]],
        *,
        on_cancel: Callable[[], Any] | None = None,
    ) -> None:
        """Drive ``source`` in a background task, publishing each payload.

        ``source`` yields JSON-ready event payloads (the ``/stream`` frame
        shapes). An exception it raises ends the run with a sanitized
        ``error`` frame. ``on_cancel`` asks the run to stop cooperatively
        (``agent.cancel(thread_id=...)``); without one, a cancel stops the
        task outright.
        """
        if record.task is not None:
            raise RuntimeError(f"run {record.run_id!r} already started")
        record.on_cancel = on_cancel
        record.task = asyncio.create_task(
            self._drive(record, source), name=f"locus-run-{record.run_id}"
        )

    def cancel(self, record: RunRecord) -> bool:
        """Request that ``record`` stop. ``False`` when it already finished."""
        if record.finished:
            return False
        record.cancel_requested = True
        task = record.task
        if task is None:
            # Not started yet; ``_drive`` checks the flag before running.
            return True
        if record.on_cancel is None:
            task.cancel()
            return True
        try:
            record.on_cancel()
        except Exception:  # noqa: BLE001 — fall back to a hard cancel below
            _logger.debug("cooperative cancel failed for run %s", record.run_id, exc_info=True)
            task.cancel()
            return True
        helper = asyncio.create_task(self._hard_cancel_after_grace(record))
        self._helpers.add(helper)
        helper.add_done_callback(self._helpers.discard)
        return True

    async def _hard_cancel_after_grace(self, record: RunRecord) -> None:
        await asyncio.sleep(self.cancel_grace_seconds)
        if not record.finished and record.task is not None:
            record.task.cancel()

    async def events(
        self, record: RunRecord, *, after: int = 0
    ) -> AsyncIterator[tuple[int, dict[str, Any]]]:
        """Yield ``(seq, payload)`` for every event of ``record`` after ``after``.

        Replays the retained history first, then follows the run live; ends
        when the run's stream closes. Events no longer retained are summarised
        by one ``replay_gap`` payload (``from``/``to`` are the missing sequence
        numbers) carrying the sequence number of the last one missed.
        """
        expected = after + 1
        async for event in self._bus.subscribe(record.run_id):
            seq = int(event.event_id)
            if seq <= after:
                continue
            if seq > expected:
                yield seq - 1, {"type": REPLAY_GAP, "from": expected, "to": seq - 1}
            yield seq, event.data
            expected = seq + 1
        if record.finished and expected <= record.event_count:
            yield (
                record.event_count,
                {"type": REPLAY_GAP, "from": expected, "to": record.event_count},
            )

    async def aclose(self) -> None:
        """Stop every running run; called at application shutdown."""
        tasks = [r.task for r in self._runs.values() if r.task is not None and not r.finished]
        for task in [*tasks, *self._helpers]:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(BaseException):
                await task

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _publish(self, record: RunRecord, payload: dict[str, Any]) -> int:
        record.event_count += 1
        seq = record.event_count
        await self._bus.publish(
            StreamEvent(
                run_id=record.run_id,
                event_type=str(payload.get("type", "event")),
                data=payload,
                event_id=str(seq),
            )
        )
        return seq

    def _note(self, record: RunRecord, payload: dict[str, Any]) -> RunStatus | None:
        """Update the record from a payload; return the status it implies."""
        kind = payload.get("type")
        if kind == "interrupt":
            record.interrupt = payload
            return "interrupted"
        if kind == "done":
            record.interrupt = None
            message = payload.get("message")
            record.message = message if isinstance(message, str) else None
            reason = payload.get("reason")
            record.stop_reason = reason if isinstance(reason, str) else None
            return "cancelled" if record.stop_reason == "cancelled" else "completed"
        if kind == "error":
            cid = payload.get("correlation_id")
            record.error_correlation_id = cid if isinstance(cid, str) else None
            return "failed"
        return None

    async def _drive(self, record: RunRecord, source: AsyncIterator[dict[str, Any]]) -> None:
        outcome: RunStatus | None = None
        saw_terminal = False
        try:
            if record.cancel_requested:
                outcome = "cancelled"
                return
            record.status = "running"
            record.started_at = self._clock()
            async for payload in source:
                implied = self._note(record, payload)
                if implied is not None:
                    outcome = implied
                    saw_terminal = True
                await self._publish(record, payload)
        except asyncio.CancelledError:
            outcome = "cancelled"
        except Exception as exc:  # noqa: BLE001 — every run error is sanitized
            frame = sanitized_error_payload(exc)
            self._note(record, frame)
            outcome = "failed"
            saw_terminal = True
            with contextlib.suppress(Exception):
                await self._publish(record, frame)
        finally:
            if outcome == "cancelled" and not saw_terminal:
                record.stop_reason = "cancelled"
                with contextlib.suppress(Exception):
                    await self._publish(
                        record, {"type": "done", "message": "", "reason": "cancelled"}
                    )
            record.status = outcome or "completed"
            record.finished_at = self._clock()
            thread = self._threads.get((record.principal, record.thread_id))
            if thread is not None:
                thread.last_activity = record.finished_at
            # A source left suspended (the task was cancelled mid-event) is
            # closed here so the run it drives finalises now, not at GC.
            aclose = getattr(source, "aclose", None)
            if aclose is not None:
                with contextlib.suppress(Exception):
                    await aclose()
            with contextlib.suppress(Exception):
                await self._bus.close_stream(record.run_id)

    def _forget_run(self, record: RunRecord) -> None:
        self._runs.pop(record.run_id, None)
        thread = self._threads.get((record.principal, record.thread_id))
        if thread is not None and record.run_id in thread.run_ids:
            thread.run_ids.remove(record.run_id)
        self._drop_history_soon(record.run_id)

    def _drop_history_soon(self, run_id: str) -> None:
        """``drop_history`` is async; schedule it when a loop is running."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        helper = loop.create_task(self._bus.drop_history(run_id))
        self._helpers.add(helper)
        helper.add_done_callback(self._helpers.discard)

    def _sweep(self) -> None:
        """Forget expired and surplus finished runs, then idle expired threads."""
        now = self._clock()
        finished = sorted(
            (r for r in self._runs.values() if r.finished),
            key=lambda r: r.finished_at or r.created_at,
        )
        surplus = len(finished) - self.max_runs
        for index, run in enumerate(finished):
            expired = (run.finished_at or run.created_at) + self.ttl_seconds <= now
            if expired or index < surplus:
                self._forget_run(run)
        for key, thread in list(self._threads.items()):
            if thread.last_activity + self.ttl_seconds > now:
                continue
            if any(not self._runs[r].finished for r in thread.run_ids if r in self._runs):
                continue
            del self._threads[key]
            for run_id in thread.run_ids:
                orphan = self._runs.get(run_id)
                if orphan is not None:
                    self._forget_run(orphan)


__all__ = [
    "FINISHED_STATUSES",
    "REPLAY_GAP",
    "RegistryError",
    "RunExistsError",
    "RunInProgressError",
    "RunKind",
    "RunRecord",
    "RunRegistry",
    "RunStatus",
    "ThreadExistsError",
    "ThreadRecord",
    "sanitized_error_payload",
]
