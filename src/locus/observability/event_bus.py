# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Singleton pub/sub event bus for locus telemetry.

A leaner adaptation of Oracle's optic ``EventBus`` (see
``optic.observability.core.event_bus``). Same semantic surface
— per-run channels + global subscribers, bounded queues, history
replay for late connections, sentinel-based stream close — without
the production-grade backpressure-warning machinery, which we don't
need at v1 scale.

The locus hook system (:class:`locus.hooks.HookProvider`) and the
router both publish through here. The workbench's SSE endpoint at
``/api/events/{run_id}`` is the HTTP wrapper around
:meth:`EventBus.subscribe`.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


logger = logging.getLogger(__name__)


# Bounded buffers — pinned for the v1 release. Optic ratchets these via
# config; we'll add config when we have a real production deployment.
_DEFAULT_QUEUE_SIZE = 1024
_DEFAULT_HISTORY_PER_RUN = 500
_DEFAULT_MAX_RUNS_RETAINED = 200
_PUBLISH_TIMEOUT_SECONDS = 1.0


@dataclass
class StreamEvent:
    """Single event flowing through the bus.

    Fields are intentionally narrow — every consumer (CLI, SSE, JSON
    log) sees the same shape, and ``data`` is the open extension
    point.
    """

    run_id: str
    event_type: str
    data: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    event_id: str = field(default_factory=lambda: str(uuid4()))

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable shape used by the SSE wire format."""
        return {
            "event_id": self.event_id,
            "run_id": self.run_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
        }


class EventBus:
    """In-process pub/sub bus.

    Thread-safe under asyncio (a single ``asyncio.Lock`` guards all
    queue mutations). Not safe across processes — for multi-worker
    deployments, swap the implementation for one backed by Redis /
    NATS / similar.
    """

    def __init__(
        self,
        *,
        max_queue_size: int = _DEFAULT_QUEUE_SIZE,
        history_per_run: int = _DEFAULT_HISTORY_PER_RUN,
        max_runs_retained: int = _DEFAULT_MAX_RUNS_RETAINED,
    ) -> None:
        self._max_queue_size = max_queue_size
        self._history_per_run = history_per_run
        self._max_runs_retained = max_runs_retained

        self._lock = asyncio.Lock()

        # Per-run subscribers: each subscriber owns one bounded queue.
        # Sentinel ``None`` on the queue terminates the subscriber loop.
        self._queues: dict[str, list[asyncio.Queue[StreamEvent | None]]] = defaultdict(list)

        # Global subscribers fire on every event regardless of run_id.
        # Capped to prevent unbounded growth in monitoring deployments.
        self._global_subscribers: list[asyncio.Queue[StreamEvent | None]] = []
        self._max_global_subscribers = 50

        # History buffer for late-joining subscribers — they get the
        # last N events for the run on connect, then live events.
        self._history: dict[str, deque[StreamEvent]] = {}

        # Runs that have already had ``close_stream`` called. A late
        # subscriber for a closed run gets the history + immediate
        # sentinel, so the iterator terminates cleanly instead of
        # hanging on a queue no one publishes to.
        self._closed_runs: set[str] = set()

        # Drop counter for tests + diagnostics.
        self._dropped_events = 0

        # Optional reference to the asyncio loop the bus runs on. Set
        # lazily by ``emit()`` / ``emit_sync()`` on first use so worker
        # threads (the ``@tool`` decorator's executor, ``asyncio.to_thread``
        # callers) can reach back to the right loop via
        # ``run_coroutine_threadsafe``. Never set unless someone actually
        # publishes — preserves the SDK-without-SSE invariant.
        self._owner_loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def publish(self, event: StreamEvent) -> None:
        """Deliver ``event`` to every subscriber on its run + global.

        Non-blocking on slow consumers: each queue.put is wrapped in a
        ``wait_for`` with :data:`_PUBLISH_TIMEOUT_SECONDS`. If a
        subscriber's queue is saturated past the timeout, the event is
        dropped for that subscriber (counted in
        :attr:`_dropped_events`) and the publish continues for
        everyone else.
        """
        if not event.run_id:
            logger.error(
                "EventBus: dropping event with empty run_id (event_type=%s)",
                event.event_type,
            )
            return

        async with self._lock:
            history = self._history.setdefault(event.run_id, deque(maxlen=self._history_per_run))
            history.append(event)
            self._evict_old_runs_locked()

            run_subscribers = list(self._queues.get(event.run_id, ()))
            global_subscribers = list(self._global_subscribers)

        for queue in run_subscribers:
            await self._deliver(queue, event)
        for queue in global_subscribers:
            await self._deliver(queue, event)

    async def _deliver(self, queue: asyncio.Queue[StreamEvent | None], event: StreamEvent) -> None:
        try:
            await asyncio.wait_for(queue.put(event), timeout=_PUBLISH_TIMEOUT_SECONDS)
        except (TimeoutError, asyncio.QueueFull):
            self._dropped_events += 1
            logger.debug(
                "EventBus: dropped event for slow subscriber "
                "(run_id=%s event_type=%s queue_size=%s)",
                event.run_id,
                event.event_type,
                queue.qsize(),
            )

    # ------------------------------------------------------------------
    # Subscribing
    # ------------------------------------------------------------------

    async def subscribe(self, run_id: str) -> AsyncIterator[StreamEvent]:
        """Subscribe to a single run's events.

        On connect, every event currently in history for the run is
        replayed in order, then live events follow. The iterator
        terminates when :meth:`close_stream` is called for the run.
        """
        queue = await self._register(run_id)
        try:
            while True:
                event = await queue.get()
                if event is None:
                    return
                yield event
        finally:
            await self._unregister(run_id, queue)

    async def subscribe_global(self) -> AsyncIterator[StreamEvent]:
        """Subscribe to every event on the bus.

        Useful for monitoring dashboards. Does not replay history —
        only live events. Capped at :attr:`_max_global_subscribers`
        concurrent subscribers; over-cap connections receive an
        immediate sentinel.
        """
        queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue(maxsize=self._max_queue_size)

        async with self._lock:
            if len(self._global_subscribers) >= self._max_global_subscribers:
                logger.warning(
                    "EventBus: global subscriber cap reached (%d); rejecting new subscriber",
                    self._max_global_subscribers,
                )
                return
            self._global_subscribers.append(queue)

        try:
            while True:
                event = await queue.get()
                if event is None:
                    return
                yield event
        finally:
            async with self._lock:
                try:
                    self._global_subscribers.remove(queue)
                except ValueError:
                    pass

    async def _register(self, run_id: str) -> asyncio.Queue[StreamEvent | None]:
        """Create + register a queue, pre-load it with history.

        If the run has already been closed, the queue receives the
        history followed by the sentinel — the subscriber's
        ``async for`` loop yields the history then terminates without
        ever waiting on a live publisher.
        """
        queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue(maxsize=self._max_queue_size)
        async with self._lock:
            for past in self._history.get(run_id, ()):
                # History is bounded; if the queue can't hold it all,
                # drop the oldest entries silently (the live stream is
                # what consumers actually care about).
                try:
                    queue.put_nowait(past)
                except asyncio.QueueFull:
                    break
            if run_id in self._closed_runs:
                # Run is finished — give the consumer the sentinel
                # immediately after the history. Skip live registration.
                try:
                    queue.put_nowait(None)
                except asyncio.QueueFull:
                    self._dropped_events += 1
                return queue
            self._queues[run_id].append(queue)
        return queue

    async def _unregister(
        self,
        run_id: str,
        queue: asyncio.Queue[StreamEvent | None],
    ) -> None:
        async with self._lock:
            queues = self._queues.get(run_id, [])
            try:
                queues.remove(queue)
            except ValueError:
                pass
            if not queues and run_id in self._queues:
                del self._queues[run_id]

    # ------------------------------------------------------------------
    # Stream lifecycle
    # ------------------------------------------------------------------

    async def close_stream(self, run_id: str) -> None:
        """Signal end-of-stream to every active subscriber of ``run_id``.

        Idempotent. **Keeps** the history buffer — a late subscriber
        joining within the retention window still sees the run's
        events plus an immediate sentinel terminating the iterator.
        Use :meth:`drop_history` to forcibly forget a run.
        """
        async with self._lock:
            queues = list(self._queues.get(run_id, ()))
            self._queues.pop(run_id, None)
            self._closed_runs.add(run_id)

        for queue in queues:
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                # Drop the oldest queued event to make room for the
                # sentinel — close must always reach the consumer.
                try:
                    queue.get_nowait()
                    queue.put_nowait(None)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    self._dropped_events += 1

    async def drop_history(self, run_id: str) -> None:
        """Forget a run's history buffer + closed-state.

        Late subscribers after this call see nothing for ``run_id``.
        Useful for explicit cleanup; not called automatically.
        """
        async with self._lock:
            self._history.pop(run_id, None)
            self._closed_runs.discard(run_id)

    async def close_global(self) -> None:
        """Terminate every global subscriber. Used at shutdown."""
        async with self._lock:
            queues = list(self._global_subscribers)
            self._global_subscribers.clear()
        for queue in queues:
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                self._dropped_events += 1

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Snapshot of subscriber counts + drop totals.

        Cheap to call. Read-only — does not lock; treat as approximate
        when called concurrently with publish.
        """
        return {
            "active_runs": len(self._queues),
            "global_subscribers": len(self._global_subscribers),
            "history_runs": len(self._history),
            "dropped_events_total": self._dropped_events,
            "queue_depths": {
                run_id: [q.qsize() for q in queues] for run_id, queues in self._queues.items()
            },
        }

    def _evict_old_runs_locked(self) -> None:
        """Cap the number of runs we hold history for.

        Called under ``self._lock``. Evicts in insertion order — the
        oldest ``run_id`` entries in ``_history`` are dropped first
        once we exceed :attr:`_max_runs_retained`.
        """
        overflow = len(self._history) - self._max_runs_retained
        if overflow <= 0:
            return
        for run_id in list(self._history.keys())[:overflow]:
            del self._history[run_id]
            self._closed_runs.discard(run_id)


# Module-level singleton — matches optic's pattern.
_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Return the process-wide :class:`EventBus`, creating it lazily."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def reset_event_bus() -> None:
    """Reset the singleton — tests only.

    Calling this in production discards every active subscription
    silently. The function is here purely so each pytest test can
    start with a clean bus.
    """
    global _event_bus
    _event_bus = None
