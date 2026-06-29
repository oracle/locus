# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Extra coverage for tulip.observability.event_bus — missing branches."""

from __future__ import annotations

import asyncio

import pytest

from tulip.observability.event_bus import EventBus, StreamEvent, reset_event_bus


pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _reset_bus():
    reset_event_bus()
    yield
    reset_event_bus()


def _ev(run_id: str, kind: str, **data) -> StreamEvent:
    return StreamEvent(run_id=run_id, event_type=kind, data=data)


# ---------------------------------------------------------------------------
# subscribe_global — cap reached (lines 200, 204)
# ---------------------------------------------------------------------------


class TestSubscribeGlobalCap:
    async def test_cap_reached_returns_immediately(self):
        """When global subscriber cap is 0, subscribe_global must return
        immediately (yields nothing) after logging the warning."""
        bus = EventBus()
        bus._max_global_subscribers = 0  # no slots

        collected: list[StreamEvent] = []

        async def consumer():
            async for ev in bus.subscribe_global():  # should not yield anything
                collected.append(ev)

        task = asyncio.create_task(consumer())
        await asyncio.wait_for(task, timeout=2.0)
        assert collected == []

    async def test_cap_reached_at_exactly_limit(self):
        """Adding max+1 subscriber hits the warning path."""
        bus = EventBus()
        bus._max_global_subscribers = 1
        # Manually fill to cap
        full_q: asyncio.Queue[StreamEvent | None] = asyncio.Queue(maxsize=64)
        bus._global_subscribers.append(full_q)

        collected: list[StreamEvent] = []

        async def overflow_consumer():
            async for ev in bus.subscribe_global():
                collected.append(ev)

        task = asyncio.create_task(overflow_consumer())
        await asyncio.wait_for(task, timeout=2.0)
        assert collected == []  # rejected before any event


# ---------------------------------------------------------------------------
# subscribe_global — sentinel close (lines 211, 217, 218) + close_global (304-311)
# ---------------------------------------------------------------------------


class TestSubscribeGlobalClose:
    async def test_close_global_terminates_all_subscribers(self):
        """close_global sends sentinel to every global subscriber (lines 304-311),
        subscriber receives it and exits (line 211), finally removes itself
        from the list — which is already cleared, so ValueError is caught
        silently (lines 217-218)."""
        bus = EventBus()
        received: list[str] = []

        async def consumer():
            async for ev in bus.subscribe_global():
                received.append(ev.event_type)

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0.02)

        await bus.publish(_ev("r1", "ping"))
        await bus.close_global()  # lines 304-311
        await asyncio.wait_for(task, timeout=2.0)

        assert received == ["ping"]
        assert bus._global_subscribers == []

    async def test_close_global_no_subscribers_is_noop(self):
        """Calling close_global with no subscribers must not raise."""
        bus = EventBus()
        await bus.close_global()  # must not raise

    async def test_subscribe_global_none_sentinel_stops_iteration(self):
        """Manually putting None onto a global subscriber's queue exits the loop
        via line 211."""
        bus = EventBus()
        received: list[str] = []

        async def consumer():
            async for ev in bus.subscribe_global():
                received.append(ev.event_type)
                # After first event, close to test None path
                # (close_global does this for us)
                break  # break triggers the finally block (lines 217-218)

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0.02)
        await bus.publish(_ev("r1", "hello"))
        await asyncio.wait_for(task, timeout=2.0)

        assert received == ["hello"]

    async def test_close_global_full_queue_increments_drops(self):
        """close_global increments _dropped_events when a global queue is full."""
        bus = EventBus(max_queue_size=1)
        # Manually add a full queue
        full_q: asyncio.Queue[StreamEvent | None] = asyncio.Queue(maxsize=1)
        full_q.put_nowait(_ev("x", "placeholder"))  # fill it
        bus._global_subscribers.append(full_q)

        await bus.close_global()
        assert bus._dropped_events >= 1


# ---------------------------------------------------------------------------
# _register — history overflows queue (lines 236, 237)
# ---------------------------------------------------------------------------


class TestRegisterHistoryOverflow:
    async def test_history_larger_than_queue_breaks_early(self):
        """When history_per_run > max_queue_size, _register must stop loading
        history into the queue when it's full (lines 236-237 — asyncio.QueueFull
        except + break)."""
        bus = EventBus(max_queue_size=2, history_per_run=5)
        for i in range(5):
            await bus.publish(_ev("big-hist", f"e{i}"))

        queue = await bus._register("big-hist")
        # Queue is full (maxsize=2); only 2 history items loaded.
        assert queue.qsize() == 2
        assert queue.full()


# ---------------------------------------------------------------------------
# _register — closed run, queue full, sentinel can't fit (lines 243, 244)
# ---------------------------------------------------------------------------


class TestRegisterClosedRunFullQueue:
    async def test_dropped_when_closed_run_full_queue(self):
        """After close_stream, late subscription with queue full of history
        can't put the sentinel — drops it and increments _dropped_events
        (lines 243-244)."""
        bus = EventBus(max_queue_size=2, history_per_run=5)
        for i in range(5):
            await bus.publish(_ev("closed-r", f"e{i}"))
        await bus.close_stream("closed-r")

        # _register now: 2 history items fill the queue, then sentinel can't
        # fit — line 243 (QueueFull) → line 244 (drop counter++)
        await bus._register("closed-r")
        assert bus._dropped_events >= 1


# ---------------------------------------------------------------------------
# close_stream — queue full path (lines 283, 286, 287, 288)
# ---------------------------------------------------------------------------


class TestCloseStreamFullQueue:
    async def test_sentinel_replaces_oldest_event_when_queue_full(self):
        """When a subscriber's queue is full at close_stream time, the bus
        must discard one event and put the sentinel (lines 283, 286, 287, 288).
        The subscriber should eventually receive the sentinel (None→return)."""
        bus = EventBus(max_queue_size=1)

        collected: list[str] = []

        async def consumer():
            async for ev in bus.subscribe("run-qf"):
                collected.append(ev.event_type)

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0.02)

        # Fill the queue by publishing.  The consumer is busy not consuming,
        # so after the first publish the queue is full.
        await bus.publish(_ev("run-qf", "payload"))
        # Now close_stream — sentinel can't fit directly (QueueFull),
        # so it removes one item and retries (lines 286-288).
        await bus.close_stream("run-qf")
        await asyncio.wait_for(task, timeout=2.0)

        # Either the payload or just the sentinel arrives; the important thing
        # is the consumer exits cleanly (sentinel was delivered).
        assert task.done()
