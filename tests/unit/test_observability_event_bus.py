# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for ``locus.observability.event_bus``.

Covers the load-bearing behaviours: per-run filtering, history replay
on late connect, sentinel close, global subscription, drop accounting
under bounded queues.
"""

from __future__ import annotations

import asyncio

import pytest

from locus.observability.event_bus import (
    EventBus,
    StreamEvent,
    get_event_bus,
    reset_event_bus,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _reset_bus():
    """Each test gets a fresh singleton — no state leaks across tests."""
    reset_event_bus()
    yield
    reset_event_bus()


def _ev(run_id: str, kind: str, **data) -> StreamEvent:
    return StreamEvent(run_id=run_id, event_type=kind, data=data)


# ---------------------------------------------------------------------------
# Singleton behaviour
# ---------------------------------------------------------------------------


class TestSingleton:
    async def test_get_event_bus_returns_same_instance(self):
        a = get_event_bus()
        b = get_event_bus()
        assert a is b

    async def test_reset_creates_new_instance(self):
        a = get_event_bus()
        reset_event_bus()
        b = get_event_bus()
        assert a is not b


# ---------------------------------------------------------------------------
# Pub/sub correctness
# ---------------------------------------------------------------------------


class TestPubSub:
    async def test_subscriber_receives_only_matching_run(self):
        bus = EventBus()
        received: list[str] = []

        async def consumer():
            async for ev in bus.subscribe("run-A"):
                received.append(ev.event_type)
                if ev.event_type == "stop":
                    break

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0.01)
        await bus.publish(_ev("run-A", "first"))
        await bus.publish(_ev("run-B", "ignored"))  # different run
        await bus.publish(_ev("run-A", "second"))
        await bus.publish(_ev("run-A", "stop"))
        await asyncio.wait_for(task, timeout=2.0)
        assert received == ["first", "second", "stop"]

    async def test_global_subscriber_receives_every_event(self):
        bus = EventBus()
        received: list[tuple[str, str]] = []

        async def consumer():
            async for ev in bus.subscribe_global():
                received.append((ev.run_id, ev.event_type))
                if len(received) == 3:
                    return

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0.01)
        await bus.publish(_ev("run-A", "first"))
        await bus.publish(_ev("run-B", "second"))
        await bus.publish(_ev("run-C", "third"))
        await asyncio.wait_for(task, timeout=2.0)
        assert {(r, e) for r, e in received} == {
            ("run-A", "first"),
            ("run-B", "second"),
            ("run-C", "third"),
        }

    async def test_publish_with_empty_run_id_is_dropped(self):
        bus = EventBus()
        # Should log + return without raising. No subscriber for "".
        await bus.publish(_ev("", "should-be-dropped"))
        # Verify nothing landed in history.
        assert "" not in bus._history


# ---------------------------------------------------------------------------
# History replay
# ---------------------------------------------------------------------------


class TestHistoryReplay:
    async def test_late_subscriber_gets_history(self):
        bus = EventBus()
        await bus.publish(_ev("run-1", "early-1"))
        await bus.publish(_ev("run-1", "early-2"))

        replayed: list[str] = []

        async def consumer():
            async for ev in bus.subscribe("run-1"):
                replayed.append(ev.event_type)
                if ev.event_type == "stop":
                    break

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0.01)
        await bus.publish(_ev("run-1", "live-1"))
        await bus.publish(_ev("run-1", "stop"))
        await asyncio.wait_for(task, timeout=2.0)
        # History first, then live events.
        assert replayed == ["early-1", "early-2", "live-1", "stop"]

    async def test_history_capped_per_run(self):
        bus = EventBus(history_per_run=3)
        for i in range(10):
            await bus.publish(_ev("run-1", f"e{i}"))

        replayed: list[str] = []

        async def consumer():
            async for ev in bus.subscribe("run-1"):
                replayed.append(ev.event_type)
                if ev.event_type == "done":
                    break

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0.01)
        await bus.publish(_ev("run-1", "done"))
        await asyncio.wait_for(task, timeout=2.0)
        # Only the last 3 historical events are kept (deque maxlen).
        assert replayed[: len(replayed) - 1] == ["e7", "e8", "e9"]
        assert replayed[-1] == "done"


# ---------------------------------------------------------------------------
# Stream lifecycle
# ---------------------------------------------------------------------------


class TestCloseStream:
    async def test_close_terminates_active_subscribers(self):
        bus = EventBus()

        async def consumer(out: list[str]):
            async for ev in bus.subscribe("run-1"):
                out.append(ev.event_type)

        out: list[str] = []
        task = asyncio.create_task(consumer(out))
        await asyncio.sleep(0.01)
        await bus.publish(_ev("run-1", "alpha"))
        await bus.close_stream("run-1")
        await asyncio.wait_for(task, timeout=2.0)
        assert out == ["alpha"]

    async def test_late_subscriber_after_close_replays_then_terminates(self):
        bus = EventBus()
        await bus.publish(_ev("run-1", "alpha"))
        await bus.publish(_ev("run-1", "beta"))
        await bus.close_stream("run-1")

        out: list[str] = []
        async for ev in bus.subscribe("run-1"):
            out.append(ev.event_type)
        # Late subscriber sees the history then the iterator terminates
        # — without this the subscriber would hang on a closed run.
        assert out == ["alpha", "beta"]

    async def test_close_is_idempotent(self):
        bus = EventBus()
        await bus.publish(_ev("run-1", "alpha"))
        await bus.close_stream("run-1")
        await bus.close_stream("run-1")  # must not raise

    async def test_drop_history_clears_replay(self):
        bus = EventBus()
        await bus.publish(_ev("run-1", "alpha"))
        await bus.close_stream("run-1")
        await bus.drop_history("run-1")
        # After drop_history, the run's history + closed-state are gone;
        # a new subscriber sees the run as if it never existed.
        # We can't assert "hangs forever" cleanly, so we publish a new
        # event and verify the bus accepts it as a fresh run.
        await bus.publish(_ev("run-1", "beta"))
        history = list(bus._history.get("run-1", ()))
        assert [e.event_type for e in history] == ["beta"]


# ---------------------------------------------------------------------------
# Backpressure / bounded queues
# ---------------------------------------------------------------------------


class TestBackpressure:
    async def test_drop_counter_increments_on_slow_subscriber(self):
        # Tiny queue + tiny timeout to force a drop. We don't actually
        # consume from the queue — every publish past max_queue_size
        # times out and increments the drop counter.
        bus = EventBus(max_queue_size=2)

        # Pre-register a subscriber so events have somewhere to go,
        # but never read from it.
        await bus._register("slow")
        for i in range(5):
            await bus.publish(_ev("slow", f"e{i}"))
        # The first 2 fit in the queue, the next 3 should drop.
        assert bus._dropped_events >= 1, (
            f"expected drops on queue overflow; got dropped={bus._dropped_events}"
        )

    async def test_stats_reports_active_state(self):
        bus = EventBus()
        await bus._register("run-1")
        await bus._register("run-1")
        await bus.publish(_ev("run-1", "alpha"))
        stats = bus.stats()
        assert stats["active_runs"] == 1
        assert stats["history_runs"] == 1
        assert stats["queue_depths"]["run-1"] == [1, 1]


# ---------------------------------------------------------------------------
# Run-retention cap
# ---------------------------------------------------------------------------


class TestRunRetentionCap:
    async def test_old_runs_evicted_when_cap_reached(self):
        bus = EventBus(max_runs_retained=3)
        for i in range(5):
            await bus.publish(_ev(f"run-{i}", "alpha"))
        # Only 3 runs in history — the oldest 2 evicted.
        assert len(bus._history) == 3
        # Insertion order eviction: run-0 + run-1 are gone.
        assert "run-0" not in bus._history
        assert "run-1" not in bus._history
        assert "run-4" in bus._history


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestStreamEvent:
    async def test_to_dict_round_trips(self):
        ev = StreamEvent(run_id="r", event_type="foo.bar", data={"x": 1})
        d = ev.to_dict()
        assert d["run_id"] == "r"
        assert d["event_type"] == "foo.bar"
        assert d["data"] == {"x": 1}
        assert "timestamp" in d
        assert "event_id" in d
