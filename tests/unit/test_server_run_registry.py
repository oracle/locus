# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``RunRegistry`` on its own: guards, cancel paths, retention and sweeping.

The HTTP and WebSocket tests drive the registry through the routes; these
reach the branches a well-behaved client never does — a cancel before the
task starts, a cooperative cancel that raises or never stops, a thread
dropped under a running run, history lost under a finished run, and the
sweeper's edge cases with an injected clock.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from locus.server.runs import (
    REPLAY_GAP,
    RunExistsError,
    RunInProgressError,
    RunRegistry,
    ThreadExistsError,
)


async def _gen(
    payloads: list[dict[str, Any]], *, gate: asyncio.Event | None = None
) -> AsyncIterator[dict[str, Any]]:
    for payload in payloads:
        if gate is not None:
            await gate.wait()
        yield payload


async def _collect(registry: RunRegistry, record: Any, after: int = 0) -> list[Any]:
    return [(seq, payload) async for seq, payload in registry.events(record, after=after)]


async def _finished(record: Any) -> None:
    assert record.task is not None
    await asyncio.wait_for(record.task, 2.0)


DONE = {"type": "done", "message": "ok", "reason": "complete"}
CANCELLED = {"type": "done", "message": "", "reason": "cancelled"}


async def test_registration_guards() -> None:
    registry = RunRegistry()
    registry.create_thread("p", "t")
    with pytest.raises(ThreadExistsError):
        registry.create_thread("p", "t")
    gate = asyncio.Event()
    first = registry.new_run("p", "t")
    registry.start(first, _gen([DONE], gate=gate))
    with pytest.raises(RunInProgressError):
        registry.new_run("p", "t")
    with pytest.raises(RunExistsError):
        registry.new_run("p", "other", run_id=first.run_id)
    with pytest.raises(RuntimeError, match="already started"):
        registry.start(first, _gen([DONE]))
    gate.set()
    await _finished(first)
    assert first.status == "completed"
    # The thread is free again, and another principal never sees the run.
    assert registry.new_run("p", "t").run_id != first.run_id
    assert registry.get_run("q", first.run_id) is None
    assert registry.get_run("p", first.run_id) is first


async def test_cancel_before_start_skips_the_run_entirely() -> None:
    registry = RunRegistry()
    record = registry.new_run("p", "t")
    assert registry.cancel(record) is True  # not started yet
    iterated = False

    async def source() -> AsyncIterator[dict[str, Any]]:
        nonlocal iterated
        iterated = True
        yield DONE

    registry.start(record, source())
    await _finished(record)
    assert record.status == "cancelled"
    assert record.stop_reason == "cancelled"
    assert iterated is False
    assert await _collect(registry, record) == [(1, CANCELLED)]
    # A finished run cannot be cancelled again.
    assert registry.cancel(record) is False


async def test_cancel_falls_back_to_the_task_when_on_cancel_raises() -> None:
    registry = RunRegistry()
    record = registry.new_run("p", "t")
    gate = asyncio.Event()

    def broken_cancel() -> None:
        raise RuntimeError("no cancel today")

    registry.start(record, _gen([DONE], gate=gate), on_cancel=broken_cancel)
    await asyncio.sleep(0)
    assert registry.cancel(record) is True
    await _finished(record)
    assert record.status == "cancelled"
    assert await _collect(registry, record) == [(1, CANCELLED)]


async def test_cancel_turns_hard_after_the_grace_period() -> None:
    registry = RunRegistry(cancel_grace_seconds=0.01)
    record = registry.new_run("p", "t")
    gate = asyncio.Event()
    asked: list[str] = []
    registry.start(record, _gen([DONE], gate=gate), on_cancel=lambda: asked.append("stop"))
    await asyncio.sleep(0)
    assert registry.cancel(record) is True
    assert asked == ["stop"]
    await _finished(record)  # the grace helper cancelled the task
    assert record.status == "cancelled"
    assert record.cancel_requested is True


async def test_aclose_stops_runs_in_flight() -> None:
    registry = RunRegistry()
    record = registry.new_run("p", "t")
    registry.start(record, _gen([DONE], gate=asyncio.Event()))
    await asyncio.sleep(0)
    await registry.aclose()
    assert record.status == "cancelled"
    assert record.finished_at is not None


async def test_drop_thread_forgets_finished_runs_and_keeps_the_running_one() -> None:
    registry = RunRegistry()
    done = registry.new_run("p", "t")
    registry.start(done, _gen([DONE]))
    await _finished(done)
    gate = asyncio.Event()
    running = registry.new_run("p", "t")
    registry.start(running, _gen([DONE], gate=gate))
    await asyncio.sleep(0)

    assert registry.drop_thread("p", "t") is True
    assert registry.drop_thread("p", "t") is False
    assert registry.get_run("p", done.run_id) is None
    assert registry.get_run("p", running.run_id) is running

    # The run finishes under a thread that no longer exists.
    gate.set()
    await _finished(running)
    assert running.status == "completed"
    assert registry.get_thread("p", "t") is None
    # Forgetting it later tolerates the missing thread too.
    registry._forget_run(running)
    assert registry.get_run("p", running.run_id) is None


async def test_events_report_history_that_is_no_longer_held() -> None:
    registry = RunRegistry()
    record = registry.new_run("p", "t")
    registry.start(record, _gen([{"type": "think", "content": str(i)} for i in range(3)] + [DONE]))
    await _finished(record)
    assert record.event_count == 4

    class _EmptyBus:
        async def subscribe(self, run_id: str) -> AsyncIterator[Any]:
            return
            yield  # pragma: no cover — makes this an async generator

    registry._bus = _EmptyBus()  # type: ignore[assignment]
    assert await _collect(registry, record, after=1) == [
        (4, {"type": REPLAY_GAP, "from": 2, "to": 4})
    ]
    assert await _collect(registry, record, after=4) == []


async def test_sources_without_aclose_are_driven_too() -> None:
    class _Plain:
        def __init__(self) -> None:
            self.items = iter([{"type": "think", "content": "x"}, DONE])

        def __aiter__(self) -> _Plain:
            return self

        async def __anext__(self) -> dict[str, Any]:
            try:
                return next(self.items)
            except StopIteration:
                raise StopAsyncIteration from None

    registry = RunRegistry()
    record = registry.new_run("p", "t")
    registry.start(record, _Plain())
    await _finished(record)
    assert record.status == "completed"
    assert record.message == "ok"
    assert [seq for seq, _ in await _collect(registry, record)] == [1, 2]


def test_sweep_outside_an_event_loop_forgets_expired_runs() -> None:
    now = [100.0]
    registry = RunRegistry(ttl_seconds=10, clock=lambda: now[0])
    record = registry.new_run("p", "t")
    record.status = "completed"
    record.finished_at = now[0]
    now[0] += 11
    # No running loop here: dropping the bus history is skipped, the record goes.
    assert registry.get_run("p", record.run_id) is None
    assert registry.get_thread("p", "t") is None


def test_sweep_keeps_a_thread_with_a_run_in_flight_and_forgets_orphans() -> None:
    now = [100.0]
    registry = RunRegistry(ttl_seconds=10, clock=lambda: now[0])
    active = registry.new_run("p", "busy")
    active.status = "running"
    now[0] += 11
    # Idle past the TTL by the clock, but its run has not finished: kept.
    assert registry.get_thread("p", "busy") is not None
    assert registry.get_run("p", active.run_id) is active

    quiet = registry.new_run("p", "quiet")
    quiet.status = "completed"
    quiet.finished_at = now[0]
    thread = registry.get_thread("p", "quiet")
    assert thread is not None
    # The run is within its TTL, the thread is not (a clock that went backwards,
    # or a record patched by hand): the thread goes and takes the run with it.
    thread.last_activity = now[0] - 100
    registry._sweep()
    assert registry.get_thread("p", "quiet") is None
    assert registry.get_run("p", quiet.run_id) is None
    assert registry.get_run("p", active.run_id) is active


def test_record_projections() -> None:
    now = [1_700_000_000.0]
    registry = RunRegistry(clock=lambda: now[0])
    thread = registry.create_thread("p", "t", {"case": 1})
    record = registry.new_run("p", "t", kind="resume")
    assert thread.to_dict() == {
        "thread_id": "t",
        "created_at": "2023-11-14T22:13:20+00:00",
        "metadata": {"case": 1},
        "run_ids": [record.run_id],
    }
    projected = record.to_dict()
    assert projected["kind"] == "resume"
    assert projected["status"] == "queued"
    assert projected["started_at"] is None
    assert projected["event_count"] == 0
    assert registry.runs_for(thread) == [record]
