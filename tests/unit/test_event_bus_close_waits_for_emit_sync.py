"""``close_stream`` lets the publishes ``emit_sync`` scheduled land before it ends the stream.

``emit_sync`` cannot await the bus: it schedules the publish as a task (or hops it
to the owner loop from a worker thread). A consumer that closed the stream in the
same loop turn used to see the event in the history but never live.
"""

from __future__ import annotations

import asyncio

import pytest

from locus.observability import emit_sync, get_event_bus, run_context
from locus.observability.event_bus import reset_event_bus


@pytest.fixture(autouse=True)
def _fresh_bus() -> None:
    reset_event_bus()


async def _live_events_after_close(emit: object) -> list[str]:
    """Subscribe, run ``emit`` (sync or awaitable), close in the same turn, return what arrived."""
    bus = get_event_bus()
    seen: list[str] = []
    async with run_context() as run_id:

        async def drain() -> None:
            async for event in bus.subscribe(run_id):
                seen.append(event.event_type)

        consumer = asyncio.create_task(drain())
        await asyncio.sleep(0)
        result = emit()  # type: ignore[operator]
        if asyncio.iscoroutine(result):
            await result
        await bus.close_stream(run_id)
        await asyncio.wait_for(consumer, timeout=2.0)
    return seen


class TestCloseStreamWaitsForScheduledPublishes:
    @pytest.mark.asyncio
    async def test_an_event_emitted_from_sync_code_in_the_closing_turn_arrives_live(self) -> None:
        seen = await _live_events_after_close(lambda: emit_sync("sync.site", step=1))
        assert seen == ["sync.site"]

    @pytest.mark.asyncio
    async def test_an_event_emitted_from_a_worker_thread_arrives_live(self) -> None:
        def from_thread() -> None:
            emit_sync("thread.site", step=2)

        seen = await _live_events_after_close(lambda: asyncio.to_thread(from_thread))
        assert seen == ["thread.site"]

    @pytest.mark.asyncio
    async def test_settled_publishes_are_forgotten(self) -> None:
        bus = get_event_bus()
        async with run_context():
            emit_sync("sync.site")
            emit_sync("sync.site")
            await asyncio.sleep(0.01)
        assert bus.stats()["pending_publishes"] == 0

    @pytest.mark.asyncio
    async def test_closing_a_run_waits_only_for_its_own_publishes(self) -> None:
        bus = get_event_bus()
        other: list[str] = []
        async with run_context() as other_run:

            async def drain() -> None:
                async for event in bus.subscribe(other_run):
                    other.append(event.event_type)

            consumer = asyncio.create_task(drain())
            await asyncio.sleep(0)
            emit_sync("other.site")
            async with run_context() as closing_run:
                await bus.close_stream(closing_run)
            # the other run's publish is still pending or just landed; it is not cut off
            await bus.close_stream(other_run)
            await asyncio.wait_for(consumer, timeout=2.0)
        assert other == ["other.site"]
