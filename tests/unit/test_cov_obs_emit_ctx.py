# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Extra coverage for context.py and emit.py — missing exception branches."""

from __future__ import annotations

import asyncio
import sys

import pytest

from tulip.observability import reset_event_bus
from tulip.observability.context import (
    _owner_loop_var,
    current_run_id,
    reset_run_id,
    run_context,
    set_run_id,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _reset_bus():
    reset_event_bus()
    yield
    reset_event_bus()


# ---------------------------------------------------------------------------
# context.py — lines 127, 130 (RuntimeError except in run_context)
# ---------------------------------------------------------------------------


class TestRunContextNoLoop:
    async def test_run_context_no_loop_skips_owner_loop(self):
        """Patching _asyncio.get_running_loop to raise inside run_context
        hits the except RuntimeError path (lines 127-130).  The context
        still binds run_id correctly; loop_token stays None."""
        import tulip.observability.context as ctx_mod  # noqa: PLC0415

        original_grl = ctx_mod._asyncio.get_running_loop

        def _raising():
            raise RuntimeError("no loop — simulated")

        ctx_mod._asyncio.get_running_loop = _raising
        try:
            async with run_context("ctx-no-loop") as rid:
                assert rid == "ctx-no-loop"
                assert current_run_id() == "ctx-no-loop"
                # owner_loop_var stays unset because we failed to capture the loop.
                assert _owner_loop_var.get() is None
        finally:
            ctx_mod._asyncio.get_running_loop = original_grl

        assert current_run_id() is None


# ---------------------------------------------------------------------------
# emit.py — lines 161, 162 (RuntimeError in loop cache block)
# ---------------------------------------------------------------------------


class TestEmitRuntimeErrorInLoopCapture:
    async def test_emit_survives_get_running_loop_raising(self):
        """When asyncio.get_running_loop() raises inside emit's loop-cache
        block (bus._owner_loop is None), the except RuntimeError:pass path
        (lines 161-162) is taken and the event is still published."""
        emit_mod = sys.modules["tulip.observability.emit"]

        reset_event_bus()
        from tulip.observability.event_bus import get_event_bus  # noqa: PLC0415

        bus = get_event_bus()
        # Force bus._owner_loop to None so the cache block runs.
        bus._owner_loop = None

        original_grl = emit_mod.asyncio.get_running_loop

        def _raising():
            raise RuntimeError("no loop — simulated")

        emit_mod.asyncio.get_running_loop = _raising
        try:
            received: list[str] = []

            async def consumer():
                async for ev in bus.subscribe("rl-test"):
                    received.append(ev.event_type)
                    if ev.event_type == "sentinel":
                        return

            task = asyncio.create_task(consumer())
            await asyncio.sleep(0.01)

            token = set_run_id("rl-test")
            try:
                await emit_mod.emit("rl.test.event", v=1)
                await emit_mod.emit("sentinel")
            finally:
                reset_run_id(token)

            await asyncio.wait_for(task, timeout=2.0)
        finally:
            emit_mod.asyncio.get_running_loop = original_grl

        assert "rl.test.event" in received
        assert "sentinel" in received


# ---------------------------------------------------------------------------
# emit.py — lines 167, 168 (except Exception in emit's publish call)
# ---------------------------------------------------------------------------


class TestEmitExceptionInPublish:
    async def test_emit_swallows_publish_exception(self):
        """If bus.publish raises, emit catches it (lines 167-168) and
        the caller never sees the error — telemetry must not break the SDK."""
        emit_mod = sys.modules["tulip.observability.emit"]

        from tulip.observability.event_bus import get_event_bus  # noqa: PLC0415

        reset_event_bus()
        bus = get_event_bus()

        original_publish = bus.publish

        async def _exploding_publish(event):
            raise RuntimeError("bus exploded")

        bus.publish = _exploding_publish  # type: ignore[method-assign]
        try:
            token = set_run_id("explode-test")
            try:
                # Must not raise — exception is swallowed (lines 167-168).
                await emit_mod.emit("explode.test", v=1)
            finally:
                reset_run_id(token)
        finally:
            bus.publish = original_publish  # type: ignore[method-assign]

        # Reaching here means no exception was propagated.


# ---------------------------------------------------------------------------
# emit.py — lines 206, 207 (emit_sync from worker thread with owner loop)
# ---------------------------------------------------------------------------


class TestEmitSyncFromThread:
    async def test_emit_sync_from_worker_thread_uses_threadsafe(self):
        """emit_sync called from asyncio.to_thread has no running loop in
        the worker but inherits the run_id and owner_loop context vars.
        It must schedule the publish via run_coroutine_threadsafe
        (lines 206-207)."""
        from tulip.observability import emit_sync  # noqa: PLC0415
        from tulip.observability.event_bus import get_event_bus  # noqa: PLC0415

        reset_event_bus()
        bus = get_event_bus()
        received: list[str] = []

        async def collector():
            async for ev in bus.subscribe("thread-emit"):
                received.append(ev.event_type)
                if ev.event_type == "thread.event":
                    return

        task = asyncio.create_task(collector())
        await asyncio.sleep(0.02)

        async with run_context("thread-emit"):
            # asyncio.to_thread propagates context vars — including run_id
            # and owner_loop — into the worker thread.
            def sync_worker():
                emit_sync("thread.event", from_thread=True)

            await asyncio.to_thread(sync_worker)
            # Allow the threadsafe-scheduled publish to land.
            await asyncio.sleep(0.05)

        await asyncio.wait_for(task, timeout=2.0)
        assert "thread.event" in received

    async def test_emit_sync_threadsafe_exception_swallowed(self):
        """If run_coroutine_threadsafe raises (e.g., loop closed), the
        exception is swallowed (lines 208-209) and emit_sync returns normally."""
        emit_mod = sys.modules["tulip.observability.emit"]

        original_rcts = emit_mod.asyncio.run_coroutine_threadsafe

        def _exploding_rcts(coro, loop):
            coro.close()  # avoid ResourceWarning
            raise RuntimeError("loop closed")

        emit_mod.asyncio.run_coroutine_threadsafe = _exploding_rcts
        try:
            async with run_context("rcts-test"):

                def worker():
                    from tulip.observability import emit_sync as _sync  # noqa: PLC0415

                    _sync("rcts.event")  # must not raise

                await asyncio.to_thread(worker)
        finally:
            emit_mod.asyncio.run_coroutine_threadsafe = original_rcts

        # Reaching here means the exception was swallowed (lines 208-209).
