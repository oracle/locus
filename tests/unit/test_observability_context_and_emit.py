# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for ``locus.observability.context`` + ``emit`` helpers.

The contract these tests lock in is the **opt-in invariant**: an SDK
user who never enters a ``run_context`` must pay zero cost. Every
emission helper has to no-op without instantiating the bus, allocating
events, or even importing the bus module.

Other behaviours covered:

* ``run_context`` correctly binds and restores the contextvar.
* Concurrent dispatches are isolated — two coroutines under different
  ``run_context``s see different ``current_run_id`` values.
* ``emit`` propagates through nested awaits (the asyncio runtime copies
  the context to child tasks).
* ``emit_sync`` is fire-and-forget; it pins the task on the
  ``_BACKGROUND_TASKS`` set so the GC can't reap it.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from locus.observability import (
    current_run_id,
    emit,
    emit_sync,
    get_event_bus,
    reset_event_bus,
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
# Context manager mechanics
# ---------------------------------------------------------------------------


class TestRunContext:
    async def test_yields_generated_id_when_none_supplied(self):
        async with run_context() as rid:
            assert rid
            assert current_run_id() == rid

    async def test_yields_explicit_id(self):
        async with run_context("my-run") as rid:
            assert rid == "my-run"
            assert current_run_id() == "my-run"

    async def test_restores_previous_value_on_exit(self):
        assert current_run_id() is None
        async with run_context("inner"):
            assert current_run_id() == "inner"
        assert current_run_id() is None

    async def test_nested_contexts_stack_lifo(self):
        async with run_context("outer"):
            assert current_run_id() == "outer"
            async with run_context("inner"):
                assert current_run_id() == "inner"
            # Inner exits — outer restored.
            assert current_run_id() == "outer"
        assert current_run_id() is None

    async def test_set_reset_helpers_round_trip(self):
        token = set_run_id("manual")
        try:
            assert current_run_id() == "manual"
        finally:
            reset_run_id(token)
        assert current_run_id() is None


# ---------------------------------------------------------------------------
# Concurrency isolation
# ---------------------------------------------------------------------------


class TestConcurrentDispatches:
    async def test_two_concurrent_run_contexts_dont_leak(self):
        """Two coroutines running under different run_contexts must see
        different ``current_run_id`` values, even when ``asyncio.gather``
        interleaves them.

        This is the property that lets the workbench dispatch many runs
        on one event loop without cross-talk.
        """
        seen: dict[str, list[str | None]] = {"a": [], "b": []}

        async def under(rid: str, key: str) -> None:
            async with run_context(rid):
                for _ in range(5):
                    seen[key].append(current_run_id())
                    await asyncio.sleep(0)

        await asyncio.gather(under("run-A", "a"), under("run-B", "b"))

        assert all(v == "run-A" for v in seen["a"]), seen
        assert all(v == "run-B" for v in seen["b"]), seen

    async def test_child_task_inherits_parent_run_id(self):
        """``asyncio.create_task`` copies the contextvar at scheduling
        time. Verifies emits from spawned tasks tag with the parent's
        ``run_id``."""
        observed: list[str | None] = []

        async def child() -> None:
            observed.append(current_run_id())

        async with run_context("parent"):
            task = asyncio.create_task(child())
            await task

        assert observed == ["parent"]


# ---------------------------------------------------------------------------
# emit / emit_sync — opt-in invariant
# ---------------------------------------------------------------------------


class TestEmitNoOp:
    async def test_emit_returns_immediately_when_no_run_context(self):
        """The SDK user who never imports the bus must pay nothing.

        We assert the bus singleton is *not* created by ``emit`` when
        there is no run_id on the context. ``reset_event_bus`` puts us
        in a known state; we then call ``emit`` and verify the
        singleton stays uninstantiated.
        """
        # Force a clean state and remove the cached singleton.
        reset_event_bus()

        # Sanity check: import didn't already construct the bus.
        from locus.observability import event_bus as bus_mod  # noqa: PLC0415

        bus_mod._event_bus = None  # type: ignore[attr-defined]

        await emit("router.protocol.selected", protocol_id="direct_response")
        emit_sync("router.protocol.selected", protocol_id="direct_response")

        # Allow any (incorrectly-created) background task to settle
        # before we check.
        await asyncio.sleep(0)

        assert bus_mod._event_bus is None, (  # type: ignore[attr-defined]
            "emit must not instantiate the bus when no run_id is bound"
        )

    async def test_emit_publishes_when_run_context_active(self):
        bus = get_event_bus()
        received: list[str] = []

        async def consumer() -> None:
            async for ev in bus.subscribe("run-X"):
                received.append(ev.event_type)
                if ev.event_type == "stop":
                    return

        consumer_task = asyncio.create_task(consumer())
        await asyncio.sleep(0)  # let the consumer register

        async with run_context("run-X"):
            await emit("first", note="alpha")
            await emit("second", note="beta")
            await emit("stop")

        await asyncio.wait_for(consumer_task, timeout=2.0)
        assert received == ["first", "second", "stop"]


class TestEmitSync:
    async def test_emit_sync_pins_task_on_background_set(self):
        """``emit_sync`` must keep a strong ref to the task it spawns,
        otherwise the GC can reap it before the publish lands."""
        # ``locus.observability.emit`` resolves to the *function* via
        # the package's re-export, not the module — fetch the module
        # directly out of ``sys.modules``.
        emit_mod = sys.modules["locus.observability.emit"]

        emit_mod._BACKGROUND_TASKS.clear()
        async with run_context("run-Y"):
            emit_sync("ping", n=1)
            # Task must be tracked while in flight.
            assert emit_mod._BACKGROUND_TASKS, "emit_sync must keep the task on _BACKGROUND_TASKS"
            # Drain pending background tasks so the done callback fires.
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        # After the loop drains, the done callback must have removed
        # the task — no leaks across runs.
        assert not emit_mod._BACKGROUND_TASKS, (
            f"_BACKGROUND_TASKS should drain on completion, got {emit_mod._BACKGROUND_TASKS}"
        )

    async def test_emit_sync_no_running_loop_drops_silently(self):
        """``emit_sync`` is allowed to be called from sync code without
        a running loop. It must not raise, not start a new loop, not
        instantiate the bus."""
        # We can't actually call this from inside an async test (we are
        # always under a loop). Instead, just assert the absence-of-loop
        # branch is reachable and doesn't blow up.
        import asyncio as _aio  # noqa: PLC0415

        emit_mod = sys.modules["locus.observability.emit"]

        # Patch get_running_loop to raise — simulates sync code path.
        original = _aio.get_running_loop
        _aio.get_running_loop = lambda: (_ for _ in ()).throw(RuntimeError("no loop"))  # type: ignore[assignment]
        try:
            token = set_run_id("offline")
            try:
                # Must not raise.
                emit_mod.emit_sync("orphan", v=1)
            finally:
                reset_run_id(token)
        finally:
            _aio.get_running_loop = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# SDK-without-SSE invariant
# ---------------------------------------------------------------------------


class TestSDKWithoutSSE:
    async def test_importing_observability_does_not_construct_bus(self):
        """Just importing the module surface shouldn't construct the
        bus singleton. Users who don't call ``get_event_bus`` (or
        enter ``run_context``) pay nothing.

        We verify via a *subprocess* — re-importing in-process via
        ``del sys.modules[...]`` would create a new module object and
        leave other tests' cached references dangling, breaking the
        rest of the suite. The subprocess gives us a clean import.
        """
        import subprocess  # noqa: PLC0415
        import sys as _sys  # noqa: PLC0415

        proc = subprocess.run(
            [
                _sys.executable,
                "-c",
                (
                    "import locus.observability;"
                    "from locus.observability import event_bus as m;"
                    "import sys; "
                    "sys.exit(0 if m._event_bus is None else 1)"
                ),
            ],
            check=False,
            capture_output=True,
        )
        assert proc.returncode == 0, (
            "Re-importing locus.observability must not eagerly create the bus; "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )

    async def test_full_emit_call_path_without_run_context_is_nop(self):
        """End-to-end no-op: every public emit verb the SDK uses returns
        immediately and produces no observable side effects when no
        ``run_context`` is active."""
        from locus.observability.emit import (  # noqa: PLC0415
            EV_CHECKPOINT_LOADED,
            EV_CHECKPOINT_SAVED,
            EV_HANDOFF_INITIATED,
            EV_LOOP_ITERATION_COMPLETED,
            EV_ORCHESTRATOR_DECISION,
            EV_PIPELINE_FANOUT_STARTED,
            EV_SKILL_ACTIVATED,
            EV_SPECIALIST_STARTED,
        )

        # No run_context active — every emit must drop on the floor.
        await emit(EV_ORCHESTRATOR_DECISION, decision="invoke")
        await emit(EV_SPECIALIST_STARTED, specialist_id="s1")
        await emit(EV_HANDOFF_INITIATED, source="a", target="b")
        await emit(EV_PIPELINE_FANOUT_STARTED, stages=3)
        await emit(EV_LOOP_ITERATION_COMPLETED, iteration=1)
        await emit(EV_CHECKPOINT_SAVED, thread_id="t")
        await emit(EV_CHECKPOINT_LOADED, thread_id="t")
        emit_sync(EV_SKILL_ACTIVATED, skill_name="research")

        # Bus should remain unborn.
        from locus.observability import event_bus as bus_mod  # noqa: PLC0415

        assert bus_mod._event_bus is None, (  # type: ignore[attr-defined]
            "All-emits-without-context invariant violated"
        )
