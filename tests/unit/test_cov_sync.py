# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Coverage tests for locus._sync — run_sync and drain helpers."""

from __future__ import annotations

import asyncio
import contextvars

import pytest

from locus._sync import drain, run_sync
from locus.core.loop_bound import loop_bound


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _double(x: int) -> int:
    return x * 2


async def _running_loop() -> asyncio.AbstractEventLoop:
    return asyncio.get_running_loop()


async def _raise_value_error() -> int:
    raise ValueError("synthetic error")


async def _int_gen(n: int):
    for i in range(n):
        yield i


# ---------------------------------------------------------------------------
# run_sync — no running loop path (asyncio.run)
# ---------------------------------------------------------------------------


def test_run_sync_no_loop_returns_value() -> None:
    """run_sync from a sync context returns the coroutine's value."""
    result = run_sync(_double(21))
    assert result == 42


def test_run_sync_no_loop_propagates_exception() -> None:
    """run_sync propagates exceptions from the coroutine."""
    with pytest.raises(ValueError, match="synthetic error"):
        run_sync(_raise_value_error())


# ---------------------------------------------------------------------------
# run_sync — running loop path (background thread)
# ---------------------------------------------------------------------------


async def test_run_sync_inside_loop_returns_value() -> None:
    """run_sync from inside a running loop still returns the value."""
    # asyncio_mode=auto: this test body runs inside an event loop.
    result = run_sync(_double(7))
    assert result == 14


async def test_run_sync_inside_loop_propagates_exception() -> None:
    """run_sync re-raises exceptions raised on the bridge loop."""
    with pytest.raises(ValueError, match="synthetic error"):
        run_sync(_raise_value_error())


async def test_run_sync_inside_loop_runs_on_the_bridge_loop() -> None:
    """Inside a running loop the coroutine runs on the bridge loop, not the caller's."""
    caller = asyncio.get_running_loop()
    assert run_sync(_running_loop()) is not caller


def test_run_sync_reuses_one_loop_across_calls() -> None:
    """Every call runs on the same loop, so loop-bound clients survive between calls."""
    assert run_sync(_running_loop()) is run_sync(_running_loop())


def test_run_sync_keeps_a_loop_bound_resource() -> None:
    """A pool cached per loop is built once: each call sees the loop that built it.

    This is the sync wrappers' defect: a fresh loop per call rebuilt the
    ``oracledb`` pool on every call and left the previous pool's sessions
    open on the database.
    """

    class Holder:
        _client: object | None = None
        _client_loop: object | None = None

    built = 0

    def factory() -> object:
        nonlocal built
        built += 1
        return object()

    holder = Holder()

    async def use() -> object:
        return loop_bound(holder, "_client", factory)

    first = run_sync(use())
    second = run_sync(use())
    assert first is second
    assert built == 1


def test_run_sync_carries_the_callers_context() -> None:
    """The coroutine sees the caller's ContextVars, as ``asyncio.run`` did."""
    who: contextvars.ContextVar[str] = contextvars.ContextVar("who", default="nobody")
    who.set("caller")

    async def read() -> str:
        return who.get()

    assert run_sync(read()) == "caller"


def test_run_sync_from_its_own_thread_raises() -> None:
    """Blocking on the bridge loop from the bridge loop would wait forever."""

    async def nested() -> int:
        return run_sync(_double(1))

    with pytest.raises(RuntimeError, match="thread that runs it"):
        run_sync(nested())


# ---------------------------------------------------------------------------
# drain
# ---------------------------------------------------------------------------


async def test_drain_collects_all_items() -> None:
    """drain() collects every yielded value into a list (lines 66-73)."""
    result = await drain(_int_gen(5))
    assert result == [0, 1, 2, 3, 4]


async def test_drain_empty_iterator() -> None:
    """drain() returns an empty list for an empty async iterator."""
    result = await drain(_int_gen(0))
    assert result == []


async def test_drain_single_item() -> None:
    """drain() works for a single-item async iterator."""

    async def _one():
        yield 42

    result = await drain(_one())
    assert result == [42]
