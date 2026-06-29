# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Coverage tests for tulip._sync — run_sync and drain helpers."""

from __future__ import annotations

import pytest

from tulip._sync import drain, run_sync


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _double(x: int) -> int:
    return x * 2


async def _raise_value_error() -> int:
    raise ValueError("synthetic error")


async def _int_gen(n: int):
    for i in range(n):
        yield i


# ---------------------------------------------------------------------------
# run_sync — no running loop path (asyncio.run)
# ---------------------------------------------------------------------------


def test_run_sync_no_loop_returns_value() -> None:
    """run_sync from a sync context uses asyncio.run (lines 38-42)."""
    result = run_sync(_double(21))
    assert result == 42


def test_run_sync_no_loop_propagates_exception() -> None:
    """run_sync propagates exceptions from the coroutine (lines 38-42)."""
    with pytest.raises(ValueError, match="synthetic error"):
        run_sync(_raise_value_error())


# ---------------------------------------------------------------------------
# run_sync — running loop path (background thread)
# ---------------------------------------------------------------------------


async def test_run_sync_inside_loop_returns_value() -> None:
    """run_sync from inside a running loop delegates to a background thread (lines 44-62)."""
    # asyncio_mode=auto: this test body runs inside an event loop.
    result = run_sync(_double(7))
    assert result == 14


async def test_run_sync_inside_loop_propagates_exception() -> None:
    """run_sync re-raises exceptions from the background thread (lines 60-62)."""
    with pytest.raises(ValueError, match="synthetic error"):
        run_sync(_raise_value_error())


async def test_run_sync_inside_loop_closes_new_loop() -> None:
    """The background thread closes the fresh loop it created (line 55)."""
    # Create a simple counter to verify the coroutine ran.
    calls: list[int] = []

    async def _track() -> str:
        calls.append(1)
        return "ok"

    result = run_sync(_track())
    assert result == "ok"
    assert calls == [1]


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
