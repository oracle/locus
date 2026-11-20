# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Run an async coroutine from sync code, on one event loop that stays up.

Single shared helper used by every ``Sync*`` wrapper class shipped
alongside the async primitives (checkpointer / store / vector store /
embedder). It bridges synchronous callers into the async drivers
without taking any langchain or langgraph dependency.

Every call runs on the same event loop, hosted on one daemon thread for
the life of the process. That is what makes a sync wrapper usable more
than once: ``oracledb`` async pools (and every other async client) bind
to the loop that created them, so a fresh ``asyncio.run()`` per call made
each call build a new pool and leave the previous pool's sessions open
on the database, and ``close()`` then tried to close a pool whose loop
was already gone. On one loop the pool is built once, reused by every
call, and closed by ``close()``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import threading
from collections.abc import AsyncIterator, Coroutine
from typing import Any, TypeVar


T = TypeVar("T")

_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None


def _bridge_loop() -> asyncio.AbstractEventLoop:
    """The loop every :func:`run_sync` call runs on, started on first use.

    Rebuilt when its thread is gone — after ``os.fork()`` the child has
    the loop object but not the thread that was running it.
    """
    global _loop, _thread
    with _lock:
        if _loop is None or _thread is None or not _thread.is_alive():
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever, name="locus-sync-bridge", daemon=True
            )
            thread.start()
            _loop, _thread = loop, thread
        return _loop


def _settle(task: asyncio.Future[Any], done: concurrent.futures.Future[Any]) -> None:
    """Carry ``task``'s outcome over to the future the calling thread waits on."""
    if task.cancelled():
        done.cancel()
        return
    exc = task.exception()
    if exc is not None:
        done.set_exception(exc)
    else:
        done.set_result(task.result())


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Drive a coroutine to completion from synchronous context.

    Works whether or not the calling thread has a running loop: the
    coroutine runs on the bridge loop's thread, in a copy of the caller's
    :mod:`contextvars` context, and the caller blocks until it finishes.
    Its return value comes back, its exception is re-raised.

    Raises:
        RuntimeError: Called from the bridge loop's own thread, where
            blocking on it would wait forever; code already running there
            awaits the coroutine instead.
    """
    loop = _bridge_loop()
    if threading.current_thread() is _thread:
        coro.close()
        raise RuntimeError(
            "run_sync() was called from the thread that runs it; "
            "code on the bridge loop awaits the coroutine instead."
        )

    done: concurrent.futures.Future[T] = concurrent.futures.Future()
    context = contextvars.copy_context()

    def start() -> None:
        try:
            task = loop.create_task(coro, context=context)
        except BaseException as exc:  # noqa: BLE001 — re-raised in the calling thread
            done.set_exception(exc)
            return
        task.add_done_callback(lambda finished: _settle(finished, done))

    loop.call_soon_threadsafe(start)
    return done.result()


async def drain(agen: AsyncIterator[T]) -> list[T]:
    """Drain an async iterator into a list.

    Tiny helper used by every sync wrapper that needs to surface an
    async generator as a plain ``list[...]`` to its synchronous
    caller. Kept here so each wrapper module doesn't redefine it.
    """
    return [x async for x in agen]
