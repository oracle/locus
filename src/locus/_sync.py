# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Run an async coroutine from sync code, regardless of loop state.

Single shared helper used by every ``Sync*`` wrapper class shipped
alongside the async Oracle primitives (checkpointer / store / vector
store / loader / chunker / embedder). Mirrors the same trick
``langgraph-oracledb``'s ``OracleSaver`` uses to bridge sync callers
into an asyncpg-style driver — without taking any langchain or
langgraph dependency.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Coroutine
from typing import Any, TypeVar


T = TypeVar("T")


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Drive a coroutine to completion from synchronous context.

    Handles two cases:

      1. **No running loop** (script / Jupyter cell with ``await``
         disabled): use :func:`asyncio.run`.
      2. **Inside a running loop** (Jupyter ``await``, FastAPI
         handler, etc.): we can't reuse the active loop — instead we
         run the coroutine to completion on a *fresh* event loop
         hosted on a background thread, and block the calling thread
         until that completes.

    This is the same trick ``langgraph-oracledb``'s ``OracleSaver``
    uses to bridge sync callers into an asyncpg-style driver.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop in this thread — safe to spin one up.
        return asyncio.run(coro)

    import threading

    result: list[Any] = []
    error: list[BaseException] = []

    def runner() -> None:
        new_loop = asyncio.new_event_loop()
        try:
            result.append(new_loop.run_until_complete(coro))
        except BaseException as exc:  # noqa: BLE001 — re-raised in caller thread
            error.append(exc)
        finally:
            new_loop.close()

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join()
    if error:
        raise error[0]
    return result[0]  # type: ignore[no-any-return]


async def drain(agen: AsyncIterator[T]) -> list[T]:
    """Drain an async iterator into a list.

    Tiny helper used by every sync wrapper that needs to surface an
    async generator as a plain ``list[...]`` to its synchronous
    caller. Kept here so each wrapper module doesn't redefine it.
    """
    return [x async for x in agen]
