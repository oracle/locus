# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Per-event-loop ``oracledb`` async pool cache.

``oracledb`` async pools are bound to the asyncio event loop that
creates them. The moment a caller invokes :meth:`Agent.run_sync` a
second time (each call spins a fresh loop via :func:`asyncio.run`),
or FastAPI's lifespan / anyio's BlockingPortal opens a different
loop, the cached pool is attached to a dead loop and the next
``acquire`` raises::

    RuntimeError: Task ... got Future attached to a different loop

The fix is to detect a loop mismatch and rebuild. This helper does
that for every Oracle-touching backend in locus so the per-class
copies stay one line.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol


class _PoolHolder(Protocol):
    """Duck-typed surface every Oracle backend already exposes.

    ``_pool`` is the cached pool; ``_pool_loop`` is the event loop it
    was created on. Backends declare these as instance attributes (or
    Pydantic private attrs) and call :func:`get_pool` to read /
    refresh.
    """

    _pool: Any
    _pool_loop: Any


async def get_pool(
    holder: _PoolHolder,
    builder: Callable[[], Any] | Callable[[], Awaitable[Any]],
) -> Any:
    """Return a pool bound to the current event loop, building if needed.

    Args:
        holder: any object with mutable ``_pool`` / ``_pool_loop``
            attributes. Most locus backends already satisfy this.
        builder: zero-arg callable that produces a fresh
            ``oracledb`` async pool. May be sync (the common case —
            ``oracledb.create_pool_async`` returns the pool directly,
            not a coroutine) or async.

    The cached pool is invalidated when the running loop changes;
    the dropped pool is left for ``oracledb`` to GC.
    """
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:  # pragma: no cover — backend methods are async
        running = None

    if holder._pool is not None and holder._pool_loop is not running:
        # Cross-loop call — drop the dead reference, reinitialise table
        # provisioning state, and rebuild below.
        holder._pool = None
        # ``_initialized`` is a sibling flag on most backends that gates
        # CREATE TABLE checks; reset it so the new pool re-verifies
        # against the database.
        if hasattr(holder, "_initialized"):
            holder._initialized = False

    if holder._pool is None:
        pool = builder()
        if asyncio.iscoroutine(pool):
            pool = await pool
        holder._pool = pool
        holder._pool_loop = running

    return holder._pool


# Oracle network-layer errors that mean the underlying TCP connection
# is dead — Autonomous Database aggressively closes idle sockets, and
# the pool's cached connections may have been killed since the last
# use. The fix is the same in every case: drop the pool, rebuild on the
# next call, and retry the operation once.
_RECONNECT_PREFIXES: tuple[str, ...] = (
    "DPY-4011",  # the database or network closed the connection
    "DPY-1001",  # not connected to the database
    "DPY-6000",  # cannot connect to database (broken connection)
    "ORA-12537",  # TNS:connection closed
    "ORA-03113",  # end-of-file on communication channel
    "ORA-03114",  # not connected to ORACLE
    "ORA-03138",  # connection terminated due to security policy violation
    "ORA-03146",  # invalid buffer length for TTC field (protocol desync)
    "ORA-12541",  # TNS:no listener
    "ORA-12170",  # TNS:Connect timeout occurred
)


def is_reconnectable(exc: BaseException) -> bool:
    """True when *exc* names a network-layer disconnect we can recover from.

    Matches against the error string rather than the exception class
    because ``oracledb.DatabaseError`` wraps every database-side
    condition; the ORA / DPY prefix tells reconnectable apart from
    application-level errors (ORA-00001 etc.) we should not retry.
    """
    msg = str(exc)
    return any(p in msg for p in _RECONNECT_PREFIXES)


async def with_reconnect(
    holder: _PoolHolder,
    op: Callable[[Any], Awaitable[Any]],
    builder: Callable[[], Any] | Callable[[], Awaitable[Any]],
) -> Any:
    """Run *op(pool)* with a one-shot rebuild on a network-layer error.

    The retry path drops the pool and any sibling ``_initialized`` flag
    so the second attempt walks the full provisioning path again. We
    retry exactly once — a persistent disconnect (DB shutdown, wrong
    DSN) should surface to the caller rather than loop.
    """
    pool = await get_pool(holder, builder)
    try:
        return await op(pool)
    except BaseException as exc:  # noqa: BLE001
        if not is_reconnectable(exc):
            raise
        # Drop the dead pool + sibling provisioning state so the next
        # ``get_pool`` rebuilds cleanly. Don't await ``pool.close()`` —
        # the broken socket would just raise again.
        holder._pool = None
        if hasattr(holder, "_initialized"):
            holder._initialized = False
        pool = await get_pool(holder, builder)
        return await op(pool)


@contextlib.asynccontextmanager
async def safe_acquire(holder: _PoolHolder, pool: Any) -> AsyncIterator[Any]:
    """Drop-in replacement for ``async with pool.acquire() as conn``.

    Behaves like ``pool.acquire()`` for the body of the block — yields a
    connection, no surprises. The wrapper exists for the *close* side:
    Autonomous Database aggressively closes idle TCP connections, so a
    pooled connection's socket may be dead by the time we're done with
    it. When that close-time handshake fails with DPY-4011 (or a sibling
    network-layer error) we:

    1. Swallow the exception — the work inside the block already
       committed, surfacing a close-time disconnect would mask a
       successful save with a false error.
    2. Invalidate the pool by setting ``holder._pool = None`` so the
       next operation rebuilds it. Stale connections in the pool
       likely share the same fate.

    Errors raised *inside* the block propagate untouched — pair with
    :func:`with_reconnect` if you also need body-time retry.

    Implementation note: ``pool.acquire()`` in ``oracledb`` returns the
    connection object directly (synchronously) and the connection
    itself implements the async context-manager protocol. We drive
    ``__aenter__`` / ``__aexit__`` manually so close-time errors can
    be caught separately from body errors.
    """
    ctx = pool.acquire()
    conn = await ctx.__aenter__()
    body_exc: BaseException | None = None
    try:
        yield conn
    except BaseException as exc:
        body_exc = exc
        raise
    finally:
        try:
            if body_exc is None:
                await ctx.__aexit__(None, None, None)
            else:
                await ctx.__aexit__(type(body_exc), body_exc, body_exc.__traceback__)
        except BaseException as close_exc:  # noqa: BLE001
            if is_reconnectable(close_exc):
                # Socket's dead — drop the pool so the next call rebuilds.
                # Suppress the close-time error: the body either succeeded
                # already (no point surfacing a false failure) or has its
                # own ``body_exc`` that's about to propagate.
                holder._pool = None
                if hasattr(holder, "_initialized"):
                    holder._initialized = False
            elif body_exc is None:
                # Real close-time failure with no body error to propagate.
                raise
            # else: body_exc is about to propagate; don't mask it.
