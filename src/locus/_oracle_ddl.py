# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""First-time DDL that several processes may run at once, for every Oracle backend.

Each Oracle backend creates its table and indexes the first time an
instance is used: probe ``user_tables``, run the ``CREATE`` statements,
commit. Two things go wrong when that happens concurrently. Within one
process, every coroutine (or thread) that arrives before the flag is set
runs the DDL, so a burst of twenty first saves is twenty ``CREATE TABLE``
statements. Between processes, the loser of the race meets ``ORA-00955``
(the object exists), or a lock collision with the winner's DDL or first
writes (``ORA-00054`` resource busy, ``ORA-04020`` library-cache
deadlock). The PostgreSQL backend serialises its own first-time DDL with
an advisory lock; Oracle's DDL is its own arbiter, so here the loser's
``CREATE`` reads as created and a collision is retried briefly.

:func:`provisioning_lock` gives one lock per instance and event loop for
the coroutines of a process; sync backends hold a :class:`threading.Lock`
of their own. :func:`run_ddl` and :func:`run_ddl_sync` run one statement
against the other process.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from locus.core.loop_bound import loop_bound


__all__ = [
    "DDL_ATTEMPTS",
    "already_exists",
    "ddl_contended",
    "provisioning_lock",
    "run_ddl",
    "run_ddl_sync",
]

#: Attempts at a first-time DDL statement another session is contending for.
DDL_ATTEMPTS = 5

#: Seconds waited before the second attempt; each later wait grows by this much.
_BACKOFF_STEP = 0.2


def already_exists(exc: BaseException) -> bool:
    """Whether ``exc`` is ``ORA-00955``: another session created the object first.

    Two pools provisioning one table at the same instant both see it
    missing and both run the DDL; the loser's ``CREATE`` fails with this
    error, and the table (or index) is there.
    """
    return "ORA-00955" in str(exc)


def ddl_contended(exc: BaseException) -> bool:
    """Whether ``exc`` is a lock collision with another session on the object.

    ``ORA-00054`` (resource busy) and ``ORA-04020`` (deadlock on the object's
    library-cache lock) come from two processes provisioning one table at
    the same instant while one of them already writes to it; a short wait
    lets the other side commit.
    """
    message = str(exc)
    return "ORA-00054" in message or "ORA-04020" in message


def provisioning_lock(owner: Any) -> asyncio.Lock:
    """The lock ``owner``'s first-time DDL runs under, one per event loop.

    Kept on ``owner._provision_lock`` through :func:`~locus.core.loop_bound.loop_bound`,
    so a backend used from several loops in turn (``run_sync`` builds one per
    call) gets a lock belonging to the loop that is running. Pydantic models
    declare the two attributes it uses, ``_provision_lock`` and
    ``_provision_lock_loop``; plain classes need nothing.
    """
    return loop_bound(owner, "_provision_lock", asyncio.Lock)


def _should_retry(exc: BaseException, attempt: int) -> bool:
    return ddl_contended(exc) and attempt < DDL_ATTEMPTS


async def run_ddl(cursor: Any, statement: str) -> None:
    """Run one first-time DDL statement on an async cursor against other sessions doing the same.

    A ``CREATE`` another session won is done (``ORA-00955``); a lock
    collision with that session waits a little and tries again, up to
    :data:`DDL_ATTEMPTS` times; anything else propagates.
    """
    for attempt in range(1, DDL_ATTEMPTS + 1):
        try:
            await cursor.execute(statement)
        except Exception as exc:
            if already_exists(exc):
                return
            if _should_retry(exc, attempt):
                await asyncio.sleep(_BACKOFF_STEP * attempt)
                continue
            raise
        return


def run_ddl_sync(cursor: Any, statement: str) -> None:
    """:func:`run_ddl` for a blocking cursor (the audit trail and the report store)."""
    for attempt in range(1, DDL_ATTEMPTS + 1):
        try:
            cursor.execute(statement)
        except Exception as exc:
            if already_exists(exc):
                return
            if _should_retry(exc, attempt):
                time.sleep(_BACKOFF_STEP * attempt)
                continue
            raise
        return
