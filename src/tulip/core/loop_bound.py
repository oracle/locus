# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Caching an async client that belongs to the event loop that built it.

Async HTTP clients and connection pools bind to the loop running when they are
created — `httpx`, and therefore `openai` and `anthropic`; `redis.asyncio`;
`asyncpg`; `aiomysql`; `opensearch-py`. Cache one on `self` and it works
exactly once per process. The second loop inherits a pool whose sockets belong
to a loop that is closed, and the call fails.

How it fails is the expensive part. Redis says ``Event loop is closed``, which
at least points somewhere. The OpenAI client raises::

    APIConnectionError: Connection error.

which reads as "the provider is down" — so the first thing anyone does is
check their network, their key, and the provider's status page. Nothing in
that message suggests the client was reused across loops.

None of this is exotic. Two ``asyncio.run()`` calls in a script, a notebook
cell run twice, FastAPI's ``TestClient`` (a fresh portal per request), or two
``pytest`` tests that each start a loop all reach it.

This was fixed three times by hand — in the Redis, OpenSearch and PostgreSQL
checkpointer backends — each with its own spelling of the cache key, before it
turned out to be live in ten more places including both native model clients.
Three hand-copies of one idea is how the copies drift, so it lives here now
and the call sites share it.

Usage::

    async def _get_client(self) -> AsyncOpenAI:
        return loop_bound(self, "_client", self._build_client)

An externally assigned ``self._client`` is left alone. Tests inject a double
that way, and a cache that threw it out would send the test at a real network.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar


__all__ = ["loop_bound", "loop_bound_async", "loop_key_for"]

T = TypeVar("T")


def loop_key_for(attr: str) -> str:
    """Where the loop is recorded for a resource cached at ``attr``."""
    return f"{attr}_loop"


def loop_bound(owner: Any, attr: str, factory: Callable[[], T]) -> T:
    """Return ``owner.<attr>``, rebuilt when the running event loop changed.

    Args:
        owner: The object holding the cache.
        attr: Attribute the resource is cached on, e.g. ``"_client"``.
        factory: Builds a fresh resource. Called only when there is nothing
            usable cached, so an expensive or optional import can live inside
            it and cost nothing on the common path.

    Returns:
        A resource belonging to the loop that is running now.

    The previous resource belonged to a loop that is gone, so the reference is
    dropped rather than closed: closing it would need that loop, which is
    precisely what is no longer available. Callers that need deterministic
    teardown should close explicitly while their loop is still running.
    """
    try:
        loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
    except RuntimeError:
        # Reached from synchronous code — several of these caches sit behind a
        # plain property. With no loop running there is nothing to compare
        # against, so hand back whatever is cached and record nothing: the
        # first async use will bind it.
        loop = None
    key = loop_key_for(attr)
    cached = getattr(owner, attr, None)
    recorded = getattr(owner, key, None)

    # ``recorded is None`` means the resource did not come from here — a caller
    # or a test assigned it — so its lifecycle is not ours to manage.
    stale = loop is not None and recorded is not None and recorded is not loop
    if cached is None or stale:
        cached = factory()
        setattr(owner, attr, cached)
        setattr(owner, key, loop)
    return cached


async def loop_bound_async(owner: Any, attr: str, factory: Callable[[], Awaitable[T]]) -> T:
    """:func:`loop_bound` for a resource that has to be awaited into existence.

    Connection pools mostly are: ``asyncpg.create_pool`` and friends open
    sockets before they hand anything back, which is also why they are bound to
    the loop that did the opening.

    Always called from inside a loop, so unlike :func:`loop_bound` there is no
    no-loop fallback to make — awaiting outside one is impossible.
    """
    loop = asyncio.get_running_loop()
    key = loop_key_for(attr)
    cached = getattr(owner, attr, None)
    recorded = getattr(owner, key, None)

    stale = recorded is not None and recorded is not loop
    if cached is None or stale:
        cached = await factory()
        setattr(owner, attr, cached)
        setattr(owner, key, loop)
    return cached
