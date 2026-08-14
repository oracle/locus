# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""``RedisBackend``'s client cache, which used to outlive the loop it belonged to.

``redis.asyncio`` binds a client's connection pool to the event loop that
created it. Caching the client on the instance alone therefore worked exactly
once per process: the second loop inherited a dead pool and every call failed
with ``Event loop is closed``. Nothing exotic gets you there — FastAPI's
``TestClient`` runs each request through its own portal, and any code calling
``asyncio.run()`` twice hits it — and because the error surfaces from deep
inside ``redis.asyncio`` it reads as a Redis outage rather than a lifecycle bug
in this class.

These pin the three things the fix has to get right at once: a new loop gets a
new client, one loop still gets exactly one, and a client the caller supplied
is left alone.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from tulip.memory.backends.redis import RedisBackend


def _fake_client() -> AsyncMock:
    client = AsyncMock()
    client.set = AsyncMock()
    client.get = AsyncMock(return_value='{"ok": true}')
    client.close = AsyncMock()
    return client


@pytest.fixture
def built(monkeypatch: pytest.MonkeyPatch) -> list[AsyncMock]:
    """Record every client ``_get_client`` constructs, without touching Redis."""
    made: list[AsyncMock] = []

    class _FakeRedis:
        @staticmethod
        def from_url(*_a: Any, **_k: Any) -> AsyncMock:
            client = _fake_client()
            made.append(client)
            return client

    module = type(asyncio)("redis.asyncio")
    module.Redis = _FakeRedis  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "redis.asyncio", module)
    return made


def test_a_second_event_loop_gets_its_own_client(built: list[AsyncMock]) -> None:
    """The regression itself: two ``asyncio.run`` calls on one backend.

    Before the fix the second call reused a pool bound to the first, closed,
    loop and raised ``RuntimeError: Event loop is closed``.
    """
    backend = RedisBackend()

    asyncio.run(backend.save("t1", {"n": 1}))
    asyncio.run(backend.save("t1", {"n": 2}))

    assert len(built) == 2, "the client must be rebuilt when the loop changes"
    built[0].set.assert_awaited_once()
    built[1].set.assert_awaited_once()


def test_one_loop_still_builds_exactly_one_client(built: list[AsyncMock]) -> None:
    """Every real deployment is this case, and it must stay as cheap as before."""
    backend = RedisBackend()

    async def _many() -> None:
        for i in range(5):
            await backend.save(f"t{i}", {"n": i})

    asyncio.run(_many())

    assert len(built) == 1


@pytest.mark.asyncio
async def test_a_caller_supplied_client_is_never_discarded(built: list[AsyncMock]) -> None:
    """Assigning ``_client`` means the caller owns its lifecycle.

    An unrecorded loop has to mean "not ours to manage" rather than "stale",
    or the backend throws away an injected client and reaches for real Redis —
    which is how a naive loop check breaks every test that mocks this out.
    """
    backend = RedisBackend()
    injected = _fake_client()
    backend._client = injected

    await backend.save("t1", {"n": 1})

    assert built == [], "no client should have been constructed"
    injected.set.assert_awaited_once()


@pytest.mark.asyncio
async def test_closing_clears_the_loop_it_was_bound_to(built: list[AsyncMock]) -> None:
    """A stale loop recorded against no client would be a lie about the next one."""
    backend = RedisBackend()
    await backend.save("t1", {"n": 1})
    assert backend._client_loop is not None

    await backend.close()

    assert backend._client is None
    assert backend._client_loop is None

    await backend.save("t1", {"n": 2})
    assert len(built) == 2
