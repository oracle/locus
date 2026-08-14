# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Async backend handles must belong to the loop that uses them.

redis.asyncio and aiohttp bind their pools to the creating loop, and asyncpg
pools are explicitly single-loop. A handle cached on the instance alone is
therefore only valid inside one loop; the second loop gets a dead one and fails
with 'Event loop is closed' (or, for asyncpg, 'another operation is in
progress').

These tests drive the caching decision directly with a stub factory, so they
need no live service: what matters is *how many times* a new handle is built
and *when*.

RedisBackend is covered on its own; these are the two remaining async backends
that shared the same defect.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from tulip.memory.backends import OpenSearchBackend, PostgreSQLBackend


class TestOpenSearchLoopAffinity:
    @staticmethod
    def _patch(monkeypatch: pytest.MonkeyPatch, built: list[Any]) -> None:
        import opensearchpy._async.client as mod

        def _factory(*a: Any, **k: Any) -> Any:
            built.append(MagicMock())
            return built[-1]

        monkeypatch.setattr(mod, "AsyncOpenSearch", _factory)

    def test_a_new_loop_builds_a_new_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        built: list[Any] = []
        self._patch(monkeypatch, built)
        backend = OpenSearchBackend(hosts=["stub:9200"])

        async def once() -> Any:
            return await backend._get_client()

        assert asyncio.run(once()) is not asyncio.run(once())
        assert len(built) == 2

    def test_the_client_is_reused_inside_one_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        built: list[Any] = []
        self._patch(monkeypatch, built)
        backend = OpenSearchBackend(hosts=["stub:9200"])

        async def twice() -> tuple[Any, Any]:
            return await backend._get_client(), await backend._get_client()

        first, second = asyncio.run(twice())

        assert first is second
        assert len(built) == 1

    def test_an_injected_client_is_left_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import opensearchpy._async.client as mod

        def _boom(*a: Any, **k: Any) -> Any:
            raise AssertionError("an injected client must not be replaced")

        monkeypatch.setattr(mod, "AsyncOpenSearch", _boom)
        backend = OpenSearchBackend(hosts=["stub:9200"])
        injected = MagicMock()
        backend._client = injected

        assert asyncio.run(backend._get_client()) is injected


class TestPostgreSQLLoopAffinity:
    @staticmethod
    def _patch(monkeypatch: pytest.MonkeyPatch, built: list[Any]) -> None:
        import asyncpg

        async def _create_pool(*a: Any, **k: Any) -> Any:
            built.append(MagicMock())
            return built[-1]

        monkeypatch.setattr(asyncpg, "create_pool", _create_pool)

    def test_a_new_loop_builds_a_new_pool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        built: list[Any] = []
        self._patch(monkeypatch, built)
        backend = PostgreSQLBackend(host="stub", database="d", user="u", password="p")  # noqa: S106 — stub credentials; no server is contacted

        async def once() -> Any:
            return await backend._get_pool()

        assert asyncio.run(once()) is not asyncio.run(once())
        assert len(built) == 2

    def test_the_pool_is_reused_inside_one_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        built: list[Any] = []
        self._patch(monkeypatch, built)
        backend = PostgreSQLBackend(host="stub", database="d", user="u", password="p")  # noqa: S106 — stub credentials; no server is contacted

        async def twice() -> tuple[Any, Any]:
            return await backend._get_pool(), await backend._get_pool()

        first, second = asyncio.run(twice())

        assert first is second
        assert len(built) == 1

    def test_a_dsn_pool_is_also_loop_keyed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The dsn branch builds the pool separately and must key it too."""
        built: list[Any] = []
        self._patch(monkeypatch, built)
        backend = PostgreSQLBackend(dsn="postgresql://u:p@stub:5432/d")

        async def once() -> Any:
            return await backend._get_pool()

        assert asyncio.run(once()) is not asyncio.run(once())
        assert len(built) == 2

    def test_an_injected_pool_is_left_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import asyncpg

        async def _boom(*a: Any, **k: Any) -> Any:
            raise AssertionError("an injected pool must not be replaced")

        monkeypatch.setattr(asyncpg, "create_pool", _boom)
        backend = PostgreSQLBackend(host="stub", database="d", user="u", password="p")  # noqa: S106 — stub credentials; no server is contacted
        injected = MagicMock()
        backend._pool = injected

        assert asyncio.run(backend._get_pool()) is injected
