# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Coverage tests for ``tulip.memory.backends.adapters``.

The existing ``test_storage_adapters.py`` covers happy-path save/load.
This file targets the remaining gaps:

- Adapter ``save()`` with backends whose ``save`` signature accepts a
  ``metadata=`` kwarg (the inspect-based branch).
- ``_remove_from_index`` early-return when the index is missing.
- ``query_by_metadata`` falling back from ``query_by_metadata`` →
  ``get_by_metadata`` and the final ``NotImplementedError``.
- ``get_metadata`` native + fallback paths (latest, by id, missing).
- ``copy_thread`` happy path.
- ``list_threads`` signature variations: ``pattern=`` arg, ``limit=`` arg,
  no-arg, plus client-side pattern filter when backend doesn't support it.
- ``list_with_metadata`` and ``close`` delegation.
- Each factory function (``redis``, ``postgresql``, ``mysql``,
  ``opensearch``) — uses ``sys.modules`` stubs so we don't need the
  real SDK clients.
"""

from __future__ import annotations

import asyncio
import sys
import types
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tulip.memory.backends.adapters import StorageBackendAdapter, _callable_has_parameter


# ---------------------------------------------------------------------------
# Fake state object
# ---------------------------------------------------------------------------


class _FakeState:
    """Minimal stand-in for AgentState with to_checkpoint / from_checkpoint."""

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload or {"messages": []}

    def to_checkpoint(self) -> dict[str, Any]:
        return dict(self.payload)


# ---------------------------------------------------------------------------
# Save with metadata-aware backend
# ---------------------------------------------------------------------------


class _MetaAwareBackend:
    """Backend whose ``save`` signature includes a ``metadata=`` kwarg."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], dict[str, Any] | None]] = []

    async def save(
        self,
        key: str,
        data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.calls.append((key, data, metadata))

    async def load(self, key: str) -> dict[str, Any] | None:
        return None

    async def delete(self, key: str) -> bool:
        return True

    async def exists(self, key: str) -> bool:
        return False


class TestSaveMetadataKwargBranch:
    @pytest.mark.asyncio
    async def test_save_passes_metadata_when_supported(self) -> None:
        backend = _MetaAwareBackend()
        adapter = StorageBackendAdapter(backend)
        await adapter.save(_FakeState(), thread_id="t1", metadata={"tag": "x"})
        # First call is the actual checkpoint save (with metadata kwarg).
        assert backend.calls
        first_key, _first_data, first_meta = backend.calls[0]
        assert first_key.startswith("t1:")
        assert first_meta == {"tag": "x"}


class TestCallableHasParameter:
    def test_uninspectable_callable_returns_false(self) -> None:
        assert _callable_has_parameter(42, "metadata") is False


class TestUpdateCheckpointIndex:
    @pytest.mark.asyncio
    async def test_existing_index_replaces_duplicate_and_sorts_newest_first(self) -> None:
        backend = MagicMock()
        backend.load = AsyncMock(
            return_value={
                "checkpoints": [
                    {"checkpoint_id": "cp-old", "timestamp": "2026-01-01T00:00:00+00:00"},
                    {"checkpoint_id": "cp-1", "timestamp": "2026-01-02T00:00:00+00:00"},
                ]
            }
        )
        backend.save = AsyncMock()
        adapter = StorageBackendAdapter(backend)

        await adapter._update_checkpoint_index(
            "t1",
            "cp-1",
            timestamp=datetime(2026, 1, 3, tzinfo=UTC),
        )

        saved_index = backend.save.await_args.args[1]
        assert [cp["checkpoint_id"] for cp in saved_index["checkpoints"]] == [
            "cp-1",
            "cp-old",
        ]


# ---------------------------------------------------------------------------
# _remove_from_index — early return when index missing
# ---------------------------------------------------------------------------


class TestRemoveFromIndexNoExisting:
    @pytest.mark.asyncio
    async def test_no_existing_index_is_noop(self) -> None:
        backend = MagicMock()
        backend.load = AsyncMock(return_value=None)
        backend.save = AsyncMock()
        adapter = StorageBackendAdapter(backend)
        # Call private method directly — should not crash and should not
        # call save.
        await adapter._remove_from_index("t1", "cp1")
        backend.save.assert_not_called()


# ---------------------------------------------------------------------------
# query_by_metadata — fallback to get_by_metadata + NotImplementedError
# ---------------------------------------------------------------------------


class TestQueryByMetadata:
    @pytest.mark.asyncio
    async def test_uses_query_by_metadata_when_available(self) -> None:
        backend = MagicMock()
        backend.query_by_metadata = AsyncMock(return_value=[{"k": 1}])
        adapter = StorageBackendAdapter(backend)
        out = await adapter.query_by_metadata("k", "v", limit=5)
        assert out == [{"k": 1}]
        backend.query_by_metadata.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_falls_back_to_get_by_metadata(self) -> None:
        # Build a backend without query_by_metadata; capabilities check
        # (in StorageBackendAdapter) accepts get_by_metadata as
        # metadata_query support.
        class _Backend:
            async def get_by_metadata(
                self, k: str, v: Any, *, limit: int = 100
            ) -> list[dict[str, Any]]:
                return [{"hit": True}]

        adapter = StorageBackendAdapter(_Backend())
        out = await adapter.query_by_metadata("k", "v", limit=2)
        assert out == [{"hit": True}]

    @pytest.mark.asyncio
    async def test_final_not_implemented_branch(self) -> None:
        # Backend "supports" metadata_query because it has get_metadata
        # (which makes ``capabilities.metadata_query`` True), but lacks
        # both query_by_metadata and get_by_metadata.
        class _Backend:
            async def get_metadata(self, key: str) -> dict[str, Any] | None:
                return None

        adapter = StorageBackendAdapter(_Backend())
        with pytest.raises(NotImplementedError):
            await adapter.query_by_metadata("k", "v")


# ---------------------------------------------------------------------------
# get_metadata — native + index fallback
# ---------------------------------------------------------------------------


class TestGetMetadata:
    @pytest.mark.asyncio
    async def test_native_get_metadata_used(self) -> None:
        backend = MagicMock()
        backend.get_metadata = AsyncMock(return_value={"native": True})
        adapter = StorageBackendAdapter(backend)
        out = await adapter.get_metadata("t1", "cp1")
        assert out == {"native": True}

    @pytest.mark.asyncio
    async def test_index_fallback_returns_latest(self) -> None:
        # No get_metadata on backend; adapter falls back to the index.
        idx = {"checkpoints": [{"checkpoint_id": "cp2"}, {"checkpoint_id": "cp1"}]}

        class _Backend:
            async def load(self, key: str) -> dict[str, Any] | None:
                return idx if key.endswith("_checkpoints") else None

        adapter = StorageBackendAdapter(_Backend())
        latest = await adapter.get_metadata("t1")
        assert latest == {"checkpoint_id": "cp2"}

    @pytest.mark.asyncio
    async def test_index_fallback_specific_checkpoint(self) -> None:
        idx = {"checkpoints": [{"checkpoint_id": "cp2"}, {"checkpoint_id": "cp1"}]}

        class _Backend:
            async def load(self, key: str) -> dict[str, Any] | None:
                return idx if key.endswith("_checkpoints") else None

        adapter = StorageBackendAdapter(_Backend())
        m = await adapter.get_metadata("t1", checkpoint_id="cp1")
        assert m == {"checkpoint_id": "cp1"}

    @pytest.mark.asyncio
    async def test_index_fallback_missing_checkpoint(self) -> None:
        idx = {"checkpoints": [{"checkpoint_id": "cp2"}]}

        class _Backend:
            async def load(self, key: str) -> dict[str, Any] | None:
                return idx if key.endswith("_checkpoints") else None

        adapter = StorageBackendAdapter(_Backend())
        assert await adapter.get_metadata("t1", checkpoint_id="missing") is None

    @pytest.mark.asyncio
    async def test_index_fallback_no_index(self) -> None:
        class _Backend:
            async def load(self, key: str) -> dict[str, Any] | None:
                return None

        adapter = StorageBackendAdapter(_Backend())
        assert await adapter.get_metadata("t1") is None


# ---------------------------------------------------------------------------
# copy_thread happy path
# ---------------------------------------------------------------------------


class _BranchingBackend:
    """Stores everything in an in-memory dict and has ``copy_thread``."""

    def __init__(self) -> None:
        self.store: dict[str, dict[str, Any]] = {}

    async def save(self, key: str, data: dict[str, Any]) -> None:
        self.store[key] = dict(data)

    async def load(self, key: str) -> dict[str, Any] | None:
        return dict(self.store[key]) if key in self.store else None

    async def delete(self, key: str) -> bool:
        return self.store.pop(key, None) is not None

    async def exists(self, key: str) -> bool:
        return key in self.store

    async def copy_thread(self, src: str, dst: str) -> bool:
        # Presence triggers ``branching=True`` capability; the adapter
        # uses its own logic, so this method itself doesn't need to do
        # anything.
        return True


class TestCopyThread:
    @pytest.mark.asyncio
    async def test_copy_thread_clones_checkpoints(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = _BranchingBackend()
        adapter = StorageBackendAdapter(backend)

        # Patch AgentState.from_checkpoint so we don't depend on the real
        # state-shape contract.
        from tulip.core import state as state_mod

        class _MiniState:
            def __init__(self, payload: dict[str, Any]) -> None:
                self.payload = payload

            def to_checkpoint(self) -> dict[str, Any]:
                return dict(self.payload)

            @classmethod
            def from_checkpoint(cls, data: dict[str, Any]) -> _MiniState:
                return cls(data)

        monkeypatch.setattr(state_mod, "AgentState", _MiniState)
        # Save one checkpoint on source
        await adapter.save(_MiniState({"msg": "hi"}), thread_id="src", checkpoint_id="cp1")
        ok = await adapter.copy_thread("src", "dst")
        assert ok is True
        # Latest pointer for dst should now exist.
        assert await backend.exists("dst:latest")

    @pytest.mark.asyncio
    async def test_copy_thread_no_source(self) -> None:
        backend = _BranchingBackend()
        adapter = StorageBackendAdapter(backend)
        ok = await adapter.copy_thread("nonexistent", "dst")
        assert ok is False

    @pytest.mark.asyncio
    async def test_copy_thread_skips_missing_checkpoint_state(self) -> None:
        backend = MagicMock()
        backend.copy_thread = AsyncMock(return_value=True)
        backend.load = AsyncMock(
            side_effect=lambda key: (
                {"checkpoints": [{"checkpoint_id": "cp1"}]}
                if key.endswith("_checkpoints")
                else None
            )
        )
        adapter = StorageBackendAdapter(backend)

        ok = await adapter.copy_thread("src", "dst")

        assert ok is True
        backend.save.assert_not_called()


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_all_keeps_deleted_false_when_checkpoint_delete_misses(self) -> None:
        backend = MagicMock()
        backend.load = AsyncMock(return_value={"checkpoints": [{"checkpoint_id": "cp1"}]})
        backend.delete = AsyncMock(return_value=False)
        backend.exists = AsyncMock(return_value=False)
        adapter = StorageBackendAdapter(backend)

        assert await adapter.delete("t1") is False


# ---------------------------------------------------------------------------
# list_threads variations
# ---------------------------------------------------------------------------


class TestListThreads:
    @pytest.mark.asyncio
    async def test_pattern_kwarg_supported(self) -> None:
        class _Backend:
            async def list_threads(self, pattern: str = "*", limit: int = 100) -> list[str]:
                return ["a", "b"]

        adapter = StorageBackendAdapter(_Backend())
        out = await adapter.list_threads(pattern="a*", limit=5)
        assert out == ["a", "b"]

    @pytest.mark.asyncio
    async def test_limit_kwarg_only_with_client_side_filter(self) -> None:
        class _Backend:
            async def list_threads(self, limit: int = 100) -> list[str]:
                return ["abc", "def", "ace"]

        adapter = StorageBackendAdapter(_Backend())
        out = await adapter.list_threads(pattern="a*", limit=10)
        assert "abc" in out
        assert "ace" in out
        assert "def" not in out

    @pytest.mark.asyncio
    async def test_no_kwargs_branch(self) -> None:
        class _Backend:
            async def list_threads(self) -> list[str]:
                return ["x", "y", "z"]

        adapter = StorageBackendAdapter(_Backend())
        out = await adapter.list_threads(limit=2)
        assert len(out) == 2

    @pytest.mark.asyncio
    async def test_no_list_threads_method_raises(self) -> None:
        class _Backend:
            pass

        adapter = StorageBackendAdapter(_Backend())
        # capabilities.list_threads is False → _require_capability raises
        # NotImplementedError; the exact subclass is incidental.
        with pytest.raises(NotImplementedError):
            await adapter.list_threads()


# ---------------------------------------------------------------------------
# list_with_metadata + close delegation
# ---------------------------------------------------------------------------


class TestListWithMetadataAndClose:
    @pytest.mark.asyncio
    async def test_list_with_metadata_delegates(self) -> None:
        class _Backend:
            async def list_with_metadata(self, *, limit: int = 100) -> list[dict[str, Any]]:
                return [{"thread": "a"}]

        adapter = StorageBackendAdapter(_Backend())
        out = await adapter.list_with_metadata(limit=10)
        assert out == [{"thread": "a"}]

    @pytest.mark.asyncio
    async def test_close_invokes_backend_close(self) -> None:
        backend = MagicMock()
        backend.close = AsyncMock()
        adapter = StorageBackendAdapter(backend)
        await adapter.close()
        backend.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_when_backend_has_no_close(self) -> None:
        # No ``close`` attribute on the backend — adapter must noop.
        class _Backend:
            pass

        adapter = StorageBackendAdapter(_Backend())
        await adapter.close()  # must not raise

    def test_repr_contains_backend(self) -> None:
        backend = MagicMock()
        adapter = StorageBackendAdapter(backend)
        assert "StorageBackendAdapter" in repr(adapter)


# ---------------------------------------------------------------------------
# Factory functions — stub the SDK imports so the factory bodies execute
# ---------------------------------------------------------------------------


def _stub_module(name: str, **attrs: Any) -> types.ModuleType:
    """Build and register a stub module on ``sys.modules``."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


@pytest.fixture
def fake_redis_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    class FakeRedisBackend:
        def __init__(self, **kw: Any) -> None:
            self.kwargs = kw

    _stub_module("tulip.memory.backends.redis", RedisBackend=FakeRedisBackend)
    yield FakeRedisBackend
    sys.modules.pop("tulip.memory.backends.redis", None)


@pytest.fixture
def fake_pg_module() -> Any:
    class FakePGBackend:
        def __init__(self, **kw: Any) -> None:
            self.kwargs = kw

    _stub_module("tulip.memory.backends.postgresql", PostgreSQLBackend=FakePGBackend)
    yield FakePGBackend
    sys.modules.pop("tulip.memory.backends.postgresql", None)


@pytest.fixture
def fake_mysql_module() -> Any:
    class FakeMySQLBackend:
        def __init__(self, **kw: Any) -> None:
            self.kwargs = kw

    _stub_module("tulip.memory.backends.mysql", MySQLBackend=FakeMySQLBackend)
    yield FakeMySQLBackend
    sys.modules.pop("tulip.memory.backends.mysql", None)


@pytest.fixture
def fake_os_module() -> Any:
    class FakeOSBackend:
        def __init__(self, **kw: Any) -> None:
            self.kwargs = kw

    _stub_module("tulip.memory.backends.opensearch", OpenSearchBackend=FakeOSBackend)
    yield FakeOSBackend
    sys.modules.pop("tulip.memory.backends.opensearch", None)


class TestFactoryFunctions:
    def test_redis_checkpointer(self, fake_redis_module: Any) -> None:
        from tulip.memory.backends.adapters import redis_checkpointer

        cp = redis_checkpointer(url="redis://x:1", prefix="p:")
        assert isinstance(cp, StorageBackendAdapter)
        assert isinstance(cp._backend, fake_redis_module)

    def test_postgresql_checkpointer(self, fake_pg_module: Any) -> None:
        from tulip.memory.backends.adapters import postgresql_checkpointer

        cp = postgresql_checkpointer(database="db", user="u", password="p")  # noqa: S106
        assert isinstance(cp, StorageBackendAdapter)
        assert isinstance(cp._backend, fake_pg_module)

    def test_mysql_checkpointer(self, fake_mysql_module: Any) -> None:
        from tulip.memory.backends.adapters import mysql_checkpointer

        cp = mysql_checkpointer(database="db", user="u", password="p")  # noqa: S106
        assert isinstance(cp, StorageBackendAdapter)
        assert isinstance(cp._backend, fake_mysql_module)

    def test_opensearch_checkpointer(self, fake_os_module: Any) -> None:
        from tulip.memory.backends.adapters import opensearch_checkpointer

        cp = opensearch_checkpointer(hosts=["http://x:9200"])
        assert isinstance(cp, StorageBackendAdapter)
        assert isinstance(cp._backend, fake_os_module)


# ---------------------------------------------------------------------------
# Concurrency — checkpoint index must not lose entries (regression for #297)
# ---------------------------------------------------------------------------


class _YieldingBackend:
    """In-memory backend that yields on every op, forcing realistic interleaving
    (stands in for a real DB's await points) so index read-modify-write races
    surface deterministically without a live database."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    async def save(self, key: str, data: dict[str, Any]) -> None:
        await asyncio.sleep(0)
        self._data[key] = data

    async def load(self, key: str) -> dict[str, Any] | None:
        await asyncio.sleep(0)
        return self._data.get(key)

    async def delete(self, key: str) -> bool:
        await asyncio.sleep(0)
        return self._data.pop(key, None) is not None

    async def exists(self, key: str) -> bool:
        await asyncio.sleep(0)
        return key in self._data


class TestConcurrentIndexUpdates:
    @pytest.mark.asyncio
    async def test_concurrent_same_thread_saves_keep_all_checkpoints(self) -> None:
        """Concurrent saves to one thread must not drop index entries (#297)."""
        adapter = StorageBackendAdapter(_YieldingBackend())
        n = 20
        await asyncio.gather(
            *(
                adapter.save(_FakeState(), thread_id="hot", checkpoint_id=f"cp-{i}")
                for i in range(n)
            )
        )
        listed = await adapter.list_checkpoints("hot", limit=1000)
        assert len(set(listed)) == n

    @pytest.mark.asyncio
    async def test_concurrent_removes_keep_index_consistent(self) -> None:
        """Concurrent removals of distinct checkpoints leave the rest intact (#297)."""
        adapter = StorageBackendAdapter(_YieldingBackend())
        n = 20
        for i in range(n):
            await adapter.save(_FakeState(), thread_id="hot", checkpoint_id=f"cp-{i}")
        await asyncio.gather(
            *(adapter._remove_from_index("hot", f"cp-{i}") for i in range(0, n, 2))
        )
        listed = set(await adapter.list_checkpoints("hot", limit=1000))
        assert listed == {f"cp-{i}" for i in range(1, n, 2)}
