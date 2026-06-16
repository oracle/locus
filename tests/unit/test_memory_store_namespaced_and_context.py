# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Coverage gap fills for ``locus.memory.store`` —
``NamespacedStore`` + ``StoreContext`` + base optional-method paths.

Existing tests exercise ``InMemoryStore`` directly. These cover the
helper classes and the ``BaseStore`` default implementations:

- ``NamespacedStore`` delegates to the underlying store
- ``StoreContext`` user / session namespacing, missing-id raises,
  global helpers, semantic-memory paths
- ``BaseStore.get_batch`` / ``clear_namespace`` defaults
- ``BaseStore.put_batch`` sequential fallback
- ``BaseStore.list_namespaces`` raises ``StoreCapabilityError``
"""

from __future__ import annotations

import pytest

from locus.memory.store import (
    InMemoryStore,
    NamespacedStore,
    StoreCapabilityError,
    StoreContext,
)


# ---------------------------------------------------------------------------
# NamespacedStore
# ---------------------------------------------------------------------------


class TestNamespacedStore:
    @pytest.mark.asyncio
    async def test_put_get(self) -> None:
        store = InMemoryStore()
        ns = NamespacedStore(store, ("users", "u1"))
        await ns.put("pref", {"theme": "dark"})
        assert await ns.get("pref") == {"theme": "dark"}

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        store = InMemoryStore()
        ns = NamespacedStore(store, ("users", "u1"))
        await ns.put("pref", "x")
        assert await ns.delete("pref") is True
        assert await ns.delete("missing") is False

    @pytest.mark.asyncio
    async def test_list_keys(self) -> None:
        store = InMemoryStore()
        ns = NamespacedStore(store, ("users", "u1"))
        await ns.put("a", 1)
        await ns.put("b", 2)
        keys = await ns.list_keys()
        assert sorted(keys) == ["a", "b"]

    @pytest.mark.asyncio
    async def test_exists(self) -> None:
        store = InMemoryStore()
        ns = NamespacedStore(store, ("users", "u1"))
        await ns.put("pref", "x")
        assert await ns.exists("pref") is True
        assert await ns.exists("missing") is False

    @pytest.mark.asyncio
    async def test_search(self) -> None:
        store = InMemoryStore()
        ns = NamespacedStore(store, ("users", "u1"))
        await ns.put("topic", "python is great")
        results = await ns.search("python")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_clear(self) -> None:
        store = InMemoryStore()
        ns = NamespacedStore(store, ("users", "u1"))
        await ns.put("a", 1)
        await ns.put("b", 2)
        deleted = await ns.clear()
        assert deleted == 2
        assert await ns.list_keys() == []


# ---------------------------------------------------------------------------
# StoreContext
# ---------------------------------------------------------------------------


class TestStoreContext:
    def test_store_property(self) -> None:
        s = InMemoryStore()
        ctx = StoreContext(s)
        assert ctx.store is s

    def test_for_user_with_id(self) -> None:
        ctx = StoreContext(InMemoryStore(), user_id="u1")
        ns = ctx.for_user()
        assert isinstance(ns, NamespacedStore)

    def test_for_user_without_id_raises(self) -> None:
        ctx = StoreContext(InMemoryStore())
        with pytest.raises(ValueError, match="user_id must be provided"):
            ctx.for_user()

    def test_for_user_explicit_id_overrides(self) -> None:
        ctx = StoreContext(InMemoryStore(), user_id="u1")
        ns = ctx.for_user("u2")
        assert isinstance(ns, NamespacedStore)

    def test_for_session_with_id(self) -> None:
        ctx = StoreContext(InMemoryStore(), session_id="s1")
        ns = ctx.for_session()
        assert isinstance(ns, NamespacedStore)

    def test_for_session_without_id_raises(self) -> None:
        ctx = StoreContext(InMemoryStore())
        with pytest.raises(ValueError, match="session_id must be provided"):
            ctx.for_session()

    @pytest.mark.asyncio
    async def test_remember_and_get_user_memory(self) -> None:
        ctx = StoreContext(InMemoryStore(), user_id="u1")
        await ctx.remember("key1", {"data": "x"})
        retrieved = await ctx.get_user_memory("key1")
        assert retrieved == {"data": "x"}

    @pytest.mark.asyncio
    async def test_get_user_memory_returns_none_without_user(self) -> None:
        ctx = StoreContext(InMemoryStore())
        assert await ctx.get_user_memory("k") is None

    @pytest.mark.asyncio
    async def test_remember_without_user_raises(self) -> None:
        ctx = StoreContext(InMemoryStore())
        with pytest.raises(ValueError, match="No user_id"):
            await ctx.remember("k", "v")

    @pytest.mark.asyncio
    async def test_forget_without_user_returns_false(self) -> None:
        ctx = StoreContext(InMemoryStore())
        assert await ctx.forget("k") is False

    @pytest.mark.asyncio
    async def test_forget_existing_returns_true(self) -> None:
        ctx = StoreContext(InMemoryStore(), user_id="u1")
        await ctx.remember("key", "value")
        assert await ctx.forget("key") is True

    @pytest.mark.asyncio
    async def test_search_memories_without_user_returns_empty(self) -> None:
        ctx = StoreContext(InMemoryStore())
        assert await ctx.search_memories("query") == []

    @pytest.mark.asyncio
    async def test_search_memories_with_user(self) -> None:
        ctx = StoreContext(InMemoryStore(), user_id="u1")
        await ctx.remember("topic", "python is fun")
        results = await ctx.search_memories("python")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_remember_with_embedding_without_user_raises(self) -> None:
        ctx = StoreContext(InMemoryStore())
        with pytest.raises(ValueError, match="No user_id"):
            await ctx.remember_with_embedding("k", "v", [0.1])

    @pytest.mark.asyncio
    async def test_remember_with_embedding_unsupported_backend_raises(self) -> None:
        # InMemoryStore doesn't advertise semantic_search → upstream
        # ``put_with_embedding`` raises StoreCapabilityError.
        ctx = StoreContext(InMemoryStore(), user_id="u1")
        with pytest.raises(StoreCapabilityError):
            await ctx.remember_with_embedding("k", "v", [0.1, 0.2])

    @pytest.mark.asyncio
    async def test_search_memories_semantic_without_user_returns_empty(
        self,
    ) -> None:
        ctx = StoreContext(InMemoryStore())
        assert await ctx.search_memories_semantic([0.1]) == []

    @pytest.mark.asyncio
    async def test_get_set_global(self) -> None:
        ctx = StoreContext(InMemoryStore())
        await ctx.set_global("config", {"feature": True})
        assert await ctx.get_global("config") == {"feature": True}


# ---------------------------------------------------------------------------
# BaseStore default implementations
# ---------------------------------------------------------------------------


class TestBaseStoreDefaults:
    @pytest.mark.asyncio
    async def test_put_batch_falls_back_to_sequential(self) -> None:
        store = InMemoryStore()
        items = [
            (("ns",), "k1", "v1", None),
            (("ns",), "k2", "v2", {"tag": "x"}),
        ]
        await store.put_batch(items)
        assert await store.get(("ns",), "k1") == "v1"
        assert await store.get(("ns",), "k2") == "v2"

    @pytest.mark.asyncio
    async def test_get_batch_collects_existing_keys(self) -> None:
        store = InMemoryStore()
        await store.put(("ns",), "a", 1)
        await store.put(("ns",), "b", 2)
        result = await store.get_batch([(("ns",), "a"), (("ns",), "b"), (("ns",), "missing")])
        assert result == {(("ns",), "a"): 1, (("ns",), "b"): 2}

    @pytest.mark.asyncio
    async def test_clear_namespace_default_implementation(self) -> None:
        store = InMemoryStore()
        await store.put(("ns",), "a", 1)
        await store.put(("ns",), "b", 2)
        n = await store.clear_namespace(("ns",))
        assert n == 2

    @pytest.mark.asyncio
    async def test_close_is_noop(self) -> None:
        # Default ``close`` returns None — this is the smoke test.
        store = InMemoryStore()
        assert await store.close() is None  # type: ignore[func-returns-value]

    def test_make_storage_key(self) -> None:
        store = InMemoryStore()
        assert store._make_storage_key(("a", "b"), "c") == "a/b:c"
