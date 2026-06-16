# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for cross-thread Store module."""

import pytest

from locus.memory.store import (
    InMemoryStore,
    NamespacedStore,
    StoreCapabilities,
    StoreContext,
    StoreItem,
)


class TestStoreCapabilities:
    """Tests for StoreCapabilities."""

    def test_default_capabilities(self):
        """Test default capabilities are False."""
        caps = StoreCapabilities()
        assert not caps.search
        assert not caps.ttl
        assert not caps.list_namespaces
        assert not caps.batch_operations
        assert not caps.transactions


class TestStoreItem:
    """Tests for StoreItem."""

    def test_to_dict(self):
        """Test to_dict method."""
        from datetime import UTC, datetime

        now = datetime.now(UTC)
        item = StoreItem(
            namespace=("users", "alice"),
            key="preferences",
            value={"theme": "dark"},
            metadata={"updated_by": "system"},
            created_at=now,
            updated_at=now,
            version=1,
        )
        d = item.to_dict()
        assert d["namespace"] == ["users", "alice"]
        assert d["key"] == "preferences"
        assert d["value"] == {"theme": "dark"}
        assert d["version"] == 1

    def test_from_dict(self):
        """Test from_dict method."""
        d = {
            "namespace": ["users", "alice"],
            "key": "preferences",
            "value": {"theme": "dark"},
            "metadata": {},
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
            "version": 2,
        }
        item = StoreItem.from_dict(d)
        assert item.namespace == ("users", "alice")
        assert item.key == "preferences"
        assert item.version == 2


class TestInMemoryStore:
    """Tests for InMemoryStore."""

    @pytest.fixture
    def store(self):
        """Create a fresh store for each test."""
        return InMemoryStore()

    @pytest.mark.asyncio
    async def test_put_and_get(self, store):
        """Test basic put and get."""
        await store.put(("users", "alice"), "name", "Alice")
        result = await store.get(("users", "alice"), "name")
        assert result == "Alice"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store):
        """Test get returns None for nonexistent key."""
        result = await store.get(("users", "alice"), "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_existing(self, store):
        """Test updating existing key."""
        await store.put(("users", "alice"), "count", 1)
        await store.put(("users", "alice"), "count", 2)
        result = await store.get(("users", "alice"), "count")
        assert result == 2

    @pytest.mark.asyncio
    async def test_version_increments(self, store):
        """Test version increments on update."""
        await store.put(("users", "alice"), "data", "v1")
        item1 = await store.get_item(("users", "alice"), "data")
        assert item1.version == 1

        await store.put(("users", "alice"), "data", "v2")
        item2 = await store.get_item(("users", "alice"), "data")
        assert item2.version == 2

    @pytest.mark.asyncio
    async def test_delete(self, store):
        """Test delete operation."""
        await store.put(("users", "alice"), "name", "Alice")
        deleted = await store.delete(("users", "alice"), "name")
        assert deleted

        result = await store.get(("users", "alice"), "name")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store):
        """Test delete returns False for nonexistent key."""
        deleted = await store.delete(("users", "alice"), "nonexistent")
        assert not deleted

    @pytest.mark.asyncio
    async def test_list_keys(self, store):
        """Test list_keys operation."""
        await store.put(("users", "alice"), "name", "Alice")
        await store.put(("users", "alice"), "age", 30)
        await store.put(("users", "bob"), "name", "Bob")

        keys = await store.list_keys(("users", "alice"))
        assert sorted(keys) == ["age", "name"]

    @pytest.mark.asyncio
    async def test_list_keys_empty_namespace(self, store):
        """Test list_keys with empty namespace."""
        keys = await store.list_keys(("users", "nonexistent"))
        assert keys == []

    @pytest.mark.asyncio
    async def test_exists(self, store):
        """Test exists operation."""
        await store.put(("users", "alice"), "name", "Alice")
        assert await store.exists(("users", "alice"), "name")
        assert not await store.exists(("users", "alice"), "nonexistent")

    @pytest.mark.asyncio
    async def test_search(self, store):
        """Test search operation."""
        await store.put(("docs",), "doc1", {"title": "Python Guide"})
        await store.put(("docs",), "doc2", {"title": "Java Guide"})
        await store.put(("docs",), "doc3", {"title": "Python Basics"})

        results = await store.search(("docs",), query="Python")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_limit(self, store):
        """Test search with limit."""
        for i in range(10):
            await store.put(("docs",), f"doc{i}", {"content": "test data"})

        results = await store.search(("docs",), query="test", limit=5)
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_list_namespaces(self, store):
        """Test list_namespaces operation."""
        await store.put(("users", "alice"), "name", "Alice")
        await store.put(("users", "bob"), "name", "Bob")
        await store.put(("settings",), "theme", "dark")

        namespaces = await store.list_namespaces()
        assert ("users", "alice") in namespaces
        assert ("users", "bob") in namespaces
        assert ("settings",) in namespaces

    @pytest.mark.asyncio
    async def test_clear_namespace(self, store):
        """Test clear_namespace operation."""
        await store.put(("users", "alice"), "name", "Alice")
        await store.put(("users", "alice"), "age", 30)
        await store.put(("users", "bob"), "name", "Bob")

        count = await store.clear_namespace(("users", "alice"))
        assert count == 2

        assert await store.get(("users", "alice"), "name") is None
        assert await store.get(("users", "bob"), "name") == "Bob"

    @pytest.mark.asyncio
    async def test_put_batch(self, store):
        """Test batch put operation."""
        items = [
            (("users", "alice"), "name", "Alice", None),
            (("users", "alice"), "age", 30, None),
            (("users", "bob"), "name", "Bob", None),
        ]
        await store.put_batch(items)

        assert await store.get(("users", "alice"), "name") == "Alice"
        assert await store.get(("users", "alice"), "age") == 30
        assert await store.get(("users", "bob"), "name") == "Bob"

    @pytest.mark.asyncio
    async def test_capabilities(self, store):
        """Test InMemoryStore capabilities."""
        caps = store.capabilities
        assert caps.search
        assert caps.list_namespaces
        assert caps.batch_operations
        assert not caps.ttl


class TestNamespacedStore:
    """Tests for NamespacedStore wrapper."""

    @pytest.fixture
    def store(self):
        return InMemoryStore()

    @pytest.fixture
    def user_store(self, store):
        return NamespacedStore(store, ("users", "alice"))

    @pytest.mark.asyncio
    async def test_put_and_get(self, user_store):
        """Test put and get through namespaced store."""
        await user_store.put("name", "Alice")
        result = await user_store.get("name")
        assert result == "Alice"

    @pytest.mark.asyncio
    async def test_scoped(self, user_store):
        """Test scoped method creates extended namespace."""
        memories = user_store.scoped("memories")
        assert memories.namespace == ("users", "alice", "memories")

        await memories.put("topic", "python")
        result = await memories.get("topic")
        assert result == "python"

    @pytest.mark.asyncio
    async def test_list_keys(self, user_store):
        """Test list_keys through namespaced store."""
        await user_store.put("name", "Alice")
        await user_store.put("age", 30)
        keys = await user_store.list_keys()
        assert sorted(keys) == ["age", "name"]

    @pytest.mark.asyncio
    async def test_delete(self, user_store):
        """Test delete through namespaced store."""
        await user_store.put("name", "Alice")
        deleted = await user_store.delete("name")
        assert deleted
        assert await user_store.get("name") is None

    @pytest.mark.asyncio
    async def test_clear(self, user_store):
        """Test clear through namespaced store."""
        await user_store.put("name", "Alice")
        await user_store.put("age", 30)
        count = await user_store.clear()
        assert count == 2


class TestStoreContext:
    """Tests for StoreContext."""

    @pytest.fixture
    def store(self):
        return InMemoryStore()

    @pytest.fixture
    def context(self, store):
        return StoreContext(store, user_id="alice", session_id="session1")

    @pytest.mark.asyncio
    async def test_for_user(self, context):
        """Test for_user method."""
        user_store = context.for_user("bob")
        assert user_store.namespace == ("users", "bob")

    @pytest.mark.asyncio
    async def test_for_session(self, context):
        """Test for_session method."""
        session_store = context.for_session("sess123")
        assert session_store.namespace == ("sessions", "sess123")

    @pytest.mark.asyncio
    async def test_remember_and_get_user_memory(self, context):
        """Test remember and get_user_memory."""
        await context.remember("favorite_color", "blue")
        result = await context.get_user_memory("favorite_color")
        assert result == "blue"

    @pytest.mark.asyncio
    async def test_forget(self, context):
        """Test forget method."""
        await context.remember("temp_data", "value")
        deleted = await context.forget("temp_data")
        assert deleted
        assert await context.get_user_memory("temp_data") is None

    @pytest.mark.asyncio
    async def test_global_operations(self, context):
        """Test get_global and set_global."""
        await context.set_global("config", {"debug": True})
        result = await context.get_global("config")
        assert result == {"debug": True}

    @pytest.mark.asyncio
    async def test_search_memories(self, context):
        """Test search_memories."""
        await context.remember("topic1", {"topic": "python programming"})
        await context.remember("topic2", {"topic": "java development"})

        results = await context.search_memories("python")
        assert len(results) >= 1
