# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for memory store."""

from datetime import UTC, datetime

import pytest

from locus.memory.store import (
    BaseStore,
    InMemoryStore,
    StoreCapabilities,
    StoreItem,
)


class TestStoreCapabilities:
    """Tests for StoreCapabilities."""

    def test_default_capabilities(self):
        """Test default capabilities."""
        caps = StoreCapabilities()
        assert caps.search is False
        assert caps.semantic_search is False
        assert caps.embedding_dimension is None
        assert caps.ttl is False
        assert caps.list_namespaces is False
        assert caps.batch_operations is False
        assert caps.transactions is False

    def test_custom_capabilities(self):
        """Test custom capabilities."""
        caps = StoreCapabilities(
            search=True,
            semantic_search=True,
            embedding_dimension=1536,
            ttl=True,
            list_namespaces=True,
        )
        assert caps.search is True
        assert caps.semantic_search is True
        assert caps.embedding_dimension == 1536
        assert caps.ttl is True


class TestStoreItem:
    """Tests for StoreItem."""

    def test_create_item(self):
        """Test creating a store item."""
        now = datetime.now(UTC)
        item = StoreItem(
            namespace=("users", "123"),
            key="preferences",
            value={"theme": "dark"},
            metadata={"source": "test"},
            created_at=now,
            updated_at=now,
            version=1,
        )
        assert item.namespace == ("users", "123")
        assert item.key == "preferences"
        assert item.value == {"theme": "dark"}
        assert item.version == 1

    def test_to_dict(self):
        """Test converting item to dict."""
        now = datetime.now(UTC)
        item = StoreItem(
            namespace=("users", "123"),
            key="key1",
            value="value1",
            metadata={},
            created_at=now,
            updated_at=now,
        )
        d = item.to_dict()
        assert d["namespace"] == ["users", "123"]
        assert d["key"] == "key1"
        assert d["value"] == "value1"
        assert "created_at" in d


class TestInMemoryStoreInit:
    """Tests for InMemoryStore initialization."""

    def test_create_store(self):
        """Test creating an in-memory store."""
        store = InMemoryStore()
        assert store._data == {}
        assert store._namespaces == set()

    def test_capabilities(self):
        """Test store capabilities."""
        store = InMemoryStore()
        caps = store.capabilities
        assert caps.search is True
        assert caps.list_namespaces is True
        assert caps.batch_operations is True
        assert caps.semantic_search is False


class TestInMemoryStorePut:
    """Tests for put operations."""

    @pytest.fixture
    def store(self):
        """Create store for testing."""
        return InMemoryStore()

    @pytest.mark.asyncio
    async def test_put_new_value(self, store):
        """Test putting a new value."""
        await store.put(("users",), "key1", "value1")
        value = await store.get(("users",), "key1")
        assert value == "value1"

    @pytest.mark.asyncio
    async def test_put_update_value(self, store):
        """Test updating an existing value."""
        await store.put(("users",), "key1", "value1")
        await store.put(("users",), "key1", "value2")
        value = await store.get(("users",), "key1")
        assert value == "value2"

    @pytest.mark.asyncio
    async def test_put_with_metadata(self, store):
        """Test putting with metadata."""
        await store.put(("users",), "key1", "value1", metadata={"source": "test"})
        item = await store.get_item(("users",), "key1")
        assert item is not None
        assert item.metadata == {"source": "test"}

    @pytest.mark.asyncio
    async def test_put_increments_version(self, store):
        """Test that version increments on update."""
        await store.put(("users",), "key1", "v1")
        item1 = await store.get_item(("users",), "key1")
        assert item1.version == 1

        await store.put(("users",), "key1", "v2")
        item2 = await store.get_item(("users",), "key1")
        assert item2.version == 2

    @pytest.mark.asyncio
    async def test_put_adds_namespace(self, store):
        """Test that put adds namespace to set."""
        await store.put(("users", "123"), "key1", "value1")
        assert ("users", "123") in store._namespaces


class TestInMemoryStoreGet:
    """Tests for get operations."""

    @pytest.fixture
    def store(self):
        """Create store for testing."""
        return InMemoryStore()

    @pytest.mark.asyncio
    async def test_get_existing(self, store):
        """Test getting an existing value."""
        await store.put(("ns",), "key1", {"data": "test"})
        value = await store.get(("ns",), "key1")
        assert value == {"data": "test"}

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store):
        """Test getting a nonexistent value."""
        value = await store.get(("ns",), "nonexistent")
        assert value is None

    @pytest.mark.asyncio
    async def test_get_item_existing(self, store):
        """Test getting full item."""
        await store.put(("ns",), "key1", "value1")
        item = await store.get_item(("ns",), "key1")
        assert item is not None
        assert item.key == "key1"
        assert item.value == "value1"

    @pytest.mark.asyncio
    async def test_get_item_nonexistent(self, store):
        """Test getting nonexistent item."""
        item = await store.get_item(("ns",), "nonexistent")
        assert item is None


class TestInMemoryStoreDelete:
    """Tests for delete operations."""

    @pytest.fixture
    def store(self):
        """Create store for testing."""
        return InMemoryStore()

    @pytest.mark.asyncio
    async def test_delete_existing(self, store):
        """Test deleting an existing value."""
        await store.put(("ns",), "key1", "value1")
        result = await store.delete(("ns",), "key1")
        assert result is True
        value = await store.get(("ns",), "key1")
        assert value is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store):
        """Test deleting a nonexistent value."""
        result = await store.delete(("ns",), "nonexistent")
        assert result is False


class TestInMemoryStoreListKeys:
    """Tests for list_keys operations."""

    @pytest.fixture
    def store(self):
        """Create store for testing."""
        return InMemoryStore()

    @pytest.mark.asyncio
    async def test_list_keys_empty(self, store):
        """Test listing keys from empty namespace."""
        keys = await store.list_keys(("ns",))
        assert keys == []

    @pytest.mark.asyncio
    async def test_list_keys(self, store):
        """Test listing keys."""
        await store.put(("ns",), "key1", "value1")
        await store.put(("ns",), "key2", "value2")
        await store.put(("other",), "key3", "value3")

        keys = await store.list_keys(("ns",))
        assert set(keys) == {"key1", "key2"}

    @pytest.mark.asyncio
    async def test_list_keys_with_limit(self, store):
        """Test listing keys with limit."""
        for i in range(10):
            await store.put(("ns",), f"key{i}", f"value{i}")

        keys = await store.list_keys(("ns",), limit=5)
        assert len(keys) == 5


class TestInMemoryStoreSearch:
    """Tests for search operations."""

    @pytest.fixture
    def store(self):
        """Create store for testing."""
        return InMemoryStore()

    @pytest.mark.asyncio
    async def test_search_by_value(self, store):
        """Test searching by value content."""
        await store.put(("ns",), "key1", "hello world")
        await store.put(("ns",), "key2", "goodbye world")
        await store.put(("ns",), "key3", "hello there")

        results = await store.search(("ns",), query="hello")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_returns_items_with_metadata(self, store):
        """Test that search returns items with metadata."""
        await store.put(("ns",), "key1", "hello world", metadata={"tag": "important"})
        await store.put(("ns",), "key2", "hello there", metadata={"tag": "normal"})

        results = await store.search(("ns",), query="hello")
        assert len(results) == 2
        # Check that metadata is preserved
        assert all(item.metadata for item in results)

    @pytest.mark.asyncio
    async def test_search_with_limit(self, store):
        """Test search with limit."""
        for i in range(10):
            await store.put(("ns",), f"key{i}", "searchable content")

        results = await store.search(("ns",), query="searchable", limit=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_search_empty_results(self, store):
        """Test search with no matches."""
        await store.put(("ns",), "key1", "hello")

        results = await store.search(("ns",), query="nonexistent")
        assert results == []


class TestInMemoryStoreListNamespaces:
    """Tests for list_namespaces operations."""

    @pytest.fixture
    def store(self):
        """Create store for testing."""
        return InMemoryStore()

    @pytest.mark.asyncio
    async def test_list_namespaces_empty(self, store):
        """Test listing namespaces when empty."""
        namespaces = await store.list_namespaces()
        assert namespaces == []

    @pytest.mark.asyncio
    async def test_list_namespaces(self, store):
        """Test listing namespaces."""
        await store.put(("users", "1"), "key", "value")
        await store.put(("users", "2"), "key", "value")
        await store.put(("sessions",), "key", "value")

        namespaces = await store.list_namespaces()
        assert len(namespaces) == 3


class TestInMemoryStoreClearNamespace:
    """Tests for clear_namespace operation."""

    @pytest.fixture
    def store(self):
        """Create store for testing."""
        return InMemoryStore()

    @pytest.mark.asyncio
    async def test_clear_namespace(self, store):
        """Test clearing a namespace."""
        await store.put(("ns1",), "key1", "value1")
        await store.put(("ns1",), "key2", "value2")
        await store.put(("ns2",), "key3", "value3")

        count = await store.clear_namespace(("ns1",))

        assert count == 2
        v1 = await store.get(("ns1",), "key1")
        v3 = await store.get(("ns2",), "key3")
        assert v1 is None
        assert v3 == "value3"

    @pytest.mark.asyncio
    async def test_clear_namespace_empty(self, store):
        """Test clearing an empty namespace."""
        count = await store.clear_namespace(("empty",))
        assert count == 0


class TestStoreItemFromDict:
    """Tests for StoreItem.from_dict."""

    def test_from_dict(self):
        """Test creating StoreItem from dict."""
        now = datetime.now(UTC)
        data = {
            "namespace": ["users", "123"],
            "key": "prefs",
            "value": {"theme": "dark"},
            "metadata": {"source": "test"},
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "version": 5,
        }
        item = StoreItem.from_dict(data)

        assert item.namespace == ("users", "123")
        assert item.key == "prefs"
        assert item.value == {"theme": "dark"}
        assert item.metadata == {"source": "test"}
        assert item.version == 5

    def test_from_dict_defaults(self):
        """Test from_dict with defaults."""
        now = datetime.now(UTC)
        data = {
            "namespace": ["ns"],
            "key": "k",
            "value": "v",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }
        item = StoreItem.from_dict(data)

        assert item.metadata == {}
        assert item.version == 1


class TestBaseStoreDefaultImplementations:
    """Tests for BaseStore default implementations."""

    @pytest.fixture
    def minimal_store(self):
        """Create a minimal concrete store implementation."""

        class MinimalStore(BaseStore):
            def __init__(self):
                self._data = {}

            async def put(self, namespace, key, value, metadata=None):
                self._data[(namespace, key)] = value

            async def get(self, namespace, key):
                return self._data.get((namespace, key))

            async def delete(self, namespace, key):
                if (namespace, key) in self._data:
                    del self._data[(namespace, key)]
                    return True
                return False

            async def list_keys(self, namespace, limit=100):
                return [k for ns, k in self._data if ns == namespace][:limit]

        return MinimalStore()

    @pytest.mark.asyncio
    async def test_exists_true(self, minimal_store):
        """Test exists returns True when value exists."""
        await minimal_store.put(("ns",), "key", "value")
        assert await minimal_store.exists(("ns",), "key") is True

    @pytest.mark.asyncio
    async def test_exists_false(self, minimal_store):
        """Test exists returns False when value doesn't exist."""
        assert await minimal_store.exists(("ns",), "nonexistent") is False

    @pytest.mark.asyncio
    async def test_get_item_default(self, minimal_store):
        """Test get_item default implementation."""
        await minimal_store.put(("ns",), "key", {"data": "value"})
        item = await minimal_store.get_item(("ns",), "key")

        assert item is not None
        assert item.namespace == ("ns",)
        assert item.key == "key"
        assert item.value == {"data": "value"}
        assert item.metadata == {}

    @pytest.mark.asyncio
    async def test_get_item_not_found(self, minimal_store):
        """Test get_item returns None when not found."""
        item = await minimal_store.get_item(("ns",), "nonexistent")
        assert item is None

    def test_capabilities_default(self, minimal_store):
        """Test default capabilities."""
        caps = minimal_store.capabilities
        assert caps.search is False
        assert caps.semantic_search is False

    @pytest.mark.asyncio
    async def test_search_without_capability(self, minimal_store):
        """Search on an unsupporting backend raises a typed capability error."""
        from locus.memory.store import StoreCapabilityError

        with pytest.raises(StoreCapabilityError) as exc_info:
            await minimal_store.search(("ns",), query="test")
        # Old code caught NotImplementedError — preserved via inheritance.
        assert isinstance(exc_info.value, NotImplementedError)
        assert exc_info.value.capability == "search"
        assert exc_info.value.store_class == "MinimalStore"

    @pytest.mark.asyncio
    async def test_put_with_embedding_without_capability(self, minimal_store):
        """put_with_embedding raises StoreCapabilityError."""
        from locus.memory.store import StoreCapabilityError

        with pytest.raises(StoreCapabilityError) as exc_info:
            await minimal_store.put_with_embedding(("ns",), "key", "value", [0.1, 0.2, 0.3])
        assert exc_info.value.capability == "semantic_search"
        assert isinstance(exc_info.value, NotImplementedError)

    @pytest.mark.asyncio
    async def test_search_by_embedding_without_capability(self, minimal_store):
        """search_by_embedding raises StoreCapabilityError."""
        from locus.memory.store import StoreCapabilityError

        with pytest.raises(StoreCapabilityError) as exc_info:
            await minimal_store.search_by_embedding(("ns",), [0.1, 0.2, 0.3])
        assert exc_info.value.capability == "semantic_search"

    @pytest.mark.asyncio
    async def test_get_embedding_without_capability(self, minimal_store):
        """get_embedding raises StoreCapabilityError."""
        from locus.memory.store import StoreCapabilityError

        with pytest.raises(StoreCapabilityError) as exc_info:
            await minimal_store.get_embedding(("ns",), "key")
        assert exc_info.value.capability == "semantic_search"


class TestStoreCapabilityError:
    """Direct tests for the StoreCapabilityError shape (#33)."""

    def test_carries_structured_payload(self):
        """The exception exposes capability/store_class/hint as attributes."""
        from locus.memory.store import StoreCapabilityError

        err = StoreCapabilityError(
            capability="semantic_search",
            store_class="StubStore",
            hint="Use a vector-capable backend.",
        )
        assert err.capability == "semantic_search"
        assert err.store_class == "StubStore"
        assert err.hint == "Use a vector-capable backend."
        assert "StubStore does not support 'semantic_search'" in str(err)
        assert "Use a vector-capable backend." in str(err)

    def test_subclasses_not_implemented_error_for_back_compat(self):
        """Legacy ``except NotImplementedError`` paths still catch it."""
        from locus.memory.store import StoreCapabilityError

        err = StoreCapabilityError(capability="search", store_class="StubStore")
        assert isinstance(err, NotImplementedError)

    def test_hint_is_optional(self):
        """Constructed without a hint, the message is just capability + class."""
        from locus.memory.store import StoreCapabilityError

        err = StoreCapabilityError(capability="search", store_class="StubStore")
        assert err.hint is None
        assert str(err) == "StubStore does not support 'search'"


class TestInMemoryStoreBatchOperations:
    """Tests for batch operations."""

    @pytest.fixture
    def store(self):
        """Create store for testing."""
        return InMemoryStore()

    @pytest.mark.asyncio
    async def test_put_batch(self, store):
        """Test batch put operation."""
        items = [
            (("ns",), "key1", "value1", None),
            (("ns",), "key2", "value2", {"meta": "data"}),
            (("ns",), "key3", "value3", None),
        ]
        await store.put_batch(items)

        v1 = await store.get(("ns",), "key1")
        v2 = await store.get(("ns",), "key2")
        v3 = await store.get(("ns",), "key3")

        assert v1 == "value1"
        assert v2 == "value2"
        assert v3 == "value3"

    @pytest.mark.asyncio
    async def test_get_batch(self, store):
        """Test batch get operation."""
        await store.put(("ns1",), "key1", "value1")
        await store.put(("ns2",), "key2", "value2")

        results = await store.get_batch(
            [
                (("ns1",), "key1"),
                (("ns2",), "key2"),
                (("ns3",), "nonexistent"),
            ]
        )

        assert results[(("ns1",), "key1")] == "value1"
        assert results[(("ns2",), "key2")] == "value2"
        # Nonexistent keys are not included in results
        assert (("ns3",), "nonexistent") not in results


class TestInMemoryStoreClose:
    """Tests for close operation."""

    @pytest.mark.asyncio
    async def test_close(self):
        """Test close doesn't raise error."""
        store = InMemoryStore()
        await store.put(("ns",), "key", "value")

        # close() is a no-op for InMemoryStore (inherits from BaseStore)
        await store.close()  # Should not raise

        # Data is still there (close is just for cleanup resources)
        value = await store.get(("ns",), "key")
        assert value == "value"


class TestInMemoryStoreSearchEdgeCases:
    """Edge case tests for search."""

    @pytest.mark.asyncio
    async def test_search_with_query_filter(self):
        """Test search with query substring filter."""
        store = InMemoryStore()
        await store.put(("ns",), "key1", {"text": "hello world"})
        await store.put(("ns",), "key2", {"text": "goodbye world"})
        await store.put(("ns",), "key3", {"text": "hello again"})

        # Search for items containing "hello"
        results = await store.search(("ns",), query="hello", limit=10)

        # Should find items containing "hello"
        keys = [r.key for r in results]
        assert "key1" in keys
        assert "key3" in keys

    @pytest.mark.asyncio
    async def test_search_query_in_metadata(self):
        """Test search finds query in metadata."""
        store = InMemoryStore()
        await store.put(("ns",), "key1", "value1", {"tag": "important"})
        await store.put(("ns",), "key2", "value2", {"tag": "normal"})

        results = await store.search(("ns",), query="important", limit=10)

        assert len(results) == 1
        assert results[0].key == "key1"

    @pytest.mark.asyncio
    async def test_search_respects_namespace_prefix(self):
        """Test search only returns items with matching namespace prefix."""
        store = InMemoryStore()
        await store.put(("ns1",), "key1", "value1")
        await store.put(("ns2",), "key2", "value2")

        results = await store.search(("ns1",), limit=10)

        # Should only find items in ns1
        assert len(results) == 1
        assert results[0].key == "key1"

    @pytest.mark.asyncio
    async def test_search_limit(self):
        """Test search respects limit."""
        store = InMemoryStore()
        for i in range(10):
            await store.put(("ns",), f"key{i}", f"value{i}")

        results = await store.search(("ns",), limit=3)

        assert len(results) == 3


class TestInMemoryStoreListNamespacesEdgeCases:
    """Edge case tests for list_namespaces."""

    @pytest.mark.asyncio
    async def test_list_namespaces_with_prefix_filter(self):
        """Test list_namespaces filters by prefix."""
        store = InMemoryStore()
        await store.put(("users", "123"), "data", "value")
        await store.put(("users", "456"), "data", "value")
        await store.put(("orders", "789"), "data", "value")

        results = await store.list_namespaces(prefix=("users",))

        assert len(results) == 2
        assert all(ns[0] == "users" for ns in results)

    @pytest.mark.asyncio
    async def test_list_namespaces_limit(self):
        """Test list_namespaces respects limit."""
        store = InMemoryStore()
        for i in range(10):
            await store.put((f"ns{i}",), "key", "value")

        results = await store.list_namespaces(limit=3)

        assert len(results) == 3


class TestInMemoryStorePutBatchEdgeCases:
    """Edge case tests for put_batch."""

    @pytest.mark.asyncio
    async def test_put_batch_update_existing(self):
        """Test put_batch updates existing items."""
        store = InMemoryStore()

        # Create initial item
        await store.put(("ns",), "key1", "original_value")

        # Batch update including existing key
        items = [
            (("ns",), "key1", "updated_value", None),
            (("ns",), "key2", "new_value", None),
        ]
        await store.put_batch(items)

        # Verify existing was updated
        value1 = await store.get(("ns",), "key1")
        assert value1 == "updated_value"

        # Verify new was created
        value2 = await store.get(("ns",), "key2")
        assert value2 == "new_value"


class TestSemanticSearchResult:
    """Tests for SemanticSearchResult."""

    def test_create_semantic_search_result(self):
        """Test creating SemanticSearchResult."""
        from locus.memory.store import SemanticSearchResult

        now = datetime.now(UTC)
        item = StoreItem(
            namespace=("ns",),
            key="key1",
            value="value1",
            metadata={},
            created_at=now,
            updated_at=now,
        )
        result = SemanticSearchResult(item=item, score=0.95)

        assert result.item == item
        assert result.score == 0.95
        assert result.distance is None

    def test_semantic_search_result_with_distance(self):
        """Test SemanticSearchResult with distance."""
        from locus.memory.store import SemanticSearchResult

        now = datetime.now(UTC)
        item = StoreItem(
            namespace=("ns",),
            key="key1",
            value="value1",
            metadata={},
            created_at=now,
            updated_at=now,
        )
        result = SemanticSearchResult(item=item, score=0.95, distance=0.05)

        assert result.distance == 0.05


class TestStoreItemConversion:
    """Tests for StoreItem conversion methods."""

    def test_store_item_from_dict(self):
        """Test StoreItem.from_dict."""
        now = datetime.now(UTC)
        d = {
            "namespace": ["ns", "sub"],
            "key": "key1",
            "value": {"data": "value"},
            "metadata": {"tag": "test"},
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "version": 2,
        }

        item = StoreItem.from_dict(d)

        assert item.namespace == ("ns", "sub")
        assert item.key == "key1"
        assert item.value == {"data": "value"}
        assert item.version == 2

    def test_store_item_from_dict_defaults(self):
        """Test StoreItem.from_dict with minimal data."""
        now = datetime.now(UTC)
        d = {
            "namespace": ["ns"],
            "key": "key1",
            "value": "value1",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }

        item = StoreItem.from_dict(d)

        assert item.namespace == ("ns",)
        assert item.metadata == {}
        assert item.version == 1
