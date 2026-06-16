# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for in-memory vector store."""

import math

import pytest

from locus.rag.stores.base import Document
from locus.rag.stores.memory import InMemoryVectorStore


class TestInMemoryVectorStoreInit:
    """Tests for InMemoryVectorStore initialization."""

    def test_default_init(self):
        """Test creating store with defaults."""
        store = InMemoryVectorStore()
        assert store._dimension == 1024
        assert store._distance_metric == "cosine"

    def test_custom_dimension(self):
        """Test creating store with custom dimension."""
        store = InMemoryVectorStore(dimension=512)
        assert store._dimension == 512

    def test_custom_distance_metric(self):
        """Test creating store with custom distance metric."""
        store = InMemoryVectorStore(distance_metric="euclidean")
        assert store._distance_metric == "euclidean"

    def test_config_property(self):
        """Test config property returns VectorStoreConfig."""
        store = InMemoryVectorStore(dimension=256, distance_metric="dot_product")
        config = store.config
        assert config.dimension == 256
        assert config.distance_metric == "dot_product"
        assert config.index_type == "flat"

    def test_repr(self):
        """Test string representation."""
        store = InMemoryVectorStore(dimension=128)
        assert "InMemoryVectorStore" in repr(store)
        assert "128" in repr(store)
        assert "count=0" in repr(store)


class TestCosineSimilarity:
    """Tests for cosine similarity calculation."""

    @pytest.fixture
    def store(self):
        """Create store for testing."""
        return InMemoryVectorStore()

    def test_identical_vectors(self, store):
        """Test cosine similarity of identical vectors."""
        v = [1.0, 2.0, 3.0]
        similarity = store._cosine_similarity(v, v)
        assert similarity == pytest.approx(1.0)

    def test_orthogonal_vectors(self, store):
        """Test cosine similarity of orthogonal vectors."""
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        similarity = store._cosine_similarity(v1, v2)
        assert similarity == pytest.approx(0.0)

    def test_opposite_vectors(self, store):
        """Test cosine similarity of opposite vectors."""
        v1 = [1.0, 1.0]
        v2 = [-1.0, -1.0]
        similarity = store._cosine_similarity(v1, v2)
        assert similarity == pytest.approx(-1.0)

    def test_zero_vector_a(self, store):
        """Test with zero vector a."""
        v1 = [0.0, 0.0]
        v2 = [1.0, 1.0]
        similarity = store._cosine_similarity(v1, v2)
        assert similarity == 0.0

    def test_zero_vector_b(self, store):
        """Test with zero vector b."""
        v1 = [1.0, 1.0]
        v2 = [0.0, 0.0]
        similarity = store._cosine_similarity(v1, v2)
        assert similarity == 0.0


class TestEuclideanDistance:
    """Tests for Euclidean distance calculation."""

    @pytest.fixture
    def store(self):
        """Create store for testing."""
        return InMemoryVectorStore(distance_metric="euclidean")

    def test_identical_vectors(self, store):
        """Test distance between identical vectors."""
        v = [1.0, 2.0, 3.0]
        distance = store._euclidean_distance(v, v)
        assert distance == 0.0

    def test_simple_distance(self, store):
        """Test simple 3-4-5 triangle distance."""
        v1 = [0.0, 0.0]
        v2 = [3.0, 4.0]
        distance = store._euclidean_distance(v1, v2)
        assert distance == 5.0

    def test_negative_values(self, store):
        """Test with negative values."""
        v1 = [-1.0, -1.0]
        v2 = [1.0, 1.0]
        distance = store._euclidean_distance(v1, v2)
        expected = math.sqrt(8)
        assert distance == pytest.approx(expected)


class TestDotProduct:
    """Tests for dot product calculation."""

    @pytest.fixture
    def store(self):
        """Create store for testing."""
        return InMemoryVectorStore(distance_metric="dot_product")

    def test_simple_dot_product(self, store):
        """Test simple dot product."""
        v1 = [1.0, 2.0, 3.0]
        v2 = [4.0, 5.0, 6.0]
        result = store._dot_product(v1, v2)
        assert result == 32.0  # 1*4 + 2*5 + 3*6

    def test_orthogonal_dot_product(self, store):
        """Test dot product of orthogonal vectors."""
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        result = store._dot_product(v1, v2)
        assert result == 0.0


class TestInMemoryVectorStoreAdd:
    """Tests for add operations."""

    @pytest.fixture
    def store(self):
        """Create store for testing."""
        return InMemoryVectorStore()

    @pytest.mark.asyncio
    async def test_add_document(self, store):
        """Test adding a document."""
        doc = Document(id="doc1", content="test", embedding=[0.1, 0.2, 0.3])
        doc_id = await store.add(doc)
        assert doc_id == "doc1"
        assert await store.count() == 1

    @pytest.mark.asyncio
    async def test_add_document_without_embedding_raises(self, store):
        """Test adding document without embedding raises error."""
        doc = Document(id="doc1", content="test")
        with pytest.raises(ValueError, match="must have an embedding"):
            await store.add(doc)

    @pytest.mark.asyncio
    async def test_add_batch(self, store):
        """Test adding multiple documents."""
        docs = [
            Document(id="doc1", content="test1", embedding=[0.1, 0.2]),
            Document(id="doc2", content="test2", embedding=[0.3, 0.4]),
        ]
        ids = await store.add_batch(docs)
        assert ids == ["doc1", "doc2"]
        assert await store.count() == 2


class TestInMemoryVectorStoreGet:
    """Tests for get operations."""

    @pytest.fixture
    def store(self):
        """Create store for testing."""
        return InMemoryVectorStore()

    @pytest.mark.asyncio
    async def test_get_existing_document(self, store):
        """Test getting an existing document."""
        doc = Document(id="doc1", content="test", embedding=[0.1, 0.2])
        await store.add(doc)

        result = await store.get("doc1")
        assert result is not None
        assert result.id == "doc1"
        assert result.content == "test"

    @pytest.mark.asyncio
    async def test_get_nonexistent_document(self, store):
        """Test getting a nonexistent document."""
        result = await store.get("nonexistent")
        assert result is None


class TestInMemoryVectorStoreDelete:
    """Tests for delete operations."""

    @pytest.fixture
    def store(self):
        """Create store for testing."""
        return InMemoryVectorStore()

    @pytest.mark.asyncio
    async def test_delete_existing_document(self, store):
        """Test deleting an existing document."""
        doc = Document(id="doc1", content="test", embedding=[0.1, 0.2])
        await store.add(doc)

        result = await store.delete("doc1")
        assert result is True
        assert await store.count() == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent_document(self, store):
        """Test deleting a nonexistent document."""
        result = await store.delete("nonexistent")
        assert result is False


class TestInMemoryVectorStoreSearch:
    """Tests for search operations."""

    @pytest.fixture
    def store(self):
        """Create store for testing."""
        return InMemoryVectorStore(dimension=3)

    @pytest.mark.asyncio
    async def test_search_cosine(self, store):
        """Test search with cosine similarity."""
        docs = [
            Document(id="doc1", content="close", embedding=[1.0, 0.0, 0.0]),
            Document(id="doc2", content="far", embedding=[0.0, 1.0, 0.0]),
        ]
        await store.add_batch(docs)

        results = await store.search([0.9, 0.1, 0.0], limit=2)

        assert len(results) == 2
        # First result should be doc1 (more similar to query)
        assert results[0].document.id == "doc1"
        assert results[0].score > results[1].score

    @pytest.mark.asyncio
    async def test_search_euclidean(self):
        """Test search with Euclidean distance."""
        store = InMemoryVectorStore(distance_metric="euclidean")
        docs = [
            Document(id="doc1", content="close", embedding=[1.0, 1.0]),
            Document(id="doc2", content="far", embedding=[10.0, 10.0]),
        ]
        await store.add_batch(docs)

        results = await store.search([0.0, 0.0], limit=2)

        # doc1 should be closer
        assert results[0].document.id == "doc1"

    @pytest.mark.asyncio
    async def test_search_dot_product(self):
        """Test search with dot product."""
        store = InMemoryVectorStore(distance_metric="dot_product")
        docs = [
            Document(id="doc1", content="high", embedding=[2.0, 2.0]),
            Document(id="doc2", content="low", embedding=[0.1, 0.1]),
        ]
        await store.add_batch(docs)

        results = await store.search([1.0, 1.0], limit=2)

        # doc1 should have higher dot product
        assert results[0].document.id == "doc1"

    @pytest.mark.asyncio
    async def test_search_with_limit(self, store):
        """Test search with limit."""
        docs = [
            Document(id=f"doc{i}", content=f"content{i}", embedding=[float(i), 0.0, 0.0])
            for i in range(5)
        ]
        await store.add_batch(docs)

        results = await store.search([1.0, 0.0, 0.0], limit=2)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_with_threshold(self, store):
        """Test search with threshold filter."""
        docs = [
            Document(id="doc1", content="close", embedding=[1.0, 0.0, 0.0]),
            Document(id="doc2", content="far", embedding=[0.0, 1.0, 0.0]),
        ]
        await store.add_batch(docs)

        # High threshold should filter out dissimilar docs
        results = await store.search([1.0, 0.0, 0.0], threshold=0.9)

        assert len(results) == 1
        assert results[0].document.id == "doc1"

    @pytest.mark.asyncio
    async def test_search_with_metadata_filter(self, store):
        """Test search with metadata filter."""
        docs = [
            Document(id="doc1", content="a", embedding=[1.0, 0.0, 0.0], metadata={"type": "a"}),
            Document(id="doc2", content="b", embedding=[1.0, 0.1, 0.0], metadata={"type": "b"}),
        ]
        await store.add_batch(docs)

        results = await store.search([1.0, 0.0, 0.0], metadata_filter={"type": "b"})

        assert len(results) == 1
        assert results[0].document.id == "doc2"

    @pytest.mark.asyncio
    async def test_search_skips_docs_without_embedding(self, store):
        """Test that search skips documents without embedding."""
        doc = Document(id="doc1", content="test", embedding=[1.0, 0.0, 0.0])
        await store.add(doc)

        # Manually add a doc without embedding (simulating edge case)
        store._documents["doc2"] = Document(id="doc2", content="test")

        results = await store.search([1.0, 0.0, 0.0])

        assert len(results) == 1
        assert results[0].document.id == "doc1"


class TestInMemoryVectorStoreClear:
    """Tests for clear operation."""

    @pytest.fixture
    def store(self):
        """Create store for testing."""
        return InMemoryVectorStore()

    @pytest.mark.asyncio
    async def test_clear(self, store):
        """Test clearing all documents."""
        docs = [
            Document(id="doc1", content="test1", embedding=[0.1, 0.2]),
            Document(id="doc2", content="test2", embedding=[0.3, 0.4]),
        ]
        await store.add_batch(docs)

        count = await store.clear()

        assert count == 2
        assert await store.count() == 0
