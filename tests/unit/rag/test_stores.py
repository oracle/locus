# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for vector stores."""

import pytest

from locus.rag.stores.base import Document, SearchResult, VectorStoreConfig
from locus.rag.stores.memory import InMemoryVectorStore


class TestDocument:
    """Tests for Document dataclass."""

    def test_create_document(self):
        """Test creating a document."""
        doc = Document(
            id="doc1",
            content="Hello world",
            embedding=[0.1, 0.2, 0.3],
            metadata={"source": "test"},
        )

        assert doc.id == "doc1"
        assert doc.content == "Hello world"
        assert doc.embedding == [0.1, 0.2, 0.3]
        assert doc.metadata == {"source": "test"}
        assert doc.content_type == "text"
        assert doc.raw_content is None

    def test_document_with_raw_content(self):
        """Test document with multimodal content."""
        raw = b"binary data"
        doc = Document(
            id="img1",
            content="Image description",
            embedding=[0.5, 0.5],
            content_type="image",
            raw_content=raw,
        )

        assert doc.content_type == "image"
        assert doc.raw_content == raw

    def test_to_dict(self):
        """Test document serialization."""
        doc = Document(
            id="doc1",
            content="Test",
            embedding=[0.1, 0.2],
            metadata={"key": "value"},
            content_type="pdf",
            raw_content=b"pdf bytes",
        )

        data = doc.to_dict()

        assert data["id"] == "doc1"
        assert data["content"] == "Test"
        assert data["content_type"] == "pdf"
        assert "raw_content" in data  # Base64 encoded

    def test_from_dict(self):
        """Test document deserialization."""
        import base64

        data = {
            "id": "doc1",
            "content": "Test",
            "embedding": [0.1, 0.2],
            "metadata": {"key": "value"},
            "created_at": "2024-01-01T00:00:00+00:00",
            "content_type": "audio",
            "raw_content": base64.b64encode(b"audio bytes").decode(),
        }

        doc = Document.from_dict(data)

        assert doc.id == "doc1"
        assert doc.content_type == "audio"
        assert doc.raw_content == b"audio bytes"


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_create_result(self):
        """Test creating a search result."""
        doc = Document(id="doc1", content="Test")
        result = SearchResult(document=doc, score=0.95, distance=0.05)

        assert result.document.id == "doc1"
        assert result.score == 0.95
        assert result.distance == 0.05


class TestInMemoryVectorStore:
    """Tests for in-memory vector store."""

    @pytest.fixture
    def store(self):
        """Create a test store."""
        return InMemoryVectorStore(dimension=4)

    @pytest.mark.asyncio
    async def test_add_document(self, store):
        """Test adding a document."""
        doc = Document(
            id="doc1",
            content="Hello world",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        doc_id = await store.add(doc)

        assert doc_id == "doc1"
        assert await store.count() == 1

    @pytest.mark.asyncio
    async def test_add_without_embedding_fails(self, store):
        """Test that adding without embedding fails."""
        doc = Document(id="doc1", content="No embedding")

        with pytest.raises(ValueError, match="must have an embedding"):
            await store.add(doc)

    @pytest.mark.asyncio
    async def test_get_document(self, store):
        """Test retrieving a document."""
        doc = Document(
            id="doc1",
            content="Test content",
            embedding=[0.1, 0.2, 0.3, 0.4],
            metadata={"key": "value"},
        )
        await store.add(doc)

        retrieved = await store.get("doc1")

        assert retrieved is not None
        assert retrieved.id == "doc1"
        assert retrieved.content == "Test content"
        assert retrieved.metadata == {"key": "value"}

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store):
        """Test getting a nonexistent document."""
        result = await store.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_document(self, store):
        """Test deleting a document."""
        doc = Document(id="doc1", content="Test", embedding=[0.1, 0.2, 0.3, 0.4])
        await store.add(doc)

        deleted = await store.delete("doc1")

        assert deleted is True
        assert await store.get("doc1") is None
        assert await store.count() == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store):
        """Test deleting a nonexistent document."""
        deleted = await store.delete("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_search_cosine(self, store):
        """Test cosine similarity search."""
        # Add documents with known embeddings
        await store.add(
            Document(
                id="doc1",
                content="Similar",
                embedding=[1.0, 0.0, 0.0, 0.0],
            )
        )
        await store.add(
            Document(
                id="doc2",
                content="Different",
                embedding=[0.0, 1.0, 0.0, 0.0],
            )
        )
        await store.add(
            Document(
                id="doc3",
                content="Also similar",
                embedding=[0.9, 0.1, 0.0, 0.0],
            )
        )

        # Search with query similar to doc1 and doc3
        results = await store.search(
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            limit=2,
        )

        assert len(results) == 2
        assert results[0].document.id == "doc1"  # Exact match
        assert results[0].score == pytest.approx(1.0, abs=0.01)
        assert results[1].document.id == "doc3"  # Similar

    @pytest.mark.asyncio
    async def test_search_with_threshold(self, store):
        """Test search with similarity threshold."""
        await store.add(
            Document(
                id="doc1",
                content="Match",
                embedding=[1.0, 0.0, 0.0, 0.0],
            )
        )
        await store.add(
            Document(
                id="doc2",
                content="No match",
                embedding=[0.0, 1.0, 0.0, 0.0],
            )
        )

        results = await store.search(
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            threshold=0.9,
        )

        assert len(results) == 1
        assert results[0].document.id == "doc1"

    @pytest.mark.asyncio
    async def test_search_with_metadata_filter(self, store):
        """Test search with metadata filtering."""
        await store.add(
            Document(
                id="doc1",
                content="Python doc",
                embedding=[1.0, 0.0, 0.0, 0.0],
                metadata={"language": "python"},
            )
        )
        await store.add(
            Document(
                id="doc2",
                content="Java doc",
                embedding=[0.9, 0.1, 0.0, 0.0],
                metadata={"language": "java"},
            )
        )

        results = await store.search(
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            metadata_filter={"language": "java"},
        )

        assert len(results) == 1
        assert results[0].document.id == "doc2"

    @pytest.mark.asyncio
    async def test_add_batch(self, store):
        """Test batch document addition."""
        docs = [
            Document(id=f"doc{i}", content=f"Content {i}", embedding=[0.1 * i, 0.2, 0.3, 0.4])
            for i in range(5)
        ]

        ids = await store.add_batch(docs)

        assert len(ids) == 5
        assert await store.count() == 5

    @pytest.mark.asyncio
    async def test_clear(self, store):
        """Test clearing all documents."""
        for i in range(3):
            await store.add(
                Document(
                    id=f"doc{i}",
                    content=f"Content {i}",
                    embedding=[0.1 * i, 0.2, 0.3, 0.4],
                )
            )

        count = await store.clear()

        assert count == 3
        assert await store.count() == 0

    def test_config(self, store):
        """Test store configuration."""
        config = store.config

        assert config.dimension == 4
        assert config.distance_metric == "cosine"
        assert config.index_type == "flat"


class TestVectorStoreConfig:
    """Tests for VectorStoreConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = VectorStoreConfig(dimension=1024)

        assert config.dimension == 1024
        assert config.distance_metric == "cosine"
        assert config.index_type == "hnsw"

    def test_custom_config(self):
        """Test custom configuration."""
        config = VectorStoreConfig(
            dimension=384,
            distance_metric="l2",
            index_type="ivf",
        )

        assert config.dimension == 384
        assert config.distance_metric == "l2"
        assert config.index_type == "ivf"
