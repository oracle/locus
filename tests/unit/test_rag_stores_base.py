# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for RAG stores base classes."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from locus.rag.stores.base import (
    BaseVectorStore,
    Document,
    SearchResult,
    VectorStore,
    VectorStoreConfig,
)


class TestDocument:
    """Tests for Document dataclass."""

    def test_create_minimal(self):
        """Test creating document with minimal fields."""
        doc = Document(id="doc1", content="Hello world")
        assert doc.id == "doc1"
        assert doc.content == "Hello world"
        assert doc.embedding is None
        assert doc.metadata == {}
        assert doc.content_type == "text"
        assert doc.raw_content is None

    def test_create_full(self):
        """Test creating document with all fields."""
        created = datetime.now(UTC)
        doc = Document(
            id="doc1",
            content="Hello",
            embedding=[0.1, 0.2, 0.3],
            metadata={"source": "test"},
            created_at=created,
            content_type="pdf",
            raw_content=b"raw data",
        )
        assert doc.embedding == [0.1, 0.2, 0.3]
        assert doc.metadata == {"source": "test"}
        assert doc.created_at == created
        assert doc.content_type == "pdf"
        assert doc.raw_content == b"raw data"

    def test_to_dict_minimal(self):
        """Test converting minimal document to dict."""
        doc = Document(id="doc1", content="Hello")
        d = doc.to_dict()

        assert d["id"] == "doc1"
        assert d["content"] == "Hello"
        assert d["embedding"] is None
        assert d["metadata"] == {}
        assert "created_at" in d
        assert d["content_type"] == "text"
        assert "raw_content" not in d

    def test_to_dict_with_raw_content(self):
        """Test to_dict includes base64 encoded raw content."""
        doc = Document(
            id="doc1",
            content="Hello",
            raw_content=b"binary data",
        )
        d = doc.to_dict()

        assert "raw_content" in d
        import base64

        assert d["raw_content"] == base64.b64encode(b"binary data").decode()

    def test_from_dict_minimal(self):
        """Test creating document from minimal dict."""
        data = {"id": "doc1", "content": "Hello"}
        doc = Document.from_dict(data)

        assert doc.id == "doc1"
        assert doc.content == "Hello"
        assert doc.embedding is None

    def test_from_dict_full(self):
        """Test creating document from full dict."""
        data = {
            "id": "doc1",
            "content": "Hello",
            "embedding": [0.1, 0.2],
            "metadata": {"key": "value"},
            "created_at": "2024-01-15T10:30:00+00:00",
            "content_type": "image",
        }
        doc = Document.from_dict(data)

        assert doc.embedding == [0.1, 0.2]
        assert doc.metadata == {"key": "value"}
        assert doc.content_type == "image"

    def test_from_dict_with_raw_content(self):
        """Test from_dict decodes raw content."""
        import base64

        encoded = base64.b64encode(b"binary data").decode()
        data = {
            "id": "doc1",
            "content": "Hello",
            "raw_content": encoded,
        }
        doc = Document.from_dict(data)

        assert doc.raw_content == b"binary data"

    def test_from_dict_no_created_at(self):
        """Test from_dict handles missing created_at."""
        data = {"id": "doc1", "content": "Hello"}
        doc = Document.from_dict(data)

        assert doc.created_at is not None

    def test_round_trip(self):
        """Test document survives to_dict/from_dict round trip."""
        original = Document(
            id="doc1",
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            metadata={"source": "test", "page": 1},
            content_type="pdf",
            raw_content=b"binary",
        )

        restored = Document.from_dict(original.to_dict())

        assert restored.id == original.id
        assert restored.content == original.content
        assert restored.embedding == original.embedding
        assert restored.metadata == original.metadata
        assert restored.content_type == original.content_type
        assert restored.raw_content == original.raw_content


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_create_minimal(self):
        """Test creating search result with minimal fields."""
        doc = Document(id="doc1", content="Hello")
        result = SearchResult(document=doc, score=0.95)

        assert result.document is doc
        assert result.score == 0.95
        assert result.distance is None

    def test_create_full(self):
        """Test creating search result with all fields."""
        doc = Document(id="doc1", content="Hello")
        result = SearchResult(document=doc, score=0.95, distance=0.05)

        assert result.distance == 0.05


class TestVectorStoreConfig:
    """Tests for VectorStoreConfig dataclass."""

    def test_create_minimal(self):
        """Test creating config with minimal fields."""
        config = VectorStoreConfig(dimension=1024)

        assert config.dimension == 1024
        assert config.distance_metric == "cosine"
        assert config.index_type == "hnsw"

    def test_create_full(self):
        """Test creating config with all fields."""
        config = VectorStoreConfig(
            dimension=512,
            distance_metric="l2",
            index_type="flat",
        )

        assert config.dimension == 512
        assert config.distance_metric == "l2"
        assert config.index_type == "flat"

    def test_config_is_frozen(self):
        """Test that config is immutable."""
        config = VectorStoreConfig(dimension=1024)
        with pytest.raises(FrozenInstanceError):
            config.dimension = 512


class TestVectorStoreProtocol:
    """Tests for VectorStore protocol."""

    def test_protocol_checking(self):
        """Test that protocol can be used for type checking."""
        # InMemoryVectorStore should implement VectorStore protocol
        from locus.rag.stores.memory import InMemoryVectorStore

        store = InMemoryVectorStore()
        assert isinstance(store, VectorStore)


class TestBaseVectorStore:
    """Tests for BaseVectorStore abstract class."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseVectorStore cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseVectorStore()

    def test_subclass_must_implement_abstract_methods(self):
        """Test that subclass must implement abstract methods."""

        class IncompleteStore(BaseVectorStore):
            pass

        with pytest.raises(TypeError):
            IncompleteStore()

    @pytest.mark.asyncio
    async def test_default_add_batch(self):
        """Test default add_batch implementation."""

        class MinimalStore(BaseVectorStore):
            def __init__(self):
                self.added = []

            @property
            def config(self):
                return VectorStoreConfig(dimension=128)

            async def add(self, document):
                self.added.append(document)
                return document.id

            async def get(self, doc_id):
                return None

            async def delete(self, doc_id):
                return False

            async def search(self, query_embedding, limit=10, threshold=None, metadata_filter=None):
                return []

        store = MinimalStore()
        docs = [
            Document(id="doc1", content="Hello", embedding=[0.1]),
            Document(id="doc2", content="World", embedding=[0.2]),
        ]

        ids = await store.add_batch(docs)

        assert ids == ["doc1", "doc2"]
        assert len(store.added) == 2

    @pytest.mark.asyncio
    async def test_default_count(self):
        """Test default count returns 0."""

        class MinimalStore(BaseVectorStore):
            @property
            def config(self):
                return VectorStoreConfig(dimension=128)

            async def add(self, document):
                return document.id

            async def get(self, doc_id):
                return None

            async def delete(self, doc_id):
                return False

            async def search(self, query_embedding, limit=10, threshold=None, metadata_filter=None):
                return []

        store = MinimalStore()
        count = await store.count()
        assert count == 0

    @pytest.mark.asyncio
    async def test_default_clear(self):
        """Test default clear returns 0."""

        class MinimalStore(BaseVectorStore):
            @property
            def config(self):
                return VectorStoreConfig(dimension=128)

            async def add(self, document):
                return document.id

            async def get(self, doc_id):
                return None

            async def delete(self, doc_id):
                return False

            async def search(self, query_embedding, limit=10, threshold=None, metadata_filter=None):
                return []

        store = MinimalStore()
        count = await store.clear()
        assert count == 0

    @pytest.mark.asyncio
    async def test_default_close(self):
        """Test default close does nothing."""

        class MinimalStore(BaseVectorStore):
            @property
            def config(self):
                return VectorStoreConfig(dimension=128)

            async def add(self, document):
                return document.id

            async def get(self, doc_id):
                return None

            async def delete(self, doc_id):
                return False

            async def search(self, query_embedding, limit=10, threshold=None, metadata_filter=None):
                return []

        store = MinimalStore()
        await store.close()  # Should not raise
