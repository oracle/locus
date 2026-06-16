# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for vector stores."""

from datetime import UTC, datetime

from locus.rag.stores.base import Document, SearchResult, VectorStoreConfig


class TestDocument:
    """Tests for Document model."""

    def test_create_document(self):
        """Create document with all fields."""
        now = datetime.now(UTC)
        doc = Document(
            id="doc1",
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            metadata={"key": "value"},
            created_at=now,
        )
        assert doc.id == "doc1"
        assert doc.content == "Test content"
        assert doc.embedding == [0.1, 0.2, 0.3]
        assert doc.metadata == {"key": "value"}
        assert doc.created_at == now

    def test_create_document_minimal(self):
        """Create document with minimal fields."""
        doc = Document(id="doc1", content="Test")
        assert doc.id == "doc1"
        assert doc.content == "Test"


class TestSearchResult:
    """Tests for SearchResult model."""

    def test_create_search_result(self):
        """Create search result."""
        doc = Document(id="doc1", content="Test")
        result = SearchResult(
            document=doc,
            score=0.95,
            distance=0.05,
        )
        assert result.document == doc
        assert result.score == 0.95
        assert result.distance == 0.05


class TestVectorStoreConfig:
    """Tests for VectorStoreConfig."""

    def test_create_config(self):
        """Test creating configuration."""
        config = VectorStoreConfig(
            dimension=1024,
            distance_metric="cosine",
            index_type="hnsw",
        )
        assert config.dimension == 1024
        assert config.distance_metric == "cosine"
        assert config.index_type == "hnsw"
