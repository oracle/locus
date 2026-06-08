# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Unit tests for RAG stores __init__ lazy imports."""

import pytest


class TestRagStoresDirectImports:
    """Tests for directly imported classes."""

    def test_import_base_vector_store(self):
        """Test importing BaseVectorStore."""
        from tulip.rag.stores import BaseVectorStore

        assert BaseVectorStore is not None

    def test_import_document(self):
        """Test importing Document."""
        from tulip.rag.stores import Document

        assert Document is not None

    def test_import_search_result(self):
        """Test importing SearchResult."""
        from tulip.rag.stores import SearchResult

        assert SearchResult is not None

    def test_import_vector_store_protocol(self):
        """Test importing VectorStore protocol."""
        from tulip.rag.stores import VectorStore

        assert VectorStore is not None

    def test_import_vector_store_config(self):
        """Test importing VectorStoreConfig."""
        from tulip.rag.stores import VectorStoreConfig

        assert VectorStoreConfig is not None


class TestRagStoresLazyImports:
    """Tests for lazy imported stores."""

    def test_lazy_import_in_memory_store(self):
        """Test lazy importing InMemoryVectorStore."""
        from tulip.rag.stores import InMemoryVectorStore

        assert InMemoryVectorStore is not None

    def test_lazy_import_opensearch_store(self):
        """Test lazy importing OpenSearchVectorStore."""
        try:
            from tulip.rag.stores import OpenSearchVectorStore

            assert OpenSearchVectorStore is not None
        except ImportError:
            pytest.skip("OpenSearch dependencies not available")

    def test_lazy_import_pgvector_store(self):
        """Test lazy importing PgVectorStore."""
        try:
            from tulip.rag.stores import PgVectorStore

            assert PgVectorStore is not None
        except ImportError:
            pytest.skip("PgVector dependencies not available")

    def test_lazy_import_unknown_raises(self):
        """Test that unknown attribute raises AttributeError."""
        from tulip.rag import stores

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = stores.NonExistentStore


class TestRagStoresAll:
    """Tests for __all__ attribute."""

    def test_all_defined(self):
        """Test that __all__ is defined."""
        from tulip.rag import stores

        assert hasattr(stores, "__all__")
        assert isinstance(stores.__all__, list)

    def test_all_contains_base_classes(self):
        """Test __all__ contains base classes."""
        from tulip.rag import stores

        assert "BaseVectorStore" in stores.__all__
        assert "Document" in stores.__all__
        assert "SearchResult" in stores.__all__

    def test_all_contains_stores(self):
        """Test __all__ contains store implementations."""
        from tulip.rag import stores

        assert "InMemoryVectorStore" in stores.__all__
