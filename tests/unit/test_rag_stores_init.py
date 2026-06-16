# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for RAG stores __init__ lazy imports."""

import pytest


class TestRagStoresDirectImports:
    """Tests for directly imported classes."""

    def test_import_base_vector_store(self):
        """Test importing BaseVectorStore."""
        from locus.rag.stores import BaseVectorStore

        assert BaseVectorStore is not None

    def test_import_document(self):
        """Test importing Document."""
        from locus.rag.stores import Document

        assert Document is not None

    def test_import_search_result(self):
        """Test importing SearchResult."""
        from locus.rag.stores import SearchResult

        assert SearchResult is not None

    def test_import_vector_store_protocol(self):
        """Test importing VectorStore protocol."""
        from locus.rag.stores import VectorStore

        assert VectorStore is not None

    def test_import_vector_store_config(self):
        """Test importing VectorStoreConfig."""
        from locus.rag.stores import VectorStoreConfig

        assert VectorStoreConfig is not None


class TestRagStoresLazyImports:
    """Tests for lazy imported stores."""

    def test_lazy_import_in_memory_store(self):
        """Test lazy importing InMemoryVectorStore."""
        from locus.rag.stores import InMemoryVectorStore

        assert InMemoryVectorStore is not None

    def test_lazy_import_oracle_store(self):
        """Test lazy importing OracleVectorStore."""
        try:
            from locus.rag.stores import OracleVectorStore

            assert OracleVectorStore is not None
        except ImportError:
            pytest.skip("Oracle dependencies not available")

    def test_lazy_import_opensearch_store(self):
        """Test lazy importing OpenSearchVectorStore."""
        try:
            from locus.rag.stores import OpenSearchVectorStore

            assert OpenSearchVectorStore is not None
        except ImportError:
            pytest.skip("OpenSearch dependencies not available")

    def test_lazy_import_pgvector_store(self):
        """Test lazy importing PgVectorStore."""
        try:
            from locus.rag.stores import PgVectorStore

            assert PgVectorStore is not None
        except ImportError:
            pytest.skip("PgVector dependencies not available")

    def test_lazy_import_unknown_raises(self):
        """Test that unknown attribute raises AttributeError."""
        from locus.rag import stores

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = stores.NonExistentStore


class TestRagStoresAll:
    """Tests for __all__ attribute."""

    def test_all_defined(self):
        """Test that __all__ is defined."""
        from locus.rag import stores

        assert hasattr(stores, "__all__")
        assert isinstance(stores.__all__, list)

    def test_all_contains_base_classes(self):
        """Test __all__ contains base classes."""
        from locus.rag import stores

        assert "BaseVectorStore" in stores.__all__
        assert "Document" in stores.__all__
        assert "SearchResult" in stores.__all__

    def test_all_contains_stores(self):
        """Test __all__ contains store implementations."""
        from locus.rag import stores

        assert "InMemoryVectorStore" in stores.__all__
