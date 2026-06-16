# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for RAG module __init__ (lazy imports)."""

import pytest


class TestRAGDirectImports:
    """Tests for directly imported classes."""

    def test_import_embedding_base_classes(self):
        """Test importing embedding base classes."""
        from locus.rag import (
            BaseEmbedding,
            EmbeddingConfig,
            EmbeddingProvider,
            EmbeddingResult,
        )

        assert BaseEmbedding is not None
        assert EmbeddingConfig is not None
        assert EmbeddingProvider is not None
        assert EmbeddingResult is not None

    def test_import_store_base_classes(self):
        """Test importing store base classes."""
        from locus.rag import (
            BaseVectorStore,
            Document,
            SearchResult,
            VectorStore,
            VectorStoreConfig,
        )

        assert BaseVectorStore is not None
        assert Document is not None
        assert SearchResult is not None
        assert VectorStore is not None
        assert VectorStoreConfig is not None

    def test_import_retriever(self):
        """Test importing retriever."""
        from locus.rag import RAGRetriever, RetrievalResult

        assert RAGRetriever is not None
        assert RetrievalResult is not None

    def test_import_multimodal(self):
        """Test importing multimodal classes."""
        from locus.rag import (
            ContentType,
            MultimodalProcessor,
            ProcessedContent,
            process_content,
        )

        assert ContentType is not None
        assert MultimodalProcessor is not None
        assert ProcessedContent is not None
        assert process_content is not None

    def test_import_tools(self):
        """Test importing tools."""
        from locus.rag import (
            RAGToolkit,
            create_rag_context_tool,
            create_rag_tool,
        )

        assert RAGToolkit is not None
        assert create_rag_context_tool is not None
        assert create_rag_tool is not None


class TestRAGLazyImports:
    """Tests for lazy imported classes."""

    def test_lazy_import_in_memory_store(self):
        """Test lazy importing InMemoryVectorStore."""
        from locus.rag import InMemoryVectorStore

        assert InMemoryVectorStore is not None

    def test_lazy_import_oci_embeddings(self):
        """Test lazy importing OCIEmbeddings."""
        try:
            from locus.rag import OCIEmbeddings

            assert OCIEmbeddings is not None
        except ImportError:
            pytest.skip("OCI dependencies not available")

    def test_lazy_import_oracle_store(self):
        """Test lazy importing OracleVectorStore."""
        try:
            from locus.rag import OracleVectorStore

            assert OracleVectorStore is not None
        except ImportError:
            pytest.skip("Oracle dependencies not available")

    def test_lazy_import_opensearch_store(self):
        """Test lazy importing OpenSearchVectorStore."""
        try:
            from locus.rag import OpenSearchVectorStore

            assert OpenSearchVectorStore is not None
        except ImportError:
            pytest.skip("OpenSearch dependencies not available")

    def test_lazy_import_unknown_raises(self):
        """Test that unknown attribute raises AttributeError."""
        from locus import rag

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = rag.NonExistentClass


class TestRAGAll:
    """Tests for __all__ attribute."""

    def test_all_defined(self):
        """Test that __all__ is defined."""
        from locus import rag

        assert hasattr(rag, "__all__")
        assert isinstance(rag.__all__, list)
        assert "RAGRetriever" in rag.__all__
        assert "Document" in rag.__all__
