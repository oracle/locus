# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for RAG embeddings module init (lazy imports)."""

import pytest


class TestRAGEmbeddingsDirectImports:
    """Tests for direct imports from RAG embeddings module."""

    def test_import_base_classes(self):
        """Test importing base classes."""
        from locus.rag.embeddings import (
            BaseEmbedding,
            EmbeddingConfig,
            EmbeddingProvider,
            EmbeddingResult,
        )

        assert BaseEmbedding is not None
        assert EmbeddingConfig is not None
        assert EmbeddingProvider is not None
        assert EmbeddingResult is not None


class TestRAGEmbeddingsLazyImports:
    """Tests for lazy imports in RAG embeddings module."""

    def test_lazy_import_oci_embeddings(self):
        """Test lazy importing OCIEmbeddings."""
        try:
            from locus.rag.embeddings import OCIEmbeddings

            assert OCIEmbeddings is not None
        except ImportError:
            pytest.skip("OCI dependencies not available")

    def test_lazy_import_openai_embeddings(self):
        """Test lazy importing OpenAIEmbeddings."""
        try:
            from locus.rag.embeddings import OpenAIEmbeddings

            assert OpenAIEmbeddings is not None
        except ImportError:
            pytest.skip("OpenAI dependencies not available")

    def test_lazy_import_unknown_raises(self):
        """Test that unknown attribute raises AttributeError."""
        from locus.rag import embeddings

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = embeddings.NonExistentProvider
