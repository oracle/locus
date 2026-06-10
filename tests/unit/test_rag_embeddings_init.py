# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for RAG embeddings module init (lazy imports)."""

import pytest


class TestRAGEmbeddingsDirectImports:
    """Tests for direct imports from RAG embeddings module."""

    def test_import_base_classes(self):
        """Test importing base classes."""
        from tulip.rag.embeddings import (
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

    def test_lazy_import_openai_embeddings(self):
        """Test lazy importing OpenAIEmbeddings."""
        try:
            from tulip.rag.embeddings import OpenAIEmbeddings

            assert OpenAIEmbeddings is not None
        except ImportError:
            pytest.skip("OpenAI dependencies not available")

    def test_lazy_import_unknown_raises(self):
        """Test that unknown attribute raises AttributeError."""
        from tulip.rag import embeddings

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = embeddings.NonExistentProvider
