# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Unit tests for embedding providers."""

import pytest

from tulip.rag.embeddings.base import (
    BaseEmbedding,
    EmbeddingConfig,
    EmbeddingResult,
)


class TestEmbeddingResult:
    """Tests for EmbeddingResult dataclass."""

    def test_create_result(self):
        """Test creating an embedding result."""
        result = EmbeddingResult(
            embedding=[0.1, 0.2, 0.3],
            text="Hello world",
            model="test-model",
            tokens=2,
        )

        assert result.embedding == [0.1, 0.2, 0.3]
        assert result.text == "Hello world"
        assert result.model == "test-model"
        assert result.tokens == 2

    def test_result_without_tokens(self):
        """Test result without token count."""
        result = EmbeddingResult(
            embedding=[0.1, 0.2],
            text="Test",
            model="model",
        )

        assert result.tokens is None


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = EmbeddingConfig(dimension=1024)

        assert config.dimension == 1024
        assert config.max_tokens == 8192
        assert config.batch_size == 96

    def test_custom_config(self):
        """Test custom configuration."""
        config = EmbeddingConfig(
            dimension=384,
            max_tokens=4096,
            batch_size=32,
        )

        assert config.dimension == 384
        assert config.max_tokens == 4096
        assert config.batch_size == 32


class MockEmbedding(BaseEmbedding):
    """Mock embedding provider for testing."""

    def __init__(self, dimension: int = 1024):
        self._dimension = dimension

    @property
    def config(self) -> EmbeddingConfig:
        return EmbeddingConfig(dimension=self._dimension)

    async def embed(self, text: str) -> EmbeddingResult:
        # Return deterministic embedding based on text hash
        import hashlib

        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        embedding = [(hash_val >> i & 0xFF) / 255.0 for i in range(self._dimension)]
        return EmbeddingResult(
            embedding=embedding,
            text=text,
            model="mock-model",
            tokens=len(text.split()),
        )


class TestBaseEmbedding:
    """Tests for BaseEmbedding."""

    @pytest.mark.asyncio
    async def test_embed(self):
        """Test single text embedding."""
        embedder = MockEmbedding(dimension=128)
        result = await embedder.embed("Hello world")

        assert len(result.embedding) == 128
        assert result.text == "Hello world"
        assert result.model == "mock-model"

    @pytest.mark.asyncio
    async def test_embed_batch(self):
        """Test batch embedding."""
        embedder = MockEmbedding(dimension=64)
        results = await embedder.embed_batch(["Hello", "World", "Test"])

        assert len(results) == 3
        assert all(len(r.embedding) == 64 for r in results)
        assert [r.text for r in results] == ["Hello", "World", "Test"]

    @pytest.mark.asyncio
    async def test_embed_query(self):
        """Test query embedding (default uses embed)."""
        embedder = MockEmbedding()
        result = await embedder.embed_query("search query")

        assert result.text == "search query"
        assert len(result.embedding) == 1024

    @pytest.mark.asyncio
    async def test_embed_documents(self):
        """Test document embedding (default uses embed_batch)."""
        embedder = MockEmbedding(dimension=256)
        results = await embedder.embed_documents(["doc1", "doc2"])

        assert len(results) == 2
        assert all(len(r.embedding) == 256 for r in results)

    def test_dimension_property(self):
        """Test dimension property."""
        embedder = MockEmbedding(dimension=512)
        assert embedder.dimension == 512
