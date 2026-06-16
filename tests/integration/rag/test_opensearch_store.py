# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Integration tests for OpenSearch Vector Store.

Configuration via environment variables:
- OPENSEARCH_HOSTS: Comma-separated host list
- OPENSEARCH_USER: Username
- OPENSEARCH_PASSWORD: Password
- OPENSEARCH_USE_SSL: Use SSL (default: true)
- OPENSEARCH_VERIFY_CERTS: Verify certs (default: false)
"""

import pytest


class TestOpenSearchVectorStoreIntegration:
    """Integration tests for OpenSearch vector store."""

    @pytest.fixture
    async def store(self, opensearch_config):
        """Create OpenSearch store for testing."""
        from locus.rag.stores.opensearch import OpenSearchVectorStore

        store = OpenSearchVectorStore(
            hosts=opensearch_config["hosts"],
            http_auth=opensearch_config["http_auth"],
            use_ssl=opensearch_config["use_ssl"],
            index_name="locus_test_vectors",
            dimension=128,  # Small dimension for testing
        )

        # Clean up before test
        try:
            await store._ensure_index()
            await store.clear()
        except Exception:
            pass

        yield store

        # Clean up after test
        try:
            await store.clear()
            await store.close()
        except Exception:
            pass

    @pytest.fixture
    def sample_embedding(self):
        """Create a sample embedding."""
        return [0.1] * 128

    @pytest.mark.asyncio
    async def test_add_document(self, store, sample_embedding):
        """Test adding a document."""
        from locus.rag.stores.base import Document

        doc = Document(
            id="test_doc_1",
            content="This is a test document for OpenSearch.",
            embedding=sample_embedding,
            metadata={"source": "test", "category": "integration"},
        )

        doc_id = await store.add(doc)

        assert doc_id == "test_doc_1"

    @pytest.mark.asyncio
    async def test_get_document(self, store, sample_embedding):
        """Test retrieving a document."""
        from locus.rag.stores.base import Document

        doc = Document(
            id="test_doc_2",
            content="Document for retrieval test.",
            embedding=sample_embedding,
            metadata={"key": "value"},
        )
        await store.add(doc)

        retrieved = await store.get("test_doc_2")

        assert retrieved is not None
        assert retrieved.id == "test_doc_2"
        assert retrieved.content == "Document for retrieval test."
        assert retrieved.metadata["key"] == "value"

    @pytest.mark.asyncio
    async def test_delete_document(self, store, sample_embedding):
        """Test deleting a document."""
        from locus.rag.stores.base import Document

        doc = Document(
            id="test_doc_3",
            content="Document to delete.",
            embedding=sample_embedding,
        )
        await store.add(doc)

        deleted = await store.delete("test_doc_3")

        assert deleted is True
        assert await store.get("test_doc_3") is None

    @pytest.mark.asyncio
    async def test_search(self, store):
        """Test similarity search."""
        from locus.rag.stores.base import Document

        # Add documents with different embeddings
        docs = [
            Document(
                id="search_1",
                content="Python programming",
                embedding=[1.0] + [0.0] * 127,  # Points in one direction
            ),
            Document(
                id="search_2",
                content="Java programming",
                embedding=[0.9, 0.1] + [0.0] * 126,  # Similar direction
            ),
            Document(
                id="search_3",
                content="Cat pictures",
                embedding=[0.0, 1.0] + [0.0] * 126,  # Different direction
            ),
        ]

        for doc in docs:
            await store.add(doc)

        # Search with query similar to Python/Java
        results = await store.search(
            query_embedding=[1.0] + [0.0] * 127,
            limit=2,
        )

        assert len(results) >= 1
        # First result should be the most similar
        assert results[0].document.id in ("search_1", "search_2")

    @pytest.mark.asyncio
    async def test_search_with_threshold(self, store, sample_embedding):
        """Test search with similarity threshold."""
        from locus.rag.stores.base import Document

        await store.add(
            Document(
                id="threshold_1",
                content="Matching document",
                embedding=sample_embedding,
            )
        )

        results = await store.search(
            query_embedding=sample_embedding,
            threshold=0.5,
        )

        assert all(r.score >= 0.5 for r in results)

    @pytest.mark.asyncio
    async def test_batch_add(self, store, sample_embedding):
        """Test adding multiple documents."""
        from locus.rag.stores.base import Document

        docs = [
            Document(
                id=f"batch_{i}",
                content=f"Batch document {i}",
                embedding=sample_embedding,
            )
            for i in range(5)
        ]

        ids = await store.add_batch(docs)

        assert len(ids) == 5
        assert await store.count() >= 5

    @pytest.mark.asyncio
    async def test_count(self, store, sample_embedding):
        """Test document count."""
        from locus.rag.stores.base import Document

        initial_count = await store.count()

        for i in range(3):
            await store.add(
                Document(
                    id=f"count_{i}",
                    content=f"Count doc {i}",
                    embedding=sample_embedding,
                )
            )

        final_count = await store.count()
        assert final_count == initial_count + 3

    @pytest.mark.asyncio
    async def test_clear(self, store, sample_embedding):
        """Test clearing all documents."""
        from locus.rag.stores.base import Document

        for i in range(3):
            await store.add(
                Document(
                    id=f"clear_{i}",
                    content=f"Clear doc {i}",
                    embedding=sample_embedding,
                )
            )

        count = await store.clear()

        assert count >= 3
        assert await store.count() == 0

    def test_config(self, store):
        """Test store configuration."""
        config = store.config

        assert config.dimension == 128
        assert config.index_type == "hnsw"
