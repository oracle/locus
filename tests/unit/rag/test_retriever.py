# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for RAG retriever."""

import pytest

from locus.rag.embeddings.base import EmbeddingConfig, EmbeddingResult
from locus.rag.retriever import RAGRetriever, RetrievalResult
from locus.rag.stores.base import Document, SearchResult
from locus.rag.stores.memory import InMemoryVectorStore


class MockEmbedder:
    """Mock embedding provider for testing."""

    def __init__(self, dimension: int = 128):
        self._dimension = dimension

    @property
    def config(self) -> EmbeddingConfig:
        return EmbeddingConfig(dimension=self._dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> EmbeddingResult:
        """Generate deterministic embedding from text."""
        import hashlib

        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        embedding = [(hash_val >> i & 0xFF) / 255.0 for i in range(self._dimension)]
        return EmbeddingResult(
            embedding=embedding,
            text=text,
            model="mock",
        )

    async def embed_query(self, query: str) -> EmbeddingResult:
        return await self.embed(query)

    async def embed_documents(self, documents: list[str]) -> list[EmbeddingResult]:
        return [await self.embed(doc) for doc in documents]

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        return await self.embed_documents(texts)


class TestRAGRetriever:
    """Tests for RAGRetriever."""

    @pytest.fixture
    def embedder(self):
        return MockEmbedder(dimension=64)

    @pytest.fixture
    def store(self):
        return InMemoryVectorStore(dimension=64)

    @pytest.fixture
    def retriever(self, embedder, store):
        return RAGRetriever(
            embedder=embedder,
            store=store,
            chunk_size=100,
            chunk_overlap=20,
        )

    @pytest.mark.asyncio
    async def test_add_document(self, retriever):
        """Test adding a document."""
        ids = await retriever.add_document("This is a test document.")

        assert len(ids) == 1
        assert await retriever.count() == 1

    @pytest.mark.asyncio
    async def test_add_document_with_chunking(self, retriever):
        """Test adding a large document that gets chunked."""
        # Create content larger than chunk_size
        long_content = "Word " * 50  # ~250 chars

        ids = await retriever.add_document(long_content)

        assert len(ids) > 1  # Should be chunked
        assert await retriever.count() == len(ids)

    @pytest.mark.asyncio
    async def test_add_document_without_chunking(self, retriever):
        """Test adding document without chunking."""
        long_content = "Word " * 50

        ids = await retriever.add_document(long_content, chunk=False)

        assert len(ids) == 1

    @pytest.mark.asyncio
    async def test_add_documents(self, retriever):
        """Test adding multiple documents."""
        contents = ["Document one", "Document two", "Document three"]

        ids = await retriever.add_documents(contents)

        assert len(ids) == 3
        assert await retriever.count() == 3

    @pytest.mark.asyncio
    async def test_add_document_with_metadata(self, retriever):
        """Test adding document with metadata."""
        ids = await retriever.add_document(
            "Test content",
            metadata={"source": "test", "category": "docs"},
        )

        # Retrieve and check metadata
        doc = await retriever.store.get(ids[0])
        assert doc.metadata["source"] == "test"
        assert doc.metadata["category"] == "docs"

    @pytest.mark.asyncio
    async def test_retrieve(self, retriever):
        """Test retrieval."""
        await retriever.add_documents(
            [
                "Python is a programming language.",
                "Java is also a programming language.",
                "Cats are fluffy animals.",
            ]
        )

        result = await retriever.retrieve("programming languages", limit=2)

        assert isinstance(result, RetrievalResult)
        assert len(result.documents) == 2
        assert result.query == "programming languages"

    @pytest.mark.asyncio
    async def test_retrieve_with_threshold(self, retriever):
        """Test retrieval with similarity threshold."""
        await retriever.add_documents(
            [
                "Exact match content",
                "Completely different topic",
            ]
        )

        result = await retriever.retrieve(
            "Exact match content",
            threshold=0.9,
        )

        # Should only return high-similarity results
        assert all(r.score >= 0.9 for r in result.documents)

    @pytest.mark.asyncio
    async def test_retrieve_text(self, retriever):
        """Test retrieve_text convenience method."""
        await retriever.add_documents(
            [
                "First document content.",
                "Second document content.",
            ]
        )

        text = await retriever.retrieve_text("document", limit=2)

        assert isinstance(text, str)
        assert "content" in text.lower()

    @pytest.mark.asyncio
    async def test_retrieve_empty(self, retriever):
        """Test retrieval from empty store."""
        result = await retriever.retrieve("query")

        assert len(result.documents) == 0

    @pytest.mark.asyncio
    async def test_delete_document(self, retriever):
        """Test deleting a document."""
        ids = await retriever.add_document("Test content")

        deleted = await retriever.delete_document(ids[0])

        assert deleted is True
        assert await retriever.count() == 0

    @pytest.mark.asyncio
    async def test_clear(self, retriever):
        """Test clearing all documents."""
        await retriever.add_documents(["Doc 1", "Doc 2", "Doc 3"])

        count = await retriever.clear()

        assert count == 3
        assert await retriever.count() == 0

    @pytest.mark.asyncio
    async def test_count(self, retriever):
        """Test document count."""
        assert await retriever.count() == 0

        await retriever.add_documents(["Doc 1", "Doc 2"])

        assert await retriever.count() == 2

    def test_chunk_text(self, retriever):
        """Test text chunking."""
        # Short text - no chunking
        chunks = retriever._chunk_text("Short text")
        assert len(chunks) == 1

        # Long text - should be chunked
        long_text = "Word " * 100  # ~500 chars
        chunks = retriever._chunk_text(long_text)
        assert len(chunks) > 1

    def test_chunk_text_with_separator(self, retriever):
        """Test chunking with paragraph separator."""
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = retriever._chunk_text(text, separator="\n\n")

        # Should respect paragraph boundaries
        assert all("Paragraph" in chunk for chunk in chunks)

    @pytest.mark.asyncio
    async def test_as_tool(self, retriever):
        """Test creating a tool from retriever."""
        await retriever.add_document("Test content for tool")

        tool = retriever.as_tool(name="test_search")

        assert tool is not None
        assert callable(tool)


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_create_result(self):
        """Test creating a retrieval result."""
        doc = Document(id="doc1", content="Test")
        search_result = SearchResult(document=doc, score=0.9)

        result = RetrievalResult(
            documents=[search_result],
            query="test query",
            total_results=1,
        )

        assert len(result.documents) == 1
        assert result.query == "test query"
        assert result.total_results == 1
