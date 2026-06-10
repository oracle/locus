# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for RAG notebooks 22-24.

Tests validate that all RAG notebook examples work correctly.
"""

from __future__ import annotations

import os

import pytest


# Skip all tests if no embedding provider is available
pytestmark = pytest.mark.integration


def has_embedder_available() -> bool:
    """Check if an embedding provider is available."""
    return bool(os.environ.get("OPENAI_API_KEY"))


def get_embedder():
    """Get embedder based on available credentials."""
    if os.environ.get("OPENAI_API_KEY"):
        from tulip.rag.embeddings import OpenAIEmbeddings

        return OpenAIEmbeddings(model="text-embedding-3-small")

    return None


def get_model():
    """Get LLM model based on available credentials.

    OpenAI when ``OPENAI_API_KEY`` is set; otherwise ``None``.
    """
    if os.environ.get("OPENAI_API_KEY"):
        from tulip.models.native.openai import OpenAIModel

        return OpenAIModel(model="gpt-4o-mini", max_tokens=256)

    return None


# =============================================================================
# Notebook 23: RAG Basics Tests
# =============================================================================


@pytest.mark.skipif(not has_embedder_available(), reason="No embedder available")
class TestNotebook22RAGBasics:
    """Tests for Notebook 23: RAG Basics."""

    @pytest.mark.asyncio
    async def test_embedding_single_text(self):
        """Test embedding a single text."""
        embedder = get_embedder()
        if not embedder:
            pytest.skip("No embedder available")

        result = await embedder.embed("Hello world")

        assert result.embedding is not None
        assert len(result.embedding) > 0
        assert result.text == "Hello world"

    @pytest.mark.asyncio
    async def test_embedding_batch(self):
        """Test batch embedding."""
        embedder = get_embedder()
        if not embedder:
            pytest.skip("No embedder available")

        texts = ["First text", "Second text", "Third text"]
        results = await embedder.embed_batch(texts)

        assert len(results) == 3
        assert all(len(r.embedding) > 0 for r in results)

    @pytest.mark.asyncio
    async def test_embedding_similarity(self):
        """Test that similar texts have similar embeddings."""
        import math

        embedder = get_embedder()
        if not embedder:
            pytest.skip("No embedder available")

        def cosine_similarity(a, b):
            dot = sum(x * y for x, y in zip(a, b, strict=False))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            return dot / (norm_a * norm_b)

        results = await embedder.embed_batch(
            [
                "Python programming language",
                "Python coding language",
                "Cats and dogs",
            ]
        )

        sim_similar = cosine_similarity(results[0].embedding, results[1].embedding)
        sim_different = cosine_similarity(results[0].embedding, results[2].embedding)

        # Similar texts should have higher similarity
        assert sim_similar > sim_different

    @pytest.mark.asyncio
    async def test_inmemory_vector_store(self):
        """Test InMemoryVectorStore operations."""
        from tulip.rag.stores.base import Document
        from tulip.rag.stores.memory import InMemoryVectorStore

        embedder = get_embedder()
        if not embedder:
            pytest.skip("No embedder available")

        store = InMemoryVectorStore(dimension=embedder.config.dimension)

        # Add document
        result = await embedder.embed("Test document content")
        doc = Document(
            id="test_doc",
            content="Test document content",
            embedding=result.embedding,
        )
        doc_id = await store.add(doc)
        assert doc_id == "test_doc"

        # Get document
        retrieved = await store.get("test_doc")
        assert retrieved is not None
        assert retrieved.content == "Test document content"

        # Search
        query_result = await embedder.embed("document")
        search_results = await store.search(
            query_embedding=query_result.embedding,
            limit=1,
        )
        assert len(search_results) == 1
        assert search_results[0].document.id == "test_doc"

        # Count
        count = await store.count()
        assert count == 1

        # Delete
        deleted = await store.delete("test_doc")
        assert deleted is True
        assert await store.count() == 0

    @pytest.mark.asyncio
    async def test_rag_retriever(self):
        """Test RAGRetriever end-to-end."""
        from tulip.rag import RAGRetriever
        from tulip.rag.stores.memory import InMemoryVectorStore

        embedder = get_embedder()
        if not embedder:
            pytest.skip("No embedder available")

        store = InMemoryVectorStore(dimension=embedder.config.dimension)
        retriever = RAGRetriever(
            embedder=embedder,
            store=store,
            chunk_size=500,
        )

        # Add documents
        await retriever.add_documents(
            [
                "Python is a programming language.",
                "JavaScript runs in browsers.",
                "Cats are fluffy pets.",
            ]
        )

        # Retrieve
        result = await retriever.retrieve("programming languages", limit=2)

        assert len(result.documents) >= 1
        # Should find Python or JavaScript
        contents = [r.document.content for r in result.documents]
        assert any("Python" in c or "JavaScript" in c for c in contents)

    @pytest.mark.asyncio
    async def test_rag_retriever_with_metadata(self):
        """Test RAGRetriever with metadata."""
        from tulip.rag import RAGRetriever
        from tulip.rag.stores.memory import InMemoryVectorStore

        embedder = get_embedder()
        if not embedder:
            pytest.skip("No embedder available")

        store = InMemoryVectorStore(dimension=embedder.config.dimension)
        retriever = RAGRetriever(embedder=embedder, store=store)

        await retriever.add_document(
            "Test content",
            metadata={"author": "test", "category": "demo"},
        )

        result = await retriever.retrieve("test", limit=1)

        assert len(result.documents) == 1
        assert result.documents[0].document.metadata["author"] == "test"


# =============================================================================
# Notebook 24: RAG Providers Tests
# =============================================================================


@pytest.mark.skipif(not has_embedder_available(), reason="No embedder available")
class TestNotebook23RAGProviders:
    """Tests for Notebook 24: RAG Providers."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OpenAI not configured")
    async def test_openai_embeddings(self):
        """Test OpenAI embeddings."""
        from tulip.rag.embeddings import OpenAIEmbeddings

        embedder = OpenAIEmbeddings(model="text-embedding-3-small")

        result = await embedder.embed("Test text")

        assert result.embedding is not None
        assert len(result.embedding) == 1536  # text-embedding-3-small dimension
        assert result.model == "text-embedding-3-small"

        await embedder.close()

    @pytest.mark.asyncio
    async def test_embedder_config(self):
        """Test embedder configuration properties."""
        embedder = get_embedder()
        if not embedder:
            pytest.skip("No embedder available")

        config = embedder.config

        assert config.dimension > 0
        assert config.batch_size > 0


# =============================================================================
# Notebook 25: RAG Agents Tests
# =============================================================================


@pytest.mark.skipif(not has_embedder_available(), reason="No embedder available")
class TestNotebook24RAGAgents:
    """Tests for Notebook 25: RAG Agents."""

    @pytest.mark.asyncio
    async def test_rag_as_tool(self):
        """Test converting RAG retriever to a tool."""
        from tulip.rag import RAGRetriever
        from tulip.rag.stores.memory import InMemoryVectorStore

        embedder = get_embedder()
        if not embedder:
            pytest.skip("No embedder available")

        store = InMemoryVectorStore(dimension=embedder.config.dimension)
        retriever = RAGRetriever(embedder=embedder, store=store)

        await retriever.add_documents(
            [
                "Tulip is a Python framework for AI agents.",
                "Tulip supports multiple LLM providers.",
            ]
        )

        # Create tool
        tool = retriever.as_tool(
            name="search_docs",
            description="Search documentation",
        )

        assert tool.name == "search_docs"
        assert "Search documentation" in tool.description

        # Test tool execution
        result = await tool("What is Tulip?")

        assert "results" in result
        assert "total" in result
        assert result["total"] > 0

    @pytest.mark.asyncio
    async def test_create_rag_tool(self):
        """Test create_rag_tool function."""
        from tulip.rag import RAGRetriever
        from tulip.rag.stores.memory import InMemoryVectorStore
        from tulip.rag.tools import create_rag_tool

        embedder = get_embedder()
        if not embedder:
            pytest.skip("No embedder available")

        store = InMemoryVectorStore(dimension=embedder.config.dimension)
        retriever = RAGRetriever(embedder=embedder, store=store)

        await retriever.add_documents(["Test document"])

        tool = create_rag_tool(
            retriever,
            name="kb_search",
            description="Search knowledge base",
        )

        assert tool.name == "kb_search"

    @pytest.mark.asyncio
    async def test_rag_agent_simple(self):
        """Test simple RAG agent."""
        from tulip.agent import Agent
        from tulip.rag import RAGRetriever
        from tulip.rag.stores.memory import InMemoryVectorStore

        embedder = get_embedder()
        model = get_model()
        if not embedder or not model:
            pytest.skip("No embedder or model available")

        store = InMemoryVectorStore(dimension=embedder.config.dimension)
        retriever = RAGRetriever(embedder=embedder, store=store)

        await retriever.add_documents(
            [
                "The capital of France is Paris.",
                "The capital of Germany is Berlin.",
            ]
        )

        search_tool = retriever.as_tool(
            name="search",
            description="Search for country information",
        )

        agent = Agent(
            model=model,
            tools=[search_tool],
            system_prompt="Use the search tool to answer questions.",
            max_iterations=3,
        )

        result = agent.run_sync("What is the capital of France?")

        assert result.success is True
        assert "Paris" in result.message or len(result.tool_executions) > 0

    @pytest.mark.asyncio
    async def test_retrieve_text(self):
        """Test retrieve_text convenience method."""
        from tulip.rag import RAGRetriever
        from tulip.rag.stores.memory import InMemoryVectorStore

        embedder = get_embedder()
        if not embedder:
            pytest.skip("No embedder available")

        store = InMemoryVectorStore(dimension=embedder.config.dimension)
        retriever = RAGRetriever(embedder=embedder, store=store)

        await retriever.add_documents(
            [
                "Python is great for data science.",
                "Machine learning uses Python extensively.",
            ]
        )

        text = await retriever.retrieve_text("Python programming", limit=2)

        assert isinstance(text, str)
        assert len(text) > 0
        assert "Python" in text


# =============================================================================
# Integration Tests: Full Pipeline
# =============================================================================


@pytest.mark.skipif(not has_embedder_available(), reason="No embedder available")
class TestRAGFullPipeline:
    """Full pipeline integration tests."""

    @pytest.mark.asyncio
    async def test_document_chunking(self):
        """Test that long documents are chunked correctly."""
        from tulip.rag import RAGRetriever
        from tulip.rag.stores.memory import InMemoryVectorStore

        embedder = get_embedder()
        if not embedder:
            pytest.skip("No embedder available")

        store = InMemoryVectorStore(dimension=embedder.config.dimension)
        retriever = RAGRetriever(
            embedder=embedder,
            store=store,
            chunk_size=100,  # Small chunks for testing
            chunk_overlap=20,
        )

        long_doc = "This is a test. " * 50  # ~800 chars

        ids = await retriever.add_document(long_doc)

        # Should be chunked into multiple documents
        assert len(ids) > 1

    @pytest.mark.asyncio
    async def test_similarity_ordering(self):
        """Test that results are ordered by similarity."""
        from tulip.rag import RAGRetriever
        from tulip.rag.stores.memory import InMemoryVectorStore

        embedder = get_embedder()
        if not embedder:
            pytest.skip("No embedder available")

        store = InMemoryVectorStore(dimension=embedder.config.dimension)
        retriever = RAGRetriever(embedder=embedder, store=store)

        await retriever.add_documents(
            [
                "Python programming language",
                "JavaScript programming language",
                "Cats and dogs pets",
            ]
        )

        result = await retriever.retrieve("Python code", limit=3)

        # Scores should be in descending order
        scores = [r.score for r in result.documents]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_empty_retrieval(self):
        """Test retrieval from empty store."""
        from tulip.rag import RAGRetriever
        from tulip.rag.stores.memory import InMemoryVectorStore

        embedder = get_embedder()
        if not embedder:
            pytest.skip("No embedder available")

        store = InMemoryVectorStore(dimension=embedder.config.dimension)
        retriever = RAGRetriever(embedder=embedder, store=store)

        result = await retriever.retrieve("anything", limit=5)

        assert len(result.documents) == 0
