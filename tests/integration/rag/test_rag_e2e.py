# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""End-to-end integration tests for RAG pipeline.

Tests the complete flow: embeddings -> store -> retrieval.

Configuration via environment variables (see conftest.py).
"""

import os

import pytest


# Skip if OCI not configured
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.expanduser("~/.oci/config")),
    reason="OCI config not found",
)


class TestRAGEndToEnd:
    """End-to-end tests for RAG pipeline."""

    @pytest.fixture
    def embedder(self, oci_config):
        """Create OCI embedder."""
        from locus.rag.embeddings.oci import OCIEmbeddings

        return OCIEmbeddings(
            model_id=os.getenv("OCI_EMBED_MODEL", "cohere.embed-english-v3.0"),
            profile_name=oci_config["profile_name"],
            auth_type=oci_config["auth_type"],
            compartment_id=oci_config["compartment_id"],
            service_endpoint=oci_config.get("service_endpoint"),
        )

    @pytest.fixture
    async def retriever_memory(self, embedder):
        """Create retriever with in-memory store."""
        from locus.rag import RAGRetriever
        from locus.rag.stores.memory import InMemoryVectorStore

        store = InMemoryVectorStore(dimension=1024)

        retriever = RAGRetriever(
            embedder=embedder,
            store=store,
            chunk_size=500,
        )

        return retriever

    @pytest.mark.asyncio
    async def test_add_and_retrieve_memory(self, retriever_memory):
        """Test adding documents and retrieving with in-memory store."""
        # Add documents
        await retriever_memory.add_documents(
            [
                "Python is a high-level programming language known for its simplicity.",
                "Oracle Database is a powerful relational database management system.",
                "Machine learning is a subset of artificial intelligence.",
                "Cats are popular pets known for their independence.",
            ]
        )

        # Retrieve
        result = await retriever_memory.retrieve(
            "What programming languages are easy to learn?",
            limit=2,
        )

        assert len(result.documents) >= 1
        # Python document should be most relevant
        contents = [r.document.content for r in result.documents]
        assert any("Python" in c for c in contents)

    @pytest.mark.asyncio
    async def test_chunking(self, retriever_memory):
        """Test document chunking."""
        # Create a long document
        long_doc = " ".join(
            [f"Paragraph {i}: This is some content for testing chunking. " * 10 for i in range(10)]
        )

        ids = await retriever_memory.add_document(long_doc)

        # Should be chunked into multiple documents
        assert len(ids) > 1

    @pytest.mark.asyncio
    async def test_retrieve_text(self, retriever_memory):
        """Test retrieve_text convenience method."""
        await retriever_memory.add_documents(
            [
                "Document about Python programming.",
                "Document about Java programming.",
            ]
        )

        text = await retriever_memory.retrieve_text(
            "programming languages",
            limit=2,
        )

        assert isinstance(text, str)
        assert "programming" in text.lower()

    @pytest.mark.asyncio
    async def test_metadata_preservation(self, retriever_memory):
        """Test that metadata is preserved through the pipeline."""
        await retriever_memory.add_document(
            "Test document content",
            metadata={"author": "test", "category": "docs"},
        )

        result = await retriever_memory.retrieve("test document", limit=1)

        assert len(result.documents) == 1
        metadata = result.documents[0].document.metadata
        assert metadata["author"] == "test"
        assert metadata["category"] == "docs"

    @pytest.mark.asyncio
    async def test_similarity_ordering(self, retriever_memory):
        """Test that results are ordered by similarity."""
        await retriever_memory.add_documents(
            [
                "Python is a programming language",  # Most similar
                "JavaScript is also a programming language",  # Similar
                "Cats like to sleep in the sun",  # Not similar
            ]
        )

        result = await retriever_memory.retrieve(
            "Python programming",
            limit=3,
        )

        # Scores should be in descending order
        scores = [r.score for r in result.documents]
        assert scores == sorted(scores, reverse=True)

        # Python should be first
        assert "Python" in result.documents[0].document.content


class TestRAGTool:
    """Tests for RAG tool integration."""

    @pytest.fixture
    def embedder(self, oci_config):
        """Create OCI embedder."""
        from locus.rag.embeddings.oci import OCIEmbeddings

        return OCIEmbeddings(
            model_id=os.getenv("OCI_EMBED_MODEL", "cohere.embed-english-v3.0"),
            profile_name=oci_config["profile_name"],
            auth_type=oci_config["auth_type"],
            compartment_id=oci_config["compartment_id"],
            service_endpoint=oci_config.get("service_endpoint"),
        )

    @pytest.fixture
    async def retriever(self, embedder):
        """Create retriever."""
        from locus.rag import RAGRetriever
        from locus.rag.stores.memory import InMemoryVectorStore

        store = InMemoryVectorStore(dimension=1024)

        return RAGRetriever(
            embedder=embedder,
            store=store,
        )

    @pytest.mark.asyncio
    async def test_as_tool(self, retriever):
        """Test creating a tool from retriever."""
        await retriever.add_documents(
            [
                "Python is great for data science.",
                "Oracle offers cloud services.",
            ]
        )

        tool = retriever.as_tool(name="search_docs")

        # Tool should be callable
        result = await tool("What is Python used for?")

        assert "results" in result
        assert "total" in result
        assert "query" in result

    @pytest.mark.asyncio
    async def test_tool_with_custom_description(self, retriever):
        """Test tool with custom description."""
        from locus.rag.tools import create_rag_tool

        tool = create_rag_tool(
            retriever,
            name="kb_search",
            description="Search the knowledge base for information.",
        )

        assert tool is not None


class TestMultimodalRAG:
    """Tests for multimodal RAG support."""

    @pytest.fixture
    def embedder(self, oci_config):
        """Create OCI embedder."""
        from locus.rag.embeddings.oci import OCIEmbeddings

        return OCIEmbeddings(
            model_id=os.getenv("OCI_EMBED_MODEL", "cohere.embed-english-v3.0"),
            profile_name=oci_config["profile_name"],
            auth_type=oci_config["auth_type"],
            compartment_id=oci_config["compartment_id"],
            service_endpoint=oci_config.get("service_endpoint"),
        )

    @pytest.fixture
    async def retriever(self, embedder):
        """Create retriever."""
        from locus.rag import RAGRetriever
        from locus.rag.stores.memory import InMemoryVectorStore

        store = InMemoryVectorStore(dimension=1024)

        return RAGRetriever(
            embedder=embedder,
            store=store,
        )

    @pytest.mark.asyncio
    async def test_text_content_type(self, retriever):
        """Test that text documents have correct content type."""
        await retriever.add_document("Plain text document")

        result = await retriever.retrieve("document", limit=1)

        # Text documents should have text content type
        doc = result.documents[0].document
        assert doc.content_type == "text"
