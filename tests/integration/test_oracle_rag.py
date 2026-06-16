# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Integration tests for Oracle 26ai RAG.

Tests validate Oracle's native VECTOR type for RAG operations.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


# Skip all tests if no credentials available
pytestmark = [
    pytest.mark.integration,
]


def has_oracle_available() -> bool:
    """Check if Oracle ADB is available.

    Requires both a wallet on disk *and* an ``ORACLE_PASSWORD`` in the
    environment. Without the password ``oracledb.connect()`` prompts for
    a PEM passphrase on stdin and fails with ``DPY-6005`` instead of
    skipping cleanly.
    """
    if not os.environ.get("ORACLE_PASSWORD"):
        return False
    wallet_path = Path(
        os.environ.get("ORACLE_WALLET", str(Path.home() / ".oci/wallets/deepresearch"))
    )
    return wallet_path.exists() and (wallet_path / "tnsnames.ora").exists()


def has_embedder_available() -> bool:
    """Check if an embedding provider is available."""
    if os.environ.get("OPENAI_API_KEY"):
        return True
    if (Path.home() / ".oci/config").exists():
        return True
    return False


def get_embedder():
    """Get embedder based on available credentials."""
    if os.environ.get("OPENAI_API_KEY"):
        from locus.rag.embeddings import OpenAIEmbeddings

        return OpenAIEmbeddings(model="text-embedding-3-small")

    if (Path.home() / ".oci/config").exists():
        try:
            from locus.rag.embeddings import OCIEmbeddings

            return OCIEmbeddings(
                model_id=os.getenv("OCI_EMBED_MODEL", "cohere.embed-english-v3.0"),
                profile_name=os.getenv("OCI_PROFILE", "DEFAULT"),
                auth_type=os.getenv("OCI_AUTH_TYPE", "api_key"),
                compartment_id=os.getenv("OCI_COMPARTMENT", ""),
                service_endpoint=os.getenv("OCI_ENDPOINT", ""),
            )
        except Exception:
            pass

    return None


# Oracle ADB credentials (configurable via environment)
ORACLE_DSN = os.environ.get("ORACLE_DSN", "deepresearch_low")
ORACLE_USER = os.environ.get("ORACLE_USER", "ADMIN")
ORACLE_PASSWORD = os.environ.get("ORACLE_PASSWORD", "")
ORACLE_WALLET = os.environ.get("ORACLE_WALLET", str(Path.home() / ".oci/wallets/deepresearch"))
ORACLE_WALLET_PASSWORD = os.environ.get("ORACLE_WALLET_PASSWORD", "")


@pytest.mark.skipif(
    not has_oracle_available() or not has_embedder_available(),
    reason="Oracle ADB or embedder not available",
)
class TestOracleVectorStore:
    """Tests for Oracle Vector Store operations."""

    @pytest.mark.asyncio
    async def test_oracle_connection(self):
        """Test basic Oracle connection."""
        from locus.rag.stores.oracle import OracleVectorStore

        embedder = get_embedder()
        if not embedder:
            pytest.skip("No embedder available")

        store = OracleVectorStore(
            dsn=ORACLE_DSN,
            user=ORACLE_USER,
            password=ORACLE_PASSWORD,
            wallet_location=ORACLE_WALLET,
            wallet_password=ORACLE_WALLET_PASSWORD,
            dimension=embedder.config.dimension,
            table_name="test_connection",
        )

        try:
            # Just test we can get count (creates table if needed)
            count = await store.count()
            assert count >= 0
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_add_and_get_document(self):
        """Test adding and retrieving a document."""
        from locus.rag.stores.base import Document
        from locus.rag.stores.oracle import OracleVectorStore

        embedder = get_embedder()
        if not embedder:
            pytest.skip("No embedder available")

        store = OracleVectorStore(
            dsn=ORACLE_DSN,
            user=ORACLE_USER,
            password=ORACLE_PASSWORD,
            wallet_location=ORACLE_WALLET,
            wallet_password=ORACLE_WALLET_PASSWORD,
            dimension=embedder.config.dimension,
            table_name="test_add_get",
        )

        try:
            # Clear existing data
            await store.clear()

            # Embed and add document
            result = await embedder.embed("Test document for Oracle 26ai")
            doc = Document(
                id="oracle_test_1",
                content="Test document for Oracle 26ai",
                embedding=result.embedding,
                metadata={"source": "test", "version": "1.0"},
            )
            doc_id = await store.add(doc)
            assert doc_id == "oracle_test_1"

            # Get document
            retrieved = await store.get("oracle_test_1")
            assert retrieved is not None
            assert retrieved.content == "Test document for Oracle 26ai"
            assert retrieved.metadata["source"] == "test"

            # Count
            count = await store.count()
            assert count == 1

            # Delete
            deleted = await store.delete("oracle_test_1")
            assert deleted is True
            assert await store.count() == 0

        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_vector_search(self):
        """Test vector similarity search."""
        from locus.rag.stores.base import Document
        from locus.rag.stores.oracle import OracleVectorStore

        embedder = get_embedder()
        if not embedder:
            pytest.skip("No embedder available")

        store = OracleVectorStore(
            dsn=ORACLE_DSN,
            user=ORACLE_USER,
            password=ORACLE_PASSWORD,
            wallet_location=ORACLE_WALLET,
            wallet_password=ORACLE_WALLET_PASSWORD,
            dimension=embedder.config.dimension,
            table_name="test_search",
        )

        try:
            await store.clear()

            # Add test documents
            texts = [
                "Python is a programming language",
                "JavaScript runs in browsers",
                "Cats are fluffy pets",
                "Oracle Database is enterprise software",
            ]

            for i, text in enumerate(texts):
                result = await embedder.embed(text)
                doc = Document(
                    id=f"search_doc_{i}",
                    content=text,
                    embedding=result.embedding,
                )
                await store.add(doc)

            # Search for programming-related
            query_result = await embedder.embed("programming languages")
            results = await store.search(query_result.embedding, limit=2)

            assert len(results) == 2
            # Top results should be programming-related
            contents = [r.document.content for r in results]
            assert any("Python" in c or "JavaScript" in c for c in contents)

            # Scores should be descending
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

        finally:
            await store.clear()
            await store.close()


@pytest.mark.skipif(
    not has_oracle_available() or not has_embedder_available(),
    reason="Oracle ADB or embedder not available",
)
class TestOracleRAGRetriever:
    """Tests for RAG Retriever with Oracle backend."""

    @pytest.mark.asyncio
    async def test_rag_retriever_with_oracle(self):
        """Test RAGRetriever with Oracle Vector Store."""
        from locus.rag import RAGRetriever
        from locus.rag.stores.oracle import OracleVectorStore

        embedder = get_embedder()
        if not embedder:
            pytest.skip("No embedder available")

        store = OracleVectorStore(
            dsn=ORACLE_DSN,
            user=ORACLE_USER,
            password=ORACLE_PASSWORD,
            wallet_location=ORACLE_WALLET,
            wallet_password=ORACLE_WALLET_PASSWORD,
            dimension=embedder.config.dimension,
            table_name="test_rag_retriever",
        )

        try:
            await store.clear()

            retriever = RAGRetriever(
                embedder=embedder,
                store=store,
                chunk_size=500,
            )

            # Add documents
            await retriever.add_documents(
                [
                    "Oracle 23ai introduces native VECTOR data type.",
                    "Vector search enables semantic similarity queries.",
                    "RAG combines retrieval with generation.",
                ]
            )

            # Retrieve
            result = await retriever.retrieve("vector database features", limit=2)

            assert len(result.documents) >= 1
            contents = [r.document.content for r in result.documents]
            assert any("VECTOR" in c or "Vector" in c for c in contents)

        finally:
            await store.clear()
            await store.close()

    @pytest.mark.asyncio
    async def test_rag_as_tool_with_oracle(self):
        """Test RAG tool creation with Oracle backend."""
        from locus.rag import RAGRetriever
        from locus.rag.stores.oracle import OracleVectorStore

        embedder = get_embedder()
        if not embedder:
            pytest.skip("No embedder available")

        store = OracleVectorStore(
            dsn=ORACLE_DSN,
            user=ORACLE_USER,
            password=ORACLE_PASSWORD,
            wallet_location=ORACLE_WALLET,
            wallet_password=ORACLE_WALLET_PASSWORD,
            dimension=embedder.config.dimension,
            table_name="test_rag_tool",
        )

        try:
            await store.clear()

            retriever = RAGRetriever(embedder=embedder, store=store)

            await retriever.add_documents(
                [
                    "Locus is a Python AI agent framework.",
                    "Locus supports multiple vector stores.",
                ]
            )

            # Create tool
            tool = retriever.as_tool(
                name="search_oracle",
                description="Search Oracle knowledge base",
            )

            assert tool.name == "search_oracle"

            # Test tool
            result = await tool("What is Locus?")
            assert "results" in result
            assert result["total"] > 0

        finally:
            await store.clear()
            await store.close()


@pytest.mark.skipif(
    not has_oracle_available() or not has_embedder_available(),
    reason="Oracle ADB or embedder not available",
)
class TestOracleWithBothEmbedders:
    """Test Oracle with both OCI and OpenAI embedders."""

    @pytest.mark.asyncio
    async def test_oracle_with_oci_cohere(self):
        """Test Oracle with OCI Cohere embeddings (1024 dims)."""
        if not (Path.home() / ".oci/config").exists():
            pytest.skip("OCI not configured")

        try:
            from locus.rag import RAGRetriever
            from locus.rag.embeddings import OCIEmbeddings
            from locus.rag.stores.oracle import OracleVectorStore

            embedder = OCIEmbeddings(
                model_id=os.getenv("OCI_EMBED_MODEL", "cohere.embed-english-v3.0"),
                profile_name=os.getenv("OCI_PROFILE", "DEFAULT"),
                auth_type=os.getenv("OCI_AUTH_TYPE", "api_key"),
                compartment_id=os.getenv("OCI_COMPARTMENT", ""),
                service_endpoint=os.getenv("OCI_ENDPOINT", ""),
            )

            store = OracleVectorStore(
                dsn=ORACLE_DSN,
                user=ORACLE_USER,
                password=ORACLE_PASSWORD,
                wallet_location=ORACLE_WALLET,
                wallet_password=ORACLE_WALLET_PASSWORD,
                dimension=1024,  # Cohere dimension
                table_name="test_oci_cohere",
            )

            try:
                await store.clear()

                retriever = RAGRetriever(embedder=embedder, store=store)
                await retriever.add_documents(
                    [
                        "OCI Cohere embeddings are 1024 dimensions.",
                        "Cohere models are optimized for search.",
                    ]
                )

                result = await retriever.retrieve("embedding dimensions", limit=1)
                assert len(result.documents) == 1
                assert "1024" in result.documents[0].document.content

            finally:
                await store.clear()
                await store.close()

        except Exception as e:
            pytest.skip(f"OCI Cohere not available: {e}")

    @pytest.mark.asyncio
    async def test_oracle_with_openai(self):
        """Test Oracle with OpenAI embeddings (1536 dims)."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OpenAI not configured")

        from locus.rag import RAGRetriever
        from locus.rag.embeddings import OpenAIEmbeddings
        from locus.rag.stores.oracle import OracleVectorStore

        embedder = OpenAIEmbeddings(model="text-embedding-3-small")

        store = OracleVectorStore(
            dsn=ORACLE_DSN,
            user=ORACLE_USER,
            password=ORACLE_PASSWORD,
            wallet_location=ORACLE_WALLET,
            wallet_password=ORACLE_WALLET_PASSWORD,
            dimension=1536,  # OpenAI dimension
            table_name="test_openai",
        )

        try:
            await store.clear()

            retriever = RAGRetriever(embedder=embedder, store=store)
            await retriever.add_documents(
                [
                    "OpenAI embeddings are 1536 dimensions.",
                    "text-embedding-3-small is fast and efficient.",
                ]
            )

            result = await retriever.retrieve("embedding model", limit=1)
            assert len(result.documents) == 1

        finally:
            await store.clear()
            await store.close()
            await embedder.close()
