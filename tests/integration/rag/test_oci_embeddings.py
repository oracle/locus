# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Integration tests for OCI GenAI Embeddings (Cohere).

Configuration via environment variables:
- OCI_PROFILE: OCI config profile name
- OCI_AUTH_TYPE: Auth type (api_key, security_token, etc.)
- OCI_COMPARTMENT_ID: Compartment OCID
"""

import os
from pathlib import Path

import pytest


# Skip if OCI not configured
pytestmark = pytest.mark.skipif(
    not Path("~/.oci/config").expanduser().exists(),
    reason="OCI config not found",
)


class TestOCIEmbeddingsIntegration:
    """Integration tests for OCI GenAI embeddings."""

    @pytest.fixture
    def embedder(self, oci_config):
        """Create OCI embedder with configured auth."""
        from locus.rag.embeddings.oci import OCIEmbeddings

        return OCIEmbeddings(
            model_id=os.getenv("OCI_EMBED_MODEL", "cohere.embed-v4.0"),
            profile_name=oci_config["profile_name"],
            auth_type=oci_config["auth_type"],
            compartment_id=oci_config["compartment_id"],
            service_endpoint=oci_config.get("service_endpoint"),
        )

    @pytest.mark.asyncio
    async def test_embed_single(self, embedder):
        """Test embedding a single text."""
        result = await embedder.embed("Hello world, this is a test.")

        assert result.embedding is not None
        assert len(result.embedding) == embedder.config.dimension
        assert result.text == "Hello world, this is a test."
        assert result.model == embedder.oci_config.model_id

    @pytest.mark.asyncio
    async def test_embed_batch(self, embedder):
        """Test batch embedding."""
        texts = [
            "Python is a programming language.",
            "Oracle Database is a relational database.",
            "Machine learning enables AI applications.",
        ]

        results = await embedder.embed_batch(texts)

        assert len(results) == 3
        assert all(len(r.embedding) == embedder.dimension for r in results)
        assert [r.text for r in results] == texts

    @pytest.mark.asyncio
    async def test_embed_query(self, embedder):
        """Test query embedding (uses SEARCH_QUERY input type)."""
        result = await embedder.embed_query("What is machine learning?")

        assert len(result.embedding) == embedder.dimension
        assert result.text == "What is machine learning?"

    @pytest.mark.asyncio
    async def test_embed_documents(self, embedder):
        """Test document embedding (uses SEARCH_DOCUMENT input type)."""
        docs = [
            "Document about Python programming.",
            "Document about data science.",
        ]

        results = await embedder.embed_documents(docs)

        assert len(results) == 2
        assert all(len(r.embedding) == embedder.dimension for r in results)

    @pytest.mark.asyncio
    async def test_embedding_similarity(self, embedder):
        """Test that similar texts produce similar embeddings."""
        import math

        # Similar texts
        result1 = await embedder.embed("Python is a programming language")
        result2 = await embedder.embed("Python is a coding language")

        # Different text
        result3 = await embedder.embed("Cats are fluffy animals")

        def cosine_similarity(a, b):
            dot = sum(x * y for x, y in zip(a, b, strict=False))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            return dot / (norm_a * norm_b)

        sim_12 = cosine_similarity(result1.embedding, result2.embedding)
        sim_13 = cosine_similarity(result1.embedding, result3.embedding)

        # Similar texts should have higher similarity
        assert sim_12 > sim_13
        assert sim_12 > 0.8  # Should be quite similar

    def test_config(self, embedder):
        """Test embedder configuration."""
        config = embedder.config

        assert config.dimension == embedder.dimension
        assert config.batch_size == 96


class TestOCIEmbeddingsMultilingual:
    """Tests for multilingual embedding model."""

    @pytest.fixture
    def embedder(self, oci_config):
        """Create multilingual embedder."""
        from locus.rag.embeddings.oci import OCIEmbeddings

        return OCIEmbeddings(
            model_id=os.getenv("OCI_EMBED_MULTILINGUAL_MODEL", "cohere.embed-v4.0"),
            profile_name=oci_config["profile_name"],
            auth_type=oci_config["auth_type"],
            compartment_id=oci_config["compartment_id"],
            service_endpoint=oci_config.get("service_endpoint"),
        )

    @pytest.mark.asyncio
    async def test_embed_multiple_languages(self, embedder):
        """Test embedding texts in different languages."""
        texts = [
            "Hello, how are you?",  # English
            "Hola, como estas?",  # Spanish
            "Bonjour, comment allez-vous?",  # French
        ]

        results = await embedder.embed_batch(texts)

        assert len(results) == 3
        assert all(len(r.embedding) == embedder.dimension for r in results)


class TestOCIEmbeddingsLight:
    """Tests for a light (smaller) embedding model.

    Runs only with ``OCI_EMBED_LIGHT_MODEL`` set: the Cohere v3 light models
    (384-dim) were OCI's only light ones, and on-demand retired them on
    2026-09-30. A dedicated AI cluster that still hosts one can name it here.
    """

    @pytest.fixture
    def embedder(self, oci_config):
        """Create light embedder."""
        from locus.rag.embeddings.oci import OCIEmbeddings

        model_id = os.getenv("OCI_EMBED_LIGHT_MODEL")
        if not model_id:
            # The v3 light models were the only light ones, and OCI retired
            # them from on-demand on 2026-09-30.
            pytest.skip("no light embedding model on OCI on-demand; set OCI_EMBED_LIGHT_MODEL")
        return OCIEmbeddings(
            model_id=model_id,
            profile_name=oci_config["profile_name"],
            auth_type=oci_config["auth_type"],
            compartment_id=oci_config["compartment_id"],
            service_endpoint=oci_config.get("service_endpoint"),
        )

    @pytest.mark.asyncio
    async def test_embed_light(self, embedder):
        """The light model's vectors match its configured dimension."""
        result = await embedder.embed("Test text")

        # 384 for the Cohere v3 light models; whatever the table says for an
        # ``OCI_EMBED_LIGHT_MODEL`` override on tenancies where v3 is retired.
        assert len(result.embedding) == embedder.config.dimension

    def test_config_dimension(self, embedder):
        """The Cohere v3 light models are configured as 384-dim."""
        if "light" not in embedder.oci_config.model_id:
            pytest.skip("OCI_EMBED_LIGHT_MODEL overrides the light model; no 384-dim claim")
        assert embedder.config.dimension == 384
