# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the ``EmbeddingCapabilities`` surface.

Every concrete embedding provider advertises its capabilities so
callers can pick the right method without catching exceptions. This
test locks in the cross-provider contract.
"""

from __future__ import annotations

import pytest

from tulip.rag.embeddings.base import (
    BaseEmbedding,
    EmbeddingCapabilities,
    EmbeddingConfig,
    EmbeddingResult,
)


class _Stub(BaseEmbedding):
    """Minimal concrete subclass to exercise the default implementation."""

    @property
    def config(self) -> EmbeddingConfig:
        return EmbeddingConfig(dimension=3, max_tokens=1024, batch_size=4)

    async def embed(self, text: str) -> EmbeddingResult:  # pragma: no cover
        return EmbeddingResult(embedding=[0.0, 0.0, 0.0], text=text, model="stub")


class TestEmbeddingCapabilities:
    def test_default_capabilities_use_config_bounds(self) -> None:
        """BaseEmbedding default capabilities derive from config."""
        stub = _Stub()
        caps = stub.capabilities
        assert isinstance(caps, EmbeddingCapabilities)
        assert caps.supports_batching is True
        assert caps.max_batch_size == 4
        assert caps.max_input_tokens == 1024
        # Unadvertised features default to False.
        assert caps.supports_query_vs_doc is False
        assert caps.supports_multimodal is False

    def test_capabilities_are_immutable(self) -> None:
        """Frozen dataclass — callers can't mutate a provider's advertised surface."""
        caps = _Stub().capabilities
        with pytest.raises(AttributeError):
            caps.supports_multimodal = True  # type: ignore[misc]

    def test_openai_capabilities_are_declared(self) -> None:
        """OpenAI embeddings advertise text-only, native batching."""
        openai_module = pytest.importorskip("openai")
        assert openai_module is not None  # sanity
        from tulip.rag.embeddings.openai import OpenAIEmbeddings

        embedder = OpenAIEmbeddings(api_key="sk-test", model="text-embedding-3-small")
        caps = embedder.capabilities
        assert caps.supports_query_vs_doc is False
        assert caps.supports_multimodal is False
        assert caps.supports_batching is True
        assert caps.max_batch_size == 2048
