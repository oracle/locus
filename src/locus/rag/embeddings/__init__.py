# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Embedding providers for RAG.

Available providers:
- OCIEmbeddings: OCI GenAI with Cohere models
- OpenAIEmbeddings: OpenAI text-embedding models
- OracleInDBEmbeddings: Oracle 23ai/26ai in-database ONNX embeddings
"""

from typing import Any

from locus.rag.embeddings.base import (
    BaseEmbedding,
    EmbeddingConfig,
    EmbeddingProvider,
    EmbeddingResult,
)


__all__ = [
    # Base
    "BaseEmbedding",
    "EmbeddingConfig",
    "EmbeddingProvider",
    "EmbeddingResult",
    # Providers (lazy imports)
    "OCIEmbeddings",
    "OpenAIEmbeddings",
    "OracleInDBEmbeddings",
    "OracleSyncInDBEmbeddings",
]


def __getattr__(name: str) -> Any:
    """Lazy import providers to avoid requiring all dependencies."""
    if name == "OCIEmbeddings":
        from locus.rag.embeddings.oci import OCIEmbeddings

        return OCIEmbeddings

    if name == "OpenAIEmbeddings":
        from locus.rag.embeddings.openai import OpenAIEmbeddings

        return OpenAIEmbeddings

    if name == "OracleInDBEmbeddings":
        from locus.rag.embeddings.oracle_indb import OracleInDBEmbeddings

        return OracleInDBEmbeddings

    if name == "OracleSyncInDBEmbeddings":
        from locus.rag.embeddings.oracle_sync import OracleSyncInDBEmbeddings

        return OracleSyncInDBEmbeddings

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
