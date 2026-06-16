# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Synchronous-API variant of :class:`OracleInDBEmbeddings`.

Wraps the async in-DB embedder and exposes blocking equivalents. All
SQL construction (``UTL_TO_EMBEDDING`` / ``UTL_TO_EMBEDDINGS``),
``VECTOR_SERIALIZE`` parsing, and CLOB handling live on the async
class; this wrapper only bridges the I/O methods through
:func:`locus._sync.run_sync`.

**Zero** langchain / langgraph imports.
"""

from __future__ import annotations

from pydantic import SecretStr

from locus._sync import run_sync
from locus.rag.embeddings.base import EmbeddingCapabilities, EmbeddingConfig, EmbeddingResult
from locus.rag.embeddings.oracle_indb import OracleInDBEmbeddings


class OracleSyncInDBEmbeddings:
    """Sync companion to :class:`OracleInDBEmbeddings`."""

    def __init__(
        self,
        *,
        model_name: str,
        dimension: int,
        dsn: str | None = None,
        user: str = "admin",
        password: str | SecretStr = "",
        wallet_location: str | None = None,
        wallet_password: str | SecretStr | None = None,
        host: str | None = None,
        port: int = 1521,
        service_name: str | None = None,
        min_pool_size: int = 1,
        max_pool_size: int = 5,
        use_batch_function: bool = True,
    ) -> None:
        self._async = OracleInDBEmbeddings(
            model_name=model_name,
            dimension=dimension,
            dsn=dsn,
            user=user,
            password=password,
            wallet_location=wallet_location,
            wallet_password=wallet_password,
            host=host,
            port=port,
            service_name=service_name,
            min_pool_size=min_pool_size,
            max_pool_size=max_pool_size,
            use_batch_function=use_batch_function,
        )

    # -- Public introspection ----------------------------------------------

    @property
    def model_name(self) -> str:
        return self._async.model_name

    @property
    def config(self) -> EmbeddingConfig:
        return self._async.config

    @property
    def capabilities(self) -> EmbeddingCapabilities:
        return self._async.capabilities

    # -- embed / embed_batch ------------------------------------------------

    def embed(self, text: str) -> EmbeddingResult:
        return run_sync(self._async.embed(text))

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        return run_sync(self._async.embed_batch(texts))

    def embed_query(self, query: str) -> EmbeddingResult:
        return run_sync(self._async.embed_query(query))

    # -- Lifecycle ---------------------------------------------------------

    def close(self) -> None:
        run_sync(self._async.close())

    def __repr__(self) -> str:
        return f"OracleSyncInDBEmbeddings(wrapping={self._async!r})"


# Public re-export so ``from locus.rag.embeddings.oracle_sync import *``
# picks up the wrapper class.
__all__ = ["OracleSyncInDBEmbeddings"]
