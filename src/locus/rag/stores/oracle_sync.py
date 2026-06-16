# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Synchronous-API variant of :class:`OracleVectorStore`.

Wraps the async vector store and exposes every public method as a
blocking call via :func:`locus._sync.run_sync`. Same constructor
signature, identical method names — porting from sync code paths needs
only an import change.

**Zero** langchain / langgraph imports.
"""

from __future__ import annotations

from typing import Any

from pydantic import SecretStr

from locus._sync import run_sync
from locus.rag.stores.base import Document, SearchResult, VectorStoreConfig
from locus.rag.stores.oracle import OracleVectorStore


class OracleSyncVectorStore:
    """Sync companion to :class:`OracleVectorStore`.

    Same constructor signature as the async class. Every public method
    is mirrored with a blocking call. The DDL builders (e.g.
    ``_vector_index_ddl``) and config introspection remain async-class
    properties accessed via the underlying instance — sync wrappers
    only need to bridge I/O methods.
    """

    def __init__(
        self,
        dsn: str | None = None,
        user: str = "admin",
        password: str | SecretStr = "",
        host: str | None = None,
        port: int = 1521,
        service_name: str | None = None,
        dimension: int = 1024,
        distance_metric: str = "COSINE",
        **kwargs: Any,
    ) -> None:
        self._async = OracleVectorStore(
            dsn=dsn,
            user=user,
            password=password,
            host=host,
            port=port,
            service_name=service_name,
            dimension=dimension,
            distance_metric=distance_metric,
            **kwargs,
        )

    # -- Introspection -----------------------------------------------------

    @property
    def oracle_config(self) -> Any:
        return self._async.oracle_config

    @property
    def config(self) -> VectorStoreConfig:
        return self._async.config

    # -- CRUD --------------------------------------------------------------

    def add(self, document: Document) -> str:
        return run_sync(self._async.add(document))

    def add_batch(self, documents: list[Document]) -> list[str]:
        return run_sync(self._async.add_batch(documents))

    def get(self, doc_id: str) -> Document | None:
        return run_sync(self._async.get(doc_id))

    def delete(self, doc_id: str) -> bool:
        return run_sync(self._async.delete(doc_id))

    # -- Search ------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
        *,
        mmr: bool = False,
        mmr_lambda: float = 0.5,
        mmr_candidate_pool: int | None = None,
    ) -> list[SearchResult]:
        return run_sync(
            self._async.search(
                query_embedding,
                limit=limit,
                threshold=threshold,
                metadata_filter=metadata_filter,
                mmr=mmr,
                mmr_lambda=mmr_lambda,
                mmr_candidate_pool=mmr_candidate_pool,
            )
        )

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        *,
        limit: int = 10,
        alpha: float = 0.5,
        threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
        use_text_index: bool = False,
    ) -> list[SearchResult]:
        return run_sync(
            self._async.hybrid_search(
                query_text,
                query_embedding,
                limit=limit,
                alpha=alpha,
                threshold=threshold,
                metadata_filter=metadata_filter,
                use_text_index=use_text_index,
            )
        )

    # -- Index / table management -----------------------------------------

    def build_index(self, *, rebuild: bool = False) -> None:
        return run_sync(self._async.build_index(rebuild=rebuild))

    def ensure_text_index(self, *, drop_existing: bool = False) -> None:
        return run_sync(self._async.ensure_text_index(drop_existing=drop_existing))

    # -- Bulk operations ---------------------------------------------------

    def count(self) -> int:
        return run_sync(self._async.count())

    def clear(self) -> int:
        return run_sync(self._async.clear())

    # -- Lifecycle ---------------------------------------------------------

    def close(self) -> None:
        run_sync(self._async.close())

    def __repr__(self) -> str:
        return f"OracleSyncVectorStore(wrapping={self._async!r})"
