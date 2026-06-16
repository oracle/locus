# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Synchronous-API variant of :class:`OracleStore`.

Wraps an internal async :class:`OracleStore` instance and exposes every
public method as a blocking call via :func:`locus._sync.run_sync`.
Method names are identical to the async surface so swap-in is a single
import change.

**Zero** langchain / langgraph imports.
"""

from __future__ import annotations

from typing import Any

from pydantic import SecretStr

from locus._sync import run_sync
from locus.memory.store import SemanticSearchResult, StoreCapabilities, StoreItem
from locus.memory.store_backends.oracle import OracleStore


class OracleSyncStore:
    """Sync companion to :class:`OracleStore`.

    Same constructor signature as the async class; every public method
    is mirrored with a blocking call. Capability gating (``dimension``
    None disables semantic-search methods) is delegated to the underlying
    async instance so the gating logic lives in exactly one place.
    """

    def __init__(
        self,
        *,
        dsn: str | None = None,
        user: str = "admin",
        password: str | SecretStr = "",
        wallet_location: str | None = None,
        wallet_password: str | SecretStr | None = None,
        host: str | None = None,
        port: int = 1521,
        service_name: str | None = None,
        table_name: str = "locus_store",
        schema_name: str | None = None,
        min_pool_size: int = 1,
        max_pool_size: int = 5,
        dimension: int | None = None,
        distance_metric: str = "COSINE",
        auto_create_table: bool = True,
    ) -> None:
        self._async = OracleStore(
            dsn=dsn,
            user=user,
            password=password,
            wallet_location=wallet_location,
            wallet_password=wallet_password,
            host=host,
            port=port,
            service_name=service_name,
            table_name=table_name,
            schema_name=schema_name,
            min_pool_size=min_pool_size,
            max_pool_size=max_pool_size,
            dimension=dimension,
            distance_metric=distance_metric,
            auto_create_table=auto_create_table,
        )

    # -- Capabilities ------------------------------------------------------

    @property
    def capabilities(self) -> StoreCapabilities:
        return self._async.capabilities

    @property
    def config(self) -> Any:
        return self._async.config

    # -- BaseStore: put / get / delete / list_keys -------------------------

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        return run_sync(self._async.put(namespace, key, value, metadata))

    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> Any | None:
        return run_sync(self._async.get(namespace, key))

    def delete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> bool:
        return run_sync(self._async.delete(namespace, key))

    def list_keys(
        self,
        namespace: tuple[str, ...],
        limit: int = 100,
    ) -> list[str]:
        return run_sync(self._async.list_keys(namespace, limit=limit))

    # -- BaseStore: search / list_namespaces -------------------------------

    def search(
        self,
        namespace: tuple[str, ...],
        query: str | None = None,
        limit: int = 10,
    ) -> list[StoreItem]:
        return run_sync(self._async.search(namespace, query=query, limit=limit))

    def list_namespaces(
        self,
        prefix: tuple[str, ...] | None = None,
        limit: int = 100,
    ) -> list[tuple[str, ...]]:
        return run_sync(self._async.list_namespaces(prefix=prefix, limit=limit))

    # -- BaseStore: semantic-search hooks ----------------------------------

    def put_with_embedding(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: Any,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        return run_sync(self._async.put_with_embedding(namespace, key, value, embedding, metadata))

    def search_by_embedding(
        self,
        namespace: tuple[str, ...],
        query_embedding: list[float],
        limit: int = 10,
        threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SemanticSearchResult]:
        return run_sync(
            self._async.search_by_embedding(
                namespace,
                query_embedding,
                limit=limit,
                threshold=threshold,
                metadata_filter=metadata_filter,
            )
        )

    # -- Lifecycle ---------------------------------------------------------

    def close(self) -> None:
        run_sync(self._async.close())

    def __repr__(self) -> str:
        return f"OracleSyncStore(wrapping={self._async!r})"
