# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Synchronous-API variant of :class:`OracleADBLoader`.

Wraps the async loader and exposes blocking equivalents. The streaming
:meth:`OracleADBLoader.lazy_load` async generator is surfaced as
:meth:`OracleSyncADBLoader.lazy_load` returning a fully-drained
``list[Document]`` — the iterator-of-document shape doesn't survive the
sync boundary cleanly (it'd require a new background loop per
``__next__``), and drained-list parity is what every other Sync wrapper
exposes.

**Zero** langchain / langgraph imports.
"""

from __future__ import annotations

from typing import Any

from pydantic import SecretStr

from locus._sync import drain, run_sync
from locus.rag.loaders.oracle import OracleADBLoader
from locus.rag.stores.base import Document


class OracleSyncADBLoader:
    """Sync companion to :class:`OracleADBLoader`.

    Same constructor signature. ``lazy_load`` and ``load`` both return
    a ``list[Document]`` — the async streaming semantics collapse to a
    drained list at the sync boundary. Callers that need true row-by-row
    streaming should stay on the async class.
    """

    def __init__(
        self,
        sql: str,
        content_column: str,
        bind_params: dict[str, Any] | None = None,
        id_column: str | None = None,
        metadata_columns: list[str] | None = None,
        fetch_arraysize: int = 100,
        dsn: str | None = None,
        user: str = "admin",
        password: str | SecretStr = "",
        wallet_location: str | None = None,
        wallet_password: str | SecretStr | None = None,
        host: str | None = None,
        port: int = 1521,
        service_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._async = OracleADBLoader(
            sql=sql,
            content_column=content_column,
            bind_params=bind_params,
            id_column=id_column,
            metadata_columns=metadata_columns,
            fetch_arraysize=fetch_arraysize,
            dsn=dsn,
            user=user,
            password=password,
            wallet_location=wallet_location,
            wallet_password=wallet_password,
            host=host,
            port=port,
            service_name=service_name,
            **kwargs,
        )

    # -- Loading -----------------------------------------------------------

    def lazy_load(self) -> list[Document]:
        """Drain the async generator into a list.

        Async ``lazy_load`` streams rows; the sync equivalent has to
        materialise — yielding from a background-thread loop one row
        at a time would create a fresh loop per call, which is far
        worse than just buffering. Callers needing true streaming
        should use the async class directly.
        """
        return run_sync(drain(self._async.lazy_load()))

    def load(self) -> list[Document]:
        return run_sync(self._async.load())

    # -- Lifecycle ---------------------------------------------------------

    def close(self) -> None:
        run_sync(self._async.close())

    def __repr__(self) -> str:
        return f"OracleSyncADBLoader(wrapping={self._async!r})"
