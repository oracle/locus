# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Synchronous-API variant of :class:`OracleInDBChunker`.

Wraps the async chunker and exposes blocking equivalents.
:meth:`chunk_column` is an async generator on the async class; the sync
counterpart drains it into a list (same pattern
:class:`OracleSyncADBLoader.lazy_load` follows).

**Zero** langchain / langgraph imports.
"""

from __future__ import annotations

from typing import Any

from pydantic import SecretStr

from locus._sync import drain, run_sync
from locus.rag.chunkers.oracle_indb import OracleInDBChunker


class OracleSyncInDBChunker:
    """Sync companion to :class:`OracleInDBChunker`."""

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
        max_tokens: int = 100,
        overlap: int = 0,
        by: str = "words",
        split: str = "recursively",
        normalize: str = "all",
        **kwargs: Any,
    ) -> None:
        self._async = OracleInDBChunker(
            dsn=dsn,
            user=user,
            password=password,
            wallet_location=wallet_location,
            wallet_password=wallet_password,
            host=host,
            port=port,
            service_name=service_name,
            max_tokens=max_tokens,
            overlap=overlap,
            by=by,
            split=split,
            normalize=normalize,
            **kwargs,
        )

    # -- Public surface ----------------------------------------------------

    def chunk_text(self, text: str) -> list[dict[str, Any]]:
        return run_sync(self._async.chunk_text(text))

    def chunk_column(
        self,
        *,
        table_name: str,
        text_column: str,
        id_column: str = "id",
        where: str | None = None,
    ) -> list[dict[str, Any]]:
        """Drain the async chunk_column generator into a list.

        Async ``chunk_column`` yields per-chunk dicts as the cursor
        produces them; the sync version materialises the full list.
        For huge tables, prefer the async class.
        """
        return run_sync(
            drain(
                self._async.chunk_column(
                    table_name=table_name,
                    text_column=text_column,
                    id_column=id_column,
                    where=where,
                )
            )
        )

    def close(self) -> None:
        run_sync(self._async.close())

    def __repr__(self) -> str:
        return f"OracleSyncInDBChunker(wrapping={self._async!r})"
