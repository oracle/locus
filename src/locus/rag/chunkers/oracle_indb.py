# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Server-side text chunking via ``DBMS_VECTOR_CHAIN.UTL_TO_CHUNKS``.

The Oracle 23ai/26ai database ships a chunking primitive that
tokenises, segments, and yields chunks **inside the database**, with
no Python round-trip. Native locus equivalent of langchain-oracle's
``OracleTextSplitter`` — implemented here without depending on the
langchain ecosystem.

When to reach for this over the in-Python ``ChunkConfig`` on
``RAGRetriever``:

* You're ingesting a large CLOB column already in the database —
  this lets you go ``text → chunks → embedding → VECTOR`` without ever
  pulling the CLOB out.
* Your text-prep pipeline already lives in PL/SQL and you want one
  uniform tokenisation strategy across applications.
* You need deterministic chunk boundaries based on Oracle's tokeniser
  rather than the Python ``split()`` defaults.

Prerequisites (out-of-band, one-time per schema):

```sql
GRANT EXECUTE ON DBMS_VECTOR_CHAIN TO <app_user>;
```
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, SecretStr

from locus._oracle_pool_cache import safe_acquire
from locus.memory.backends._oracle_config import (
    OracleConfig,
    validate_sql_identifier,
)


_VALID_BY_VALUES = frozenset({"chars", "words", "vocabulary"})
_VALID_NORMALIZE = frozenset({"none", "all"})


class _ChunkParams(BaseModel):
    """Parameters mirroring DBMS_VECTOR_CHAIN.UTL_TO_CHUNKS' params JSON.

    Field names match the SQL-side keys (``by``, ``max``, ``overlap``,
    ``split``, ``normalize``) so the JSON serialisation is a direct
    ``model_dump`` away.
    """

    by: str = "words"  # chars | words | vocabulary
    max: int = 100  # max tokens per chunk (in `by` units)
    overlap: int = 0  # overlap between chunks, same units
    split: str = "recursively"  # recursively | sentence | paragraph | none
    normalize: str = "all"  # none | all (whitespace collapse, etc.)


class OracleInDBChunker(BaseModel):
    """Split text into chunks using ``DBMS_VECTOR_CHAIN.UTL_TO_CHUNKS``.

    Two call shapes:

    * :meth:`chunk_text` — pass a Python string, get back a list of
      chunks (each ``{chunk_id, text, offset, length}``).
    * :meth:`chunk_column` — point at an existing ``(table, id_col,
      text_col)`` and stream chunks of every row, no Python round-trip
      for the source text.

    Args:
        max_tokens: Soft cap per chunk in the unit of ``by``.
        overlap: Token overlap between adjacent chunks (0 by default).
        by: Tokenisation unit. ``"chars"`` is the simplest; ``"words"``
            is the Oracle default; ``"vocabulary"`` defers to the
            tokeniser of the configured ONNX vocabulary model.
        split: Boundary strategy. ``"recursively"`` (default) tries
            paragraph → sentence → word boundaries in order.
        normalize: ``"all"`` collapses whitespace and trims punctuation;
            ``"none"`` returns slices verbatim.
    """

    model_config = {"arbitrary_types_allowed": True}

    oracle_config: OracleConfig
    params: _ChunkParams = _ChunkParams()
    _pool: Any = None
    _pool_loop: Any = None  # asyncio loop the pool is bound to

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
        if by not in _VALID_BY_VALUES:
            raise ValueError(f"by must be one of {sorted(_VALID_BY_VALUES)}, got {by!r}")
        if normalize not in _VALID_NORMALIZE:
            raise ValueError(
                f"normalize must be one of {sorted(_VALID_NORMALIZE)}, got {normalize!r}"
            )
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")

        oracle_config = OracleConfig(
            dsn=dsn,
            user=user,
            password=SecretStr(password) if isinstance(password, str) else password,
            wallet_location=wallet_location,
            wallet_password=(
                SecretStr(wallet_password) if isinstance(wallet_password, str) else wallet_password
            ),
            host=host,
            port=port,
            service_name=service_name,
            **kwargs,
        )
        super().__init__(
            oracle_config=oracle_config,
            params=_ChunkParams(
                by=by, max=max_tokens, overlap=overlap, split=split, normalize=normalize
            ),
        )

    def _params_json(self) -> str:
        """Render the params block as a JSON literal for the SQL call."""
        return json.dumps(self.params.model_dump())

    async def _get_pool(self) -> Any:
        if self._pool is not None:
            return self._pool
        try:
            import oracledb
        except ImportError as e:  # pragma: no cover — depends on env
            raise ImportError("OracleInDBChunker requires the 'oracledb' package.") from e

        cfg = self.oracle_config
        params: dict[str, Any] = {}
        if cfg.wallet_location:
            params["config_dir"] = cfg.wallet_location
            params["wallet_location"] = cfg.wallet_location
            # ``is not None``: SecretStr("") is falsy in Pydantic v2, but
            # "" is the auto-login (cwallet.sso) wallet idiom. See #289.
            if cfg.wallet_password is not None:
                params["wallet_password"] = cfg.wallet_password.get_secret_value()

        dsn = cfg.dsn
        if dsn is None and cfg.host and cfg.service_name:
            dsn = oracledb.makedsn(cfg.host, cfg.port, service_name=cfg.service_name)

        self._pool = oracledb.create_pool_async(
            user=cfg.user,
            password=cfg.password.get_secret_value(),
            dsn=dsn,
            min=cfg.min_pool_size,
            max=cfg.max_pool_size,
            **params,
        )
        return self._pool

    async def chunk_text(self, text: str) -> list[dict[str, Any]]:
        """Split a Python string into chunks, returning structured rows.

        Each row carries ``chunk_id`` (1-based index from
        ``UTL_TO_CHUNKS``), the chunk ``text``, byte ``offset`` into
        the original document, and ``length``.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        if not text:
            return []

        pool = await self._get_pool()
        sql = """
            SELECT
                JSON_VALUE(t.column_value, '$.chunk_id') AS chunk_id,
                JSON_VALUE(t.column_value, '$.chunk_offset') AS chunk_offset,
                JSON_VALUE(t.column_value, '$.chunk_length') AS chunk_length,
                JSON_VALUE(t.column_value, '$.chunk_data') AS chunk_text
            FROM TABLE(
                DBMS_VECTOR_CHAIN.UTL_TO_CHUNKS(
                    :text,
                    JSON(:params)
                )
            ) t
            ORDER BY chunk_id
        """
        async with safe_acquire(self, pool) as conn, conn.cursor() as cursor:
            await cursor.execute(sql, {"text": text, "params": self._params_json()})
            rows = await cursor.fetchall()

        return [
            {
                "chunk_id": int(r[0]) if r[0] is not None else None,
                "offset": int(r[1]) if r[1] is not None else None,
                "length": int(r[2]) if r[2] is not None else None,
                "text": r[3] if r[3] is not None else "",
            }
            for r in rows
        ]

    async def chunk_column(
        self,
        *,
        table_name: str,
        text_column: str,
        id_column: str = "id",
        where: str | None = None,
    ) -> Any:  # AsyncIterator[dict[str, Any]] — typed as Any to dodge mypy v1.x quirks
        """Stream chunks of every row in ``table_name``, no Python round-trip.

        Yields ``{source_id, chunk_id, text, offset, length}`` per chunk.
        Set ``where`` to a parameter-less SQL fragment to restrict the
        scan; the chunker doesn't bind into ``where``, so callers MUST
        ensure it's free of user input (or pre-bind separately).
        """
        validate_sql_identifier(table_name, "table_name")
        validate_sql_identifier(text_column, "text_column")
        validate_sql_identifier(id_column, "id_column")
        if where is not None and re.search(r"[;\\]", where):
            raise ValueError("where clause must not contain ';' or '\\'")

        where_sql = f" WHERE {where}" if where else ""
        pool = await self._get_pool()
        sql = f"""
            SELECT
                src.{id_column} AS source_id,
                JSON_VALUE(t.column_value, '$.chunk_id') AS chunk_id,
                JSON_VALUE(t.column_value, '$.chunk_offset') AS chunk_offset,
                JSON_VALUE(t.column_value, '$.chunk_length') AS chunk_length,
                JSON_VALUE(t.column_value, '$.chunk_data') AS chunk_text
            FROM {table_name} src,
                 TABLE(
                    DBMS_VECTOR_CHAIN.UTL_TO_CHUNKS(
                        src.{text_column},
                        JSON(:params)
                    )
                 ) t
            {where_sql}
            ORDER BY src.{id_column}, chunk_id
        """
        async with safe_acquire(self, pool) as conn, conn.cursor() as cursor:
            await cursor.execute(sql, {"params": self._params_json()})
            async for row in cursor:
                yield {
                    "source_id": row[0],
                    "chunk_id": int(row[1]) if row[1] is not None else None,
                    "offset": int(row[2]) if row[2] is not None else None,
                    "length": int(row[3]) if row[3] is not None else None,
                    "text": row[4] if row[4] is not None else "",
                }

    async def close(self) -> None:
        if self._pool is not None:
            try:
                await self._pool.close()
            finally:
                self._pool = None
