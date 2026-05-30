# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Oracle Autonomous Database document loader.

Native locus analog of langchain-oracle's
``OracleAutonomousDatabaseLoader``: it runs a single SELECT against an
ADB (wallet-secured or plain TCPS) and yields one
:class:`locus.rag.stores.base.Document` per row, mapping a designated
column to ``Document.content``, an optional column to ``Document.id``,
and the rest of the projected columns into ``Document.metadata``.

Why a locus-native loader instead of pulling in langchain-oracle?

* **Zero langchain dependency.** locus's RAG stack already speaks
  ``Document`` directly — chaining through ``langchain_core.documents``
  would add a heavy import graph (langchain-core, pydantic-v1 shims,
  tracers) just to translate dataclasses. The loader is ~150 lines of
  pure ``oracledb`` async code, so we keep it in-tree.
* **Same connection envelope as the rest of locus.** Both
  :class:`locus.memory.backends.oracle.OracleBackend` (checkpointer)
  and :class:`locus.rag.stores.oracle.OracleVectorStore` already accept
  ``dsn`` / ``user`` / ``password`` / ``wallet_location`` /
  ``wallet_password`` and call ``oracledb.create_pool_async`` the same
  way. This loader mirrors those field names so an app can reuse a
  single config block for checkpoint + vector store + raw row pull.
* **Async-first.** ``lazy_load()`` yields rows as the cursor produces
  them (fetched in batches of ``fetch_arraysize``), which keeps memory
  flat on multi-million-row pulls. langchain-oracle's loader is sync;
  ``load()`` here is just an eager wrapper for parity.

The SQL string is taken **verbatim** — the caller owns query shape and
identifier escaping. Bind parameters, however, are validated: every
key in ``bind_params`` must be a valid Oracle identifier, which is
enforced before the query is ever handed to the driver.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, Field, SecretStr

from locus._oracle_pool_cache import safe_acquire
from locus.rag.stores.base import Document


if TYPE_CHECKING:
    import oracledb


_SAFE_SQL_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_$#]{0,127}$")


def _validate_sql_identifier(value: str, field_name: str) -> str:
    """Validate that a string is a safe Oracle SQL identifier.

    Mirrors the helper in :mod:`locus.memory.backends.oracle` so bind
    keys can never smuggle ``;`` or whitespace into a bound query name.
    """
    if not _SAFE_SQL_IDENTIFIER.match(value):
        msg = (
            f"Invalid {field_name}: {value!r}. "
            "Must start with a letter or underscore and contain only "
            "alphanumeric characters, underscores, $, or # (max 128 chars)."
        )
        raise ValueError(msg)
    return value


class OracleADBLoaderConfig(BaseModel):
    """Connection envelope for :class:`OracleADBLoader`.

    Field names match :class:`locus.memory.backends.oracle.OracleConfig`
    on purpose so a single dict can configure both the checkpointer and
    the loader.
    """

    # Connection (same shape as memory.backends.oracle.OracleConfig)
    dsn: str | None = None
    user: str = "admin"
    password: SecretStr = SecretStr("")
    wallet_location: str | None = None
    wallet_password: SecretStr | None = None
    host: str | None = None
    port: int = 1521
    service_name: str | None = None

    # Pool sizing — small by default; loaders are typically single-shot.
    min_pool_size: int = 1
    max_pool_size: int = 2


class OracleADBLoader(BaseModel):
    """Stream rows out of an Oracle Autonomous Database as ``Document``s.

    Example:
        >>> loader = OracleADBLoader(
        ...     dsn="mydb_low",
        ...     user="locus_app",
        ...     password=os.environ["LOCUS_DB_PASSWORD"],
        ...     wallet_location="~/.oci/wallets/mydb",
        ...     sql="SELECT id, body, author, created FROM articles WHERE topic = :topic",
        ...     bind_params={"topic": "oracle"},
        ...     content_column="body",
        ...     id_column="id",
        ...     metadata_columns=["author", "created"],
        ... )
        >>> async for doc in loader.lazy_load():
        ...     print(doc.id, doc.content[:80])
        >>> await loader.close()

    Column-to-field mapping:

    * ``content_column`` → :attr:`Document.content` (required, must be
      one of the SELECTed column names — validated lazily at fetch
      time, since the SQL is taken verbatim).
    * ``id_column`` → :attr:`Document.id` (optional; falls back to a
      generated UUID hex when omitted or when the row value is NULL).
    * ``metadata_columns`` → :attr:`Document.metadata` keyed by column
      name. When omitted, **all non-content/non-id columns** end up in
      metadata — matching langchain-oracle's default.

    CLOB / NCLOB columns are awaited via ``.read()`` before being
    placed into the Document (Oracle 26ai returns LOB locator objects
    in thin mode; calling code that doesn't read them will see the
    locator stringified, not the content).
    """

    config: OracleADBLoaderConfig = Field(default_factory=OracleADBLoaderConfig)

    sql: str
    bind_params: dict[str, Any] = Field(default_factory=dict)
    content_column: str
    id_column: str | None = None
    metadata_columns: list[str] | None = None
    fetch_arraysize: int = 100

    _pool: oracledb.AsyncConnectionPool | None = None
    _pool_loop: Any = None  # asyncio loop the pool is bound to

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        sql: str,
        content_column: str,
        bind_params: dict[str, Any] | None = None,
        id_column: str | None = None,
        metadata_columns: list[str] | None = None,
        fetch_arraysize: int = 100,
        # Connection passthrough — mirrors OracleConfig field names
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
        # Validate required string params before Pydantic touches them so
        # the error messages stay readable for the common typo cases.
        if not sql or not isinstance(sql, str):
            raise ValueError("sql is required and must be a non-empty string")
        if not content_column or not isinstance(content_column, str):
            raise ValueError("content_column is required and must be a non-empty string")

        bind_params = bind_params if bind_params is not None else {}
        if not isinstance(bind_params, dict):
            raise TypeError("bind_params must be a dict (may be empty)")

        # Bind-key safety: the SQL itself is the caller's responsibility,
        # but bind keys hit cursor.execute as a kwargs-style mapping. A
        # malformed key would surface as an oracledb DPI error far from
        # the source. Validate here for a clearer trace.
        for key in bind_params:
            _validate_sql_identifier(key, "bind_params key")

        config = OracleADBLoaderConfig(
            dsn=dsn,
            user=user,
            password=SecretStr(password) if isinstance(password, str) else password,
            wallet_location=wallet_location,
            wallet_password=SecretStr(wallet_password)
            if isinstance(wallet_password, str)
            else wallet_password,
            host=host,
            port=port,
            service_name=service_name,
            **kwargs,
        )

        super().__init__(
            config=config,
            sql=sql,
            bind_params=bind_params,
            content_column=content_column,
            id_column=id_column,
            metadata_columns=metadata_columns,
            fetch_arraysize=fetch_arraysize,
        )

    async def _get_pool(self) -> oracledb.AsyncConnectionPool:
        """Get or create the underlying async connection pool.

        Same shape as :meth:`OracleVectorStore._get_pool` so wallet
        handling stays consistent across the Oracle backends.
        """
        if self._pool is None:
            try:
                import oracledb
            except ImportError as e:  # pragma: no cover - import guard
                raise ImportError(
                    "OracleADBLoader requires the 'oracledb' package. "
                    "Install with: pip install oracledb"
                ) from e

            dsn = self.config.dsn
            if dsn is None and self.config.host and self.config.service_name:
                dsn = oracledb.makedsn(
                    self.config.host,
                    self.config.port,
                    service_name=self.config.service_name,
                )

            params: dict[str, Any] = {}
            if self.config.wallet_location:
                params["config_dir"] = self.config.wallet_location
                params["wallet_location"] = self.config.wallet_location
                # ``is not None``: SecretStr("") is falsy in Pydantic v2, but
                # "" is the auto-login (cwallet.sso) wallet idiom. See #289.
                if self.config.wallet_password is not None:
                    params["wallet_password"] = self.config.wallet_password.get_secret_value()

            # create_pool_async returns the pool directly (the "async"
            # refers to the pool type, not the constructor). Same quirk
            # OracleBackend documents.
            self._pool = oracledb.create_pool_async(
                user=self.config.user,
                password=self.config.password.get_secret_value(),
                dsn=dsn,
                min=self.config.min_pool_size,
                max=self.config.max_pool_size,
                **params,
            )

        return self._pool

    @staticmethod
    async def _materialise(value: Any) -> Any:
        """Read a CLOB / NCLOB locator if present, otherwise pass through.

        oracledb in thin mode returns ``AsyncLOB`` for CLOB columns; the
        consumer must ``await .read()`` to get the text. Anything else
        (str, int, datetime, None) is returned as-is.
        """
        if value is not None and hasattr(value, "read"):
            return await value.read()
        return value

    async def lazy_load(self) -> AsyncIterator[Document]:
        """Stream rows from the SQL as ``Document`` instances.

        Yields one Document per row. The cursor's arraysize is set to
        :attr:`fetch_arraysize` so the driver bulk-fetches in batches
        without us pulling everything into Python memory.
        """
        pool = await self._get_pool()

        async with safe_acquire(self, pool) as conn, conn.cursor() as cursor:
            cursor.arraysize = self.fetch_arraysize
            await cursor.execute(self.sql, self.bind_params)

            # description is a list of (name, type, ...) tuples after
            # execute(). Lowercase for case-insensitive lookup against
            # the user-supplied column names.
            columns = [d[0] for d in cursor.description]
            col_index = {name.lower(): idx for idx, name in enumerate(columns)}

            content_idx = col_index.get(self.content_column.lower())
            if content_idx is None:
                raise ValueError(
                    f"content_column {self.content_column!r} not found in SELECT columns: {columns}"
                )

            id_idx = col_index.get(self.id_column.lower()) if self.id_column else None

            if self.metadata_columns is not None:
                meta_indices = [
                    (name, col_index[name.lower()])
                    for name in self.metadata_columns
                    if name.lower() in col_index
                ]
            else:
                # Default: everything that isn't content/id becomes metadata.
                skip = {content_idx}
                if id_idx is not None:
                    skip.add(id_idx)
                meta_indices = [(columns[i], i) for i in range(len(columns)) if i not in skip]

            while True:
                rows = await cursor.fetchmany(self.fetch_arraysize)
                if not rows:
                    break
                for row in rows:
                    content_val = await self._materialise(row[content_idx])
                    content_str = (
                        content_val
                        if isinstance(content_val, str)
                        else ""
                        if content_val is None
                        else str(content_val)
                    )

                    if id_idx is not None:
                        id_raw = await self._materialise(row[id_idx])
                        doc_id = str(id_raw) if id_raw is not None else uuid4().hex
                    else:
                        doc_id = uuid4().hex

                    metadata: dict[str, Any] = {}
                    for name, idx in meta_indices:
                        metadata[name] = await self._materialise(row[idx])

                    yield Document(
                        id=doc_id,
                        content=content_str,
                        metadata=metadata,
                    )

    async def load(self) -> list[Document]:
        """Eagerly materialise :meth:`lazy_load` into a list.

        Convenience wrapper for callers that don't need streaming —
        e.g. small reference tables, or sync code paths that just want
        to feed a vector store ``add_batch``.
        """
        return [doc async for doc in self.lazy_load()]

    async def close(self) -> None:
        """Close the underlying pool, if one was opened."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def __repr__(self) -> str:
        target = self.config.dsn or self.config.host or "<no-target>"
        return (
            f"OracleADBLoader(target={target!r}, "
            f"content_column={self.content_column!r}, "
            f"id_column={self.id_column!r})"
        )
