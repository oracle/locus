# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Oracle 23ai/26ai in-database embedding generation.

Native locus wrapper around ``DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDING`` and
``DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDINGS``. Functionally equivalent to
``langchain-oracle``'s ``OracleEmbeddings(provider="database")`` but with
**zero** langchain / langgraph imports — locus owns the contract.

What this is for
----------------

Oracle 23ai / 26ai can host ONNX embedding models *inside* the database
(see ``DBMS_VECTOR.LOAD_ONNX_MODEL``). When the model lives in the DB
the embedding generation happens DB-side: the application ships text
over the wire, the database produces the vector locally, and the
caller gets back a serialized ``VECTOR`` ready to write into a
``VECTOR`` column. This is the canonical pattern when:

* **Data residency / sovereignty** rules forbid sending raw text out
  to OCI GenAI / OpenAI / a third-party endpoint.
* **Latency budget** can't absorb a round-trip to a remote inference
  service.
* The ONNX model is already loaded into the DB and you want one
  pipeline for both embedding *and* storage.

Prerequisites
-------------

* Oracle Database 23ai (or newer) with the AI Vector Search option.
* The desired ONNX model loaded into the database. Typical path::

      BEGIN
          DBMS_VECTOR.LOAD_ONNX_MODEL(
              directory  => 'DM_DUMP',
              file_name  => 'all_MiniLM_L12_v2.onnx',
              model_name => 'ALL_MINILM_L12_V2');
      END;
      /

* The connecting user has ``EXECUTE`` on ``DBMS_VECTOR_CHAIN`` and
  read access on the model::

      GRANT EXECUTE ON DBMS_VECTOR_CHAIN TO locus_app;
      GRANT MINING MODEL SELECT ON ALL_MINILM_L12_V2 TO locus_app;

Usage
-----

::

    from locus.rag.embeddings.oracle_indb import OracleInDBEmbeddings

    emb = OracleInDBEmbeddings(
        model_name="ALL_MINILM_L12_V2",
        dimension=384,
        dsn="mydb_low",
        user="locus_app",
        password="...",
        wallet_location="~/.oci/wallets/mydb",
    )
    vec = await emb.embed("hello world")
    vecs = await emb.embed_batch(["a", "b", "c"])
    await emb.close()

The SQL the wrapper executes
----------------------------

Single text → one ``VECTOR``::

    SELECT TO_CLOB(VECTOR_SERIALIZE(
        DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDING(
            :text,
            JSON('{"provider":"database","model":"<MODEL_NAME>"}')
        )
    )) AS emb
    FROM dual

Batch — ``UTL_TO_EMBEDDINGS`` over a chunk-object JSON array, with
``JSON_TABLE`` to break the response back into one row per input::

    SELECT et.embed_vector, et.embed_id
    FROM TABLE(DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDINGS(
        JSON_ARRAY(
          JSON_OBJECT('chunk_id' VALUE 1, 'chunk_data' VALUE :t0),
          JSON_OBJECT('chunk_id' VALUE 2, 'chunk_data' VALUE :t1),
          ... RETURNING CLOB
        ),
        JSON('{"provider":"database","model":"<MODEL_NAME>"}')
    )) t,
    JSON_TABLE(t.column_value, '$' COLUMNS (
      embed_id INTEGER PATH '$.embed_id',
      embed_vector CLOB PATH '$.embed_vector'
    )) et
    ORDER BY et.embed_id

The ``VECTOR_SERIALIZE`` (single-input) / ``embed_vector`` (batch)
outputs are the canonical ``[0.123, -0.456, …]`` text form which the
wrapper parses into ``list[float]`` for the
:class:`EmbeddingResult.embedding` field.

``use_batch_function`` controls whether the batch path uses
``UTL_TO_EMBEDDINGS`` or loops ``UTL_TO_EMBEDDING`` per text. The
default is **False** — sequential — because the batch TABLE function
is unreliable on Autonomous Database 23.26 (zero rows from the
``VECTOR_ARRAY_T`` overload, single combined embedding from the CLOB
overload). Enable explicitly on DBs where the batch path is known to
work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, SecretStr

from locus._oracle_pool_cache import safe_acquire
from locus.memory.backends._oracle_config import validate_sql_identifier
from locus.rag.embeddings.base import (
    BaseEmbedding,
    EmbeddingCapabilities,
    EmbeddingConfig,
    EmbeddingResult,
)


if TYPE_CHECKING:
    import oracledb


class OracleInDBEmbeddingsConfig(BaseModel):
    """Configuration envelope for :class:`OracleInDBEmbeddings`.

    Same connection shape as
    :class:`locus.memory.backends._oracle_config.OracleConfig` — DSN +
    optional wallet, or host triplet — plus the ONNX model identity.
    """

    # ONNX model identity (inside the DB)
    model_name: str = Field(
        ...,
        description="Name of the ONNX model loaded into the DB (DBMS_VECTOR.LOAD_ONNX_MODEL)",
    )
    dimension: int = Field(
        ...,
        description="Output embedding dimension (known a priori for the chosen model)",
    )

    # Connection options
    dsn: str | None = None
    user: str = "admin"
    password: SecretStr = SecretStr("")

    # Wallet (Autonomous DB / mTLS)
    wallet_location: str | None = None
    wallet_password: SecretStr | None = None

    # Host triplet (alternative to DSN)
    host: str | None = None
    port: int = 1521
    service_name: str | None = None

    # Pool settings
    min_pool_size: int = 1
    max_pool_size: int = 5

    # When False (default), batch calls loop ``UTL_TO_EMBEDDING`` per
    # text rather than calling ``UTL_TO_EMBEDDINGS``. The batch TABLE
    # function is unreliable on Oracle Autonomous Database 23.26 — the
    # VECTOR_ARRAY_T overload returns zero rows for many in-DB ONNX
    # models, and the CLOB overload collapses every input into a single
    # combined embedding. Set this True only if your DB exposes a
    # working batch path (older on-prem 23ai patches in some cases).
    use_batch_function: bool = False

    def model_post_init(self, _: Any, /) -> None:
        """Validate identifiers and positive dimension."""
        # model_name is spliced into the JSON literal of the SQL call; the
        # JSON parser would happily accept "X","Y":"injection but we
        # constrain it to a SQL identifier shape anyway, both as
        # defence-in-depth and because Oracle model names *must* match
        # that shape on disk.
        validate_sql_identifier(self.model_name, "model_name")
        if self.dimension < 1:
            msg = f"dimension must be a positive int, got {self.dimension}"
            raise ValueError(msg)


class OracleInDBEmbeddings(BaseEmbedding):
    """Oracle 23ai/26ai in-database embedding generator.

    Calls ``DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDING`` /
    ``UTL_TO_EMBEDDINGS`` over an async ``oracledb`` pool, parses
    the ``VECTOR_SERIALIZE`` text representation into ``list[float]``,
    and returns :class:`EmbeddingResult` for parity with the rest of
    the locus embedding providers.

    Example::

        emb = OracleInDBEmbeddings(
            model_name="ALL_MINILM_L12_V2",
            dimension=384,
            dsn="mydb_low",
            user="locus_app",
            password="...",
            wallet_location="~/.oci/wallets/mydb",
        )
        vec = await emb.embed("hello world")
        # vec.embedding is list[float] of length 384

    See module docstring for prerequisite DB grants and ONNX model
    loading.
    """

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
        use_batch_function: bool = False,
    ) -> None:
        super().__init__()
        self._cfg = OracleInDBEmbeddingsConfig(
            model_name=model_name,
            dimension=dimension,
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
            min_pool_size=min_pool_size,
            max_pool_size=max_pool_size,
            use_batch_function=use_batch_function,
        )
        self._pool: oracledb.AsyncConnectionPool | None = None

    _pool_loop: Any = None  # asyncio loop the pool is bound to

    # -- Public introspection -----------------------------------------------

    @property
    def model_name(self) -> str:
        """Name of the in-DB ONNX model this embedder uses."""
        return self._cfg.model_name

    @property
    def config(self) -> EmbeddingConfig:
        """Embedding configuration.

        ``dimension`` is supplied by the caller (it's known a priori
        for any ONNX model loaded into the DB). ``max_tokens`` /
        ``batch_size`` use generic safe defaults — the DB itself
        truncates per the model's tokenizer.
        """
        return EmbeddingConfig(
            dimension=self._cfg.dimension,
            max_tokens=8192,
            batch_size=96,
        )

    @property
    def capabilities(self) -> EmbeddingCapabilities:
        """Capabilities for the in-DB embedder.

        ``supports_batching`` flips with ``use_batch_function`` — when
        the batch SQL is disabled the wrapper still implements
        :meth:`embed_batch` but it loops single calls internally.
        """
        return EmbeddingCapabilities(
            supports_query_vs_doc=False,
            supports_multimodal=False,
            supports_batching=self._cfg.use_batch_function,
            max_batch_size=96 if self._cfg.use_batch_function else 1,
            max_input_tokens=8192,
        )

    # -- Pool ---------------------------------------------------------------

    async def _get_pool(self) -> oracledb.AsyncConnectionPool:
        """Lazily create the oracledb async pool.

        ``oracledb`` is imported inside the function (not at module
        load) so installs without the driver can still ``import`` this
        module — same pattern as :class:`OracleStore`.
        """
        if self._pool is None:
            try:
                import oracledb
            except ImportError as e:
                msg = (
                    "OracleInDBEmbeddings requires the 'oracledb' package. "
                    "Install with: pip install oracledb"
                )
                raise ImportError(msg) from e

            cfg = self._cfg
            dsn = cfg.dsn
            if dsn is None and cfg.host and cfg.service_name:
                dsn = oracledb.makedsn(cfg.host, cfg.port, service_name=cfg.service_name)

            params: dict[str, Any] = {}
            if cfg.wallet_location:
                params["config_dir"] = cfg.wallet_location
                params["wallet_location"] = cfg.wallet_location
                # ``is not None``: SecretStr("") is falsy in Pydantic v2, but
                # "" is the auto-login (cwallet.sso) wallet idiom. See #289.
                if cfg.wallet_password is not None:
                    params["wallet_password"] = cfg.wallet_password.get_secret_value()

            self._pool = oracledb.create_pool_async(
                user=cfg.user,
                password=cfg.password.get_secret_value(),
                dsn=dsn,
                min=cfg.min_pool_size,
                max=cfg.max_pool_size,
                **params,
            )
        return self._pool

    # -- SQL builders -------------------------------------------------------

    def _single_sql(self) -> str:
        """SQL for ``UTL_TO_EMBEDDING`` — single text → single vector.

        The model name is *interpolated* (not bound) because it lives
        inside a JSON literal that Oracle parses at execution time;
        ``validate_sql_identifier`` constrains it to a SQL-safe shape.
        Only the text payload is bound, as ``:text``.
        """
        return (
            "SELECT TO_CLOB(VECTOR_SERIALIZE("
            "DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDING("
            ":text, "
            f"""JSON('{{"provider":"database","model":"{self._cfg.model_name}"}}')"""
            "))) AS emb "
            "FROM dual"
        )

    def _batch_sql(self, n: int) -> str:
        """SQL for ``UTL_TO_EMBEDDINGS`` — N chunk objects → N vectors.

        ``UTL_TO_EMBEDDINGS`` expects each input as a chunk object
        with ``chunk_id`` + ``chunk_data`` keys, not a bare string.
        Texts are bound as ``:t0, :t1, ..., :t{n-1}``; the returned
        rows are CLOBs containing ``{embed_id, embed_data, embed_vector}``
        — ``JSON_TABLE`` extracts the ``embed_vector`` field, and
        ``embed_id`` preserves caller order.
        """
        chunks = ", ".join(
            f"JSON_OBJECT('chunk_id' VALUE {i + 1}, 'chunk_data' VALUE :t{i})" for i in range(n)
        )
        return (
            "SELECT et.embed_vector, et.embed_id "
            "FROM TABLE(DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDINGS("
            f"JSON_ARRAY({chunks} RETURNING CLOB), "
            f"""JSON('{{"provider":"database","model":"{self._cfg.model_name}"}}')"""
            ")) t, "
            "JSON_TABLE(t.column_value, '$' COLUMNS ("
            "  embed_id INTEGER PATH '$.embed_id', "
            "  embed_vector CLOB PATH '$.embed_vector'"
            ")) et "
            "ORDER BY et.embed_id"
        )

    # -- VECTOR_SERIALIZE parser -------------------------------------------

    @staticmethod
    def _parse_serialized_vector(text: str) -> list[float]:
        """Parse ``"[0.123, -0.456, 7.89]"`` into ``[0.123, -0.456, 7.89]``.

        The serialized form is JSON-array-shaped but we can't blindly
        use ``json.loads`` because the floats may use ``E``-notation
        without a leading digit before the decimal in some locales —
        manual split keeps the parser lenient on whitespace and on a
        trailing newline.
        """
        if text is None:
            msg = "VECTOR_SERIALIZE returned NULL — model may not exist or input was empty"
            raise ValueError(msg)
        stripped = text.strip()
        if not (stripped.startswith("[") and stripped.endswith("]")):
            msg = f"unexpected VECTOR_SERIALIZE output (no brackets): {text!r}"
            raise ValueError(msg)
        inner = stripped[1:-1].strip()
        if not inner:
            return []
        return [float(part) for part in inner.split(",")]

    @staticmethod
    async def _read_clob(value: Any) -> str:
        """Resolve a value that may be a ``str`` or an ``AsyncLOB``.

        Oracle returns ``TO_CLOB(...)`` as an async LOB by default;
        callers can also set ``cursor.outputtypehandler`` to fetch as
        ``str``. Handle both transparently.
        """
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        # AsyncLOB exposes .read() that returns a coroutine. If a
        # synchronous LOB sneaks through, .read() is still callable.
        read = value.read()
        if hasattr(read, "__await__"):
            return await read  # type: ignore[no-any-return]
        return read  # type: ignore[no-any-return]

    # -- embed / embed_batch ------------------------------------------------

    async def embed(self, text: str) -> EmbeddingResult:
        """Embed a single text via ``UTL_TO_EMBEDDING``.

        Returns an :class:`EmbeddingResult` with the parsed vector,
        the original text, and the model name. ``tokens`` is left
        ``None`` because the DB doesn't surface a token count via
        this call path.
        """
        pool = await self._get_pool()
        async with safe_acquire(self, pool) as conn, conn.cursor() as cursor:
            await cursor.execute(self._single_sql(), {"text": text})
            row = await cursor.fetchone()
        if row is None:
            msg = "UTL_TO_EMBEDDING returned no rows"
            raise RuntimeError(msg)
        clob_value = row[0]
        serialized = await self._read_clob(clob_value)
        vector = self._parse_serialized_vector(serialized)
        return EmbeddingResult(
            embedding=vector,
            text=text,
            model=self._cfg.model_name,
            tokens=None,
        )

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed multiple texts.

        Uses ``UTL_TO_EMBEDDINGS`` when ``use_batch_function`` is True
        (default). Falls back to a sequential loop of
        ``UTL_TO_EMBEDDING`` calls when the batch function is unavailable
        on the target DB (older 23ai patch levels) — still cheaper than
        opening a fresh connection per text because the loop reuses one
        pool connection.
        """
        if not texts:
            return []

        if not self._cfg.use_batch_function:
            return [await self.embed(t) for t in texts]

        binds = {f"t{i}": t for i, t in enumerate(texts)}
        sql = self._batch_sql(len(texts))

        pool = await self._get_pool()
        async with safe_acquire(self, pool) as conn, conn.cursor() as cursor:
            await cursor.execute(sql, binds)
            rows = await cursor.fetchall()

        # rownum-ordered, so row i corresponds to texts[i].
        results: list[EmbeddingResult] = []
        for i, row in enumerate(rows):
            serialized = await self._read_clob(row[0])
            vector = self._parse_serialized_vector(serialized)
            results.append(
                EmbeddingResult(
                    embedding=vector,
                    text=texts[i],
                    model=self._cfg.model_name,
                    tokens=None,
                )
            )
        return results

    async def embed_query(self, query: str) -> EmbeddingResult:
        """Embed a query — alias of :meth:`embed`.

        In-DB ONNX embedding models don't differentiate query vs
        document spaces; the same SQL path is used for both.
        """
        return await self.embed(query)

    # -- Lifecycle ----------------------------------------------------------

    async def close(self) -> None:
        """Release the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def __repr__(self) -> str:
        return (
            f"OracleInDBEmbeddings(model_name={self._cfg.model_name!r}, "
            f"dimension={self._cfg.dimension})"
        )
