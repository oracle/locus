# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Oracle 26ai Vector Store - Native vector support.

Uses Oracle Database 23ai/26ai with native VECTOR data type.
Requires python-oracledb in thin mode (no Oracle Client needed).
"""

from __future__ import annotations

import json
import re
import warnings
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from pydantic import BaseModel, Field, SecretStr

from locus._oracle_ddl import provisioning_lock, run_ddl
from locus._oracle_pool_cache import safe_acquire
from locus.core.errors import VectorStoreError
from locus.core.warnings import LocusWarning
from locus.memory.backends._oracle_config import connect_params
from locus.rag.stores._mmr import mmr_rerank as _mmr_rerank
from locus.rag.stores._oracle_filter import compile_metadata_filter as _compile_metadata_filter
from locus.rag.stores.base import (
    BaseVectorStore,
    Document,
    SearchResult,
    VectorStoreConfig,
)


if TYPE_CHECKING:
    import oracledb


_SAFE_SQL_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_$#]{0,127}$")
_ALLOWED_DISTANCE_METRICS = frozenset({"COSINE", "DOT", "EUCLIDEAN", "MANHATTAN", "HAMMING"})
_ALLOWED_INDEX_TYPES = frozenset({"HNSW", "IVF", "NONE"})


def _validate_sql_identifier(value: str, field_name: str) -> str:
    """Validate that a string is a safe Oracle SQL identifier."""
    if not _SAFE_SQL_IDENTIFIER.match(value):
        msg = (
            f"Invalid {field_name}: {value!r}. "
            "Must start with a letter or underscore and contain only "
            "alphanumeric characters, underscores, $, or # (max 128 chars)."
        )
        raise ValueError(msg)
    return value


def _validate_int_in_range(
    value: int | None,
    field_name: str,
    min_value: int,
    max_value: int | None = None,
) -> None:
    """Bounds-check an int parameter before it reaches DDL.

    Mirrors langchain-oracle's ``_validate_int_param`` (oracle/langchain-oracle#215)
    so vector-index tuning knobs can't smuggle a bool or an out-of-range
    integer into a CREATE VECTOR INDEX statement.
    """
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer.")
    if value < min_value:
        raise ValueError(f"{field_name} must be at least {min_value}.")
    if max_value is not None and value > max_value:
        raise ValueError(f"{field_name} must be at most {max_value}.")


class OracleVectorConfig(BaseModel):
    """Configuration for Oracle Vector Store."""

    # Connection options
    dsn: str | None = None
    user: str = "admin"
    password: SecretStr = SecretStr("")

    # Directory holding tnsnames.ora, for a TNS alias without a wallet
    config_dir: str | None = None

    # For Autonomous Database with wallet
    wallet_location: str | None = None
    wallet_password: SecretStr | None = None

    # Connection string components (alternative to DSN)
    host: str | None = None
    port: int = 1521
    service_name: str | None = None

    # Table settings
    table_name: str = "locus_vectors"
    schema_name: str | None = None

    # Column overrides — let an existing table written by another tool be read
    # without re-ingestion. The langchain_oracledb.OracleVS schema, for
    # example, uses ``text`` instead of ``content`` and has no ``created_at``
    # column. Set ``content_column="text"`` and ``created_at_column=None`` to
    # point at one of those tables.
    id_column: str = "id"
    content_column: str = "content"
    embedding_column: str = "embedding"
    metadata_column: str = "metadata"
    created_at_column: str | None = "created_at"

    # When False, ``_ensure_table`` won't issue CREATE statements — useful for
    # read-only attachment to a foreign-schema table populated by a different
    # ingestion pipeline.
    auto_create_table: bool = True

    # Vector settings
    dimension: int = 1024  # Cohere embed-v3 default
    distance_metric: str = "COSINE"  # COSINE, DOT, EUCLIDEAN, MANHATTAN, HAMMING

    # Vector index settings — Oracle 23ai/26ai supports two
    # ORGANIZATION clauses for native VECTOR columns:
    #
    #   * ``HNSW`` — Hierarchical Navigable Small World. Memory-resident,
    #     best for sub-second query latency on large corpora; tune with
    #     ``hnsw_neighbors`` (M) and ``hnsw_ef_construction``. Needs the
    #     database's vector memory pool: without one (Oracle Database
    #     Free as installed) provisioning warns and searches the table
    #     with an exact scan — see ``_needs_vector_memory``.
    #   * ``IVF`` — Inverted File w/ neighbor partitions. Disk-friendly,
    #     ideal when the index can't fit in memory; tune with
    #     ``ivf_neighbor_partitions``.
    #   * ``NONE`` — skip CREATE VECTOR INDEX entirely. The store still
    #     works (exact scan over VECTOR_DISTANCE); useful for tiny demos
    #     or pre-created indexes managed out-of-band.
    #
    # ``accuracy`` / ``parallel`` apply to both index types.
    # All knobs are validated bounds-checked before reaching DDL.
    index_type: str = "HNSW"
    hnsw_neighbors: int | None = None  # 2..2048 (Oracle default ~32)
    hnsw_ef_construction: int | None = None  # 1..65535 (Oracle default ~200)
    ivf_neighbor_partitions: int | None = None  # 1..10_000_000
    index_accuracy: int | None = None  # 1..100 (Oracle default 95)
    index_parallel: int | None = None  # >=1

    # Pool settings
    min_pool_size: int = 1
    max_pool_size: int = 5

    def model_post_init(self, __context: Any) -> None:
        """Validate SQL identifiers and distance metric to prevent injection."""
        _validate_sql_identifier(self.table_name, "table_name")
        if self.schema_name is not None:
            _validate_sql_identifier(self.schema_name, "schema_name")
        for col_field in (
            "id_column",
            "content_column",
            "embedding_column",
            "metadata_column",
        ):
            _validate_sql_identifier(getattr(self, col_field), col_field)
        if self.created_at_column is not None:
            _validate_sql_identifier(self.created_at_column, "created_at_column")
        if self.distance_metric.upper() not in _ALLOWED_DISTANCE_METRICS:
            raise ValueError(
                f"Invalid distance_metric: {self.distance_metric!r}. "
                f"Must be one of: {sorted(_ALLOWED_DISTANCE_METRICS)}"
            )
        if self.index_type.upper() not in _ALLOWED_INDEX_TYPES:
            raise ValueError(
                f"Invalid index_type: {self.index_type!r}. "
                f"Must be one of: {sorted(_ALLOWED_INDEX_TYPES)}"
            )
        _validate_int_in_range(self.hnsw_neighbors, "hnsw_neighbors", 2, 2048)
        _validate_int_in_range(self.hnsw_ef_construction, "hnsw_ef_construction", 1, 65535)
        _validate_int_in_range(
            self.ivf_neighbor_partitions, "ivf_neighbor_partitions", 1, 10_000_000
        )
        _validate_int_in_range(self.index_accuracy, "index_accuracy", 1, 100)
        _validate_int_in_range(self.index_parallel, "index_parallel", 1)


def _needs_vector_memory(exc: BaseException) -> bool:
    """Whether ``exc`` is ``ORA-51962``: no room in the vector memory pool for the index.

    An HNSW index lives in the database's vector memory pool. Oracle
    Database Free sets ``vector_memory_size`` to 0 as installed, so its
    first ``CREATE VECTOR INDEX ... INMEMORY NEIGHBOR GRAPH`` fails with
    this error; a sized pool that is already full fails the same way.
    """
    return "ORA-51962" in str(exc)


def _vector_memory_hint(index_type: str, table: str) -> str:
    """What to do about ``ORA-51962`` on ``table``'s ``index_type`` index."""
    return (
        f"The {index_type} vector index on {table} needs room in the database's "
        "vector memory pool and there is none (ORA-51962; vector_memory_size is 0 "
        'on Oracle Database Free as installed). Pass index_type="IVF" for an '
        'on-disk index, index_type="NONE" to search without one, or size '
        "vector_memory_size."
    )


_TEXT_QUERY_MAX_TERMS = 20


def _oracle_text_query(text: str) -> str | None:
    """Turn free text into an Oracle Text query that scores by terms matched.

    Each whitespace-separated token is wrapped in ``{}`` so hyphens,
    operators (``&``, ``|``, ``-``), reserved words (``AND``, ``NEAR``) and
    stopwords are matched as plain words rather than parsed as query syntax,
    and the tokens are joined with ``ACCUM``, which ranks a row by how many
    of the terms it contains. A brace inside a token would end the escape,
    so braces are dropped. Returns ``None`` when the text holds no term.
    """
    terms: list[str] = []
    for token in text.split():
        term = token.replace("{", "").replace("}", "")
        if term:
            terms.append("{" + term + "}")
        if len(terms) >= _TEXT_QUERY_MAX_TERMS:
            break
    return " ACCUM ".join(terms) if terms else None


class OracleVectorStore(BaseModel, BaseVectorStore):
    """
    Oracle 26ai Vector Store with native VECTOR support.

    Uses Oracle Database 23ai/26ai's native VECTOR data type for
    efficient similarity search. Supports cosine, dot product,
    and Euclidean distance metrics.

    **Production setup — use a least-privileged schema, not ADMIN.**

    Using ``ADMIN`` against an Autonomous Database is an Oracle security
    anti-pattern: every connection runs with full DBA privileges, so a
    compromised credential or a malformed query has unbounded blast
    radius. Create a dedicated application schema with only the
    privileges this store needs. Run once as ADMIN to provision::

        -- Create a least-privileged owner for Locus's vector tables.
        CREATE USER locus_app IDENTIFIED BY "<strong-password>";
        GRANT CONNECT, RESOURCE TO locus_app;
        ALTER USER locus_app QUOTA 1G ON DATA;

        -- (Optional) pre-create the table yourself so locus runs with
        -- DML-only privileges. See ``auto_create_table=False`` below.
        CREATE TABLE locus_app.locus_documents (
            id            VARCHAR2(255) PRIMARY KEY,
            content       CLOB,
            embedding     VECTOR(1024, FLOAT32),
            metadata      CLOB DEFAULT '{}' CHECK (metadata IS JSON)
        );
        CREATE VECTOR INDEX idx_locus_documents_vec
            ON locus_app.locus_documents (embedding)
            ORGANIZATION NEIGHBOR PARTITIONS
            WITH DISTANCE COSINE;

    Then connect as the app user — never ADMIN — at application startup.

    **Table provisioning: auto vs. pre-create.**

    ``auto_create_table=True`` (the default) issues ``CREATE TABLE`` and
    ``CREATE VECTOR INDEX`` on first use. Convenient for demos and
    notebooks; **requires DDL privileges** on the schema. For production
    workloads use ``auto_create_table=False`` and pre-create the table
    out-of-band (DDL above) so the application user can be restricted
    to ``INSERT`` / ``SELECT`` / ``UPDATE`` on the table only.

    The default ``HNSW`` index lives in the database's vector memory
    pool. On a database without one (Oracle Database Free sets
    ``vector_memory_size`` to 0 as installed) the index cannot be built:
    the store warns with :class:`~locus.core.warnings.LocusWarning` and
    searches the table with an exact scan, as ``index_type="NONE"``
    does. Pass ``index_type="IVF"`` for an on-disk index there.

    Example with DSN (least-privileged app schema):
        >>> store = OracleVectorStore(
        ...     dsn="mydb_high",
        ...     user="locus_app",
        ...     password=os.environ["LOCUS_DB_PASSWORD"],
        ...     wallet_location="~/.oci/wallets/mydb",
        ...     dimension=1024,
        ... )
        >>> await store.add(document)
        >>> results = await store.search(query_embedding, limit=5)

    Example with host/service_name + pre-created table:
        >>> store = OracleVectorStore(
        ...     host="adb.us-ashburn-1.oraclecloud.com",
        ...     port=1522,
        ...     service_name="xxx_high.adb.oraclecloud.com",
        ...     user="locus_app",
        ...     password=os.environ["LOCUS_DB_PASSWORD"],
        ...     auto_create_table=False,  # locus_app has DML only
        ...     dimension=1024,
        ... )

    Example attaching to an existing langchain_oracledb-formatted table
    (column names differ, no ``created_at`` column, table already
    exists):
        >>> store = OracleVectorStore(
        ...     dsn="mydb_low",
        ...     user="locus_app",
        ...     password=os.environ["LOCUS_DB_PASSWORD"],
        ...     wallet_location="~/.oci/wallets/mydb",
        ...     table_name="VECTOR_DOCUMENTS",
        ...     content_column="text",
        ...     created_at_column=None,
        ...     auto_create_table=False,  # don't try to CREATE TABLE
        ...     dimension=1536,  # match the existing column
        ... )
    """

    oracle_config: OracleVectorConfig = Field(default_factory=OracleVectorConfig)
    _pool: oracledb.AsyncConnectionPool | None = None
    _pool_loop: Any = None  # asyncio loop the pool is bound to
    _initialized: bool = False
    # The lock the first-time DDL runs under, per event loop (``locus._oracle_ddl``).
    _provision_lock: Any = None
    _provision_lock_loop: Any = None

    model_config = {"arbitrary_types_allowed": True}

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
        oracle_config = OracleVectorConfig(
            dsn=dsn,
            user=user,
            password=SecretStr(password) if isinstance(password, str) else password,
            host=host,
            port=port,
            service_name=service_name,
            dimension=dimension,
            distance_metric=distance_metric,
            **kwargs,
        )
        super().__init__(oracle_config=oracle_config)

    @property
    def config(self) -> VectorStoreConfig:
        """Get store configuration."""
        return VectorStoreConfig(
            dimension=self.oracle_config.dimension,
            distance_metric=self.oracle_config.distance_metric.lower(),
            index_type="hnsw",
        )

    async def _get_pool(self) -> oracledb.AsyncConnectionPool:
        """Get or create connection pool, rebuilding on loop change."""
        from locus._oracle_pool_cache import get_pool

        def _build() -> oracledb.AsyncConnectionPool:
            try:
                import oracledb
            except ImportError as e:
                raise ImportError(
                    "OracleVectorStore requires the 'oracledb' package. "
                    "Install with: pip install oracledb"
                ) from e

            dsn = self.oracle_config.dsn
            if dsn is None and self.oracle_config.host and self.oracle_config.service_name:
                dsn = oracledb.makedsn(
                    self.oracle_config.host,
                    self.oracle_config.port,
                    service_name=self.oracle_config.service_name,
                )

            return oracledb.create_pool_async(
                user=self.oracle_config.user,
                password=self.oracle_config.password.get_secret_value(),
                dsn=dsn,
                min=self.oracle_config.min_pool_size,
                max=self.oracle_config.max_pool_size,
                **connect_params(self.oracle_config),
            )

        return cast("oracledb.AsyncConnectionPool", await get_pool(self, _build))

    @property
    def _full_table_name(self) -> str:
        """Get fully qualified table name."""
        if self.oracle_config.schema_name:
            return f"{self.oracle_config.schema_name}.{self.oracle_config.table_name}"
        return self.oracle_config.table_name

    async def _ensure_table(self) -> None:
        """Create table if not exists, unless ``auto_create_table=False``.

        One provisioning pass per instance; a ``CREATE`` another process won
        or is still running is handled (see ``locus._oracle_ddl``).
        """
        if self._initialized:
            return

        async with provisioning_lock(self):
            if self._initialized:
                return
            await self._provision()
            self._initialized = True

    async def _provision(self) -> None:
        pool = await self._get_pool()
        cfg = self.oracle_config
        dim = cfg.dimension
        id_col = cfg.id_column
        content_col = cfg.content_column
        embedding_col = cfg.embedding_column
        metadata_col = cfg.metadata_column
        created_col = cfg.created_at_column

        async with safe_acquire(self, pool) as conn, conn.cursor() as cursor:
            # Check if table exists
            await cursor.execute(
                """
                    SELECT COUNT(*) FROM user_tables
                    WHERE table_name = UPPER(:table_name)
                    """,
                {"table_name": cfg.table_name},
            )
            result = await cursor.fetchone()
            table_exists = result[0] > 0 if result else False

            if not table_exists and cfg.auto_create_table:
                # Create table with VECTOR column (Oracle 23ai/26ai)
                created_at_ddl = (
                    f", {created_col} TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP"
                    if created_col
                    else ""
                )
                await run_ddl(
                    cursor,
                    f"""
                        CREATE TABLE {self._full_table_name} (
                            {id_col} VARCHAR2(255) PRIMARY KEY,
                            {content_col} CLOB,
                            {embedding_col} VECTOR({dim}, FLOAT32),
                            {metadata_col} CLOB DEFAULT '{{}}'
                                CHECK ({metadata_col} IS JSON)
                            {created_at_ddl}
                        )
                    """,
                )

                # Create the vector index using the configured organisation
                # (HNSW / IVF / NONE). Centralised in _vector_index_ddl so
                # build_index() can fire the same DDL out-of-band later.
                ddl = self._vector_index_ddl()
                if ddl is not None:
                    try:
                        await run_ddl(cursor, ddl)
                    except Exception as exc:
                        if not _needs_vector_memory(exc):
                            raise
                        # Oracle commits each DDL on its own, so the table
                        # is there; keep it and search it with an exact
                        # scan, as ``index_type="NONE"`` would.
                        hint = _vector_memory_hint(cfg.index_type.upper(), self._full_table_name)
                        warnings.warn(
                            f"{hint} Searching {self._full_table_name} with an exact scan.",
                            LocusWarning,
                            stacklevel=2,
                        )

                if created_col:
                    await run_ddl(
                        cursor,
                        f"""
                            CREATE INDEX idx_{cfg.table_name}_created
                            ON {self._full_table_name} ({created_col} DESC)
                        """,
                    )

                await conn.commit()
            # When auto_create_table=False and the table is missing, we leave
            # it for the first SQL operation to surface ORA-00942 with full
            # table context.

    @property
    def _vector_index_name(self) -> str:
        return f"idx_{self.oracle_config.table_name}_vec"

    def _vector_index_ddl(self) -> str | None:
        """Build the CREATE VECTOR INDEX DDL for the configured index type.

        Returns ``None`` when ``index_type == "NONE"`` so the caller can
        skip the DDL entirely (exact-scan fallback). Centralising the DDL
        here means ``_ensure_table`` and ``build_index()`` always emit
        the same statement.

        Grammar (Oracle 23ai/26ai)::

            CREATE VECTOR INDEX <idx> ON <table> (<col>)
              ORGANIZATION {INMEMORY NEIGHBOR GRAPH | NEIGHBOR PARTITIONS}
              [WITH DISTANCE <metric>]
              [WITH TARGET ACCURACY <n>]
              [PARAMETERS (TYPE {HNSW|IVF} [, NEIGHBORS <n>]
                                          [, EFCONSTRUCTION <n>]
                                          [, NEIGHBOR_PARTITIONS <n>])]
              [PARALLEL <n>]
        """
        cfg = self.oracle_config
        index_type = cfg.index_type.upper()
        if index_type == "NONE":
            return None

        idx_name = self._vector_index_name
        param_pairs: list[str] = [f"TYPE {index_type}"]
        if index_type == "HNSW":
            organisation = "ORGANIZATION INMEMORY NEIGHBOR GRAPH"
            if cfg.hnsw_neighbors is not None:
                param_pairs.append(f"NEIGHBORS {cfg.hnsw_neighbors}")
            if cfg.hnsw_ef_construction is not None:
                param_pairs.append(f"EFCONSTRUCTION {cfg.hnsw_ef_construction}")
        elif index_type == "IVF":
            organisation = "ORGANIZATION NEIGHBOR PARTITIONS"
            if cfg.ivf_neighbor_partitions is not None:
                # NB: Oracle 23ai/26ai uses ``NEIGHBOR PARTITIONS`` with a
                # space inside PARAMETERS(...) — NOT an underscore.
                param_pairs.append(f"NEIGHBOR PARTITIONS {cfg.ivf_neighbor_partitions}")
        else:  # pragma: no cover — model_post_init guards
            raise ValueError(f"Unsupported index_type: {cfg.index_type!r}")

        # Distance / PARAMETERS clause ordering quirk: when a PARAMETERS
        # block is present, the grammar wants bare ``DISTANCE <metric>``
        # without the ``WITH`` keyword; the ``WITH DISTANCE`` form is
        # only accepted when no PARAMETERS clause follows. ORA-00922 if
        # that rule is violated.
        clauses = [
            f"CREATE VECTOR INDEX {idx_name}",
            f"ON {self._full_table_name} ({cfg.embedding_column})",
            organisation,
        ]
        has_params = len(param_pairs) > 1
        distance_clause = (
            f"DISTANCE {cfg.distance_metric.upper()}"
            if has_params
            else f"WITH DISTANCE {cfg.distance_metric.upper()}"
        )
        clauses.append(distance_clause)
        if cfg.index_accuracy is not None:
            clauses.append(f"WITH TARGET ACCURACY {cfg.index_accuracy}")
        if has_params:
            clauses.append(f"PARAMETERS ({', '.join(param_pairs)})")
        if cfg.index_parallel is not None:
            clauses.append(f"PARALLEL {cfg.index_parallel}")

        return " ".join(clauses)

    async def build_index(self, *, rebuild: bool = False) -> None:
        """Create (or rebuild) the vector index on demand.

        Use this when you set ``auto_create_table=False`` and want to
        provision the index out-of-band, or when you've changed
        ``index_type`` / tuning knobs after data has already been loaded.

        Args:
            rebuild: When True, drop the existing index (if any) before
                creating it. Lets you switch from IVF to HNSW on a
                populated table without DROP TABLE.

        Raises:
            VectorStoreError: The index needs room in the database's
                vector memory pool and there is none (``ORA-51962``);
                the message says which ``index_type`` to pass instead.
        """
        cfg = self.oracle_config
        if cfg.index_type.upper() == "NONE":
            return
        ddl = self._vector_index_ddl()
        if ddl is None:  # pragma: no cover
            return

        pool = await self._get_pool()
        async with safe_acquire(self, pool) as conn, conn.cursor() as cursor:
            if rebuild:
                # Oracle has no "DROP IF EXISTS" for indexes; swallow the
                # ORA-01418 (index does not exist) so callers can use
                # rebuild=True idempotently.
                try:
                    await cursor.execute(f"DROP INDEX {self._vector_index_name}")
                except Exception as exc:  # pragma: no cover — diagnostic only
                    if "ORA-01418" not in str(exc):
                        raise
            try:
                await cursor.execute(ddl)
            except Exception as exc:
                if _needs_vector_memory(exc):
                    raise VectorStoreError(
                        _vector_memory_hint(cfg.index_type.upper(), self._full_table_name)
                    ) from exc
                raise
            await conn.commit()

    def _vector_to_string(self, embedding: list[float]) -> str:
        """Convert embedding list to Oracle VECTOR string format."""
        return "[" + ",".join(str(f) for f in embedding) + "]"

    @staticmethod
    async def _read_embedding(value: Any) -> list[float] | None:
        """Read a fetched vector column into a list of floats.

        python-oracledb returns a ``VECTOR`` column as an ``array.array``;
        the serialised ``[x,y,...]`` text, as ``str`` or as a LOB, is read
        too for tables reached through another SQL path.
        """
        if value is None:
            return None
        if hasattr(value, "read"):
            value = await value.read()
        if isinstance(value, bytes):
            value = value.decode()
        if isinstance(value, str):
            text = value.strip().strip("[]")
            return [float(x) for x in text.split(",")] if text else None
        return [float(x) for x in value]

    def _insert_sql(self) -> str:
        """Build an INSERT statement that respects column-name overrides."""
        cfg = self.oracle_config
        cols = [cfg.id_column, cfg.content_column, cfg.embedding_column, cfg.metadata_column]
        values = [":id", ":content", "TO_VECTOR(:embedding)", ":metadata"]
        if cfg.created_at_column:
            cols.append(cfg.created_at_column)
            values.append(":created_at")
        return (
            f"INSERT INTO {self._full_table_name} ({', '.join(cols)}) VALUES ({', '.join(values)})"
        )

    def _insert_params(self, doc_id: str, doc: Document) -> dict[str, Any]:
        params: dict[str, Any] = {
            "id": doc_id,
            "content": doc.content,
            "embedding": self._vector_to_string(doc.embedding or []),
            "metadata": json.dumps(doc.metadata),
        }
        if self.oracle_config.created_at_column:
            params["created_at"] = doc.created_at
        return params

    def _pin_clob_inputs(self, cursor: Any) -> None:
        """Tell oracledb to bind :content / :metadata as CLOB.

        Mirrors the explicit-LOB binding pattern langgraph-oracledb
        adopted for its checkpoint columns (oracle/langchain-oracle#224):
        without this hint oracledb thin mode can fail on large text
        payloads with ORA-01461 (character-to-LOB conversion).
        """
        import oracledb as _oracledb

        cursor.setinputsizes(
            content=_oracledb.DB_TYPE_CLOB,
            metadata=_oracledb.DB_TYPE_CLOB,
        )

    async def add(self, document: Document) -> str:
        """Add a document with embedding."""
        await self._ensure_table()
        pool = await self._get_pool()

        doc_id = document.id or uuid4().hex

        if document.embedding is None:
            raise ValueError("Document must have an embedding")

        async with safe_acquire(self, pool) as conn, conn.cursor() as cursor:
            self._pin_clob_inputs(cursor)
            await cursor.execute(self._insert_sql(), self._insert_params(doc_id, document))
            await conn.commit()

        return doc_id

    async def add_batch(self, documents: list[Document]) -> list[str]:
        """Add multiple documents."""
        await self._ensure_table()
        pool = await self._get_pool()

        ids = []
        sql = self._insert_sql()
        async with safe_acquire(self, pool) as conn, conn.cursor() as cursor:
            for doc in documents:
                doc_id = doc.id or uuid4().hex
                ids.append(doc_id)

                if doc.embedding is None:
                    raise ValueError(f"Document {doc_id} must have an embedding")

                self._pin_clob_inputs(cursor)
                await cursor.execute(sql, self._insert_params(doc_id, doc))
            await conn.commit()

        return ids

    def _compile_metadata_filter(
        self,
        filter_: dict[str, Any] | None,
        params: dict[str, Any],
        prefix: str = "mf",
    ) -> str:
        """Compile a Mongo-style filter — delegates to ``_oracle_filter``.

        Thin shim that supplies the ``metadata_column`` so the compiler
        stays decoupled from :class:`OracleVectorConfig`. The full
        operator grammar lives in
        ``src/locus/rag/stores/_oracle_filter.py``.
        """
        return _compile_metadata_filter(
            filter_,
            params,
            metadata_column=self.oracle_config.metadata_column,
            prefix=prefix,
        )

    def _select_columns_sql(self, *, with_distance: str | None = None) -> str:
        """Build the SELECT clause, optionally with a distance expression."""
        cfg = self.oracle_config
        parts = [
            f"{cfg.id_column} AS id_",
            f"{cfg.content_column} AS content_",
            # Fetched natively (python-oracledb returns a VECTOR as an
            # array.array). VECTOR_SERIALIZE to VARCHAR2 overflows 4000 bytes
            # at 1024 dimensions and above when MAX_STRING_SIZE is STANDARD.
            f"{cfg.embedding_column} AS embedding_",
            f"{cfg.metadata_column} AS metadata_",
        ]
        if cfg.created_at_column:
            parts.append(f"{cfg.created_at_column} AS created_at_")
        else:
            parts.append("NULL AS created_at_")
        if with_distance:
            parts.append(f"{with_distance} AS distance_")
        return ", ".join(parts)

    async def get(self, doc_id: str) -> Document | None:
        """Get a document by ID."""
        await self._ensure_table()
        pool = await self._get_pool()

        cfg = self.oracle_config
        async with safe_acquire(self, pool) as conn, conn.cursor() as cursor:
            await cursor.execute(
                f"""
                SELECT {self._select_columns_sql()}
                FROM {self._full_table_name}
                WHERE {cfg.id_column} = :id
                """,
                {"id": doc_id},
            )
            row = await cursor.fetchone()

        if row is None:
            return None

        embedding = await self._read_embedding(row[2])

        # Parse metadata (handle async LOB)
        metadata = row[3]
        if hasattr(metadata, "read"):
            metadata = await metadata.read()
        if isinstance(metadata, str):
            metadata = json.loads(metadata) if metadata else {}

        # Parse content (handle async LOB)
        content = row[1]
        if hasattr(content, "read"):
            content = await content.read()

        return Document(
            id=row[0],
            content=content,
            embedding=embedding,
            metadata=metadata,
            created_at=row[4] if row[4] else datetime.now(UTC),
        )

    async def delete(self, doc_id: str) -> bool:
        """Delete a document."""
        await self._ensure_table()
        pool = await self._get_pool()

        cfg = self.oracle_config
        async with safe_acquire(self, pool) as conn, conn.cursor() as cursor:
            await cursor.execute(
                f"DELETE FROM {self._full_table_name} WHERE {cfg.id_column} = :id",
                {"id": doc_id},
            )
            deleted: bool = cursor.rowcount > 0
            await conn.commit()

        return deleted

    async def search(
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
        """Search for similar documents using vector similarity.

        Uses Oracle's VECTOR_DISTANCE function for efficient similarity search.

        Args:
            query_embedding: Query vector.
            limit: Top-N to return.
            threshold: Optional minimum similarity score (post-MMR if enabled).
            metadata_filter: Mongo-style filter dict (see :meth:`hybrid_search`).
            mmr: When True, apply Maximal Marginal Relevance re-ranking
                to the candidate pool — fetches ``mmr_candidate_pool``
                rows from Oracle, then picks ``limit`` that balance
                relevance vs. diversity in Python.
            mmr_lambda: Trade-off in ``[0.0, 1.0]``. ``1.0`` = pure
                relevance (collapses to plain top-N), ``0.0`` = pure
                diversity, ``0.5`` is the standard balance.
            mmr_candidate_pool: Candidate pool size when MMR is on.
                Defaults to ``max(limit * 4, 20)`` so the diversity
                pass has enough material to choose from.
        """
        # Some LLMs (notably gpt-5.x via tool calls) JSON-encode floats as
        # strings (e.g. "0.5"); coerce defensively so the `score < threshold`
        # comparison below doesn't TypeError.
        if isinstance(threshold, str):
            try:
                threshold = float(threshold)
            except ValueError:
                threshold = None
        await self._ensure_table()
        pool = await self._get_pool()

        cfg = self.oracle_config
        # Build distance function based on metric
        metric = cfg.distance_metric.upper()
        distance_func = f"VECTOR_DISTANCE({cfg.embedding_column}, TO_VECTOR(:query_vec), {metric})"

        # Build WHERE clause for metadata filtering
        where_clauses = []
        # When MMR is on, oversample the candidate pool — Python-side
        # diversity rerank needs more material to pick from than the
        # final ``limit`` count.
        sql_limit = limit
        if mmr:
            if not 0.0 <= mmr_lambda <= 1.0:
                raise ValueError(f"mmr_lambda must be in [0.0, 1.0], got {mmr_lambda}")
            sql_limit = mmr_candidate_pool or max(limit * 4, 20)
        params: dict[str, Any] = {
            "query_vec": self._vector_to_string(query_embedding),
            "limit": sql_limit,
        }

        filter_sql = self._compile_metadata_filter(metadata_filter, params, prefix="mf")
        if filter_sql:
            where_clauses.append(filter_sql)

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        # For cosine distance, lower is better (0 = identical)
        # Convert to similarity score: 1 - distance for cosine
        async with safe_acquire(self, pool) as conn, conn.cursor() as cursor:
            await cursor.execute(
                f"""
                SELECT {self._select_columns_sql(with_distance=distance_func)}
                FROM {self._full_table_name}
                {where_sql}
                ORDER BY distance_ ASC
                FETCH FIRST :limit ROWS ONLY
                """,
                params,
            )
            rows = await cursor.fetchall()

        results = []
        for row in rows:
            distance = row[5]

            # Convert distance to similarity score (0-1, higher is better)
            if metric == "COSINE":
                # Cosine distance is 0-2, convert to similarity
                score = 1.0 - (distance / 2.0)
            elif metric == "DOT":
                # Dot product: higher is better, normalize
                score = max(0.0, min(1.0, (distance + 1.0) / 2.0))
            else:  # EUCLIDEAN
                # Euclidean: lower is better, use exponential decay
                score = 1.0 / (1.0 + distance)

            # Apply threshold filter
            if threshold is not None and score < threshold:
                continue

            embedding = await self._read_embedding(row[2])

            # Parse metadata (handle async LOB)
            metadata = row[3]
            if hasattr(metadata, "read"):
                metadata = await metadata.read()
            if isinstance(metadata, str):
                metadata = json.loads(metadata) if metadata else {}

            # Parse content (handle async LOB)
            content = row[1]
            if hasattr(content, "read"):
                content = await content.read()

            doc = Document(
                id=row[0],
                content=content,
                embedding=embedding,
                metadata=metadata,
                created_at=row[4] if row[4] else datetime.now(UTC),
            )

            results.append(
                SearchResult(
                    document=doc,
                    score=score,
                    distance=distance,
                )
            )

        if mmr:
            results = _mmr_rerank(
                results,
                query_embedding=query_embedding,
                limit=limit,
                lambda_=mmr_lambda,
            )
        return results

    async def ensure_text_index(self, *, drop_existing: bool = False) -> None:
        """Create an Oracle Text CONTEXT index on the content column.

        Required only when calling :meth:`hybrid_search` with
        ``use_text_index=True``. The index is named ``idx_<table>_txt``
        and provides relevance scoring via ``SCORE(label)`` on top of
        ``CONTAINS()`` queries. It is created with ``SYNC (ON COMMIT)``,
        so rows written after the index exists are searchable as soon as
        their transaction commits; a CONTEXT index without that parameter
        covers only the rows present when it was built. Costs disk space
        and adds index-maintenance overhead on writes — skip it if your
        corpus is small enough that the LIKE fallback is fast enough.

        Args:
            drop_existing: When True, drop the existing index first so
                the call is idempotent across reconfigurations. Use it
                once to rebuild an index created by an earlier release,
                which synchronised manually.
        """
        await self._ensure_table()
        pool = await self._get_pool()
        cfg = self.oracle_config
        idx_name = f"idx_{cfg.table_name}_txt"

        async with safe_acquire(self, pool) as conn, conn.cursor() as cursor:
            if drop_existing:
                try:
                    await cursor.execute(f"DROP INDEX {idx_name}")
                except Exception as exc:
                    if "ORA-01418" not in str(exc):
                        raise
            try:
                await cursor.execute(
                    f"CREATE INDEX {idx_name} ON {self._full_table_name} "
                    f"({cfg.content_column}) INDEXTYPE IS CTXSYS.CONTEXT "
                    "PARAMETERS ('SYNC (ON COMMIT)')"
                )
                await conn.commit()
            except Exception as exc:
                # ORA-29879: cannot create multiple CONTEXT indexes — already exists.
                if "ORA-29879" in str(exc) or "ORA-00955" in str(exc):
                    return
                raise

    async def hybrid_search(
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
        """Blend vector similarity with lexical relevance on the same row.

        Each row's final score is::

            alpha * vector_score + (1 - alpha) * lexical_score

        where both legs are normalised to [0, 1]. ``alpha=1.0``
        collapses to pure vector search, ``alpha=0.0`` to pure
        lexical. Ranges:

        * ``vector_score``: ``1 - distance/2`` for COSINE,
          ``(distance + 1) / 2`` for DOT, ``1 / (1 + distance)`` for
          EUCLIDEAN — same scoring as :meth:`search`.
        * ``lexical_score``: when ``use_text_index=True``, the Oracle Text
          ``SCORE(...)`` of an ``ACCUM`` query over the whitespace-split
          tokens of ``query_text`` (each escaped in ``{}``), divided by 100
          (Oracle returns 0..100). When False, the fraction of those tokens
          that appear as a case-insensitive substring of the content (0..1).

        Neither leg filters rows: a document that matches no query term
        scores 0 on the lexical leg and still ranks by its vector
        similarity, so ``alpha=1.0`` is the same ranking as :meth:`search`.
        The text index path requires :meth:`ensure_text_index` to have
        provisioned a ``CTXSYS.CONTEXT`` index on the content column.

        Args:
            query_text: Natural-language query string (used for the
                lexical leg).
            query_embedding: Query vector for the dense leg.
            limit: Top-N to return.
            alpha: Blend weight in ``[0.0, 1.0]``. 0.5 by default.
            threshold: Optional minimum *blended* score.
            metadata_filter: Same shape as :meth:`search`.
            use_text_index: Drive the lexical leg through Oracle Text
                instead of LIKE. Set True after calling
                :meth:`ensure_text_index`.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")
        await self._ensure_table()
        pool = await self._get_pool()

        cfg = self.oracle_config
        metric = cfg.distance_metric.upper()
        distance_func = f"VECTOR_DISTANCE({cfg.embedding_column}, TO_VECTOR(:query_vec), {metric})"

        params: dict[str, Any] = {
            "query_vec": self._vector_to_string(query_embedding),
            "limit": limit,
        }

        # ---- lexical leg -------------------------------------------------
        # Neither leg filters: a row with no lexical match scores 0 on that
        # leg and still ranks by vector similarity.
        lexical_join = ""
        text_query = _oracle_text_query(query_text or "") if use_text_index else None
        if use_text_index and text_query is not None:
            # SCORE(label) is the CTXSYS.CONTEXT relevance (0..100) of the
            # ACCUM query. SCORE must sit in the query block that holds the
            # CONTAINS, so the text leg is a LEFT JOIN on a subquery and
            # unmatched rows come back with NULL, read as 0.
            params["text_query"] = text_query
            lexical_expr = "NVL(lex_.lex_score_, 0) / 100.0"
            lexical_join = (
                f"LEFT JOIN (SELECT {cfg.id_column} AS lex_id_, SCORE(1) AS lex_score_ "
                f"FROM {self._full_table_name} "
                f"WHERE CONTAINS({cfg.content_column}, :text_query, 1) > 0) lex_ "
                f"ON lex_.lex_id_ = t.{cfg.id_column}"
            )
        elif use_text_index:
            # No term to look up → pure vector.
            lexical_expr = "0.0"
        else:
            tokens = [t for t in (query_text or "").split() if t]
            if not tokens:
                # No tokens → fall back to pure vector.
                lexical_expr = "0.0"
            else:
                # Cap to avoid runaway bind counts on absurd queries.
                tokens = tokens[:20]
                token_terms = []
                for i, tok in enumerate(tokens):
                    pname = f"tok_{i}"
                    params[pname] = f"%{tok.lower()}%"
                    token_terms.append(
                        f"CASE WHEN LOWER({cfg.content_column}) LIKE :{pname} THEN 1 ELSE 0 END"
                    )
                lexical_expr = f"({' + '.join(token_terms)}) / {len(tokens)}.0"

        # ---- vector→similarity expression (Oracle SQL) -------------------
        if metric == "COSINE":
            vector_score_expr = f"GREATEST(0, 1.0 - ({distance_func}) / 2.0)"
        elif metric == "DOT":
            vector_score_expr = f"GREATEST(0, ({distance_func} + 1.0) / 2.0)"
        else:  # EUCLIDEAN / MANHATTAN / HAMMING
            vector_score_expr = f"1.0 / (1.0 + ({distance_func}))"

        params["alpha"] = float(alpha)
        params["one_minus_alpha"] = 1.0 - float(alpha)
        blended_expr = f"({vector_score_expr}) * :alpha + ({lexical_expr}) * :one_minus_alpha"

        filter_sql = self._compile_metadata_filter(metadata_filter, params, prefix="hmf")
        where_sql = f"WHERE {filter_sql}" if filter_sql else ""

        sql = f"""
            SELECT {self._select_columns_sql(with_distance=distance_func)},
                   {lexical_expr} AS lexical_score_,
                   ({blended_expr}) AS blended_score_
            FROM {self._full_table_name} t
            {lexical_join}
            {where_sql}
            ORDER BY blended_score_ DESC
            FETCH FIRST :limit ROWS ONLY
        """

        async with safe_acquire(self, pool) as conn, conn.cursor() as cursor:
            await cursor.execute(sql, params)
            rows = await cursor.fetchall()

        results: list[SearchResult] = []
        for row in rows:
            distance = row[5]
            lexical_score = float(row[6] or 0.0)
            blended = float(row[7] or 0.0)

            if threshold is not None and blended < threshold:
                continue

            embedding = await self._read_embedding(row[2])

            metadata = row[3]
            if hasattr(metadata, "read"):
                metadata = await metadata.read()
            if isinstance(metadata, str):
                metadata = json.loads(metadata) if metadata else {}

            content = row[1]
            if hasattr(content, "read"):
                content = await content.read()

            doc = Document(
                id=row[0],
                content=content,
                embedding=embedding,
                metadata=metadata,
                created_at=row[4] if row[4] else datetime.now(UTC),
            )
            sr = SearchResult(
                document=doc,
                score=blended,
                distance=distance,
            )
            # Expose the lexical leg for callers that want to compare or
            # debug the blend without rerunning the query.
            object.__setattr__(sr, "_lexical_score", lexical_score)
            results.append(sr)

        return results

    async def count(self) -> int:
        """Count documents in store."""
        await self._ensure_table()
        pool = await self._get_pool()

        async with safe_acquire(self, pool) as conn, conn.cursor() as cursor:
            await cursor.execute(f"SELECT COUNT(*) FROM {self._full_table_name}")
            row = await cursor.fetchone()

        return row[0] if row else 0

    async def clear(self) -> int:
        """Delete all documents."""
        await self._ensure_table()
        pool = await self._get_pool()

        async with safe_acquire(self, pool) as conn, conn.cursor() as cursor:
            await cursor.execute(f"SELECT COUNT(*) FROM {self._full_table_name}")
            row = await cursor.fetchone()
            count = row[0] if row else 0

            await cursor.execute(f"TRUNCATE TABLE {self._full_table_name}")
            await conn.commit()

        return count

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    def __repr__(self) -> str:
        if self.oracle_config.dsn:
            return f"OracleVectorStore(dsn={self.oracle_config.dsn!r})"
        if self.oracle_config.host:
            return f"OracleVectorStore(host={self.oracle_config.host!r})"
        return "OracleVectorStore()"
