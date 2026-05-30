# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Oracle Database 26ai store backend — namespace-keyed key/value with vectors.

Native locus implementation of the long-term-memory store contract
(:class:`locus.memory.store.BaseStore`) on top of Oracle Database 26ai's
JSON and ``VECTOR`` data types. Functionally equivalent to
``langgraph-oracledb``'s ``OracleStore`` / ``AsyncOracleStore`` but with
**zero** langchain / langgraph imports — locus owns the contract.

Schema (single table, auto-created with ``auto_create_table=True``)::

    CREATE TABLE locus_store (
        namespace  VARCHAR2(255) NOT NULL,
        key        VARCHAR2(255) NOT NULL,
        value      CLOB CHECK (value IS JSON),
        embedding  VECTOR(<dim>, FLOAT32),     -- nullable, omitted when dim=None
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP,
        CONSTRAINT pk_locus_store PRIMARY KEY (namespace, key)
    );
    CREATE INDEX idx_locus_store_ns ON locus_store (namespace);

Namespaces (``tuple[str, ...]``) are flattened to a ``/``-joined string
for storage::

    ("memory", "user-42") -> "memory/user-42"

``list_namespaces`` reverses the join. ``prefix`` matching uses
``LIKE :prefix || '%'`` for fast index scans.

Uses ``python-oracledb`` in thin mode (no Oracle Client required). The
``oracledb`` dependency is imported lazily inside ``_get_pool`` so the
module imports fine on installs without it — the same pattern
``OracleBackend`` uses.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, SecretStr

from locus._oracle_pool_cache import safe_acquire
from locus.memory.store import (
    BaseStore,
    SemanticSearchResult,
    StoreCapabilities,
    StoreItem,
)


if TYPE_CHECKING:
    import oracledb


# ---------------------------------------------------------------------------
# Identifier validation
# ---------------------------------------------------------------------------

_SAFE_SQL_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_$#]{0,127}$")
_ALLOWED_DISTANCE_METRICS = frozenset({"COSINE", "DOT", "EUCLIDEAN", "MANHATTAN", "HAMMING"})


def _validate_sql_identifier(value: str, field_name: str) -> str:
    """Validate that a string is a safe Oracle SQL identifier.

    Mirrors the helper in ``locus.memory.backends.oracle`` and
    ``locus.rag.stores.oracle`` — duplicated locally so this module
    doesn't need a private import from a sibling package.
    """
    if not _SAFE_SQL_IDENTIFIER.match(value):
        msg = (
            f"Invalid {field_name}: {value!r}. "
            "Must start with a letter or underscore and contain only "
            "alphanumeric characters, underscores, $, or # (max 128 chars)."
        )
        raise ValueError(msg)
    return value


# ---------------------------------------------------------------------------
# Namespace flattening helpers
# ---------------------------------------------------------------------------

_NS_SEP = "/"


def _flatten_namespace(namespace: tuple[str, ...]) -> str:
    """Flatten a namespace tuple to a ``/``-joined string for storage.

    Empty tuple maps to the empty string. We deliberately reject parts
    that contain the separator itself so the round-trip with
    :func:`_parse_namespace` is lossless.
    """
    for part in namespace:
        if not isinstance(part, str):
            raise TypeError(f"namespace parts must be str, got {type(part).__name__}: {part!r}")
        if _NS_SEP in part:
            raise ValueError(
                f"namespace part {part!r} contains the separator {_NS_SEP!r} — "
                "use a different naming convention."
            )
    return _NS_SEP.join(namespace)


def _parse_namespace(flat: str) -> tuple[str, ...]:
    """Inverse of :func:`_flatten_namespace`."""
    if not flat:
        return ()
    return tuple(flat.split(_NS_SEP))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class OracleStoreConfig(BaseModel):
    """Configuration for :class:`OracleStore`.

    Mirrors :class:`locus.memory.backends.oracle.OracleConfig` for the
    connection envelope so the same set of arguments work for both
    checkpointer and store side-by-side.
    """

    # Connection options
    dsn: str | None = None
    user: str = "admin"
    password: SecretStr = SecretStr("")

    # For Autonomous Database with wallet
    wallet_location: str | None = None
    wallet_password: SecretStr | None = None

    # Connection string components (alternative to DSN)
    host: str | None = None
    port: int = 1521
    service_name: str | None = None

    # Table settings
    table_name: str = "locus_store"
    schema_name: str | None = None

    # Pool settings
    min_pool_size: int = 1
    max_pool_size: int = 5

    # Vector / semantic-search settings
    # When ``dimension`` is None the table is created WITHOUT the VECTOR
    # column (text-only store mode); ``put_with_embedding`` /
    # ``search_by_embedding`` raise StoreCapabilityError in that case.
    dimension: int | None = None
    distance_metric: str = "COSINE"

    # When False, _ensure_table won't issue CREATE statements — useful
    # when the table is pre-provisioned by a DBA out-of-band so the app
    # user can run with INSERT/SELECT/UPDATE only (no DDL).
    auto_create_table: bool = True

    def model_post_init(self, __context: Any) -> None:
        _validate_sql_identifier(self.table_name, "table_name")
        if self.schema_name is not None:
            _validate_sql_identifier(self.schema_name, "schema_name")
        if self.distance_metric.upper() not in _ALLOWED_DISTANCE_METRICS:
            raise ValueError(
                f"Invalid distance_metric: {self.distance_metric!r}. "
                f"Must be one of: {sorted(_ALLOWED_DISTANCE_METRICS)}"
            )
        if self.dimension is not None and self.dimension < 1:
            raise ValueError(f"dimension must be a positive int, got {self.dimension}")


# ---------------------------------------------------------------------------
# OracleStore
# ---------------------------------------------------------------------------


class OracleStore(BaseStore):
    """Oracle Database 26ai-backed long-term memory store.

    Implements the :class:`locus.memory.store.BaseStore` contract on top
    of Oracle 26ai. Persists namespaced key/value rows and, when
    ``dimension`` is set, vector embeddings for semantic search inside a
    namespace.

    **Production setup — use a least-privileged schema, not ADMIN.**

    Running against ``ADMIN`` is an Oracle anti-pattern: every connection
    has full DBA privileges, so a compromised credential has unbounded
    blast radius. Provision a dedicated app user with only the
    privileges this store needs::

        CREATE USER locus_app IDENTIFIED BY "<strong-password>";
        GRANT CONNECT, RESOURCE TO locus_app;
        ALTER USER locus_app QUOTA 1G ON DATA;

    Then either set ``auto_create_table=True`` and let the store
    provision its own table, or pre-create it as DBA and pass
    ``auto_create_table=False`` so the app user runs with DML-only.

    Example — Autonomous DB with wallet, vector mode::

        store = OracleStore(
            dsn="mydb_high",
            user="locus_app",
            password=os.environ["LOCUS_DB_PASSWORD"],
            wallet_location="~/.oci/wallets/mydb",
            dimension=1024,  # Cohere embed-v3
        )
        await store.put(("memory", "u1"), "theme", {"value": "dark"})
        await store.put_with_embedding(
            ("memory", "u1"),
            "fact1",
            {"text": "user likes dark theme"},
            embedding=embedder("user likes dark theme"),
        )
        hits = await store.search_by_embedding(
            ("memory", "u1"),
            query_embedding=embedder("display preferences"),
            limit=5,
        )

    Example — text-only mode (no vector column)::

        store = OracleStore(
            dsn="mydb_high",
            user="u",
            password="p",
            dimension=None,  # store still works; semantic_search disabled
        )
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
        super().__init__()
        self.config = OracleStoreConfig(
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
            table_name=table_name,
            schema_name=schema_name,
            min_pool_size=min_pool_size,
            max_pool_size=max_pool_size,
            dimension=dimension,
            distance_metric=distance_metric,
            auto_create_table=auto_create_table,
        )
        self._pool: oracledb.AsyncConnectionPool | None = None
        self._pool_loop: Any = None  # asyncio loop the pool is bound to
        self._initialized: bool = False

    # -- Capabilities --------------------------------------------------------

    @property
    def capabilities(self) -> StoreCapabilities:
        """Report what this backend supports.

        ``semantic_search`` is gated on ``dimension`` being set — the
        VECTOR column is omitted from the DDL in text-only mode, so the
        vector queries would fail at the SQL level anyway.
        """
        has_vector = self.config.dimension is not None
        return StoreCapabilities(
            search=True,  # LIKE substring on JSON value
            semantic_search=has_vector,
            embedding_dimension=self.config.dimension,
            list_namespaces=True,
            batch_operations=False,
        )

    # -- Pool / DDL ---------------------------------------------------------

    async def _get_pool(self) -> oracledb.AsyncConnectionPool:
        """Lazily create the oracledb async pool.

        ``oracledb`` is imported here (not at module load) so installs
        without the driver can still ``import`` this module.
        """
        if self._pool is None:
            try:
                import oracledb
            except ImportError as e:
                raise ImportError(
                    "OracleStore requires the 'oracledb' package. "
                    "Install with: pip install oracledb"
                ) from e

            cfg = self.config
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

            # create_pool_async returns the pool directly — the "async"
            # refers to the pool type, not the call signature.
            self._pool = oracledb.create_pool_async(
                user=cfg.user,
                password=cfg.password.get_secret_value(),
                dsn=dsn,
                min=cfg.min_pool_size,
                max=cfg.max_pool_size,
                **params,
            )
        return self._pool

    @property
    def _full_table_name(self) -> str:
        if self.config.schema_name:
            return f"{self.config.schema_name}.{self.config.table_name}"
        return self.config.table_name

    def _create_table_ddl(self) -> str:
        """Build the CREATE TABLE statement.

        When ``dimension`` is ``None`` the VECTOR column is omitted
        entirely so the table is valid on Oracle versions without
        native vector support (and so the column doesn't sit empty
        wasting space in text-only mode).
        """
        cfg = self.config
        if cfg.dimension is not None:
            vector_col = f", embedding VECTOR({cfg.dimension}, FLOAT32)"
        else:
            vector_col = ""
        return (
            f"CREATE TABLE {self._full_table_name} (\n"
            f"    namespace  VARCHAR2(255) NOT NULL,\n"
            f"    key        VARCHAR2(255) NOT NULL,\n"
            f"    value      CLOB CHECK (value IS JSON),\n"
            f"    updated_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP"
            f"{vector_col},\n"
            f"    CONSTRAINT pk_{cfg.table_name} PRIMARY KEY (namespace, key)\n"
            f")"
        )

    def _ns_index_ddl(self) -> str:
        cfg = self.config
        return f"CREATE INDEX idx_{cfg.table_name}_ns ON {self._full_table_name} (namespace)"

    async def _ensure_table(self) -> None:
        """Create the table once per instance, idempotently."""
        if self._initialized:
            return

        if not self.config.auto_create_table:
            # DBA-managed mode: trust the table exists; let the first
            # SQL surface ORA-00942 if not.
            self._initialized = True
            return

        pool = await self._get_pool()
        async with safe_acquire(self, pool) as conn, conn.cursor() as cursor:
            await cursor.execute(
                """
                SELECT COUNT(*) FROM user_tables
                WHERE table_name = UPPER(:table_name)
                """,
                {"table_name": self.config.table_name},
            )
            row = await cursor.fetchone()
            table_exists = bool(row and row[0] > 0)

            if not table_exists:
                await cursor.execute(self._create_table_ddl())
                await cursor.execute(self._ns_index_ddl())
                await conn.commit()

        self._initialized = True

    # -- MERGE / SELECT / DELETE SQL ----------------------------------------

    def _merge_sql(self, *, with_embedding: bool) -> str:
        """Build the upsert MERGE statement.

        Two variants: one that touches the VECTOR column, one that
        doesn't. We keep them as separate strings so the SQL parser
        can prepare each variant separately.
        """
        if with_embedding:
            update_set = (
                "value = :value, embedding = TO_VECTOR(:embedding), updated_at = SYSTIMESTAMP"
            )
            insert_cols = "(namespace, key, value, embedding)"
            insert_vals = "(:namespace, :key, :value, TO_VECTOR(:embedding))"
        else:
            update_set = "value = :value, updated_at = SYSTIMESTAMP"
            insert_cols = "(namespace, key, value)"
            insert_vals = "(:namespace, :key, :value)"

        return (
            f"MERGE INTO {self._full_table_name} t "
            f"USING (SELECT :namespace AS namespace, :key AS key FROM dual) s "
            f"ON (t.namespace = s.namespace AND t.key = s.key) "
            f"WHEN MATCHED THEN UPDATE SET {update_set} "
            f"WHEN NOT MATCHED THEN INSERT {insert_cols} VALUES {insert_vals}"
        )

    @staticmethod
    def _vector_to_string(embedding: list[float]) -> str:
        """Serialize a Python embedding to Oracle's TO_VECTOR text input."""
        return "[" + ",".join(repr(float(f)) for f in embedding) + "]"

    @staticmethod
    def _pin_clob(cursor: Any) -> None:
        """Hint oracledb to bind :value as CLOB.

        Without this, thin-mode oracledb can fail with ORA-01461
        (character-to-LOB conversion) on large JSON payloads. Same
        hardening :class:`OracleBackend` applies to its ``:data`` bind.
        """
        import oracledb as _oracledb

        cursor.setinputsizes(value=_oracledb.DB_TYPE_CLOB)

    # -- BaseStore: put / get / delete / list_keys --------------------------

    async def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Upsert a value.

        ``metadata`` is accepted for BaseStore parity but folded into
        the JSON value when provided so we don't need a second column.
        Callers needing first-class metadata should query
        ``value['_meta']`` from JSON_VALUE.
        """
        if metadata:
            # Don't mutate the caller's dict.
            payload = {"value": value, "_meta": metadata}
        else:
            payload = value

        await self._ensure_table()
        pool = await self._get_pool()
        ns_flat = _flatten_namespace(namespace)

        async with safe_acquire(self, pool) as conn, conn.cursor() as cursor:
            self._pin_clob(cursor)
            await cursor.execute(
                self._merge_sql(with_embedding=False),
                {
                    "namespace": ns_flat,
                    "key": key,
                    "value": json.dumps(payload, default=str),
                },
            )
            await conn.commit()

    async def get(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> Any | None:
        await self._ensure_table()
        pool = await self._get_pool()
        ns_flat = _flatten_namespace(namespace)

        async with safe_acquire(self, pool) as conn, conn.cursor() as cursor:
            await cursor.execute(
                f"SELECT value FROM {self._full_table_name} "
                f"WHERE namespace = :namespace AND key = :key",
                {"namespace": ns_flat, "key": key},
            )
            row = await cursor.fetchone()

        if row is None:
            return None
        return await _decode_value(row[0])

    async def delete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> bool:
        await self._ensure_table()
        pool = await self._get_pool()
        ns_flat = _flatten_namespace(namespace)

        async with safe_acquire(self, pool) as conn, conn.cursor() as cursor:
            await cursor.execute(
                f"DELETE FROM {self._full_table_name} WHERE namespace = :namespace AND key = :key",
                {"namespace": ns_flat, "key": key},
            )
            deleted = cursor.rowcount > 0
            await conn.commit()

        return bool(deleted)

    async def list_keys(
        self,
        namespace: tuple[str, ...],
        limit: int = 100,
    ) -> list[str]:
        await self._ensure_table()
        pool = await self._get_pool()
        ns_flat = _flatten_namespace(namespace)

        async with safe_acquire(self, pool) as conn, conn.cursor() as cursor:
            await cursor.execute(
                f"SELECT key FROM {self._full_table_name} "
                f"WHERE namespace = :namespace "
                f"ORDER BY updated_at DESC "
                f"FETCH FIRST :lim ROWS ONLY",
                {"namespace": ns_flat, "lim": limit},
            )
            rows = await cursor.fetchall()

        return [row[0] for row in rows]

    # -- BaseStore: search / list_namespaces --------------------------------

    async def search(
        self,
        namespace: tuple[str, ...],
        query: str | None = None,
        limit: int = 10,
    ) -> list[StoreItem]:
        """Text search inside a namespace.

        Uses ``LOWER(value) LIKE LOWER(:q)`` — sufficient for the
        small payloads memory stores typically carry. For
        Oracle-Text-grade ranking, swap to ``CONTAINS()`` with an
        Oracle Text index out-of-band.
        """
        await self._ensure_table()
        pool = await self._get_pool()
        ns_flat = _flatten_namespace(namespace)

        params: dict[str, Any] = {"namespace": ns_flat, "lim": limit}
        where = "namespace = :namespace"
        if query:
            where += " AND LOWER(value) LIKE LOWER(:pattern)"
            params["pattern"] = f"%{query}%"

        async with safe_acquire(self, pool) as conn, conn.cursor() as cursor:
            await cursor.execute(
                f"SELECT namespace, key, value, updated_at "
                f"FROM {self._full_table_name} "
                f"WHERE {where} "
                f"ORDER BY updated_at DESC "
                f"FETCH FIRST :lim ROWS ONLY",
                params,
            )
            rows = await cursor.fetchall()

        items: list[StoreItem] = []
        for row in rows:
            ns_str, key_str, value_lob, updated_at = row
            decoded = await _decode_value(value_lob)
            # If the put() helper wrapped metadata into the payload,
            # split it back out so the StoreItem matches what callers expect.
            if isinstance(decoded, dict) and set(decoded.keys()) == {"value", "_meta"}:
                value = decoded["value"]
                meta = decoded["_meta"]
            else:
                value = decoded
                meta = {}
            items.append(
                StoreItem(
                    namespace=_parse_namespace(ns_str),
                    key=key_str,
                    value=value,
                    metadata=meta,
                    created_at=updated_at or datetime.now(UTC),
                    updated_at=updated_at or datetime.now(UTC),
                )
            )
        return items

    async def list_namespaces(
        self,
        prefix: tuple[str, ...] | None = None,
        limit: int = 100,
    ) -> list[tuple[str, ...]]:
        await self._ensure_table()
        pool = await self._get_pool()

        params: dict[str, Any] = {"lim": limit}
        where = ""
        if prefix:
            prefix_flat = _flatten_namespace(prefix)
            # Match either the prefix exactly or anything beginning with
            # the prefix followed by the separator. Plain ``LIKE prefix||'%'``
            # would also match ``("memorial", ...)`` when prefix is
            # ``("memo",)`` — which is rarely what callers want.
            where = "WHERE namespace = :pfx OR namespace LIKE :pfx_sep "
            params["pfx"] = prefix_flat
            params["pfx_sep"] = prefix_flat + _NS_SEP + "%"

        async with safe_acquire(self, pool) as conn, conn.cursor() as cursor:
            await cursor.execute(
                f"SELECT DISTINCT namespace FROM {self._full_table_name} "
                f"{where}"
                f"FETCH FIRST :lim ROWS ONLY",
                params,
            )
            rows = await cursor.fetchall()

        return [_parse_namespace(row[0]) for row in rows]

    # -- BaseStore: semantic-search hooks -----------------------------------

    async def put_with_embedding(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: Any,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.config.dimension is None:
            # Capability gate — surface the typed error rather than
            # ORA-00904 on the missing column.
            return await super().put_with_embedding(namespace, key, value, embedding, metadata)
        if len(embedding) != self.config.dimension:
            raise ValueError(
                f"embedding has {len(embedding)} dims, store configured for {self.config.dimension}"
            )

        if metadata:
            payload = {"value": value, "_meta": metadata}
        else:
            payload = value

        await self._ensure_table()
        pool = await self._get_pool()
        ns_flat = _flatten_namespace(namespace)

        async with safe_acquire(self, pool) as conn, conn.cursor() as cursor:
            self._pin_clob(cursor)
            await cursor.execute(
                self._merge_sql(with_embedding=True),
                {
                    "namespace": ns_flat,
                    "key": key,
                    "value": json.dumps(payload, default=str),
                    "embedding": self._vector_to_string(embedding),
                },
            )
            await conn.commit()
        return None

    async def search_by_embedding(
        self,
        namespace: tuple[str, ...],
        query_embedding: list[float],
        limit: int = 10,
        threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SemanticSearchResult]:
        if self.config.dimension is None:
            return await super().search_by_embedding(
                namespace, query_embedding, limit, threshold, metadata_filter
            )

        await self._ensure_table()
        pool = await self._get_pool()
        ns_flat = _flatten_namespace(namespace)

        metric = self.config.distance_metric.upper()
        distance_expr = f"VECTOR_DISTANCE(embedding, TO_VECTOR(:q), {metric})"

        async with safe_acquire(self, pool) as conn, conn.cursor() as cursor:
            await cursor.execute(
                f"SELECT namespace, key, value, updated_at, "
                f"       {distance_expr} AS distance_ "
                f"FROM {self._full_table_name} "
                f"WHERE namespace = :namespace AND embedding IS NOT NULL "
                f"ORDER BY distance_ ASC "
                f"FETCH FIRST :lim ROWS ONLY",
                {
                    "namespace": ns_flat,
                    "q": self._vector_to_string(query_embedding),
                    "lim": limit,
                },
            )
            rows = await cursor.fetchall()

        results: list[SemanticSearchResult] = []
        for row in rows:
            ns_str, key_str, value_lob, updated_at, distance = row
            score = _distance_to_score(metric, float(distance))
            if threshold is not None and score < threshold:
                continue
            decoded = await _decode_value(value_lob)
            if isinstance(decoded, dict) and set(decoded.keys()) == {"value", "_meta"}:
                value = decoded["value"]
                meta = decoded["_meta"]
            else:
                value = decoded
                meta = {}
            item = StoreItem(
                namespace=_parse_namespace(ns_str),
                key=key_str,
                value=value,
                metadata=meta,
                created_at=updated_at or datetime.now(UTC),
                updated_at=updated_at or datetime.now(UTC),
            )
            results.append(SemanticSearchResult(item=item, score=score, distance=float(distance)))
        return results

    # -- Lifecycle ----------------------------------------------------------

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
        self._initialized = False

    # -- Async-prefix aliases (langgraph-oracledb parity) -------------------
    # langgraph-oracledb names its store methods aput / aget / etc.
    # We expose those as aliases so callers porting from that library
    # don't need to rename every call site — the canonical methods are
    # the BaseStore ones above.

    aput = put
    aget = get
    adelete = delete
    asearch = search
    alist_namespaces = list_namespaces

    def __repr__(self) -> str:
        cfg = self.config
        if cfg.dsn:
            return f"OracleStore(dsn={cfg.dsn!r}, table={cfg.table_name!r})"
        if cfg.host:
            return (
                f"OracleStore(host={cfg.host!r}, service={cfg.service_name!r}, "
                f"table={cfg.table_name!r})"
            )
        return f"OracleStore(table={cfg.table_name!r})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _decode_value(raw: Any) -> Any:
    """Decode the ``value`` column.

    oracledb may hand back the JSON column as a dict, a str, a sync
    CLOB locator (``raw.read()`` returns a string), or an async CLOB
    locator (``raw.read()`` is a coroutine). Normalise all four to a
    Python value.
    """
    if raw is None:
        return None
    if hasattr(raw, "read"):
        read_result = raw.read()
        raw = await read_result if asyncio.iscoroutine(read_result) else read_result
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def _distance_to_score(metric: str, distance: float) -> float:
    """Convert raw VECTOR_DISTANCE output to a 0..1 similarity score.

    Same formulas the RAG OracleVectorStore uses — duplicated locally
    so this module doesn't reach into the RAG package.
    """
    if metric == "COSINE":
        return 1.0 - (distance / 2.0)
    if metric == "DOT":
        return max(0.0, min(1.0, (distance + 1.0) / 2.0))
    # EUCLIDEAN / MANHATTAN / HAMMING: lower is better.
    return 1.0 / (1.0 + distance)
