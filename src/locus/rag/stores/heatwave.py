# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""MySQL HeatWave vector store.

Documents live in one InnoDB table with a native ``VECTOR(n)`` column for
the embedding, a ``JSON`` column for the metadata and a ``FULLTEXT`` index
on the content. The store speaks plain MySQL through the official
Connector/Python driver (the ``heatwave`` extra, the same package as the
``mysql`` checkpoint extra) and adapts to what the server can do:

* **HeatWave MySQL** (MySQL HeatWave on OCI, MySQL AI) has ``DISTANCE()``,
  so similarity is computed and ordered on the server, and a HeatWave
  cluster accelerates it once the table is loaded (:meth:`load_into_heatwave`).
* **Community / Commercial MySQL 9+** has the ``VECTOR`` type and the
  ``STRING_TO_VECTOR`` / ``VECTOR_TO_STRING`` functions but no
  ``DISTANCE()`` and no vector index. The store then fetches the candidate
  rows (bounded by ``candidate_limit``) and computes the distances here.

The store probes ``DISTANCE()`` once on first use and reports the outcome
as :attr:`server_side_distance`. Metadata filters are equality predicates
over ``JSON_EXTRACT`` with every path and value bound as a parameter;
hybrid search blends the vector leg with ``MATCH ... AGAINST`` relevance
client-side (:mod:`locus.rag.stores._hybrid`), like the Elasticsearch store.

Usage::

    from locus.rag.stores import HeatWaveVectorStore

    store = HeatWaveVectorStore(
        host="10.0.1.5", database="kb", user="locus", password="...", dimension=1024
    )
    store = HeatWaveVectorStore(
        dsn="mysql://locus:...@10.0.1.5:3306/kb", dimension=1024
    )
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import math
import re
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypeVar
from urllib.parse import parse_qsl, unquote, urlparse

from pydantic import BaseModel, Field, SecretStr

from locus.rag.stores._hybrid import blend_hybrid
from locus.rag.stores.base import (
    BaseVectorStore,
    Document,
    SearchResult,
    VectorStoreConfig,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


logger = logging.getLogger(__name__)

T = TypeVar("T")

_MISSING_DRIVER = (
    'mysql-connector-python is not installed. Install with: pip install "locus-sdk[heatwave]"'
)

_SAFE_SQL_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,63}$")
_ALLOWED_DISTANCE_METRICS = frozenset({"COSINE", "DOT", "EUCLIDEAN"})
# MySQL caps a VECTOR column at 16383 entries.
_MAX_VECTOR_DIMENSION = 16383
# ER_SP_DOES_NOT_EXIST: "FUNCTION <db>.DISTANCE does not exist" on servers
# without the HeatWave vector functions.
_ER_SP_DOES_NOT_EXIST = 1305
_DISTANCE_PROBE_SQL = (
    "SELECT DISTANCE(STRING_TO_VECTOR('[1,0]'), STRING_TO_VECTOR('[0,1]'), 'COSINE')"
)


def _validate_sql_identifier(value: str, field_name: str) -> str:
    """Validate that a string is a safe MySQL identifier."""
    if not _SAFE_SQL_IDENTIFIER.match(value):
        msg = (
            f"Invalid {field_name}: {value!r}. "
            "Must start with a letter or underscore and contain only "
            "alphanumeric characters and underscores (max 64 chars)."
        )
        raise ValueError(msg)
    return value


def _quote_identifier(value: str) -> str:
    """Quote a validated MySQL identifier."""
    return f"`{value}`"


def _parse_dsn(dsn: str) -> dict[str, Any]:
    """Parse a ``mysql://`` DSN into Connector/Python keyword arguments.

    Same grammar as the MySQL checkpoint backend: ``mysql://`` or
    ``mysql+connector://``, user and password URL-decoded, the first path
    segment the database, query parameters passed through to the driver.
    """
    parsed = urlparse(dsn)
    if parsed.scheme not in {"mysql", "mysql+connector"}:
        msg = "MySQL DSN must use the mysql:// or mysql+connector:// scheme"
        raise ValueError(msg)

    config: dict[str, Any] = {}
    if parsed.hostname:
        config["host"] = parsed.hostname
    if parsed.port:
        config["port"] = parsed.port
    if parsed.username:
        config["user"] = unquote(parsed.username)
    if parsed.password:
        config["password"] = unquote(parsed.password)
    if parsed.path and parsed.path != "/":
        config["database"] = unquote(parsed.path.lstrip("/"))
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        config[key] = value
    return config


def _decode_json(value: Any) -> Any:
    """Decode a JSON column across the driver's text / bytes / native paths."""
    if value is None:
        return None
    if isinstance(value, bytes | bytearray):
        value = bytes(value).decode("utf-8")
    if isinstance(value, str):
        return json.loads(value) if value else None
    return value


def _vector_to_string(embedding: list[float]) -> str:
    """Serialise an embedding for ``STRING_TO_VECTOR``."""
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


def _string_to_vector(value: Any) -> list[float] | None:
    """Parse the ``VECTOR_TO_STRING`` output (``[1.00000e+00,...]``)."""
    if value is None:
        return None
    if isinstance(value, bytes | bytearray):
        value = bytes(value).decode("ascii")
    text = str(value).strip().strip("[]").strip()
    if not text:
        return []
    return [float(x) for x in text.split(",")]


def _is_missing_function(exc: BaseException) -> bool:
    """True for the driver error a server raises when ``DISTANCE()`` is unknown."""
    if getattr(exc, "errno", None) == _ER_SP_DOES_NOT_EXIST:
        return True
    text = str(exc)
    return "does not exist" in text and "DISTANCE" in text.upper()


def build_filter_sql(
    metadata_filter: dict[str, Any] | None, metadata_column: str = "metadata"
) -> tuple[str, list[Any]]:
    """Compile equality filters into ``JSON_EXTRACT`` predicates with bound parameters.

    Each ``{key: value}`` pair becomes
    ``JSON_EXTRACT(metadata, %s) = CAST(%s AS JSON)`` with the JSON path and
    the JSON-encoded value both passed as parameters, so neither the key nor
    the value is ever spliced into the statement. Comparing JSON to JSON
    keeps the match typed: ``2026`` does not match ``"2026"`` and ``True``
    does not match ``1``. Values must be scalars.

    Returns the ``AND``-joined predicate (empty when there is nothing to
    filter) and the parameter list in placeholder order.
    """
    if not metadata_filter:
        return "", []
    column = _quote_identifier(_validate_sql_identifier(metadata_column, "metadata_column"))
    clauses: list[str] = []
    params: list[Any] = []
    for key, value in metadata_filter.items():
        if not isinstance(key, str) or not key:
            raise TypeError("metadata_filter keys must be non-empty strings")
        if value is not None and not isinstance(value, str | int | float | bool):
            raise TypeError(
                "metadata_filter values must be scalars (str, int, float, bool or None); "
                f"got {type(value).__name__} for {key!r}"
            )
        escaped = key.replace("\\", "\\\\").replace('"', '\\"')
        clauses.append(f"JSON_EXTRACT({column}, %s) = CAST(%s AS JSON)")
        params.extend([f'$."{escaped}"', json.dumps(value)])
    return " AND ".join(clauses), params


class HeatWaveVectorConfig(BaseModel):
    """Configuration for :class:`HeatWaveVectorStore`.

    Unknown keys are rejected so a misspelled option fails loudly instead of
    being absorbed.
    """

    model_config = {"extra": "forbid"}

    # Connection
    host: str = "localhost"
    port: int = 3306
    database: str = "locus"
    user: str = "root"
    password: SecretStr = SecretStr("")
    dsn: str | None = None
    ssl_ca: str | None = None
    #: Extra keyword arguments handed to ``mysql.connector.connect`` verbatim
    #: (``ssl_disabled``, ``connection_timeout``, ``use_pure`` and so on).
    connect_args: dict[str, Any] = Field(default_factory=dict)
    pool_size: int = 5
    charset: str = "utf8mb4"
    collation: str = "utf8mb4_unicode_ci"

    # Table
    table_name: str = "locus_vectors"
    dimension: int = 1024
    distance_metric: str = "COSINE"  # COSINE, DOT, EUCLIDEAN
    #: Issue ``CREATE TABLE IF NOT EXISTS`` on first use. Set ``False`` when
    #: the application user has DML privileges only and the table exists.
    auto_create_table: bool = True
    #: The table carries a ``FULLTEXT`` index on the content column, which
    #: :meth:`HeatWaveVectorStore.hybrid_search` needs. Set ``False`` when
    #: attaching to a table without one; ``hybrid_search_available`` follows.
    fulltext: bool = True
    #: ``None`` probes the server once; ``True`` / ``False`` skips the probe.
    server_side_distance: bool | None = None
    #: Rows fetched per query when distances are computed client-side.
    candidate_limit: int = 10_000

    def model_post_init(self, __context: Any) -> None:
        """Validate identifiers and bounds before anything reaches SQL."""
        _validate_sql_identifier(self.database, "database")
        _validate_sql_identifier(self.table_name, "table_name")
        _validate_sql_identifier(self.charset, "charset")
        _validate_sql_identifier(self.collation, "collation")
        if self.distance_metric.upper() not in _ALLOWED_DISTANCE_METRICS:
            raise ValueError(
                f"Invalid distance_metric: {self.distance_metric!r}. "
                f"Must be one of: {sorted(_ALLOWED_DISTANCE_METRICS)}"
            )
        if not 1 <= self.dimension <= _MAX_VECTOR_DIMENSION:
            raise ValueError(f"dimension must be between 1 and {_MAX_VECTOR_DIMENSION}")
        if self.candidate_limit < 1:
            raise ValueError("candidate_limit must be >= 1")
        if self.pool_size < 1:
            raise ValueError("pool_size must be >= 1")


class _SyncPool:
    """A small thread-safe pool of synchronous Connector/Python connections.

    Every statement runs in a worker thread (``asyncio.to_thread``), so the
    pool is guarded by threading primitives and its connections never bind
    to an event loop — a store built in one ``asyncio.run()`` keeps working
    in the next.
    """

    def __init__(self, connect: Callable[[], Any], max_size: int) -> None:
        self._connect = connect
        self._idle: list[Any] = []
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(max_size)
        self._closed = False

    @contextmanager
    def acquire(self) -> Iterator[Any]:
        """Hand out a healthy connection and return it afterwards.

        A connection whose statement raised is discarded rather than pooled:
        a failed statement may leave the session in an unknown state.
        """
        if self._closed:
            raise RuntimeError("HeatWaveVectorStore is closed")
        self._slots.acquire()
        conn: Any = None
        try:
            conn = self._take()
            yield conn
        except BaseException:
            if conn is not None:
                self._discard(conn)
            raise
        else:
            self._release(conn)
        finally:
            self._slots.release()

    def _take(self) -> Any:
        while True:
            with self._lock:
                conn = self._idle.pop() if self._idle else None
            if conn is None:
                return self._connect()
            if self._is_healthy(conn):
                return conn
            self._discard(conn)

    @staticmethod
    def _is_healthy(conn: Any) -> bool:
        try:
            return bool(conn.is_connected())
        except Exception:  # noqa: BLE001 — any driver error means the connection is unusable
            return False

    def _release(self, conn: Any) -> None:
        # Roll back so no pooled connection sits idle in a transaction holding
        # a metadata lock or a stale snapshot (see the MySQL checkpoint backend).
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001 — a connection that cannot roll back is not pooled
            self._discard(conn)
            return
        with self._lock:
            if not self._closed:
                self._idle.append(conn)
                return
        self._discard(conn)

    @staticmethod
    def _discard(conn: Any) -> None:
        try:
            conn.close()
        except Exception:  # noqa: BLE001 — best-effort close of a connection already given up on
            logger.debug("HeatWaveVectorStore: closing a bad connection failed", exc_info=True)

    def close(self) -> None:
        """Close every idle connection and refuse new acquisitions."""
        with self._lock:
            self._closed = True
            idle, self._idle = self._idle, []
        for conn in idle:
            self._discard(conn)


class HeatWaveVectorStore(BaseVectorStore):
    """Vector store on MySQL HeatWave (and plain MySQL 9+) ``VECTOR`` columns.

    Args:
        host: Server host. Ignored when ``dsn`` is given.
        port: Server port (default ``3306``).
        database: Schema holding the table.
        user: Database user.
        password: Password (``str`` or ``SecretStr``).
        dsn: ``mysql://user:password@host:port/database?opt=val`` instead of
            the individual settings.
        dimension: Embedding dimension; fixed in the ``VECTOR(n)`` column.
        distance_metric: ``COSINE`` (default), ``DOT`` or ``EUCLIDEAN`` — the
            metrics ``DISTANCE()`` accepts, computed the same way client-side.
        **kwargs: Any other :class:`HeatWaveVectorConfig` field
            (``table_name``, ``auto_create_table``, ``fulltext``,
            ``server_side_distance``, ``candidate_limit``, ``pool_size``,
            ``ssl_ca``, ``connect_args``, ...).

    The table is created on first use (``auto_create_table=True``)::

        CREATE TABLE IF NOT EXISTS `locus_vectors` (
            id           VARCHAR(255) PRIMARY KEY,
            content      LONGTEXT NOT NULL,
            content_type VARCHAR(64) NOT NULL DEFAULT 'text',
            embedding    VECTOR(1024) NOT NULL,
            metadata     JSON NOT NULL,
            created_at   DATETIME(6) NOT NULL,
            FULLTEXT KEY `ft_locus_vectors_content` (content)
        ) ENGINE=InnoDB

    Scores are ``1 - cosine distance`` for ``COSINE``, ``(dot + 1) / 2`` for
    ``DOT`` (unit vectors land in ``[0, 1]``) and ``1 / (1 + distance)`` for
    ``EUCLIDEAN``; ``SearchResult.distance`` is lower-is-closer on every
    metric — for ``DOT`` it is the negated dot product (``DISTANCE()`` itself
    returns the dot product for that metric, so the store negates it).

    Example::

        store = HeatWaveVectorStore(
            host="10.0.1.5",
            database="kb",
            user="locus",
            password=os.environ["LOCUS_DB_PASSWORD"],
            dimension=1024,
        )
        await store.add(document)
        results = await store.search(query_embedding, limit=5)
        print(store.server_side_distance)  # True on HeatWave MySQL
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        database: str = "locus",
        user: str = "root",
        password: str | SecretStr = "",
        dsn: str | None = None,
        dimension: int = 1024,
        distance_metric: str = "COSINE",
        **kwargs: Any,
    ) -> None:
        self.heatwave_config = HeatWaveVectorConfig(
            host=host,
            port=port,
            database=database,
            user=user,
            password=SecretStr(password) if isinstance(password, str) else password,
            dsn=dsn,
            dimension=dimension,
            distance_metric=distance_metric,
            **kwargs,
        )
        # A thread pool of synchronous connections, not an asyncio resource:
        # nothing in it belongs to an event loop, so no loop binding is needed.
        self._sync_pool: _SyncPool | None = None
        self._sync_pool_lock = threading.Lock()
        self._table_ready = False
        self._server_side_distance: bool | None = self.heatwave_config.server_side_distance

    # ----------------------------------------------------------------- config

    @property
    def config(self) -> VectorStoreConfig:
        """Store configuration in the protocol's vocabulary."""
        metric = self.heatwave_config.distance_metric.upper()
        return VectorStoreConfig(
            dimension=self.heatwave_config.dimension,
            distance_metric={"COSINE": "cosine", "DOT": "dot_product", "EUCLIDEAN": "l2"}[metric],
            index_type="hnsw" if self._server_side_distance else "flat",
        )

    @property
    def server_side_distance(self) -> bool:
        """``True`` when the server computes ``DISTANCE()`` (HeatWave MySQL).

        Detected on the first operation (or :meth:`ensure_ready`) unless the
        ``server_side_distance`` option pinned it; ``False`` before that and
        on community / commercial MySQL, where distances are computed here.
        """
        return bool(self._server_side_distance)

    @property
    def hybrid_search_available(self) -> bool:
        """``True`` when the table carries the ``FULLTEXT`` index hybrid search needs."""
        return self.heatwave_config.fulltext

    @property
    def _table(self) -> str:
        return _quote_identifier(self.heatwave_config.table_name)

    @property
    def _metric(self) -> str:
        return self.heatwave_config.distance_metric.upper()

    # ------------------------------------------------------------ connections

    def _connect_kwargs(self) -> dict[str, Any]:
        cfg = self.heatwave_config
        if cfg.dsn:
            kwargs = _parse_dsn(cfg.dsn)
        else:
            kwargs = {
                "host": cfg.host,
                "port": cfg.port,
                "database": cfg.database,
                "user": cfg.user,
                "password": cfg.password.get_secret_value(),
            }
        kwargs.setdefault("charset", cfg.charset)
        kwargs.setdefault("collation", cfg.collation)
        kwargs.setdefault("autocommit", False)
        if cfg.ssl_ca is not None:
            kwargs.setdefault("ssl_ca", cfg.ssl_ca)
        kwargs.update(cfg.connect_args)
        return kwargs

    def _connect(self) -> Any:
        try:
            connector = importlib.import_module("mysql.connector")
        except ImportError as e:
            raise ImportError(_MISSING_DRIVER) from e
        return connector.connect(**self._connect_kwargs())

    def _get_pool(self) -> _SyncPool:
        with self._sync_pool_lock:
            if self._sync_pool is None:
                self._sync_pool = _SyncPool(self._connect, self.heatwave_config.pool_size)
            return self._sync_pool

    async def _run(self, fn: Callable[[Any], T]) -> T:
        """Run ``fn(connection)`` on a worker thread with a pooled connection."""

        def work() -> T:
            with self._get_pool().acquire() as conn:
                return fn(conn)

        return await asyncio.to_thread(work)

    @staticmethod
    def _execute(
        conn: Any, sql: str, params: list[Any] | tuple[Any, ...] | None = None
    ) -> tuple[list[tuple[Any, ...]], int]:
        """Execute one statement; return ``(rows, rowcount)``."""
        cursor = conn.cursor()
        try:
            cursor.execute(sql, tuple(params) if params else None)
            rows = cursor.fetchall() if cursor.description else []
            return list(rows), int(cursor.rowcount or 0)
        finally:
            cursor.close()

    # ------------------------------------------------------------------ setup

    def _create_table_sql(self) -> str:
        cfg = self.heatwave_config
        fulltext = (
            f",\n    FULLTEXT KEY `ft_{cfg.table_name[:50]}_content` (content)"
            if cfg.fulltext
            else ""
        )
        return (
            f"CREATE TABLE IF NOT EXISTS {self._table} (\n"
            "    id VARCHAR(255) PRIMARY KEY,\n"
            "    content LONGTEXT NOT NULL,\n"
            "    content_type VARCHAR(64) NOT NULL DEFAULT 'text',\n"
            f"    embedding VECTOR({cfg.dimension}) NOT NULL,\n"
            "    metadata JSON NOT NULL,\n"
            "    created_at DATETIME(6) NOT NULL"
            f"{fulltext}\n"
            f") ENGINE=InnoDB DEFAULT CHARSET={cfg.charset} COLLATE={cfg.collation}"
        )

    def _ensure_ready_sync(self, conn: Any) -> None:
        if not self._table_ready:
            if self.heatwave_config.auto_create_table:
                self._execute(conn, self._create_table_sql())
                conn.commit()
            self._table_ready = True
        if self._server_side_distance is None:
            self._server_side_distance = self._probe_distance(conn)
            if self._server_side_distance:
                logger.info(
                    "HeatWaveVectorStore: server computes DISTANCE(); using server-side search"
                )
            else:
                logger.info(
                    "HeatWaveVectorStore: server has no DISTANCE(); computing similarity "
                    "client-side over up to %d candidate rows per query",
                    self.heatwave_config.candidate_limit,
                )

    def _probe_distance(self, conn: Any) -> bool:
        try:
            self._execute(conn, _DISTANCE_PROBE_SQL)
        except Exception as exc:
            if _is_missing_function(exc):
                conn.rollback()
                return False
            raise
        return True

    async def ensure_ready(self) -> None:
        """Create the table (when allowed) and detect ``DISTANCE()`` support."""
        await self._run(self._ensure_ready_sync)

    async def load_into_heatwave(self) -> None:
        """Load the table into the HeatWave cluster for accelerated queries.

        Issues ``ALTER TABLE ... SECONDARY_ENGINE = RAPID`` followed by
        ``ALTER TABLE ... SECONDARY_LOAD``. Needs a HeatWave cluster attached
        to the DB system; on a server without one the statements fail with
        the server's error.
        """

        def work(conn: Any) -> None:
            self._ensure_ready_sync(conn)
            self._execute(conn, f"ALTER TABLE {self._table} SECONDARY_ENGINE = RAPID")
            self._execute(conn, f"ALTER TABLE {self._table} SECONDARY_LOAD")
            conn.commit()

        await self._run(work)

    # ------------------------------------------------------------------- rows

    _SELECT_COLUMNS = "id, content, content_type, VECTOR_TO_STRING(embedding), metadata, created_at"

    @staticmethod
    def _to_document(row: tuple[Any, ...]) -> Document:
        created = row[5]
        if isinstance(created, datetime):
            created_at = created if created.tzinfo else created.replace(tzinfo=UTC)
        else:
            created_at = datetime.now(UTC)
        content = row[1]
        if isinstance(content, bytes | bytearray):
            content = bytes(content).decode("utf-8")
        return Document(
            id=str(row[0]),
            content=content or "",
            embedding=_string_to_vector(row[3]),
            metadata=dict(_decode_json(row[4]) or {}),
            created_at=created_at,
            content_type=str(row[2] or "text"),
        )

    def _upsert_sql(self) -> str:
        return (
            f"INSERT INTO {self._table} "
            "(id, content, content_type, embedding, metadata, created_at) "
            "VALUES (%s, %s, %s, STRING_TO_VECTOR(%s), %s, %s) AS new "
            "ON DUPLICATE KEY UPDATE content = new.content, "
            "content_type = new.content_type, embedding = new.embedding, "
            "metadata = new.metadata, created_at = new.created_at"
        )

    def _upsert_params(self, doc: Document) -> list[Any]:
        if doc.embedding is None:
            raise ValueError(f"Document {doc.id!r} must have an embedding")
        if len(doc.embedding) != self.heatwave_config.dimension:
            raise ValueError(
                f"Document {doc.id!r} embedding has {len(doc.embedding)} dimensions; "
                f"the store expects {self.heatwave_config.dimension}"
            )
        created = doc.created_at
        if created.tzinfo is not None:
            created = created.astimezone(UTC).replace(tzinfo=None)
        return [
            doc.id,
            doc.content,
            doc.content_type,
            _vector_to_string(doc.embedding),
            json.dumps(doc.metadata),
            created,
        ]

    # -------------------------------------------------------------------- API

    async def add(self, document: Document) -> str:
        """Insert or replace one document (by id)."""
        return (await self.add_batch([document]))[0]

    async def add_batch(self, documents: list[Document]) -> list[str]:
        """Insert or replace documents in one transaction."""
        if not documents:
            return []
        rows = [self._upsert_params(doc) for doc in documents]
        sql = self._upsert_sql()

        def work(conn: Any) -> None:
            self._ensure_ready_sync(conn)
            cursor = conn.cursor()
            try:
                cursor.executemany(sql, rows)
            finally:
                cursor.close()
            conn.commit()

        await self._run(work)
        return [doc.id for doc in documents]

    async def get(self, doc_id: str) -> Document | None:
        """Fetch one document by id."""
        sql = f"SELECT {self._SELECT_COLUMNS} FROM {self._table} WHERE id = %s"

        def work(conn: Any) -> Document | None:
            self._ensure_ready_sync(conn)
            rows, _ = self._execute(conn, sql, [doc_id])
            return self._to_document(rows[0]) if rows else None

        return await self._run(work)

    async def delete(self, doc_id: str) -> bool:
        """Delete one document; ``True`` when a row went away."""
        sql = f"DELETE FROM {self._table} WHERE id = %s"

        def work(conn: Any) -> bool:
            self._ensure_ready_sync(conn)
            _, rowcount = self._execute(conn, sql, [doc_id])
            conn.commit()
            return rowcount > 0

        return await self._run(work)

    # ----------------------------------------------------------------- search

    def _distance_expr(self) -> str:
        """Server-side distance, lower is closer for every metric.

        ``DISTANCE(..., 'DOT')`` on HeatWave MySQL returns the dot product
        itself (``1`` for identical unit vectors, ``0`` for orthogonal ones),
        so it is negated here to be a distance like the other two metrics and
        like the client-side computation.
        """
        expr = f"DISTANCE(embedding, STRING_TO_VECTOR(%s), '{self._metric}')"
        return f"-{expr}" if self._metric == "DOT" else expr

    def _score(self, distance: float) -> float:
        """Map a raw distance to a similarity in ``[0, 1]``."""
        if self._metric == "COSINE":
            return max(0.0, min(1.0, 1.0 - distance))
        if self._metric == "DOT":
            return max(0.0, min(1.0, (1.0 - distance) / 2.0))
        return 1.0 / (1.0 + max(0.0, distance))

    def _client_distance(self, a: list[float], b: list[float]) -> float:
        """Compute the configured distance here, as ``DISTANCE()`` would."""
        if len(a) != len(b):
            raise ValueError(f"vector dimension mismatch: {len(a)} vs {len(b)}")
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        if self._metric == "DOT":
            return -dot
        if self._metric == "EUCLIDEAN":
            return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))
        norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
        if norm == 0.0:
            return 1.0
        return 1.0 - dot / norm

    def _search_sync(
        self,
        conn: Any,
        query_embedding: list[float],
        limit: int,
        metadata_filter: dict[str, Any] | None,
    ) -> list[tuple[Document, float]]:
        """Return ``(document, distance)`` pairs, nearest first."""
        self._ensure_ready_sync(conn)
        where_sql, where_params = build_filter_sql(metadata_filter)
        where = f" WHERE {where_sql}" if where_sql else ""
        if self._server_side_distance:
            sql = (
                f"SELECT {self._SELECT_COLUMNS}, {self._distance_expr()} AS distance_ "
                f"FROM {self._table}{where} ORDER BY distance_ ASC LIMIT %s"
            )
            params = [_vector_to_string(query_embedding), *where_params, limit]
            rows, _ = self._execute(conn, sql, params)
            return [(self._to_document(row), float(row[6])) for row in rows]

        sql = f"SELECT {self._SELECT_COLUMNS} FROM {self._table}{where} LIMIT %s"
        rows, _ = self._execute(conn, sql, [*where_params, self.heatwave_config.candidate_limit])
        scored: list[tuple[Document, float]] = []
        for row in rows:
            doc = self._to_document(row)
            if doc.embedding is None:
                continue
            scored.append((doc, self._client_distance(query_embedding, doc.embedding)))
        scored.sort(key=lambda pair: pair[1])
        return scored[:limit]

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Nearest documents by the configured metric.

        On HeatWave MySQL the server orders by ``DISTANCE()``; elsewhere up
        to ``candidate_limit`` rows (after the metadata filter) are fetched
        and ranked here. ``threshold`` is a minimum similarity score.
        """
        if isinstance(threshold, str):
            try:
                threshold = float(threshold)
            except ValueError:
                threshold = None
        query = [float(x) for x in query_embedding]
        pairs = await self._run(lambda conn: self._search_sync(conn, query, limit, metadata_filter))
        results: list[SearchResult] = []
        for doc, distance in pairs:
            score = self._score(distance)
            if threshold is not None and score < threshold:
                continue
            results.append(SearchResult(document=doc, score=score, distance=distance))
        return results

    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        *,
        limit: int = 10,
        alpha: float = 0.5,
        threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Blend vector similarity with ``MATCH ... AGAINST`` relevance on ``content``.

        Both legs fetch ``max(limit * 3, 20)`` candidates under the same
        metadata filter; the lexical leg is InnoDB full-text search in
        natural language mode over the table's ``FULLTEXT`` index. Scores are
        normalised to ``[0, 1]`` and combined as
        ``alpha * vector + (1 - alpha) * lexical``; an empty ``query_text``
        is answered by :meth:`search`.

        ``threshold`` applies to the blended score. When only one leg returns
        hits, that leg's scores are returned unblended, so a threshold chosen
        for vector search still keeps an exact match.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")
        if not self.heatwave_config.fulltext:
            raise RuntimeError(
                "hybrid_search needs the FULLTEXT index; the store was built with fulltext=False"
            )
        if not query_text.strip():
            return await self.search(
                query_embedding, limit=limit, threshold=threshold, metadata_filter=metadata_filter
            )
        pool = max(limit * 3, 20)
        vector_hits = await self.search(
            query_embedding, limit=pool, metadata_filter=metadata_filter
        )

        where_sql, where_params = build_filter_sql(metadata_filter)
        match = "MATCH(content) AGAINST (%s IN NATURAL LANGUAGE MODE)"
        where = f" WHERE {match}" + (f" AND {where_sql}" if where_sql else "")
        sql = (
            f"SELECT {self._SELECT_COLUMNS}, {match} AS lexical_ "
            f"FROM {self._table}{where} ORDER BY lexical_ DESC LIMIT %s"
        )
        params = [query_text, query_text, *where_params, pool]

        def work(conn: Any) -> list[tuple[Document, float]]:
            self._ensure_ready_sync(conn)
            rows, _ = self._execute(conn, sql, params)
            return [(self._to_document(row), float(row[6] or 0.0)) for row in rows]

        lexical_hits = await self._run(work)
        return blend_hybrid(
            vector_hits, lexical_hits, alpha=alpha, limit=limit, threshold=threshold
        )

    # ------------------------------------------------------------ bookkeeping

    async def count(self) -> int:
        """Number of documents in the table."""
        sql = f"SELECT COUNT(*) FROM {self._table}"

        def work(conn: Any) -> int:
            self._ensure_ready_sync(conn)
            rows, _ = self._execute(conn, sql)
            return int(rows[0][0]) if rows else 0

        return await self._run(work)

    async def clear(self) -> int:
        """Delete every document; returns how many went away."""
        sql = f"DELETE FROM {self._table}"

        def work(conn: Any) -> int:
            self._ensure_ready_sync(conn)
            _, rowcount = self._execute(conn, sql)
            conn.commit()
            return rowcount

        return await self._run(work)

    async def close(self) -> None:
        """Close the pooled connections; the store reconnects on next use."""
        with self._sync_pool_lock:
            pool, self._sync_pool = self._sync_pool, None
        if pool is not None:
            await asyncio.to_thread(pool.close)
        self._table_ready = False

    def __repr__(self) -> str:
        cfg = self.heatwave_config
        if cfg.dsn:
            return f"HeatWaveVectorStore(dsn=..., table={cfg.table_name!r})"
        return (
            f"HeatWaveVectorStore(host={cfg.host!r}, database={cfg.database!r}, "
            f"table={cfg.table_name!r})"
        )
