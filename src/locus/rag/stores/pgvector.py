# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""PostgreSQL pgvector store.

pgvector is a PostgreSQL extension for vector similarity search,
perfect for adding vector capabilities to existing PostgreSQL databases.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, Field, SecretStr

from locus.rag.stores.base import (
    BaseVectorStore,
    Document,
    SearchResult,
    VectorStoreConfig,
)


if TYPE_CHECKING:
    import asyncpg


_SAFE_SQL_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,62}$")
_ALLOWED_DISTANCE_METRICS = frozenset({"cosine", "l2", "inner_product", "ip"})
_ALLOWED_INDEX_TYPES = frozenset({"ivfflat", "hnsw", "none"})


def _validate_sql_identifier(value: str, field_name: str) -> str:
    """Validate that a string is a safe SQL identifier."""
    if not _SAFE_SQL_IDENTIFIER.match(value):
        msg = (
            f"Invalid {field_name}: {value!r}. "
            "Must start with a letter or underscore and contain only "
            "alphanumeric characters and underscores (max 63 chars)."
        )
        raise ValueError(msg)
    return value


class PgVectorConfig(BaseModel):
    """Configuration for PostgreSQL pgvector Store."""

    # Connection options
    dsn: str | None = Field(
        default=None,
        description="PostgreSQL connection string (postgresql://user:pass@host:port/db)",
    )
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    database: str = Field(default="postgres", description="Database name")
    user: str = Field(default="postgres", description="Database user")
    password: SecretStr = Field(default=SecretStr(""), description="Database password")

    # Table settings
    table_name: str = Field(default="locus_vectors", description="Table name")
    schema_name: str = Field(default="public", description="Schema name")

    # Vector settings
    dimension: int = Field(default=1536, description="Vector dimension")
    distance_metric: str = Field(
        default="cosine",
        description="Distance metric: cosine, l2, inner_product",
    )

    # Index settings
    index_type: str = Field(
        default="ivfflat",
        description="Index type: ivfflat, hnsw, none",
    )
    auto_index: bool = Field(
        default=False,
        description="Auto-create vector index (set False for small datasets)",
    )
    min_rows_for_index: int = Field(
        default=1000,
        description="Minimum rows before creating IVFFlat index",
    )
    ivf_lists: int = Field(default=100, description="IVF lists for ivfflat index")
    hnsw_m: int = Field(default=16, description="HNSW M parameter")
    hnsw_ef_construction: int = Field(default=64, description="HNSW ef_construction")

    # Pool settings
    min_pool_size: int = Field(default=1, description="Minimum pool size")
    max_pool_size: int = Field(default=10, description="Maximum pool size")

    def model_post_init(self, __context: Any) -> None:
        """Validate SQL identifiers and metric/index allowlists to prevent injection."""
        _validate_sql_identifier(self.table_name, "table_name")
        _validate_sql_identifier(self.schema_name, "schema_name")
        if self.distance_metric.lower() not in _ALLOWED_DISTANCE_METRICS:
            raise ValueError(
                f"Invalid distance_metric: {self.distance_metric!r}. "
                f"Must be one of: {sorted(_ALLOWED_DISTANCE_METRICS)}"
            )
        if self.index_type.lower() not in _ALLOWED_INDEX_TYPES:
            raise ValueError(
                f"Invalid index_type: {self.index_type!r}. "
                f"Must be one of: {sorted(_ALLOWED_INDEX_TYPES)}"
            )


class PgVectorStore(BaseModel, BaseVectorStore):
    """
    PostgreSQL pgvector store.

    pgvector adds vector similarity search to PostgreSQL with:
    - IVFFlat and HNSW indexing
    - Cosine, L2, and inner product distance
    - Integration with existing PostgreSQL data
    - ACID transactions

    Prerequisites:
        1. Install pgvector extension: CREATE EXTENSION vector;
        2. Install asyncpg: pip install asyncpg

    Example (DSN):
        >>> store = PgVectorStore(
        ...     dsn="postgresql://user:pass@localhost:5432/mydb",
        ...     table_name="documents",
        ...     dimension=1536,
        ... )
        >>> await store.add(document)
        >>> results = await store.search(query_embedding, limit=5)

    Example (Individual params):
        >>> store = PgVectorStore(
        ...     host="localhost",
        ...     database="mydb",
        ...     user="postgres",
        ...     password="secret",
        ...     dimension=1536,
        ... )

    Note:
        The pgvector extension must be installed in your PostgreSQL database.
        Run: CREATE EXTENSION IF NOT EXISTS vector;
    """

    pgvector_config: PgVectorConfig = Field(default_factory=PgVectorConfig)
    _pool: asyncpg.Pool | None = None
    _initialized: bool = False

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        dsn: str | None = None,
        host: str = "localhost",
        port: int = 5432,
        database: str = "postgres",
        user: str = "postgres",
        password: str | SecretStr = "",
        table_name: str = "locus_vectors",
        dimension: int = 1536,
        distance_metric: str = "cosine",
        **kwargs: Any,
    ) -> None:
        pgvector_config = PgVectorConfig(
            dsn=dsn,
            host=host,
            port=port,
            database=database,
            user=user,
            password=SecretStr(password) if isinstance(password, str) else password,
            table_name=table_name,
            dimension=dimension,
            distance_metric=distance_metric,
            **kwargs,
        )
        super().__init__(pgvector_config=pgvector_config)

    @property
    def config(self) -> VectorStoreConfig:
        """Get store configuration."""
        return VectorStoreConfig(
            dimension=self.pgvector_config.dimension,
            distance_metric=self.pgvector_config.distance_metric,
            index_type=self.pgvector_config.index_type,
        )

    @property
    def _full_table_name(self) -> str:
        """Get fully qualified table name."""
        return f"{self.pgvector_config.schema_name}.{self.pgvector_config.table_name}"

    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create connection pool."""
        if self._pool is None:
            try:
                import asyncpg
            except ImportError as e:
                raise ImportError(
                    "PgVectorStore requires 'asyncpg'. Install with: pip install asyncpg"
                ) from e

            # Build DSN if not provided
            dsn = self.pgvector_config.dsn
            if dsn is None:
                dsn = (
                    f"postgresql://{self.pgvector_config.user}:"
                    f"{self.pgvector_config.password.get_secret_value()}@"
                    f"{self.pgvector_config.host}:{self.pgvector_config.port}/"
                    f"{self.pgvector_config.database}"
                )

            self._pool = await asyncpg.create_pool(
                dsn,
                min_size=self.pgvector_config.min_pool_size,
                max_size=self.pgvector_config.max_pool_size,
            )

        return self._pool

    async def _ensure_table(self) -> None:
        """Create table and index if not exists."""
        if self._initialized:
            return

        pool = await self._get_pool()
        dim = self.pgvector_config.dimension
        table = self._full_table_name
        table_name = self.pgvector_config.table_name

        async with pool.acquire() as conn:
            # Ensure pgvector extension exists
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    embedding vector({dim}),
                    metadata JSONB DEFAULT '{{}}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Note: Vector indexes (IVFFlat/HNSW) are NOT created automatically
            # IVFFlat requires data to work properly; creating on empty table causes issues
            # Use create_index() method after loading data, or set auto_index=True
            # to have the index created automatically when min_rows_for_index is reached

            # Create metadata index for filtering
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_metadata
                ON {table} USING gin (metadata)
            """)

        self._initialized = True

    async def add(self, document: Document) -> str:
        """Add a document."""
        await self._ensure_table()
        pool = await self._get_pool()

        doc_id = document.id or uuid4().hex

        if document.embedding is None:
            raise ValueError("Document must have an embedding")

        # Convert embedding to pgvector format
        embedding_str = "[" + ",".join(str(x) for x in document.embedding) + "]"

        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self._full_table_name}
                (id, content, embedding, metadata, created_at)
                VALUES ($1, $2, $3::vector, $4, $5)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    created_at = EXCLUDED.created_at
            """,
                doc_id,
                document.content,
                embedding_str,
                json.dumps(document.metadata),
                document.created_at,
            )

        return doc_id

    async def add_batch(self, documents: list[Document]) -> list[str]:
        """Add multiple documents."""
        await self._ensure_table()
        pool = await self._get_pool()

        ids = []

        async with pool.acquire() as conn, conn.transaction():
            for doc in documents:
                doc_id = doc.id or uuid4().hex
                ids.append(doc_id)

                if doc.embedding is None:
                    raise ValueError(f"Document {doc_id} must have an embedding")

                embedding_str = "[" + ",".join(str(x) for x in doc.embedding) + "]"

                await conn.execute(
                    f"""
                        INSERT INTO {self._full_table_name}
                        (id, content, embedding, metadata, created_at)
                        VALUES ($1, $2, $3::vector, $4, $5)
                        ON CONFLICT (id) DO UPDATE SET
                            content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata,
                            created_at = EXCLUDED.created_at
                    """,
                    doc_id,
                    doc.content,
                    embedding_str,
                    json.dumps(doc.metadata),
                    doc.created_at,
                )

        return ids

    async def get(self, doc_id: str) -> Document | None:
        """Get a document by ID."""
        await self._ensure_table()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT id, content, embedding::text, metadata, created_at
                FROM {self._full_table_name}
                WHERE id = $1
            """,
                doc_id,
            )

        if row is None:
            return None

        # Parse embedding from text format [x,y,z]
        embedding_str = row["embedding"]
        if embedding_str:
            embedding_str = embedding_str.strip("[]")
            embedding = [float(x) for x in embedding_str.split(",")]
        else:
            embedding = None

        return Document(
            id=row["id"],
            content=row["content"],
            embedding=embedding,
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=row["created_at"] or datetime.now(UTC),
        )

    async def delete(self, doc_id: str) -> bool:
        """Delete a document."""
        await self._ensure_table()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result: str = await conn.execute(
                f"""
                DELETE FROM {self._full_table_name}
                WHERE id = $1
            """,
                doc_id,
            )

        return result == "DELETE 1"

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents."""
        await self._ensure_table()
        pool = await self._get_pool()

        # Convert embedding to pgvector format
        query_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        # Map distance metric to operator
        operator_map = {
            "cosine": "<=>",  # Cosine distance
            "l2": "<->",  # L2 distance
            "inner_product": "<#>",  # Negative inner product
            "ip": "<#>",
        }
        operator = operator_map.get(
            self.pgvector_config.distance_metric.lower(),
            "<=>",
        )

        # Build WHERE clause for metadata filtering
        where_clauses = []
        params = [query_str, limit]
        param_idx = 3

        if metadata_filter:
            for key, value in metadata_filter.items():
                # Keys are interpolated into SQL; reject anything that is not a safe identifier.
                if not isinstance(key, str) or not key.isidentifier():
                    raise ValueError(
                        f"Invalid metadata filter key: {key!r}. "
                        "Keys must be valid Python identifiers."
                    )
                where_clauses.append(f"metadata->>'{key}' = ${param_idx}")
                params.append(str(value))
                param_idx += 1

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, content, embedding::text, metadata, created_at,
                       embedding {operator} $1::vector AS distance
                FROM {self._full_table_name}
                {where_sql}
                ORDER BY distance ASC
                LIMIT $2
            """,
                *params,
            )

        results = []
        for row in rows:
            distance = row["distance"]

            # Convert distance to similarity score (0-1, higher is better)
            if self.pgvector_config.distance_metric.lower() == "cosine":
                # Cosine distance is 0-2, convert to similarity
                score = 1.0 - (distance / 2.0)
            elif self.pgvector_config.distance_metric.lower() == "l2":
                # L2 distance: use exponential decay
                score = 1.0 / (1.0 + distance)
            else:  # inner_product
                # Negative inner product, higher is better
                score = max(0.0, min(1.0, -distance))

            if threshold is not None and score < threshold:
                continue

            # Parse embedding
            embedding_str = row["embedding"]
            if embedding_str:
                embedding_str = embedding_str.strip("[]")
                embedding = [float(x) for x in embedding_str.split(",")]
            else:
                embedding = None

            doc = Document(
                id=row["id"],
                content=row["content"],
                embedding=embedding,
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                created_at=row["created_at"] or datetime.now(UTC),
            )

            results.append(
                SearchResult(
                    document=doc,
                    score=score,
                    distance=distance,
                )
            )

        return results

    async def count(self) -> int:
        """Count documents."""
        await self._ensure_table()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            count = await conn.fetchval(f"""
                SELECT COUNT(*) FROM {self._full_table_name}
            """)

        return count or 0

    async def clear(self) -> int:
        """Delete all documents."""
        await self._ensure_table()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            count = await conn.fetchval(f"""
                SELECT COUNT(*) FROM {self._full_table_name}
            """)
            await conn.execute(f"TRUNCATE TABLE {self._full_table_name}")

        return count or 0

    async def create_index(self, index_type: str | None = None) -> bool:
        """
        Create vector index for faster similarity search.

        Should be called after loading data. IVFFlat indexes require
        data to determine optimal list assignments.

        Args:
            index_type: Override index type ("ivfflat" or "hnsw")

        Returns:
            True if index was created, False if already exists

        Example:
            >>> await store.add_batch(documents)
            >>> await store.create_index()  # Now create the index
        """
        await self._ensure_table()
        pool = await self._get_pool()
        table = self._full_table_name
        table_name = self.pgvector_config.table_name
        idx_type = index_type or self.pgvector_config.index_type

        async with pool.acquire() as conn:
            # Check if index exists
            index_exists = await conn.fetchval(f"""
                SELECT EXISTS (
                    SELECT 1 FROM pg_indexes
                    WHERE indexname = 'idx_{table_name}_embedding'
                )
            """)

            if index_exists:
                return False

            # Get row count to adjust IVFFlat lists
            row_count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")

            # Map distance metric to operator class
            op_class_map = {
                "cosine": "vector_cosine_ops",
                "l2": "vector_l2_ops",
                "inner_product": "vector_ip_ops",
                "ip": "vector_ip_ops",
            }
            op_class = op_class_map.get(
                self.pgvector_config.distance_metric.lower(),
                "vector_cosine_ops",
            )

            if idx_type == "hnsw":
                await conn.execute(f"""
                    CREATE INDEX idx_{table_name}_embedding
                    ON {table}
                    USING hnsw (embedding {op_class})
                    WITH (m = {self.pgvector_config.hnsw_m},
                          ef_construction = {self.pgvector_config.hnsw_ef_construction})
                """)
            elif idx_type == "ivfflat":
                # Adjust lists based on data size
                # Recommended: lists = sqrt(rows) for < 1M rows
                lists = max(1, min(self.pgvector_config.ivf_lists, int(row_count**0.5)))
                await conn.execute(f"""
                    CREATE INDEX idx_{table_name}_embedding
                    ON {table}
                    USING ivfflat (embedding {op_class})
                    WITH (lists = {lists})
                """)
            else:
                # No index (exact search)
                return False

        return True

    async def has_index(self) -> bool:
        """Check if vector index exists."""
        await self._ensure_table()
        pool = await self._get_pool()
        table_name = self.pgvector_config.table_name

        async with pool.acquire() as conn:
            exists: bool = await conn.fetchval(f"""
                SELECT EXISTS (
                    SELECT 1 FROM pg_indexes
                    WHERE indexname = 'idx_{table_name}_embedding'
                )
            """)
            return exists

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    def __repr__(self) -> str:
        return f"PgVectorStore(table={self._full_table_name!r})"
