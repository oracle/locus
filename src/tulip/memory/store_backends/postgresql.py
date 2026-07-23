# SPDX-License-Identifier: Apache-2.0
#
# The holographic (HRR) text encoding reused here is ported/adapted from
# NousResearch/hermes-agent (MIT, © 2025 Nous Research) — see ``holographic.py``.
# The pgvector persistence, per-tenant Row-Level-Security isolation, and the
# [cos φ, sin φ] embedding that makes pgvector cosine distance equal HRR phase
# similarity are Tulip (Apache-2.0).
"""PgMemory — the multi-tenant, RLS-isolated enterprise memory store.

A :class:`~tulip.memory.store.BaseStore` backed by **PostgreSQL + pgvector**
(Aurora in the paid tier, any Postgres locally). It is the enterprise counterpart
to :class:`~tulip.memory.store_backends.holographic.HolographicStore`: same HRR
associative recall, but **persisted, shared, and tenant-isolated**.

Two properties make it the governed backend:

* **Tenant isolation is absolute.** ``namespace[0]`` is the ``tenant`` — a hard
  boundary. Every row carries it, **Row-Level Security** enforces it on read AND
  write (for any non-owner role), and every query *also* filters on it
  explicitly (defence in depth). There is no global index, embedding, or cache
  shared across tenants.
* **No external embedding API.** The HRR phase vector ``φ`` is stored as
  ``[cos φ, sin φ]`` (length ``2·dim``). For unit vectors, pgvector cosine
  similarity of that encoding equals ``mean(cos(φ_a − φ_b))`` — exactly the HRR
  phase similarity — so semantic recall runs entirely inside Postgres with no
  embedding service and no data egress.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from tulip.memory.store import BaseStore, StoreCapabilities, StoreItem
from tulip.memory.store_backends.holographic import _DEFAULT_DIM, _numpy, encode_text


if TYPE_CHECKING:
    from asyncpg import Pool

    from tulip.rag.embeddings.base import BaseEmbedding

_NS_SEP = "\x1f"
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
#: The GUC that carries the active tenant to the RLS policy. `SET LOCAL` scopes it
#: to the transaction, so a pooled connection never leaks one tenant's id to the
#: next checkout.
_TENANT_GUC = "tulip.tenant"


def _validate_ident(value: str, field: str) -> str:
    """Guard a table name that is interpolated into DDL/DML (no bind for identifiers)."""
    if not _IDENT_RE.match(value):
        raise ValueError(f"invalid {field}: {value!r}")
    return value


def _embed_literal(content: str, dim: int) -> str | None:
    """HRR encode ``content`` to the ``[cos φ, sin φ]`` pgvector literal, or ``None``.

    Returns ``None`` when numpy is unavailable — the store then falls back to a
    lexical (``ILIKE``) search instead of vector recall, degrading not failing.
    """
    np = _numpy()
    if np is None:  # pragma: no cover - exercised only on numpy-less installs
        return None
    phases = encode_text(content, dim)
    vec = np.concatenate([np.cos(phases), np.sin(phases)])
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


class PgMemory(BaseStore):
    """Postgres + pgvector memory store with per-tenant Row-Level Security.

    ``namespace[0]`` is the tenant (the isolation boundary); the full namespace
    tuple scopes within a tenant (e.g. ``(tenant, "user", user_id)``). The schema,
    RLS policy, and HNSW index are created on first use, so it works against a
    bare Postgres locally and a managed Aurora cluster identically.

    **Recall quality depends on the vector.** Pass an ``embedder`` (any
    :class:`~tulip.rag.embeddings.base.BaseEmbedding`, e.g. brokered OpenAI
    ``text-embedding-3-small``) for **true semantic recall** — the enterprise
    default. Without one it falls back to the HRR ``[cos φ, sin φ]`` encoding,
    which is **lexical/associative** (matches shared or hashed tokens), not
    trained semantics — fine offline, but it will not match paraphrases the way
    an embedding model does. Choose the embedder to match the value of recall.
    """

    def __init__(
        self,
        dsn: str,
        *,
        table: str = "tulip_memories",
        dim: int = _DEFAULT_DIM,
        embedder: BaseEmbedding | None = None,
    ) -> None:
        self._dsn = dsn
        self._table = _validate_ident(table, "table")
        self._dim = dim
        self._embedder = embedder
        # A real embedder fixes the column width; HRR [cos, sin] doubles `dim`.
        self._vdim = embedder.dimension if embedder is not None else 2 * dim
        self._pool: Pool | None = None

    async def _embed(self, content: str) -> str | None:
        """The pgvector literal for ``content`` — real embedding if configured,
        else the HRR ``[cos φ, sin φ]`` fallback (``None`` without numpy)."""
        if self._embedder is not None:
            result = await self._embedder.embed(content)
            return "[" + ",".join(f"{x:.6f}" for x in result.embedding) + "]"
        return _embed_literal(content, self._dim)

    async def _get_pool(self) -> Pool:
        if self._pool is None:
            import asyncpg  # noqa: PLC0415

            self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=8)
            await self._ensure_schema()
        return self._pool

    async def _ensure_schema(self) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self._table} ("
                "tenant text NOT NULL, ns text NOT NULL, key text NOT NULL, "
                "value jsonb NOT NULL, metadata jsonb NOT NULL DEFAULT '{}', "
                "content text NOT NULL DEFAULT '', "
                f"embedding vector({self._vdim}), "
                "created_at timestamptz NOT NULL DEFAULT now(), "
                "updated_at timestamptz NOT NULL DEFAULT now(), "
                "version integer NOT NULL DEFAULT 1, "
                "PRIMARY KEY (tenant, ns, key))"
            )
            # RLS — the hard tenant boundary. Enforced for every non-owner role;
            # FORCE also applies it to the table owner (belt for admin roles).
            await conn.execute(f"ALTER TABLE {self._table} ENABLE ROW LEVEL SECURITY")
            await conn.execute(f"ALTER TABLE {self._table} FORCE ROW LEVEL SECURITY")
            await conn.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {self._table}")
            await conn.execute(
                f"CREATE POLICY tenant_isolation ON {self._table} "
                f"USING (tenant = current_setting('{_TENANT_GUC}', true)) "
                f"WITH CHECK (tenant = current_setting('{_TENANT_GUC}', true))"
            )
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self._table}_ann "
                f"ON {self._table} USING hnsw (embedding vector_cosine_ops)"
            )

    @staticmethod
    def _tenant_of(namespace: tuple[str, ...]) -> str:
        if not namespace:
            raise ValueError("namespace must have at least a tenant element")
        return namespace[0]

    @property
    def capabilities(self) -> StoreCapabilities:
        return StoreCapabilities(
            search=True,
            # True *semantic* recall needs a real embedder; the HRR fallback is
            # lexical/associative, so it does not claim semantic_search.
            semantic_search=self._embedder is not None,
            embedding_dimension=self._vdim,
            list_namespaces=True,
        )

    async def _scoped(self, conn: Any, tenant: str) -> None:
        """Pin the tenant for this transaction so RLS admits only its rows."""
        await conn.execute(f"SELECT set_config('{_TENANT_GUC}', $1, true)", tenant)

    async def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        tenant = self._tenant_of(namespace)
        ns = _NS_SEP.join(namespace)
        content = _content_for(value)
        emb = await self._embed(content)
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            await self._scoped(conn, tenant)
            await conn.execute(
                f"INSERT INTO {self._table} "
                "(tenant, ns, key, value, metadata, content, embedding) "
                f"VALUES ($1,$2,$3,$4,$5,$6,$7::vector) "
                "ON CONFLICT (tenant, ns, key) DO UPDATE SET "
                f"value=EXCLUDED.value, metadata=EXCLUDED.metadata, "
                f"content=EXCLUDED.content, embedding=EXCLUDED.embedding, "
                f"updated_at=now(), version={self._table}.version+1",
                tenant,
                ns,
                key,
                json.dumps(value),
                json.dumps(metadata or {}),
                content,
                emb,
            )

    async def get(self, namespace: tuple[str, ...], key: str) -> Any | None:
        tenant = self._tenant_of(namespace)
        ns = _NS_SEP.join(namespace)
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            await self._scoped(conn, tenant)
            row = await conn.fetchrow(
                f"SELECT value FROM {self._table} WHERE tenant=$1 AND ns=$2 AND key=$3",
                tenant,
                ns,
                key,
            )
        return json.loads(row["value"]) if row else None

    async def delete(self, namespace: tuple[str, ...], key: str) -> bool:
        tenant = self._tenant_of(namespace)
        ns = _NS_SEP.join(namespace)
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            await self._scoped(conn, tenant)
            result: str = await conn.execute(
                f"DELETE FROM {self._table} WHERE tenant=$1 AND ns=$2 AND key=$3",
                tenant,
                ns,
                key,
            )
        # asyncpg returns e.g. "DELETE 1" / "DELETE 0" — the tail is the row count.
        return result.rsplit(maxsplit=1)[-1] != "0"

    async def list_keys(self, namespace: tuple[str, ...], limit: int = 100) -> list[str]:
        tenant = self._tenant_of(namespace)
        ns = _NS_SEP.join(namespace)
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            await self._scoped(conn, tenant)
            rows = await conn.fetch(
                f"SELECT key FROM {self._table} WHERE tenant=$1 AND ns=$2 "
                "ORDER BY updated_at DESC LIMIT $3",
                tenant,
                ns,
                limit,
            )
        return [r["key"] for r in rows]

    async def search(
        self, namespace: tuple[str, ...], query: str | None = None, limit: int = 10
    ) -> list[StoreItem]:
        tenant = self._tenant_of(namespace)
        ns = _NS_SEP.join(namespace)
        cols = "key, value, metadata, created_at, updated_at, version"
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            await self._scoped(conn, tenant)
            if not query:
                rows = await conn.fetch(
                    f"SELECT {cols} FROM {self._table} WHERE tenant=$1 AND ns=$2 "
                    "ORDER BY updated_at DESC LIMIT $3",
                    tenant,
                    ns,
                    limit,
                )
            else:
                qvec = await self._embed(query)
                if qvec is None:  # pragma: no cover - numpy-less fallback
                    rows = await conn.fetch(
                        f"SELECT {cols} FROM {self._table} "
                        "WHERE tenant=$1 AND ns=$2 AND content ILIKE '%'||$3||'%' "
                        "ORDER BY updated_at DESC LIMIT $4",
                        tenant,
                        ns,
                        query,
                        limit,
                    )
                else:
                    rows = await conn.fetch(
                        f"SELECT {cols} FROM {self._table} "
                        "WHERE tenant=$1 AND ns=$2 AND embedding IS NOT NULL "
                        f"ORDER BY embedding <=> $3::vector LIMIT $4",
                        tenant,
                        ns,
                        qvec,
                        limit,
                    )
        return [self._row_to_item(namespace, r) for r in rows]

    def _row_to_item(self, namespace: tuple[str, ...], r: Any) -> StoreItem:
        return StoreItem(
            namespace=namespace,
            key=r["key"],
            value=json.loads(r["value"]),
            metadata=json.loads(r["metadata"]) if r["metadata"] else {},
            created_at=_as_dt(r["created_at"]),
            updated_at=_as_dt(r["updated_at"]),
            version=r["version"],
        )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


def _as_dt(value: Any) -> datetime:
    """asyncpg returns ``datetime`` already; be defensive for str rows in tests."""
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value)).replace(tzinfo=UTC)


def _content_for(value: Any) -> str:
    """The searchable text of a stored value — its scalar strings, or a JSON dump."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = [str(v) for v in value.values() if isinstance(v, (str, int, float))]
        return " ".join(parts) if parts else json.dumps(value, sort_keys=True, default=str)
    return json.dumps(value, sort_keys=True, default=str)
