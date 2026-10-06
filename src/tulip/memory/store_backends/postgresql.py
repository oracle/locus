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

import asyncio
import importlib.util
import json
import logging
import re
import warnings
from collections import OrderedDict
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from tulip.core.loop_bound import loop_bound_async
from tulip.memory.store import BaseStore, StoreCapabilities, StoreItem
from tulip.memory.store_backends.holographic import _numpy, encode_text


if TYPE_CHECKING:
    from asyncpg import Pool

    from tulip.rag.embeddings.base import BaseEmbedding

logger = logging.getLogger(__name__)

_NS_SEP = "\x1f"
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
#: The GUC that carries the active tenant to the RLS policy. `SET LOCAL` scopes it
#: to the transaction, so a pooled connection never leaks one tenant's id to the
#: next checkout.
_TENANT_GUC = "tulip.tenant"

#: pgvector's hard ceiling on the width of an indexable column: it refuses to
#: build an HNSW (or IVFFlat) index over more than 2000 dimensions. Storage is
#: fine up to 16000 — only the ANN index is capped.
PGVECTOR_ANN_MAX_DIM = 2000

#: PgMemory's default HRR ``dim``. Deliberately **not** the HolographicStore
#: default (1024): the HRR encoding stores ``[cos φ, sin φ]``, so the pgvector
#: column is ``2·dim`` wide and 1024 would ask for 2048 columns — over
#: :data:`PGVECTOR_ANN_MAX_DIM`, which made ``CREATE INDEX … USING hnsw`` fail
#: and left every default-constructed store unusable. 512 → 1024 columns, well
#: inside the limit with room for pgvector's own headroom.
_DEFAULT_PG_DIM = 512

#: Encodings a store keeps by default (see ``PgMemory(encoding_cache=)``).
_DEFAULT_ENCODING_CACHE = 1024

#: Said once per store, the first time a search falls back to substring matching.
_SUBSTRING_ONLY = (
    "PgMemory: numpy is not installed and no embedder was given, so memories are "
    "stored without vectors and recall is a substring match (content ILIKE "
    "'%query%'): a query only finds memories that contain it word for word. "
    "Install numpy (`pip install 'tulip-agents[pgvector]'` pulls it) for HRR "
    "recall, or pass an embedder."
)


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

    **Width and the ANN index.** Without an embedder the pgvector column is
    ``2·dim`` wide (the ``[cos φ, sin φ]`` pair), and pgvector will not build an
    HNSW index over more than :data:`PGVECTOR_ANN_MAX_DIM` dimensions. ``dim``
    therefore defaults to 512 (a 1024-wide column) and a larger explicit ``dim``
    is rejected at construction rather than at ``CREATE INDEX``. An *embedder*
    wider than the limit is allowed — its width is the model's, not ours — but
    the table is then created without an ANN index and the constructor emits a
    ``RuntimeWarning`` saying so.

    **numpy.** The HRR encoding needs numpy (the ``[pgvector]`` extra installs
    it). Without numpy and without an embedder nothing is wrong at write time,
    but rows are stored with no vector and recall degrades to a substring match;
    the constructor warns (``RuntimeWarning``) and the first degraded search
    logs a warning, so this never happens quietly.

    **Encoding cache.** Encoding a text costs a pure-Python HRR pass (or an
    embedding call), and one recall often encodes the same words several times.
    Each store keeps the last ``encoding_cache`` encodings (``0`` turns the
    cache off), keyed by ``(tenant, text)``: an entry is only ever reused within
    the tenant that computed it, so the cache is not a surface two tenants
    share.

    **Several namespaces at once.** :meth:`search_many` ranks several namespaces
    of ONE tenant in one transaction and one statement, encoding the query once.
    Namespaces of different tenants in one call are refused.
    """

    def __init__(
        self,
        dsn: str,
        *,
        table: str = "tulip_memories",
        dim: int = _DEFAULT_PG_DIM,
        embedder: BaseEmbedding | None = None,
        encoding_cache: int = _DEFAULT_ENCODING_CACHE,
    ) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if encoding_cache < 0:
            raise ValueError(f"encoding_cache must be >= 0, got {encoding_cache}")
        # asyncpg is an optional dependency imported lazily inside the pool
        # builder, so a missing package used to surface as a bare
        # ModuleNotFoundError from deep inside a coroutine on first *use* —
        # long after the mistake, and far from anything that explains it. Same
        # reasoning as the dim check below: a caller mistake belongs at
        # construction, in words that say what to install.
        if importlib.util.find_spec("asyncpg") is None:
            raise ImportError(
                "PgMemory needs the asyncpg driver, which ships as an optional "
                "dependency. Install it with:\n\n"
                "    pip install 'tulip-agents[pgvector]'\n\n"
                "(or `pip install asyncpg numpy` if you manage dependencies yourself)."
            )
        self._dsn = dsn
        self._table = _validate_ident(table, "table")
        self._dim = dim
        self._embedder = embedder
        # A real embedder fixes the column width; HRR [cos, sin] doubles `dim`.
        self._vdim = embedder.dimension if embedder is not None else 2 * dim
        # Fail here, not thousands of lines into a run with a raw driver error:
        # a `dim` we control that cannot carry an ANN index is a caller mistake.
        if embedder is None and self._vdim > PGVECTOR_ANN_MAX_DIM:
            raise ValueError(
                f"PgMemory(dim={dim}) is not indexable: the HRR encoding stores "
                f"[cos φ, sin φ], so the pgvector column would be 2·{dim} = "
                f"{self._vdim} dimensions wide, over pgvector's "
                f"{PGVECTOR_ANN_MAX_DIM}-dimension limit for an HNSW index "
                f"(CREATE INDEX would raise ProgramLimitExceededError). Pass "
                f"dim <= {PGVECTOR_ANN_MAX_DIM // 2} (the default is "
                f"{_DEFAULT_PG_DIM}), or pass an embedder."
            )
        # An embedder's width is *not* ours to choose (text-embedding-3-large is
        # 3072). Rather than reject a legitimate model we keep the column and skip
        # the index — but say so loudly, because a silent sequential scan is
        # exactly the failure this store had.
        self._ann = self._vdim <= PGVECTOR_ANN_MAX_DIM
        if not self._ann:
            warnings.warn(
                f"PgMemory: embedder dimension {self._vdim} exceeds pgvector's "
                f"{PGVECTOR_ANN_MAX_DIM}-dimension limit for an HNSW index, so "
                f"table {self._table!r} will be created WITHOUT an ANN index — "
                f"every search is a sequential scan. Use an embedder with "
                f"<= {PGVECTOR_ANN_MAX_DIM} dimensions (or a model that supports "
                f"shortening its output) for indexed recall.",
                RuntimeWarning,
                stacklevel=2,
            )
        # Without numpy (and without an embedder) there is no vector to store or
        # to rank by: recall silently became a substring match. Say so now.
        self._substring_only = embedder is None and _numpy() is None
        self._substring_warned = False
        if self._substring_only:
            warnings.warn(_SUBSTRING_ONLY, RuntimeWarning, stacklevel=2)
        self._encoding_cache = encoding_cache
        self._encodings: OrderedDict[tuple[str, str], str | None] = OrderedDict()
        self._pool: Pool | None = None
        # Serialises first use so two concurrent calls cannot each build a pool
        # (and each run schema creation).
        self._pool_lock = asyncio.Lock()

    async def _embed(self, content: str) -> str | None:
        """The pgvector literal for ``content`` — real embedding if configured,
        else the HRR ``[cos φ, sin φ]`` fallback (``None`` without numpy)."""
        if self._embedder is not None:
            result = await self._embedder.embed(content)
            return "[" + ",".join(f"{x:.6f}" for x in result.embedding) + "]"
        return _embed_literal(content, self._dim)

    async def _vector(self, tenant: str, content: str) -> str | None:
        """:meth:`_embed`, remembered per ``(tenant, content)``.

        The key carries the tenant on purpose: an encoding computed for one
        tenant is never handed to another, so the cache is per tenant even
        though the store is shared.
        """
        if self._encoding_cache == 0:
            return await self._embed(content)
        key = (tenant, content)
        if key in self._encodings:
            self._encodings.move_to_end(key)
            return self._encodings[key]
        vector = await self._embed(content)
        self._encodings[key] = vector
        while len(self._encodings) > self._encoding_cache:
            self._encodings.popitem(last=False)
        return vector

    def _warn_substring(self) -> None:
        if not self._substring_warned:
            self._substring_warned = True
            logger.warning(_SUBSTRING_ONLY)

    async def _get_pool(self) -> Pool:
        """The connection pool, built (and its schema created) on first use.

        ``self._pool`` is published **only after** ``_ensure_schema`` succeeds.
        Assigning it first would make a schema failure surface on the first call
        and then vanish — every later call would find a pool, skip schema
        creation, and run against a half-built table. A half-initialised store
        must not present as a working one.
        """

        async def build() -> Pool:
            import asyncpg  # noqa: PLC0415

            pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=8)
            try:
                await self._ensure_schema(pool)
            except BaseException:
                await pool.close()
                raise
            return pool

        # The lock still guards concurrent first use within one loop; the
        # loop key handles the case the lock cannot see, which is a second
        # loop inheriting a pool whose sockets belong to the first.
        async with self._pool_lock:
            return await loop_bound_async(self, "_pool", build)

    async def _ensure_schema(self, pool: Pool) -> None:
        async with pool.acquire() as conn:
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
            await self._check_existing_width(conn)
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
            if self._ann:
                await conn.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{self._table}_ann "
                    f"ON {self._table} USING hnsw (embedding vector_cosine_ops)"
                )

    async def _check_existing_width(self, conn: Any) -> None:
        """Refuse to run against a table whose ``embedding`` column is a different width.

        ``CREATE TABLE IF NOT EXISTS`` silently keeps a pre-existing table, so a
        store configured for a different ``dim`` would only fail later, per
        INSERT, with asyncpg's ``expected N dimensions, not M``. Detect it here
        and name both widths and the remedy.
        """
        # For pgvector, ``atttypmod`` is the declared dimension verbatim
        # (0/-1 when the column is unmodified, which cannot happen for us).
        actual = await conn.fetchval(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = $1::regclass AND attname = 'embedding' AND NOT attisdropped",
            self._table,
        )
        if actual is None or actual <= 0 or actual == self._vdim:
            return
        how = (
            f"pass dim={actual // 2}"
            if self._embedder is None and actual % 2 == 0
            else "use the embedder it was written with"
        )
        raise RuntimeError(
            f"PgMemory: table {self._table!r} already has embedding vector({actual}), "
            f"but this store is configured for vector({self._vdim}) — every write "
            f"would be rejected by Postgres. To read the existing rows, {how}; "
            f"to adopt the new width, re-embed the table (ALTER TABLE "
            f"{self._table} ALTER COLUMN embedding TYPE vector({self._vdim}) after "
            f"clearing it) or write to a different table=."
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
        emb = await self._vector(tenant, content)
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
                qvec = await self._vector(tenant, query)
                if qvec is None:
                    self._warn_substring()
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

    async def search_many(
        self,
        namespaces: Sequence[tuple[str, ...]],
        query: str | None = None,
        limit: int = 10,
    ) -> list[list[StoreItem]]:
        """:meth:`search` over several namespaces of ONE tenant, in one statement.

        One transaction (the tenant pin and one ``SELECT``), the query encoded
        once, and each namespace ranked on its own exactly as :meth:`search`
        would rank it (``LATERAL`` per namespace), so the result is
        ``[search(ns, query, limit) for ns in namespaces]`` at the cost of one
        round trip.

        Raises:
            ValueError: The namespaces belong to more than one tenant
                (``namespace[0]``). A call never spans tenants.
        """
        if not namespaces:
            return []
        tenants = {self._tenant_of(ns) for ns in namespaces}
        if len(tenants) > 1:
            raise ValueError(
                f"search_many spans tenants {sorted(tenants)}: every namespace of one "
                "call must share namespace[0], the tenant"
            )
        tenant = tenants.pop()
        names = [_NS_SEP.join(ns) for ns in namespaces]
        cols = "key, value, metadata, created_at, updated_at, version"
        where = f"FROM {self._table} WHERE tenant=$1 AND ns=n.ns"
        args: list[Any] = [tenant, names]
        if not query:
            inner = f"SELECT {cols}, updated_at AS _rank {where} ORDER BY updated_at DESC"
            order = "n.idx, m._rank DESC"
        else:
            qvec = await self._vector(tenant, query)
            if qvec is None:
                self._warn_substring()
                inner = (
                    f"SELECT {cols}, updated_at AS _rank {where} "
                    "AND content ILIKE '%'||$3||'%' ORDER BY updated_at DESC"
                )
                order = "n.idx, m._rank DESC"
                args.append(query)
            else:
                inner = (
                    f"SELECT {cols}, embedding <=> $3::vector AS _rank {where} "
                    "AND embedding IS NOT NULL ORDER BY embedding <=> $3::vector"
                )
                order = "n.idx, m._rank"
                args.append(qvec)
        args.append(limit)
        sql = (
            f"SELECT n.idx, m.* FROM unnest($2::text[]) WITH ORDINALITY AS n(ns, idx) "
            f"CROSS JOIN LATERAL ({inner} LIMIT ${len(args)}) m ORDER BY {order}"
        )
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            await self._scoped(conn, tenant)
            rows = await conn.fetch(sql, *args)
        out: list[list[StoreItem]] = [[] for _ in namespaces]
        for r in rows:
            i = int(r["idx"]) - 1
            out[i].append(self._row_to_item(tuple(namespaces[i]), r))
        return out

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
