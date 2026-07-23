# SPDX-License-Identifier: Apache-2.0
#
# The holographic (HRR) encoding primitives in this module — encode_atom, bind,
# bundle, similarity, encode_text, and the phase byte (de)serialisation — are
# ported and adapted from NousResearch/hermes-agent
# (plugins/memory/holographic/holographic.py), MIT License, © 2025 Nous Research.
# The BaseStore integration, SQLite/FTS5 persistence, and namespacing are Tulip.
"""Holographic long-term memory store — the zero-infra default.

A :class:`~tulip.memory.store.BaseStore` backed by **local SQLite** (a file, or
in-memory) with **FTS5** lexical search and **Holographic Reduced
Representation** (HRR) associative recall. It needs nothing but the machine — no
server, no vector database, no embedding API — which makes it the free / local /
personal default of Tulip's open-core memory (the enterprise path is a
pgvector-backed store).

HRR encodes text as a high-dimensional **phase** vector; ``bundle`` superposes
tokens and ``similarity`` is a phase cosine — so recall is compositional and
fully offline. ``numpy`` is imported lazily; without it the store still works as
a lexical (FTS5) store, degrading rather than failing.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import struct
import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from tulip.memory.store import BaseStore, StoreCapabilities, StoreItem


if TYPE_CHECKING:
    import numpy as np

_TWO_PI = 2.0 * math.pi
#: Default HRR dimensionality. 1024 phases hold O(sqrt(dim)) superposed tokens
#: before recall degrades — plenty for per-fact/per-card encoding.
_DEFAULT_DIM = 1024
_NS_SEP = "\x1f"  # unit separator — joins a namespace tuple into a stable string


def _numpy() -> Any:
    """The numpy module, or ``None`` when it isn't installed (lexical-only mode)."""
    try:
        import numpy as np  # noqa: PLC0415

        return np
    except ImportError:  # pragma: no cover - exercised only on numpy-less installs
        return None


def encode_atom(word: str, dim: int = _DEFAULT_DIM) -> np.ndarray:
    """A deterministic phase vector for one token, via SHA-256 counter blocks.

    Uses hashlib (not a numpy RNG) so the encoding is byte-for-byte reproducible
    across platforms and processes — a fact stored today recalls the same way
    tomorrow. Ported from hermes-agent (MIT).
    """
    np = _numpy()
    values_per_block = 16  # each SHA-256 digest = 32 bytes = 16 uint16 values
    blocks_needed = math.ceil(dim / values_per_block)
    uint16_values: list[int] = []
    for i in range(blocks_needed):
        digest = hashlib.sha256(f"{word}:{i}".encode()).digest()
        uint16_values.extend(struct.unpack("<16H", digest))
    return np.array(uint16_values[:dim], dtype=np.float64) * (_TWO_PI / 65536.0)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Associate two concepts (element-wise phase addition). Ported (MIT)."""
    return (a + b) % _TWO_PI


def bundle(*vectors: np.ndarray) -> np.ndarray:
    """Superpose vectors via the circular mean of complex exponentials (MIT)."""
    np = _numpy()
    complex_sum = np.sum([np.exp(1j * v) for v in vectors], axis=0)
    return np.angle(complex_sum) % _TWO_PI


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Phase cosine similarity in ``[-1, 1]`` (1.0 identical, ~0 unrelated). MIT."""
    np = _numpy()
    return float(np.mean(np.cos(a - b)))


def encode_text(text: str, dim: int = _DEFAULT_DIM) -> np.ndarray:
    """Bag-of-words HRR: the bundle of each token's atom vector. Ported (MIT)."""
    tokens = [tok.strip(".,!?;:\"'()[]{}") for tok in text.lower().split()]
    tokens = [t for t in tokens if t]
    if not tokens:
        return encode_atom("__hrr_empty__", dim)
    return bundle(*[encode_atom(t, dim) for t in tokens])


def phases_to_bytes(phases: np.ndarray) -> bytes:
    """Serialise a phase vector to uint16 bytes for compact on-disk storage (MIT)."""
    np = _numpy()
    quantized = np.round(phases * (65536.0 / _TWO_PI)).astype(np.uint16)
    return bytes(quantized.tobytes())


def bytes_to_phases(data: bytes) -> np.ndarray:
    """Deserialise uint16 bytes back into a phase vector (MIT)."""
    np = _numpy()
    quantized = np.frombuffer(data, dtype=np.uint16)
    return quantized.astype(np.float64) * (_TWO_PI / 65536.0)


def _content_of(value: Any) -> str:
    """The searchable text of a stored value — its strings, or a JSON dump."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = [str(v) for v in value.values() if isinstance(v, (str, int, float))]
        return " ".join(parts) if parts else json.dumps(value, sort_keys=True, default=str)
    return json.dumps(value, sort_keys=True, default=str)


class HolographicStore(BaseStore):
    """Local SQLite + FTS5 + HRR store — zero external infra.

    Every value is kept as a row plus an FTS5 index entry and (when numpy is
    present) an HRR phase vector. ``search`` ranks by HRR associative similarity,
    fused with FTS5 lexical match; without numpy it is FTS5-only. A single
    connection guarded by a lock keeps it safe across the store's threads.
    """

    def __init__(self, path: str = ":memory:", *, dim: int = _DEFAULT_DIM) -> None:
        self._dim = dim
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS items ("
            "ns TEXT, key TEXT, value TEXT, metadata TEXT, created_at TEXT, "
            "updated_at TEXT, version INTEGER, content TEXT, hrr BLOB, "
            "PRIMARY KEY (ns, key))"
        )
        self._conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS items_fts "
            "USING fts5(ns UNINDEXED, key UNINDEXED, content)"
        )
        self._conn.commit()

    @property
    def capabilities(self) -> StoreCapabilities:
        return StoreCapabilities(
            search=True,
            semantic_search=_numpy() is not None,
            embedding_dimension=self._dim,
            list_namespaces=True,
        )

    async def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        ns = _NS_SEP.join(namespace)
        content = _content_of(value)
        now = datetime.now(UTC).isoformat()
        np = _numpy()
        hrr = phases_to_bytes(encode_text(content, self._dim)) if np is not None else None
        with self._lock:
            existing = self._conn.execute(
                "SELECT created_at, version FROM items WHERE ns=? AND key=?", (ns, key)
            ).fetchone()
            created_at = existing[0] if existing else now
            version = (existing[1] + 1) if existing else 1
            self._conn.execute(
                "INSERT OR REPLACE INTO items "
                "(ns, key, value, metadata, created_at, updated_at, version, content, hrr) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    ns,
                    key,
                    json.dumps(value),
                    json.dumps(metadata or {}),
                    created_at,
                    now,
                    version,
                    content,
                    hrr,
                ),
            )
            self._conn.execute("DELETE FROM items_fts WHERE ns=? AND key=?", (ns, key))
            self._conn.execute(
                "INSERT INTO items_fts (ns, key, content) VALUES (?,?,?)", (ns, key, content)
            )
            self._conn.commit()

    async def get(self, namespace: tuple[str, ...], key: str) -> Any | None:
        ns = _NS_SEP.join(namespace)
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM items WHERE ns=? AND key=?", (ns, key)
            ).fetchone()
        return json.loads(row[0]) if row else None

    async def delete(self, namespace: tuple[str, ...], key: str) -> bool:
        ns = _NS_SEP.join(namespace)
        with self._lock:
            cur = self._conn.execute("DELETE FROM items WHERE ns=? AND key=?", (ns, key))
            self._conn.execute("DELETE FROM items_fts WHERE ns=? AND key=?", (ns, key))
            self._conn.commit()
            return cur.rowcount > 0

    async def list_keys(self, namespace: tuple[str, ...], limit: int = 100) -> list[str]:
        ns = _NS_SEP.join(namespace)
        with self._lock:
            rows = self._conn.execute(
                "SELECT key FROM items WHERE ns=? ORDER BY updated_at DESC LIMIT ?", (ns, limit)
            ).fetchall()
        return [r[0] for r in rows]

    async def search(
        self, namespace: tuple[str, ...], query: str | None = None, limit: int = 10
    ) -> list[StoreItem]:
        ns = _NS_SEP.join(namespace)
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, value, metadata, created_at, updated_at, version, hrr "
                "FROM items WHERE ns=? ORDER BY updated_at DESC",
                (ns,),
            ).fetchall()
        if not query:
            return [self._row_to_item(namespace, r) for r in rows[:limit]]
        np = _numpy()
        if np is None:
            return await self._lexical_search(namespace, ns, query, limit)
        q = encode_text(query, self._dim)
        scored = [(similarity(q, bytes_to_phases(r[6])) if r[6] else 0.0, r) for r in rows]
        scored.sort(key=lambda s: s[0], reverse=True)
        return [self._row_to_item(namespace, r) for _, r in scored[:limit]]

    async def _lexical_search(
        self, namespace: tuple[str, ...], ns: str, query: str, limit: int
    ) -> list[StoreItem]:
        with self._lock:
            keys = [
                r[0]
                for r in self._conn.execute(
                    "SELECT key FROM items_fts WHERE ns=? AND items_fts MATCH ? LIMIT ?",
                    (ns, query, limit),
                ).fetchall()
            ]
            items = []
            for k in keys:
                r = self._conn.execute(
                    "SELECT key, value, metadata, created_at, updated_at, version, hrr "
                    "FROM items WHERE ns=? AND key=?",
                    (ns, k),
                ).fetchone()
                if r:
                    items.append(self._row_to_item(namespace, r))
        return items

    def _row_to_item(self, namespace: tuple[str, ...], r: tuple[Any, ...]) -> StoreItem:
        return StoreItem(
            namespace=namespace,
            key=r[0],
            value=json.loads(r[1]),
            metadata=json.loads(r[2]) if r[2] else {},
            created_at=datetime.fromisoformat(r[3]),
            updated_at=datetime.fromisoformat(r[4]),
            version=r[5],
        )

    async def close(self) -> None:
        with self._lock:
            self._conn.close()
