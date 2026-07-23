"""Store backends for Tulip long-term memory.

A :class:`~tulip.memory.store.BaseStore` is the cross-thread persistent
key/value store used for long-term memory (see ``store.py``). Where the
``backends/`` package holds **checkpointer** drivers (per-thread agent
state), this package holds **store** drivers — namespace-keyed value
storage that survives across threads, optionally with vector search.

Two long-term memory stores ship here: :class:`HolographicStore` (zero-infra
SQLite + FTS5 + HRR — the free/personal default) and :class:`PgMemory`
(Postgres + pgvector with per-tenant Row-Level Security — the enterprise
backend). Both implement ``BaseStore``; implement a custom store by subclassing
it. The built-in :class:`~tulip.memory.store.InMemoryStore` covers small tests.
"""

__all__: list[str] = []

from tulip.memory.store_backends.holographic import HolographicStore  # noqa: E402
from tulip.memory.store_backends.postgresql import PgMemory  # noqa: E402


__all__ = ["HolographicStore", "PgMemory"]
