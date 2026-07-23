"""Store backends for Tulip long-term memory.

A :class:`~tulip.memory.store.BaseStore` is the cross-thread persistent
key/value store used for long-term memory (see ``store.py``). Where the
``backends/`` package holds **checkpointer** drivers (per-thread agent
state), this package holds **store** drivers — namespace-keyed value
storage that survives across threads, optionally with vector search.

No external store driver ships in the core package today; the built-in
:class:`~tulip.memory.store.InMemoryStore` covers testing and small
deployments. Implement a custom store by subclassing ``BaseStore``.
"""

__all__: list[str] = []

from tulip.memory.store_backends.holographic import HolographicStore  # noqa: E402


__all__ = ["HolographicStore"]
