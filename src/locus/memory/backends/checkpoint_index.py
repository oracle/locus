# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The checkpoint index a storage backend can keep atomically for the adapter.

:class:`~locus.memory.backends.adapters.StorageBackendAdapter` records which
checkpoints a thread has in one index per thread: a list of entries, each
the checkpoint's id, write time, metadata, iteration and size. Through the
plain ``save`` / ``load`` interface that index is a read-modify-write, which
only a lock inside one process can serialise; two processes sharing a store
interleave it and lose entries (issue #301).

A backend that implements :class:`CheckpointIndexStore` keeps the index in
an operation its store makes atomic — a Redis hash field, a single SQL
``UPDATE`` on the index row, a scripted document update — and the adapter
uses it instead of the locked read-modify-write. The store decides where
the index lives; the adapter only asks for its entries.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable


__all__ = ["INDEX_SUFFIX", "CheckpointIndexStore", "index_key", "newest_first"]

#: The key suffix of a thread's checkpoint index, as the adapter has always stored it.
INDEX_SUFFIX = ":_checkpoints"


def index_key(thread_id: str) -> str:
    """The storage key of ``thread_id``'s checkpoint index (``<thread_id>:_checkpoints``)."""
    return f"{thread_id}{INDEX_SUFFIX}"


def newest_first(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Index entries ordered newest first by their ``timestamp``, as the adapter lists them.

    The sort is stable, so entries with one timestamp keep the order the
    store returned them in.
    """
    return sorted(entries, key=lambda entry: str(entry.get("timestamp") or ""), reverse=True)


@runtime_checkable
class CheckpointIndexStore(Protocol):
    """A storage backend that updates a thread's checkpoint index atomically.

    Each method is one atomic operation on the store, safe against another
    process doing the same to the same thread. An entry is a dict with at
    least ``checkpoint_id`` and ``timestamp`` (an ISO-8601 string); the
    adapter adds ``metadata``, ``iteration`` and ``size``. The order
    ``index_list`` returns entries in does not matter: the adapter sorts
    them newest first.
    """

    async def index_add(self, thread_id: str, entry: dict[str, Any]) -> None:
        """Add ``entry`` to the thread's index, replacing one with the same ``checkpoint_id``."""
        ...

    async def index_remove(self, thread_id: str, checkpoint_id: str) -> bool:
        """Remove the entry for ``checkpoint_id``; ``True`` when there was one."""
        ...

    async def index_list(self, thread_id: str) -> list[dict[str, Any]]:
        """Every entry of the thread's index, in any order; ``[]`` for an unknown thread."""
        ...

    async def index_clear(self, thread_id: str) -> bool:
        """Drop the thread's whole index; ``True`` when there was one."""
        ...
