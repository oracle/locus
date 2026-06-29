# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Coverage tests for ``tulip.memory.delta.DeltaCheckpointer``.

Targets the fall-back-to-full-checkpoint arms (parent checkpoint absent
or unreconstructable), the cache-miss load path, the empty
``get_storage_stats`` shape, and the graceful handling of a delta
checkpoint whose parent is missing or ``None``.
"""

from __future__ import annotations

import json
import zlib
from typing import Any

from tulip.core.state import AgentState
from tulip.memory.delta import (
    CheckpointMetadata,
    DeltaCheckpoint,
    DeltaCheckpointer,
    DeltaStorage,
)


def _full_checkpoint_bytes() -> bytes:
    state = AgentState(run_id="x", messages=[])
    return zlib.compress(json.dumps(state.to_checkpoint()).encode("utf-8"))


def _delta_bytes() -> bytes:
    delta = {"__added__": {}, "__removed__": [], "__changed__": {}}
    return zlib.compress(json.dumps(delta).encode("utf-8"))


# ---------------------------------------------------------------------------
# Storage doubles
# ---------------------------------------------------------------------------


class _ListButNoRetrieveStorage(DeltaStorage):
    """list_checkpoints reports a parent, but retrieve never finds it."""

    def __init__(self) -> None:
        self.stored: DeltaCheckpoint | None = None

    async def store(self, thread_id: str, checkpoint_id: str, checkpoint: DeltaCheckpoint) -> None:
        self.stored = checkpoint

    async def retrieve(self, thread_id: str, checkpoint_id: str) -> DeltaCheckpoint | None:
        return None

    async def list_checkpoints(self, thread_id: str, limit: int = 10) -> list[CheckpointMetadata]:
        return [CheckpointMetadata(checkpoint_id="ghost", thread_id=thread_id, chain_depth=0)]

    async def delete(self, thread_id: str, checkpoint_id: str | None = None) -> bool:
        return True


class _GhostParentStorage(DeltaStorage):
    """Parent is retrievable but is itself a delta whose parent is missing."""

    def __init__(self) -> None:
        self._parent = DeltaCheckpoint(
            metadata=CheckpointMetadata(
                checkpoint_id="parent",
                thread_id="t",
                parent_id="missing-grandparent",
                is_full=False,
                chain_depth=0,
            ),
            data=_delta_bytes(),
            is_delta=True,
        )
        self.stored: DeltaCheckpoint | None = None

    async def store(self, thread_id: str, checkpoint_id: str, checkpoint: DeltaCheckpoint) -> None:
        self.stored = checkpoint

    async def retrieve(self, thread_id: str, checkpoint_id: str) -> DeltaCheckpoint | None:
        return self._parent if checkpoint_id == "parent" else None

    async def list_checkpoints(self, thread_id: str, limit: int = 10) -> list[CheckpointMetadata]:
        return [self._parent.metadata]

    async def delete(self, thread_id: str, checkpoint_id: str | None = None) -> bool:
        return True


class _DeltaWithoutParentStorage(DeltaStorage):
    """Returns a delta checkpoint whose parent_id is None (degenerate case)."""

    def __init__(self) -> None:
        self._cp = DeltaCheckpoint(
            metadata=CheckpointMetadata(
                checkpoint_id="d1",
                thread_id="t",
                parent_id=None,
                is_full=False,
                chain_depth=1,
            ),
            data=_full_checkpoint_bytes(),
            is_delta=True,
        )

    async def store(self, thread_id: str, checkpoint_id: str, checkpoint: DeltaCheckpoint) -> None:
        return None

    async def retrieve(self, thread_id: str, checkpoint_id: str) -> DeltaCheckpoint | None:
        return self._cp if checkpoint_id == "d1" else None

    async def list_checkpoints(self, thread_id: str, limit: int = 10) -> list[CheckpointMetadata]:
        return [self._cp.metadata]

    async def delete(self, thread_id: str, checkpoint_id: str | None = None) -> bool:
        return True


# ---------------------------------------------------------------------------
# save() fall-back arms
# ---------------------------------------------------------------------------


async def test_save_creates_full_when_parent_not_retrievable() -> None:
    storage = _ListButNoRetrieveStorage()
    dc = DeltaCheckpointer(storage=storage, max_chain_depth=5)
    await dc.save(AgentState(run_id="x", messages=[]), "t", "new")
    assert storage.stored is not None
    assert storage.stored.metadata.is_full is True


async def test_save_creates_full_when_parent_state_unreconstructable() -> None:
    storage = _GhostParentStorage()
    dc = DeltaCheckpointer(storage=storage, max_chain_depth=5)
    await dc.save(AgentState(run_id="x", messages=[]), "t", "new")
    assert storage.stored is not None
    assert storage.stored.metadata.is_full is True


# ---------------------------------------------------------------------------
# load() cache-miss path
# ---------------------------------------------------------------------------


async def test_load_reconstructs_after_cache_cleared() -> None:
    dc = DeltaCheckpointer()
    await dc.save(AgentState(run_id="cached", messages=[]), "t", "cp1")
    dc._state_cache.clear()  # force the reconstruct-from-storage path
    loaded = await dc.load("t", "cp1")
    assert loaded is not None
    assert loaded.run_id == "cached"


async def test_load_delta_with_none_parent_returns_data() -> None:
    dc = DeltaCheckpointer(storage=_DeltaWithoutParentStorage())
    loaded = await dc.load("t", "d1")
    assert loaded is not None
    assert loaded.run_id == "x"


# ---------------------------------------------------------------------------
# get_storage_stats empty thread
# ---------------------------------------------------------------------------


async def test_get_storage_stats_empty_thread() -> None:
    dc = DeltaCheckpointer()
    stats: dict[str, Any] = await dc.get_storage_stats("never-saved")
    assert stats["total_checkpoints"] == 0
    assert stats["compression_ratio"] == 1.0
    assert stats["full_checkpoints"] == 0
