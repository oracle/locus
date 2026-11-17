# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Time travel on a thread: what its checkpoints are, forking one, running from one.

Every checkpointer keeps a thread's checkpoints — one per turn by default,
one per iteration with ``keep_iteration_checkpoints``, one per pause and
per edit — and :meth:`~locus.memory.BaseCheckpointer.list_checkpoints`
names them. This module is the uniform layer above that:

* :class:`CheckpointInfo` is what :meth:`~locus.memory.BaseCheckpointer.list`
  returns for each checkpoint: its id, when it was written, the iteration
  the state had reached, the serialised state's size, the checkpoint it
  descends from, and the thread it was forked from (for a fork's first
  checkpoint). A backend fills what it can read without loading the state;
  the default implementation here loads each state and reads it off.
* :func:`default_fork` is :meth:`~locus.memory.BaseCheckpointer.fork`:
  the state at a checkpoint saved as the first checkpoint of a new thread,
  with the thread's tool-effect ledger (:mod:`locus.memory.ledger`) copied
  up to that step and no further.

Lineage. The ``parent_checkpoint_id`` of a checkpoint is recorded in its
metadata when the writer knew it was not the thread's previous checkpoint:
a fork's first checkpoint (its parent is in the source thread), an edit
(``update_state``) of an older checkpoint, and the first lasting save of a
run started ``from_checkpoint=`` (the run's later saves follow on from it).
Everywhere else a thread's history is a line, and the parent is the
checkpoint before it (:func:`link_parents`); only a thread's first
checkpoint has none.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4


if TYPE_CHECKING:
    from locus.memory.checkpointer import BaseCheckpointer


__all__ = [
    "FORKED_FROM_KEY",
    "FORK_SOURCE",
    "PARENT_KEY",
    "SOURCE_KEY",
    "CheckpointInfo",
    "checkpoint_iteration",
    "checkpoint_size",
    "default_fork",
    "default_list",
    "fork_metadata",
    "info_from",
    "link_parents",
    "new_fork_id",
    "order_by_write_time",
    "parse_time",
    "split_stored_metadata",
]

#: Checkpoint-metadata key naming the checkpoint this one descends from.
PARENT_KEY = "parent_checkpoint_id"
#: Checkpoint-metadata key naming the thread a fork's first checkpoint came from.
FORKED_FROM_KEY = "forked_from"
#: Checkpoint-metadata key saying what wrote the checkpoint (``"fork"``, ``"update_state"``).
SOURCE_KEY = "source"
#: The ``source`` a fork's first checkpoint carries.
FORK_SOURCE = "fork"


@dataclass(frozen=True)
class CheckpointInfo:
    """One checkpoint of a thread, as :meth:`BaseCheckpointer.list` describes it.

    Attributes:
        checkpoint_id: The id ``load(thread_id, checkpoint_id)`` takes.
        thread_id: The thread the checkpoint belongs to.
        created_at: When it was written; ``None`` when the backend does not keep it.
        iteration: The state's iteration counter when it was saved — the step
            of the turn it was written in (0 before the first model call).
        size: Bytes of the checkpoint as its backend stores it — the
            serialised state as compact JSON for most, the stored row's text
            where the database keeps its own encoding (PostgreSQL ``jsonb``)
            — so a thread's storage can be read off its history. Sizes
            compare within one backend, not across backends.
        parent_checkpoint_id: The checkpoint this one descends from: in the
            same thread unless ``forked_from`` is set; ``None`` for a thread's
            first checkpoint.
        forked_from: The thread the parent lives in, for a fork's first
            checkpoint; ``None`` otherwise.
        metadata: The metadata the checkpoint was saved with.
    """

    checkpoint_id: str
    thread_id: str
    created_at: datetime | None = None
    iteration: int | None = None
    size: int | None = None
    parent_checkpoint_id: str | None = None
    forked_from: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str | None:
        """What wrote the checkpoint, when its metadata says (``"fork"``, ``"update_state"``)."""
        source = self.metadata.get(SOURCE_KEY)
        return str(source) if source is not None else None

    def to_dict(self) -> dict[str, Any]:
        """The info as plain data (``created_at`` as an ISO-8601 string)."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "thread_id": self.thread_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "iteration": self.iteration,
            "size": self.size,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "forked_from": self.forked_from,
            "metadata": dict(self.metadata),
        }


def checkpoint_size(data: Mapping[str, Any]) -> int:
    """Bytes of a checkpoint's state as compact JSON — the ``size`` most backends report."""
    return len(json.dumps(data, separators=(",", ":"), default=str).encode("utf-8"))


def checkpoint_iteration(data: Mapping[str, Any]) -> int | None:
    """The iteration a serialised state had reached, or ``None`` when it records none."""
    iteration = data.get("iteration")
    return int(iteration) if isinstance(iteration, (int, float, str)) else None


def parse_time(value: Any) -> datetime | None:
    """``value`` as a datetime: one already, an ISO-8601 string, or ``None``.

    A naive value is UTC wall-clock time, which is what every backend
    stores, and comes back aware, so infos from different backends compare
    (a naive and an aware datetime cannot be ordered).
    """
    moment: datetime | None = None
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str) and value:
        try:
            moment = datetime.fromisoformat(value)
        except ValueError:
            return None
    if moment is None:
        return None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def new_fork_id(thread_id: str) -> str:
    """A fresh thread id for a fork of ``thread_id``."""
    return f"{thread_id}-fork-{uuid4().hex[:8]}"


def fork_metadata(thread_id: str, checkpoint_id: str | None) -> dict[str, Any]:
    """The metadata a fork's first checkpoint is saved with."""
    return {SOURCE_KEY: FORK_SOURCE, FORKED_FROM_KEY: thread_id, PARENT_KEY: checkpoint_id}


def info_from(
    checkpoint_id: str,
    thread_id: str,
    *,
    created_at: Any = None,
    metadata: Mapping[str, Any] | None = None,
    iteration: Any = None,
    size: Any = None,
) -> CheckpointInfo:
    """A :class:`CheckpointInfo` from what a backend stored, reading lineage off the metadata.

    Values a backend hands over as they came out of storage (an ISO string
    for the time, a number that may be a string) are normalised here, so
    each backend's ``list`` is a projection and nothing more.
    """
    meta = dict(metadata or {})
    parent = meta.get(PARENT_KEY)
    forked_from = meta.get(FORKED_FROM_KEY)
    return CheckpointInfo(
        checkpoint_id=checkpoint_id,
        thread_id=thread_id,
        created_at=parse_time(created_at),
        iteration=int(iteration) if isinstance(iteration, (int, float, str)) else None,
        size=int(size) if isinstance(size, (int, float, str)) else None,
        parent_checkpoint_id=str(parent) if parent is not None else None,
        forked_from=str(forked_from) if forked_from is not None else None,
        metadata=meta,
    )


def link_parents(infos: list[CheckpointInfo], older: str | None = None) -> list[CheckpointInfo]:
    """Fill in the parent of each checkpoint that did not record one.

    ``infos`` is a thread's history, newest first — or the newest part of
    it, with ``older`` the checkpoint just before that window. A checkpoint
    whose metadata names its parent keeps it; any other descends from the
    one before it in the thread (``older`` for the last of ``infos``), and
    only a thread's first checkpoint from nothing.
    """
    linked: list[CheckpointInfo] = []
    for position, info in enumerate(infos):
        if info.parent_checkpoint_id is not None or info.forked_from is not None:
            linked.append(info)
            continue
        before = infos[position + 1].checkpoint_id if position + 1 < len(infos) else older
        linked.append(replace(info, parent_checkpoint_id=before))
    return linked


def _utc(when: datetime) -> datetime:
    return when if when.tzinfo is not None else when.replace(tzinfo=UTC)


async def order_by_write_time(
    objects: Sequence[tuple[str, Any]],
    read_meta: Callable[[str], Awaitable[Mapping[str, Any] | None]],
    time_key: str,
) -> list[tuple[str, Any, dict[str, Any] | None]]:
    """An object store's checkpoints newest first, exact where the store's clock is coarse.

    ``objects`` are ``(checkpoint_id, modified)`` from a bucket listing. The
    store's modification time orders them, but S3 keeps it to the second
    (OCI Object Storage to the millisecond) and a turn writes several
    checkpoints in that; each checkpoint's metadata object records its
    write time to the microsecond under ``time_key``, so for the checkpoints
    that share a second of the store's clock — those alone — ``read_meta``
    fetches it and it orders them. Each result is ``(checkpoint_id,
    modified, meta)``: ``meta`` is the metadata object where it was read,
    ``None`` where the listing was enough.
    """
    epoch = datetime.fromtimestamp(0, UTC)

    def second(modified: Any) -> datetime:
        when = _utc(modified) if isinstance(modified, datetime) else epoch
        return when.replace(microsecond=0)

    groups: dict[datetime, list[str]] = {}
    for checkpoint_id, modified in objects:
        groups.setdefault(second(modified), []).append(checkpoint_id)
    tied = [cid for ids in groups.values() if len(ids) > 1 for cid in ids]
    metas = dict(zip(tied, await asyncio.gather(*(read_meta(cid) for cid in tied)), strict=True))

    def key(item: tuple[str, Any]) -> tuple[datetime, datetime]:
        checkpoint_id, modified = item
        exact = parse_time((metas.get(checkpoint_id) or {}).get(time_key))
        return second(modified), _utc(exact) if exact else second(modified)

    return [
        (checkpoint_id, modified, dict(meta) if (meta := metas.get(checkpoint_id)) else None)
        for checkpoint_id, modified in sorted(objects, key=key, reverse=True)
    ]


def split_stored_metadata(raw: Mapping[str, Any] | None) -> tuple[dict[str, Any], datetime | None]:
    """``(metadata, created_at)`` from what a backend's ``get_metadata`` returns.

    Backends differ: some return the metadata the checkpoint was saved with,
    others an envelope with ``metadata`` inside and the write time beside it.
    """
    if raw is None:
        return {}, None
    inner = raw.get("metadata")
    if isinstance(inner, Mapping):
        when = raw.get("created_at") or raw.get("timestamp") or raw.get("updated_at")
        return dict(inner), parse_time(when)
    return dict(raw), None


async def default_list(
    checkpointer: BaseCheckpointer, thread_id: str, limit: int
) -> list[CheckpointInfo]:
    """:meth:`BaseCheckpointer.list` for a backend with no projection of its own.

    Loads each checkpoint's state to read its iteration and size, and asks
    ``get_metadata`` for the rest when the backend has it. Correct on any
    backend; a backend that can read the fields without the state should
    override ``list``.
    """
    infos: list[CheckpointInfo] = []
    has_metadata = checkpointer.capabilities.metadata_query
    # One id past the window names the parent of the oldest checkpoint in it.
    ids = await checkpointer.list_checkpoints(thread_id, limit=limit + 1)
    older = ids[limit] if len(ids) > limit else None
    for checkpoint_id in ids[:limit]:
        state = await checkpointer.load(thread_id, checkpoint_id)
        if state is None:
            continue
        metadata: dict[str, Any] = {}
        created_at: datetime | None = None
        if has_metadata:
            try:
                metadata, created_at = split_stored_metadata(
                    await checkpointer.get_metadata(thread_id, checkpoint_id)
                )
            except NotImplementedError:
                has_metadata = False
        infos.append(
            info_from(
                checkpoint_id,
                thread_id,
                created_at=created_at,
                metadata=metadata,
                iteration=state.iteration,
                size=checkpoint_size(state.to_checkpoint()),
            )
        )
    return link_parents(infos, older)


async def default_fork(
    checkpointer: BaseCheckpointer,
    thread_id: str,
    checkpoint_id: str | None,
    new_thread_id: str | None,
) -> str:
    """:meth:`BaseCheckpointer.fork`: a new thread whose first checkpoint is the given state.

    The state is saved under the new thread with :func:`fork_metadata`, and
    the source thread's tool-effect ledger is copied up to the forked
    state's iteration (:meth:`~locus.memory.ledger.ToolEffectLedger.fork`),
    so a run on the fork replays nothing the source did past that point.

    Raises:
        LookupError: No such thread or checkpoint.
        ValueError: ``new_thread_id`` is the source thread.
    """
    from locus.memory.ledger import ToolEffectLedger  # noqa: PLC0415

    state = await checkpointer.load(thread_id, checkpoint_id)
    if state is None:
        where = f"checkpoint {checkpoint_id!r} of " if checkpoint_id else ""
        raise LookupError(f"no {where}thread {thread_id!r}")
    source_id = checkpoint_id
    if source_id is None:
        latest = await checkpointer.list_checkpoints(thread_id, limit=1)
        source_id = latest[0] if latest else None
    forked = new_thread_id or new_fork_id(thread_id)
    if forked == thread_id:
        raise ValueError("a fork needs a thread id different from its parent")
    await checkpointer.save(state, forked, metadata=fork_metadata(thread_id, source_id))
    await ToolEffectLedger(checkpointer).fork(thread_id, forked, up_to_step=state.iteration)
    return forked
