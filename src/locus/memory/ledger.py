# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The tool-effect ledger: every executed tool call, recorded as it completes.

A checkpoint holds a tool call's result only once the iteration that made it
is saved. Between the call returning and that save — the whole turn, with
``durability="exit"`` — a process that dies leaves no record that the refund
went out, and a resumed run would issue it again. The ledger closes that
window: the executor writes one small record per executed call, keyed by
``(thread id, step, tool name, canonical arguments)``, the moment the call
returns, and consults the ledger before running a call when a run is
replaying. A hit returns the recorded result; the tool body does not run.

The ledger lives in the thread's own checkpointer, so every backend has it
without code of its own: the records of thread ``t1`` are checkpoints of the
sidecar thread ``t1~ledger`` (:func:`ledger_thread_id`), one per executed
call under its key, plus a ``manifest`` naming the invocation (the run id),
its prompt and metadata. The turn's final checkpoint supersedes the records,
so the run deletes the sidecar once that save has landed; a sidecar found
later is the trace of a turn that never finished.

Replay is on for the runs that continue a turn — ``Agent.continue_turn``,
``Agent.resume(invocation_id=...)``, and ``Agent.run(invocation_id=...)``
with the id of the turn to retry — and off for a fresh turn, which can have
no records yet. A hit is served only when the record's invocation matches
the run's: a new turn on the thread never reads a crashed turn's results.

What the ledger guarantees, by tool annotation:

* A tool that is **not** ``idempotent`` runs **at least once**: once its
  record is written, a replay returns the record and the body does not run
  again. A process killed between the body returning and the record landing
  leaves no record, and the replayed call runs the body again.
* An ``idempotent=True`` tool is served from the ledger the same way, unless
  ``AgentConfig.rerun_idempotent_on_replay`` is set, in which case a replay
  runs the body again (which the annotation declares safe) and the record is
  refreshed with the new result.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from locus.core.messages import ToolResult


if TYPE_CHECKING:
    from locus.core.messages import ToolCall


__all__ = [
    "LEDGER_KEY",
    "LEDGER_SUFFIX",
    "MANIFEST_ID",
    "LedgerEntry",
    "LedgerManifest",
    "RunLedger",
    "ToolEffectLedger",
    "canonical_arguments",
    "effect_key",
    "ledger_thread_id",
]

logger = logging.getLogger(__name__)

#: Appended to a thread id to name the thread that holds its ledger.
LEDGER_SUFFIX = "~ledger"

#: The checkpoint id of the sidecar's manifest.
MANIFEST_ID = "manifest"

#: The metadata key a sidecar checkpoint carries its record under.
LEDGER_KEY = "_locus_tool_ledger"

#: How many records :meth:`ToolEffectLedger.entries` reads at most.
_MAX_ENTRIES = 10_000


def ledger_thread_id(thread_id: str) -> str:
    """The sidecar thread that holds ``thread_id``'s ledger."""
    return f"{thread_id}{LEDGER_SUFFIX}"


def canonical_arguments(arguments: Mapping[str, Any]) -> str:
    """``arguments`` as one string that is the same for equal arguments.

    Keys are sorted and separators fixed, so argument order and whitespace do
    not change the key; a value ``json`` cannot encode is written as text,
    which keeps the key defined for every argument the model can send.
    """
    return json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)


def effect_key(thread_id: str, step: int, tool: str, arguments: Mapping[str, Any]) -> str:
    """The ledger key of one call: ``(thread, step, tool, canonical arguments)``, hashed.

    A hash so the key is a safe checkpoint id on every backend (a file name,
    an object key, a row), whatever the arguments contain.
    """
    raw = "\0".join((thread_id, str(step), tool, canonical_arguments(arguments)))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class LedgerEntry:
    """One executed tool call, as the ledger records it."""

    thread_id: str
    invocation_id: str
    step: int
    tool: str
    #: The call's arguments in canonical form (:func:`canonical_arguments`).
    arguments: str
    #: The :class:`~locus.core.messages.ToolResult` the call produced, as JSON data.
    result: dict[str, Any]
    #: The id of the call that produced the result; a replay answers a new id.
    tool_call_id: str = ""
    recorded_at: str = field(default_factory=_now)

    @property
    def key(self) -> str:
        return effect_key(self.thread_id, self.step, self.tool, json.loads(self.arguments))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LedgerEntry:
        """The entry a record holds; unknown keys are ignored."""
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in names})

    def tool_result(self, tool_call_id: str) -> ToolResult:
        """The recorded result, answering ``tool_call_id`` (the replayed call's id)."""
        result = ToolResult.model_validate(self.result)
        return result.model_copy(update={"tool_call_id": tool_call_id, "replayed": True})


@dataclass(frozen=True)
class LedgerManifest:
    """The turn a thread's ledger belongs to."""

    thread_id: str
    invocation_id: str
    prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    durability: str = "sync"
    started_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LedgerManifest:
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in names})


class ToolEffectLedger:
    """A thread's tool-effect ledger, kept in its checkpointer.

    ``checkpointer`` is anything with the :class:`~locus.memory.BaseCheckpointer`
    ``save`` / ``load`` / ``delete`` methods. Every method takes the *agent's*
    thread id and addresses the sidecar itself.
    """

    def __init__(self, checkpointer: Any) -> None:
        self._checkpointer = checkpointer

    @property
    def checkpointer(self) -> Any:
        return self._checkpointer

    async def record(self, entry: LedgerEntry) -> None:
        """Write ``entry`` under its key, replacing an earlier record of the same call."""
        await self._put(entry.thread_id, entry.key, entry.to_dict())

    async def lookup(
        self, thread_id: str, step: int, tool: str, arguments: Mapping[str, Any]
    ) -> LedgerEntry | None:
        """The record of this call, or ``None``."""
        data = await self._get(thread_id, effect_key(thread_id, step, tool, arguments))
        return LedgerEntry.from_dict(data) if data is not None else None

    async def write_manifest(self, manifest: LedgerManifest) -> None:
        await self._put(manifest.thread_id, MANIFEST_ID, manifest.to_dict())

    async def manifest(self, thread_id: str) -> LedgerManifest | None:
        """The turn the thread's ledger belongs to, or ``None`` when there is no ledger."""
        data = await self._get(thread_id, MANIFEST_ID)
        return LedgerManifest.from_dict(data) if data is not None else None

    async def entries(self, thread_id: str) -> list[LedgerEntry]:
        """Every record of the thread, oldest first. For inspection; replay uses :meth:`lookup`."""
        sidecar = ledger_thread_id(thread_id)
        found: list[LedgerEntry] = []
        for checkpoint_id in await self._checkpointer.list_checkpoints(sidecar, limit=_MAX_ENTRIES):
            if checkpoint_id == MANIFEST_ID:
                continue
            data = await self._get(thread_id, checkpoint_id)
            if data is not None and "tool" in data:
                found.append(LedgerEntry.from_dict(data))
        return sorted(found, key=lambda e: (e.recorded_at, e.step))

    async def fork(
        self, source_thread_id: str, dest_thread_id: str, *, up_to_step: int | None = None
    ) -> int:
        """Copy the source thread's ledger to a fork of it, up to ``up_to_step``.

        A fork starts from a checkpoint, and that checkpoint already holds
        the results of every call at or before its iteration — those records
        are copied, re-keyed under the fork's thread id, with the manifest.
        Records past ``up_to_step`` are the source thread's effects beyond
        the fork point: a run on the fork must never replay them, so they
        are not copied. ``None`` copies every record.

        Returns the records copied; 0 when the source has no ledger.
        """
        manifest = await self.manifest(source_thread_id)
        entries = await self.entries(source_thread_id)
        if manifest is None and not entries:
            return 0
        if manifest is not None:
            await self.write_manifest(replace(manifest, thread_id=dest_thread_id))
        copied = 0
        for entry in entries:
            if up_to_step is not None and entry.step > up_to_step:
                continue
            await self.record(replace(entry, thread_id=dest_thread_id))
            copied += 1
        return copied

    async def clear(self, thread_id: str) -> bool:
        """Delete the thread's ledger. ``False`` when there was none or the backend cannot delete."""
        try:
            deleted: bool = await self._checkpointer.delete(ledger_thread_id(thread_id))
        except NotImplementedError:
            logger.warning(
                "%s cannot delete, so the tool-effect ledger of thread %s stays behind",
                type(self._checkpointer).__name__,
                thread_id,
            )
            return False
        return deleted

    # -- storage: records are checkpoints of the sidecar thread ---------------

    async def _put(self, thread_id: str, checkpoint_id: str, data: dict[str, Any]) -> None:
        from locus.core.state import AgentState  # noqa: PLC0415

        await self._checkpointer.save(
            AgentState(metadata={LEDGER_KEY: data}), ledger_thread_id(thread_id), checkpoint_id
        )

    async def _get(self, thread_id: str, checkpoint_id: str) -> dict[str, Any] | None:
        state = await self._checkpointer.load(ledger_thread_id(thread_id), checkpoint_id)
        if state is None:
            return None
        data = state.metadata.get(LEDGER_KEY)
        return dict(data) if isinstance(data, dict) else None


class RunLedger:
    """One run's view of the ledger: records what it executes, replays when asked.

    Created by the runtime for a run with a checkpointer and a thread, and
    handed to the tool executor on the run's
    :class:`~locus.tools.executor.ToolContextFactory`. ``replay`` is on for a
    run that continues a turn and off for a fresh one. The manifest is
    written before the run's first record.
    """

    def __init__(
        self,
        ledger: ToolEffectLedger,
        *,
        thread_id: str,
        invocation_id: str,
        replay: bool,
        prompt: str = "",
        metadata: Mapping[str, Any] | None = None,
        durability: str = "sync",
        rerun_idempotent: bool = False,
    ) -> None:
        self.ledger = ledger
        self.thread_id = thread_id
        self.invocation_id = invocation_id
        self.replay_enabled = replay
        self.rerun_idempotent = rerun_idempotent
        self._manifest = LedgerManifest(
            thread_id=thread_id,
            invocation_id=invocation_id,
            prompt=prompt,
            metadata=dict(metadata or {}),
            durability=durability,
        )
        self._manifest_written = not prompt
        #: Replays served and records written by this run (for tests and logs).
        self.replays = 0
        self.records = 0

    def without_replay(self) -> RunLedger:
        """This run's ledger for a call that must run: a hook's retry."""
        clone = RunLedger.__new__(RunLedger)
        clone.__dict__.update(self.__dict__)
        clone.replay_enabled = False
        return clone

    async def replay(
        self, tool_call: ToolCall, *, step: int, idempotent: bool = False
    ) -> ToolResult | None:
        """The recorded result of this call at ``step``, or ``None`` to run it.

        Only a record this run's invocation wrote is served. An idempotent
        tool is run again instead when the run prefers (``rerun_idempotent``).
        """
        if not self.replay_enabled or (idempotent and self.rerun_idempotent):
            return None
        try:
            entry = await self.ledger.lookup(
                self.thread_id, step, tool_call.name, tool_call.arguments
            )
        except Exception:  # noqa: BLE001 — a ledger that cannot be read is a miss, not a failure
            logger.warning(
                "tool-effect ledger lookup failed for %s on thread %s; running the call",
                tool_call.name,
                self.thread_id,
                exc_info=True,
            )
            return None
        if entry is None or entry.invocation_id != self.invocation_id:
            return None
        self.replays += 1
        logger.debug(
            "replayed %s at step %d of thread %s from the tool-effect ledger",
            tool_call.name,
            step,
            self.thread_id,
        )
        return entry.tool_result(tool_call.id)

    async def record(self, tool_call: ToolCall, result: ToolResult, *, step: int) -> None:
        """Record ``result`` for ``tool_call`` at ``step``; the manifest first, once."""
        entry = LedgerEntry(
            thread_id=self.thread_id,
            invocation_id=self.invocation_id,
            step=step,
            tool=tool_call.name,
            arguments=canonical_arguments(tool_call.arguments),
            result=result.model_dump(mode="json", exclude={"replayed"}),
            tool_call_id=tool_call.id,
        )
        try:
            if not self._manifest_written:
                await self.ledger.write_manifest(self._manifest)
                self._manifest_written = True
            await self.ledger.record(entry)
        except Exception:  # noqa: BLE001 — the call already ran; losing its record must not fail it
            logger.warning(
                "tool-effect ledger write failed for %s on thread %s; a replay after a crash "
                "may run it again",
                tool_call.name,
                self.thread_id,
                exc_info=True,
            )
            return
        self.records += 1

    async def clear(self) -> None:
        """Delete the thread's ledger: the turn's checkpoint now holds every result."""
        try:
            await self.ledger.clear(self.thread_id)
        except Exception:  # noqa: BLE001 — clean-up only; the turn is saved
            logger.warning(
                "could not clear the tool-effect ledger of thread %s", self.thread_id, exc_info=True
            )

    def __repr__(self) -> str:
        return (
            f"RunLedger(thread_id={self.thread_id!r}, invocation_id={self.invocation_id!r}, "
            f"replay={self.replay_enabled}, records={self.records}, replays={self.replays})"
        )
