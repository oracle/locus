# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Durability modes: *when* a run writes its checkpoints.

A run with a checkpointer saves its state at three kinds of point: after each
tool-calling iteration, before it pauses on an interrupt, and when it ends.
The durability mode decides how the iteration saves are made; the pause and
the end are always written before control returns, in every mode:

``"sync"``
    Each iteration checkpoint is written, and awaited, before the next model
    call. A process killed at any point loses at most the iteration in flight.
    The default, and what every earlier release did.

``"async"``
    Each iteration checkpoint is scheduled on the event loop and runs while
    the next model call is in flight. The run awaits every pending write
    before its final save, so the thread's history is complete and in order
    when the run returns; a process killed mid-run can lose the writes still
    in flight.

``"exit"``
    No iteration checkpoints. The run writes once when it ends (and before a
    pause, as always). The cheapest mode; a crash mid-run leaves the thread at
    its previous checkpoint.

The tool-effect ledger (:mod:`locus.memory.ledger`) is what makes the two
cheaper modes safe to crash in: each executed tool call is recorded as it
completes, so a resumed run replays what already happened instead of running
it again. :class:`CheckpointWriter` is the object the runtime writes through;
one per run, created by ``Agent._begin_run`` and the graph runner.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Literal, get_args


if TYPE_CHECKING:
    from locus.agent.config import AgentConfig
    from locus.agent.run_context import RunContext
    from locus.memory.ledger import RunLedger


__all__ = [
    "DURABILITY_MODES",
    "CheckpointWriter",
    "DurabilityMode",
    "attach_durability",
    "check_durability",
    "ledger_enabled",
    "retry_ledger",
]

logger = logging.getLogger(__name__)

DurabilityMode = Literal["sync", "async", "exit"]

#: The accepted values of ``durability=``, in the order the docs list them.
DURABILITY_MODES: tuple[DurabilityMode, ...] = get_args(DurabilityMode)


def check_durability(mode: Any) -> DurabilityMode:
    """``mode`` as a :data:`DurabilityMode`, or ``ValueError`` naming the valid ones."""
    if mode not in DURABILITY_MODES:
        raise ValueError(f"durability must be one of {', '.join(DURABILITY_MODES)}; got {mode!r}")
    checked: DurabilityMode = mode
    return checked


class CheckpointWriter:
    """Writes one run's checkpoints according to its durability mode.

    ``checkpointer`` is anything with the :class:`~locus.memory.BaseCheckpointer`
    ``save`` signature. The writer keeps the set of scheduled (``"async"``)
    writes so :meth:`flush` can await them; a write that fails is logged, not
    raised, because the run's final save supersedes it.
    """

    def __init__(self, checkpointer: Any, mode: DurabilityMode = "sync") -> None:
        self._checkpointer = checkpointer
        self._mode = check_durability(mode)
        self._pending: set[asyncio.Task[Any]] = set()
        #: Iteration saves this writer made or scheduled.
        self.iteration_writes = 0

    @property
    def mode(self) -> DurabilityMode:
        return self._mode

    @property
    def writes_iterations(self) -> bool:
        """Whether this mode writes iteration checkpoints at all."""
        return self._mode != "exit"

    @property
    def pending(self) -> int:
        """Scheduled writes not yet complete."""
        return len(self._pending)

    async def save_iteration(
        self,
        state: Any,
        thread_id: str,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Write (``sync``) or schedule (``async``) an iteration checkpoint.

        Returns ``False`` when the mode skips iteration checkpoints, so the
        caller does no bookkeeping for a save that never happens.
        """
        if self._mode == "exit":
            return False
        self.iteration_writes += 1
        if self._mode == "sync":
            await self._checkpointer.save(state, thread_id, checkpoint_id, metadata)
            return True
        task = asyncio.create_task(
            self._write(state, thread_id, checkpoint_id, metadata),
            name=f"locus-checkpoint-{thread_id}",
        )
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        return True

    async def save_now(
        self,
        state: Any,
        thread_id: str,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Await every scheduled write, then write this checkpoint and return its id.

        For the saves every mode makes before control returns: the run's
        final state, and the state it pauses on. Flushing first keeps the
        thread's newest checkpoint newest on backends that order by write.
        """
        await self.flush()
        checkpoint_id_out: str = await self._checkpointer.save(
            state, thread_id, checkpoint_id, metadata
        )
        return checkpoint_id_out

    async def flush(self) -> None:
        """Await the scheduled writes. A failed write is logged; the run goes on."""
        if not self._pending:
            return
        pending = list(self._pending)
        self._pending.clear()
        for task, outcome in zip(
            pending, await asyncio.gather(*pending, return_exceptions=True), strict=True
        ):
            if isinstance(outcome, BaseException):
                logger.warning(
                    "scheduled checkpoint write %s failed; the run's final save supersedes it",
                    task.get_name(),
                    exc_info=outcome,
                )

    async def _write(
        self,
        state: Any,
        thread_id: str,
        checkpoint_id: str | None,
        metadata: dict[str, Any] | None,
    ) -> str:
        written: str = await self._checkpointer.save(state, thread_id, checkpoint_id, metadata)
        return written

    def __repr__(self) -> str:
        return f"CheckpointWriter(mode={self._mode!r}, pending={self.pending})"


def ledger_enabled(config: AgentConfig, mode: DurabilityMode) -> bool:
    """Whether a run in ``mode`` keeps a tool-effect ledger under ``config``.

    ``tool_ledger=None`` means on for the modes whose iteration checkpoints
    are deferred or absent (``async``, ``exit``) and off for ``sync``, where
    the checkpoint follows each tool batch at once.
    """
    if config.tool_ledger is not None:
        return bool(config.tool_ledger)
    return mode != "sync"


def attach_durability(
    rc: RunContext,
    config: AgentConfig,
    *,
    durability: DurabilityMode | None = None,
    replay: bool = False,
) -> None:
    """Give ``rc`` its checkpoint writer and, on a thread, its tool-effect ledger.

    ``durability`` is the per-run override of ``config.durability``;
    ``replay`` is on for a run that continues a turn (``continue_turn``,
    ``resume(invocation_id=...)``, ``run(invocation_id=...)``) and off for a
    fresh turn, which has no records to find.
    """
    checkpointer = config.checkpointer
    if checkpointer is None:
        rc.checkpoints = None
        rc.ledger = None
        return
    mode = check_durability(durability if durability is not None else config.durability)
    rc.checkpoints = CheckpointWriter(checkpointer, mode)
    rc.ledger = None
    if rc.thread_id and ledger_enabled(config, mode):
        from locus.memory.ledger import RunLedger, ToolEffectLedger  # noqa: PLC0415

        rc.ledger = RunLedger(
            ToolEffectLedger(checkpointer),
            thread_id=rc.thread_id,
            invocation_id=rc.run_id,
            replay=replay,
            prompt=rc.prompt,
            metadata=rc.metadata,
            durability=mode,
            rerun_idempotent=config.rerun_idempotent_on_replay,
        )


def retry_ledger(rc: RunContext) -> RunLedger | None:
    """The ledger a hook-requested retry records to: the run's, with replay off.

    A retry must run the call, so the record of the first attempt is not
    served back; the retry's result replaces it.
    """
    return rc.ledger.without_replay() if rc.ledger is not None else None
