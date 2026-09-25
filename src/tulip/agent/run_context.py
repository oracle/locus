# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-run bookkeeping for :class:`~tulip.agent.agent.Agent`.

One ``Agent`` instance is meant to serve many conversations at once — a chat
backend builds it once and calls ``run(..., thread_id=...)`` for every user.
Anything that belongs to ONE run (its cancel signal, its termination clock,
its final state, the events its hooks emit, its unverified-writes flag) must
therefore live on an object owned by that run, never on the agent. Before
2.17 these lived on the agent, and ``asyncio.gather`` over two runs returned
run A's state as run B's result.

Everything here is private: the public surface is :class:`tulip.RunInfo` (on
hook events), :meth:`Agent.cancel` and the ``thread_id`` keyed
:meth:`Agent.resume`.
"""

from __future__ import annotations

import contextvars
import copy
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tulip.core.events import CustomEvent, RunInfo


if TYPE_CHECKING:
    from tulip.core.state import AgentState
    from tulip.core.termination import TerminationCondition


@dataclass
class ResultSlot:
    """Where a run leaves its final state for the ``arun`` that drives it.

    ``arun`` publishes a slot through :data:`ARUN_RESULT_SLOT` before it
    iterates ``run()``; the first ``run()`` of the SAME agent to start in that
    context claims it. A nested run (a tool that runs another agent, or this
    one recursively) finds the slot claimed or owned by another agent and
    leaves it alone, so the outer ``arun`` always reads its own run's state.
    """

    owner: Any
    claimed: bool = False
    state: AgentState | None = None


#: The slot the innermost active ``Agent.arun`` is waiting on.
ARUN_RESULT_SLOT: contextvars.ContextVar[ResultSlot | None] = contextvars.ContextVar(
    "tulip_arun_result_slot", default=None
)


@dataclass
class PendingInterrupt:
    """A run paused on an interrupt, parked in memory under its thread id."""

    state: AgentState
    prompt: str
    thread_id: str | None
    metadata: dict[str, Any] | None


def _copy_termination(
    termination: TerminationCondition | None,
) -> TerminationCondition | None:
    """Give a run its own copy of a stateful termination condition.

    ``TimeLimit`` and friends keep a clock on the instance; resetting the one
    shared ``config.termination`` at every run start restarted every OTHER
    in-flight run's clock too. A deep copy isolates them. A condition that
    cannot be copied (it closes over a lock, a client…) falls back to the
    shared instance — the pre-2.17 behaviour — rather than failing the run.
    """
    if termination is None:
        return None
    try:
        return copy.deepcopy(termination)
    except Exception:  # noqa: BLE001 — uncopyable user object: degrade, don't fail
        return termination


@dataclass
class RunContext:
    """Mutable state owned by exactly one run (a turn, or a resumed segment)."""

    info: RunInfo
    prompt: str
    #: The metadata object as the caller passed it (tools receive this).
    metadata: dict[str, Any] | None
    termination: TerminationCondition | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    has_unverified_writes: bool = False
    result_slot: ResultSlot | None = None
    pending_events: list[CustomEvent] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        thread_id: str | None,
        prompt: str,
        metadata: dict[str, Any] | None,
        agent_name: str | None,
        termination: TerminationCondition | None,
    ) -> RunContext:
        run_termination = _copy_termination(termination)
        if run_termination is not None:
            run_termination.reset()
        return cls(
            info=RunInfo.build(
                run_id=run_id,
                thread_id=thread_id,
                metadata=metadata,
                agent_name=agent_name,
            ),
            prompt=prompt,
            metadata=metadata,
            termination=run_termination,
        )

    @property
    def run_id(self) -> str:
        return self.info.run_id

    @property
    def thread_id(self) -> str | None:
        return self.info.thread_id

    def emit(self, event: CustomEvent, *, tool_call_id: str | None = None) -> None:
        """Queue a UI-only event; the loop yields it after the current hook."""
        if not isinstance(event, CustomEvent):
            raise TypeError(
                f"hook events can only emit CustomEvent, got {type(event).__name__}; "
                "loop events are produced by the runtime itself"
            )
        update: dict[str, Any] = {}
        if event.run_id is None:
            update["run_id"] = self.info.run_id
        if event.thread_id is None:
            update["thread_id"] = self.info.thread_id
        if event.tool_call_id is None and tool_call_id:
            update["tool_call_id"] = tool_call_id
        self.pending_events.append(event.model_copy(update=update) if update else event)

    def drain(self) -> list[CustomEvent]:
        """Take every queued custom event, oldest first."""
        if not self.pending_events:
            return []
        out = self.pending_events
        self.pending_events = []
        return out


def claim_result_slot(owner: Any) -> ResultSlot | None:
    """Claim the enclosing ``arun``'s result slot if it is waiting on ``owner``."""
    slot = ARUN_RESULT_SLOT.get()
    if slot is None or slot.owner is not owner or slot.claimed:
        return None
    slot.claimed = True
    return slot
