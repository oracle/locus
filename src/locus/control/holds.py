# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The hold a paused thread carries in its checkpoint.

When a gated tool holds a call in interrupt mode, the run pauses and the
checkpointer saves the conversation. Before this module, that checkpoint said
only that a tool call had no result yet; which approval it waited on, in which
store, was known to the process that paused and to nobody else. Now the runtime
writes the hold into the state it checkpoints, under
``state.metadata["_locus_pending_holds"]``, keyed by the held tool call's id:

.. code-block:: json

    {"appr-…": {"approval_id": "appr-…", "tool": "issue_refund", "action": "issue_refund",
                "asset": "o1", "principal": "svc-billing", "reason": "…",
                "arguments": {"order_id": "o1"}, "store": "file://approvals.json", …}}

So a process that finds the thread later -- the one that will call
``agent.resume(thread_id, perform_dangling=True)``, or an operator deciding what
to approve -- reads what the thread waits on from the checkpoint alone::

    for hold in await pending_holds(checkpointer, "t1"):
        print(hold.approval_id, hold.action, hold.asset, hold.state(store))

The hold is removed when the held call is answered: performed after a decision,
or folded with a text response. A checkpoint written before this existed has no
holds; ``resume`` then falls back, as before, to the dangling tool call in the
conversation, so nothing about old threads changes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from locus.control.approvals import ApprovalStore
    from locus.core.state import AgentState


__all__ = [
    "HOLDS_KEY",
    "PendingHold",
    "hold_from_interrupt",
    "holds_of",
    "pending_holds",
    "with_hold",
    "without_hold",
]

#: Where the holds live in ``AgentState.metadata`` (and so in every checkpoint).
HOLDS_KEY = "_locus_pending_holds"


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class PendingHold:
    """One held tool call a thread waits on, as its checkpoint records it."""

    #: The id of the tool call the model made; what the fold answers.
    tool_call_id: str
    #: The record in the approval store.
    approval_id: str
    #: The tool that was called.
    tool: str
    #: The action the policy weighed, and what it acts on.
    action: str
    asset: str
    #: Who the held action is attributed to.
    principal: str
    #: The policy's reason for holding.
    reason: str
    arguments: dict[str, Any] = field(default_factory=dict)
    #: Approvers accepted so far (a quorum in progress).
    approvers: list[str] = field(default_factory=list)
    #: What the approval is bound to: a policy version, a thread.
    context: dict[str, str] = field(default_factory=dict)
    #: The store's spec (:func:`locus.control.approval_store` opens it), when known.
    store: str | None = None
    held_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        """The hold as the checkpoint stores it."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PendingHold:
        """The hold a checkpoint stored. Unknown keys are ignored, so a newer
        runtime's holds still load here."""
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in names})

    def state(self, store: ApprovalStore) -> str | None:
        """The approval's status in ``store`` now: re-checks the hold against the store."""
        return store.state(self.approval_id)


def hold_from_interrupt(
    tool_call_id: str, tool: str, payload: dict[str, Any]
) -> PendingHold | None:
    """The hold an interrupt marker describes, or ``None`` for one that is not an approval.

    A gated tool's marker carries the approval in ``metadata`` (``approval_id``,
    ``action``, ``asset``, ``principal``, ``reason``, ``arguments``, ``approvers``,
    ``context``, ``store``); ``ask_user``'s has no ``approval_id`` and is not a hold.
    """
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or not metadata.get("approval_id"):
        return None
    return PendingHold(
        tool_call_id=tool_call_id,
        approval_id=str(metadata["approval_id"]),
        tool=tool,
        action=str(metadata.get("action") or tool),
        asset=str(metadata.get("asset") or ""),
        principal=str(metadata.get("principal") or ""),
        reason=str(metadata.get("reason") or ""),
        arguments=dict(metadata.get("arguments") or {}),
        approvers=list(metadata.get("approvers") or []),
        context=dict(metadata.get("context") or {}),
        store=metadata.get("store"),
    )


def holds_of(state: AgentState) -> list[PendingHold]:
    """The holds ``state`` carries, oldest first; none for a checkpoint written before holds."""
    raw = state.metadata.get(HOLDS_KEY)
    if not isinstance(raw, dict):
        return []
    holds = [PendingHold.from_dict(v) for v in raw.values() if isinstance(v, dict)]
    return sorted(holds, key=lambda h: h.held_at)


def with_hold(state: AgentState, hold: PendingHold) -> AgentState:
    """``state`` carrying ``hold``, replacing any earlier hold on the same call."""
    current = state.metadata.get(HOLDS_KEY)
    holds = dict(current) if isinstance(current, dict) else {}
    holds[hold.tool_call_id] = hold.to_dict()
    return state.with_metadata(HOLDS_KEY, holds)


def without_hold(state: AgentState, tool_call_id: str) -> AgentState:
    """``state`` no longer waiting on ``tool_call_id``; unchanged when it never was."""
    current = state.metadata.get(HOLDS_KEY)
    if not isinstance(current, dict) or tool_call_id not in current:
        return state
    holds = {k: v for k, v in current.items() if k != tool_call_id}
    metadata = {k: v for k, v in state.metadata.items() if k != HOLDS_KEY}
    if holds:
        metadata[HOLDS_KEY] = holds
    return state.model_copy(update={"metadata": metadata})


async def pending_holds(checkpointer: Any, thread_id: str) -> list[PendingHold]:
    """The holds the thread's latest checkpoint carries; empty for no checkpoint."""
    state = await checkpointer.load(thread_id)
    return holds_of(state) if state is not None else []
