# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Approval stores: a held action that waits for a person, across a restart.

``gate_tool(..., on_refusal="interrupt", approval=store)`` turns a
``require_human`` hold into a pause. The run stops at the held call, the
checkpointer keeps the conversation, and the store keeps the pending approval.
A person decides with :meth:`ApprovalStore.decide`, and
``agent.resume(..., thread_id=..., perform_dangling=True)`` re-issues the held
call, which finds the decision::

    store = FileApprovals("approvals.json")
    refund = gate_tool(
        issue_refund,
        policy=policy,
        approval=store,
        on_refusal="interrupt",
        trail=trail,
    )
    agent = Agent(
        model=model, tools=[refund], checkpointer=FileCheckpointer("checkpoints")
    )

    async for event in agent.run("refund order 4821", thread_id="t1"):
        if isinstance(event, InterruptEvent):
            approval_id = event.metadata["approval_id"]  # the run is parked

    # Later, from any process that can open the same file:
    store.decide(approval_id, "approved", by="alice@example.com")
    async for event in agent.resume(
        "approved", thread_id="t1", perform_dangling=True
    ):
        ...

An approval names one call. Its id is derived from the principal, the tool and
the canonical JSON of the arguments, so a call with different arguments is a
different approval and waits for its own decision. An approval is used once:
the gate consumes it immediately before the side effect, so a repeated call
holds again instead of riding an old yes.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable


if TYPE_CHECKING:
    from collections.abc import Mapping


Status = Literal["pending", "approved", "denied", "consumed"]
Verdict = Literal["approved", "denied"]

#: Records a held call can still be matched to: waiting, or decided but not yet
#: acted on. A consumed record is history; the next identical call opens a new one.
_LIVE: frozenset[str] = frozenset({"pending", "approved", "denied"})


def call_digest(principal: str, tool: str, arguments: Mapping[str, Any]) -> str:
    """SHA-256 over the canonical form of one call.

    Key order does not matter; any change to a value, the tool or the principal
    does.
    """
    canonical = json.dumps(
        {"principal": principal, "tool": tool, "arguments": dict(arguments)},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class ApprovalRecord:
    """One held call and what became of it."""

    approval_id: str
    digest: str
    principal: str
    tool: str
    arguments: dict[str, Any]
    #: ``pending`` → ``approved`` / ``denied`` → ``consumed``.
    status: Status = "pending"
    #: The policy's reason for holding the call.
    reason: str = ""
    created_at: str = field(default_factory=_now)
    verdict: Verdict | None = None
    decided_by: str | None = None
    decided_at: str | None = None


@runtime_checkable
class ApprovalStore(Protocol):
    """Where held calls wait for a decision.

    A superset of :class:`~tulip.control.ApprovalBridge`: ``submit`` and
    ``state`` keep that shape, so a store works anywhere a bridge does.
    """

    def submit(
        self, principal: str, tool: str, args: Mapping[str, Any], *, reason: str = ""
    ) -> str:
        """Record a held call, or return the live record for the same call."""
        ...

    def state(self, approval_id: str) -> str | None:
        """The record's status, or ``None`` for an unknown id."""
        ...

    def get(self, approval_id: str) -> ApprovalRecord | None:
        """The full record, or ``None`` for an unknown id."""
        ...

    def decide(self, approval_id: str, verdict: Verdict, *, by: str) -> ApprovalRecord:
        """Approve or deny a pending record, naming who decided."""
        ...

    def consume(self, approval_id: str) -> ApprovalRecord:
        """Mark a decided record as acted on, so it cannot be used again."""
        ...

    def pending(self) -> list[ApprovalRecord]:
        """Records still waiting for a decision, oldest first."""
        ...


class _Approvals:
    """The state machine, over a load/save pair a subclass provides."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def _load(self) -> dict[str, ApprovalRecord]:
        raise NotImplementedError

    def _save(self, records: dict[str, ApprovalRecord]) -> None:
        raise NotImplementedError

    def submit(
        self, principal: str, tool: str, args: Mapping[str, Any], *, reason: str = ""
    ) -> str:
        digest = call_digest(principal, tool, args)
        with self._lock:
            records = self._load()
            same_call = [r for r in records.values() if r.digest == digest]
            live = [r for r in same_call if r.status in _LIVE]
            if live:
                return max(live, key=lambda r: r.created_at).approval_id
            approval_id = f"appr-{digest[:16]}-{len(same_call) + 1}"
            records[approval_id] = ApprovalRecord(
                approval_id=approval_id,
                digest=digest,
                principal=principal,
                tool=tool,
                arguments=dict(args),
                reason=reason,
            )
            self._save(records)
            return approval_id

    def state(self, approval_id: str) -> str | None:
        record = self.get(approval_id)
        return record.status if record is not None else None

    def get(self, approval_id: str) -> ApprovalRecord | None:
        with self._lock:
            return self._load().get(approval_id)

    def decide(self, approval_id: str, verdict: Verdict, *, by: str) -> ApprovalRecord:
        if verdict not in ("approved", "denied"):
            raise ValueError(f"verdict must be 'approved' or 'denied', not {verdict!r}")
        if not by:
            raise ValueError("a decision names who made it")
        with self._lock:
            records = self._load()
            record = records.get(approval_id)
            if record is None:
                raise KeyError(approval_id)
            if record.status != "pending":
                raise ValueError(f"approval {approval_id} is already {record.status}")
            record = replace(
                record, status=verdict, verdict=verdict, decided_by=by, decided_at=_now()
            )
            records[approval_id] = record
            self._save(records)
            return record

    def consume(self, approval_id: str) -> ApprovalRecord:
        with self._lock:
            records = self._load()
            record = records.get(approval_id)
            if record is None:
                raise KeyError(approval_id)
            if record.status not in ("approved", "denied"):
                raise ValueError(f"approval {approval_id} is {record.status}, not decided")
            record = replace(record, status="consumed")
            records[approval_id] = record
            self._save(records)
            return record

    def pending(self) -> list[ApprovalRecord]:
        with self._lock:
            waiting = [r for r in self._load().values() if r.status == "pending"]
        return sorted(waiting, key=lambda r: r.created_at)


class InMemoryApprovals(_Approvals):
    """An approval store in this process only. Gone on restart; for tests and demos."""

    def __init__(self) -> None:
        super().__init__()
        self._records: dict[str, ApprovalRecord] = {}

    def _load(self) -> dict[str, ApprovalRecord]:
        return dict(self._records)

    def _save(self, records: dict[str, ApprovalRecord]) -> None:
        self._records = dict(records)


class FileApprovals(_Approvals):
    """An approval store in one JSON file.

    Every call re-reads the file, so a decision written by another process is
    seen by the next call. Writes are atomic (temporary file, then rename), so a
    crash never leaves a half-written store. Concurrent writers are serialised
    within a process only: use one process to decide, or a database-backed store
    for many.
    """

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)

    def _load(self) -> dict[str, ApprovalRecord]:
        if not self.path.exists():
            return {}
        data: dict[str, dict[str, Any]] = json.loads(self.path.read_text(encoding="utf-8"))
        return {key: ApprovalRecord(**value) for key, value in data.items()}

    def _save(self, records: dict[str, ApprovalRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {key: asdict(record) for key, record in records.items()},
            indent=2,
            sort_keys=True,
            default=str,
        )
        fd, tmp = tempfile.mkstemp(
            dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            Path(tmp).replace(self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


__all__ = [
    "ApprovalRecord",
    "ApprovalStore",
    "FileApprovals",
    "InMemoryApprovals",
    "Status",
    "Verdict",
    "call_digest",
]
