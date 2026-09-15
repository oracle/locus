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

**Who may decide.** Without an :class:`ApprovalAuthority`, any named principal
can decide. With one, a decision is checked against it at decision time::

    authority = ApprovalAuthority(
        rules=(
            ApproverRule(
                labels=frozenset({"payment"}), roles=frozenset({"finance"})
            ),
            ApproverRule(
                labels=frozenset({"production"}),
                approvers=frozenset({"olga"}),
                quorum=2,
            ),
        ),
        roles_of=directory.roles_for,
        delegations=(
            Delegation(
                grantor="olga", grantee="dan", expires_at="2026-10-01T00:00:00Z"
            ),
        ),
    )
    store = FileApprovals("approvals.json", authority=authority)

Every rule whose labels the held action carries must reach its quorum of
distinct authorised approvers before the call is approved; one authorised
denial ends it. The principal that requested the action cannot approve it,
directly or through a delegation it granted, unless every matching rule says
``allow_self_approval``. A decision by someone without authority raises
:class:`ApprovalAuthorityError` and is kept on the record. An action no rule
covers cannot be approved by anyone: the authority fails closed.
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
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol, runtime_checkable


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping


Status = Literal["pending", "approved", "denied", "consumed"]
Verdict = Literal["approved", "denied"]

#: Records a held call can still be matched to: waiting, or decided but not yet
#: acted on. A consumed record is history; the next identical call opens a new one.
_LIVE: frozenset[str] = frozenset({"pending", "approved", "denied"})


def call_digest(
    principal: str,
    tool: str,
    arguments: Mapping[str, Any],
    context: Mapping[str, str] | None = None,
) -> str:
    """SHA-256 over the canonical form of one call.

    Key order does not matter; any change to a value, the tool, the principal
    or the context does. ``context`` binds the call to where it was made, such
    as a policy version or a thread. An empty context hashes exactly like no
    context, so records written before contexts existed still match.
    """
    body: dict[str, Any] = {"principal": principal, "tool": tool, "arguments": dict(arguments)}
    if context:
        body["context"] = dict(context)
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _utcnow() -> datetime:
    return datetime.now(UTC)


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
    #: Who decided. With a quorum, every approver, comma-separated.
    decided_by: str | None = None
    decided_at: str | None = None
    #: The held action's labels (environment, kind, tags); what approver rules match.
    labels: list[str] = field(default_factory=list)
    #: Each accepted decision: ``by``, ``at``, ``verdict``, and with an authority
    #: the ``basis`` it was accepted on and the rule positions it counted toward.
    approvals: list[dict[str, Any]] = field(default_factory=list)
    #: Decisions refused by the authority: ``by``, ``at``, ``verdict``, ``reason``.
    rejections: list[dict[str, Any]] = field(default_factory=list)
    #: What the call was bound to (a policy version, a thread); part of the digest.
    context: dict[str, str] = field(default_factory=dict)
    #: The arguments the approvers approved, when they edited the call.
    approved_arguments: dict[str, Any] | None = None

    @property
    def approvers(self) -> list[str]:
        """Distinct principals whose approval was accepted, in order."""
        seen: list[str] = []
        for entry in self.approvals:
            if entry.get("verdict") == "approved" and entry["by"] not in seen:
                seen.append(entry["by"])
        return seen


@dataclass(frozen=True)
class ApproverRule:
    """Who may decide actions that carry some labels.

    Args:
        labels: Labels this rule governs. Empty matches every action.
        approvers: Principals allowed to decide, by name.
        roles: Roles allowed to decide, resolved per principal by
            :attr:`ApprovalAuthority.roles_of`.
        quorum: Distinct authorised approvals needed.
        allow_self_approval: Whether the principal that requested the action may
            approve it. Off by default: separation of duties.
    """

    labels: frozenset[str] = frozenset()
    approvers: frozenset[str] = frozenset()
    roles: frozenset[str] = frozenset()
    quorum: int = 1
    allow_self_approval: bool = False

    def __post_init__(self) -> None:
        if self.quorum < 1:
            raise ValueError("quorum must be at least 1")
        if not (self.approvers or self.roles):
            raise ValueError("an approver rule names approvers or roles")

    def matches(self, labels: frozenset[str]) -> bool:
        """Whether this rule governs an action carrying ``labels``."""
        return not self.labels or bool(self.labels & labels)


@dataclass(frozen=True)
class Delegation:
    """An approver lends their authority to someone else until a deadline.

    Args:
        grantor: The principal whose authority is lent.
        grantee: The principal who may use it.
        expires_at: ISO-8601 deadline; a naive timestamp is read as UTC.
        labels: Limit the delegation to actions carrying these labels. Empty
            lends everything the grantor may decide.
    """

    grantor: str
    grantee: str
    expires_at: str
    labels: frozenset[str] = frozenset()

    def active(self, now: datetime) -> bool:
        """Whether the deadline is still in the future."""
        deadline = datetime.fromisoformat(self.expires_at)
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=UTC)
        return now < deadline


class ApprovalAuthorityError(PermissionError):
    """A decision was refused because the decider lacks authority for it."""


class _Basis(NamedTuple):
    rules: frozenset[int]
    label: str


@dataclass(frozen=True)
class ApprovalAuthority:
    """Approver rules, delegations, and how to look up a principal's roles.

    Rules are evaluated by position, so keep their order stable for records
    that are still pending.
    """

    rules: tuple[ApproverRule, ...]
    delegations: tuple[Delegation, ...] = ()
    roles_of: Callable[[str], Iterable[str]] | None = None
    clock: Callable[[], datetime] = _utcnow

    def matching(self, labels: Iterable[str]) -> dict[int, ApproverRule]:
        """The rules governing an action carrying ``labels``, by position."""
        wanted = frozenset(labels)
        return {i: rule for i, rule in enumerate(self.rules) if rule.matches(wanted)}

    def _holds(self, rule: ApproverRule, principal: str) -> bool:
        if principal in rule.approvers:
            return True
        if rule.roles and self.roles_of is not None:
            return bool(rule.roles & frozenset(self.roles_of(principal)))
        return False

    def check(self, record: ApprovalRecord, by: str) -> _Basis:
        """Which matching rules ``by`` may decide under, and on what basis.

        Raises:
            ApprovalAuthorityError: No rule covers the action, ``by`` requested
                it, or ``by`` holds none of the matching rules, directly or by an
                active delegation.
        """
        labels = frozenset(record.labels)
        rules = self.matching(labels)
        if not rules:
            raise ApprovalAuthorityError(
                f"no approver rule covers labels {sorted(labels)}; nobody may decide it"
            )
        self_allowed = all(rule.allow_self_approval for rule in rules.values())
        if by == record.principal and not self_allowed:
            raise ApprovalAuthorityError(f"{by} requested this action and may not decide it")

        direct = frozenset(i for i, rule in rules.items() if self._holds(rule, by))
        if direct:
            return _Basis(direct, by)

        now = self.clock()
        for grant in self.delegations:
            if grant.grantee != by or not grant.active(now):
                continue
            if grant.labels and not (grant.labels & labels):
                continue
            if grant.grantor == record.principal and not self_allowed:
                continue
            lent = frozenset(i for i, rule in rules.items() if self._holds(rule, grant.grantor))
            if lent:
                return _Basis(lent, f"delegated by {grant.grantor}")
        raise ApprovalAuthorityError(f"{by} may not decide {record.tool} (labels {sorted(labels)})")

    def satisfied(self, record: ApprovalRecord, approvals: list[dict[str, Any]]) -> bool:
        """Whether every matching rule has its quorum of distinct approvers."""
        for i, rule in self.matching(record.labels).items():
            approvers = {
                a["by"]
                for a in approvals
                if a.get("verdict") == "approved" and i in a.get("rules", ())
            }
            if len(approvers) < rule.quorum:
                return False
        return True


@runtime_checkable
class ApprovalStore(Protocol):
    """Where held calls wait for a decision.

    A superset of :class:`~tulip.control.ApprovalBridge`: ``submit`` and
    ``state`` keep that shape, so a store works anywhere a bridge does.
    """

    def submit(
        self,
        principal: str,
        tool: str,
        args: Mapping[str, Any],
        *,
        reason: str = "",
        labels: Iterable[str] = (),
        context: Mapping[str, str] | None = None,
    ) -> str:
        """Record a held call, or return the live record for the same call."""
        ...

    def state(self, approval_id: str) -> str | None:
        """The record's status, or ``None`` for an unknown id."""
        ...

    def get(self, approval_id: str) -> ApprovalRecord | None:
        """The full record, or ``None`` for an unknown id."""
        ...

    def decide(
        self,
        approval_id: str,
        verdict: Verdict,
        *,
        by: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> ApprovalRecord:
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

    def __init__(self, authority: ApprovalAuthority | None = None) -> None:
        self._lock = threading.Lock()
        self.authority = authority

    def _load(self) -> dict[str, ApprovalRecord]:
        raise NotImplementedError

    def _save(self, records: dict[str, ApprovalRecord]) -> None:
        raise NotImplementedError

    def submit(
        self,
        principal: str,
        tool: str,
        args: Mapping[str, Any],
        *,
        reason: str = "",
        labels: Iterable[str] = (),
        context: Mapping[str, str] | None = None,
    ) -> str:
        digest = call_digest(principal, tool, args, context)
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
                labels=sorted(set(labels)),
                context=dict(context or {}),
            )
            self._save(records)
            return approval_id

    def state(self, approval_id: str) -> str | None:
        record = self.get(approval_id)
        return record.status if record is not None else None

    def get(self, approval_id: str) -> ApprovalRecord | None:
        with self._lock:
            return self._load().get(approval_id)

    def decide(
        self,
        approval_id: str,
        verdict: Verdict,
        *,
        by: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> ApprovalRecord:
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
            if arguments is not None:
                if verdict != "approved":
                    raise ValueError("only an approval can edit the call's arguments")
                if set(arguments) != set(record.arguments):
                    raise ValueError(
                        f"an edit keeps the call's arguments {sorted(record.arguments)}; "
                        f"got {sorted(arguments)}"
                    )
            edited = dict(arguments) if arguments is not None else None
            record = self._decided(records, record, verdict, by, edited)
            records[approval_id] = record
            self._save(records)
            return record

    def _decided(
        self,
        records: dict[str, ApprovalRecord],
        record: ApprovalRecord,
        verdict: Verdict,
        by: str,
        edited: dict[str, Any] | None = None,
    ) -> ApprovalRecord:
        now = _now()
        entry: dict[str, Any] = {"by": by, "at": now, "verdict": verdict}
        if edited is not None:
            entry["arguments"] = edited
        if self.authority is None:
            return replace(
                record,
                status=verdict,
                verdict=verdict,
                decided_by=by,
                decided_at=now,
                approvals=[*record.approvals, entry],
                approved_arguments=edited,
            )

        try:
            basis = self.authority.check(record, by)
        except ApprovalAuthorityError as error:
            records[record.approval_id] = replace(
                record, rejections=[*record.rejections, {**entry, "reason": str(error)}]
            )
            self._save(records)
            raise

        entry.update({"basis": basis.label, "rules": sorted(basis.rules)})
        if verdict == "denied":
            return replace(
                record,
                status="denied",
                verdict="denied",
                decided_by=by,
                decided_at=now,
                approvals=[*record.approvals, entry],
            )
        if by in record.approvers:
            raise ValueError(f"{by} has already approved {record.approval_id}")
        earlier = [a.get("arguments") for a in record.approvals if a.get("verdict") == "approved"]
        if any(previous != edited for previous in earlier):
            raise ValueError(
                "approvers must approve the same arguments; this approval differs "
                "from an earlier one"
            )
        approvals = [*record.approvals, entry]
        if not self.authority.satisfied(record, approvals):
            return replace(record, approvals=approvals)
        approved = replace(record, approvals=approvals)
        return replace(
            approved,
            status="approved",
            verdict="approved",
            decided_by=", ".join(approved.approvers),
            decided_at=now,
            approved_arguments=edited,
        )

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

    def __init__(self, authority: ApprovalAuthority | None = None) -> None:
        super().__init__(authority)
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

    def __init__(self, path: str | Path, authority: ApprovalAuthority | None = None) -> None:
        super().__init__(authority)
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
    "ApprovalAuthority",
    "ApprovalAuthorityError",
    "ApprovalRecord",
    "ApprovalStore",
    "ApproverRule",
    "Delegation",
    "FileApprovals",
    "InMemoryApprovals",
    "Status",
    "Verdict",
    "call_digest",
]
