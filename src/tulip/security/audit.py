# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Tamper-evident audit trail — every agent action as replayable evidence.

A :class:`AuditTrail` records agent actions into a hash chain: each record
commits to the hash of the one before it, so a later edit, reorder, or
deletion from the middle breaks the chain and :meth:`AuditTrail.verify`
returns ``False``. The trail exports as JSONL for shipping to a SIEM.

Truncation is the exception, and it is worth stating plainly: lopping records
off the *end* of a chain leaves a shorter chain that still verifies, because
no link can attest to one that was never handed to it. Anchor
:attr:`AuditTrail.head` externally and pass it to
``verify(expected_head=...)`` to close that gap — see :meth:`AuditTrail.verify`.

This is a *supporting property* of a trustworthy security agent — the agent
doing red-team / assurance work leaves a forensic record that holds up —
not a governance/policy product. It does not block or enforce; it makes the
record auditable after the fact.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any


# The chain's anchor — `prev_hash` of the first record.
_GENESIS = "0" * 64


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _entry_hash(
    seq: int,
    ts: str,
    event_type: str,
    payload: Mapping[str, Any],
    prev_hash: str,
) -> str:
    """SHA-256 over the canonical (sorted, compact) record body + prev hash."""
    canonical = json.dumps(
        {
            "seq": seq,
            "ts": ts,
            "event_type": event_type,
            "payload": dict(payload),
            "prev_hash": prev_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AuditRecord:
    """One link in the audit chain. ``hash`` commits to ``prev_hash``."""

    seq: int
    ts: str
    event_type: str
    payload: dict[str, Any]
    prev_hash: str
    hash: str


class AuditTrail:
    """An append-only, hash-chained log of agent actions.

    Append with :meth:`record` (or :meth:`record_event` for a Tulip event);
    check integrity with :meth:`verify`; ship with :meth:`export_jsonl`.
    Pass ``clock`` to make timestamps deterministic in tests.
    """

    def __init__(self, *, clock: Callable[[], str] | None = None) -> None:
        self._records: list[AuditRecord] = []
        self._clock = clock or _utc_now_iso

    @property
    def head(self) -> str:
        """Hash of the latest record, or the genesis anchor when empty."""
        return self._records[-1].hash if self._records else _GENESIS

    def __len__(self) -> int:
        return len(self._records)

    def record(self, event_type: str, payload: Mapping[str, Any] | None = None) -> AuditRecord:
        """Append a record committing to the current chain head."""
        seq = len(self._records)
        prev = self.head
        ts = self._clock()
        body = dict(payload or {})
        rec = AuditRecord(
            seq=seq,
            ts=ts,
            event_type=event_type,
            payload=body,
            prev_hash=prev,
            hash=_entry_hash(seq, ts, event_type, body, prev),
        )
        self._records.append(rec)
        return rec

    def record_event(self, event: Any) -> AuditRecord:
        """Append a record for a Tulip event (duck-typed; safe scalar fields)."""
        payload: dict[str, Any] = {}
        for key in ("name", "tool", "final_message", "reason", "content", "asset"):
            val = getattr(event, key, None)
            if isinstance(val, str | int | float | bool):
                payload[key] = val
        return self.record(type(event).__name__, payload)

    def records(self) -> list[AuditRecord]:
        """A copy of the records, in order."""
        return list(self._records)

    def verify(self, *, expected_head: str | None = None) -> bool:
        """Whether the chain is internally consistent, and optionally un-truncated.

        On its own this catches every edit, reorder, and deletion **from the
        middle** of the chain: each of those leaves a record whose stored hash
        no longer matches its contents, or whose ``prev_hash`` no longer points
        at the record before it.

        It cannot, on its own, catch a *truncation*. Dropping records from the
        end — or discarding the trail entirely — leaves a shorter chain that is
        perfectly valid on its own terms, so this returns ``True``. That is a
        property of hash chains in general, not of this implementation: nothing
        inside a chain can attest to a link that was never handed to it.

        Truncation is what ``expected_head`` is for. Persist :attr:`head`
        somewhere the agent cannot reach — a WORM bucket, a append-only log, a
        transparency log, a co-signer — and pass it back here. Every attack
        above, truncation included, changes the head:

        ```python
        anchor = trail.head  # written to durable, external storage
        ...
        trail.verify(expected_head=anchor)  # False if anything was removed
        ```

        Args:
            expected_head: The chain head recorded out-of-band. When given, the
                trail must also *end* on this hash. Omit it and truncation goes
                undetected — see above.

        Returns:
            ``True`` if the chain is intact, and ends at ``expected_head`` when
            one was supplied.
        """
        prev = _GENESIS
        for i, rec in enumerate(self._records):
            if rec.seq != i or rec.prev_hash != prev:
                return False
            if _entry_hash(rec.seq, rec.ts, rec.event_type, rec.payload, rec.prev_hash) != rec.hash:
                return False
            prev = rec.hash
        if expected_head is not None and self.head != expected_head:
            return False
        return True

    def export_jsonl(self) -> str:
        """The chain as newline-delimited JSON — one record per line, SIEM-ready."""
        return "\n".join(json.dumps(asdict(rec), default=str) for rec in self._records)

    @classmethod
    def from_records(cls, records: Iterable[AuditRecord]) -> AuditTrail:
        """Rebuild a trail from records (e.g. to :meth:`verify` an exported chain)."""
        trail = cls()
        trail._records = list(records)
        return trail


__all__ = ["AuditRecord", "AuditTrail"]
