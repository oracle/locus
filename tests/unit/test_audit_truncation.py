# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""What the audit chain does and does not detect.

``verify()`` used to promise "no edit, deletion, or reorder". Three of those
four attacks below break the chain. Truncation does not, and no amount of
hashing *inside* a chain can make it: the records that remain are a valid
shorter chain, and nothing in them attests to a link that was never handed
over. These tests pin the real boundary so the docstring cannot drift back.
"""

from __future__ import annotations

import dataclasses

import pytest

from tulip.security.audit import AuditRecord, AuditTrail, _entry_hash


def _trail(n: int = 4) -> AuditTrail:
    trail = AuditTrail(clock=lambda: "2026-01-01T00:00:00+00:00")
    for i in range(n):
        trail.record("action-admission", {"outcome": "allow", "n": i})
    return trail


def test_intact_chain_verifies() -> None:
    trail = _trail()
    assert trail.verify() is True
    assert trail.verify(expected_head=trail.head) is True


def test_edit_is_detected() -> None:
    """Change a payload, keep the stored hash — the recomputation disagrees."""
    records = _trail().records()
    forged = dataclasses.replace(records[2], payload={**records[2].payload, "outcome": "deny"})
    assert AuditTrail.from_records([*records[:2], forged, *records[3:]]).verify() is False


def test_edit_with_recomputed_hash_is_detected() -> None:
    """An attacker who knows the algorithm still breaks the NEXT record's link."""
    records = _trail().records()
    bad = dataclasses.replace(records[1], payload={**records[1].payload, "outcome": "deny"})
    rehashed = dataclasses.replace(
        bad, hash=_entry_hash(bad.seq, bad.ts, bad.event_type, bad.payload, bad.prev_hash)
    )
    assert AuditTrail.from_records([records[0], rehashed, *records[2:]]).verify() is False


def test_reorder_is_detected() -> None:
    assert AuditTrail.from_records(list(reversed(_trail().records()))).verify() is False


def test_deletion_from_the_middle_is_detected() -> None:
    records = _trail().records()
    assert AuditTrail.from_records([records[0], *records[2:]]).verify() is False


@pytest.mark.parametrize("keep", [3, 2, 1, 0], ids=["drop-1", "drop-2", "drop-3", "drop-all"])
def test_truncation_is_NOT_detected_without_an_anchor(keep: int) -> None:
    """The documented limitation, pinned.

    If this ever starts failing, someone has strengthened the chain and the
    docstring on :meth:`AuditTrail.verify` needs to be relaxed to match.
    """
    truncated = AuditTrail.from_records(_trail().records()[:keep])
    assert truncated.verify() is True


@pytest.mark.parametrize("keep", [3, 2, 1, 0], ids=["drop-1", "drop-2", "drop-3", "drop-all"])
def test_truncation_IS_detected_with_an_anchor(keep: int) -> None:
    """The fix: an externally recorded head catches every truncation."""
    trail = _trail()
    anchor = trail.head  # written somewhere the agent cannot reach
    truncated = AuditTrail.from_records(trail.records()[:keep])
    assert truncated.verify(expected_head=anchor) is False


def test_anchor_also_catches_a_rehashed_tail() -> None:
    """Rewrite the last record convincingly; the head still moves."""
    trail = _trail()
    anchor, records = trail.head, trail.records()
    last = records[-1]
    bad = dataclasses.replace(last, payload={**last.payload, "outcome": "deny"})
    rehashed = dataclasses.replace(
        bad, hash=_entry_hash(bad.seq, bad.ts, bad.event_type, bad.payload, bad.prev_hash)
    )
    forged = AuditTrail.from_records([*records[:-1], rehashed])
    assert forged.verify() is True  # internally consistent...
    assert forged.verify(expected_head=anchor) is False  # ...but not the chain we anchored


def test_anchor_of_an_empty_trail() -> None:
    empty = AuditTrail()
    assert empty.verify(expected_head=empty.head) is True
    assert empty.verify(expected_head="0" * 64) is True  # genesis is the empty head


def test_records_returns_a_copy() -> None:
    trail = _trail(2)
    got = trail.records()
    got.append(AuditRecord(seq=99, ts="x", event_type="x", payload={}, prev_hash="x", hash="x"))
    assert len(trail.records()) == 2
    assert trail.verify() is True
