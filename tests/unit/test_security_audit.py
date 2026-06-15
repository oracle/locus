# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Tests for the tamper-evident, hash-chained audit trail."""

from __future__ import annotations

import dataclasses
import itertools
import json
from collections.abc import Callable

from tulip.security.audit import AuditTrail


def _fixed_clock() -> Callable[[], str]:
    counter = itertools.count()
    return lambda: f"2026-06-15T00:00:{next(counter):02d}+00:00"


def test_empty_trail_head_is_genesis_and_verifies() -> None:
    trail = AuditTrail(clock=_fixed_clock())
    assert trail.head == "0" * 64
    assert len(trail) == 0
    assert trail.verify()


def test_records_chain_and_verify() -> None:
    trail = AuditTrail(clock=_fixed_clock())
    r0 = trail.record("red_team.start", {"target": "bot"})
    r1 = trail.record("probe.finding", {"probe": "direct-prompt-injection"})
    assert r0.seq == 0
    assert r0.prev_hash == "0" * 64
    assert r1.prev_hash == r0.hash  # each record commits to the previous
    assert trail.head == r1.hash
    assert trail.verify()


def test_tamper_with_payload_breaks_chain() -> None:
    trail = AuditTrail(clock=_fixed_clock())
    trail.record("a", {"v": 1})
    trail.record("b", {"v": 2})
    trail.record("c", {"v": 3})
    records = trail.records()
    # Mutate a middle record's payload (its stored hash now no longer matches).
    records[1] = dataclasses.replace(records[1], payload={"v": 999})
    assert not AuditTrail.from_records(records).verify()


def test_tamper_with_reorder_breaks_chain() -> None:
    trail = AuditTrail(clock=_fixed_clock())
    trail.record("a")
    trail.record("b")
    records = trail.records()
    records[0], records[1] = records[1], records[0]
    assert not AuditTrail.from_records(records).verify()


def test_deletion_breaks_chain() -> None:
    trail = AuditTrail(clock=_fixed_clock())
    trail.record("a")
    trail.record("b")
    trail.record("c")
    records = trail.records()
    del records[1]
    assert not AuditTrail.from_records(records).verify()


def test_export_jsonl_roundtrips_and_is_siem_shippable() -> None:
    trail = AuditTrail(clock=_fixed_clock())
    trail.record("x", {"k": "v"})
    trail.record("y")
    lines = trail.export_jsonl().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["event_type"] == "x"
    assert all("hash" in p and "prev_hash" in p for p in parsed)


def test_record_event_extracts_scalar_fields() -> None:
    class _Event:
        def __init__(self) -> None:
            self.name = "tool-call"
            self.tool = "scan_endpoint"
            self.payload = {"not": "scalar"}  # ignored — not a scalar field name

    rec = AuditTrail(clock=_fixed_clock()).record_event(_Event())
    assert rec.event_type == "_Event"
    assert rec.payload == {"name": "tool-call", "tool": "scan_endpoint"}
