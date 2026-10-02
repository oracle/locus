# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""A trail on disk: durable as it is written, continued when reopened, and
checkable from the file alone — with the break located, not just detected."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from tulip.control import AuditTrail, check_jsonl, verify_jsonl
from tulip.control.audit import _GENESIS


def _lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_every_record_is_on_disk_when_record_returns(tmp_path: Path) -> None:
    path = tmp_path / "s1" / "audit.jsonl"
    trail = AuditTrail(path=path)
    assert trail.path == path
    trail.record("gate", {"tool": "bash", "verdict": "allow"})
    assert len(_lines(path)) == 1
    trail.record("gate", {"tool": "write", "verdict": "deny"})
    rows = _lines(path)
    assert [r["seq"] for r in rows] == [0, 1]
    assert rows[1]["prev_hash"] == rows[0]["hash"]
    assert check_jsonl(path.read_text()).ok


def test_reopening_continues_the_chain(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    first = AuditTrail(path=path)
    first.record("gate", {"n": 1})
    head = first.head

    second = AuditTrail(path=path)
    assert len(second) == 1
    assert second.head == head
    second.record("gate", {"n": 2})
    report = check_jsonl(path.read_text(), expected_head=second.head)
    assert report.ok
    assert report.records == 2
    assert report.head == second.head
    assert report.broken_at is None
    assert report.problem == ""


def test_an_edited_line_is_located(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    trail = AuditTrail(path=path)
    for n in range(4):
        trail.record("gate", {"verdict": "deny", "n": n})
    rows = _lines(path)
    rows[2]["payload"] = {"verdict": "allow", "n": 2}
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    report = check_jsonl(path.read_text())
    assert not report.ok
    assert report.broken_at == 2
    assert "changed after it was written" in report.problem
    assert not AuditTrail(path=path).verify()


def test_a_removed_line_is_located(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    trail = AuditTrail(path=path)
    for n in range(3):
        trail.record("gate", {"n": n})
    rows = _lines(path)
    del rows[1]
    report = check_jsonl("\n".join(json.dumps(r) for r in rows))
    assert (report.ok, report.broken_at) == (False, 1)
    assert "removed or moved" in report.problem


def test_a_relinked_line_is_located() -> None:
    trail = AuditTrail()
    for n in range(3):
        trail.record("gate", {"n": n})
    rows = [json.loads(line) for line in trail.export_jsonl().splitlines()]
    rows[2]["prev_hash"] = _GENESIS
    report = check_jsonl("\n".join(json.dumps(r) for r in rows))
    assert (report.ok, report.broken_at) == (False, 2)
    assert "does not follow" in report.problem


def test_a_line_that_is_not_a_record_breaks_the_trail(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    trail = AuditTrail(path=path)
    trail.record("gate", {"n": 0})
    with path.open("a") as fh:
        fh.write("not json\n\n")
    reopened = AuditTrail(path=path)
    report = reopened.check()
    assert not report.ok
    assert report.broken_at is None
    assert "not audit records" in report.problem
    assert not verify_jsonl(path.read_text())


def test_truncation_needs_an_anchor() -> None:
    trail = AuditTrail()
    trail.record("gate", {"n": 0})
    anchor = trail.head
    trail.record("gate", {"n": 1})
    cut = trail.export_jsonl().splitlines()[0]
    assert check_jsonl(cut).ok
    report = check_jsonl(cut, expected_head=trail.head)
    assert not report.ok
    assert "cut off" in report.problem
    assert check_jsonl(cut, expected_head=anchor).ok


def test_an_empty_trail_reports_the_genesis_head() -> None:
    report = check_jsonl("")
    assert report.ok
    assert report.records == 0
    assert report.head == _GENESIS


def test_a_failed_write_keeps_nothing(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("")
    trail = AuditTrail(path=blocker / "audit.jsonl")  # a file where a directory must be
    with pytest.raises(OSError):
        trail.record("gate", {"n": 0})
    assert len(trail) == 0


def test_concurrent_records_never_share_a_parent(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    trail = AuditTrail(path=path)

    def write(n: int) -> None:
        for i in range(20):
            trail.record("gate", {"thread": n, "i": i})

    threads = [threading.Thread(target=write, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    report = check_jsonl(path.read_text(), expected_head=trail.head)
    assert report.ok
    assert report.records == 80


def test_signed_records_are_checked_per_record() -> None:
    pytest.importorskip("cryptography")
    from tulip.control import Ed25519Signer

    signer = Ed25519Signer.generate(key_id="k1")
    trail = AuditTrail(signer=signer)
    trail.record("gate", {"n": 0})
    trail.use_signer(None)
    trail.record("gate", {"n": 1})
    keys = {"k1": signer.public_key_pem()}
    report = check_jsonl(trail.export_jsonl(), keys=keys)
    assert (report.ok, report.broken_at) == (False, 1)
    assert "not signed" in report.problem
