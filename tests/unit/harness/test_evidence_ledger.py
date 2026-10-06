# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Exec evidence and the read ledger."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from tulip.harness import ExecRecord, MemoryBackend, ReadLedger
from tulip.harness.backend import ExecResult, FileStat
from tulip.harness.evidence import exec_record, publish
from tulip.harness.ledger import _same
from tulip.observability.emit import EV_HARNESS_EXEC


def test_a_record_carries_digests_not_text() -> None:
    record = exec_record(
        "echo $TOKEN",
        ExecResult(exit_code=0, output=b"s3cret\n", duration_s=0.12345),
        backend_label="local",
    )
    assert record.command_sha256 == hashlib.sha256(b"echo $TOKEN").hexdigest()
    assert record.output_sha256 == hashlib.sha256(b"s3cret\n").hexdigest()
    assert record.output_bytes == 7
    assert record.duration_s == 0.123
    assert "s3cret" not in repr(record)
    assert "TOKEN" not in repr(record)


def test_publish_emits_on_the_bus_and_to_the_sink(monkeypatch: pytest.MonkeyPatch) -> None:
    emitted: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        "tulip.harness.evidence.emit_sync", lambda kind, **data: emitted.append((kind, data))
    )
    sunk: list[ExecRecord] = []
    record = exec_record("ls", ExecResult(exit_code=0, output=b""), backend_label="x")
    publish(record, sunk.append)
    publish(record)
    assert sunk == [record]
    assert emitted[0][0] == EV_HARNESS_EXEC == "harness.exec"
    assert emitted[0][1]["backend_label"] == "x"


def test_without_a_hash_size_and_time_decide() -> None:
    assert _same(FileStat(1, 2.0), FileStat(1, 2.0))
    assert not _same(FileStat(1, 2.0), FileStat(1, 3.0))
    assert _same(FileStat(1, 2.0, "a"), FileStat(1, 9.0, "a"))


def test_the_ledger_can_forget_and_ignores_what_is_not_there() -> None:
    backend = MemoryBackend({"a.txt": "x"})
    ledger = ReadLedger()
    ledger.note(backend, "ghost.txt")  # nothing to note
    ledger.note(backend, "a.txt")
    assert ledger.unseen(backend, "a.txt", "a.txt") is None
    ledger.forget()
    assert "has not been read" in str(ledger.unseen(backend, "a.txt", "a.txt"))
    assert ledger.unseen(backend, "ghost.txt", "ghost.txt") is None


def test_a_path_that_is_not_text_is_invalid() -> None:
    from tulip.harness import BackendError

    with pytest.raises(BackendError):
        MemoryBackend().resolve(42)  # type: ignore[arg-type]
