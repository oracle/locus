# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""The mechanism ledger: records, the JSONL file, counters, and the stream."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from tulip.core.events import CompactionEvent, CustomEvent, FinalAnswerVerificationEvent
from tulip.observability.mechanisms import (
    COMPACTION,
    COMPLETION_CHECK,
    LOOP_WARNING,
    MechanismLedger,
    bind_ledger,
    current_ledger,
    record_mechanism,
)


def test_records_go_to_memory_and_to_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "s1" / "mechanisms.jsonl"
    ledger = MechanismLedger(path, run="r-1")
    ledger.record("action_fusion", outcome="ran", steps_saved=1, detail={"exit_code": 0})
    ledger.record("action_fusion", triggered=False, outcome="disabled", bytes_saved=10)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["outcome"] for r in rows] == ["ran", "disabled"]
    assert rows[0]["run"] == "r-1"
    assert rows[0]["detail"] == {"exit_code": 0}
    assert set(rows[0]) == {
        "ts",
        "run",
        "mechanism",
        "triggered",
        "outcome",
        "steps_saved",
        "tokens_saved",
        "bytes_saved",
        "detail",
    }
    assert len(ledger.records) == 2


def test_a_write_that_fails_keeps_the_record(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("", encoding="utf-8")
    ledger = MechanismLedger(blocker / "mechanisms.jsonl")
    ledger.record("compaction")
    assert len(ledger.records) == 1
    assert "could not write" in caplog.text


def test_summary_counts_each_mechanism_and_lists_switched_off_ones() -> None:
    ledger = MechanismLedger(enabled={"action_fusion": False, "loop_warning": True})
    ledger.record("compaction", outcome="prune", tokens_saved=5000)
    ledger.record("compaction", outcome="prune", tokens_saved=1000)
    ledger.record("leaked_tool_call_recovery", outcome="dsml", bytes_saved=40)
    ledger.record("custom", triggered=False)
    summary = ledger.summary(tokens_per_step=100.0)
    assert summary["action_fusion"] == {
        "enabled": False,
        "events": 0,
        "triggered": 0,
        "steps_saved": 0,
        "tokens_saved_est": 0,
        "bytes_saved": 0,
        "outcomes": {},
    }
    assert summary["compaction"]["events"] == 2
    assert summary["compaction"]["tokens_saved_est"] == 6000
    assert summary["compaction"]["outcomes"] == {"prune": 2}
    assert summary["leaked_tool_call_recovery"]["bytes_saved"] == 40
    assert summary["custom"]["triggered"] == 0
    assert summary["custom"]["outcomes"] == {}
    assert "enabled" not in summary["compaction"]


def test_steps_saved_become_a_token_estimate_only_when_asked() -> None:
    ledger = MechanismLedger()
    ledger.record("action_fusion", steps_saved=2)
    assert ledger.summary()["action_fusion"]["tokens_saved_est"] == 0
    assert ledger.summary(tokens_per_step=1500.4)["action_fusion"]["tokens_saved_est"] == 3001


def test_record_mechanism_is_a_no_op_without_a_bound_ledger() -> None:
    bind_ledger(None)
    assert current_ledger() is None
    assert record_mechanism("x") is None


async def test_a_bound_ledger_is_seen_by_worker_threads_and_tasks() -> None:
    ledger = MechanismLedger()
    bind_ledger(ledger)
    try:
        # A sync tool body runs on a worker thread under a copy of the context.
        await asyncio.to_thread(record_mechanism, "action_fusion", outcome="ran")
        await asyncio.create_task(asyncio.to_thread(record_mechanism, "compaction"))
    finally:
        bind_ledger(None)
    assert [r.mechanism for r in ledger.records] == ["action_fusion", "compaction"]


def test_observe_reads_the_mechanisms_the_loop_announces() -> None:
    ledger = MechanismLedger()
    continuation = FinalAnswerVerificationEvent(
        passed=False, attempt=0, replanning=True, continuation=True, reason="unchecked_edits"
    )
    verdict = FinalAnswerVerificationEvent(passed=True, attempt=0)
    compaction = CompactionEvent(
        iteration=4,
        stage="prune",
        tokens_before=90_000,
        tokens_after=40_000,
        threshold=80_000,
        context_window=128_000,
        messages_before=40,
        messages_after=40,
    )
    exhausted = compaction.model_copy(update={"exhausted": True, "tokens_after": 95_000})
    warning = CustomEvent(name="tool_loop_warning", data={"tool": "bash", "repeats": 3})
    other = CustomEvent(name="budget_nudge", data={})
    for event in (continuation, verdict, compaction, exhausted, warning, other, object()):
        ledger.observe(event)
    summary = ledger.summary()
    assert summary[COMPLETION_CHECK]["outcomes"] == {"unchecked_edits": 1}
    assert ledger.records[0].detail == {"attempt": 0, "continuing": True}
    assert summary[COMPACTION]["outcomes"] == {"prune": 1, "exhausted": 1}
    assert summary[COMPACTION]["tokens_saved_est"] == 50_000
    assert summary[LOOP_WARNING]["events"] == 1
    assert ledger.records[-1].detail == {"tool": "bash", "repeats": 3}
    assert len(ledger.records) == 4
