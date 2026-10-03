# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""A per-run ledger of the harness mechanisms that fired, for ablations.

A coding harness is a stack of small mechanisms — fused tool calls, the
completion check sending a run back, leaked tool calls recovered, context
compaction, loop warnings. Each one is meant to make runs cheaper or better,
and the only way to know which one does is to switch it off and compare. That
needs every run to say which mechanisms fired, how often, what they saved and
how each attempt ended, in a form two runs can be diffed in.

:class:`MechanismLedger` is that record. A mechanism reports with
:func:`record_mechanism`, which finds the ledger bound to the current context
(:func:`bind_ledger`) and does nothing when there is none — so instrumenting a
mechanism costs one line and no plumbing, and a program that never binds a
ledger pays nothing. Tool bodies on worker threads see the binding, because
the SDK runs them under a copy of the caller's context. A subagent started
from a run inherits its ledger too, so its mechanisms count toward the run
that started it.

Mechanisms that already announce themselves on the event stream are recorded
from the stream instead (:meth:`MechanismLedger.observe`): the completion
check's continuations, compactions and tool-loop warnings. Only top-level
events are seen there; a subagent's arrive wrapped and are not unwrapped.

Each record is one JSON line when the ledger has a path::

    {
        "ts": "2026-10-03T12:00:00+00:00",
        "run": "r-1",
        "mechanism": "action_fusion",
        "triggered": true,
        "outcome": "ran",
        "steps_saved": 1,
        "tokens_saved": null,
        "bytes_saved": null,
        "detail": {"command": "pytest -q", "exit_code": 0},
    }

and :meth:`MechanismLedger.summary` folds them into counters per mechanism.
``tokens_saved`` is an estimate, and only where the mechanism can make one
honestly (a compaction knows its before and after). For a mechanism that saves
whole model calls, the summary estimates tokens from the run's average input
per call when the caller passes it.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

#: Names the SDK's own mechanisms record under.
ACTION_FUSION = "action_fusion"
COMPLETION_CHECK = "completion_check"
LEAKED_TOOL_CALLS = "leaked_tool_call_recovery"
COMPACTION = "compaction"
LOOP_WARNING = "loop_warning"
OBSERVATION_PACK = "observation_pack"


@dataclass(frozen=True)
class MechanismRecord:
    """One time a mechanism fired, or was asked to and did not."""

    mechanism: str
    triggered: bool = True
    outcome: str = ""
    steps_saved: int = 0
    tokens_saved: int | None = None
    bytes_saved: int | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    run: str = ""

    def as_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["detail"] = dict(self.detail)
        return row


class MechanismLedger:
    """The mechanisms one run used, in memory and optionally as JSONL on disk.

    Args:
        path: Where to append one JSON line per record. ``None`` keeps the
            records in memory only. A failed write is logged and the record
            kept: losing a ledger line must never fail the run it describes.
        run: An id stamped on every record, so one file can hold many runs.
        enabled: Mechanism name to whether it is switched on in this run.
            Listed mechanisms appear in :meth:`summary` even when nothing
            fired, so an ablation's "off" run says so instead of being silent.
    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        run: str = "",
        enabled: Mapping[str, bool] | None = None,
    ) -> None:
        self.path = path
        self.run = run
        self.enabled: dict[str, bool] = dict(enabled or {})
        self.records: list[MechanismRecord] = []
        self._lock = threading.Lock()

    def record(  # noqa: PLR0913 — every field is an independent, optional fact
        self,
        mechanism: str,
        *,
        triggered: bool = True,
        outcome: str = "",
        steps_saved: int = 0,
        tokens_saved: int | None = None,
        bytes_saved: int | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> MechanismRecord:
        """Add one record, and append it to the file when there is one."""
        entry = MechanismRecord(
            mechanism=mechanism,
            triggered=triggered,
            outcome=outcome,
            steps_saved=steps_saved,
            tokens_saved=tokens_saved,
            bytes_saved=bytes_saved,
            detail=dict(detail or {}),
            run=self.run,
        )
        with self._lock:
            self.records.append(entry)
            if self.path is not None:
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with self.path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(entry.as_dict(), default=str) + "\n")
                except OSError:
                    logger.warning("mechanism ledger: could not write %s", self.path, exc_info=True)
        return entry

    def observe(self, event: Any) -> MechanismRecord | None:
        """Record the mechanism a loop event announces, if it announces one."""
        name = type(event).__name__
        if name == "FinalAnswerVerificationEvent" and getattr(event, "continuation", False):
            return self.record(
                COMPLETION_CHECK,
                outcome=str(getattr(event, "reason", None) or "continuation"),
                detail={
                    "attempt": getattr(event, "attempt", None),
                    "continuing": bool(getattr(event, "replanning", False)),
                },
            )
        if name == "CompactionEvent":
            before = int(getattr(event, "tokens_before", 0) or 0)
            after = int(getattr(event, "tokens_after", 0) or 0)
            exhausted = bool(getattr(event, "exhausted", False))
            return self.record(
                COMPACTION,
                outcome="exhausted" if exhausted else str(getattr(event, "stage", "")),
                tokens_saved=max(0, before - after),
                detail={
                    "tokens_before": before,
                    "tokens_after": after,
                    "messages_before": getattr(event, "messages_before", None),
                    "messages_after": getattr(event, "messages_after", None),
                },
            )
        if name == "CustomEvent" and getattr(event, "name", None) == "tool_loop_warning":
            data = getattr(event, "data", None)
            return self.record(
                LOOP_WARNING,
                outcome="warned",
                detail=dict(data) if isinstance(data, Mapping) else {},
            )
        return None

    def summary(self, *, tokens_per_step: float | None = None) -> dict[str, dict[str, Any]]:
        """Counters per mechanism: how often, how it ended, what it saved.

        ``tokens_per_step`` — the run's average input tokens per model call —
        turns ``steps_saved`` into a token estimate for records that made
        none of their own. The figure is labelled an estimate because it is
        one: a skipped call would have sent a context of about that size.
        """
        with self._lock:
            records = list(self.records)
        out: dict[str, dict[str, Any]] = {
            name: _empty(on) for name, on in sorted(self.enabled.items())
        }
        for entry in records:
            row = out.setdefault(entry.mechanism, _empty(self.enabled.get(entry.mechanism)))
            row["events"] += 1
            row["triggered"] += int(entry.triggered)
            row["steps_saved"] += entry.steps_saved
            if entry.outcome:
                row["outcomes"][entry.outcome] = row["outcomes"].get(entry.outcome, 0) + 1
            if entry.bytes_saved is not None:
                row["bytes_saved"] += entry.bytes_saved
            if entry.tokens_saved is not None:
                row["tokens_saved_est"] += entry.tokens_saved
            elif entry.steps_saved and tokens_per_step:
                row["tokens_saved_est"] += round(entry.steps_saved * tokens_per_step)
        return out


def _empty(enabled: bool | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "events": 0,
        "triggered": 0,
        "steps_saved": 0,
        "tokens_saved_est": 0,
        "bytes_saved": 0,
        "outcomes": {},
    }
    if enabled is not None:
        row["enabled"] = enabled
    return row


_LEDGER: ContextVar[MechanismLedger | None] = ContextVar("tulip_mechanism_ledger", default=None)


def bind_ledger(ledger: MechanismLedger | None) -> None:
    """Make ``ledger`` the one this context, and what it starts, records into."""
    _LEDGER.set(ledger)


def current_ledger() -> MechanismLedger | None:
    """The ledger bound to this context, if any."""
    return _LEDGER.get()


def record_mechanism(  # noqa: PLR0913 — mirrors MechanismLedger.record
    mechanism: str,
    *,
    triggered: bool = True,
    outcome: str = "",
    steps_saved: int = 0,
    tokens_saved: int | None = None,
    bytes_saved: int | None = None,
    detail: Mapping[str, Any] | None = None,
) -> MechanismRecord | None:
    """Record into the bound ledger; a no-op when none is bound."""
    ledger = _LEDGER.get()
    if ledger is None:
        return None
    return ledger.record(
        mechanism,
        triggered=triggered,
        outcome=outcome,
        steps_saved=steps_saved,
        tokens_saved=tokens_saved,
        bytes_saved=bytes_saved,
        detail=detail,
    )


__all__ = [
    "ACTION_FUSION",
    "COMPACTION",
    "COMPLETION_CHECK",
    "LEAKED_TOOL_CALLS",
    "LOOP_WARNING",
    "OBSERVATION_PACK",
    "MechanismLedger",
    "MechanismRecord",
    "bind_ledger",
    "current_ledger",
    "record_mechanism",
]
