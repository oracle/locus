# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Tool-loop detection: a run repeating itself, told apart from a run re-reading.

A model that is stuck calls the same tool with the same arguments, gets the
same result, and calls it again. A model that is working also repeats calls —
it re-reads a test file it is about to edit, greps again after an edit, lists
a directory it listed ten steps ago — and stopping that run throws its work
away. The difference is what happens *between* the repeats:

- a **loop** is the same step back to back — the same calls (name and
  arguments) and the same results — with nothing else in between, or the same
  short cycle of steps (A, B, A, B, …) repeated whole;
- **progress** is anything else: a repeat with another call in between, a
  repeat whose result changed (the file was edited, the job finished), or a
  repeat of a read separated by other work.

Reads are cheap and re-reading is normal, so a step made only of read-only
tools (:data:`DEFAULT_READ_ONLY_TOOLS`) needs one repeat more than the
threshold before it counts.

opencode's ``doom_loop`` counts three identical calls inside one assistant
message, which misses the common case (one call per step, repeated across
steps); counting a tool's name across the whole run, as a naive detector does,
stops real work. This module counts consecutive steps, by name, arguments and
result.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final


if TYPE_CHECKING:
    from tulip.core.messages import ToolCall
    from tulip.core.state import ReasoningStep, ToolExecution


__all__ = [
    "DEFAULT_READ_ONLY_TOOLS",
    "MAX_CYCLE",
    "ToolLoop",
    "call_signature",
    "detect_tool_loop",
    "warning_text",
]

#: Tools whose repeats are normal work: reading, listing and searching change
#: nothing, and a model re-reads what it is about to edit. Names as the SDK's
#: own tools and the common coding-agent tool sets spell them.
DEFAULT_READ_ONLY_TOOLS: Final[frozenset[str]] = frozenset(
    {
        "read",
        "read_file",
        "view",
        "cat",
        "ls",
        "list_dir",
        "list_directory",
        "list_files",
        "glob",
        "grep",
        "find_files",
        "search_files",
        "todo_read",
    }
)

#: The longest cycle looked for. Longer cycles exist, but a model repeating a
#: four-step routine with identical results is rare and a false positive there
#: costs a working run.
MAX_CYCLE: Final[int] = 3

#: Characters of a call's arguments quoted back in a warning.
_ARGS_SHOWN: Final[int] = 80


def call_signature(tc: ToolCall) -> tuple[str, str]:
    """Stable ``(name, arguments)`` for one call.

    JSON with ``sort_keys=True`` canonicalises dict argument order, so
    ``{"a": 1, "b": 2}`` matches ``{"b": 2, "a": 1}``; arguments json cannot
    serialise fall back to a sorted-items repr.
    """
    try:
        canonical = json.dumps(tc.arguments, sort_keys=True, default=str)
    except (TypeError, ValueError):
        canonical = repr(sorted(tc.arguments.items()))
    return (tc.name, canonical)


def _result_digest(execution: ToolExecution | None) -> str | None:
    """A digest of what a call returned; ``None`` when no result was recorded."""
    if execution is None:
        return None
    text = json.dumps([execution.result, execution.error], default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class _Step:
    """One step reduced to what makes two steps the same."""

    key: tuple[tuple[str, str, str | None], ...]
    names: frozenset[str]
    shown: str


def _step_of(step: ReasoningStep) -> _Step | None:
    """The comparable form of ``step``; ``None`` for a step that called no tool."""
    if not step.tool_calls:
        return None
    by_id: dict[str, ToolExecution] = {e.tool_call_id: e for e in step.tool_results}
    parts: list[tuple[str, str, str | None]] = []
    for tc in step.tool_calls:
        name, args = call_signature(tc)
        parts.append((name, args, _result_digest(by_id.get(tc.id))))
    # Sorted: parallel calls in one step have no order the model meant.
    # Duplicates inside one step collapse — several identical calls fanned
    # out at once are a model being redundant, not a model stuck.
    key = tuple(sorted(set(parts)))
    shown = ", ".join(f"{name}({_clip(args)})" for name, args, _ in key)
    return _Step(key=key, names=frozenset(name for name, _, _ in key), shown=shown)


def _clip(args: str) -> str:
    return args if len(args) <= _ARGS_SHOWN else args[: _ARGS_SHOWN - 1] + "…"


@dataclass(frozen=True)
class ToolLoop:
    """A repeating pattern at the end of a run's steps.

    Attributes:
        signature: Stable id of the repeating unit; the same loop seen again
            later in the run has the same signature.
        steps: The calls of each step in the unit, as text, for a warning.
        period: Steps in the unit: 1 for one step repeated, 2 or more for a
            cycle.
        repeats: Consecutive times the unit occurred, ending at the last step.
        read_only: Whether every call in the unit is a read-only tool.
        threshold: The repeats that made it a loop. The run is warned when
            ``repeats`` reaches it and stopped when the unit repeats once more
            (:attr:`past_warning`).
    """

    signature: str
    steps: tuple[str, ...]
    period: int
    repeats: int
    read_only: bool
    threshold: int

    @property
    def past_warning(self) -> bool:
        """Whether the loop went on after the point where it was warned about."""
        return self.repeats > self.threshold

    def as_dict(self) -> dict[str, Any]:
        """The loop as JSON-ready data, for the event that announces a warning."""
        return {
            "signature": self.signature,
            "steps": list(self.steps),
            "period": self.period,
            "repeats": self.repeats,
            "read_only": self.read_only,
            "threshold": self.threshold,
        }

    def describe(self) -> str:
        """The loop in one line: ``read({"path": "a.py"}) 4 times in a row``."""
        if self.period == 1:
            return f"{self.steps[0]} {self.repeats} times in a row, with the same result each time"
        cycle = " -> ".join(self.steps)
        return f"the cycle {cycle} {self.repeats} times in a row, with the same results each time"


def _signature(unit: Sequence[_Step]) -> str:
    """Id of a repeating unit, the same for every rotation of a cycle.

    (A, B) seen from its B step is (B, A); both are the one loop, and a
    rotation must not read as a new loop that earns a second warning.
    """
    keys = [list(s.key) for s in unit]
    rotations = [keys[i:] + keys[:i] for i in range(len(keys))]
    payload = json.dumps(min(rotations, key=lambda r: json.dumps(r, default=str)), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _repeats(keys: Sequence[_Step | None], period: int) -> int:
    """How many times the last ``period`` steps occur back to back, ending at the last."""
    unit = keys[-period:]
    count = 1
    end = len(keys) - period
    while end - period >= 0 and keys[end - period : end] == unit:
        count += 1
        end -= period
    return count


def detect_tool_loop(
    steps: Iterable[ReasoningStep],
    *,
    threshold: int = 3,
    read_only_threshold: int | None = None,
    read_only_tools: Iterable[str] = DEFAULT_READ_ONLY_TOOLS,
) -> ToolLoop | None:
    """The loop the run's last steps are in, or ``None``.

    Args:
        steps: The run's reasoning steps, oldest first.
        threshold: Consecutive repeats of a step (or a cycle) that make a loop.
        read_only_threshold: The same for a unit made only of read-only
            tools; ``None`` means ``threshold + 1``.
        read_only_tools: Tool names whose repeats are normal work.

    Returns:
        The shortest repeating unit that reached its threshold, ending at the
        last step; ``None`` when the run is not looping.
    """
    reads = frozenset(read_only_tools)
    read_threshold = read_only_threshold if read_only_threshold is not None else threshold + 1
    keys: list[_Step | None] = [_step_of(s) for s in steps]
    for period in range(1, MAX_CYCLE + 1):
        if len(keys) < period * 2:
            break
        unit = keys[-period:]
        if any(s is None for s in unit):
            # A step without a tool call (the model said something and was
            # sent back) breaks the streak: something other than the call
            # happened in between.
            continue
        present: list[_Step] = [s for s in unit if s is not None]
        if period > 1 and len({s.key for s in present}) < period:
            # (A, A) is period 1 seen twice, and (A, B, A) is not a cycle
            # unit; only a unit of distinct steps is a cycle.
            continue
        read_only = all(s.names <= reads for s in present)
        needed = read_threshold if read_only else threshold
        repeats = _repeats(keys, period)
        if repeats >= needed:
            return ToolLoop(
                signature=_signature(present),
                steps=tuple(s.shown for s in present),
                period=period,
                repeats=repeats,
                read_only=read_only,
                threshold=needed,
            )
    return None


def warning_text(loop: ToolLoop) -> str:
    """The note that tells the model it is repeating itself, before anything stops it."""
    return (
        "[Repeated tool call] You have called "
        f"{loop.describe()}. Calling it again will not produce anything new. "
        "Change approach: use the result you already have, try a different tool "
        "or different arguments, or make the change the task needs. If you repeat "
        "the same call again unchanged, the run will be stopped."
    )
