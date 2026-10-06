# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""What every harness tool shares: the workspace, the settings, the helpers.

A :class:`HarnessContext` is one session's worth of state — the backend, the
read ledger, the plan, where each background command was last read up to.
The tool factories close over it, so two harnesses in one process never see
each other's reads or jobs.

The file-changing tools are written as a *plan* and a *commit*. Planning
reads the workspace and works out every file's before and after, or the
reason it cannot; committing writes them. The split is what lets a gate show
the exact diff before anything is written (:meth:`Harness.preview
<tulip.harness.toolset.Harness.preview>`) without the tool body ever calling
the gate itself.
"""

from __future__ import annotations

import difflib
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from tulip.deepagent.backends.protocol import BackendError
from tulip.deepagent.todos import TodoState
from tulip.harness.backend import WorkspaceBackend
from tulip.harness.evidence import ExecRecord
from tulip.harness.ledger import ReadLedger
from tulip.tools.output import ToolOutput
from tulip.tools.result_storage import ToolResultStore
from tulip.tools.text_edit import EditOutcome


__all__ = [
    "FileChange",
    "HarnessConfig",
    "HarnessContext",
    "Plan",
    "cap_output",
    "describe",
    "diff",
    "refusal",
]


@dataclass(frozen=True)
class HarnessConfig:
    """How the harness tools behave. The defaults are tulip-code's.

    Attributes:
        require_read: Refuse a change to a file the agent has not read, or
            that changed since it read it.
        bash_timeout: Seconds ``bash`` waits when the model names none.
        bash_max_timeout: The most a model may ask for. A model that sets
            ``timeout=86400`` on a hung test runner turns one bad guess into a
            session that never ends.
        on_timeout: What a command that outlives its timeout does.
            ``"background"`` keeps it running under a handle the model can
            read or kill — killing a build that was nearly done throws away
            minutes of work. ``"kill"`` stops it and everything it started.
        output_chars: What one command hands back to the model, in
            characters, kept as a head and a tail: the line that matters in a
            test run — the summary, the failing assertion — is at the end.
        output_head_chars: How much of that is the head.
        vision: Whether the model can see images. ``read`` attaches an image
            only when it can; otherwise it says so in a sentence the model can
            act on. ``tulip.models.profiles.profile_for(model).vision`` is the
            usual source.
        result_store: Where the whole output of a command too long to show
            goes, so it can be fetched back by key.
        on_exec: Receives an :class:`~tulip.harness.evidence.ExecRecord` for
            every command, besides the event bus.
        on_change: Receives each :class:`FileChange` just before it is
            written — an undo journal's hook.
        environment: The ``environment`` label on every action the harness
            derives for a gate.
    """

    require_read: bool = True
    bash_timeout: int = 120
    bash_max_timeout: int = 600
    on_timeout: Literal["background", "kill"] = "background"
    output_chars: int = 30_000
    output_head_chars: int = 8_000
    vision: bool = False
    result_store: ToolResultStore | None = None
    on_exec: Callable[[ExecRecord], None] | None = None
    on_change: Callable[[FileChange], None] | None = None
    environment: str = "unknown"


@dataclass(frozen=True)
class FileChange:
    """One file's change: what it was, what it becomes.

    ``before`` is ``None`` for a file being created, ``after`` is ``None``
    for one being deleted. ``path`` is resolved; ``shown`` is how the model
    wrote it.
    """

    path: str
    shown: str
    before: str | None
    after: str | None
    action: str


@dataclass
class Plan:
    """A file change worked out, not yet written: the changes and the report."""

    changes: list[FileChange]
    report: str
    preview: str


@dataclass
class HarnessContext:
    """One session's state, shared by the tools built over it."""

    backend: WorkspaceBackend
    config: HarnessConfig
    ledger: ReadLedger
    todos: TodoState = field(default_factory=TodoState)
    #: Where each background command was last read up to, so "what is new"
    #: needs no bookkeeping by the model.
    cursors: dict[str, int] = field(default_factory=dict)
    #: Held for every file change, so two calls cannot interleave a read and
    #: a write of the same file.
    write_lock: threading.Lock = field(default_factory=threading.Lock)
    #: Counts spills when no tool context names the run.
    spills: int = 0

    def commit(self, plan: Plan) -> None:
        """Write every change, putting back what was done if one fails."""
        done: list[FileChange] = []
        try:
            for change in plan.changes:
                if self.config.on_change is not None:
                    self.config.on_change(change)
                if change.after is None:
                    self.backend.remove(change.path)
                else:
                    self.backend.write_bytes(change.path, change.after.encode("utf-8"))
                done.append(change)
        except BackendError:
            for change in reversed(done):
                if change.before is None:
                    self.backend.remove(change.path)
                else:
                    self.backend.write_bytes(change.path, change.before.encode("utf-8"))
            raise
        for change in plan.changes:
            if change.after is not None:
                # The agent's own change does not make the file stale to it.
                self.ledger.note(self.backend, change.path)

    def read_text(self, path: str) -> str:
        """A file's text with its line endings as they are.

        Bytes are decoded without newline translation: turning CRLF into LF
        and writing it back converts every line of a Windows file in an edit
        that touched one.
        """
        return self.backend.read_bytes(path).decode("utf-8", errors="replace")


def refusal(message: str) -> ToolOutput:
    """A tool result that reports a failure the model should act on."""
    return ToolOutput(message, is_error=True)


def outside(path: str, root: str) -> ToolOutput:
    """The refusal for a path that resolves outside the workspace."""
    return refusal(f"path escapes the workspace ({root}): {path}")


def diff(old: str, new: str) -> str:
    """A short unified diff — for the person at the gate, and for the model after."""
    lines = list(
        difflib.unified_diff(
            old.splitlines(), new.splitlines(), lineterm="", n=2, fromfile="before", tofile="after"
        )
    )
    if not lines:
        return "(no textual change)"
    body = "\n".join(lines[:40])
    return body + (f"\n... {len(lines) - 40} more diff lines" if len(lines) > 40 else "")


def crlf(text: str) -> bool:
    """Whether ``text`` mostly ends its lines with CRLF."""
    count = text.count("\r\n")
    return count > 0 and count * 2 >= text.count("\n")


#: How each fuzzy reading is described to the model. It needs to know its
#: quote was off — and how — so the next one is exact.
_HOW = {
    "line_trimmed": "ignoring indentation and trailing whitespace",
    "whitespace_normalized": "ignoring the spacing between tokens",
    "escape_normalized": "after removing one level of backslash escaping",
    "block_anchor": "by its first and last lines — the lines between differed",
}


def describe(outcome: EditOutcome) -> str:
    """How an edit was placed, or empty when it matched exactly, once."""
    notes: list[str] = []
    if outcome.replacements > 1:
        notes.append(f"{outcome.replacements} places")
    if not outcome.exact:
        how = _HOW.get(outcome.strategy, f"by {outcome.strategy}")
        notes.append(
            f"old text did not match exactly; matched {how} at line {outcome.lines[0]}"
            " — check the diff is what you meant"
        )
    if outcome.crlf:
        notes.append("kept the file's CRLF line endings")
    return f" ({'; '.join(notes)})" if notes else ""


def cap_output(text: str, limit: int = 30_000, head: int = 8_000) -> str:
    """Keep the start and the end of ``text``, and say how much went missing."""
    if len(text) <= limit:
        return text
    tail = limit - head
    dropped = len(text) - limit
    return (
        f"{text[:head]}\n... {dropped:,} characters truncated from the middle ...\n{text[-tail:]}"
    )
