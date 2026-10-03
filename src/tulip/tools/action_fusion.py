# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Action Fusion: a file change and the command that checks it, in one call.

A coding agent's most common pair of turns is *edit a file*, then *run the
test that covers it*. The second turn is not a decision the model needs to be
asked for — it decided when it made the edit — yet it costs a whole model
round trip: the full context sent again, a short reply, a tool call. Fusion
lets a file-changing tool take an optional ``then_run`` argument, run that
command once the change has landed, and hand back one observation holding
both: the diff and the command's exit code and output.

The idea is SoL-Pi's (NVIDIA, arXiv 2609.20519, MIT), where Action Fusion alone
cut tokens 12% and raised the benchmark score. This module is a Python
reimplementation of the mechanism, not a port of its code. What it keeps,
because each closes a real hole:

- **The change must have succeeded.** A failed edit skips the command: running
  the tests against a file the model believes it changed and did not is worse
  than not running them.
- **The file must still be what was written.** The SHA-256 of each written
  file is compared with the text the tool wrote just before the command runs.
  A formatter, a watcher or a concurrent call that touched the file in between
  means the command would test something the model has not seen, so it is
  skipped and the result says why.
- **One fused call per file at a time.** :class:`FileLocks` serialises calls
  on the same canonical path — the change and its command together — so two
  fused edits of one file never interleave their writes and their tests.

What this module does *not* do is run the command. The host runs it through
its own shell path, so the gate, the audit record, the hooks, the output caps
and the timeout ceiling a standalone shell call gets apply unchanged. A
second way to run a command that skipped any of those would make ``then_run``
the way around governance.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import threading
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tulip.observability.mechanisms import ACTION_FUSION, record_mechanism


if TYPE_CHECKING:
    from tulip.tools.decorator import Tool


#: The argument a fusable tool takes.
THEN_RUN = "then_run"

#: The ledger name fused calls are recorded under.
MECHANISM = ACTION_FUSION

#: Markers that open the command's part of a fused result. Stable strings, so
#: a completion check or a transcript reader can tell a command that ran from
#: one that was skipped without parsing prose.
RAN = "[then_run]"
SKIPPED = "[then_run skipped]"

#: The JSON Schema of ``then_run``, as the model sees it.
THEN_RUN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "A command to run right after this change succeeds — usually the "
        "narrowest test or check that covers it, e.g. "
        '{"command": "pytest tests/test_parser.py -q"}. Its exit code and '
        "output come back with the diff, so no separate shell call is needed. "
        "Skipped if the change fails; a failing command does not undo the change."
    ),
    "properties": {
        "command": {"type": "string", "description": "The shell command to run."},
        "timeout": {
            "type": "integer",
            "description": "Seconds to wait for it (the shell tool's default and ceiling apply).",
        },
    },
    "required": ["command"],
    "additionalProperties": False,
}

#: The sentence a fusable tool's description gains.
THEN_RUN_NOTE = (
    "Pass then_run to run the test or check that covers this change in the "
    "same call; the result holds the diff and the command's output."
)


class ThenRunError(ValueError):
    """``then_run`` was given in a shape that names no command."""


@dataclass(frozen=True)
class ThenRun:
    """The command a fused call runs once its change has landed."""

    command: str
    timeout: int | None = None

    @classmethod
    def parse(cls, value: Any) -> ThenRun | None:
        """``value`` as the model sent it, or ``None`` when it asked for nothing.

        A bare string is taken as the command: models that drop the object
        wrapper mean the same thing, and refusing them costs a round trip,
        which is what fusion exists to save.

        Raises:
            ThenRunError: the value is present but names no command.
        """
        if value is None or value in ({}, ""):
            return None
        if isinstance(value, str):
            return cls(value.strip()) if value.strip() else None
        if not isinstance(value, Mapping):
            raise ThenRunError(
                f'then_run must be an object like {{"command": "..."}}, not {type(value).__name__}'
            )
        command = value.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ThenRunError('then_run needs a "command" string')
        timeout = value.get("timeout")
        if timeout is None:
            return cls(command.strip())
        try:
            seconds = int(timeout)
        except (TypeError, ValueError):
            raise ThenRunError("then_run.timeout must be a whole number of seconds") from None
        return cls(command.strip(), seconds if seconds > 0 else None)

    def arguments(self) -> dict[str, Any]:
        """The command as a shell tool's arguments — what a hook is shown."""
        out: dict[str, Any] = {"command": self.command}
        if self.timeout is not None:
            out["timeout"] = self.timeout
        return out


def fused_command(arguments: Mapping[str, Any]) -> str | None:
    """The command a tool call's arguments fuse onto its change, if any."""
    try:
        spec = ThenRun.parse(arguments.get(THEN_RUN))
    except ThenRunError:
        return None
    return spec.command if spec is not None else None


# ------------------------------------------------------------ the hash guard --


def sha256_text(text: str) -> str:
    """The digest of ``text`` as a tool writes it: UTF-8, line endings as given."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    """The digest of the file at ``path``, or ``None`` when it cannot be read."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def changed_since_write(written: Mapping[Path, str | None]) -> list[str]:
    """Each written file that is no longer what was written, with how.

    ``written`` maps each path to the text written there, or ``None`` for a
    file the change deleted. An empty list means every file is as the change
    left it.
    """
    problems: list[str] = []
    for path, text in written.items():
        if text is None:
            if path.exists():
                problems.append(f"{path} was deleted by the change and exists again")
            continue
        digest = sha256_file(path)
        if digest is None:
            problems.append(f"{path} can no longer be read")
        elif digest != sha256_text(text):
            problems.append(f"{path} changed after it was written")
    return problems


# ------------------------------------------------------------ per-file locks --


@dataclass
class _Held:
    lock: threading.Lock = field(default_factory=threading.Lock)
    users: int = 0


class FileLocks:
    """One lock per canonical file path, for sync and async callers alike.

    A thread lock rather than an :class:`asyncio.Lock`: the tools that take
    these run on worker threads (a sync tool body) as often as on the event
    loop, and an asyncio lock serialises only callers on its own loop. The
    async form waits for the same lock off the loop.

    Paths are canonicalised (symlinks resolved, ``..`` collapsed), so
    ``src/a.py`` and ``./src/../src/a.py`` are one file. Several paths are
    taken in sorted order, so two calls that touch the same pair cannot
    deadlock. An entry is dropped once nobody holds or waits for it.
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._entries: dict[str, _Held] = {}

    @staticmethod
    def key(path: str | Path) -> str:
        """The canonical name a path is locked under."""
        return str(Path(path).expanduser().resolve())

    def _take(self, keys: list[str]) -> list[_Held]:
        with self._guard:
            held = []
            for key in keys:
                entry = self._entries.setdefault(key, _Held())
                entry.users += 1
                held.append(entry)
            return held

    def _drop(self, keys: list[str], held: list[_Held]) -> None:
        with self._guard:
            for key, entry in zip(keys, held, strict=True):
                entry.users -= 1
                if entry.users == 0 and self._entries.get(key) is entry:
                    del self._entries[key]

    def held(self) -> int:
        """How many paths are locked or waited on now."""
        with self._guard:
            return len(self._entries)

    @contextlib.contextmanager
    def hold(self, *paths: str | Path) -> Iterator[None]:
        """Hold every path's lock for the block, blocking until each is free."""
        keys = sorted({self.key(p) for p in paths})
        held = self._take(keys)
        acquired: list[threading.Lock] = []
        try:
            for entry in held:
                entry.lock.acquire()
                acquired.append(entry.lock)
            yield
        finally:
            for lock in reversed(acquired):
                lock.release()
            self._drop(keys, held)

    @contextlib.asynccontextmanager
    async def ahold(self, *paths: str | Path) -> AsyncIterator[None]:
        """:meth:`hold` for a coroutine: waits off the event loop."""
        keys = sorted({self.key(p) for p in paths})
        held = self._take(keys)
        acquired: list[threading.Lock] = []
        try:
            for entry in held:
                if not entry.lock.acquire(blocking=False):
                    await asyncio.to_thread(entry.lock.acquire)
                acquired.append(entry.lock)
            yield
        finally:
            for lock in reversed(acquired):
                lock.release()
            self._drop(keys, held)


#: The process's file locks. Every fusable tool in one process shares them, so
#: a fused edit in a subagent waits for one in its parent.
FILE_LOCKS = FileLocks()


# ------------------------------------------------------------------ fusion --


@dataclass(frozen=True)
class Mutation:
    """What a file-changing tool did, in the terms fusion needs.

    ``report`` is what the tool returns on its own. ``written`` maps each path
    the change wrote to the exact text written (``None``: deleted); it is
    empty when the change did not happen.
    """

    report: str
    written: Mapping[Path, str | None] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.written)


@dataclass(frozen=True)
class CommandOutcome:
    """What running the fused command produced.

    ``text`` is what the host's shell tool would have returned for the
    command, notes from its hooks included. ``exit_code`` is ``None`` when the
    command did not finish (moved to the background, killed at its timeout)
    or was refused before it ran — ``refused`` says which. ``command`` is set
    when what ran is not what the model asked for (a hook rewrote it), so the
    result shows the command that produced the output.
    """

    text: str
    exit_code: int | None = None
    refused: bool = False
    #: The command as it actually ran, when the host's hooks rewrote it.
    command: str | None = None


def fuse(
    mutate: Callable[[], Mutation],
    then_run: ThenRun | None,
    run: Callable[[ThenRun], CommandOutcome],
    *,
    paths: tuple[str | Path, ...],
    locks: FileLocks = FILE_LOCKS,
) -> str:
    """Make a change and, when asked, run its command; one combined result.

    ``mutate`` makes the change. ``run`` runs the command through the host's
    shell path — the same gate, hooks and limits as its shell tool. Both run
    under ``paths``' locks, held for the whole call. ``paths`` should name
    every file the change may write; a path that is not known up front (a
    patch's rename target) is still covered by the hash guard, which checks
    whatever ``mutate`` reports it wrote.
    """
    with locks.hold(*paths):
        mutation = mutate()
        if then_run is None:
            return mutation.report
        if not mutation.ok:
            _record("edit_failed", then_run)
            return f"{mutation.report}\n\n{SKIPPED} the change did not succeed, so `{then_run.command}` was not run."
        moved = changed_since_write(mutation.written)
        if moved:
            _record("file_changed", then_run, detail={"why": moved})
            return (
                f"{mutation.report}\n\n{SKIPPED} {'; '.join(moved)}, so `{then_run.command}` "
                "was not run — it would have checked something other than this change. "
                "Read the file again before running it."
            )
        outcome = run(then_run)
    if outcome.refused:
        _record("refused", then_run)
        return f"{mutation.report}\n\n{SKIPPED} {outcome.text}\nThe change stands."
    ran = outcome.command or then_run.command
    _record(
        "ran" if outcome.exit_code is not None else "unfinished",
        ThenRun(ran, then_run.timeout),
        steps_saved=1,
        detail={"exit_code": outcome.exit_code},
    )
    return f"{mutation.report}\n\n{RAN} $ {ran}\n{outcome.text}"


def refuse_disabled(report: str, then_run: ThenRun) -> str:
    """The result when the host has fusion switched off and the model sent then_run anyway."""
    _record("disabled", then_run)
    return (
        f"{report}\n\n{SKIPPED} then_run is switched off in this session, so "
        f"`{then_run.command}` was not run. Run it with the shell tool."
    )


def _record(
    outcome: str,
    then_run: ThenRun,
    *,
    steps_saved: int = 0,
    detail: Mapping[str, Any] | None = None,
) -> None:
    record_mechanism(
        MECHANISM,
        outcome=outcome,
        steps_saved=steps_saved,
        detail={"command": then_run.command[:200], **(detail or {})},
    )


# ------------------------------------------------------------ tool schemas --


def fusable(tool: Tool, *, enabled: bool = True) -> Tool:
    """``tool`` as the model should see it: with ``then_run`` described, or without it.

    The tool's function takes ``then_run`` either way; this decides only
    whether the model is offered it. Switched off, the property is removed and
    the description says nothing about it, so an ablation run sees exactly the
    tool it would have seen before fusion existed.
    """
    parameters = dict(tool.parameters)
    properties = dict(parameters.get("properties") or {})
    required = [name for name in parameters.get("required") or [] if name != THEN_RUN]
    description = tool.description
    if enabled:
        properties[THEN_RUN] = dict(THEN_RUN_SCHEMA)
        if THEN_RUN_NOTE not in description:
            description = f"{description.rstrip()}\n\n{THEN_RUN_NOTE}"
    else:
        properties.pop(THEN_RUN, None)
    parameters["properties"] = properties
    parameters["required"] = required
    return tool.model_copy(update={"parameters": parameters, "description": description})


__all__ = [
    "FILE_LOCKS",
    "MECHANISM",
    "RAN",
    "SKIPPED",
    "THEN_RUN",
    "THEN_RUN_NOTE",
    "THEN_RUN_SCHEMA",
    "CommandOutcome",
    "FileLocks",
    "Mutation",
    "ThenRun",
    "ThenRunError",
    "changed_since_write",
    "fuse",
    "fusable",
    "fused_command",
    "refuse_disabled",
    "sha256_file",
    "sha256_text",
]
