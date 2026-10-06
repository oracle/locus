# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""``apply_patch`` — the edit tool GPT-family models are trained on.

A model edits best in the format it learned. Claude and most open-weight
models are good at exact search-and-replace, which is what ``edit`` asks
for. OpenAI's models are trained on a patch envelope instead, and asked to
write ``old``/``new`` pairs they paraphrase whitespace and miss — so for
that family this tool replaces ``edit``, ``write`` and ``multi_edit``
(``tulip.models.profiles.profile_for(model).edit_format`` says which)::

    *** Begin Patch
    *** Add File: docs/notes.md
    +first line
    *** Update File: src/app.py
    *** Move to: src/main.py
    @@ def handler():
    -    return None
    +    return 42
    *** Delete File: old.py
    *** End Patch

Every path goes through the same containment check as ``edit``, the whole
patch is worked out before anything is written, and it is all or nothing: a
hunk that does not match, or a write that fails, leaves every file as it
was.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tulip.deepagent.backends.protocol import BackendError
from tulip.harness.tools.common import FileChange, HarnessContext, Plan, diff
from tulip.tools.decorator import Tool, tool


__all__ = ["PatchError", "apply_hunks", "make_apply_patch", "parse", "plan_patch"]

BEGIN = "*** Begin Patch"
END = "*** End Patch"
ADD = "*** Add File: "
UPDATE = "*** Update File: "
DELETE = "*** Delete File: "
MOVE = "*** Move to: "
EOF_MARK = "*** End of File"


class PatchError(ValueError):
    """The patch cannot be applied as written. Nothing was changed."""


@dataclass
class Hunk:
    """One ``@@`` section: the lines it expects, and what replaces them."""

    anchor: str = ""
    old: list[str] = field(default_factory=list)
    new: list[str] = field(default_factory=list)
    at_end: bool = False


@dataclass
class FileOp:
    """One file's part of a patch."""

    kind: str  # "add", "update" or "delete"
    path: str
    move_to: str | None = None
    added: list[str] = field(default_factory=list)
    hunks: list[Hunk] = field(default_factory=list)


def parse(text: str) -> list[FileOp]:  # noqa: PLR0912 - one branch per line kind
    """Read a patch into per-file operations, or raise :class:`PatchError`."""
    lines = text.strip().splitlines()
    if not lines or lines[0].strip() != BEGIN:
        raise PatchError(f"a patch starts with {BEGIN!r}")
    if lines[-1].strip() != END:
        raise PatchError(f"a patch ends with {END!r}")
    ops: list[FileOp] = []
    current: FileOp | None = None
    hunk: Hunk | None = None
    for number, line in enumerate(lines[1:-1], start=2):
        if line.startswith(ADD):
            current, hunk = FileOp("add", line[len(ADD) :].strip()), None
            ops.append(current)
        elif line.startswith(UPDATE):
            current, hunk = FileOp("update", line[len(UPDATE) :].strip()), None
            ops.append(current)
        elif line.startswith(DELETE):
            current, hunk = FileOp("delete", line[len(DELETE) :].strip()), None
            ops.append(current)
        elif current is None:
            raise PatchError(f"line {number}: expected a file header, got {line[:60]!r}")
        elif current.kind == "add":
            if not line.startswith("+"):
                raise PatchError(f"line {number}: every line of an added file starts with '+'")
            current.added.append(line[1:])
        elif current.kind == "delete":
            raise PatchError(f"line {number}: a deleted file takes no body")
        elif line.startswith(MOVE):
            current.move_to = line[len(MOVE) :].strip()
        elif line.startswith("@@"):
            hunk = Hunk(anchor=line[2:].strip())
            current.hunks.append(hunk)
        elif line.strip() == EOF_MARK:
            if hunk is None:
                raise PatchError(f"line {number}: {EOF_MARK!r} outside a hunk")
            hunk.at_end = True
        else:
            if hunk is None:
                # A first hunk without its own ``@@`` line is common; start one.
                hunk = Hunk()
                current.hunks.append(hunk)
            marker, body = (line[:1], line[1:]) if line else (" ", "")
            if marker == " ":
                hunk.old.append(body)
                hunk.new.append(body)
            elif marker == "-":
                hunk.old.append(body)
            elif marker == "+":
                hunk.new.append(body)
            else:
                raise PatchError(
                    f"line {number}: a hunk line starts with ' ', '-' or '+', got {line[:60]!r}"
                )
    if not ops:
        raise PatchError("the patch changes no files")
    for op in ops:
        if op.kind == "update" and not op.hunks and not op.move_to:
            raise PatchError(f"{op.path}: an update with no hunks and no move")
    return ops


def _find(lines: list[str], wanted: list[str], start: int, at_end: bool) -> int:
    """Where ``wanted`` occurs in ``lines`` at or after ``start``, or -1.

    Exact first, then ignoring trailing whitespace, then ignoring indentation
    on both sides — the same three levels the format's reference applier
    uses, so a patch that applies there applies here.
    """
    if not wanted:
        return len(lines) if at_end else start
    for norm in (lambda s: s, str.rstrip, str.strip):
        target = [norm(w) for w in wanted]
        candidates = (
            [len(lines) - len(wanted)] if at_end else range(start, len(lines) - len(wanted) + 1)
        )
        for i in candidates:
            if i >= start and [norm(x) for x in lines[i : i + len(wanted)]] == target:
                return i
    return -1


def apply_hunks(path: str, body: str, hunks: list[Hunk]) -> str:
    """``body`` with every hunk applied in order, or raise :class:`PatchError`."""
    trailing_newline = body.endswith("\n") or not body
    lines = body.splitlines()
    cursor = 0
    for index, hunk in enumerate(hunks, 1):
        if hunk.anchor:
            anchored = _find(lines, [hunk.anchor], cursor, at_end=False)
            if anchored < 0:
                raise PatchError(f"{path}: hunk {index}: no line matching @@ {hunk.anchor!r}")
            cursor = anchored + (0 if hunk.old and hunk.old[0].strip() == hunk.anchor else 1)
        at = _find(lines, hunk.old, cursor, hunk.at_end)
        if at < 0:
            shown = "\n".join(hunk.old[:6])
            raise PatchError(
                f"{path}: hunk {index} does not match the file — re-read it and patch what is "
                f"there now. Expected:\n{shown}"
            )
        lines[at : at + len(hunk.old)] = hunk.new
        cursor = at + len(hunk.new)
    text = "\n".join(lines)
    return text + "\n" if trailing_newline and text else text


def _resolve(h: HarnessContext, path: str) -> str:
    try:
        return h.backend.resolve(path)
    except BackendError as exc:
        raise PatchError(f"path escapes the workspace: {path}") from exc


def _exists(h: HarnessContext, path: str) -> bool | None:
    """True for a file, False for nothing, None for a directory."""
    try:
        return None if h.backend.stat(path).is_dir else True
    except BackendError:
        return False


def _plan_ops(h: HarnessContext, ops: list[FileOp]) -> list[FileChange]:
    """Every file's before and after, or raise. Reads only; writes nothing."""
    planned: list[FileChange] = []
    seen: set[str] = set()
    for op in ops:
        target = _resolve(h, op.path)
        if target in seen:
            raise PatchError(f"{op.path} appears twice in one patch")
        seen.add(target)
        if op.kind == "add":
            if _exists(h, op.path) is not False:
                raise PatchError(f"{op.path} already exists — use *** Update File")
            content = "\n".join(op.added) + ("\n" if op.added else "")
            planned.append(FileChange(target, op.path, None, content, "created"))
            continue
        if not _exists(h, op.path):
            raise PatchError(f"no such file: {op.path}")
        body = h.read_text(op.path)
        if op.kind == "delete":
            planned.append(FileChange(target, op.path, body, None, "deleted"))
            continue
        updated = apply_hunks(op.path, body, op.hunks)
        destination = _resolve(h, op.move_to) if op.move_to else target
        if destination != target:
            if _exists(h, str(op.move_to)) is not False:
                raise PatchError(f"cannot move {op.path} to {op.move_to}: it exists")
            if destination in seen:
                raise PatchError(f"{op.move_to} appears twice in one patch")
            seen.add(destination)
            planned.append(FileChange(target, op.path, body, None, "moved"))
            planned.append(FileChange(destination, str(op.move_to), None, updated, "created"))
        else:
            planned.append(FileChange(target, op.path, body, updated, "edited"))
    return planned


def plan_patch(h: HarnessContext, input: str) -> Plan | str:  # noqa: A002 - the tool's parameter
    """The changes ``apply_patch`` would make, or why it cannot."""
    try:
        changes = _plan_ops(h, parse(input))
    except PatchError as exc:
        return f"patch not applied: {exc}"
    previews = []
    for change in changes:
        if change.after is None:
            previews.append(f"{change.shown}: {change.action} — the file is removed")
        else:
            previews.append(f"{change.shown}:\n{diff(change.before or '', change.after)}")
    report = "applied:\n" + "\n".join(f"  {c.action} {c.shown}" for c in changes)
    return Plan(changes, report, "\n".join(previews))


def make_apply_patch(h: HarnessContext) -> Tool:
    """The ``apply_patch`` tool over ``h``."""

    # The parameter is ``input`` because that is the name GPT-family models are
    # trained to fill for this tool; another name costs malformed calls.
    def apply_patch(input: str) -> str:  # noqa: A002 - the trained parameter name
        """Edit files by applying a patch. Use it for every file change.

        The patch is plain text in this envelope::

            *** Begin Patch
            *** Add File: path/new_file.py
            +every line of the new file, each starting with +
            *** Update File: path/existing.py
            @@ the line just above the change, e.g. def handler():
             context line (starts with a space)
            -line to remove
            +line to add
            *** Delete File: path/obsolete.py
            *** End Patch

        Give about three lines of unchanged context around each change, and an
        ``@@`` line naming the enclosing function or class when the context
        alone could match more than one place. ``*** Move to: new/path`` after
        an Update header renames the file. Paths are relative to the workspace
        root. Either the whole patch applies or none of it does.

        Args:
            input: The complete patch, from ``*** Begin Patch`` to ``*** End Patch``.
        """
        with h.write_lock:
            plan = plan_patch(h, input)
            if isinstance(plan, str):
                return plan
            try:
                h.commit(plan)
            except BackendError as exc:
                return f"patch not applied: {exc} — every file was put back as it was"
            return plan.report

    return tool(apply_patch)
