# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Reading and changing files: read, write, edit, multi_edit, ls.

Where ``old`` lands in an edit is decided by
:func:`tulip.tools.text_edit.apply_edit`, which forgives a near miss —
indentation, spacing, escaping — and refuses an ambiguous one. What is
decided here is what only a harness on a real workspace has: the record of
what the agent has read (:class:`~tulip.harness.ledger.ReadLedger`), line
endings as they are on disk, and the diff that goes back to the model.
"""

from __future__ import annotations

import base64
import json
import posixpath
from typing import Any

from tulip.core.media import encode_image
from tulip.deepagent.backends.protocol import BackendError, FileInfo
from tulip.harness.backend import SKIP_DIRS, take_window
from tulip.harness.tools.common import (
    FileChange,
    HarnessContext,
    Plan,
    crlf,
    describe,
    diff,
    outside,
)
from tulip.tools.decorator import Tool, tool
from tulip.tools.output import ToolOutput
from tulip.tools.text_edit import EditMatchError, apply_edit


__all__ = [
    "IMAGE_TYPES",
    "MAX_IMAGE_BYTES",
    "make_edit",
    "make_ls",
    "make_multi_edit",
    "make_read",
    "make_write",
    "plan_edit",
    "plan_multi_edit",
    "plan_write",
]

#: The image types every vision-capable provider accepts.
IMAGE_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

#: Larger images are declined rather than sent. Five megabytes is the
#: strictest provider's per-image limit, and an image past it fails the whole
#: request rather than just itself.
MAX_IMAGE_BYTES = 5_000_000

#: How many entries one directory shows in ``ls``, and how long the whole
#: listing may get.
LS_FILES_PER_DIR = 40
LS_MAX_LINES = 400


def _suffix(path: str) -> str:
    return posixpath.splitext(path)[1].lower()


# ------------------------------------------------------------------- read --


def _image(h: HarnessContext, path: str, shown: str, size: int) -> str:
    """``path`` as something a model can use: the image itself, or why not."""
    if not h.config.vision:
        return (
            f"{shown} is an image ({size:,} bytes), and this model cannot see images. "
            "Say what you need from it, or ask for a vision-capable model."
        )
    if size > MAX_IMAGE_BYTES:
        return (
            f"{shown} is an image of {size:,} bytes — over the {MAX_IMAGE_BYTES:,}-byte "
            "limit for sending one to a model. A smaller copy would work."
        )
    data = h.backend.read_bytes(path)
    media_type = IMAGE_TYPES[_suffix(path)]
    # The image goes to the model embedded in the result string, which every
    # provider adapter sends in its own wire format; the content block is for
    # an application that renders tool results.
    embedded = encode_image(data, media_type)
    block = {"type": "image", "mimeType": media_type, "data": base64.b64encode(data).decode()}
    return ToolOutput(f"{shown} ({media_type}, {size:,} bytes){embedded}", content_blocks=[block])


def _notebook_lines(raw: bytes) -> list[bytes] | None:
    """A notebook rendered cell by cell, as lines; ``None`` if it is not one."""
    try:
        notebook = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    cells = notebook.get("cells") if isinstance(notebook, dict) else None
    if not isinstance(cells, list):
        return None
    out: list[str] = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            continue
        label = f" id={cell['id']}" if cell.get("id") else ""
        out.append(f"--- cell {index}{label} [{cell.get('cell_type', 'code')}]")
        source = cell.get("source", "")
        out.extend(("".join(source) if isinstance(source, list) else str(source)).splitlines())
        for output in cell.get("outputs", []) or []:
            text = output.get("text") if isinstance(output, dict) else None
            if text is None and isinstance(output, dict):
                text = (output.get("data") or {}).get("text/plain")
            if text:
                body = "".join(text) if isinstance(text, list) else str(text)
                out.extend(f"> {line}" for line in body.splitlines()[:20])
    return [line.encode("utf-8") for line in out]


def make_read(h: HarnessContext) -> Tool:
    """The ``read`` tool over ``h``."""

    def read(path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read a file. Returns numbered lines.

        Files of any size can be read a window at a time; one call returns at
        most about 200 KB. An image file (png, jpg, gif, webp) is shown to you
        as an image when the model can see. A notebook (.ipynb) is shown cell
        by cell, with each cell's index and id.

        Args:
            path: File to read, relative to the workspace root.
            offset: First line to return, 0-indexed.
            limit: How many lines. Keep it tight on large files.
        """
        try:
            stat = h.backend.stat(path)
        except BackendError as exc:
            if exc.code == "invalid_path":
                return outside(path, h.backend.capabilities.root)
            return f"no such file: {path}"
        if stat.is_dir:
            return f"{path} is a directory — use ls or glob"
        if _suffix(path) in IMAGE_TYPES:
            return _image(h, path, path, stat.size)
        offset, limit = max(0, int(offset)), max(1, int(limit))
        rendered = (
            _notebook_lines(h.backend.read_bytes(path)) if _suffix(path) == ".ipynb" else None
        )
        if rendered is not None:
            window = take_window(rendered, offset=offset, limit=limit)
        else:
            window = h.backend.read_lines(path, offset=offset, limit=limit)
        h.ledger.note(h.backend, path)
        if not window.lines:
            if window.total == 0:
                return f"{path} is empty"
            return f"{path} has {window.total:,} lines — offset {offset} is past the end"
        text = "\n".join(window.lines)
        more = window.total - (offset + len(window.lines))
        if more <= 0:
            return text
        following = offset + len(window.lines)
        return text + (
            f"\n... {more:,} more lines (of {window.total:,}) — read again with offset={following}"
        )

    return tool(read)


# ------------------------------------------------------------------ write --


def plan_write(h: HarnessContext, path: str, content: str) -> Plan | str:
    """The change ``write`` would make, or why it cannot."""
    try:
        target = h.backend.resolve(path)
        stat = h.backend.stat(path)
    except BackendError as exc:
        if exc.code == "invalid_path":
            return outside(path, h.backend.capabilities.root)
        stat = None
    if stat is not None and stat.is_dir:
        return f"{path} is a directory"
    existed = stat is not None
    refused = h.ledger.unseen(h.backend, path, path)
    if refused:
        return refused
    old = h.read_text(path) if existed else ""
    if existed and crlf(old) and "\r" not in content:
        # The file's line endings are a property of the file, not of the text
        # the model happened to send.
        content = content.replace("\n", "\r\n")
    preview = diff(old, content)
    action = "overwrote" if existed else "created"
    report = f"{action} {path} ({len(content):,} bytes)"
    # The diff only when overwriting: what was displaced is what the model
    # cannot otherwise know, while a new file's diff is the content it sent.
    if existed:
        report = f"{report}\n{preview}"
    change = FileChange(target, path, old if existed else None, content, action)
    return Plan([change], report, preview)


def make_write(h: HarnessContext) -> Tool:
    """The ``write`` tool over ``h``."""

    def write(path: str, content: str) -> str:
        """Create or overwrite a file.

        An existing file must have been read first.

        Args:
            path: File to write, relative to the workspace root.
            content: Full new contents. This replaces the file.
        """
        with h.write_lock:
            plan = plan_write(h, path, content)
            if isinstance(plan, str):
                return plan
            h.commit(plan)
            return plan.report

    return tool(write)


# ------------------------------------------------------------------- edit --


def _editable(h: HarnessContext, path: str) -> tuple[str, str] | str:
    """The resolved path and text of a file that may be edited, or why not."""
    try:
        target = h.backend.resolve(path)
        stat = h.backend.stat(path)
    except BackendError as exc:
        if exc.code == "invalid_path":
            return outside(path, h.backend.capabilities.root)
        return f"no such file: {path}"
    if stat.is_dir:
        return f"{path} is a directory"
    refused = h.ledger.unseen(h.backend, path, path)
    if refused:
        return refused
    return target, h.read_text(path)


def plan_edit(
    h: HarnessContext, path: str, old: str, new: str, replace_all: bool = False
) -> Plan | str:
    """The change ``edit`` would make, or why it cannot."""
    found = _editable(h, path)
    if isinstance(found, str):
        return found
    target, body = found
    try:
        outcome = apply_edit(body, old, new, replace_all=replace_all)
    except EditMatchError as exc:
        return f"{path}: {exc}"
    preview = diff(body, outcome.content)
    change = FileChange(target, path, body, outcome.content, "edited")
    return Plan([change], f"edited {path}{describe(outcome)}\n{preview}", preview)


def make_edit(h: HarnessContext) -> Tool:
    """The ``edit`` tool over ``h``."""

    def edit(path: str, old: str, new: str, replace_all: bool = False) -> str:
        """Replace text in a file. Read the file first.

        ``old`` should be copied from the file. A near miss — indentation,
        spacing, escaping — is matched anyway and reported; ``old`` must still
        identify one place, unless ``replace_all`` is set. Returns the diff
        applied.

        Args:
            path: File to edit.
            old: Text to replace, including indentation.
            new: Replacement text.
            replace_all: Replace every occurrence instead of exactly one.
        """
        with h.write_lock:
            plan = plan_edit(h, path, old, new, bool(replace_all))
            if isinstance(plan, str):
                return plan
            h.commit(plan)
            return plan.report

    return tool(edit)


def plan_multi_edit(h: HarnessContext, path: str, edits: list[dict[str, Any]]) -> Plan | str:
    """The change ``multi_edit`` would make, or why it cannot."""
    found = _editable(h, path)
    if isinstance(found, str):
        return found
    target, original = found
    working = original
    notes: list[str] = []
    for i, item in enumerate(edits, 1):
        old, new = str(item.get("old", "")), str(item.get("new", ""))
        if not old:
            return f"edit {i} has no 'old' text — nothing was written"
        try:
            outcome = apply_edit(working, old, new, replace_all=bool(item.get("replace_all")))
        except EditMatchError as exc:
            return f"edit {i}: {exc}\nnothing was written — the file is unchanged"
        working = outcome.content
        note = describe(outcome)
        if note:
            notes.append(f"  edit {i}{note}")
    preview = diff(original, working)
    said = "\n".join([f"applied {len(edits)} edit(s) to {path}", *notes])
    change = FileChange(target, path, original, working, "edited")
    return Plan([change], f"{said}\n{preview}", preview)


def make_multi_edit(h: HarnessContext) -> Tool:
    """The ``multi_edit`` tool over ``h``."""

    def multi_edit(path: str, edits: list[dict[str, Any]]) -> str:
        """Apply several replacements to one file, atomically. Read the file first.

        Either every edit applies or none do. A partially-edited file is worse
        than an unedited one — it compiles less often and reads as if it were
        finished. Each ``old`` is matched the way ``edit`` matches it.

        Args:
            path: File to edit.
            edits: ``[{"old": "...", "new": "...", "replace_all": false}, ...]``,
                applied in order, each to the result of the one before. Each
                ``old`` must identify one place unless its ``replace_all`` is
                true.
        """
        with h.write_lock:
            plan = plan_multi_edit(h, path, edits)
            if isinstance(plan, str):
                return plan
            h.commit(plan)
            return plan.report

    return tool(multi_edit)


# --------------------------------------------------------------------- ls --


def _tree(entries: list[FileInfo], top: str, depth: int) -> list[str]:
    """Entries as an indented tree, top-down, the way ``os.walk`` visits them."""
    prefix = top.rstrip("/") + "/"
    children: dict[str, list[FileInfo]] = {}
    for entry in entries:
        rel = entry.path.removeprefix(prefix)
        parts = rel.split("/")
        # Hidden files and directories, and anything under them, are noise in
        # an orientation listing; the SKIP_DIRS are never walked at all.
        if any(p.startswith(".") or p in SKIP_DIRS for p in parts):
            continue
        children.setdefault(posixpath.dirname(rel), []).append(entry)

    lines: list[str] = []

    def visit(rel: str, level: int) -> bool:
        indent = "  " * level
        if level > 0:
            lines.append(f"{indent}{posixpath.basename(rel)}/")
        here = children.get(rel, [])
        files = sorted((e for e in here if not e.is_dir), key=lambda e: e.path)
        for entry in files[:LS_FILES_PER_DIR]:
            lines.append(f"{indent}  {posixpath.basename(entry.path)}")
        if len(files) > LS_FILES_PER_DIR:
            lines.append(f"{indent}  ... {len(files) - LS_FILES_PER_DIR} more files")
        if len(lines) > LS_MAX_LINES:
            lines.append("... truncated, use a smaller depth or a subdirectory")
            return False
        subdirs = sorted((e for e in here if e.is_dir and level + 1 < depth), key=lambda e: e.path)
        return all(visit(sub.path.removeprefix(prefix), level + 1) for sub in subdirs)

    visit("", 0)
    return lines


def make_ls(h: HarnessContext) -> Tool:
    """The ``ls`` tool over ``h``."""

    def ls(path: str = ".", depth: int = 2) -> str:
        """List a directory as a tree. The fastest way to learn a repo's shape.

        Args:
            path: Directory to list.
            depth: How many levels down. Two is usually enough to orient.
        """
        try:
            stat = h.backend.stat(path)
        except BackendError as exc:
            if exc.code == "invalid_path":
                return outside(path, h.backend.capabilities.root)
            return f"no such directory: {path}"
        if not stat.is_dir:
            return f"{path} is a file ({stat.size:,} bytes)"
        depth = max(1, int(depth))
        top = h.backend.resolve(path)
        entries = h.backend.ls(path, recursive=True, depth=depth)
        return "\n".join(_tree(entries, top, depth)) or "(empty)"

    return tool(ls)
