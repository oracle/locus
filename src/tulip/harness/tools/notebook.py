# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Editing Jupyter notebooks a cell at a time.

A notebook is JSON, and the line-oriented tools handle it badly: ``edit`` has
to match a cell's source as it is escaped inside a JSON string — every quote
a ``\\"``, every line a separate list item ending in ``\\n`` — and ``write``
means re-emitting the whole document, outputs and base64 plots included, to
change one line. Models get both wrong in ways that leave a file Jupyter
will not open.

``notebook_edit`` works on cells: replace a cell's source, insert a cell, or
delete one, found by its id or its position. The file keeps its own layout —
indentation, key order, trailing newline — so a one-cell change is a
one-cell diff in review. An edited code cell loses its outputs and execution
count: they were produced by the old source, and keeping them would show
results the new code never computed.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from tulip.deepagent.backends.protocol import BackendError
from tulip.harness.tools.common import FileChange, HarnessContext, Plan, diff, outside
from tulip.tools.decorator import Tool, tool


__all__ = ["CELL_TYPES", "MODES", "make_notebook_edit", "plan_notebook_edit"]

MODES = ("replace", "insert", "delete")
CELL_TYPES = ("code", "markdown", "raw")


def _source_lines(text: str) -> list[str]:
    """A cell's source as nbformat stores it: lines that keep their newline."""
    return text.splitlines(keepends=True)


def _source_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _indent_of(raw: str) -> int | None:
    """The indent the file was written with, so the rewrite matches it.

    Jupyter writes one space; other tools write two or four, or nothing at
    all. Reading it off the second line is enough: the first is ``{``.
    """
    lines = raw.splitlines()
    if len(lines) < 2:
        return None
    second = lines[1]
    return len(second) - len(second.lstrip(" "))


def _dump(notebook: dict[str, Any], raw: str) -> str:
    text = json.dumps(notebook, indent=_indent_of(raw), ensure_ascii=False)
    return text + "\n" if raw.endswith("\n") else text


def _wants_ids(notebook: dict[str, Any]) -> bool:
    """Whether new cells need an ``id`` (nbformat 4.5 and later require one)."""
    if any("id" in cell for cell in notebook.get("cells", [])):
        return True
    major = int(notebook.get("nbformat", 4) or 4)
    minor = int(notebook.get("nbformat_minor", 0) or 0)
    return (major, minor) >= (4, 5)


def _new_cell(cell_type: str, source: str, with_id: bool) -> dict[str, Any]:
    cell: dict[str, Any] = {"cell_type": cell_type}
    if with_id:
        cell["id"] = uuid.uuid4().hex[:8]
    cell["metadata"] = {}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    cell["source"] = _source_lines(source)
    return cell


def _set_type(cell: dict[str, Any], cell_type: str) -> None:
    """Change a cell's type, adding or dropping the fields that type has."""
    cell["cell_type"] = cell_type
    if cell_type == "code":
        cell.setdefault("execution_count", None)
        cell.setdefault("outputs", [])
    else:
        cell.pop("execution_count", None)
        cell.pop("outputs", None)


def _locate(cells: list[dict[str, Any]], cell_id: str, index: int | None) -> int | str:
    """The position of the target cell, or why there is none."""
    if cell_id:
        for i, cell in enumerate(cells):
            if cell.get("id") == cell_id:
                return i
        known = ", ".join(str(c.get("id")) for c in cells[:20] if c.get("id"))
        return f"no cell with id {cell_id!r}" + (f" — ids: {known}" if known else "")
    if index is None:
        return "name the cell with cell_id or index"
    if not 0 <= index < len(cells):
        return f"index {index} is out of range — the notebook has {len(cells)} cell(s)"
    return index


def plan_notebook_edit(  # noqa: PLR0911 - one return per fault the model can make
    h: HarnessContext,
    path: str,
    new_source: str = "",
    cell_id: str = "",
    index: int | None = None,
    mode: str = "replace",
    cell_type: str = "",
) -> Plan | str:
    """The change ``notebook_edit`` would make, or why it cannot."""
    if mode not in MODES:
        return f"mode must be one of {', '.join(MODES)}, not {mode!r}"
    if cell_type and cell_type not in CELL_TYPES:
        return f"cell_type must be one of {', '.join(CELL_TYPES)}, not {cell_type!r}"
    if not path.endswith(".ipynb"):
        return f"{path} is not a notebook — use edit for other files"
    try:
        target = h.backend.resolve(path)
        raw = h.read_text(path)
    except BackendError as exc:
        if exc.code == "invalid_path":
            return outside(path, h.backend.capabilities.root)
        return f"no such file: {path}"
    try:
        notebook = json.loads(raw)
    except json.JSONDecodeError as exc:
        return f"{path} is not valid notebook JSON ({exc}) — not touching it"
    cells = notebook.get("cells") if isinstance(notebook, dict) else None
    if not isinstance(cells, list):
        return f"{path} has no cell list — not touching it"

    if mode == "insert":
        if not cell_type:
            return "insert needs cell_type: code, markdown or raw"
        if cell_id:
            found = _locate(cells, cell_id, None)
            if isinstance(found, str):
                return found
            position = found + 1
        else:
            position = len(cells) if index is None else max(0, min(index, len(cells)))
        cell = _new_cell(cell_type, new_source, _wants_ids(notebook))
        cells.insert(position, cell)
        before, after = "", new_source
        done = f"inserted {cell_type} cell at index {position}"
    else:
        found = _locate(cells, cell_id, index)
        if isinstance(found, str):
            return found
        position = found
        cell = cells[position]
        before = _source_text(cell)
        if mode == "delete":
            del cells[position]
            after = ""
            done = f"deleted cell {position}"
        else:
            cell["source"] = _source_lines(new_source)
            if cell_type and cell_type != cell.get("cell_type"):
                _set_type(cell, cell_type)
            if cell.get("cell_type") == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
            after = new_source
            done = f"replaced cell {position}"

    label = f" (id {cell['id']})" if cell.get("id") else ""
    # The preview is the cell's source, not the JSON: that is what a person
    # can judge at a gate.
    preview = f"{done}{label}\n{diff(before, after)}"
    change = FileChange(target, path, raw, _dump(notebook, raw), "edited")
    return Plan([change], f"{done}{label} in {path}", preview)


def make_notebook_edit(h: HarnessContext) -> Tool:
    """The ``notebook_edit`` tool over ``h``."""

    def notebook_edit(
        path: str,
        new_source: str = "",
        cell_id: str = "",
        index: int | None = None,
        mode: str = "replace",
        cell_type: str = "",
    ) -> str:
        """Edit a Jupyter notebook (.ipynb) cell: replace, insert or delete it.

        Use this rather than edit or write on .ipynb files. Name the cell by
        its id or by its 0-based index; read the notebook first to see both.

        Args:
            path: The .ipynb file, relative to the workspace root.
            new_source: The cell's full new source (replace, insert).
            cell_id: The cell to act on. For insert, the new cell goes after it.
            index: The cell's 0-based position, when it has no id. For insert,
                the position the new cell takes; leave both empty to append.
            mode: "replace", "insert" or "delete".
            cell_type: "code", "markdown" or "raw". Required for insert; for
                replace it changes the cell's type.
        """
        with h.write_lock:
            plan = plan_notebook_edit(h, path, new_source, cell_id, index, mode, cell_type)
            if isinstance(plan, str):
                return plan
            h.commit(plan)
            return plan.report

    return tool(notebook_edit)
