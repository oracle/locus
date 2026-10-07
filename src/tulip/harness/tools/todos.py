# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""The agent's plan for the current task, surviving across turns.

Stored in a deepagent :class:`~tulip.deepagent.todos.TodoState`, one per
harness, so an application reads the same list the deepagent tools would.
The tool takes the shape coding models fill in — ``task`` and a status of
``pending``, ``in_progress`` or ``done`` — and accepts the deepagent
spelling (``content``, ``completed``) too.
"""

from __future__ import annotations

from typing import Any

from tulip.deepagent.todos import Todo
from tulip.harness.tools.common import HarnessContext
from tulip.tools.decorator import Tool, tool


__all__ = ["make_todo_read", "make_todo_write", "render"]

_STATUS = {
    "pending": "pending",
    "in_progress": "in_progress",
    "done": "completed",
    "completed": "completed",
}
_MARKS = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}


def render(todos: list[Todo]) -> str:
    """The list as checkboxes."""
    if not todos:
        return "(no todos)"
    return "\n".join(f"{_MARKS.get(t.status, '[ ]')} {t.content}" for t in todos)


def make_todo_write(h: HarnessContext) -> Tool:
    """The ``todo_write`` tool over ``h``."""

    def todo_write(items: list[dict[str, Any]]) -> str:
        """Record or update the plan for the current task.

        Use it when a task has more than about three steps, so progress
        survives a long tool sequence and the user can see where you are.
        Overwrites the whole list, so send the full plan each time.

        Args:
            items: ``[{"task": "...", "status": "pending|in_progress|done"}, ...]``
        """
        cleaned: list[Todo] = []
        for raw in items:
            if not isinstance(raw, dict):
                continue
            task = str(raw.get("task") or raw.get("content") or "").strip()
            if not task:
                continue
            status = _STATUS.get(str(raw.get("status", "pending")), "pending")
            cleaned.append(Todo(content=task, status=status))
        stored = h.todos.replace(cleaned)
        done = sum(1 for t in stored if t.status == "completed")
        return f"{done}/{len(stored)} done\n{render(stored)}"

    return tool(todo_write)


def make_todo_read(h: HarnessContext) -> Tool:
    """The ``todo_read`` tool over ``h``."""

    def todo_read() -> str:
        """Read the current plan back."""
        return render(h.todos.snapshot())

    return tool(todo_read)
