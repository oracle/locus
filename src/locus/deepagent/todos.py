# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Structured task tracker for the deep agent.

Mirrors deepagents' ``write_todos`` tool: the agent maintains a list
of ``{content, status}`` items it updates between steps. The status
is a plain string (``"pending" | "in_progress" | "completed"``) —
not a Pydantic ``Literal``, per the project convention that
LLM-populated fields stay loose so the model doesn't crash on a
status it hasn't seen before.

The TODO list is *agent-driven*: nothing forces the agent to use it.
Tools-attached, prompt-encouraged, agent-decided.
"""

from __future__ import annotations

import json
import threading
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from locus.tools.decorator import tool


class Todo(BaseModel):
    """One task on the agent's structured task list."""

    model_config = ConfigDict(extra="ignore")

    content: str = Field(..., description="What the agent is doing.")
    status: str = Field(
        default="pending",
        description="One of: pending | in_progress | completed.",
    )


class TodoState:
    """Thread-safe holder for the agent's TODO list.

    A single TodoState is shared between ``write_todos`` and
    ``read_todos`` for one agent run. Pass the same state to
    :func:`make_todo_tools` if you want to inspect the list externally
    after the run.
    """

    def __init__(self) -> None:
        self._todos: list[Todo] = []
        self._lock = threading.Lock()

    def replace(self, todos: list[Todo]) -> list[Todo]:
        with self._lock:
            self._todos = list(todos)
            return list(self._todos)

    def snapshot(self) -> list[Todo]:
        with self._lock:
            return list(self._todos)


def make_todo_tools(state: TodoState | None = None) -> list[Any]:
    """Build the ``write_todos`` and ``read_todos`` tools.

    Both tools take / return a JSON string (a list of
    ``{content, status}`` objects). Strings keep OpenAI's strict
    structured-output mode happy — nested object schemas would
    require recursive ``additionalProperties: false`` annotations.
    """
    state = state if state is not None else TodoState()

    @tool
    def write_todos(todos_json: str) -> str:
        """Replace the agent's TODO list.

        ``todos_json`` is a JSON-encoded list of ``{content, status}``
        objects. Status is one of ``pending | in_progress | completed``.
        Returns the canonical list as JSON so the agent can confirm
        what was stored.

        Use this aggressively for multi-step research: write the plan
        as a list of pending TODOs at the start, mark each
        ``in_progress`` before working it, mark it ``completed`` after.
        """
        try:
            raw = json.loads(todos_json)
        except json.JSONDecodeError as exc:
            return f"invalid JSON: {exc}"
        if not isinstance(raw, list):
            return "todos_json must be a JSON array of {content, status} objects"
        items: list[Todo] = []
        for entry in raw:
            if not isinstance(entry, dict):
                return "each todo must be an object with `content` and optional `status`"
            try:
                items.append(Todo(**entry))
            except (TypeError, ValueError) as exc:
                return f"invalid todo entry {entry!r}: {exc}"
        stored = state.replace(items)
        from locus.observability.emit import (
            EV_DEEPAGENT_TODO_ADDED,
            EV_DEEPAGENT_TODO_COMPLETED,
            emit_sync,
        )

        for todo in stored:
            ev = (
                EV_DEEPAGENT_TODO_COMPLETED
                if todo.status == "completed"
                else EV_DEEPAGENT_TODO_ADDED
            )
            emit_sync(ev, content=todo.content, status=todo.status)
        return json.dumps([t.model_dump() for t in stored])

    @tool
    def read_todos() -> str:
        """Return the current TODO list as a JSON-encoded array."""
        return json.dumps([t.model_dump() for t in state.snapshot()])

    return [write_todos, read_todos]
