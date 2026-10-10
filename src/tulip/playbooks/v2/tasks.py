# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Task steps: a step a person does, not the model.

A step with ``kind: task`` (the registry's task step) is work for a person::

    - id: confirm_by_phone
      kind: task
      title: Call the vendor back on the number on file
      instructions: Use the number in the vendor master, never one from the request.
      assignee: {by: ap-clerks}          # a grant label; the requester is excluded
      form:                              # unless allow_requester: true
        - {name: confirmed_by, type: text, sensitive: false}
        - {name: call_recording, type: file}
      due: 8h
      escalate_to: ap-leads

The model has no tools in it and cannot complete it. When it becomes active the runtime
says so (:meth:`~tulip.playbooks.v2.engine.PlaybookRuntime.pending_task`, a
:class:`TaskRequest`): the gateway files a hold of kind ``task`` carrying the form and
parks the run. The person who completes it gives the form's values, which are checked
against the form (:func:`form_problems`, the registry's own sentences) and become the
step's outputs (:meth:`~tulip.playbooks.v2.engine.PlaybookRuntime.complete_task`).

Reading is tolerant, as for the rest of a v2 playbook: the registry checked the shape
strictly at publish, so an unreadable ``due`` reads as none and a stray key is ignored.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tulip.playbooks.v2.approvals import group_name, parse_duration
from tulip.playbooks.v2.fields import Field, parse_fields, validate_value


#: The ``kind`` of a step a person does.
TASK = "task"
#: The ``kind`` of a step the model does (a step that names no kind).
AGENT = "agent"


@dataclass(frozen=True)
class TaskAssignee:
    """Who may complete a task: the holders of ``approvals/<by>``.

    The person the run acts for is excluded unless ``allow_requester``; the registry,
    which knows who that is, enforces both. ``by`` is empty only for a task whose
    definition named nobody (the registry refuses one at publish).
    """

    by: str
    allow_requester: bool = False

    @property
    def shown_as(self) -> str:
        """The group as a person reads it: ``ap-clerks`` is "AP Clerks"."""
        return group_name(self.by) if self.by else "the workspace's approvers"


@dataclass(frozen=True)
class TaskSpec:
    """What a task step asks of a person (the registry's task step, read for running).

    ``form`` are the fields the person fills in; they are the step's outputs.
    ``due_seconds`` is how long they have (``None`` = the deployment's default) before
    the task escalates to ``escalate_to`` (``None`` = no escalation of its own).
    """

    assignee: TaskAssignee
    form: tuple[Field, ...] = ()
    instructions: str = ""
    due: str = ""
    due_seconds: int | None = None
    escalate_to: str | None = None


@dataclass(frozen=True)
class TaskRequest:
    """A task step that is active and waiting on its person (``pending_task``).

    Everything the gateway needs to file the hold: which step, what to do, the form,
    who may do it, and by when.
    """

    step_id: str
    title: str
    instructions: str
    form: tuple[Field, ...]
    assignee: TaskAssignee
    due_seconds: int | None = None
    escalate_to: str | None = None

    def form_schema(self) -> list[dict[str, Any]]:
        """The form as plain data, one mapping per field, for the hold to carry."""
        return [field_schema(f) for f in self.form]

    def hold_fields(self) -> dict[str, Any]:
        """The registry hold's fields for this task, beside the hold's own ``kind``.

        As for a step approval (:meth:`ResolvedApproval.hold_fields
        <tulip.playbooks.v2.approvals.ResolvedApproval.hold_fields>`): the assignee is one
        group that must give one completion and the only group that may (``approver_groups``,
        ``only_named_groups``); with a ``due``, ``ttl_seconds`` -- twice the due when it
        escalates, so the escalation gets its turn -- and, with ``escalate_to``,
        ``escalate_to_label`` and ``escalate_after_seconds``. ``task`` carries the step,
        its instructions, the form and whether the requester may complete it.
        """
        fields: dict[str, Any] = {
            "approver_groups": (
                [{"label": self.assignee.by, "count": 1}] if self.assignee.by else []
            ),
            "only_named_groups": bool(self.assignee.by),
            "task": {
                "step_id": self.step_id,
                "title": self.title,
                "instructions": self.instructions,
                "form": self.form_schema(),
                "allow_requester": self.assignee.allow_requester,
            },
        }
        if self.escalate_to:
            fields["escalate_to_label"] = self.escalate_to
        if self.due_seconds is not None:
            if self.escalate_to:
                fields["ttl_seconds"] = 2 * self.due_seconds
                fields["escalate_after_seconds"] = self.due_seconds
            else:
                fields["ttl_seconds"] = self.due_seconds
        return fields


def field_schema(field: Field) -> dict[str, Any]:
    """One field as plain data (the registry's ``PlaybookField``)."""
    out: dict[str, Any] = {
        "name": field.name,
        "label": field.shown_as,
        "type": field.type,
        "required": field.required,
        "sensitive": field.sensitive,
    }
    if field.choices:
        out["choices"] = list(field.choices)
    if field.description:
        out["description"] = field.description
    return out


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def is_task(raw: Mapping[str, Any]) -> bool:
    """Whether a step definition is a task step (``kind: task``)."""
    return _text(raw.get("kind")).lower() == TASK


def _assignee(raw: Any) -> TaskAssignee:
    if isinstance(raw, str):
        return TaskAssignee(by=raw.strip())
    if not isinstance(raw, Mapping):
        return TaskAssignee(by="")
    return TaskAssignee(by=_text(raw.get("by")), allow_requester=raw.get("allow_requester") is True)


def _due(value: Any) -> int | None:
    try:
        return parse_duration(value)
    except ValueError:
        return None


def parse_task(raw: Mapping[str, Any]) -> TaskSpec:
    """Read a task step's own fields tolerantly (call only for :func:`is_task`).

    ``form`` is the form; a task step that carries ``outputs`` instead reads them as it.
    """
    form = parse_fields(raw.get("form")) or parse_fields(raw.get("outputs"))
    due = _text(raw.get("due"))
    return TaskSpec(
        assignee=_assignee(raw.get("assignee")),
        form=form,
        instructions=_text(raw.get("instructions")),
        due=due,
        due_seconds=_due(due) if due else None,
        escalate_to=_text(raw.get("escalate_to")) or None,
    )


def form_problems(form: tuple[Field, ...], values: Any) -> dict[str, str]:
    """Why ``values`` cannot complete a task with ``form``, by field name (empty = fine).

    The registry's check, in its words: the values are names and their values; each
    names a field of the form; each field's value suits it
    (:func:`~tulip.playbooks.v2.fields.validate_value` -- a required field left out is
    "<Label> is required."). A value that names no field is keyed by that name.
    """
    if not isinstance(values, Mapping):
        return {"": "The values must be given as names and their values."}
    fields = {f.name: f for f in form}
    problems: dict[str, str] = {
        str(name): f"The form has no field named {name!r}." for name in values if name not in fields
    }
    for name, f in fields.items():
        problem = validate_value(f, values.get(name))
        if problem is not None:
            problems[name] = problem
    return problems


__all__ = [
    "AGENT",
    "TASK",
    "TaskAssignee",
    "TaskRequest",
    "TaskSpec",
    "field_schema",
    "form_problems",
    "is_task",
    "parse_task",
]
