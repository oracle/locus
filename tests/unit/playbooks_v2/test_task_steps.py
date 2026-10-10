# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit: task steps a person completes, and the playbook's ``notify`` rules.

A step with ``kind: task`` is a person's: the model has no tools in it and cannot complete
it. ``pending_task`` hands the gateway what it files the task hold from; ``complete_task``
checks the person's values against the form in the registry's words, records them as the
step's outputs and routes on. The steps after it see what the person gave, the
``sensitive: false`` values in clear. A task step's outputs restore like any step's.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import pytest

from tulip.playbooks.v2 import (
    NOTIFY_EVENTS,
    ROUTING_UNKNOWN,
    NotifyRule,
    NotifyTarget,
    TaskAssignee,
    TaskRequest,
    form_problems,
    parse_notify_rules,
)
from tulip.playbooks.v2.engine import (
    ACTIVE,
    BLOCKED,
    DONE,
    PENDING,
    STEP_EVENT,
    WAIVED,
    PlaybookRuntime,
    PlaybookV2,
    parse_playbook_v2,
    playbook_prose,
    step_brief,
)
from tulip.playbooks.v2.fields import Field
from tulip.playbooks.v2.tasks import field_schema, is_task, parse_task


REF = {"ref": "file_01H", "name": "call.m4a", "size": 2048, "media_type": "audio/mp4"}


def _definition(**over: Any) -> dict[str, Any]:
    """Vendor bank change: ``review`` (the agent), ``call`` (a person), then a branch on
    what the person found: ``pay`` when they confirmed the account, ``reject`` otherwise."""
    call: dict[str, Any] = {
        "id": "call",
        "kind": "task",
        "title": "Call the vendor back on the number on file",
        "instructions": "Use the number in the vendor master, never one from the request.",
        "after": ["review"],
        "assignee": {"by": "ap-clerks"},
        "form": [
            {"name": "confirmed_by", "label": "Spoke with", "sensitive": False},
            {"name": "confirmed", "type": "boolean", "sensitive": False},
            {"name": "call_recording", "type": "file", "required": False},
            {"name": "callback_on", "type": "date", "sensitive": False, "required": False},
        ],
        "due": "8h",
        "escalate_to": "ap-leads",
        # A task step has no tools, whatever its definition says.
        "allowed_tools": ["lookup_vendor"],
        "branches": [
            {"id": "ok", "when": "confirmed == true", "next_step_id": "pay"},
            {"id": "no", "when": "confirmed == false", "next_step_id": "reject"},
        ],
    }
    call.update(over)
    return {
        "version": "playbook.v2",
        "id": "vendor-bank-change",
        "title": "Vendor bank change",
        "inputs": [{"name": "vendor_id", "sensitive": False}],
        "step_groups": [
            {
                "id": "g",
                "title": "Change",
                "steps": [
                    {
                        "id": "review",
                        "title": "Review the request",
                        "allowed_tools": ["lookup_vendor"],
                        "expected_outputs": ["summary"],
                    },
                    call,
                    {
                        "id": "pay",
                        "title": "Update the account",
                        "allowed_tools": ["update_vendor"],
                        "after": ["call"],
                    },
                    {
                        "id": "reject",
                        "title": "Reject the change",
                        "allowed_tools": [],
                        "after": ["call"],
                    },
                ],
            }
        ],
        "notify": [
            {
                "on": ["step.waiting", "step.failed", "bogus"],
                "steps": ["call"],
                "include": ["vendor_id"],
                "to": [{"by": "finance"}, "requester", {"channel": "slack:finance-alerts"}],
            }
        ],
    }


class _Run:
    def __init__(self, playbook: PlaybookV2, *, enforce: bool = False) -> None:
        self.events: list[dict[str, Any]] = []
        self.rt = PlaybookRuntime(playbook, emit=self.events.append, enforce=enforce)
        self.rt.start()

    def to_task(self) -> None:
        assert self.rt._admit("lookup_vendor") == ""
        self.rt._credit("lookup_vendor", '{"ok": true}')
        assert self.rt.complete_step("review", {"summary": "new IBAN"})["ok"]


def _run(enforce: bool = False, **over: Any) -> _Run:
    run = _Run(parse_playbook_v2(_definition(**over)), enforce=enforce)
    run.to_task()
    return run


def _values(**extra: Any) -> dict[str, Any]:
    return {"confirmed_by": "Dana Reyes", "confirmed": True, "call_recording": REF, **extra}


# ── parsing ──────────────────────────────────────────────────────────────────


def test_a_task_step_parses_into_its_spec() -> None:
    step = parse_playbook_v2(_definition()).step("call")
    assert step is not None
    assert step.is_task
    assert step.task is not None
    task = step.task
    assert task.assignee == TaskAssignee(by="ap-clerks", allow_requester=False)
    assert task.assignee.shown_as == "AP Clerks"
    assert [f.name for f in task.form] == [
        "confirmed_by",
        "confirmed",
        "call_recording",
        "callback_on",
    ]
    assert task.due == "8h"
    assert task.due_seconds == 8 * 3600
    assert task.escalate_to == "ap-leads"
    assert task.instructions.startswith("Use the number")
    # No tools, whatever the definition said; the form is the step's outputs.
    assert step.allowed_tools == ()
    assert not step.allows("lookup_vendor")
    assert not step.allows("ask_user")
    assert step.output_fields() == list(task.form)
    assert set(step.data_fields()) == {f.name for f in task.form}
    assert step.goal == task.instructions
    assert not step.decides


def test_task_parsing_is_tolerant() -> None:
    assert is_task({"kind": " Task "})
    assert not is_task({"kind": "agent"})
    assert not is_task({})
    bare = parse_task({"assignee": "ap-clerks", "due": "soon", "outputs": [{"name": "x"}]})
    assert bare.assignee == TaskAssignee("ap-clerks")
    assert bare.due == "soon"
    assert bare.due_seconds is None
    assert [f.name for f in bare.form] == ["x"]  # ``outputs`` stands in for ``form``
    nobody = parse_task({"assignee": 7, "escalate_to": "  "})
    assert nobody.assignee.by == ""
    assert nobody.assignee.shown_as == "the workspace's approvers"
    assert nobody.escalate_to is None
    assert nobody.form == ()
    assert nobody.due_seconds is None
    allowed = parse_task({"assignee": {"by": "ap", "allow_requester": True}})
    assert allowed.assignee.allow_requester is True
    assert parse_task({"assignee": {"by": "ap", "allow_requester": "yes"}}).assignee == (
        TaskAssignee("ap")
    )


def test_a_step_without_kind_is_the_agent_s() -> None:
    step = parse_playbook_v2(_definition()).step("review")
    assert step is not None
    assert not step.is_task
    assert step.task is None


def test_notify_rules_parse_as_the_registry_reads_them() -> None:
    playbook = parse_playbook_v2(_definition())
    assert playbook.notify_rules == (
        NotifyRule(
            on=("step.waiting", "step.failed"),
            to=(
                NotifyTarget("by", "finance"),
                NotifyTarget("requester"),
                NotifyTarget("slack", "finance-alerts"),
            ),
            steps=("call",),
            include=("vendor_id",),
        ),
    )
    rule = playbook.notify_rules[0]
    assert rule.applies("step.waiting", "call")
    assert not rule.applies("step.waiting", "review")
    assert not rule.applies("step.done", "call")
    assert "process.done" in NOTIFY_EVENTS


def test_notify_rules_fall_back_to_metadata_and_drop_what_does_not_read() -> None:
    rules = parse_notify_rules(
        {
            "metadata": {
                "notify": [
                    "not a rule",
                    {"on": "process.done", "to": [{"requester": True}]},
                    {"on": ["step.done"], "to": ["someone", {"channel": "email:x"}]},
                    {"on": ["nope"], "to": ["requester"]},
                    {"on": ["step.done"], "to": "requester"},
                ]
            }
        }
    )
    assert rules == (NotifyRule(on=("process.done",), to=(NotifyTarget("requester"),)),)
    assert rules[0].applies("process.done", "any-step")
    assert parse_notify_rules({}) == ()
    # Top-level rules win over metadata ones.
    top = {"on": ["step.done"], "to": [{"by": "ops"}]}
    meta = {"on": ["process.done"], "to": ["requester"]}
    assert parse_notify_rules({"notify": [top], "metadata": {"notify": [meta]}}) == (
        NotifyRule(on=("step.done",), to=(NotifyTarget("by", "ops"),)),
    )


@pytest.mark.parametrize("stored", [[], (), None])
def test_an_empty_top_level_notify_falls_back_to_metadata(stored: Any) -> None:
    """The registry stores every playbook with ``notify: []``; the rules may still be in
    ``metadata.notify``, and its dispatcher reads them there."""
    meta = {"on": ["step.waiting"], "steps": ["call"], "to": [{"by": "finance"}]}
    definition: dict[str, Any] = {"metadata": {"notify": [meta]}}
    if stored is not None:
        definition["notify"] = stored
    expected = (
        NotifyRule(on=("step.waiting",), to=(NotifyTarget("by", "finance"),), steps=("call",)),
    )
    assert parse_notify_rules(definition) == expected
    playbook = _definition()
    playbook["notify"] = stored if stored is not None else []
    playbook["metadata"] = {"notify": [meta]}
    assert parse_playbook_v2(playbook).notify_rules == expected
    assert parse_notify_rules({"notify": stored}) == ()
    assert parse_playbook_v2({**_definition(), "notify": None}).notify_rules == ()


# ── the model cannot act in a task step ──────────────────────────────────────


def test_the_model_cannot_complete_a_task_step() -> None:
    run = _run()
    assert [s.id for s in run.rt.active_steps()] == ["call"]
    result = run.rt.complete_step("call", _values())
    assert result["ok"] is False
    assert result["done_by_person"] is True
    assert result["error"].startswith("This step is done by a person: call (AP Clerks).")
    assert run.rt.graph.status["call"] == ACTIVE
    # Not yet active either: still a person's.
    early = _Run(parse_playbook_v2(_definition()))
    assert early.rt.complete_step("call", {})["error"].startswith("This step is done by a person")


def test_the_model_cannot_route_a_task_step() -> None:
    run = _run()
    result = run.rt.select_branches("call", ["ok"], "they said yes")
    assert result["ok"] is False
    assert result["done_by_person"] is True
    assert "call" not in run.rt.graph.selected


@pytest.mark.parametrize("enforce", [False, True])
@pytest.mark.parametrize("tool", ["lookup_vendor", "update_vendor", "ask_user"])
def test_every_call_is_refused_while_only_a_task_is_active(enforce: bool, tool: str) -> None:
    run = _run(enforce=enforce)
    refusal = run.rt._admit(tool)
    assert refusal.startswith("This step is done by a person: call (AP Clerks).")
    assert refusal.endswith(f"{tool} cannot run until they complete it.")
    deviation = run.events[-1]
    assert deviation["type"] == "playbook_deviation"
    assert deviation["step"] == "call"
    assert deviation["blocked"] is True
    # Control tools still pass: the model is told why by the tool's own refusal.
    assert run.rt._admit("complete_step") == ""


def test_a_task_beside_an_agent_step_leaves_the_agent_step_its_tools() -> None:
    definition = _definition(parallel_group="p", after=[])
    steps = definition["step_groups"][0]["steps"]
    steps[0]["parallel_group"] = "p"
    run = _Run(parse_playbook_v2(definition))
    assert {s.id for s in run.rt.active_steps()} == {"review", "call"}
    assert run.rt._admit("lookup_vendor") == ""
    owner = run.rt.owner_of("lookup_vendor")
    assert owner is not None
    assert owner.id == "review"
    # Outside every step's tools, the usual rule: recorded, not refused.
    assert run.rt._admit("update_vendor") == ""
    assert run.events[-1]["violation"] == "unexpected_tool"
    assert run.events[-1]["blocked"] is False


def test_the_brief_and_prose_say_a_person_does_it() -> None:
    playbook = parse_playbook_v2(_definition())
    step = playbook.step("call")
    assert step is not None
    brief = step_brief(step, {})
    assert "### Step call: Call the vendor back on the number on file (done by a person)" in brief
    assert "This step is done by a person (AP Clerks), not by you." in brief
    assert "What they are asked to do: Use the number in the vendor master" in brief
    assert "confirmed (true or false)" in brief
    assert "Due within 8h." in brief
    assert "Branches (routed by what the person gives):" in brief
    assert "Tools:" not in brief
    assert "Goal:" not in brief
    prose = playbook_prose(playbook, {}, ["review"])
    assert "- call: Call the vendor back on the number on file (done by a person;" in prose
    assert "A step done by a person is theirs" in prose
    plain = {**_definition(), "step_groups": [_definition()["step_groups"][0]]}
    plain["step_groups"][0]["steps"] = plain["step_groups"][0]["steps"][:1]
    assert "A step done by a person" not in playbook_prose(parse_playbook_v2(plain), {}, [])


def test_a_task_with_a_goal_of_its_own_shows_it() -> None:
    step = parse_playbook_v2(_definition(goal="Confirm the account", form=[])).step("call")
    assert step is not None
    brief = step_brief(step, {})
    assert "Goal: Confirm the account" in brief
    assert "What they fill in" not in brief


# ── pending_task ─────────────────────────────────────────────────────────────


def test_pending_task_is_the_active_task() -> None:
    run = _Run(parse_playbook_v2(_definition()))
    assert run.rt.pending_task() is None  # review is active, not the task
    run.to_task()
    task = run.rt.pending_task()
    assert isinstance(task, TaskRequest)
    assert task.step_id == "call"
    assert task.title.startswith("Call the vendor")
    assert task.instructions.startswith("Use the number")
    assert task.assignee.by == "ap-clerks"
    assert task.due_seconds == 8 * 3600
    assert task.escalate_to == "ap-leads"
    assert [f.name for f in task.form][:2] == ["confirmed_by", "confirmed"]
    # The gateway parks the run: the step is blocked, still pending as a task.
    run.rt.pause("waiting on AP Clerks")
    assert run.rt.graph.status["call"] == BLOCKED
    assert run.rt.pending_task() == task


def test_hold_fields_carry_the_form_and_the_deadline() -> None:
    run = _run()
    task = run.rt.pending_task()
    assert task is not None
    fields = task.hold_fields()
    assert fields["approver_groups"] == [{"label": "ap-clerks", "count": 1}]
    assert fields["only_named_groups"] is True
    assert fields["escalate_to_label"] == "ap-leads"
    assert fields["ttl_seconds"] == 16 * 3600
    assert fields["escalate_after_seconds"] == 8 * 3600
    assert fields["task"]["step_id"] == "call"
    assert fields["task"]["allow_requester"] is False
    assert fields["task"]["form"][0] == {
        "name": "confirmed_by",
        "label": "Spoke with",
        "type": "text",
        "required": True,
        "sensitive": False,
    }
    no_escalation = TaskRequest(
        "t", "T", "", (), TaskAssignee(""), due_seconds=60, escalate_to=None
    ).hold_fields()
    assert no_escalation["ttl_seconds"] == 60
    assert "escalate_after_seconds" not in no_escalation
    assert no_escalation["approver_groups"] == []
    assert no_escalation["only_named_groups"] is False
    bare = TaskRequest("t", "T", "", (), TaskAssignee("x")).hold_fields()
    assert "ttl_seconds" not in bare
    assert "escalate_to_label" not in bare
    choice = Field(name="c", type="choice", choices=("a", "b"), description="Pick one")
    assert field_schema(choice)["choices"] == ["a", "b"]
    assert field_schema(choice)["description"] == "Pick one"


def test_pending_task_is_none_once_decided() -> None:
    definition = _definition()
    definition["decision_policy"] = {
        "id": "d",
        "outcomes": [{"id": "INCONCLUSIVE", "priority": 0}],
    }
    run = _Run(parse_playbook_v2(definition))
    run.to_task()
    assert run.rt.pending_task() is not None
    assert run.rt.submit_decision("INCONCLUSIVE", "no answer", [])["ok"]
    assert run.rt.pending_task() is None


# ── complete_task ────────────────────────────────────────────────────────────


def test_complete_task_records_the_values_and_routes_on() -> None:
    run = _run()
    run.rt.pause("waiting on AP Clerks")
    result = run.rt.complete_task("call", _values(), by="dana@example.com")
    assert result["ok"] is True
    assert result["status"] == DONE
    assert result["completed_by"] == "dana@example.com"
    assert result["routed_by"] == "when"
    assert result["branches_taken"] == ["ok"]
    assert result["branches_not_taken"] == ["no"]
    assert result["active"] == ["pay"]
    assert result["waived"] == ["reject"]
    assert run.rt.graph.outputs["call"] == _values()
    done = [e for e in run.events if e.get("step_id") == "call" and e["status"] == DONE]
    assert len(done) == 1
    assert done[0]["completed_by"] == "dana@example.com"
    assert done[0]["outputs"] == _values()
    assert done[0]["enabled_steps"] == ["pay"]
    assert run.rt.pending_task() is None
    # The values are typed for conditions and the clear ones are public.
    assert run.rt.public_outputs("call") == {"confirmed_by": "Dana Reyes", "confirmed": True}


def test_the_other_branch_when_the_person_says_no() -> None:
    run = _run()
    result = run.rt.complete_task("call", {"confirmed_by": "Dana", "confirmed": False}, by="d")
    assert result["ok"]
    assert result["active"] == ["reject"]
    assert run.rt.graph.status["pay"] == WAIVED


def test_the_brief_after_a_task_shows_what_the_person_gave() -> None:
    run = _run()
    result = run.rt.complete_task("call", _values(callback_on="2026-10-12"), by="dana")
    (brief,) = result["next"]
    assert "Done by a person -- Call the vendor back on the number on file (call):" in brief
    assert "- Spoke with: Dana Reyes" in brief
    assert "- Confirmed: true" in brief
    assert "- Call recording: given (sensitive, not shown)" in brief
    assert "- Callback on: 2026-10-12" in brief
    assert "file_01H" not in brief
    assert "call.m4a" not in brief
    # The model's own briefs before the task carry nothing of it.
    assert "Done by a person" not in run.rt.brief(["review"])[0]


def test_a_field_the_person_left_out_is_said_so() -> None:
    run = _run()
    result = run.rt.complete_task("call", {"confirmed_by": "Dana", "confirmed": True}, by="dana")
    (brief,) = result["next"]
    assert "- Call recording: not given" in brief


def test_a_form_with_no_fields_says_so() -> None:
    run = _run(form=[], branches=[])
    result = run.rt.complete_task("call", {}, by="dana")
    assert result["ok"]
    assert "- (the form asked for nothing)" in result["next"][0]


@pytest.mark.parametrize(
    ("values", "invalid"),
    [
        ({"confirmed": True}, {"confirmed_by": "Spoke with is required."}),
        (
            {"confirmed_by": "Dana", "confirmed": "yes"},
            {"confirmed": "Confirmed must be yes or no (true or false)."},
        ),
        (
            {"confirmed_by": 7, "confirmed": True},
            {"confirmed_by": "Spoke with must be text."},
        ),
        (
            {"confirmed_by": "Dana", "confirmed": True, "callback_on": "12/10/2026"},
            {"callback_on": "Callback on must be a date written as YYYY-MM-DD."},
        ),
        (
            {"confirmed_by": "Dana", "confirmed": True, "notes": "x"},
            {"notes": "The form has no field named 'notes'."},
        ),
        (
            {"confirmed_by": "Dana", "confirmed": True, "call_recording": b"\x00\x01"},
            {
                "call_recording": "Call recording must be a reference to a stored file, "
                "never the file itself."
            },
        ),
        (
            {"confirmed_by": "Dana", "confirmed": True, "call_recording": {"ref": "f"}},
            {"call_recording": "Call recording must name the stored file's name."},
        ),
        (
            {
                "confirmed_by": "Dana",
                "confirmed": True,
                "call_recording": {**REF, "bytes": "AAAA"},
            },
            {
                "call_recording": "Call recording must be a reference to a stored file "
                "(ref, name, size, media_type); it cannot carry bytes."
            },
        ),
        ("all good", {"": "The values must be given as names and their values."}),
    ],
)
def test_complete_task_refuses_values_that_do_not_suit_the_form(
    values: Any, invalid: dict[str, str]
) -> None:
    run = _run()
    before = copy.deepcopy(run.events)
    result = run.rt.complete_task("call", values, by="dana")
    assert result["ok"] is False
    assert result["invalid"] == invalid
    assert result["error"] == (
        "The task Call the vendor back on the number on file cannot be completed: "
        + " ".join(invalid.values())
    )
    assert run.events == before  # nothing changed
    assert run.rt.graph.status["call"] == ACTIVE
    assert "call" not in run.rt.graph.outputs


def test_form_problems_lists_each_bad_value() -> None:
    form = (Field(name="a", type="number"), Field(name="b", required=False))
    assert form_problems(form, {"a": 1}) == {}
    assert form_problems(form, {"a": "1", "b": 2}) == {
        "a": "A must be a number.",
        "b": "B must be text.",
    }


def test_complete_task_refuses_what_is_not_a_waiting_task() -> None:
    run = _Run(parse_playbook_v2(_definition()))
    unknown = run.rt.complete_task("nope", {}, by="d")
    assert unknown["ok"] is False
    assert "no step 'nope'" in unknown["error"]
    agent = run.rt.complete_task("review", {"summary": "x"}, by="d")
    assert agent["error"] == "step review is not a task step: the agent completes it"
    pending = run.rt.complete_task("call", _values(), by="d")
    assert pending["error"] == f"step call is {PENDING}, not active"
    run.to_task()
    nobody = run.rt.complete_task("call", _values(), by="  ")
    assert nobody["error"] == "say who completed the task: `by` is required"
    assert run.rt.complete_task("call", _values(), by="d")["ok"]
    again = run.rt.complete_task("call", _values(), by="d")
    assert again["error"] == f"step call is {DONE}, not active"


def test_a_branch_that_reads_what_the_run_cannot_see_is_not_routed() -> None:
    run = _run(
        branches=[
            {"id": "ok", "when": "inputs.vendor_id == 'v1'", "next_step_id": "pay"},
            {"id": "no", "when": "NOT (inputs.vendor_id == 'v1')", "next_step_id": "reject"},
        ]
    )
    # No inputs were given: ``inputs.vendor_id`` is unavailable.
    result = run.rt.complete_task("call", _values(), by="dana")
    assert result["ok"] is False
    assert result["routing"] == ROUTING_UNKNOWN
    assert result["unavailable_inputs"] == ["vendor_id"]
    assert run.rt.graph.status["call"] == ACTIVE
    run.rt.set_inputs({"vendor_id": "v1"})
    assert run.rt.complete_task("call", _values(), by="dana")["active"] == ["pay"]


# ── restore ──────────────────────────────────────────────────────────────────


def _digest(value: Any) -> dict[str, Any]:
    """``tulip_gateway.persist._digest``, byte for byte."""
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return {"redacted": True, "sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded)}


def _outputs_digested(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {k: _digest(v) if k in ("outputs", "completed_by") else v for k, v in e.items()}
        for e in copy.deepcopy(events)
        if e.get("type") == STEP_EVENT
    ]


def _fresh() -> PlaybookRuntime:
    return PlaybookRuntime(parse_playbook_v2(_definition()), emit=lambda _e: None)


def test_a_pending_task_restores_pending() -> None:
    run = _run()
    run.rt.pause("waiting on AP Clerks")
    back = _fresh()
    back.restore(copy.deepcopy(run.events))
    assert back.graph.status["call"] == BLOCKED
    assert back.pending_task() == run.rt.pending_task()
    result = back.complete_task("call", _values(), by="dana")
    assert result["ok"]
    assert result["active"] == ["pay"]


def test_a_task_s_digested_outputs_restore_withheld() -> None:
    run = _run()
    run.rt.complete_task("call", _values(), by="dana")
    back = _fresh()
    result = back.restore(_outputs_digested(run.events))
    assert "call" in result.unavailable
    assert back.public_outputs("call") == {}
    (brief,) = back.brief(["pay"])
    assert "- Spoke with: given, not available to this run" in brief
    assert "Dana" not in brief


def test_a_task_s_outputs_supplied_on_restore_are_its_values() -> None:
    run = _run()
    run.rt.complete_task("call", _values(), by="dana")
    back = _fresh()
    result = back.restore(_outputs_digested(run.events), outputs={"call": _values()})
    assert "call" not in result.unavailable
    assert back.graph.outputs["call"] == _values()
    assert back.public_outputs("call") == {"confirmed_by": "Dana Reyes", "confirmed": True}
    assert "- Spoke with: Dana Reyes" in back.brief(["pay"])[0]
    assert back.graph.status["pay"] == ACTIVE
    assert back.graph.status["reject"] == WAIVED


def test_two_tasks_active_together_are_both_people_s() -> None:
    definition = _definition(parallel_group="p", after=[], branches=[])
    steps = definition["step_groups"][0]["steps"]
    steps[0] = {
        "id": "review",
        "kind": "task",
        "title": "Read the request",
        "assignee": {"by": "finance"},
        "parallel_group": "p",
    }
    run = _Run(parse_playbook_v2(definition))
    refusal = run.rt._admit("lookup_vendor")
    assert refusal.startswith(
        "These steps are done by people: review (Finance), call (AP Clerks). "
        "The run waits until they complete them."
    )
    # The first in definition order is the one pending; each completes on its own.
    first = run.rt.pending_task()
    assert first is not None
    assert first.step_id == "review"
    assert first.instructions == ""
    brief = step_brief(run.rt.playbook.steps[0], {})
    assert "What they are asked to do" not in brief
    assert "Due within" not in brief
    assert run.rt.complete_task("review", {}, by="fin")["ok"]
    second = run.rt.pending_task()
    assert second is not None
    assert second.step_id == "call"
