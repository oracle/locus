# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit: a runtime restored from its own ``playbook_step`` records is where the run left it.

A run rebuilt on another pod replays its trace onto a fresh runtime
(:meth:`PlaybookRuntime.restore`). The round trip runs the refund-dispute example through
the engine, restores a fresh runtime from what it emitted, and checks the two agree --
statuses, active steps, who owns a call, per-step counts -- and then keep agreeing event
for event as the run goes on. The refusals are all-or-nothing: nothing changes.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from tulip.playbooks.v2 import RestoreError, RestoreResult
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
)


EXAMPLE = Path(__file__).parent / "fixtures" / "refund-dispute.v2.yaml"
TOOLS = ("lookup_order", "get_balance", "submit_decision", "ask_user", "issue_refund")


def example() -> dict[str, Any]:
    body: dict[str, Any] = yaml.safe_load(EXAMPLE.read_text("utf-8"))
    return body


class _Run:
    """A runtime plus everything it emitted."""

    def __init__(self, playbook: PlaybookV2, *, start: bool = True) -> None:
        self.events: list[dict[str, Any]] = []
        self.rt = PlaybookRuntime(playbook, emit=self.events.append)
        if start:
            self.rt.start()

    def call(self, tool: str, result: Any = '{"ok": true}') -> None:
        assert self.rt._admit(tool) == ""
        self.rt._credit(tool, result)


def _triage(run: _Run, *branches: str) -> None:
    run.call("lookup_order")
    result = run.rt.complete_step(
        "triage",
        {
            "dispute_kind": "duplicate",
            "selected_branch_ids": list(branches),
            "customer_statement": "charged twice",
        },
    )
    assert result["ok"]


def _restored(playbook: PlaybookV2, events: list[dict[str, Any]]) -> _Run:
    fresh = _Run(playbook, start=False)
    result = fresh.rt.restore(copy.deepcopy(events))
    assert isinstance(result, RestoreResult)
    assert fresh.events == []  # restoring emits nothing
    assert fresh.rt.restored is True
    return fresh


def _assert_same(live: PlaybookRuntime, back: PlaybookRuntime) -> None:
    assert back.graph.status == live.graph.status
    assert [s.id for s in back.active_steps()] == [s.id for s in live.active_steps()]
    for tool in TOOLS:
        owner, restored_owner = live.owner_of(tool), back.owner_of(tool)
        assert (restored_owner.id if restored_owner else None) == (owner.id if owner else None)
    for step_id, work in live._work.items():
        again = back._work[step_id]
        assert len(again.executed) == len(work.executed), step_id
        assert again.attempts == work.attempts, step_id
        assert again.deviated == work.deviated, step_id
    assert back.graph.waived_because == live.graph.waived_because
    assert back.graph.outputs == live.graph.outputs
    assert back.graph.routed == live.graph.routed


# ── the round trip ───────────────────────────────────────────────────────────


def test_a_just_started_run_restores_to_its_first_step() -> None:
    pb = parse_playbook_v2(example())
    live = _Run(pb)
    back = _restored(pb, live.events)
    _assert_same(live.rt, back.rt)
    assert [s.id for s in back.rt.active_steps()] == ["triage"]


def test_a_routed_run_restores_and_then_runs_on_exactly_as_the_live_one() -> None:
    pb = parse_playbook_v2(example())
    live = _Run(pb)
    _triage(live, "duplicate_charge", "amount_mismatch")
    live.call("lookup_order")
    live.rt.complete_step("check_duplicates", {"duplicate_charges": ["ch_2"]})

    back = _restored(pb, live.events)
    _assert_same(live.rt, back.rt)
    assert back.rt.graph.status["check_amount"] == ACTIVE
    assert back.rt.graph.status["decide"] == PENDING
    owner = back.rt.owner_of("get_balance")
    assert owner is not None
    assert owner.id == "check_amount"

    # From here on the two emit the same events, step for step.
    mark = len(live.events)
    for run in (live, back):
        run.call("get_balance")
        run.rt.complete_step("check_amount", {"amount_delta": 0})
        run.rt.submit_decision("REFUND_DUPLICATE", "charged twice", ["ch_2"])
    assert back.events == live.events[mark:]
    assert back.rt.graph.status == live.rt.graph.status
    assert back.rt.outcome == live.rt.outcome


def test_waived_steps_and_their_reasons_are_restored() -> None:
    pb = parse_playbook_v2(example())
    live = _Run(pb)
    _triage(live, "duplicate_charge")
    back = _restored(pb, live.events)
    _assert_same(live.rt, back.rt)
    assert back.rt.graph.status["check_amount"] == WAIVED
    assert back.rt.graph.waived_because["check_amount"] == "no branch that names it was taken"
    assert back.rt.graph.outputs["triage"]["dispute_kind"] == "duplicate"


def test_per_step_counts_and_the_call_ceiling_survive() -> None:
    pb = parse_playbook_v2(example())
    live = _Run(pb)
    for _ in range(3):
        live.call("lookup_order")  # triage's ceiling is 3
    live.rt.complete_step(
        "triage",
        {"dispute_kind": "x", "selected_branch_ids": [], "customer_statement": "s"},
    )
    # triage closed; its count is in its done record.
    back = _restored(pb, live.events)
    _assert_same(live.rt, back.rt)
    assert back.rt._work["triage"].attempts == 3


def test_a_deviated_step_stays_unverified() -> None:
    pb = parse_playbook_v2(example())
    live = _Run(pb)
    live.call("get_balance")  # outside triage: a recorded deviation
    live.rt.pause("waiting on a person")  # a record of triage after the deviation
    back = _restored(pb, live.events)
    _assert_same(live.rt, back.rt)
    assert back.rt._work["triage"].deviated is True
    mark = len(live.events)
    live.rt.unpause()
    back.rt.unpause()
    assert back.events == live.events[mark:]
    assert back.events[-1]["verified"] is False


def test_a_paused_run_restores_blocked() -> None:
    pb = parse_playbook_v2(example())
    live = _Run(pb)
    live.rt.pause("waiting on a person")
    back = _restored(pb, live.events)
    _assert_same(live.rt, back.rt)
    assert back.rt.graph.status["triage"] == BLOCKED
    assert [s.id for s in back.rt.active_steps()] == ["triage"]


def test_a_decided_run_restores_concluded() -> None:
    pb = parse_playbook_v2(example())
    live = _Run(pb)
    _triage(live, "duplicate_charge")
    live.call("lookup_order")
    live.rt.complete_step("check_duplicates", {"duplicate_charges": ["ch_2"]})
    live.rt.submit_decision("REFUND_DUPLICATE", "charged twice", ["ch_2"])
    back = _restored(pb, live.events)
    _assert_same(live.rt, back.rt)
    assert back.rt.graph.status["decide"] == DONE
    assert back.rt.active_steps() == []


def test_start_after_restore_records_no_fresh_tree() -> None:
    pb = parse_playbook_v2(example())
    live = _Run(pb)
    _triage(live, "duplicate_charge")
    back = _restored(pb, live.events)
    back.rt.start()
    assert back.events == []
    assert back.rt.graph.status["check_duplicates"] == ACTIVE


def test_only_step_records_are_read_and_without_type_they_still_are() -> None:
    pb = parse_playbook_v2(example())
    live = _Run(pb)
    live.call("get_balance")  # a deviation among the step records
    live.rt.pause("waiting on a person")  # and a record of triage that carries it
    assert any(e["type"] != STEP_EVENT for e in live.events)
    stripped = [
        {k: v for k, v in e.items() if k != "type"} for e in live.events if e["type"] == STEP_EVENT
    ]
    back = _Run(pb, start=False)
    result = back.rt.restore(stripped)
    assert result.records == len(stripped)
    assert result.statuses == live.rt.graph.status
    assert result.active == ("triage",)
    assert result.statuses["triage"] == BLOCKED
    _assert_same(live.rt, back.rt)


def test_a_restored_runtime_starts_from_fresh_state() -> None:
    pb = parse_playbook_v2(example())
    live = _Run(pb)
    before = list(live.events)
    _triage(live, "duplicate_charge")
    # Restoring over a runtime that moved on puts it back where the records say.
    live.rt.restore(before)
    assert [s.id for s in live.rt.active_steps()] == ["triage"]
    assert live.rt._work["triage"].attempts == 0
    assert live.rt.graph.outputs == {}


# ── routing: the record's own enabled targets ────────────────────────────────


def _two_routers() -> dict[str, Any]:
    """``a`` and ``b`` both branch to ``t``: ``t`` is taken if either takes it."""

    def router(step_id: str, after: list[str]) -> dict[str, Any]:
        return {
            "id": step_id,
            "after": after,
            "allowed_tools": [],
            "expected_outputs": ["go"],
            "branches": [{"id": "go", "when": "go equals yes", "next_step_id": "t"}],
        }

    return {
        "version": "playbook.v2",
        "id": "routers",
        "title": "Routers",
        "summary": "s",
        "step_groups": [
            {
                "id": "g",
                "title": "g",
                "steps": [
                    router("a", []),
                    router("b", ["a"]),
                    {"id": "t", "required": False, "allowed_tools": []},
                ],
            }
        ],
    }


def test_a_target_another_router_may_still_take_is_not_taken_by_this_one() -> None:
    pb = parse_playbook_v2(_two_routers())
    live = _Run(pb)
    live.rt.complete_step("a", {"go": "no"})  # a does not take t; b may still
    assert live.rt.graph.status["t"] == PENDING
    back = _restored(pb, live.events)
    _assert_same(live.rt, back.rt)
    assert back.rt.graph.routed["a"] == set()
    mark = len(live.events)
    for run in (live, back):
        run.rt.complete_step("b", {"go": "no"})
    assert back.events == live.events[mark:]
    assert back.rt.graph.status["t"] == WAIVED


def test_a_record_without_its_enabled_targets_routes_by_what_was_not_waived() -> None:
    pb = parse_playbook_v2(example())
    live = _Run(pb)
    _triage(live, "duplicate_charge")
    events = [
        {k: v for k, v in e.items() if k != "enabled_steps"} for e in copy.deepcopy(live.events)
    ]
    back = _Run(pb, start=False)
    back.rt.restore(events)
    assert back.rt.graph.routed["triage"] == {"check_duplicates"}
    _assert_same(live.rt, back.rt)


# ── refusals: all or nothing ─────────────────────────────────────────────────


def _record(step: str, status: str = PENDING, playbook: str = "refund-dispute") -> dict[str, Any]:
    return {
        "type": STEP_EVENT,
        "playbook": playbook,
        "step_id": step,
        "step": step,
        "status": status,
    }


def _all_pending() -> list[dict[str, Any]]:
    return [_record(s) for s in ("triage", "check_duplicates", "check_amount", "decide")]


@pytest.mark.parametrize(
    ("events", "reason"),
    [
        ([], "the trace has no step records"),
        (
            [{"type": "playbook_deviation", "playbook": "refund-dispute", "step": "triage"}],
            "the trace has no step records",
        ),
        ([*_all_pending(), "not a record"], "a step record without its fields"),
        (
            [*_all_pending(), _record("triage", ACTIVE, playbook="other")],
            "a step record of another playbook ('other')",
        ),
        (
            [*_all_pending(), _record("ghost", ACTIVE)],
            "a step record for a step this playbook does not have ('ghost')",
        ),
        (
            [*_all_pending(), _record("triage", "running")],
            "a step record with an unknown status ('running')",
        ),
        (
            [_record("triage", ACTIVE), _record("decide")],
            "the trace has no record of step(s) check_amount, check_duplicates",
        ),
    ],
)
def test_a_trace_that_does_not_read_is_refused_and_nothing_changes(
    events: list[Any], reason: str
) -> None:
    pb = parse_playbook_v2(example())
    run = _Run(pb, start=False)
    graph, work = run.rt.graph, run.rt._work
    with pytest.raises(RestoreError) as caught:
        run.rt.restore(events)
    assert caught.value.reason == reason
    assert str(caught.value) == reason
    assert run.rt.restored is False
    assert run.rt.graph is graph
    assert run.rt._work is work
    assert set(run.rt.graph.status.values()) == {PENDING}
    assert run.events == []
    # Not started either: start() still records the step tree.
    run.rt.start()
    assert len(run.events) == 5  # four pending, then triage active


def test_a_refusal_leaves_a_running_runtime_where_it_was() -> None:
    pb = parse_playbook_v2(example())
    live = _Run(pb)
    _triage(live, "duplicate_charge")
    status = dict(live.rt.graph.status)
    bad = [*copy.deepcopy(live.events), _record("decide", "nope")]
    with pytest.raises(RestoreError):
        live.rt.restore(bad)
    assert live.rt.graph.status == status
    assert live.rt.restored is False


def test_hold_unstarted_records_nothing_and_activates_nothing() -> None:
    pb = parse_playbook_v2(example())
    run = _Run(pb, start=False)
    run.rt.hold_unstarted()
    run.rt.start()
    assert run.events == []
    assert run.rt.active_steps() == []
    assert run.rt.owner_of("lookup_order") is None
    assert run.rt.restored is False
