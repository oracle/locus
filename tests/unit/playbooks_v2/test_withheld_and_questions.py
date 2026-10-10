# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit: a restored run never mistakes a digest for data; declared questions must be answered.

Under the gateway's ``metadata_only`` custody a mirrored ``playbook_step`` record carries a
digest where a step's ``outputs`` were (``tulip_gateway.persist._digest``, copied below).
A runtime restored from such a trace must not route on the digest: the output is
UNAVAILABLE, a branch that reads it is ``unknown``, and ``complete_step`` refuses to route
-- unless the caller supplies the real values, checked against the digest. A step that
declares questions (``required_from_user``) closes only with each answer in its outputs,
and those questions are not the agent's own when an ``ask_user`` budget counts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import pytest
import yaml

from tulip.playbooks.v2 import (
    ROUTING_UNKNOWN,
    UNAVAILABLE,
    UNKNOWN,
    RestoreError,
    evaluate_when,
    is_digest,
    is_withheld,
    when_verdict,
)
from tulip.playbooks.v2.engine import (
    ACTIVE,
    DONE,
    NOT_EXECUTED,
    PENDING,
    WAIVED,
    PlaybookRuntime,
    PlaybookV2,
    parse_playbook_v2,
)
from tulip.playbooks.v2.results import not_executed_result
from tulip.playbooks.v2.when import Unavailable


EXAMPLE = Path(__file__).parent / "fixtures" / "refund-dispute.v2.yaml"


# ── the gateway's metadata_only mirror, as tulip_gateway.persist writes it ───


def _digest(value: Any) -> dict[str, Any]:
    """``tulip_gateway.persist._digest``, byte for byte."""
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return {"redacted": True, "sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded)}


#: The keys of a ``playbook_step`` record ``redact_event`` lets through as they are.
_SAFE = frozenset(
    {
        "type",
        "playbook",
        "step",
        "index",
        "status",
        "required",
        "reason",
        "version",
        "step_id",
        "group",
        "parallel_group",
        "tool_calls",
        "routed_by",
        "expected_tools",
        "allowed_tools",
        "branches_taken",
        "branches_not_taken",
        "enabled_steps",
    }
)


def metadata_only(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The trace as the control plane holds it: every other field a digest."""
    return [
        {k: v if v is None or k in _SAFE else _digest(v) for k, v in event.items()}
        for event in copy.deepcopy(events)
    ]


def outputs_digested(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only ``outputs`` digested: the steps' other fields as a full-custody trace has them."""
    return [
        {k: _digest(v) if k == "outputs" else v for k, v in event.items()}
        for event in copy.deepcopy(events)
    ]


# ── playbooks ────────────────────────────────────────────────────────────────


def _reads_earlier() -> dict[str, Any]:
    """``b`` routes on what ``a`` found: ``outputs.a.kind``."""
    return {
        "version": "playbook.v2",
        "id": "earlier",
        "title": "Earlier",
        "summary": "s",
        "step_groups": [
            {
                "id": "g",
                "title": "g",
                "steps": [
                    {"id": "a", "allowed_tools": ["probe"], "expected_outputs": ["kind"]},
                    {
                        "id": "b",
                        "after": ["a"],
                        "allowed_tools": [],
                        "expected_outputs": ["note"],
                        "branches": [
                            {"id": "big", "when": "outputs.a.kind == 'big'", "next_step_id": "c"},
                            {
                                "id": "small",
                                "when": "outputs.a.kind == 'small'",
                                "next_step_id": "d",
                            },
                        ],
                    },
                    {"id": "c", "required": False, "allowed_tools": []},
                    {"id": "d", "required": False, "allowed_tools": []},
                ],
            }
        ],
    }


class _Run:
    def __init__(self, playbook: PlaybookV2, *, start: bool = True) -> None:
        self.events: list[dict[str, Any]] = []
        self.rt = PlaybookRuntime(playbook, emit=self.events.append)
        if start:
            self.rt.start()

    def call(self, tool: str, result: Any = '{"ok": true}') -> None:
        assert self.rt._admit(tool) == ""
        self.rt._credit(tool, result)


def _a_found_big() -> tuple[PlaybookV2, _Run]:
    pb = parse_playbook_v2(_reads_earlier())
    live = _Run(pb)
    live.call("probe")
    assert live.rt.complete_step("a", {"kind": "big"})["ok"]
    assert live.rt.graph.status["b"] == ACTIVE
    return pb, live


# ── the when language: withheld is unknown ───────────────────────────────────


def test_the_gateway_digest_reads_as_a_digest() -> None:
    assert is_digest(_digest({"kind": "big"}))
    assert is_withheld(_digest("x"))
    assert is_withheld(UNAVAILABLE)
    assert copy.deepcopy(UNAVAILABLE) is UNAVAILABLE
    assert copy.copy(UNAVAILABLE) is UNAVAILABLE
    assert pickle.loads(pickle.dumps(UNAVAILABLE)) is UNAVAILABLE  # noqa: S301 -- our own bytes
    assert repr(UNAVAILABLE) == "UNAVAILABLE"
    assert Unavailable() is UNAVAILABLE
    for value in ({"redacted": False, "sha256": "ab"}, {"sha256": "ab"}, "redacted", None, {}):
        assert not is_withheld(value)


@pytest.mark.parametrize(
    "context",
    [
        {"outputs": {"a": UNAVAILABLE}},  # through it
        {"outputs": {"a": _digest({"kind": "big"})}},  # through a digest
        {"outputs": {"a": {"kind": UNAVAILABLE}}},  # at it
        {"outputs": {"a": {"kind": ["big", _digest("x")]}}},  # holding one
    ],
)
def test_a_condition_that_reads_a_withheld_value_is_unknown(context: dict[str, Any]) -> None:
    assert when_verdict("outputs.a.kind == 'big'", context) == UNKNOWN
    # The whole condition: the rest of it does not decide for the withheld part.
    assert when_verdict("always OR outputs.a.kind == 'big'", context) == UNKNOWN
    assert when_verdict("NOT (outputs.a.kind is empty)", context) == UNKNOWN
    assert evaluate_when("outputs.a.kind == 'big'", context) is False


def test_a_value_too_deep_to_look_through_is_not_known_whole() -> None:
    deep: Any = "big"
    for _ in range(100):
        deep = [deep]
    assert when_verdict("v is empty", {"v": deep}) == UNKNOWN
    assert when_verdict("v is empty", {"v": [[["big"]]]}) == "false"


def test_a_condition_that_reads_only_what_it_can_see_is_decided() -> None:
    context = {"outputs": {"a": UNAVAILABLE, "b": {"kind": "big"}}, "n": 3}
    assert when_verdict("outputs.b.kind == 'big'", context) == "true"
    assert when_verdict("n > 5", context) == "false"
    assert when_verdict("missing.path is empty", context) == "true"  # null is known
    assert when_verdict("always", context) == "true"
    assert when_verdict("((", context) == "false"


# ── restore: a digest is never data ──────────────────────────────────────────


def test_a_digested_output_restores_unavailable() -> None:
    pb, live = _a_found_big()
    back = _Run(pb, start=False)
    result = back.rt.restore(metadata_only(live.events))
    assert result.unavailable == ("a",)
    assert back.rt.unavailable_outputs() == {"a"}
    assert "a" not in back.rt.graph.outputs  # the digest is not put in as if it were outputs
    assert back.rt.graph.status == live.rt.graph.status
    step = pb.step("b")
    assert step is not None
    assert back.rt.graph.branch_verdicts(step, {"note": "n"}) == {
        "big": UNKNOWN,
        "small": UNKNOWN,
    }
    # The live run, which has the values, decides.
    assert live.rt.graph.branch_verdicts(step, {"note": "n"}) == {"big": "true", "small": "false"}
    assert live.rt.unavailable_outputs() == set()


def test_unknown_routing_refuses_to_close_and_changes_nothing() -> None:
    pb, live = _a_found_big()
    back = _Run(pb, start=False)
    back.rt.restore(metadata_only(live.events))
    result = back.rt.complete_step("b", {"note": "n"})
    assert result["ok"] is False
    assert result["routing"] == ROUTING_UNKNOWN == "unknown"
    assert result["branches_unknown"] == ["big", "small"]
    assert result["unavailable"] == ["a"]
    assert "a person must choose" in result["error"].lower()
    assert back.events == []
    assert back.rt.graph.status["b"] == ACTIVE
    assert back.rt.graph.status["c"] == back.rt.graph.status["d"] == PENDING
    assert "b" not in back.rt.graph.routed
    assert "b" not in back.rt.graph.outputs


def test_route_never_settles_on_an_unknown_branch() -> None:
    pb, live = _a_found_big()
    back = _Run(pb, start=False)
    back.rt.restore(metadata_only(live.events))
    step = pb.step("b")
    assert step is not None
    back.rt.graph.outputs["b"] = {"note": "n"}
    assert back.rt.graph.route(step) == ([], [], UNKNOWN)
    assert "b" not in back.rt.graph.routed
    back.rt.graph.status["b"] = DONE
    back.rt.graph.advance()
    assert back.rt.graph.status["c"] == back.rt.graph.status["d"] == PENDING  # nothing waived


def test_an_explicit_selection_still_routes() -> None:
    pb, live = _a_found_big()
    back = _Run(pb, start=False)
    back.rt.restore(metadata_only(live.events))
    assert back.rt.select_branches("b", ["big"], "the person said so")["ok"]
    result = back.rt.complete_step("b", {"note": "n"})
    assert result["ok"]
    assert result["routed_by"] == "select_branches"
    assert back.rt.graph.status["c"] == ACTIVE
    assert back.rt.graph.status["d"] == WAIVED


def test_supplied_outputs_route_exactly_as_the_original() -> None:
    pb, live = _a_found_big()
    back = _Run(pb, start=False)
    result = back.rt.restore(outputs_digested(live.events), outputs={"a": {"kind": "big"}})
    assert result.unavailable == ()
    assert back.rt.unavailable_outputs() == set()
    assert back.rt.graph.outputs == live.rt.graph.outputs
    mark = len(live.events)
    for run in (live, back):
        assert run.rt.complete_step("b", {"note": "n"})["ok"]
    assert back.events == live.events[mark:]
    assert back.rt.graph.status["c"] == ACTIVE
    assert back.rt.graph.status["d"] == WAIVED


def test_supplied_outputs_that_are_not_the_recorded_ones_are_refused() -> None:
    pb, live = _a_found_big()
    back = _Run(pb, start=False)
    graph = back.rt.graph
    with pytest.raises(RestoreError) as caught:
        back.rt.restore(metadata_only(live.events), outputs={"a": {"kind": "small"}})
    assert (
        caught.value.reason == "the outputs supplied for step a are not the ones its record holds"
    )
    assert back.rt.graph is graph
    assert back.rt.restored is False
    # Against a plain record too.
    with pytest.raises(RestoreError):
        back.rt.restore(copy.deepcopy(live.events), outputs={"a": {"kind": "small"}})
    assert back.rt.restored is False
    # The same values a plain record holds are accepted.
    back.rt.restore(copy.deepcopy(live.events), outputs={"a": {"kind": "big"}})
    assert back.rt.graph.outputs["a"] == {"kind": "big"}


@pytest.mark.parametrize(
    ("outputs", "reason"),
    [
        ({"ghost": {}}, "outputs supplied for a step this playbook does not have ('ghost')"),
        ({"a": ["big"]}, "the outputs supplied for step a are not a mapping"),
    ],
)
def test_supplied_outputs_that_do_not_read_are_refused(outputs: Any, reason: str) -> None:
    pb, live = _a_found_big()
    back = _Run(pb, start=False)
    with pytest.raises(RestoreError) as caught:
        back.rt.restore(metadata_only(live.events), outputs=outputs)
    assert caught.value.reason == reason
    assert back.rt.restored is False


def test_supplied_outputs_for_a_step_not_closed_are_ignored() -> None:
    pb, live = _a_found_big()
    back = _Run(pb, start=False)
    back.rt.restore(metadata_only(live.events), outputs={"a": {"kind": "big"}, "b": {"x": 1}})
    assert "b" not in back.rt.graph.outputs


def test_a_digested_value_inside_plain_outputs_is_unavailable() -> None:
    pb, live = _a_found_big()
    events = copy.deepcopy(live.events)
    for event in events:
        if event.get("step_id") == "a" and event["status"] == DONE:
            event["outputs"] = {"kind": _digest("big"), "seen": True}
    back = _Run(pb, start=False)
    back.rt.restore(events)
    assert back.rt.graph.outputs["a"] == {"kind": UNAVAILABLE, "seen": True}
    assert back.rt.unavailable_outputs() == {"a"}
    assert back.rt.complete_step("b", {"note": "n"})["routing"] == UNKNOWN


def test_a_not_executed_step_s_outputs_are_not_in_its_record() -> None:
    pb = parse_playbook_v2(_reads_earlier())
    live = _Run(pb)
    live.call("probe", not_executed_result("probe", "label-only tool", {}))
    closed = live.rt.complete_step("a", {"kind": "big"})
    assert closed["status"] == NOT_EXECUTED
    back = _Run(pb, start=False)
    assert back.rt.restore(copy.deepcopy(live.events)).unavailable == ("a",)
    assert back.rt.complete_step("b", {"note": "n"})["routing"] == UNKNOWN
    # With the values supplied (nothing in the record to check them against) it routes.
    again = _Run(pb, start=False)
    again.rt.restore(copy.deepcopy(live.events), outputs={"a": {"kind": "big"}})
    assert again.rt.complete_step("b", {"note": "n"})["branches_taken"] == ["big"]


def test_a_digested_verified_is_not_a_yes() -> None:
    pb, live = _a_found_big()
    back = _Run(pb, start=False)
    back.rt.restore(metadata_only(live.events))
    assert back.rt._work["a"].deviated is True
    plain = _Run(pb, start=False)
    plain.rt.restore(copy.deepcopy(live.events))
    assert plain.rt._work["a"].deviated is False


def test_a_run_already_routed_carries_on_under_metadata_only() -> None:
    """The refund example: triage routed before the move, so its digest is never read."""
    pb = parse_playbook_v2(yaml.safe_load(EXAMPLE.read_text("utf-8")))
    live = _Run(pb)
    live.call("lookup_order")
    assert live.rt.complete_step(
        "triage",
        {
            "dispute_kind": "duplicate",
            "selected_branch_ids": ["duplicate_charge"],
            "customer_statement": "charged twice",
        },
    )["ok"]
    back = _Run(pb, start=False)
    back.rt.restore(metadata_only(live.events))
    assert back.rt.unavailable_outputs() == {"triage"}
    assert back.rt.graph.routed == live.rt.graph.routed
    mark = len(live.events)
    for run in (live, back):
        run.call("lookup_order")
        assert run.rt.complete_step("check_duplicates", {"duplicate_charges": ["ch_2"]})["ok"]
    # The same events, but for ``verified``: the digested record cannot vouch for the step.
    unverified = [{**e, "verified": False} for e in live.events[mark:]]
    assert back.events == unverified
    assert live.events[mark]["verified"] is True
    assert back.rt.graph.status == live.rt.graph.status
    assert back.rt.graph.status["decide"] == ACTIVE


# ── declared questions ───────────────────────────────────────────────────────


def _refund() -> _Run:
    return _Run(parse_playbook_v2(yaml.safe_load(EXAMPLE.read_text("utf-8"))))


@pytest.mark.parametrize("blank", [None, "", "   ", [], {}])
def test_a_declared_question_closes_only_with_its_answer(blank: Any) -> None:
    run = _refund()
    run.call("lookup_order")
    run.call("ask_user")  # asking once is not the answer
    owed = {"dispute_kind": "x", "selected_branch_ids": []}
    result = run.rt.complete_step("triage", {**owed, "customer_statement": blank})
    assert result["ok"] is False
    assert result["missing"] == ["customer_statement"]
    assert "customer_statement" in result["error"]
    assert run.rt.graph.status["triage"] == ACTIVE
    assert run.rt.complete_step("triage", {**owed, "customer_statement": "twice"})["ok"]


def test_a_false_or_zero_answer_is_an_answer() -> None:
    for answer in (False, 0):
        run = _refund()
        run.call("lookup_order")
        result = run.rt.complete_step(
            "triage",
            {"dispute_kind": "x", "selected_branch_ids": [], "customer_statement": answer},
        )
        assert result["ok"]


def test_declared_questions_are_named_per_step() -> None:
    run = _refund()
    assert run.rt.declared_question("triage", "customer_statement") is True
    assert run.rt.declared_question("triage", "dispute_kind") is False
    assert run.rt.declared_question("check_amount", "customer_statement") is False
    assert run.rt.declared_question("ghost", "customer_statement") is False


def test_only_the_declared_asks_are_left_out_of_the_budget() -> None:
    run = _refund()
    # triage declares one question: the first ask is the playbook's, the next the agent's.
    assert run.rt.ask_is_declared() is True
    run.call("ask_user")
    assert run.rt._work["triage"].asks == 1
    assert run.rt.ask_is_declared() is False
    run.call("ask_user")
    assert run.rt.ask_is_declared() is False
    # A step that declares nothing: every ask is the agent's own.
    run.call("lookup_order")
    run.rt.complete_step(
        "triage",
        {
            "dispute_kind": "x",
            "selected_branch_ids": ["amount_mismatch"],
            "customer_statement": "y",
        },
    )
    assert run.rt.graph.status["check_amount"] == ACTIVE
    assert run.rt.ask_is_declared() is False


def test_a_declared_ask_is_attributed_to_the_step_that_declares_it() -> None:
    body = {
        "version": "playbook.v2",
        "id": "asks",
        "title": "Asks",
        "summary": "s",
        "step_groups": [
            {
                "id": "g",
                "title": "g",
                "steps": [
                    {"id": "open", "parallel_group": "p"},  # the agent's tools, ask_user too
                    {
                        "id": "asks",
                        "parallel_group": "p",
                        "allowed_tools": [],
                        "required_from_user": [{"name": "who", "question": "Who?"}],
                    },
                ],
            }
        ],
    }
    run = _Run(parse_playbook_v2(body))
    assert [s.id for s in run.rt.active_steps()] == ["open", "asks"]
    assert run.rt.ask_is_declared() is True
    run.call("ask_user")
    assert run.rt._work["asks"].asks == 1
    assert run.rt._work["open"].asks == 0
    assert run.rt.ask_is_declared() is False
    run.call("ask_user")
    assert run.rt._work["open"].asks == 1  # the agent's own question, as before
