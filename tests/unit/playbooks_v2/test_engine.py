# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit: a v2 playbook as the engine runs it -- graph, branches, allowlists, decision.

Ported from the gateway's ``tests/unit/test_playbook_v2.py`` with the engine. The
registry's ``examples/playbooks/refund-dispute.v2.yaml`` is the fixture (copied into
``fixtures/``): triage routes to one or both evidence steps of a parallel group, the
decision step joins that group, and one outcome concludes it.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from tulip.playbooks.v2.engine import (
    ACTIVE,
    BLOCKED,
    DECISION_EVENT,
    DONE,
    NOT_EXECUTED,
    PENDING,
    STEP_EVENT,
    WAIVED,
    PlaybookRuntime,
    PlaybookV2Error,
    StepGraph,
    control_tool,
    definition_digest,
    enforcement_mode,
    forbidden_deny,
    initial_active,
    is_playbook_v2,
    parse_playbook_v2,
    playbook_prose,
    step_brief,
    with_forbidden,
)
from tulip.playbooks.v2.results import not_executed_result


EXAMPLE = Path(__file__).parent / "fixtures" / "refund-dispute.v2.yaml"


def example() -> dict[str, Any]:
    body: dict[str, Any] = yaml.safe_load(EXAMPLE.read_text("utf-8"))
    return body


def _step(body: dict[str, Any], step_id: str) -> dict[str, Any]:
    for group in body["step_groups"]:
        for step in group["steps"]:
            if step["id"] == step_id:
                return dict(step)
    raise KeyError(step_id)


def _set_step(body: dict[str, Any], step_id: str, **fields: Any) -> dict[str, Any]:
    for group in body["step_groups"]:
        for step in group["steps"]:
            if step["id"] == step_id:
                step.update(fields)
    return body


class _Audit:
    """The chain's side of ``record``: every (kind, fields) the runtime hands over."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def record(self, event: str, fields: dict[str, Any]) -> None:
        self.events.append((event, fields))


class _Run:
    """A runtime plus the events it emitted."""

    def __init__(self, body: dict[str, Any] | None = None, *, enforce: bool = False) -> None:
        self.events: list[dict[str, Any]] = []
        self.audit = _Audit()
        self.rt = PlaybookRuntime(
            parse_playbook_v2(body or example()),
            emit=self.events.append,
            skills={"refund-checklist": {"name": "refund-checklist", "instructions": "CHECK"}},
            enforce=enforce,
            ref={"version": "3", "digest": "sha256:abc"},
            record=self.audit.record,
            correlation_id="run-9",
            principal="gw",
        )
        self.rt.start()

    def of(self, etype: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["type"] == etype]

    def status(self, step: str) -> str:
        return self.rt.graph.status[step]

    def call(self, tool: str, result: Any = '{"ok": true}') -> str:
        refusal = self.rt._admit(tool)
        if refusal:
            return refusal
        self.rt._credit(tool, result)
        return ""

    def triage(self, **outputs: Any) -> dict[str, Any]:
        self.call("lookup_order")
        return self.rt.complete_step(
            "triage",
            {
                "dispute_kind": "duplicate",
                "selected_branch_ids": ["duplicate_charge"],
                "customer_statement": "charged twice",
                **outputs,
            },
        )


# ── reading the definition ───────────────────────────────────────────────────


def test_the_example_reads_as_a_graph() -> None:
    pb = parse_playbook_v2(example())
    assert [s.id for s in pb.steps] == ["triage", "check_duplicates", "check_amount", "decide"]
    assert next(o.id for o in pb.outcomes) == "INCONCLUSIVE"
    assert pb.outcome("INCONCLUSIVE") is not None
    assert pb.outcome("INCONCLUSIVE").inconclusive
    assert pb.step("decide") is not None
    assert pb.step("decide").decides
    assert [s.id for s in pb.decision_steps] == ["decide"]
    assert pb.recommendation_only
    assert pb.mode == "diagnosis"
    assert pb.inputs == (("order_id", "The order the customer disputes."),)
    assert pb.step("nope") is None
    assert pb.outcome("nope") is None
    assert is_playbook_v2(example())
    assert not is_playbook_v2({"steps": []})


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda b: b.update(version="playbook.v1"), "not a playbook.v2"),
        (lambda b: b.update(step_groups=[]), "no steps"),
        (lambda b: _set_step(b, "decide", after=["ghost"]), "do not exist"),
        (lambda b: _set_step(b, "decide", joins="ghost-group"), "which no step belongs to"),
        (lambda b: _set_step(b, "check_amount", id="check_duplicates"), "repeat"),
        (lambda b: _set_step(b, "decide", id=""), "has no id"),
        (
            lambda b: _set_step(
                b, "triage", branches=[{"id": "x", "when": "a ==", "next_step_id": "decide"}]
            ),
            "when",
        ),
    ],
)
def test_a_playbook_the_runtime_cannot_run_is_refused(mutate: Any, message: str) -> None:
    body = example()
    mutate(body)
    with pytest.raises(PlaybookV2Error, match=message):
        parse_playbook_v2(body)


def test_loose_shapes_still_read() -> None:
    body = {
        "version": "playbook.v2",
        "id": "loose",
        "step_groups": [
            "not a group",
            {"id": "g", "steps": ["not a step", {"id": "a", "max_tool_calls": True}]},
        ],
        "decision_policy": "nope",
        "completion": "nope",
        "forbidden_actions": [{"label": "bulk", "reason": "r"}, {"reason": "nothing"}],
    }
    pb = parse_playbook_v2(body)
    assert [s.id for s in pb.steps] == ["a"]
    assert pb.steps[0].title == "a"
    assert pb.steps[0].max_tool_calls is None
    assert pb.outcomes == ()
    assert pb.all_required_steps_resolved
    assert [(f.target, f.kind) for f in pb.forbidden] == [("bulk", "label")]


def test_the_digest_is_the_registry_s_and_ignores_the_publish_stamp() -> None:
    body = example()
    stamped = copy.deepcopy(body)
    stamped["metadata"] = {"tulip.publish": {"x": 1}}
    unstamped = copy.deepcopy(body)
    unstamped["metadata"] = {}
    assert definition_digest(stamped) == definition_digest(unstamped)
    assert definition_digest(body).startswith("sha256:")


# ── forbidden actions: the registry's compile, the registry's merge ──────────


def test_forbidden_actions_compile_to_deny_rules_exactly_as_the_registry() -> None:
    """The registry's ``test_forbidden_actions_compile_to_deny_rules_only`` vector."""
    assert forbidden_deny(parse_playbook_v2(example())) == ["bulk", "issue_refund"]
    quiet = example()
    quiet["forbidden_actions"] = []
    assert forbidden_deny(parse_playbook_v2(quiet)) == []


def test_the_fragment_narrows_the_effective_policy_exactly_as_the_registry() -> None:
    """The registry's ``test_the_fragment_narrows_the_effective_policy_and_never_loosens_it``.

    ``effective_policy(bundle, inline, playbook=fragment)`` there; the gateway's merge
    then :func:`with_forbidden`. Same inputs, same answers. The two policies are the
    gateway's ``effective_policy(bundle, inline)`` and ``effective_policy(None, None)``
    for the registry vector's ``bundle = {"deny_for": ["delete"], "require_human_for":
    ["production"]}`` and ``inline = {"deny_for": ["wire"], "require_human_for":
    ["payment"], "max_blast_radius": 1}``, written out (the merge is the gateway's).
    """
    deny = forbidden_deny(parse_playbook_v2(example()))
    merged_base = {
        "require_verification_score": 0.0,
        "max_blast_radius": 1,
        "require_human_for": ["payment", "production"],
        "confirm_with_requester": [],
        "deny_for": ["delete", "wire"],
        "require_sandbox_for": [],
        "min_severity": "low",
    }
    default = {
        "require_verification_score": 0.0,
        "max_blast_radius": 1,
        "require_human_for": ["production"],
        "confirm_with_requester": [],
        "deny_for": [],
        "require_sandbox_for": [],
        "min_severity": "low",
    }
    merged = with_forbidden(merged_base, deny)
    assert merged["deny_for"] == ["bulk", "delete", "issue_refund", "wire"]
    assert merged["require_human_for"] == ["payment", "production"]
    assert with_forbidden(merged_base, []) == merged_base
    alone = with_forbidden(default, deny)
    assert alone["deny_for"] == ["bulk", "issue_refund"]
    assert alone["require_human_for"] == ["production"]
    # The inline-only path keeps a partial policy partial: deny_for is added, nothing else.
    assert with_forbidden({"max_blast_radius": 2}, deny) == {
        "max_blast_radius": 2,
        "deny_for": ["bulk", "issue_refund"],
    }


def test_enforcement_is_the_playbook_s_say_then_the_deployment_s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = example()
    assert enforcement_mode(parse_playbook_v2(body)) == "record"
    assert enforcement_mode(parse_playbook_v2(body), "block") == "block"
    assert enforcement_mode(parse_playbook_v2(body), "BLOCK") == "block"
    # The engine reads no environment: the deployment's default is passed in.
    monkeypatch.setenv("TULIP_GATEWAY_PLAYBOOK_ENFORCEMENT", "block")
    assert enforcement_mode(parse_playbook_v2(body)) == "record"
    body["metadata"] = {"enforcement": "record"}
    assert enforcement_mode(parse_playbook_v2(body), "block") == "record"
    body["metadata"] = {"enforcement": "Block"}
    assert enforcement_mode(parse_playbook_v2(body), "bogus") == "block"


# ── the scheduler ────────────────────────────────────────────────────────────


def test_only_the_root_is_active_at_the_start() -> None:
    assert initial_active(parse_playbook_v2(example())) == ["triage"]


def test_a_branch_taken_by_its_condition_enables_its_step_and_waives_the_other() -> None:
    run = _Run()
    result = run.triage()
    assert result["ok"]
    assert result["status"] == DONE
    assert result["routed_by"] == "when"
    assert result["branches_taken"] == ["duplicate"]
    assert result["branches_not_taken"] == ["amount"]
    assert run.status("check_amount") == WAIVED
    assert run.status("check_duplicates") == ACTIVE
    assert run.status("decide") == PENDING  # joins the group: waits for its last member
    assert "check_duplicates" in result["next"][0]
    done = [e for e in run.of("playbook_step") if e["status"] == DONE]
    assert done[0]["outputs"]["dispute_kind"] == "duplicate"
    assert done[0]["branches_taken"] == ["duplicate"]
    assert done[0]["enabled_steps"] == ["check_duplicates"]


def test_both_branches_put_the_parallel_group_active_together() -> None:
    run = _Run()
    run.triage(selected_branch_ids=["duplicate_charge", "amount_mismatch"])
    assert run.status("check_duplicates") == ACTIVE
    assert run.status("check_amount") == ACTIVE
    active = [e for e in run.of("playbook_step") if e["status"] == ACTIVE]
    assert {e["parallel_group"] for e in active[1:]} == {"evidence"}
    # A sibling closing does not release the join; the last one does.
    run.call("lookup_order")
    run.rt.complete_step("check_duplicates", {"duplicate_charges": []})
    assert run.status("decide") == PENDING
    run.call("get_balance")
    result = run.rt.complete_step("check_amount", {"amount_delta": 0})
    assert run.status("decide") == ACTIVE
    assert result["active"] == ["decide"]


def test_no_branch_taken_waives_the_group_and_the_join_still_opens() -> None:
    run = _Run()
    run.triage(selected_branch_ids=[])
    assert run.status("check_duplicates") == WAIVED
    assert run.status("check_amount") == WAIVED
    assert run.status("decide") == ACTIVE


def test_select_branches_routes_by_id_and_wins_over_the_outputs() -> None:
    run = _Run()
    picked = run.rt.select_branches("triage", ["amount"], "the totals differ")
    assert picked["ok"]
    assert picked["waives"] == ["duplicate"]
    result = run.triage()  # its outputs would have taken `duplicate`
    assert result["routed_by"] == "select_branches"
    assert run.status("check_amount") == ACTIVE
    assert run.status("check_duplicates") == WAIVED


def test_select_branches_is_validated() -> None:
    run = _Run()
    assert not run.rt.select_branches("decide", ["x"], "r")["ok"]  # no branches
    assert not run.rt.select_branches("ghost", ["x"], "r")["ok"]
    bad = run.rt.select_branches("triage", ["duplicate", "refund_now"], "r")
    assert not bad["ok"]
    assert bad["branches"] == ["duplicate", "amount"]
    run.triage()
    late = run.rt.select_branches("triage", ["amount"], "r")
    assert not late["ok"]
    assert "route it while active" in late["error"]


def test_a_step_after_a_waived_step_is_waived_too() -> None:
    body = example()
    body["step_groups"][1]["steps"].append(
        {"id": "follow_up", "title": "Follow up", "after": ["check_amount"], "required": False}
    )
    run = _Run(body)
    run.triage()
    assert run.status("follow_up") == WAIVED
    assert "after a step that was waived" in run.rt.graph.waived_because["follow_up"]


def test_independent_roots_run_one_at_a_time() -> None:
    body = {
        "version": "playbook.v2",
        "id": "two",
        "title": "Two",
        "summary": "s",
        "step_groups": [{"id": "g", "title": "g", "steps": [{"id": "a"}, {"id": "b"}]}],
    }
    graph = StepGraph(parse_playbook_v2(body))
    assert graph.advance() == ([], ["a"])
    graph.status["a"] = DONE
    assert graph.advance() == ([], ["b"])
    assert graph.converged() is False
    graph.status["b"] = DONE
    assert graph.converged() is True


# ── allowlists at the hook seam ──────────────────────────────────────────────


def test_a_tool_outside_the_active_step_is_recorded_by_default() -> None:
    run = _Run()
    assert run.call("get_balance") == ""  # runs: recording is the default
    deviation = run.of("playbook_deviation")[-1]
    assert deviation["violation"] == "unexpected_tool"
    assert deviation["step"] == "triage"
    assert deviation["blocked"] is False
    assert deviation["enforcement"] == "record"
    assert deviation["version"] == "playbook.v2"


def test_under_block_it_is_refused_with_what_is_allowed() -> None:
    run = _Run(enforce=True)
    refusal = run.call("get_balance")
    assert "not allowed in the active step" in refusal
    assert "lookup_order" in refusal
    assert run.of("playbook_deviation")[-1]["blocked"] is True


async def test_the_hook_cancels_the_call_under_block() -> None:
    class _Event:
        tool_name = "issue_refund"
        cancel: Any = False
        result = None
        error = None

    run = _Run(enforce=True)
    event = _Event()
    await run.rt.on_before_tool_call(event)  # type: ignore[arg-type]
    assert isinstance(event.cancel, str)
    assert "refund-dispute" in event.cancel
    allowed = _Event()
    allowed.tool_name = "lookup_order"
    await run.rt.on_before_tool_call(allowed)  # type: ignore[arg-type]
    assert allowed.cancel is False
    allowed.result = '{"ok": true}'
    await run.rt.on_after_tool_call(allowed)  # type: ignore[arg-type]
    assert run.rt._work["triage"].executed == ["lookup_order"]


def test_control_and_proposal_tools_are_always_allowed() -> None:
    run = _Run(enforce=True)
    for tool in ("complete_step", "select_branches", "submit_decision", "propose_skill"):
        assert run.call(tool) == ""
    assert run.of("playbook_deviation") == []


def test_null_allows_the_agent_s_tools_and_empty_allows_none() -> None:
    body = example()
    _set_step(body, "triage", allowed_tools=None, required_from_user=[])
    run = _Run(body, enforce=True)
    assert run.call("anything_the_agent_has") == ""
    assert run.call("ask_user") == ""
    body = example()
    _set_step(body, "triage", allowed_tools=[], required_from_user=[])
    run = _Run(body, enforce=True)
    assert run.call("lookup_order")
    assert run.call("ask_user")  # no answers owed, so no question either


def test_ask_user_is_allowed_where_answers_are_owed() -> None:
    run = _Run(enforce=True)
    assert run.call("ask_user") == ""
    assert run.rt._work["triage"].asked is True
    assert run.rt._work["triage"].attempts == 0  # asking is not one of the step's calls


def test_max_tool_calls_is_a_ceiling() -> None:
    run = _Run(enforce=True)
    for _ in range(3):
        assert run.call("lookup_order") == ""
    refusal = run.call("lookup_order")
    assert "at most 3" in refusal
    deviation = run.of("playbook_deviation")[-1]
    assert deviation["violation"] == "too_many_calls"
    assert deviation["limit"] == 3
    recorded = _Run()
    for _ in range(4):
        assert recorded.call("lookup_order") == ""
    assert recorded.of("playbook_deviation")[-1]["violation"] == "too_many_calls"


def test_a_refused_or_failed_call_is_not_the_step_s_work() -> None:
    run = _Run()
    run.rt._admit("lookup_order")
    run.rt._credit("lookup_order", "⛔ denied: no")
    run.rt._admit("lookup_order")
    run.rt._credit("lookup_order", None, error="boom")
    assert run.rt._work["triage"].executed == []
    run.rt._credit("never_admitted", "x")  # nothing to attribute it to
    assert run.rt._work["triage"].executed == []


def test_after_the_decision_nothing_else_belongs_to_the_playbook() -> None:
    run = _Run(enforce=True)
    run.rt.submit_decision("INCONCLUSIVE", "the order could not be read", [])
    assert "is concluded" in run.call("lookup_order")
    assert run.of("playbook_deviation")[-1]["violation"] == "playbook_complete"


def test_observe_counts_a_performed_call_and_its_not_executed_twin() -> None:
    run = _Run()
    run.rt.observe("lookup_order")
    assert run.rt._work["triage"].executed == ["lookup_order"]
    run.rt.observe("lookup_order", not_executed="label-only")
    assert run.rt._work["triage"].unrun == {"lookup_order": "label-only"}
    blocked = _Run(enforce=True)
    blocked.rt.observe("get_balance")
    assert blocked.rt._work["triage"].executed == []


# ── complete_step ────────────────────────────────────────────────────────────


def test_complete_step_names_the_outputs_still_owed() -> None:
    run = _Run()
    run.call("lookup_order")
    result = run.rt.complete_step("triage", {"dispute_kind": "x", "customer_statement": "y"})
    assert not result["ok"]
    assert result["missing"] == ["selected_branch_ids"]
    assert run.status("triage") == ACTIVE


def test_complete_step_wants_the_person_s_answers() -> None:
    run = _Run()
    run.call("lookup_order")
    result = run.rt.complete_step("triage", {"dispute_kind": "x", "selected_branch_ids": []})
    assert not result["ok"]
    assert result["missing"] == ["customer_statement"]
    run.call("ask_user")
    # Asking is not answering: the answer itself must be in the outputs.
    again = run.rt.complete_step("triage", {"dispute_kind": "x", "selected_branch_ids": []})
    assert not again["ok"]
    assert again["missing"] == ["customer_statement"]
    assert "customer_statement" in again["error"]
    owed = {"dispute_kind": "x", "selected_branch_ids": []}
    assert run.rt.complete_step("triage", {**owed, "customer_statement": "charged twice"})["ok"]


def test_complete_step_only_closes_an_active_step() -> None:
    run = _Run()
    assert "no step" in run.rt.complete_step("ghost", {})["error"]
    result = run.rt.complete_step("decide", {})
    assert not result["ok"]
    assert "pending, not active" in result["error"]
    assert result["active"] == ["triage"]


def test_a_step_whose_tools_ran_nothing_closes_not_executed() -> None:
    run = _Run()
    run.call("lookup_order", not_executed_result("lookup_order", "label-only tool", {}))
    assert run.of("playbook_deviation")[-1]["violation"] == "tool_not_executed"
    result = run.rt.complete_step(
        "triage",
        {"dispute_kind": "x", "selected_branch_ids": [], "customer_statement": "y"},
    )
    assert result["status"] == NOT_EXECUTED
    assert not result["ok"]
    assert "did not execute" in result["reason"]
    assert run.status("triage") == NOT_EXECUTED
    assert run.of("playbook_deviation")[-1]["violation"] == "required_step_not_executed"
    step = next(e for e in run.of("playbook_step") if e["status"] == NOT_EXECUTED)
    assert step["verified"] is False
    # The graph moves on rather than deadlocking the run.
    assert run.status("decide") == ACTIVE


def test_the_effort_floor_records_by_default_and_refuses_under_block() -> None:
    owed = {"dispute_kind": "x", "selected_branch_ids": [], "customer_statement": "y"}
    run = _Run()
    assert run.rt.complete_step("triage", owed)["ok"]
    deviation = run.of("playbook_deviation")[-1]
    assert deviation["violation"] == "insufficient_effort"
    assert deviation["calls_short"] == 1
    done = next(e for e in run.of("playbook_step") if e["status"] == DONE)
    assert done["verified"] is False
    blocked = _Run(enforce=True)
    refused = blocked.rt.complete_step("triage", owed)
    assert not refused["ok"]
    assert "1 more tool call" in refused["error"]
    assert blocked.status("triage") == ACTIVE


# ── submit_decision ──────────────────────────────────────────────────────────


def _to_decide(run: _Run) -> None:
    run.triage()
    run.call("lookup_order")
    run.rt.complete_step("check_duplicates", {"duplicate_charges": ["ch_2"]})


def test_the_decision_concludes_the_playbook() -> None:
    run = _Run()
    _to_decide(run)
    assert run.status("decide") == ACTIVE
    result = run.rt.submit_decision("REFUND_DUPLICATE", "charged twice", ["ch_1", "ch_2"])
    assert result["ok"]
    assert result["recommendation_only"] is True
    assert "do not act on it" in result["next"]
    assert run.status("decide") == DONE
    decision = run.of("playbook_decision")[0]
    assert decision["outcome_id"] == "REFUND_DUPLICATE"
    assert decision["evidence_refs"] == ["ch_1", "ch_2"]
    assert decision["converged"] is True
    assert decision["step"] == "decide"
    assert decision["action"] == "Recommend refunding the duplicate."
    assert run.rt.outcome == {"outcome_id": "REFUND_DUPLICATE", "classification": None}


def test_the_decision_is_validated() -> None:
    run = _Run()
    _to_decide(run)
    unknown = run.rt.submit_decision("REFUND_ALL", "x", [])
    assert not unknown["ok"]
    assert "INCONCLUSIVE" in unknown["outcomes"]
    assert not run.rt.submit_decision("NO_REFUND", "  ", [])["ok"]
    assert run.rt.submit_decision("NO_REFUND", "matches", [])["ok"]
    again = run.rt.submit_decision("NO_REFUND", "matches", [])
    assert not again["ok"]
    assert "already decided" in again["error"]
    body = example()
    del body["decision_policy"]
    assert "no decision policy" in _Run(body).rt.submit_decision("X", "y", [])["error"]


def test_a_premature_decision_is_recorded_or_refused() -> None:
    run = _Run()
    assert run.rt.submit_decision("NO_REFUND", "looks fine", [])["ok"]
    deviations = [e["violation"] for e in run.of("playbook_deviation")]
    assert "decision_before_convergence" in deviations
    assert "required_step_skipped" in deviations  # triage and decide were never done
    assert run.status("triage") == WAIVED
    assert run.of("playbook_decision")[0]["converged"] is False
    blocked = _Run(enforce=True)
    refused = blocked.rt.submit_decision("NO_REFUND", "looks fine", [])
    assert not refused["ok"]
    assert "triage" in refused["error"]
    assert blocked.of("playbook_decision") == []


def test_inconclusive_is_always_an_honest_answer() -> None:
    run = _Run(enforce=True)
    result = run.rt.submit_decision("INCONCLUSIVE", "the order could not be read", [])
    assert result["ok"]
    assert "required_step_skipped" not in [e["violation"] for e in run.of("playbook_deviation")]
    assert run.rt.outcome == {"outcome_id": "INCONCLUSIVE", "classification": "INCONCLUSIVE"}


def test_finish_records_what_was_left_open_and_the_missing_decision() -> None:
    run = _Run()
    run.rt.finish()
    run.rt.finish()  # once
    deviations = [(e["violation"], e["step"]) for e in run.of("playbook_deviation")]
    assert ("required_step_skipped", "triage") in deviations
    assert ("no_decision", "") in deviations
    assert len(deviations) == 3  # triage, decide, no decision


def test_a_decided_run_finishes_clean() -> None:
    run = _Run()
    _to_decide(run)
    run.rt.submit_decision("REFUND_DUPLICATE", "charged twice", ["ch_2"])
    before = len(run.of("playbook_deviation"))
    run.rt.finish()
    assert len(run.of("playbook_deviation")) == before


# ── pauses ───────────────────────────────────────────────────────────────────


def test_a_pause_blocks_the_active_steps_until_the_run_resumes() -> None:
    run = _Run()
    run.rt.pause("waiting for the person's answer")
    blocked = run.of("playbook_step")[-1]
    assert blocked["status"] == BLOCKED
    assert blocked["reason"] == "waiting for the person's answer"
    assert run.status("triage") == BLOCKED
    # A blocked step is still the active one: its tools and its closing still work.
    assert run.call("lookup_order") == ""
    run.rt.unpause()
    assert run.status("triage") == ACTIVE
    assert run.of("playbook_step")[-1]["status"] == ACTIVE


# ── the record on the chain ──────────────────────────────────────────────────


def test_the_step_trail_and_the_decision_reach_the_audit_chain() -> None:
    run = _Run()
    _to_decide(run)
    run.rt.submit_decision("REFUND_DUPLICATE", "charged twice", ["ch_2"])
    steps = [f for kind, f in run.audit.events if kind == STEP_EVENT]
    assert [(e["step"], e["status"]) for e in steps] == [
        ("triage", DONE),
        ("check_amount", WAIVED),
        ("check_duplicates", DONE),
        ("decide", DONE),
    ]
    assert {e["correlation_id"] for e in steps} == {"run-9"}
    assert {e["principal"] for e in steps} == {"gw"}
    assert {e["playbook_version"] for e in steps} == {"3"}
    (decision,) = [f for kind, f in run.audit.events if kind == DECISION_EVENT]
    assert decision["outcome_id"] == "REFUND_DUPLICATE"
    assert decision["converged"]
    assert decision["steps"]["check_amount"] == WAIVED
    # The model's words about the case stay in the trace; the chain holds digests.
    assert "charged twice" not in json.dumps(decision)


# ── what the model is told ───────────────────────────────────────────────────


def test_the_prose_discloses_only_the_active_step_in_full() -> None:
    pb = parse_playbook_v2(example())
    skills = {
        "refund-checklist": {
            "name": "refund-checklist",
            "description": "How to check a refund.",
            "instructions": "STEP-SKILL-BODY",
        },
    }
    agent_skills = [
        {"name": "tone", "description": "Write kindly.", "instructions": "AGENT-SKILL-BODY"}
    ]
    prose = playbook_prose(
        pb, skills, initial_active(pb), agent_skills=agent_skills, enforcement="block"
    )
    assert "# Playbook: Refund dispute" in prose
    assert "STEP-SKILL-BODY" in prose  # triage is active and uses it
    assert "AGENT-SKILL-BODY" not in prose  # listed, not disclosed
    assert "- tone: Write kindly." in prose
    assert "- refund-checklist: How to check a refund." in prose
    assert "issue_refund: This playbook diagnoses" in prose
    assert "INCONCLUSIVE [INCONCLUSIVE]" in prose
    assert "is refused" in prose
    assert "- check_amount: Compare the charge with the order (optional; after triage" in prose
    assert "Ask the person with ask_user" in prose
    assert "customer_statement: What does the customer say happened?" in prose
    assert "recommends; it does not act" in prose


def test_a_step_brief_says_what_it_owes() -> None:
    pb = parse_playbook_v2(example())
    decide = pb.step("decide")
    assert decide is not None
    brief = step_brief(decide, {})
    assert "Tools: submit_decision (at most 1 calls)" in brief
    triage = pb.step("triage")
    assert triage is not None
    brief = step_brief(triage, {})
    assert "Skill refund-checklist: not available" in brief
    assert "Expected outputs (pass each to complete_step): dispute_kind" in brief
    assert "- duplicate (Charged twice): when selected_branch_ids contains" in brief
    body = example()
    _set_step(body, "triage", allowed_tools=None, rules=["Be exact."], facts=["Orders are EUR."])
    _set_step(body, "check_amount", allowed_tools=[], min_tool_calls=0)
    pb = parse_playbook_v2(body)
    brief = step_brief(pb.steps[0], {})
    assert "Tools: any of your tools" in brief
    assert "- Be exact." in brief
    assert "- Orders are EUR." in brief
    assert "Tools: none" in step_brief(pb.steps[2], {})
    assert "at least 0" in step_brief(pb.steps[2], {})


async def test_the_control_tools_drive_the_runtime() -> None:
    run = _Run()
    run.call("lookup_order")
    complete = control_tool("complete_step", run.rt)
    select = control_tool("select_branches", run.rt)
    submit = control_tool("submit_decision", run.rt)
    assert [t.name for t in (complete, select, submit)] == [
        "complete_step",
        "select_branches",
        "submit_decision",
    ]
    picked = json.loads(await select.execute(step_id="triage", branch_ids=[], reason="none"))
    assert picked["ok"]
    closed = json.loads(
        await complete.execute(
            step_id="triage",
            outputs={"dispute_kind": "x", "selected_branch_ids": [], "customer_statement": "y"},
        )
    )
    assert closed["ok"]
    assert closed["active"] == ["decide"]
    decided = json.loads(
        await submit.execute(outcome_id="NO_REFUND", rationale="matches", evidence_refs=[])
    )
    assert decided["ok"]
    assert run.rt.name == "TulipPlaybookRuntime"
    assert run.rt.priority > 0
    run.rt.bind(None)  # a v2 run adopts no SDK enforcer
