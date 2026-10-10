# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit: a step's approval rules, and which one applies to a held call.

First match wins, the default rule last. A rule whose ``when`` cannot be told is never a
reason to ask fewer people: the strictest of it and the rules after it applies. The
vectors in ``fixtures/approval_rules.json`` pin the choice; every copy of ``pick_rule``
must agree on them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tulip.playbooks.v2 import (
    ApprovalRule,
    ResolvedApproval,
    StepApproval,
    parse_duration,
    pick_rule,
)
from tulip.playbooks.v2.approvals import (
    call_context,
    group_name,
    parse_rule,
    parse_step_approval,
    who_must_approve,
)
from tulip.playbooks.v2.engine import PlaybookRuntime, parse_playbook_v2
from tulip.playbooks.v2.when import Money


VECTORS = json.loads(
    (Path(__file__).parent / "fixtures" / "approval_rules.json").read_text("utf-8")
)

VENDOR_CHANGE: dict[str, Any] = VECTORS["approvals"]["vendor_change"]


def typed(value: Any) -> Any:
    """A vector's context as the runtime builds it: money values as :class:`Money`."""
    if isinstance(value, dict):
        if set(value) == {"amount", "currency"}:
            return Money(value["amount"], value["currency"])
        return {k: typed(v) for k, v in value.items()}
    return value


# ── the vectors ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("case", VECTORS["cases"], ids=[c["name"] for c in VECTORS["cases"]])
def test_the_vectors(case: dict[str, Any]) -> None:
    approval = parse_step_approval(VECTORS["approvals"][case["approval"]])
    assert approval is not None
    resolved = pick_rule(approval, typed(case["context"]))
    expect = case["expect"]
    assert resolved.rule_index == expect["rule_index"]
    assert resolved.matched == expect["matched"]
    if "groups" in expect:
        assert [list(g) for g in resolved.groups] == expect["groups"]
    for key in ("due_seconds", "escalate_to", "only_named"):
        if key in expect:
            assert getattr(resolved, key) == expect[key], key


# ── parsing ──────────────────────────────────────────────────────────────────


def test_rules_parse_into_frozen_rules() -> None:
    approval = parse_step_approval(VENDOR_CHANGE)
    assert approval == StepApproval(
        by="finance",
        ask="Confirm the new account was verified with the vendor.",
        show=("vendor_id", "vendor_name"),
        rules=(
            ApprovalRule(
                groups=(("finance", 1), ("cfo", 1)),
                when="inputs.amount > 10000",
                due_seconds=4 * 3600,
                escalate_to="finance-leads",
            ),
            ApprovalRule(groups=(("finance", 1),), due_seconds=24 * 3600),
        ),
        only_these_approvers=True,
    )
    with pytest.raises(AttributeError):
        approval.rules = ()  # type: ignore[misc]


def test_the_old_by_is_one_default_rule() -> None:
    approval = StepApproval(by="leads")
    assert approval.rules == (ApprovalRule(groups=(("leads", 1),)),)
    assert parse_step_approval({"by": "leads"}) == approval


def test_by_beside_rules_without_a_default_becomes_the_default() -> None:
    approval = parse_step_approval(
        {"by": "ap", "rules": [{"when": "inputs.amount > 10", "by": "finance"}]}
    )
    assert approval is not None
    assert [r.groups for r in approval.rules] == [(("finance", 1),), (("ap", 1),)]
    assert approval.rules[-1].default


def test_by_is_not_added_when_the_rules_have_a_default() -> None:
    approval = parse_step_approval({"by": "ap", "rules": [{"by": "finance"}]})
    assert approval is not None
    assert [r.groups for r in approval.rules] == [(("finance", 1),)]


def test_rules_alone_name_the_default_as_by() -> None:
    approval = parse_step_approval(
        {"rules": [{"when": "inputs.x > 1", "by": "cfo"}, {"by": "finance"}]}
    )
    assert approval is not None
    assert approval.by == "finance"
    no_default = parse_step_approval({"rules": [{"when": "inputs.x > 1", "by": "cfo"}]})
    assert no_default is not None
    assert no_default.by == "cfo"


def test_an_approval_names_someone() -> None:
    with pytest.raises(ValueError, match="names who approves"):
        StepApproval()
    assert parse_step_approval({"rules": [{"when": "x > 1"}, "finance", 3]}) is None
    assert parse_step_approval({"rules": "finance"}) is None


@pytest.mark.parametrize(
    ("raw", "groups"),
    [
        ({"by": " finance ", "count": 2}, (("finance", 2),)),
        ({"by": "finance"}, (("finance", 1),)),
        ({"by": "finance", "count": 0}, (("finance", 1),)),
        ({"by": "finance", "count": -3}, (("finance", 1),)),
        ({"by": "finance", "count": True}, (("finance", 1),)),
        ({"by": "finance", "count": "3"}, (("finance", 3),)),
        ({"by": "finance", "count": 1.5}, (("finance", 1),)),
        (
            {"all_of": [{"by": "finance"}, {"by": "cfo", "count": 2}, {"count": 4}, "legal"]},
            (("finance", 1), ("cfo", 2)),
        ),
        ({"all_of": [{"by": "finance"}, {"by": "finance", "count": 2}]}, (("finance", 3),)),
        ({"all_of": [{"by": "cfo"}], "by": "finance"}, (("cfo", 1),)),
    ],
)
def test_a_rules_groups_read_tolerantly_and_never_fewer(
    raw: dict[str, Any], groups: tuple[tuple[str, int], ...]
) -> None:
    rule = parse_rule(raw)
    assert rule is not None
    assert rule.groups == groups


@pytest.mark.parametrize("raw", [{"when": "x > 1"}, {"by": ""}, {"all_of": []}, "finance", None])
def test_a_rule_without_a_group_is_left_out(raw: Any) -> None:
    assert parse_rule(raw) is None


def test_a_when_that_cannot_be_read_keeps_its_rule() -> None:
    broken = parse_rule({"when": "amount >", "by": "finance"})
    assert broken is not None
    assert broken.when_error
    assert not broken.default
    not_text = parse_rule({"when": 10000, "by": "finance"})
    assert not_text is not None
    assert not_text.when_error == "the condition is not text"


def test_an_unreadable_due_and_escalation_read_as_none() -> None:
    rule = parse_rule({"by": "finance", "due": "soon", "escalate_to": 7})
    assert rule == ApprovalRule(groups=(("finance", 1),))
    assert parse_rule({"by": "finance", "due": 3600}) == ApprovalRule(groups=(("finance", 1),))


def test_only_these_approvers_must_be_true() -> None:
    for value in ("yes", 1, None):
        approval = parse_step_approval({"by": "finance", "only_these_approvers": value})
        assert approval is not None
        assert approval.only_these_approvers is False


def test_a_rule_checks_its_own_shape() -> None:
    with pytest.raises(ValueError, match="at least one group"):
        ApprovalRule(groups=())
    with pytest.raises(ValueError, match="count of at least 1"):
        ApprovalRule(groups=(("finance", 0),))


# ── durations ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("30m", 1800), ("4h", 14400), ("2d", 172800), (" 1H ", 3600), ("90 m", 5400)],
)
def test_durations(text: str, seconds: int) -> None:
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["", "4", "h", "4w", "1h30m", "-1h", "0m", "1.5h", "4 hours"])
def test_durations_that_are_not(text: str) -> None:
    with pytest.raises(ValueError, match="duration"):
        parse_duration(text)


def test_a_duration_is_text() -> None:
    with pytest.raises(ValueError, match="duration"):
        parse_duration(3600)  # type: ignore[arg-type]


# ── reasons, in plain words ──────────────────────────────────────────────────


def vendor(context: dict[str, Any]) -> ResolvedApproval:
    approval = parse_step_approval(VENDOR_CHANGE)
    assert approval is not None
    return pick_rule(approval, context, step="change_bank")


def test_the_reason_says_why_and_who() -> None:
    resolved = vendor({"inputs": {"amount": Money(25000, "USD")}})
    assert (
        resolved.reason == "The amount is more than $10,000, so Finance and CFO must both approve."
    )
    assert resolved.step == "change_bank"
    assert resolved.total == 2
    assert resolved.labels == ["finance", "cfo"]


def test_the_default_reason() -> None:
    resolved = vendor({"inputs": {"amount": Money(50, "USD")}})
    assert resolved.reason == "No other rule applies, so someone from Finance must approve."
    alone = pick_rule(StepApproval(by="leads"), {})
    assert alone.reason == "Someone from Leads must approve."


def test_the_unknown_reason_says_the_run_cannot_tell() -> None:
    resolved = vendor({"inputs": {"amount": {"redacted": True, "sha256": "ab", "bytes": 3}}})
    assert resolved.reason == (
        "This run cannot tell whether the amount is more than 10,000, so the strictest rule "
        "applies: Finance and CFO must both approve."
    )
    assert resolved.unknown_when == "inputs.amount > 10000"


def test_a_broken_rule_says_so() -> None:
    approval = parse_step_approval({"rules": [{"when": "amount >", "by": "cfo"}, {"by": "ap"}]})
    assert approval is not None
    assert pick_rule(approval, {}).reason == (
        "This step's approval rule cannot be read, so the strictest rule applies: "
        "someone from CFO must approve."
    )


def test_the_no_rule_reason() -> None:
    approval = parse_step_approval({"rules": [{"when": "inputs.x > 1", "by": "cfo"}]})
    assert approval is not None
    assert pick_rule(approval, {"inputs": {"x": 0}}).reason == (
        "No rule applies and the step names no default, so the strictest rule applies: "
        "someone from CFO must approve."
    )


@pytest.mark.parametrize(
    ("when", "context", "words"),
    [
        ("args.amount >= 5000", {"args": {"amount": 6000}}, "The call's amount is at least 5,000"),
        ("inputs.amount < 99.5", {"inputs": {"amount": 1}}, "The amount is less than 99.50"),
        (
            "inputs.amount > 100",
            {"inputs": {"amount": Money(500, "CHF")}},
            "The amount is more than 100 CHF",
        ),
        (
            "outputs.verify.risk == 'high'",
            {"outputs": {"verify": {"risk": "high"}}},
            'The risk is "high"',
        ),
        (
            "args.vendor_country != 'FR'",
            {"args": {"vendor_country": "IR"}},
            'The call\'s vendor country is not "FR"',
        ),
        (
            "inputs.amount.currency == 'EUR'",
            {"inputs": {"amount": Money(1, "EUR")}},
            'The amount\'s currency is "EUR"',
        ),
        ("tags contains urgent", {"tags": ["urgent"]}, 'The tags includes "urgent"'),
        ("findings is not empty", {"findings": [1]}, "The findings is not empty"),
        ("args.expedite", {"args": {"expedite": True}}, "The call's expedite is set"),
        ("args.expedite == true", {"args": {"expedite": True}}, "The call's expedite is true"),
        ("NOT (inputs.x == 1)", {"inputs": {"x": 2}}, "It is not so that the x is 1"),
        (
            "inputs.a > 1 AND (inputs.b > 2 OR inputs.c > 3)",
            {"inputs": {"a": 2, "b": 3, "c": 0}},
            "The a is more than 1 and (the b is more than 2 or the c is more than 3)",
        ),
        ("always", {}, "It always applies"),
    ],
)
def test_conditions_in_words(when: str, context: dict[str, Any], words: str) -> None:
    approval = StepApproval(rules=(ApprovalRule(groups=(("cfo", 1),), when=when),), by="ap")
    resolved = pick_rule(approval, context)
    assert resolved.matched == "when"
    assert resolved.reason == f"{words}, so someone from CFO must approve."


@pytest.mark.parametrize(
    ("groups", "words"),
    [
        ((("finance", 1),), "someone from Finance must approve"),
        ((("finance", 2),), "2 people from Finance must approve"),
        ((("finance", 1), ("cfo", 1)), "Finance and CFO must both approve"),
        ((("finance", 1), ("cfo", 1), ("legal", 1)), "Finance, CFO and Legal must all approve"),
        ((("finance", 2), ("cfo", 1)), "2 people from Finance and someone from CFO must approve"),
    ],
)
def test_who_must_approve(groups: tuple[tuple[str, int], ...], words: str) -> None:
    assert who_must_approve(groups) == words


@pytest.mark.parametrize(
    ("label", "name"),
    [("finance", "Finance"), ("cfo", "CFO"), ("ap-leads", "AP Leads"), ("tax_ops", "Tax Ops")],
)
def test_group_names_read_as_the_registrys(label: str, name: str) -> None:
    assert group_name(label) == name


def test_hold_fields() -> None:
    resolved = vendor({"inputs": {"amount": Money(25000, "USD")}})
    assert resolved.hold_fields() == {
        "approver_groups": [{"label": "finance", "count": 1}, {"label": "cfo", "count": 1}],
        "only_named_groups": True,
        "escalate_to_label": "finance-leads",
        "ttl_seconds": 28800,  # twice the due: the escalation gets its turn
        "escalate_after_seconds": 14400,
    }
    plain = pick_rule(StepApproval(by="leads"), {})
    assert plain.hold_fields() == {
        "approver_groups": [{"label": "leads", "count": 1}],
        "only_named_groups": False,
    }


def test_hold_fields_never_name_general_approvers() -> None:
    """An ``approver_labels`` entry makes its holders count for ANY group on the registry:
    two Finance people would satisfy "Finance and CFO". The groups alone say who decides."""
    both = pick_rule(
        StepApproval(
            rules=(ApprovalRule(groups=(("finance", 1), ("cfo", 1)), due_seconds=3600),),
            only_these_approvers=False,
        ),
        {},
    )
    fields = both.hold_fields()
    assert "approver_labels" not in fields
    assert fields == {
        "approver_groups": [{"label": "finance", "count": 1}, {"label": "cfo", "count": 1}],
        "only_named_groups": False,
        "ttl_seconds": 3600,  # no escalation: the due is the hold's life
    }
    assert "escalate_after_seconds" not in fields


def test_call_context_types_money_only() -> None:
    context = call_context({"amount": {"amount": 5, "currency": "USD"}, "bad": {"amount": "5"}})
    assert isinstance(context["amount"], Money)
    assert context["bad"] == {"amount": "5"}
    assert call_context(None) == {}


# ── the runtime: the step that owns the call decides ─────────────────────────


def vendor_playbook(approval: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "version": "playbook.v2",
        "id": "vendor-bank-change",
        "title": "Vendor bank change",
        "summary": "Change a vendor's bank account once it is verified.",
        "mode": "workflow",
        "inputs": [
            {"name": "vendor_id", "type": "text"},
            {"name": "amount", "type": "money", "sensitive": False},
        ],
        "step_groups": [
            {
                "id": "work",
                "title": "Work",
                "steps": [
                    {
                        "id": "verify",
                        "title": "Verify the vendor",
                        "allowed_tools": ["lookup_vendor"],
                        "outputs": [{"name": "risk", "type": "choice", "choices": ["low", "high"]}],
                    },
                    {
                        "id": "change_bank",
                        "title": "Change the bank account",
                        "after": ["verify"],
                        "allowed_tools": ["update_bank_account"],
                        "approval": approval
                        or {
                            "ask": "Confirm the new account was verified with the vendor.",
                            "rules": [
                                {
                                    "when": "inputs.amount > 10000 OR outputs.verify.risk == 'high'",
                                    "all_of": [{"by": "finance"}, {"by": "cfo"}],
                                    "due": "4h",
                                    "escalate_to": "finance-leads",
                                },
                                {"when": "args.amount > 5000", "by": "finance", "count": 2},
                                {"by": "finance", "due": "24h"},
                            ],
                        },
                    },
                ],
            }
        ],
        "decision_policy": {
            "id": "vendor-decision",
            "outcomes": [{"id": "INCONCLUSIVE", "priority": 1, "classification": "INCONCLUSIVE"}],
        },
    }


def at_change_bank(risk: str = "low", **inputs: Any) -> PlaybookRuntime:
    rt = PlaybookRuntime(parse_playbook_v2(vendor_playbook()), emit=lambda _e: None)
    rt.start(inputs or {"vendor_id": "v-1", "amount": {"amount": 900, "currency": "USD"}})
    assert rt._admit("lookup_vendor") == ""
    rt._credit("lookup_vendor", '{"ok": true}')
    result = rt.complete_step("verify", {"risk": risk})
    assert result["ok"], result
    return rt


def test_approval_for_reads_inputs_outputs_and_the_call() -> None:
    rt = at_change_bank()
    low = rt.approval_for("update_bank_account", {"amount": 100})
    assert low is not None
    assert (low.rule_index, low.matched, low.groups) == (2, "default", (("finance", 1),))
    assert low.due_seconds == 86400
    assert low.ask == "Confirm the new account was verified with the vendor."
    assert low.step == "change_bank"

    big_call = rt.approval_for("update_bank_account", {"amount": 6000})
    assert big_call is not None
    assert (big_call.rule_index, big_call.groups) == (1, (("finance", 2),))
    assert (
        big_call.reason
        == "The call's amount is more than 5,000, so 2 people from Finance must approve."
    )

    risky = at_change_bank(risk="high").approval_for("update_bank_account", {"amount": 1})
    assert risky is not None
    assert (risky.rule_index, risky.groups) == (0, (("finance", 1), ("cfo", 1)))

    big = at_change_bank(vendor_id="v-1", amount={"amount": 20000, "currency": "USD"})
    resolved = big.approval_for("update_bank_account", {"amount": 1})
    assert resolved is not None
    assert resolved.rule_index == 0
    assert resolved.reason.startswith('The amount is more than $10,000 or the risk is "high"')


def test_an_input_the_run_does_not_have_means_the_strictest() -> None:
    rt = at_change_bank(vendor_id="v-1")  # no amount: required, so UNAVAILABLE
    resolved = rt.approval_for("update_bank_account", {"amount": 1})
    assert resolved is not None
    assert (resolved.rule_index, resolved.matched) == (0, "unknown")
    assert resolved.groups == (("finance", 1), ("cfo", 1))


def test_an_argument_the_rule_cannot_order_means_the_strictest_from_there() -> None:
    rt = at_change_bank()
    for args in ({"amount": "6000"}, {}, None, {"amount": [6000]}):
        resolved = rt.approval_for("update_bank_account", args)
        assert resolved is not None
        assert (resolved.rule_index, resolved.matched) == (1, "unknown"), args


def test_no_approval_without_an_owning_step_or_an_approval() -> None:
    rt = PlaybookRuntime(parse_playbook_v2(vendor_playbook()), emit=lambda _e: None)
    rt.start({"vendor_id": "v-1", "amount": {"amount": 1, "currency": "USD"}})
    assert rt.approval_for("lookup_vendor", {}) is None  # verify has no approval
    assert rt.approval_for("update_bank_account", {}) is None  # not active yet
    assert rt.approval_for("delete_everything", {}) is None


def test_an_argument_ordered_against_another_value() -> None:
    approval = StepApproval(
        rules=(ApprovalRule(groups=(("cfo", 1),), when="args.amount > inputs.limit"),), by="ap"
    )
    over = pick_rule(approval, {"args": {"amount": 20}, "inputs": {"limit": 10}})
    assert (over.rule_index, over.matched) == (0, "when")
    text = pick_rule(approval, {"args": {"amount": "20"}, "inputs": {"limit": 10}})
    assert (text.rule_index, text.matched) == (0, "unknown")
    # Against a value that is not a number either, ordering is false, as it always was.
    neither = pick_rule(approval, {"args": {"amount": "20"}, "inputs": {"limit": "x"}})
    assert (neither.rule_index, neither.matched) == (1, "default")
    flipped = StepApproval(
        rules=(ApprovalRule(groups=(("cfo", 1),), when="5000 < args.amount"),), by="ap"
    )
    assert pick_rule(flipped, {"args": {}}).matched == "unknown"


def test_a_blank_when_is_the_default() -> None:
    rule = parse_rule({"when": "  ", "by": "finance"})
    assert rule is not None
    assert rule.default


def test_null_in_words() -> None:
    approval = StepApproval(
        rules=(ApprovalRule(groups=(("cfo", 1),), when="args.po_number == null"),), by="ap"
    )
    assert pick_rule(approval, {"args": {}}).reason == (
        "The call's po number is empty, so someone from CFO must approve."
    )
