# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit: a step may name who approves it, and the runtime says which step owns a call.

The engine only parses ``approval`` and exposes it; the gateway compiles it into a hold
on that step's tool calls. Parsing is tolerant: the registry validates strictly at
publish, so a malformed value reads as no approval rather than refusing the run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from tulip.playbooks.v2 import StepApproval
from tulip.playbooks.v2.engine import PlaybookRuntime, parse_playbook_v2


EXAMPLE = Path(__file__).parent / "fixtures" / "refund-dispute.v2.yaml"


def example(**approvals: Any) -> dict[str, Any]:
    body: dict[str, Any] = yaml.safe_load(EXAMPLE.read_text("utf-8"))
    for group in body["step_groups"]:
        for step in group["steps"]:
            if step["id"] in approvals:
                step["approval"] = approvals[step["id"]]
    return body


def runtime(body: dict[str, Any] | None = None) -> PlaybookRuntime:
    rt = PlaybookRuntime(parse_playbook_v2(body or example()), emit=lambda _e: None)
    rt.start()
    return rt


# ── parsing ──────────────────────────────────────────────────────────────────


def test_an_approval_parses() -> None:
    body = example(
        check_amount={
            "by": " finance-approvers ",
            "ask": " Refund the difference? ",
            "show": ["order_id", "", "amount"],
        }
    )
    step = parse_playbook_v2(body).step("check_amount")
    assert step is not None
    assert step.approval == StepApproval(
        by="finance-approvers", ask="Refund the difference?", show=("order_id", "amount")
    )


def test_by_alone_is_enough() -> None:
    step = parse_playbook_v2(example(decide={"by": "leads"})).step("decide")
    assert step is not None
    assert step.approval == StepApproval(by="leads")
    assert step.approval.ask == ""
    assert step.approval.show == ()


def test_an_absent_approval_is_none() -> None:
    pb = parse_playbook_v2(example())
    assert all(s.approval is None for s in pb.steps)


@pytest.mark.parametrize(
    "value",
    [
        {"ask": "who?"},
        {"by": ""},
        {"by": "   "},
        {"by": 7},
        {"by": None},
        "finance-approvers",
        ["finance-approvers"],
        None,
        True,
    ],
)
def test_an_approval_without_a_grant_label_is_none(value: Any) -> None:
    step = parse_playbook_v2(example(triage=value)).step("triage")
    assert step is not None
    assert step.approval is None


def test_malformed_ask_and_show_read_as_empty() -> None:
    body = example(decide={"by": "leads", "ask": 3, "show": "amount"})
    step = parse_playbook_v2(body).step("decide")
    assert step is not None
    assert step.approval == StepApproval(by="leads")


def test_an_approval_is_frozen() -> None:
    approval = StepApproval(by="leads")
    with pytest.raises(AttributeError):
        approval.by = "others"  # type: ignore[misc]


# ── who owns a call, and what is active ──────────────────────────────────────


def test_active_steps_and_owner_of_follow_the_state() -> None:
    rt = runtime(example(check_amount={"by": "finance-approvers"}))
    assert [s.id for s in rt.active_steps()] == ["triage"]
    owner = rt.owner_of("lookup_order")
    assert owner is not None
    assert owner.id == "triage"
    assert rt.owner_of("get_balance") is None

    assert rt._admit("lookup_order") == ""
    rt._credit("lookup_order", '{"ok": true}')
    result = rt.complete_step(
        "triage",
        {
            "dispute_kind": "amount",
            "selected_branch_ids": ["amount_mismatch"],
            "customer_statement": "wrong amount",
        },
    )
    assert result["ok"]
    assert [s.id for s in rt.active_steps()] == ["check_amount"]
    owner = rt.owner_of("get_balance")
    assert owner is not None
    assert owner.id == "check_amount"
    assert owner.approval == StepApproval(by="finance-approvers")
    assert rt.owner_of("submit_decision") is None


def test_parallel_steps_are_active_together_and_the_first_owns_a_shared_tool() -> None:
    rt = runtime()
    rt._admit("lookup_order")
    rt._credit("lookup_order", '{"ok": true}')
    rt.complete_step(
        "triage",
        {
            "dispute_kind": "both",
            "selected_branch_ids": ["duplicate_charge", "amount_mismatch"],
            "customer_statement": "charged twice, wrong amount",
        },
    )
    assert [s.id for s in rt.active_steps()] == ["check_duplicates", "check_amount"]
    owner = rt.owner_of("lookup_order")
    assert owner is not None
    assert owner.id == "check_duplicates"
    owner = rt.owner_of("get_balance")
    assert owner is not None
    assert owner.id == "check_amount"


def test_the_private_owner_name_is_the_public_one() -> None:
    rt = runtime()
    assert rt._owner("lookup_order") == rt.owner_of("lookup_order")
