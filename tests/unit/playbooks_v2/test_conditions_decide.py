# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit: a branch's condition decides; the model cannot choose against the data.

The live dev suite (F40) caught it: a step with two branches on typed data
(``inputs.amount > 10000`` / ``NOT (inputs.amount > 10000)``), the model called
``select_branches`` itself, and the trace said ``routed_by: "select_branches"`` -- so a
model could take the branch meant for small amounts on a large one and skip the approval
the other branch's step carries. Now every branch whose ``when`` the run can tell is
decided by it: ``select_branches`` against such a verdict is refused, one that agrees is
recorded as routed by the conditions (``routed_by: "when"``), and choosing stays only for
the branches without a condition and the conditions the run cannot tell.
"""

from __future__ import annotations

from typing import Any

from tulip.playbooks.v2 import ROUTING_UNKNOWN
from tulip.playbooks.v2.engine import (
    ACTIVE,
    DONE,
    STEP_EVENT,
    WAIVED,
    PlaybookRuntime,
    parse_playbook_v2,
    step_brief,
)


def _money(amount: float, currency: str = "USD") -> dict[str, Any]:
    return {"amount": amount, "currency": currency}


_BIG = {
    "id": "big",
    "label": "Large invoice",
    "when": "inputs.amount > 10000",
    "next_step_id": "cfo",
}
_SMALL = {
    "id": "small",
    "label": "Small invoice",
    "when": "NOT (inputs.amount > 10000)",
    "next_step_id": "clerk",
}


def _invoice(branches: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """F40's shape: ``check`` routes on the amount; ``cfo`` carries the approval."""
    return {
        "version": "playbook.v2",
        "id": "invoice-approval",
        "title": "Invoice approval",
        "inputs": [
            {"name": "amount", "type": "money", "sensitive": False},
            {"name": "vendor_name", "sensitive": False},
        ],
        "step_groups": [
            {
                "id": "g",
                "title": "Review",
                "steps": [
                    {
                        "id": "check",
                        "title": "Check the invoice",
                        "allowed_tools": [],
                        "outputs": [{"name": "total", "type": "money", "sensitive": False}],
                        "branches": [_BIG, _SMALL] if branches is None else branches,
                    },
                    {
                        "id": "cfo",
                        "title": "CFO approval",
                        "allowed_tools": ["pay"],
                        "approval": {"by": "cfo", "count": 1},
                        "after": ["check"],
                    },
                    {
                        "id": "clerk",
                        "title": "Clerk review",
                        "allowed_tools": ["pay"],
                        "after": ["check"],
                    },
                ],
            }
        ],
    }


class _Run:
    def __init__(
        self,
        branches: list[dict[str, Any]] | None = None,
        inputs: dict[str, Any] | None = None,
    ) -> None:
        self.events: list[dict[str, Any]] = []
        self.rt = PlaybookRuntime(parse_playbook_v2(_invoice(branches)), emit=self.events.append)
        self.rt.start(
            inputs={"amount": _money(25000), "vendor_name": "Acme"} if inputs is None else inputs
        )

    def status(self, step_id: str) -> str:
        return self.rt.graph.status[step_id]

    def close(self, total: float = 25000) -> dict[str, Any]:
        return self.rt.complete_step("check", {"total": _money(total)})

    def done_event(self) -> dict[str, Any]:
        return next(
            e
            for e in self.events
            if e.get("type") == STEP_EVENT and e["step_id"] == "check" and e["status"] == DONE
        )


# ── F40: the model tries to route a large amount to the small branch ─────────


def test_a_choice_against_the_data_is_refused_in_plain_words() -> None:
    run = _Run()
    result = run.rt.select_branches("check", ["small"], "it looks routine")
    assert result["ok"] is False
    assert result["error"].startswith(
        "This step routes by its conditions: the amount is more than $10,000, so it goes "
        "to Large invoice."
    )
    assert "select_branches cannot choose against" in result["error"]
    assert result["routed_by"] == "when"
    assert result["branches_taken"] == ["big"]
    assert result["branches_not_taken"] == ["small"]
    # Nothing was chosen: closing the step routes it by the data.
    assert "check" not in run.rt.graph.selected
    closed = run.close()
    assert closed["ok"]
    assert closed["routed_by"] == "when"
    assert closed["branches_taken"] == ["big"]
    assert run.status("cfo") == ACTIVE
    assert run.status("clerk") == WAIVED
    assert run.done_event()["routed_by"] == "when"


def test_choosing_both_or_neither_against_the_data_is_refused() -> None:
    run = _Run()
    assert run.rt.select_branches("check", ["big", "small"], "both")["ok"] is False
    assert run.rt.select_branches("check", [], "neither")["ok"] is False
    small = _Run(inputs={"amount": _money(500), "vendor_name": "Acme"})
    refused = small.rt.select_branches("check", ["big"], "be careful")
    assert refused["ok"] is False
    assert "so it goes to Small invoice" in refused["error"]


def test_a_choice_that_agrees_is_accepted_and_recorded_as_routed_by_the_conditions() -> None:
    run = _Run()
    picked = run.rt.select_branches("check", ["big"], "it is large")
    assert picked["ok"]
    assert picked["waives"] == ["small"]
    closed = run.close()
    assert closed["ok"]
    assert closed["routed_by"] == "when"
    assert closed["branches_taken"] == ["big"]
    assert "selection_set_aside" not in closed
    assert run.done_event()["routed_by"] == "when"
    assert run.status("cfo") == ACTIVE
    assert run.status("clerk") == WAIVED


def test_without_a_choice_the_conditions_route() -> None:
    run = _Run(inputs={"amount": _money(500), "vendor_name": "Acme"})
    closed = run.close(500)
    assert closed["routed_by"] == "when"
    assert closed["branches_taken"] == ["small"]
    assert run.status("clerk") == ACTIVE
    assert run.status("cfo") == WAIVED


# ── where choosing is still the model's (or a person's) ──────────────────────


def test_an_unknown_verdict_still_allows_an_explicit_choice() -> None:
    run = _Run(inputs={"vendor_name": "Acme"})  # the amount is required and missing
    held = run.close()
    assert held["ok"] is False
    assert held["routing"] == ROUTING_UNKNOWN
    picked = run.rt.select_branches("check", ["small"], "the person said so")
    assert picked["ok"]
    closed = run.close()
    assert closed["ok"]
    assert closed["routed_by"] == "select_branches"
    assert closed["branches_taken"] == ["small"]
    assert run.status("clerk") == ACTIVE
    assert run.status("cfo") == WAIVED


def test_a_choice_must_still_agree_with_the_verdicts_the_run_can_tell() -> None:
    vendor = {
        "id": "vendor",
        "label": "Known vendor",
        "when": "inputs.vendor_name == 'Acme'",
        "next_step_id": "clerk",
    }
    run = _Run([_BIG, vendor], inputs={"amount": _money(25000)})  # no vendor_name: unknown
    refused = run.rt.select_branches("check", ["vendor"], "skip the CFO")
    assert refused["ok"] is False
    assert "the amount is more than $10,000, so it goes to Large invoice" in refused["error"]
    assert run.rt.select_branches("check", ["big"], "large, vendor unclear")["ok"]
    closed = run.close()
    assert closed["routed_by"] == "select_branches"  # the unknown one was chosen
    assert closed["branches_taken"] == ["big"]
    assert closed["branches_not_taken"] == ["vendor"]


def test_branches_without_a_condition_are_chosen_as_before() -> None:
    plain = [
        {"id": "a", "when": "always", "next_step_id": "cfo"},
        {"id": "b", "when": "always", "next_step_id": "clerk"},
    ]
    run = _Run(plain)
    assert run.rt.select_branches("check", ["b"], "the clerk can do it")["ok"]
    closed = run.close()
    assert closed["routed_by"] == "select_branches"
    assert closed["branches_taken"] == ["b"]
    assert run.status("clerk") == ACTIVE
    assert run.status("cfo") == WAIVED
    # Unchosen, as they always were: `always` holds.
    other = _Run(plain)
    closed = other.close()
    assert closed["routed_by"] == "when"
    assert closed["branches_taken"] == ["a", "b"]


def test_beside_a_condition_only_the_unconditioned_branch_is_chosen() -> None:
    extra = {"id": "extra", "when": "always", "next_step_id": "clerk"}
    run = _Run([_BIG, extra])
    assert run.rt.select_branches("check", ["extra"], "the clerk only")["ok"] is False
    assert run.rt.select_branches("check", ["big"], "the CFO only")["ok"]
    closed = run.close()
    assert closed["routed_by"] == "select_branches"
    assert closed["branches_taken"] == ["big"]
    assert closed["branches_not_taken"] == ["extra"]


# ── a condition on the step's own outputs ────────────────────────────────────


def test_a_condition_on_the_step_s_own_outputs_waits_for_them_then_decides() -> None:
    own = [
        {"id": "big", "label": "Large total", "when": "total > 10000", "next_step_id": "cfo"},
        {"id": "small", "when": "NOT (total > 10000)", "next_step_id": "clerk"},
    ]
    run = _Run(own)
    # The outputs are not given yet: the choice cannot be judged, so it is taken...
    assert run.rt.select_branches("check", ["small"], "it looks routine")["ok"]
    # ...and set aside once the outputs say otherwise.
    closed = run.close(25000)
    assert closed["ok"]
    assert closed["routed_by"] == "when"
    assert closed["branches_taken"] == ["big"]
    assert closed["selection_set_aside"].startswith(
        "This step routes by its conditions: the total is more than $10,000, so it goes "
        "to Large total."
    )
    assert run.status("cfo") == ACTIVE
    assert run.status("clerk") == WAIVED
    assert run.done_event()["routed_by"] == "when"


# ── what the model is told ───────────────────────────────────────────────────


def _brief(branches: list[dict[str, Any]]) -> str:
    step = parse_playbook_v2(_invoice(branches)).step("check")
    assert step is not None
    return step_brief(step, {})


def test_the_brief_says_when_the_conditions_route_the_step() -> None:
    assert (
        "Branches (routed automatically by their conditions when you call complete_step; "
        "select_branches cannot choose against them):"
    ) in _brief([_BIG, _SMALL])
    mixed = _brief([_BIG, {"id": "extra", "when": "always", "next_step_id": "clerk"}])
    assert "a branch with a condition is routed by it automatically" in mixed
    plain = _brief([{"id": "a", "when": "always", "next_step_id": "cfo"}])
    assert "decided by your outputs, or pick them with select_branches" in plain
