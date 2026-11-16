# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The path evaluators: trajectory, tool selection, argument correctness.

Each reads the evidence ledger with no model call and returns a typed verdict
that says what matched, what was missing and under which settings. These
tests pin the three trajectory modes, the selection arithmetic, the
schema-aware argument comparison with its tolerances, and the way the runner
turns each evaluator into one named check and fills a tool's schema in.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from locus.agent import Agent
from locus.core.state import ToolExecution
from locus.evaluation import (
    ArgumentCorrectness,
    EvalCase,
    EvalRunner,
    EvidenceLedger,
    ToolSelection,
    TrajectoryMatch,
    dump_cases,
    load_cases,
    suite_fingerprint,
)
from locus.testing import ScriptedModel, text, tool_call
from locus.tools.decorator import tool


def _ledger(
    *calls: tuple[str, dict[str, object]] | tuple[str, dict[str, object], str],
) -> EvidenceLedger:
    executions = []
    for index, call in enumerate(calls):
        name, arguments = call[0], call[1]
        result = call[2] if len(call) > 2 else "ok"
        executions.append(
            ToolExecution(
                tool_name=name, tool_call_id=f"c{index}", arguments=dict(arguments), result=result
            )
        )
    return EvidenceLedger.from_executions(executions, prompt="p")


REFUSED = json.dumps({"status": "denied", "reason": "policy"})


# --------------------------------------------------------------------------
# TrajectoryMatch
# --------------------------------------------------------------------------


def test_strict_mode_needs_the_same_calls_in_the_same_order() -> None:
    ledger = _ledger(("lookup", {}), ("refund", {}))
    assert TrajectoryMatch(expected=["lookup", "refund"]).evaluate(ledger).passed
    verdict = TrajectoryMatch(expected=["refund", "lookup"]).evaluate(ledger)
    assert not verdict.passed
    assert verdict.score == pytest.approx(0.5)
    assert "expected exactly" in verdict.reason


def test_strict_partial_credit_is_the_common_subsequence_over_the_longer_path() -> None:
    ledger = _ledger(("lookup", {}), ("retry", {}), ("refund", {}))
    verdict = TrajectoryMatch(expected=["lookup", "refund"]).evaluate(ledger)
    assert not verdict.passed
    assert verdict.score == pytest.approx(2 / 3)
    assert verdict.unexpected == ["retry"]


def test_unordered_mode_compares_multisets() -> None:
    ledger = _ledger(("refund", {}), ("lookup", {}), ("lookup", {}))
    assert (
        TrajectoryMatch(expected=["lookup", "lookup", "refund"], mode="unordered")
        .evaluate(ledger)
        .passed
    )
    verdict = TrajectoryMatch(expected=["lookup", "refund"], mode="unordered").evaluate(ledger)
    assert not verdict.passed
    assert verdict.unexpected == ["lookup"]
    assert verdict.score == pytest.approx(2 / 3)


def test_subset_mode_allows_other_calls_around_the_expected_order() -> None:
    ledger = _ledger(("auth", {}), ("lookup", {}), ("log", {}), ("refund", {}))
    verdict = TrajectoryMatch(expected=["lookup", "refund"], mode="subset").evaluate(ledger)
    assert verdict.passed
    assert verdict.score == 1.0
    assert verdict.matched == ["lookup", "refund"]
    wrong_order = TrajectoryMatch(expected=["refund", "lookup"], mode="subset").evaluate(ledger)
    assert not wrong_order.passed
    assert wrong_order.missing == ["lookup"]
    assert wrong_order.score == pytest.approx(0.5)


def test_refused_calls_are_part_of_the_path_unless_executed_only() -> None:
    ledger = _ledger(("lookup", {}), ("delete", {}, REFUSED))
    assert not TrajectoryMatch(expected=["lookup"]).evaluate(ledger).passed
    assert TrajectoryMatch(expected=["lookup"], executed_only=True).evaluate(ledger).passed


def test_the_verdict_carries_provenance_that_moves_with_the_settings() -> None:
    ledger = _ledger(("lookup", {}))
    strict = TrajectoryMatch(expected=["lookup"]).evaluate(ledger)
    subset = TrajectoryMatch(expected=["lookup"], mode="subset").evaluate(ledger)
    assert strict.provenance is not None
    assert subset.provenance is not None
    assert strict.provenance.evaluator == "trajectory_match"
    assert strict.provenance.revision != subset.provenance.revision
    assert strict.provenance.settings["mode"] == "strict"
    assert strict.provenance.evidence_ids == ["E0"]


def test_an_agent_result_is_accepted_in_place_of_a_ledger() -> None:
    @tool
    def lookup(order_id: str) -> str:
        """Look up."""
        return "found"

    agent = Agent(
        model=ScriptedModel([tool_call("lookup", order_id="o1"), text("done")]), tools=[lookup]
    )
    result = agent.run_sync("go")
    assert TrajectoryMatch(expected=["lookup"]).evaluate(result).passed


# --------------------------------------------------------------------------
# ToolSelection
# --------------------------------------------------------------------------


def test_tool_selection_scores_recall_and_names_the_missing_tools() -> None:
    ledger = _ledger(("lookup", {}), ("lookup", {}))
    verdict = ToolSelection(expected=["lookup", "refund"]).evaluate(ledger)
    assert not verdict.passed
    assert verdict.missing == ["refund"]
    assert verdict.score == pytest.approx(0.5)
    assert verdict.details["recall"] == pytest.approx(0.5)
    assert verdict.details["precision"] == 1.0
    assert verdict.details["selected"] == ["lookup"]


def test_tool_selection_extras_are_allowed_by_default_and_refused_when_asked() -> None:
    ledger = _ledger(("lookup", {}), ("log", {}))
    assert ToolSelection(expected=["lookup"]).evaluate(ledger).passed
    strict = ToolSelection(expected=["lookup"], allow_extra=False).evaluate(ledger)
    assert not strict.passed
    assert strict.unexpected == ["log"]
    assert strict.score == pytest.approx(2 / 3)  # F1 of precision 0.5 and recall 1.0


def test_a_forbidden_selection_fails_whatever_else_matched() -> None:
    ledger = _ledger(("lookup", {}), ("delete", {}))
    verdict = ToolSelection(expected=["lookup"], forbidden=["delete"]).evaluate(ledger)
    assert not verdict.passed
    assert verdict.details["forbidden_selected"] == ["delete"]
    assert verdict.score == pytest.approx(0.5)
    assert "forbidden selected" in verdict.reason


def test_no_expected_tools_passes_when_nothing_forbidden_ran() -> None:
    assert ToolSelection(forbidden=["delete"]).evaluate(_ledger()).passed
    assert ToolSelection(forbidden=["delete"]).evaluate(_ledger(("lookup", {}))).passed
    assert not ToolSelection(forbidden=["delete"]).evaluate(_ledger(("delete", {}))).passed


# --------------------------------------------------------------------------
# ArgumentCorrectness
# --------------------------------------------------------------------------

SCHEMA = {
    "type": "object",
    "properties": {
        "order_id": {"type": "string"},
        "amount": {"type": "number"},
        "count": {"type": "integer"},
        "notify": {"type": "boolean"},
        "tags": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        "steps": {"type": "array", "items": {"type": "string"}},
        "address": {"type": "object", "properties": {"zip": {"type": "string"}}},
    },
}


def test_schema_aware_comparison_coerces_by_declared_type() -> None:
    ledger = _ledger(
        ("refund", {"order_id": "ORD-1", "amount": "49.90", "count": "2", "notify": "true"}),
    )
    verdict = ArgumentCorrectness(
        tool="refund",
        expected={"order_id": "ord-1", "amount": 49.9, "count": 2, "notify": True},
        schema=SCHEMA,
    ).evaluate(ledger)
    assert verdict.passed, verdict.mismatches
    assert verdict.score == 1.0
    assert verdict.matched == ["order_id", "amount", "count", "notify"]


def test_tolerances_absolute_per_argument_and_relative() -> None:
    ledger = _ledger(("refund", {"amount": 49.94, "fee": 1.1}))
    exact = ArgumentCorrectness(tool="refund", expected={"amount": 49.9}).evaluate(ledger)
    assert not exact.passed
    assert "not within" in exact.mismatches[0].reason
    assert (
        ArgumentCorrectness(tool="refund", expected={"amount": 49.9}, tolerance=0.05)
        .evaluate(ledger)
        .passed
    )
    per_key = ArgumentCorrectness(
        tool="refund", expected={"amount": 49.9, "fee": 1.0}, tolerances={"amount": 0.05}
    ).evaluate(ledger)
    assert not per_key.passed
    assert [m.argument for m in per_key.mismatches] == ["fee"]
    assert per_key.score == pytest.approx(0.5)
    relative = ArgumentCorrectness(
        tool="refund", expected={"amount": 49.9, "fee": 1.0}, relative_tolerance=0.1
    ).evaluate(ledger)
    assert relative.passed


def test_lists_compare_in_order_unless_the_schema_says_unique_items() -> None:
    ledger = _ledger(("refund", {"tags": ["b", "a"], "steps": ["b", "a"]}))
    verdict = ArgumentCorrectness(
        tool="refund", expected={"tags": ["a", "b"], "steps": ["a", "b"]}, schema=SCHEMA
    ).evaluate(ledger)
    assert not verdict.passed
    assert verdict.matched == ["tags"]
    assert verdict.mismatches[0].argument == "steps"
    assert "item 0" in verdict.mismatches[0].reason


def test_nested_objects_compare_their_named_properties() -> None:
    ledger = _ledger(("refund", {"address": {"zip": "10001", "city": "NYC"}}))
    assert (
        ArgumentCorrectness(tool="refund", expected={"address": {"zip": "10001"}}, schema=SCHEMA)
        .evaluate(ledger)
        .passed
    )
    wrong = ArgumentCorrectness(tool="refund", expected={"address": {"zip": "10002"}}).evaluate(
        ledger
    )
    assert not wrong.passed
    assert "zip" in wrong.mismatches[0].reason


def test_without_a_schema_the_expected_value_type_decides() -> None:
    ledger = _ledger(("refund", {"amount": "50", "notify": "yes", "note": "  Broken  ITEM "}))
    verdict = ArgumentCorrectness(
        tool="refund", expected={"amount": 50, "notify": True, "note": "broken item"}
    ).evaluate(ledger)
    assert verdict.passed, verdict.mismatches
    exact = ArgumentCorrectness(
        tool="refund", expected={"note": "broken item"}, ignore_case=False
    ).evaluate(ledger)
    assert not exact.passed


def test_strict_keys_fails_on_arguments_not_expected() -> None:
    ledger = _ledger(("refund", {"order_id": "o1", "amount": 5}))
    assert ArgumentCorrectness(tool="refund", expected={"order_id": "o1"}).evaluate(ledger).passed
    strict = ArgumentCorrectness(
        tool="refund", expected={"order_id": "o1"}, strict_keys=True
    ).evaluate(ledger)
    assert not strict.passed
    assert strict.unexpected == ["amount"]


def test_occurrence_selects_which_call_must_agree() -> None:
    ledger = _ledger(("refund", {"amount": 1}), ("refund", {"amount": 2}))
    assert ArgumentCorrectness(tool="refund", expected={"amount": 2}).evaluate(ledger).passed
    assert (
        not ArgumentCorrectness(tool="refund", expected={"amount": 2}, occurrence="first")
        .evaluate(ledger)
        .passed
    )
    assert (
        ArgumentCorrectness(tool="refund", expected={"amount": 2}, occurrence="last")
        .evaluate(ledger)
        .passed
    )
    every = ArgumentCorrectness(tool="refund", expected={"amount": 2}, occurrence="all").evaluate(
        ledger
    )
    assert not every.passed
    assert every.details["judged"] == 2


def test_a_tool_never_called_scores_zero_and_lists_every_argument_as_missing() -> None:
    verdict = ArgumentCorrectness(tool="refund", expected={"order_id": "o1", "amount": 1}).evaluate(
        _ledger()
    )
    assert not verdict.passed
    assert verdict.score == 0.0
    assert verdict.missing == ["amount", "order_id"]
    assert "was not called" in verdict.reason


def test_a_missing_argument_is_a_mismatch_named_missing() -> None:
    verdict = ArgumentCorrectness(tool="refund", expected={"order_id": "o1", "amount": 1}).evaluate(
        _ledger(("refund", {"order_id": "o1"}))
    )
    assert not verdict.passed
    assert verdict.missing == ["amount"]
    assert verdict.score == pytest.approx(0.5)


def test_schema_is_accepted_under_its_public_name() -> None:
    evaluator = ArgumentCorrectness.model_validate(
        {"tool": "refund", "expected": {"count": 2}, "schema": SCHEMA}
    )
    assert evaluator.schema_ == SCHEMA
    assert evaluator.settings()["schema"] == SCHEMA


# --------------------------------------------------------------------------
# On a case
# --------------------------------------------------------------------------


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order."""
    return f"order {order_id}: total 49.90"


@tool
def issue_refund(order_id: str, amount: float) -> str:
    """Refund an order."""
    return f"refunded {amount}"


def _agent() -> Agent:
    return Agent(
        model=ScriptedModel(
            [
                tool_call("lookup_order", order_id="ord-1"),
                tool_call("issue_refund", order_id="ord-1", amount="49.9"),
                text("Refunded 49.90 on ord-1."),
            ]
        ),
        tools=[lookup_order, issue_refund],
    )


def test_each_evaluator_is_one_named_check_and_its_verdict_is_kept() -> None:
    case = EvalCase(
        name="path",
        prompt="refund ord-1",
        evaluators=[
            TrajectoryMatch(expected=["lookup_order", "issue_refund"]),
            TrajectoryMatch(expected=["issue_refund"], mode="subset"),
            TrajectoryMatch(expected=["lookup_order"], mode="subset"),
            ToolSelection(expected=["lookup_order"], forbidden=["delete_order"]),
            ArgumentCorrectness(tool="issue_refund", expected={"amount": 49.9}),
        ],
    )
    report = EvalRunner(agent=_agent()).run([case])
    result = report.results[0]
    assert result.passed, result.checks
    assert list(result.checks) == [
        "trajectory:strict",
        "trajectory:subset",
        "trajectory:subset#2",
        "tool_selection",
        "arguments:issue_refund",
    ]
    assert [v.evaluator for v in result.evaluations] == [
        "trajectory_match",
        "trajectory_match",
        "trajectory_match",
        "tool_selection",
        "argument_correctness",
    ]


def test_the_runner_fills_the_tool_schema_from_the_agent_under_test() -> None:
    """``amount`` arrives as the string "49.9"; the tool's schema says number, so it agrees."""
    case = EvalCase(
        name="path",
        prompt="refund ord-1",
        evaluators=[ArgumentCorrectness(tool="issue_refund", expected={"amount": 49.9})],
    )
    result = EvalRunner(agent=_agent()).run([case]).results[0]
    assert result.passed, result.checks
    verdict = result.evaluations[0]
    assert verdict.provenance is not None
    assert verdict.provenance.settings["schema"]["properties"]["amount"]["type"] == "number"


@pytest.mark.asyncio
async def test_evaluators_grade_the_same_on_the_async_path() -> None:
    case = EvalCase(
        name="path",
        prompt="refund ord-1",
        evaluators=[TrajectoryMatch(expected=["issue_refund", "lookup_order"])],
    )
    report = await EvalRunner(agent=_agent(), concurrency=1).arun([case])
    result = report.results[0]
    assert not result.passed
    assert result.checks == {"trajectory:strict": False}
    assert result.evaluations[0].score == pytest.approx(0.5)
    assert "trajectory:strict: FAILED" in report.summary()


def test_evaluator_settings_enter_the_fingerprint() -> None:
    base = EvalCase(name="c", prompt="p", evaluators=[TrajectoryMatch(expected=["a"])])
    variant = EvalCase(
        name="c", prompt="p", evaluators=[TrajectoryMatch(expected=["a"], mode="subset")]
    )
    assert suite_fingerprint([base]) != suite_fingerprint([variant])


def test_evaluators_round_trip_through_a_case_file(tmp_path: Path) -> None:
    case = EvalCase(
        name="path",
        prompt="refund ord-1",
        evaluators=[
            TrajectoryMatch(expected=["lookup_order"], mode="subset"),
            ToolSelection(expected=["lookup_order"], forbidden=["delete_order"]),
            ArgumentCorrectness(
                tool="issue_refund", expected={"amount": 49.9}, tolerance=0.1, schema=SCHEMA
            ),
        ],
    )
    path = tmp_path / "cases.json"
    dump_cases([case], path)
    written = json.loads(path.read_text())
    assert [e["kind"] for e in written[0]["evaluators"]] == [
        "trajectory_match",
        "tool_selection",
        "argument_correctness",
    ]
    assert "schema" in written[0]["evaluators"][2]
    loaded = load_cases(path)
    assert loaded[0].evaluators == case.evaluators
    assert suite_fingerprint(loaded) == suite_fingerprint([case])


def test_a_case_file_names_evaluators_by_kind(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "c",
                    "prompt": "p",
                    "evaluators": [
                        {"kind": "trajectory_match", "expected": ["a", "b"], "mode": "unordered"},
                        {"kind": "argument_correctness", "tool": "a", "expected": {"x": 1}},
                    ],
                }
            ]
        )
    )
    case = load_cases(path)[0]
    assert isinstance(case.evaluators[0], TrajectoryMatch)
    assert case.evaluators[0].mode == "unordered"
    assert isinstance(case.evaluators[1], ArgumentCorrectness)
