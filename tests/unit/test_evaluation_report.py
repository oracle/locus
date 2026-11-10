# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Comparable reports: fingerprints, tag summaries, save and load, the diff, the gate.

"Same suite, two providers, diff the report" needs the two reports to be
of the same suite, which is what the fingerprint says, and a per-case diff
that names what regressed rather than two averages. The gate is the rule a
build applies, reporting every reason it failed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from locus.agent import Agent
from locus.core.messages import Message, ToolCall
from locus.evaluation import (
    Dimension,
    EvalCase,
    EvalGate,
    EvalReport,
    EvalRunner,
    LLMJudge,
    Rubric,
    diff_reports,
    suite_fingerprint,
)
from locus.evaluation.judge import SUBMIT_TOOL
from locus.models.base import ModelResponse
from locus.testing import ScriptedModel, text


RUBRIC = Rubric(
    dimensions=[Dimension(key="root_cause", weight=0.5), Dimension(key="evidence", weight=0.5)],
    grade_thresholds=Rubric.default_grades(),
)

CASES = [
    EvalCase(name="a", prompt="say yes", expected_output_contains=["yes"], tags=["smoke"]),
    EvalCase(name="b", prompt="say no", expected_output_contains=["no"], tags=["smoke", "slow"]),
    EvalCase(name="c", prompt="anything", expected_output_contains=["maybe"]),
]


def _agent(reply: str) -> Agent:
    return Agent(model=ScriptedModel([text(reply)], repeat_last=True))


class _JudgeModel:
    def __init__(self, score: int) -> None:
        self.score = score
        self.config = SimpleNamespace(model="judge-model")

    async def complete(self, messages: list[Message], tools: Any = None, **_: Any) -> ModelResponse:
        call = ToolCall(
            id="v",
            name=SUBMIT_TOOL,
            arguments={
                "dimensions": [
                    {"key": "root_cause", "score": self.score, "reason": ""},
                    {"key": "evidence", "score": self.score, "reason": ""},
                ],
                "issues": [],
                "passed": True,
            },
        )
        return ModelResponse(message=Message.assistant(tool_calls=[call]), usage={})

    async def stream(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------
# fingerprint and aggregates
# --------------------------------------------------------------------------


def test_the_fingerprint_is_stable_and_order_independent() -> None:
    assert suite_fingerprint(CASES) == suite_fingerprint(list(reversed(CASES)))
    assert len(suite_fingerprint(CASES)) == 12


def test_the_fingerprint_moves_with_a_prompt_a_case_or_a_rubric_weight() -> None:
    base = suite_fingerprint(CASES)
    edited = [CASES[0].model_copy(update={"prompt": "say YES"}), *CASES[1:]]
    assert suite_fingerprint(edited) != base
    assert suite_fingerprint(CASES[:2]) != base
    with_rubric = [CASES[0].model_copy(update={"rubric": RUBRIC}), *CASES[1:]]
    reweighted = RUBRIC.model_copy(
        update={
            "dimensions": [
                Dimension(key="root_cause", weight=0.7),
                Dimension(key="evidence", weight=0.3),
            ]
        }
    )
    with_reweighted = [CASES[0].model_copy(update={"rubric": reweighted}), *CASES[1:]]
    assert suite_fingerprint(with_rubric) != suite_fingerprint(with_reweighted)


def test_metadata_does_not_move_the_fingerprint() -> None:
    noted = [CASES[0].model_copy(update={"metadata": {"owner": "x"}}), *CASES[1:]]
    assert suite_fingerprint(noted) == suite_fingerprint(CASES)


def test_the_report_carries_the_fingerprint_tags_and_check_failures() -> None:
    report = EvalRunner(agent=_agent("yes")).run(CASES)

    assert report.fingerprint == suite_fingerprint(CASES)
    assert report.by_tag["smoke"].total == 2
    assert report.by_tag["smoke"].passed == 1
    assert report.by_tag["smoke"].failed == 1
    assert report.by_tag["slow"].total == 1
    assert report.check_failures == {"output_contains:no": 1, "output_contains:maybe": 1}
    assert report.judge_models == []


@pytest.mark.asyncio
async def test_the_report_names_the_judge_models_and_rubric_revisions() -> None:
    case = EvalCase(name="graded", prompt="p", rubric=RUBRIC)
    runner = EvalRunner(agent=_agent("answer"), judge=LLMJudge(_JudgeModel(80)), concurrency=1)
    report = await runner.arun([case])
    assert report.judge_models == ["judge-model"]
    assert len(report.rubric_revisions) == 1


def test_a_report_round_trips_through_json(tmp_path: Path) -> None:
    report = EvalRunner(agent=_agent("yes")).run(CASES)
    path = report.save(tmp_path / "report.json")
    loaded = EvalReport.load(path)
    assert loaded == report
    assert loaded.fingerprint == report.fingerprint


def test_markdown_lists_every_case_with_its_failed_checks() -> None:
    report = EvalRunner(agent=_agent("yes")).run(CASES)
    table = report.to_markdown()
    assert table.startswith("**1/3 passed**")
    assert "| a | pass | 1.00 |  |  |" in table
    assert "| b | fail | 0.00 |  | output_contains:no |" in table


# --------------------------------------------------------------------------
# diff
# --------------------------------------------------------------------------


def test_the_diff_names_what_regressed_and_what_improved() -> None:
    before = EvalRunner(agent=_agent("yes")).run(CASES)
    after = EvalRunner(agent=_agent("no maybe")).run(CASES)
    diff = diff_reports(before, after)

    assert diff.same_suite
    assert diff.regressed == ["a"]
    assert diff.improved == ["b", "c"]
    assert diff.pass_delta == 1
    assert diff.avg_score_delta == pytest.approx(1 / 3, abs=1e-3)
    case_a = next(d for d in diff.cases if d.case_name == "a")
    assert case_a.change == "regressed"
    assert case_a.newly_failing_checks == ["output_contains:yes"]
    assert case_a.score_delta == -1.0
    assert "[REGRESSED] a: pass 1.00 -> fail 0.00" in diff.summary()
    assert "| a | regressed |" in diff.to_markdown()


def test_the_diff_refuses_two_different_suites_unless_forced() -> None:
    before = EvalRunner(agent=_agent("yes")).run(CASES)
    after = EvalRunner(agent=_agent("yes")).run(CASES[:2])
    with pytest.raises(ValueError, match="different suites"):
        diff_reports(before, after)
    forced = diff_reports(before, after, force=True)
    assert not forced.same_suite
    assert forced.removed == ["c"]
    assert "(different suites)" in forced.summary()


def test_the_diff_records_added_cases_and_dimension_movement() -> None:
    before = EvalRunner(agent=_agent("yes")).run(CASES[:1])
    before.fingerprint = None  # a report from before fingerprints existed
    after = EvalRunner(agent=_agent("yes")).run(CASES)
    diff = diff_reports(before, after)
    assert diff.added == ["b", "c"]
    assert not diff.same_suite


@pytest.mark.asyncio
async def test_dimension_deltas_come_from_the_verdicts() -> None:
    case = EvalCase(name="graded", prompt="p", rubric=RUBRIC)
    before = await EvalRunner(
        agent=_agent("a"), judge=LLMJudge(_JudgeModel(90)), concurrency=1
    ).arun([case])
    after = await EvalRunner(
        agent=_agent("a"), judge=LLMJudge(_JudgeModel(70)), concurrency=1
    ).arun([case])
    diff = diff_reports(before, after)
    assert diff.cases[0].dimension_deltas == {"evidence": -20, "root_cause": -20}
    assert diff.cases[0].baseline_grade == "A"
    assert diff.cases[0].candidate_grade == "C"
    assert diff.cases[0].change == "unchanged"  # both passed; the score moved


def test_a_not_graded_case_is_neither_a_regression_nor_an_improvement() -> None:
    before = EvalRunner(agent=_agent("yes")).run(CASES[:1])
    after = EvalRunner(agent=_agent("yes")).run(CASES[:1])
    after.results[0].not_graded_reason = "judge abstained"
    assert diff_reports(before, after).cases[0].change == "not_graded"


# --------------------------------------------------------------------------
# gate
# --------------------------------------------------------------------------


def test_the_default_gate_is_the_strict_reading_of_a_suite() -> None:
    report = EvalRunner(agent=_agent("yes")).run(CASES)
    result = EvalGate().check(report)
    assert not result.passed
    assert any("pass rate 0.33 below 1.00" in r for r in result.reasons)
    assert "Gate: FAILED" in result.summary()

    clean = EvalRunner(agent=_agent("yes")).run(CASES[:1])
    assert EvalGate().check(clean).passed
    assert EvalGate().check(clean).summary() == "Gate: passed"


def test_every_failing_rule_is_reported() -> None:
    report = EvalRunner(agent=_agent("yes")).run(CASES)
    report.not_graded = 1
    report.unstable = 2
    result = EvalGate(min_pass_rate=0.6, min_avg_score=0.9, max_unstable=1).check(report)
    assert len(result.reasons) == 4  # pass rate, avg score, not graded, unstable


def test_a_regression_against_the_baseline_fails_the_gate() -> None:
    before = EvalRunner(agent=_agent("yes")).run(CASES[:1])
    after = EvalRunner(agent=_agent("no")).run(CASES[:1])
    result = EvalGate(min_pass_rate=0.0).check(after, baseline=before)
    assert not result.passed
    assert any("regressed against the baseline: a" in r for r in result.reasons)
    assert EvalGate(min_pass_rate=0.0, no_regressions=False).check(after, baseline=before).passed


def test_an_average_drop_beyond_the_allowance_fails_the_gate() -> None:
    before = EvalRunner(agent=_agent("yes")).run(CASES)
    after = EvalRunner(agent=_agent("zzz")).run(CASES)
    result = EvalGate(min_pass_rate=0.0, no_regressions=False, max_avg_score_drop=0.1).check(
        after, baseline=before
    )
    assert any("dropped 0.33" in r for r in result.reasons)


def test_a_baseline_of_another_suite_is_a_gate_failure_not_a_crash() -> None:
    before = EvalRunner(agent=_agent("yes")).run(CASES[:2])
    after = EvalRunner(agent=_agent("yes")).run(CASES[:1])
    result = EvalGate().check(after, baseline=before)
    assert any("different suites" in r for r in result.reasons)
    assert EvalGate().check(after, baseline=before, force=True).passed


def test_an_empty_report_does_not_pass_the_gate() -> None:
    result = EvalGate().check(EvalReport())
    assert not result.passed
    assert result.reasons == ["the report has no cases"]
