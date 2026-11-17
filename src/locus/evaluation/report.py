# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Comparable reports: a diff between two runs of one suite, and a gate for CI.

"Same suite, two providers, diff the report" is the sentence every
evaluation page writes and no evaluation harness makes easy. Two reports
are comparable only when they measured the same thing, so each carries a
``fingerprint`` of its cases, and :func:`diff_reports` refuses to compare
two different suites unless told to. The diff is per case: what passed and
now fails, what failed and now passes, how the scores and the dimension
scores moved, what was added and removed.

An :class:`EvalGate` is the rule a build applies to a report: a pass rate
to clear, an average to hold, how many cases may be left not graded, how
many may be unstable, and, given a baseline, whether any case may regress
or the average may drop. It returns every reason it failed, not the first.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from locus.evaluation.framework import EvalReport, EvalResult, suite_fingerprint


__all__ = [
    "CaseDiff",
    "EvalGate",
    "GateResult",
    "ReportDiff",
    "diff_reports",
    "suite_fingerprint",
]


class CaseDiff(BaseModel):
    """How one case moved between a baseline report and a candidate report."""

    case_name: str
    change: Literal["regressed", "improved", "unchanged", "added", "removed", "not_graded"]
    baseline_passed: bool | None = None
    candidate_passed: bool | None = None
    baseline_score: float | None = None
    candidate_score: float | None = None
    score_delta: float | None = None
    baseline_grade: str | None = None
    candidate_grade: str | None = None
    dimension_deltas: dict[str, int] = Field(default_factory=dict)
    newly_failing_checks: list[str] = Field(default_factory=list)


class ReportDiff(BaseModel):
    """The per-case and aggregate movement between two reports."""

    baseline_fingerprint: str | None = None
    candidate_fingerprint: str | None = None
    same_suite: bool = Field(description="Both fingerprints known and equal")
    cases: list[CaseDiff] = Field(default_factory=list)
    regressed: list[str] = Field(default_factory=list)
    improved: list[str] = Field(default_factory=list)
    added: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    pass_delta: int = Field(default=0, description="Candidate passed minus baseline passed")
    avg_score_delta: float = 0.0

    def summary(self) -> str:
        """A readable account of what moved."""
        head = (
            f"Report diff: {len(self.regressed)} regressed, {len(self.improved)} improved, "
            f"{len(self.added)} added, {len(self.removed)} removed; "
            f"passed {self.pass_delta:+d}, avg score {self.avg_score_delta:+.2f}"
        )
        if not self.same_suite:
            head += " (different suites)"
        lines = [head, ""]
        for case in self.cases:
            if case.change == "unchanged":
                continue
            before = _cell(case.baseline_passed, case.baseline_score, case.baseline_grade)
            after = _cell(case.candidate_passed, case.candidate_score, case.candidate_grade)
            lines.append(f"  [{case.change.upper()}] {case.case_name}: {before} -> {after}")
            for check in case.newly_failing_checks:
                lines.append(f"         - now failing: {check}")
            for key, delta in case.dimension_deltas.items():
                if delta:
                    lines.append(f"         - {key}: {delta:+d}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """The diff as a Markdown table, for a pull-request comment."""
        rows = [
            "| Case | Change | Baseline | Candidate | Score delta |",
            "|---|---|---|---|---|",
        ]
        for case in self.cases:
            delta = f"{case.score_delta:+.2f}" if case.score_delta is not None else ""
            rows.append(
                f"| {case.case_name} | {case.change} | "
                f"{_cell(case.baseline_passed, case.baseline_score, case.baseline_grade)} | "
                f"{_cell(case.candidate_passed, case.candidate_score, case.candidate_grade)} | "
                f"{delta} |"
            )
        return "\n".join(rows)


def _cell(passed: bool | None, score: float | None, grade: str | None) -> str:
    if passed is None and score is None:
        return "-"
    status = "pass" if passed else "fail"
    text = f"{status} {score:.2f}" if score is not None else status
    return f"{text} {grade}" if grade else text


def _by_name(report: EvalReport) -> dict[str, EvalResult]:
    return {result.case_name: result for result in report.results}


def diff_reports(baseline: EvalReport, candidate: EvalReport, *, force: bool = False) -> ReportDiff:
    """Compare two reports of the same suite, case by case.

    Raises:
        ValueError: When both fingerprints are known and differ, unless
            ``force`` is set: a diff across two different suites says
            nothing about a regression.
    """
    same = (
        baseline.fingerprint is not None
        and candidate.fingerprint is not None
        and baseline.fingerprint == candidate.fingerprint
    )
    if (
        not same
        and baseline.fingerprint is not None
        and candidate.fingerprint is not None
        and not force
    ):
        raise ValueError(
            f"the reports are of different suites ({baseline.fingerprint} vs "
            f"{candidate.fingerprint}); pass force=True to diff them anyway"
        )

    before, after = _by_name(baseline), _by_name(candidate)
    names = list(before) + [name for name in after if name not in before]
    cases: list[CaseDiff] = []
    for name in names:
        b, c = before.get(name), after.get(name)
        if b is None or c is None:
            present = c if b is None else b
            cases.append(
                CaseDiff(
                    case_name=name,
                    change="added" if b is None else "removed",
                    baseline_passed=None if b is None else b.passed,
                    candidate_passed=None if c is None else c.passed,
                    baseline_score=None if b is None else b.score,
                    candidate_score=None if c is None else c.score,
                    baseline_grade=None if b is None else b.grade,
                    candidate_grade=None if c is None else c.grade,
                    newly_failing_checks=[]
                    if present is None or present is b
                    else [k for k, ok in present.checks.items() if not ok],
                )
            )
            continue
        if b.not_graded_reason or c.not_graded_reason:
            change: Literal["regressed", "improved", "unchanged", "not_graded"] = "not_graded"
        elif b.passed and not c.passed:
            change = "regressed"
        elif not b.passed and c.passed:
            change = "improved"
        else:
            change = "unchanged"
        b_dims = b.verdict.dimensions if b.verdict is not None else {}
        c_dims = c.verdict.dimensions if c.verdict is not None else {}
        cases.append(
            CaseDiff(
                case_name=name,
                change=change,
                baseline_passed=b.passed,
                candidate_passed=c.passed,
                baseline_score=b.score,
                candidate_score=c.score,
                score_delta=round(c.score - b.score, 4),
                baseline_grade=b.grade,
                candidate_grade=c.grade,
                dimension_deltas={
                    key: c_dims.get(key, 0) - b_dims.get(key, 0)
                    for key in sorted(set(b_dims) | set(c_dims))
                },
                newly_failing_checks=[
                    key for key, ok in c.checks.items() if not ok and b.checks.get(key, True)
                ],
            )
        )
    return ReportDiff(
        baseline_fingerprint=baseline.fingerprint,
        candidate_fingerprint=candidate.fingerprint,
        same_suite=same,
        cases=cases,
        regressed=[d.case_name for d in cases if d.change == "regressed"],
        improved=[d.case_name for d in cases if d.change == "improved"],
        added=[d.case_name for d in cases if d.change == "added"],
        removed=[d.case_name for d in cases if d.change == "removed"],
        pass_delta=candidate.passed - baseline.passed,
        avg_score_delta=round(candidate.avg_score - baseline.avg_score, 4),
    )


class GateResult(BaseModel):
    """Whether a report clears the gate, and every reason it does not."""

    passed: bool
    reasons: list[str] = Field(default_factory=list)

    def summary(self) -> str:
        if self.passed:
            return "Gate: passed"
        return "Gate: FAILED\n" + "\n".join(f"  - {reason}" for reason in self.reasons)


class EvalGate(BaseModel):
    """The rule a build applies to a report.

    Every threshold is evaluated and every failure reported. The defaults
    are the strict reading of a regression suite: every graded case passes,
    nothing is left not graded, and, against a baseline, nothing regresses.
    """

    min_pass_rate: float = Field(default=1.0, ge=0.0, le=1.0, description="Over graded cases")
    min_avg_score: float | None = Field(default=None, ge=0.0, le=1.0)
    max_not_graded: int = Field(default=0, ge=0)
    max_unstable: int | None = Field(default=None, ge=0)
    no_regressions: bool = Field(default=True, description="With a baseline: no case may regress")
    max_avg_score_drop: float | None = Field(
        default=None, ge=0.0, description="With a baseline: the average may not drop by more"
    )

    def check(
        self, report: EvalReport, *, baseline: EvalReport | None = None, force: bool = False
    ) -> GateResult:
        """Apply the gate to ``report``, and to its diff against ``baseline`` when given."""
        reasons: list[str] = []
        graded = report.total_cases - report.not_graded
        if report.total_cases == 0:
            reasons.append("the report has no cases")
        pass_rate = report.passed / graded if graded else 0.0
        if graded and pass_rate < self.min_pass_rate:
            reasons.append(
                f"pass rate {pass_rate:.2f} below {self.min_pass_rate:.2f} "
                f"({report.passed}/{graded} graded cases passed)"
            )
        if self.min_avg_score is not None and report.avg_score < self.min_avg_score:
            reasons.append(f"avg score {report.avg_score:.2f} below {self.min_avg_score:.2f}")
        if report.not_graded > self.max_not_graded:
            reasons.append(
                f"{report.not_graded} case(s) not graded, at most {self.max_not_graded} allowed"
            )
        if self.max_unstable is not None and report.unstable > self.max_unstable:
            reasons.append(
                f"{report.unstable} unstable case(s), at most {self.max_unstable} allowed"
            )
        if baseline is not None:
            try:
                diff = diff_reports(baseline, report, force=force)
            except ValueError as exc:
                reasons.append(str(exc))
            else:
                if self.no_regressions and diff.regressed:
                    reasons.append(f"regressed against the baseline: {', '.join(diff.regressed)}")
                if (
                    self.max_avg_score_drop is not None
                    and -diff.avg_score_delta > self.max_avg_score_drop
                ):
                    reasons.append(
                        f"avg score dropped {-diff.avg_score_delta:.2f} against the baseline, "
                        f"at most {self.max_avg_score_drop:.2f} allowed"
                    )
        return GateResult(passed=not reasons, reasons=reasons)
