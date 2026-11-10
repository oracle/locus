# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Agent evaluation framework.

Test agents systematically with defined expectations:
- Expected tool usage patterns
- Output content requirements
- Iteration and performance budgets
- Tool-call *ordering*, not just membership (``expected_tool_sequence``)
- LLM-as-judge scoring against a rubric (``rubric``, and an
  ``EvalRunner(judge=...)``); see :mod:`locus.evaluation.judge`. A string
  rubric is one pass criterion the judge answers; a
  :class:`~locus.evaluation.rubric.Rubric` names dimensions the judge scores
  and code composes into a score and a letter.

Grading rules (see :class:`EvalRunner` for the reasons):

- A case must assert something. One with no expectations and no rubric
  fails with the single check ``has_expectations`` rather than passing
  vacuously.
- ``passed`` is true only when every check passed. ``score`` is the
  fraction of checks that passed; for a judged case it is
  ``min(judge score, that fraction)``, so a failed case never scores 1.0.
- ``max_duration_ms`` is enforced while the case runs: an over-budget run is
  cancelled and reported ``timed_out``.
- An unreachable judge raises :class:`~locus.evaluation.judge.JudgeUnavailableError`
  out of :meth:`EvalRunner.arun` instead of being recorded as a failed case.
- The run's own tool results are evidence. ``require_receipts`` asks that
  every figure in the answer appear in one; ``claims`` are settled against
  them, by code where the figures line up and by the judge otherwise;
  ``forbidden_tools`` and ``expected_refusals`` read what the control layer
  did; ``max_total_tokens`` and ``max_cost_usd`` read the run's metrics.
- One reading is one opinion. ``repeats=`` runs every case several times
  and a case passes only when every graded reading passed (pass^k), with a
  ``consistency`` report saying whether the judge or the agent varied;
  ``judges=`` grades every answer with a panel and records their agreement
  and a trust decision between the first two.
- A judge that answers with something that is not a verdict has not graded
  the case. The result carries ``not_graded_reason``, counts in
  ``EvalReport.not_graded``, and is left out of ``failed`` and ``avg_score``:
  a dead judge must neither lower the agent's average nor pass it.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from locus.evaluation.evidence import (
    ClaimCheck,
    EvidenceLedger,
    ReceiptCheck,
    ToolCallExpectation,
    settle_claims,
    settle_receipts,
)
from locus.evaluation.judge import JudgeUnavailableError, Verdict, check_trajectory, model_id_of
from locus.evaluation.reliability import (
    Consistency,
    PanelAgreement,
    TrustDecision,
    consistency,
    panel_agreement,
    trust_decision,
)
from locus.evaluation.rubric import Rubric


class EvalCase(BaseModel):
    """A single evaluation test case.

    Defines what to send to the agent and what to expect back.

    Example:
        >>> case = EvalCase(
        ...     name="ioc_triage",
        ...     prompt="Is 198.51.100.23 malicious? Enrich it and decide.",
        ...     expected_tools=["enrich_indicator"],
        ...     expected_output_contains=["malicious", "198.51.100.23"],
        ...     max_iterations=5,
        ... )
    """

    name: str
    prompt: str
    expected_tools: list[str] = Field(
        default_factory=list,
        description="Tool names that should be called during execution",
    )
    expected_output_contains: list[str] = Field(
        default_factory=list,
        description="Strings that should appear in the final output (case-insensitive)",
    )
    expected_output_not_contains: list[str] = Field(
        default_factory=list,
        description="Strings that should NOT appear in the final output",
    )
    max_iterations: int | None = Field(
        default=None,
        description="Max iterations allowed (fail if exceeded)",
    )
    max_duration_ms: float | None = Field(
        default=None,
        description="Max duration in milliseconds. Enforced as a timeout: a "
        "run still going at the budget is cancelled and the case is reported "
        "timed_out.",
    )
    expected_tool_sequence: list[str] = Field(
        default_factory=list,
        description="Tool names in the order they must be called. Unlike "
        "expected_tools, this catches an agent that did the right things in "
        "the wrong order — refunding before it looked the order up.",
    )
    exact_tool_sequence: bool = Field(
        default=False,
        description="Require expected_tool_sequence to match exactly. The "
        "default allows extra calls around it, so a retry does not fail a "
        "test about ordering.",
    )
    rubric: str | Rubric | None = Field(
        default=None,
        description="Graded by an LLM judge in EvalRunner.arun when the "
        "runner has one. Use it where the right answer is not one exact "
        "string, which substring matching cannot express. A string is one "
        "pass criterion the judge answers; a Rubric names dimensions the "
        "judge scores and code composes into a score and a letter. The "
        "synchronous run() cannot call a judge and fails a case that carries "
        "a rubric.",
    )
    expected_tool_calls: list[ToolCallExpectation] = Field(
        default_factory=list,
        description="Tool calls the run must have made, with the arguments that "
        "matter: every argument given must match, others are not checked. A call "
        "the gate refused did not happen.",
    )
    forbidden_tools: list[str] = Field(
        default_factory=list,
        description="Tools that must not have executed. A call the gate refused "
        "(denied or held for approval) or that errored did not execute.",
    )
    expected_refusals: list[str] = Field(
        default_factory=list,
        description="Tools the gate must have refused at least once: the call's "
        "result carried the gate's denied or held_for_approval status.",
    )
    max_total_tokens: int | None = Field(
        default=None,
        description="Token budget for the run, read from the result's metrics. A "
        "target that reports no metrics fails the check rather than passing it.",
    )
    max_cost_usd: float | None = Field(
        default=None,
        description="Spend budget for the run, read from metrics.cost_usd. An "
        "unknown cost fails the check.",
    )
    require_receipts: bool = Field(
        default=False,
        description="Every figure and identifier in the answer must appear in a "
        "tool result or in the prompt. Deterministic; no model call.",
    )
    claims: list[str] = Field(
        default_factory=list,
        description="Claims about the run to settle against its evidence. Code "
        "confirms a claim whose figures all sit in one tool result; the judge "
        "settles the rest in arun(), citing rows and quoting them.",
    )
    grounding: float | None = Field(
        default=None,
        description="Minimum GSAR grounding score of the answer against the run's "
        "evidence, audited by the runner's judge model in arun(). An abstaining "
        "audit leaves the case not graded.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for filtering/grouping eval cases",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the eval case",
    )


def suite_fingerprint(cases: Sequence[EvalCase]) -> str:
    """A short hash of what a suite asks, so two reports are known to be comparable.

    Covers every case's name, prompt and expectations, in name order; a
    :class:`~locus.evaluation.rubric.Rubric` enters as its revision. Free-form
    ``metadata`` is left out. Editing a prompt, adding a case or retuning a
    weight moves the fingerprint, and :func:`~locus.evaluation.report.diff_reports`
    refuses two reports whose fingerprints differ.
    """
    payload = []
    for case in sorted(cases, key=lambda c: c.name):
        data = case.model_dump(mode="json", exclude={"metadata"})
        if isinstance(case.rubric, Rubric):
            data["rubric"] = {"revision": case.rubric.revision()}
        payload.append(data)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


class TagSummary(BaseModel):
    """Counts for the cases carrying one tag."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    not_graded: int = 0
    avg_score: float = 0.0


class EvalSample(BaseModel):
    """One reading of a case when it was run more than once."""

    passed: bool
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    grade: str | None = None
    checks: dict[str, bool] = Field(default_factory=dict)
    not_graded_reason: str | None = None
    duration_ms: float = 0.0
    verdict: Verdict | None = None


class EvalResult(BaseModel):
    """Result from evaluating a single case."""

    case_name: str
    passed: bool
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    output: str = ""
    tools_called: list[str] = Field(default_factory=list)
    iterations: int = 0
    duration_ms: float = 0.0
    checks: dict[str, bool] = Field(default_factory=dict)
    error: str | None = None
    timed_out: bool = Field(
        default=False,
        description="The run hit max_duration_ms and was cancelled; the case "
        "failed with score 0.0 and there is no output to grade.",
    )
    grade: str | None = Field(
        default=None, description="Letter from the rubric's bands, when the case had a Rubric"
    )
    verdict: Verdict | None = Field(
        default=None, description="The judge's full verdict, with its provenance"
    )
    not_graded_reason: str | None = Field(
        default=None,
        description="Set when the judge abstained: the case was neither passed nor "
        "failed, and is left out of avg_score.",
    )
    receipts: ReceiptCheck | None = Field(
        default=None, description="Where each figure in the answer was found, when asked"
    )
    claim_checks: list[ClaimCheck] = Field(
        default_factory=list, description="The case's claims, each settled by code or judge"
    )
    grounding_score: float | None = Field(
        default=None, description="GSAR score of the answer against the ledger, when audited"
    )
    samples: list[EvalSample] = Field(
        default_factory=list,
        description="Every reading when the case ran more than once; empty for a single run",
    )
    pass_rate: float | None = Field(
        default=None, description="Share of graded readings that passed, with repeats"
    )
    consistency: Consistency | None = Field(
        default=None, description="How much the readings disagreed, with repeats"
    )
    panel: PanelAgreement | None = Field(
        default=None, description="Agreement of the judge panel on this answer, with judges"
    )
    trust: TrustDecision | None = Field(
        default=None, description="Primary verdict against the second judge's, with judges"
    )
    panel_verdicts: list[Verdict] = Field(
        default_factory=list, description="Every judge's verdict, primary first, with judges"
    )

    model_config = {"arbitrary_types_allowed": True}


class EvalReport(BaseModel):
    """Aggregated report from running an eval suite."""

    results: list[EvalResult] = Field(default_factory=list)
    total_cases: int = 0
    passed: int = 0
    failed: int = 0
    timed_out: int = Field(default=0, description="Failed cases that hit their budget")
    not_graded: int = Field(
        default=0, description="Cases the judge abstained on; neither passed nor failed"
    )
    avg_score: float = Field(default=0.0, description="Mean score over the graded cases")
    total_duration_ms: float = 0.0
    grade_distribution: dict[str, int] = Field(
        default_factory=dict, description="How many graded cases landed on each letter"
    )
    self_graded: bool = Field(
        default=False, description="The judge model was the model under test on some case"
    )
    repeats: int = Field(default=1, description="Readings per case")
    unstable: int = Field(
        default=0, description="Cases whose readings disagreed enough to flag the judge"
    )
    fingerprint: str | None = Field(
        default=None, description="suite_fingerprint() of the cases this report measured"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    by_tag: dict[str, TagSummary] = Field(
        default_factory=dict, description="Counts per case tag, for a CI matrix"
    )
    check_failures: dict[str, int] = Field(
        default_factory=dict, description="How often each check label failed across cases"
    )
    judge_models: list[str] = Field(
        default_factory=list, description="Judge model ids seen on the verdicts"
    )
    rubric_revisions: list[str] = Field(
        default_factory=list, description="Rubric revisions seen on the verdicts"
    )

    model_config = {"arbitrary_types_allowed": True}

    def summary(self) -> str:
        """Generate a human-readable summary."""
        head = (
            f"Eval Report: {self.passed}/{self.total_cases} passed "
            f"(avg score: {self.avg_score:.2f})"
        )
        if self.not_graded:
            head += f", {self.not_graded} not graded"
        if self.repeats > 1:
            head += f", {self.repeats} readings per case"
            if self.unstable:
                head += f", {self.unstable} unstable"
        lines = [head, f"Total duration: {self.total_duration_ms:.0f}ms", ""]
        for r in self.results:
            if r.not_graded_reason:
                status = "NOT GRADED"
            elif r.passed:
                status = "PASS"
            else:
                status = "TIMEOUT" if r.timed_out else "FAIL"
            grade = f", grade {r.grade}" if r.grade else ""
            extra = ""
            if r.samples:
                graded = [s for s in r.samples if not s.not_graded_reason]
                extra = f", pass {sum(1 for s in graded if s.passed)}/{len(graded)}"
                if r.consistency is not None and r.consistency.flagged_unstable:
                    extra += ", unstable"
            if r.trust is not None:
                extra += f", {r.trust.decision}"
            lines.append(
                f"  [{status}] {r.case_name} "
                f"(score: {r.score:.2f}{grade}{extra}, {r.duration_ms:.0f}ms)"
            )
            if r.not_graded_reason:
                lines.append(f"         - not graded: {r.not_graded_reason}")
                continue
            if not r.passed:
                for check_name, check_passed in r.checks.items():
                    if not check_passed:
                        lines.append(f"         - {check_name}: FAILED")
                if r.error:
                    lines.append(f"         - error: {r.error}")
        if self.self_graded:
            lines.append("")
            lines.append(
                "warning: the judge is the model under test (self-graded); "
                "it measures self-consistency, not correctness"
            )
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """The report as a Markdown table, for a pull-request comment or a CI summary."""
        head = (
            f"**{self.passed}/{self.total_cases} passed**, avg score {self.avg_score:.2f}"
            + (f", {self.not_graded} not graded" if self.not_graded else "")
            + (f", {self.unstable} unstable" if self.unstable else "")
            + (f" · suite `{self.fingerprint}`" if self.fingerprint else "")
        )
        rows = [
            head,
            "",
            "| Case | Result | Score | Grade | Failed checks |",
            "|---|---|---|---|---|",
        ]
        for r in self.results:
            if r.not_graded_reason:
                status = "not graded"
            elif r.passed:
                status = "pass"
            else:
                status = "timeout" if r.timed_out else "fail"
            failed = ", ".join(k for k, ok in r.checks.items() if not ok)
            rows.append(
                f"| {r.case_name} | {status} | {r.score:.2f} | {r.grade or ''} | {failed} |"
            )
        return "\n".join(rows)

    def save(self, path: str | Path) -> Path:
        """Write the report as JSON; :meth:`load` reads it back."""
        file = Path(path)
        file.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return file

    @classmethod
    def load(cls, path: str | Path) -> EvalReport:
        """Read a report :meth:`save` wrote."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


#: Check recorded for a case that asserts nothing. It exists so the failure
#: is named in the report rather than being an unexplained red row.
NO_EXPECTATIONS_CHECK = "has_expectations"

#: How long a timed-out async run is given to unwind after it is cancelled
#: before the runner stops waiting for it.
_CANCEL_GRACE_S = 5.0


class _BudgetExceeded:
    """Sentinel: the run was still going when its budget ran out.

    A sentinel rather than ``TimeoutError`` so an agent that itself raises
    ``TimeoutError`` (an HTTP timeout, say) is reported as the agent's error,
    not mistaken for the runner's budget.
    """


_BUDGET_EXCEEDED = _BudgetExceeded()


class EvalRunner:
    """Run evaluation cases against an agent.

    How a case is graded:

    - **Something must be checked.** A case with no expectations and no
      rubric fails with the check ``has_expectations: False`` (score 0.0).
      Passing it would let a suite of empty cases report 100%.
    - **Pass/score.** ``passed`` requires every check to pass. ``score`` is
      the fraction of checks that passed. A judged case reports
      ``min(judge score, fraction)``: the judge's number when everything
      passed, and never more than the fraction otherwise, so a failed case
      can never score 1.0 and inflate ``avg_score``.
    - **Budget.** ``max_duration_ms`` is a timeout. In :meth:`arun` the run
      is cancelled at the budget; :meth:`run` stops waiting for it (a
      synchronous call cannot be interrupted, so it is abandoned on a daemon
      thread). Either way the case is ``timed_out`` with score 0.0. A run
      that finishes but is measured over budget still fails
      ``within_duration_budget``.
    - **Judge.** An unreachable judge raises
      :class:`~locus.evaluation.judge.JudgeUnavailableError` out of
      :meth:`arun` (the cases still running are cancelled): the suite could
      not be graded, which is not a failure of the agent. An unparseable
      verdict is still a failed check, labelled ``judge_unparseable``.
    - **Agent errors.** An agent that raises is a failed case with
      ``error`` set; one broken case does not stop the suite.

    Example:
        >>> runner = EvalRunner(agent=my_agent)
        >>> report = runner.run(
        ...     [
        ...         EvalCase(
        ...             name="basic", prompt="Hello", expected_output_contains=["hello"]
        ...         ),
        ...         EvalCase(
        ...             name="tool_use", prompt="Search for X", expected_tools=["search"]
        ...         ),
        ...     ]
        ... )
        >>> print(report.summary())
    """

    def __init__(
        self,
        agent: Any,
        *,
        judge: Any = None,
        judges: Sequence[Any] | None = None,
        concurrency: int = 4,
        repeats: int = 1,
    ) -> None:
        """
        Args:
            agent: The agent under test.
            judges: A panel. Every judged answer is graded by each of them;
                the first is the primary whose verdict scores the case, and
                the others feed ``EvalResult.panel`` and a ``trust``
                decision between the primary and the second. ``judge`` and
                ``judges`` may be combined; the single judge comes first.
            repeats: Readings per case in :meth:`arun`. With more than one,
                a case passes only when every graded reading passed, its
                score is the lowest graded reading's, and
                ``EvalResult.consistency`` says whether the readings agreed.
            judge: Optional :class:`~locus.evaluation.judge.LLMJudge`.
                Cases carrying a ``rubric`` are graded by it in
                :meth:`arun`; without one those cases fail rather than
                silently passing, so a suite cannot look green because
                nobody read the answers.
            concurrency: Cases run at once in :meth:`arun`. Evals are
                mostly latency, so running them serially wastes wall clock;
                the cap keeps a large suite from stampeding a rate limit.
                Set it to 1 if the agent's model is **stateful** — a
                scripted test double hands out its turns in order, so
                concurrent cases will consume each other's and fail in ways
                that have nothing to do with the agent.
        """
        self.agent = agent
        self.judges: list[Any] = ([judge] if judge is not None else []) + list(judges or [])
        self.judge = self.judges[0] if self.judges else None
        self.concurrency = max(1, concurrency)
        self.repeats = max(1, repeats)

    def run(self, cases: list[EvalCase]) -> EvalReport:
        """Run all eval cases, one after another, and produce a report.

        This path cannot call a judge; a case with a ``rubric`` fails with
        ``rubric:requires_arun``. Use :meth:`arun` for judged suites.
        """
        results = [self._run_case(case) for case in cases]
        return self._report(results, len(cases), cases=cases)

    async def arun(self, cases: list[EvalCase], *, repeats: int | None = None) -> EvalReport:
        """Run every case concurrently, then report.

        The async path is the one that can grade with a judge: scoring is
        a model call, and ``run()`` is synchronous.

        Raises:
            JudgeUnavailableError: The judge could not be reached. The
                cases still in flight are cancelled first.
        """
        readings = max(1, repeats if repeats is not None else self.repeats)
        semaphore = asyncio.Semaphore(self.concurrency)

        async def one(case: EvalCase) -> EvalResult:
            async with semaphore:
                return await self._arun_case(case)

        tasks = [asyncio.ensure_future(one(c)) for c in cases for _ in range(readings)]
        try:
            flat = list(await asyncio.gather(*tasks))
        except BaseException:
            # gather() does not cancel the siblings of a task that raised;
            # left alone they would keep calling the model after the suite
            # has already stopped.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        results = [
            self._combine(flat[index * readings : (index + 1) * readings])
            for index in range(len(cases))
        ]
        return self._report(results, len(cases), repeats=readings, cases=cases)

    @staticmethod
    def _combine(readings: list[EvalResult]) -> EvalResult:
        """Fold the readings of one case into one result.

        One reading is returned as it is. Several are pass^k: the case
        passes only when every graded reading passed, and the result shown
        is the most pessimistic graded reading (a failure before a pass,
        then the lowest score), with every reading kept under ``samples``
        and :func:`~locus.evaluation.reliability.consistency` saying how
        much they disagreed. A case with no graded reading is not graded.
        """
        if len(readings) == 1:
            return readings[0]
        graded = [r for r in readings if not r.not_graded_reason]
        if graded:
            shown = min(graded, key=lambda r: (r.passed, r.score)).model_copy()
            shown.passed = all(r.passed for r in graded)
            shown.pass_rate = sum(1 for r in graded if r.passed) / len(graded)
        else:
            shown = readings[0].model_copy()
            shown.not_graded_reason = (
                f"no graded reading in {len(readings)}: {readings[0].not_graded_reason}"
            )
        shown.samples = [
            EvalSample(
                passed=r.passed,
                score=r.score,
                grade=r.grade,
                checks=r.checks,
                not_graded_reason=r.not_graded_reason,
                duration_ms=r.duration_ms,
                verdict=r.verdict,
            )
            for r in readings
        ]
        shown.consistency = consistency(
            [
                r.verdict
                if r.verdict is not None
                else {
                    "passed": r.passed,
                    "score": r.score,
                    "grade": r.grade,
                    "abstained": bool(r.not_graded_reason),
                }
                for r in readings
            ]
        )
        shown.duration_ms = sum(r.duration_ms for r in readings)
        return shown

    def _report(
        self,
        results: list[EvalResult],
        total: int,
        *,
        repeats: int = 1,
        cases: Sequence[EvalCase] | None = None,
    ) -> EvalReport:
        """Aggregate results into a report.

        A case the judge abstained on is counted in ``not_graded`` and left
        out of ``failed`` and ``avg_score``: it says nothing about the agent.
        """
        passed = sum(1 for r in results if r.passed)
        graded = [r for r in results if not r.not_graded_reason]
        not_graded = len(results) - len(graded)
        scores = [r.score for r in graded]
        return EvalReport(
            results=results,
            total_cases=total,
            passed=passed,
            failed=total - passed - not_graded,
            timed_out=sum(1 for r in results if r.timed_out),
            not_graded=not_graded,
            avg_score=sum(scores) / len(scores) if scores else 0.0,
            total_duration_ms=sum(r.duration_ms for r in results),
            grade_distribution=dict(Counter(r.grade for r in graded if r.grade)),
            self_graded=any(
                r.verdict is not None
                and r.verdict.provenance is not None
                and r.verdict.provenance.self_graded is True
                for r in results
            ),
            repeats=repeats,
            unstable=sum(
                1 for r in results if r.consistency is not None and r.consistency.flagged_unstable
            ),
            fingerprint=suite_fingerprint(cases) if cases else None,
            by_tag=self._by_tag(results, cases or []),
            check_failures=dict(
                Counter(label for r in results for label, ok in r.checks.items() if not ok)
            ),
            judge_models=sorted(
                {
                    r.verdict.provenance.judge_model
                    for r in results
                    if r.verdict is not None
                    and r.verdict.provenance is not None
                    and r.verdict.provenance.judge_model
                }
            ),
            rubric_revisions=sorted(
                {
                    r.verdict.provenance.rubric_revision
                    for r in results
                    if r.verdict is not None
                    and r.verdict.provenance is not None
                    and r.verdict.provenance.rubric_revision
                }
            ),
        )

    @staticmethod
    def _by_tag(results: list[EvalResult], cases: Sequence[EvalCase]) -> dict[str, TagSummary]:
        """Fold the results under every tag their case carries."""
        tags_of = {case.name: case.tags for case in cases}
        buckets: dict[str, list[EvalResult]] = {}
        for result in results:
            for tag in tags_of.get(result.case_name, []):
                buckets.setdefault(tag, []).append(result)
        summaries: dict[str, TagSummary] = {}
        for tag, members in sorted(buckets.items()):
            graded = [r for r in members if not r.not_graded_reason]
            passed = sum(1 for r in graded if r.passed)
            summaries[tag] = TagSummary(
                total=len(members),
                passed=passed,
                failed=len(graded) - passed,
                not_graded=len(members) - len(graded),
                avg_score=sum(r.score for r in graded) / len(graded) if graded else 0.0,
            )
        return summaries

    async def _judge_panel(
        self, case: EvalCase, *, output: str, primary: Verdict
    ) -> tuple[list[Verdict], PanelAgreement | None, TrustDecision | None]:
        """Grade the answer with every other judge; agreement and trust against the primary."""
        others = [judge for judge in self.judges if judge is not self.judge]
        if not others:
            return [], None, None
        verdicts = [primary]
        for judge in others:
            second = await judge.score(prompt=case.prompt, output=output, rubric=case.rubric)
            self._stamp_self_graded(second)
            verdicts.append(second)
        return verdicts, panel_agreement(verdicts), trust_decision(primary, verdicts[1])

    def _stamp_self_graded(self, verdict: Verdict) -> None:
        """Record whether the judge is the model under test, when both ids are known."""
        if verdict.provenance is None:
            return
        agent_model = model_id_of(getattr(self.agent, "model", None))
        judge_model = verdict.provenance.judge_model
        if agent_model and judge_model:
            verdict.provenance.self_graded = agent_model == judge_model

    @staticmethod
    def _grade(checks: dict[str, bool], judged: float | None) -> tuple[bool, float]:
        """Combine checks (and an optional judge score) into pass and score.

        ``checks`` is updated in place with ``has_expectations: False`` when
        it is empty. The combination rule is documented on the class.
        """
        if not checks:
            checks[NO_EXPECTATIONS_CHECK] = False
        passed = all(checks.values())
        fraction = sum(checks.values()) / len(checks)
        score = fraction if judged is None else min(judged, fraction)
        return passed, score

    @staticmethod
    def _timed_out_result(case: EvalCase, duration_ms: float) -> EvalResult:
        return EvalResult(
            case_name=case.name,
            passed=False,
            score=0.0,
            duration_ms=duration_ms,
            checks={"within_duration_budget": False},
            error=f"timed out: still running at its {case.max_duration_ms:g}ms budget",
            timed_out=True,
        )

    @staticmethod
    def _call_sync_with_budget(fn: Callable[[], Any], budget_ms: float | None) -> Any:
        """Call ``fn``, giving up after ``budget_ms``.

        Returns ``_BUDGET_EXCEEDED`` on timeout. A synchronous call cannot
        be interrupted, so it runs on a daemon thread that is abandoned when
        the budget runs out; it must not keep the interpreter alive.
        """
        if budget_ms is None:
            return fn()

        outcome: dict[str, Any] = {}
        finished = threading.Event()
        context = contextvars.copy_context()

        def target() -> None:
            try:
                outcome["value"] = context.run(fn)
            except BaseException as exc:  # noqa: BLE001 — re-raised below
                outcome["error"] = exc
            finally:
                finished.set()

        threading.Thread(target=target, name="locus-eval-case", daemon=True).start()
        if not finished.wait(max(0.0, budget_ms) / 1000):
            return _BUDGET_EXCEEDED
        if "error" in outcome:
            raise outcome["error"]
        return outcome["value"]

    async def _call_async_with_budget(self, case: EvalCase) -> Any:
        """Await the agent, cancelling it at ``max_duration_ms``.

        Returns ``_BUDGET_EXCEEDED`` on timeout.
        """
        if case.max_duration_ms is None:
            return await self.agent.arun(case.prompt)

        task = asyncio.ensure_future(self.agent.arun(case.prompt))
        try:
            done, _ = await asyncio.wait({task}, timeout=max(0.0, case.max_duration_ms) / 1000)
        except asyncio.CancelledError:
            task.cancel()
            raise
        if task in done:
            return task.result()

        task.cancel()
        # Retrieve whatever the task ends with so an agent that swallows the
        # cancellation and fails later does not log "exception never
        # retrieved"; wait briefly for it to unwind, but never hang on it.
        task.add_done_callback(lambda t: t.cancelled() or t.exception())
        await asyncio.wait({task}, timeout=_CANCEL_GRACE_S)
        return _BUDGET_EXCEEDED

    def _run_case(self, case: EvalCase) -> EvalResult:
        """Run a single eval case."""
        start_time = time.perf_counter()
        checks: dict[str, bool] = {}

        try:
            agent_result = self._call_sync_with_budget(
                lambda: self.agent.run_sync(case.prompt), case.max_duration_ms
            )
            if agent_result is _BUDGET_EXCEEDED:
                return self._timed_out_result(case, (time.perf_counter() - start_time) * 1000)

            output = agent_result.message or ""
            iterations = agent_result.iterations
            ledger = EvidenceLedger.from_result(agent_result, prompt=case.prompt)
            tools_called = [row.tool_name for row in ledger.tool_rows]
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Shared with the async path on purpose. This used to be a
            # hand-copied second implementation that had drifted: it omitted
            # `expected_tool_sequence` entirely, so `run()` accepted a case
            # asserting the wrong tool order and reported it green. An
            # ordering assertion that is silently never evaluated is worse
            # than no assertion, because the suite says it was checked.
            checks = self._structural_checks(
                case,
                output=output,
                ledger=ledger,
                iterations=iterations,
                duration_ms=duration_ms,
                metrics=getattr(agent_result, "metrics", None),
            )
            receipts, claim_checks, open_claims = self._evidence_checks(
                case, output=output, ledger=ledger, checks=checks
            )
            # No judge can be awaited here. Ignoring a rubric, an open claim
            # or a grounding floor would report a graded case green without
            # anyone reading it.
            if case.rubric:
                checks["rubric:requires_arun"] = False
            if open_claims:
                checks["claims:requires_arun"] = False
            if case.grounding is not None:
                checks["grounding:requires_arun"] = False

            passed, score = self._grade(checks, None)

            return EvalResult(
                case_name=case.name,
                passed=passed,
                score=score,
                output=output,
                tools_called=tools_called,
                iterations=iterations,
                duration_ms=duration_ms,
                checks=checks,
                receipts=receipts,
                claim_checks=claim_checks,
            )

        except Exception as e:  # noqa: BLE001
            duration_ms = (time.perf_counter() - start_time) * 1000
            return EvalResult(
                case_name=case.name,
                passed=False,
                score=0.0,
                duration_ms=duration_ms,
                checks=checks,
                error=str(e),
            )

    def _structural_checks(
        self,
        case: EvalCase,
        *,
        output: str,
        ledger: EvidenceLedger,
        iterations: int,
        duration_ms: float,
        metrics: Any = None,
    ) -> dict[str, bool]:
        """Every check that needs no model call.

        Shared by the sync and async paths so the two cannot drift into
        grading the same case differently.
        """
        checks: dict[str, bool] = {}
        lowered = output.lower()
        tools_called = [row.tool_name for row in ledger.tool_rows]

        for tool_name in case.expected_tools:
            checks[f"tool_called:{tool_name}"] = tool_name in tools_called
        for expected in case.expected_output_contains:
            checks[f"output_contains:{expected}"] = expected.lower() in lowered
        for excluded in case.expected_output_not_contains:
            checks[f"output_not_contains:{excluded}"] = excluded.lower() not in lowered
        if case.max_iterations is not None:
            checks["within_iteration_budget"] = iterations <= case.max_iterations
        if case.max_duration_ms is not None:
            # Kept alongside the timeout: a run that finished just as the
            # budget ran out is still over it.
            checks["within_duration_budget"] = duration_ms <= case.max_duration_ms

        if case.expected_tool_sequence:
            ok, _reason = check_trajectory(
                tools_called, case.expected_tool_sequence, exact=case.exact_tool_sequence
            )
            checks["tool_sequence"] = ok

        for expectation in case.expected_tool_calls:
            label = f"tool_call:{expectation.name}"
            if expectation.arguments:
                args = ", ".join(f"{k}={v!r}" for k, v in expectation.arguments.items())
                label = f"{label}({args})"
            checks[label] = any(
                expectation.matches(row.tool_name, row.arguments)
                for row in ledger.tool_rows
                if row.executed
            )
        for tool_name in case.forbidden_tools:
            # A refused or errored call did not execute; that is the point of
            # the gate, and the point of asserting it.
            checks[f"forbidden_tool:{tool_name}"] = not ledger.executions_of(tool_name)
        for tool_name in case.expected_refusals:
            checks[f"refused:{tool_name}"] = bool(ledger.refusals_of(tool_name))

        if case.max_total_tokens is not None:
            total = getattr(metrics, "total_tokens", None)
            checks["within_token_budget"] = (
                isinstance(total, int | float) and total <= case.max_total_tokens
            )
        if case.max_cost_usd is not None:
            cost = getattr(metrics, "cost_usd", None)
            checks["within_cost_budget"] = (
                isinstance(cost, int | float) and cost <= case.max_cost_usd
            )

        return checks

    @staticmethod
    def _evidence_checks(
        case: EvalCase, *, output: str, ledger: EvidenceLedger, checks: dict[str, bool]
    ) -> tuple[ReceiptCheck | None, list[ClaimCheck], list[str]]:
        """Receipts and code-settled claims. Returns what the judge still has to settle.

        Both are deterministic. ``checks`` gains ``receipts:<found>/<total>``
        and one ``claim:<i>`` per claim code could confirm; the open claims
        get their check once the judge has looked (``arun``) or are marked
        as needing it (``run``).
        """
        receipts: ReceiptCheck | None = None
        if case.require_receipts:
            receipts = settle_receipts(output, ledger)
            checks[f"receipts:{receipts.summary}"] = receipts.passed

        claim_checks: list[ClaimCheck] = []
        open_claims: list[str] = []
        if case.claims:
            claim_checks, open_claims = settle_claims(case.claims, ledger)
            for check in claim_checks:
                checks[f"claim:{case.claims.index(check.claim)}"] = True
        return receipts, claim_checks, open_claims

    async def _judge_claims(
        self, case: EvalCase, *, open_claims: list[str], ledger: EvidenceLedger
    ) -> list[ClaimCheck] | None:
        """The judge settles the claims code could not. ``None`` when it abstained."""
        check_claims = getattr(self.judge, "check_claims", None)
        if check_claims is None:
            return []
        judged: list[ClaimCheck] | None = await check_claims(
            prompt=case.prompt, claims=open_claims, ledger=ledger
        )
        return judged

    async def _audit_grounding(self, *, output: str, ledger: EvidenceLedger) -> Any:
        """Run the GSAR judge over the answer and the ledger with the runner's judge model."""
        from locus.reasoning.gsar_judge import StructuredOutputGSARJudge

        model = getattr(self.judge, "model", None)
        audit = await StructuredOutputGSARJudge(model=model, strict=False).judge(
            report_synthesis=output, evidence_corpus=ledger.corpus()
        )
        reason = audit.abstain_reason or ""
        if audit.decision_status == "abstain" and reason.startswith("judge model call failed"):
            # The GSAR judge folds transport failures into an abstention; for
            # the suite that is the judge being down, not a finding.
            raise JudgeUnavailableError(f"grounding audit could not reach the judge: {reason}")
        return audit

    @staticmethod
    def _apply_caps(
        verdict: Verdict,
        rubric: Rubric,
        *,
        contradicted: bool,
        receipts_failed: bool,
        ungrounded: bool,
    ) -> None:
        """Drop the verdict's grade to each fired cap's ceiling, and say so."""
        if verdict.grade is None:
            return
        fired = {
            "contradicted_claim": contradicted,
            "receipts_failed": receipts_failed,
            "ungrounded": ungrounded,
        }
        for cap in rubric.caps:
            if not fired.get(cap.when, False):
                continue
            verdict.caps_applied.append(f"{cap.when}->{cap.grade}")
            current = rubric.rank(verdict.grade)
            ceiling = rubric.rank(cap.grade)
            if current is not None and ceiling is not None and ceiling > current:
                verdict.grade = cap.grade
                verdict.score = min(verdict.score, rubric.band_top(cap.grade) / 100)
                verdict.passed = round(verdict.score * 100) >= rubric.pass_score

    async def _arun_case(self, case: EvalCase) -> EvalResult:
        """Run one case asynchronously, grading with the judge when asked.

        Agent failures become a failed result. Judge failures are raised:
        they are outside the ``try`` on purpose.
        """
        started = time.perf_counter()
        checks: dict[str, bool] = {}
        try:
            agent_result = await self._call_async_with_budget(case)
            if agent_result is _BUDGET_EXCEEDED:
                return self._timed_out_result(case, (time.perf_counter() - started) * 1000)
            output = agent_result.message or ""
            iterations = agent_result.iterations
            ledger = EvidenceLedger.from_result(agent_result, prompt=case.prompt)
            tools_called = [row.tool_name for row in ledger.tool_rows]
            duration_ms = (time.perf_counter() - started) * 1000

            checks = self._structural_checks(
                case,
                output=output,
                ledger=ledger,
                iterations=iterations,
                duration_ms=duration_ms,
                metrics=getattr(agent_result, "metrics", None),
            )
            receipts, claim_checks, open_claims = self._evidence_checks(
                case, output=output, ledger=ledger, checks=checks
            )
        except Exception as e:  # noqa: BLE001
            return EvalResult(
                case_name=case.name,
                passed=False,
                score=0.0,
                duration_ms=(time.perf_counter() - started) * 1000,
                checks=checks,
                error=str(e),
            )

        not_graded_reason: str | None = None
        needs_judge = bool(case.rubric) or bool(open_claims) or case.grounding is not None
        if needs_judge and self.judge is None:
            # Not silently passed: a suite must not look green because
            # nobody supplied the judge its cases asked for.
            for label, wanted in (
                ("rubric", bool(case.rubric)),
                ("claims", bool(open_claims)),
                ("grounding", case.grounding is not None),
            ):
                if wanted:
                    checks[f"{label}:no_judge_configured"] = False

        # Judge calls below are deliberately not caught: an unreachable judge
        # raises (JudgeUnavailableError). Recording it as a failed case would
        # blame the agent for the judge being down.
        if open_claims and self.judge is not None:
            judged = await self._judge_claims(case, open_claims=open_claims, ledger=ledger)
            if judged is None:
                checks["judge_unparseable:claim checks"] = False
                not_graded_reason = "the judge returned no claim checks"
            else:
                claim_checks.extend(judged)
        for check in claim_checks:
            if check.settled_by == "judge":
                checks[f"claim:{case.claims.index(check.claim)}"] = check.verdict == "confirmed"

        grounding_score: float | None = None
        grounding_ok: bool | None = None
        if case.grounding is not None and self.judge is not None:
            from locus.reasoning.gsar import gsar_score

            audit = await self._audit_grounding(output=output, ledger=ledger)
            if audit.decision_status == "abstain":
                checks["judge_unparseable:grounding audit"] = False
                not_graded_reason = not_graded_reason or (
                    f"grounding audit abstained: {audit.abstain_reason or 'no reason given'}"
                )
            else:
                grounding_score = gsar_score(audit.to_partition())
                grounding_ok = grounding_score >= case.grounding
                checks[f"grounding:S>={case.grounding:g}"] = grounding_ok

        judged_score: float | None = None
        verdict: Verdict | None = None
        panel_verdicts: list[Verdict] = []
        panel: PanelAgreement | None = None
        trust: TrustDecision | None = None
        if case.rubric and self.judge is not None:
            verdict = await self.judge.score(prompt=case.prompt, output=output, rubric=case.rubric)
            self._stamp_self_graded(verdict)
            if verdict.abstained or verdict.unparseable:
                # The judge answered, but not with a verdict. That is a
                # finding about the judge, so the case is not graded
                # rather than failed; the check keeps the failure visible.
                checks[f"judge_unparseable:{verdict.reason[:60] or 'no verdict'}"] = False
                not_graded_reason = not_graded_reason or (
                    verdict.reason or "the judge returned no verdict"
                )
            else:
                if isinstance(case.rubric, Rubric):
                    self._apply_caps(
                        verdict,
                        case.rubric,
                        contradicted=any(c.verdict == "contradicted" for c in claim_checks),
                        receipts_failed=receipts is not None and not receipts.passed,
                        ungrounded=grounding_ok is False,
                    )
                checks[f"rubric:{verdict.reason[:60] or 'graded'}"] = verdict.passed
                judged_score = verdict.score
                panel_verdicts, panel, trust = await self._judge_panel(
                    case, output=output, primary=verdict
                )

        # A graded case reports the judge's score rather than the fraction
        # of boxes ticked -- "0.5 because one of two checks failed" says
        # much less than "0.2, the answer named no policy" -- but capped by
        # that fraction, so a failed case never reports 1.0.
        passed, score = self._grade(checks, judged_score)
        if not_graded_reason:
            passed, score = False, 0.0

        return EvalResult(
            case_name=case.name,
            passed=passed,
            score=score,
            output=output,
            tools_called=tools_called,
            iterations=iterations,
            duration_ms=duration_ms,
            checks=checks,
            grade=verdict.grade if verdict is not None and not not_graded_reason else None,
            verdict=verdict,
            not_graded_reason=not_graded_reason,
            receipts=receipts,
            claim_checks=claim_checks,
            grounding_score=grounding_score,
            panel=panel,
            trust=trust,
            panel_verdicts=panel_verdicts,
        )
