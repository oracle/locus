# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Agent evaluation framework.

Test agents systematically with defined expectations:
- Expected tool usage patterns
- Output content requirements
- Iteration and performance budgets
- Tool-call *ordering*, not just membership (``expected_tool_sequence``)
- LLM-as-judge scoring against a written rubric (``rubric``, and an
  ``EvalRunner(judge=...)``); see :mod:`tulip.evaluation.judge`

Grading rules (see :class:`EvalRunner` for the reasons):

- A case must assert something. One with no expectations and no rubric
  fails with the single check ``has_expectations`` rather than passing
  vacuously.
- ``passed`` is true only when every check passed. ``score`` is the
  fraction of checks that passed; for a judged case it is
  ``min(judge score, that fraction)``, so a failed case never scores 1.0.
- ``max_duration_ms`` is enforced while the case runs: an over-budget run is
  cancelled and reported ``timed_out``.
- An unreachable judge raises :class:`~tulip.evaluation.judge.JudgeUnavailableError`
  out of :meth:`EvalRunner.arun` instead of being recorded as a failed case.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field


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
    rubric: str | None = Field(
        default=None,
        description="Graded by an LLM judge in EvalRunner.arun when the "
        "runner has one. Use it where the right answer is not one exact "
        "string, which substring matching cannot express. The synchronous "
        "run() cannot call a judge and fails a case that carries a rubric.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for filtering/grouping eval cases",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the eval case",
    )


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

    model_config = {"arbitrary_types_allowed": True}


class EvalReport(BaseModel):
    """Aggregated report from running an eval suite."""

    results: list[EvalResult] = Field(default_factory=list)
    total_cases: int = 0
    passed: int = 0
    failed: int = 0
    timed_out: int = Field(default=0, description="Failed cases that hit their budget")
    avg_score: float = 0.0
    total_duration_ms: float = 0.0

    model_config = {"arbitrary_types_allowed": True}

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Eval Report: {self.passed}/{self.total_cases} passed "
            f"(avg score: {self.avg_score:.2f})",
            f"Total duration: {self.total_duration_ms:.0f}ms",
            "",
        ]
        for r in self.results:
            status = "PASS" if r.passed else ("TIMEOUT" if r.timed_out else "FAIL")
            lines.append(
                f"  [{status}] {r.case_name} (score: {r.score:.2f}, {r.duration_ms:.0f}ms)"
            )
            if not r.passed:
                for check_name, check_passed in r.checks.items():
                    if not check_passed:
                        lines.append(f"         - {check_name}: FAILED")
                if r.error:
                    lines.append(f"         - error: {r.error}")
        return "\n".join(lines)


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
      :class:`~tulip.evaluation.judge.JudgeUnavailableError` out of
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

    def __init__(self, agent: Any, *, judge: Any = None, concurrency: int = 4) -> None:
        """
        Args:
            agent: The agent under test.
            judge: Optional :class:`~tulip.evaluation.judge.LLMJudge`.
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
        self.judge = judge
        self.concurrency = max(1, concurrency)

    def run(self, cases: list[EvalCase]) -> EvalReport:
        """Run all eval cases, one after another, and produce a report.

        This path cannot call a judge; a case with a ``rubric`` fails with
        ``rubric:requires_arun``. Use :meth:`arun` for judged suites.
        """
        results = [self._run_case(case) for case in cases]
        return self._report(results, len(cases))

    async def arun(self, cases: list[EvalCase]) -> EvalReport:
        """Run every case concurrently, then report.

        The async path is the one that can grade with a judge: scoring is
        a model call, and ``run()`` is synchronous.

        Raises:
            JudgeUnavailableError: The judge could not be reached. The
                cases still in flight are cancelled first.
        """
        semaphore = asyncio.Semaphore(self.concurrency)

        async def one(case: EvalCase) -> EvalResult:
            async with semaphore:
                return await self._arun_case(case)

        tasks = [asyncio.ensure_future(one(c)) for c in cases]
        try:
            results = list(await asyncio.gather(*tasks))
        except BaseException:
            # gather() does not cancel the siblings of a task that raised;
            # left alone they would keep calling the model after the suite
            # has already stopped.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return self._report(results, len(cases))

    def _report(self, results: list[EvalResult], total: int) -> EvalReport:
        """Aggregate results into a report."""
        passed = sum(1 for r in results if r.passed)
        scores = [r.score for r in results]
        return EvalReport(
            results=results,
            total_cases=total,
            passed=passed,
            failed=total - passed,
            timed_out=sum(1 for r in results if r.timed_out),
            avg_score=sum(scores) / len(scores) if scores else 0.0,
            total_duration_ms=sum(r.duration_ms for r in results),
        )

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

        threading.Thread(target=target, name="tulip-eval-case", daemon=True).start()
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

            # Collect tool names from execution
            tools_called = [te.tool_name for te in agent_result.tool_executions]

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
                tools_called=tools_called,
                iterations=iterations,
                duration_ms=duration_ms,
            )
            if case.rubric:
                # No judge can be awaited here. Ignoring the rubric would
                # report a graded case green without anyone reading it.
                checks["rubric:requires_arun"] = False

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
        tools_called: list[str],
        iterations: int,
        duration_ms: float,
    ) -> dict[str, bool]:
        """Every check that needs no model call.

        Shared by the sync and async paths so the two cannot drift into
        grading the same case differently.
        """
        checks: dict[str, bool] = {}
        lowered = output.lower()

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
            from tulip.evaluation.judge import check_trajectory  # noqa: PLC0415

            ok, _reason = check_trajectory(
                tools_called, case.expected_tool_sequence, exact=case.exact_tool_sequence
            )
            checks["tool_sequence"] = ok

        return checks

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
            tools_called = [te.tool_name for te in agent_result.tool_executions]
            iterations = agent_result.iterations
            duration_ms = (time.perf_counter() - started) * 1000

            checks = self._structural_checks(
                case,
                output=output,
                tools_called=tools_called,
                iterations=iterations,
                duration_ms=duration_ms,
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

        judged: float | None = None
        if case.rubric:
            if self.judge is None:
                # Not silently passed: a suite must not look green because
                # nobody supplied the judge its cases asked for.
                checks["rubric:no_judge_configured"] = False
            else:
                # Deliberately not caught: an unreachable judge raises
                # (JudgeUnavailableError). Recording it as a failed case
                # would blame the agent for the judge being down.
                verdict = await self.judge.score(
                    prompt=case.prompt, output=output, rubric=case.rubric
                )
                label = "judge_unparseable" if verdict.unparseable else "rubric"
                checks[f"{label}:{verdict.reason[:60] or 'graded'}"] = verdict.passed
                judged = verdict.score

        # A graded case reports the judge's score rather than the fraction
        # of boxes ticked -- "0.5 because one of two checks failed" says
        # much less than "0.2, the answer named no policy" -- but capped by
        # that fraction, so a failed case never reports 1.0.
        passed, score = self._grade(checks, judged)

        return EvalResult(
            case_name=case.name,
            passed=passed,
            score=score,
            output=output,
            tools_called=tools_called,
            iterations=iterations,
            duration_ms=duration_ms,
            checks=checks,
        )
