# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Agent evaluation framework.

Test agents systematically with defined expectations:
- Expected tool usage patterns
- Output content requirements
- Iteration and performance budgets
- Tool-call *ordering*, not just membership (``expected_tool_sequence``)
- LLM-as-judge scoring against a written rubric (``rubric``, and an
  ``EvalRunner(judge=...)``); see :mod:`tulip.evaluation.judge`
"""

from __future__ import annotations

import asyncio
import time
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
        description="Max duration in milliseconds",
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
        description="Graded by an LLM judge when EvalRunner has one. Use it "
        "where the right answer is not one exact string, which substring "
        "matching cannot express.",
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

    model_config = {"arbitrary_types_allowed": True}


class EvalReport(BaseModel):
    """Aggregated report from running an eval suite."""

    results: list[EvalResult] = Field(default_factory=list)
    total_cases: int = 0
    passed: int = 0
    failed: int = 0
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
            status = "PASS" if r.passed else "FAIL"
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


class EvalRunner:
    """Run evaluation cases against an agent.

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
                Cases carrying a ``rubric`` are graded by it; without one
                those cases are skipped rather than silently passing, so a
                suite cannot look green because nobody read the answers.
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
        """Run all eval cases and produce a report."""
        results: list[EvalResult] = []

        for case in cases:
            result = self._run_case(case)
            results.append(result)

        passed = sum(1 for r in results if r.passed)
        scores = [r.score for r in results]
        total_duration = sum(r.duration_ms for r in results)

        return EvalReport(
            results=results,
            total_cases=len(cases),
            passed=passed,
            failed=len(cases) - passed,
            avg_score=sum(scores) / len(scores) if scores else 0.0,
            total_duration_ms=total_duration,
        )

    async def arun(self, cases: list[EvalCase]) -> EvalReport:
        """Run every case concurrently, then report.

        The async path is the one that can grade with a judge: scoring is
        a model call, and ``run()`` is synchronous.
        """
        semaphore = asyncio.Semaphore(self.concurrency)

        async def one(case: EvalCase) -> EvalResult:
            async with semaphore:
                return await self._arun_case(case)

        results = list(await asyncio.gather(*(one(c) for c in cases)))
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
            avg_score=sum(scores) / len(scores) if scores else 0.0,
            total_duration_ms=sum(r.duration_ms for r in results),
        )

    def _run_case(self, case: EvalCase) -> EvalResult:
        """Run a single eval case."""
        start_time = time.perf_counter()
        checks: dict[str, bool] = {}

        try:
            agent_result = self.agent.run_sync(case.prompt)

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

            # Calculate score
            all_passed = all(checks.values()) if checks else True
            score = sum(checks.values()) / len(checks) if checks else 1.0

            return EvalResult(
                case_name=case.name,
                passed=all_passed,
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
            checks["within_duration_budget"] = duration_ms <= case.max_duration_ms

        if case.expected_tool_sequence:
            from tulip.evaluation.judge import check_trajectory  # noqa: PLC0415

            ok, _reason = check_trajectory(
                tools_called, case.expected_tool_sequence, exact=case.exact_tool_sequence
            )
            checks["tool_sequence"] = ok

        return checks

    async def _arun_case(self, case: EvalCase) -> EvalResult:
        """Run one case asynchronously, grading with the judge when asked."""
        started = time.perf_counter()
        checks: dict[str, bool] = {}
        try:
            agent_result = await self.agent.arun(case.prompt)
            output = agent_result.message or ""
            tools_called = [te.tool_name for te in agent_result.tool_executions]
            duration_ms = (time.perf_counter() - started) * 1000

            checks = self._structural_checks(
                case,
                output=output,
                tools_called=tools_called,
                iterations=agent_result.iterations,
                duration_ms=duration_ms,
            )

            judged: float | None = None
            if case.rubric:
                if self.judge is None:
                    # Not silently passed: a suite must not look green because
                    # nobody supplied the judge its cases asked for.
                    checks["rubric:no_judge_configured"] = False
                else:
                    verdict = await self.judge.score(
                        prompt=case.prompt, output=output, rubric=case.rubric
                    )
                    label = "judge_unparseable" if verdict.unparseable else "rubric"
                    checks[f"{label}:{verdict.reason[:60] or 'graded'}"] = verdict.passed
                    judged = verdict.score

            passed = all(checks.values()) if checks else True
            score = sum(checks.values()) / len(checks) if checks else 1.0
            # A graded case reports the judge's score rather than the fraction
            # of boxes ticked -- "0.5 because one of two checks failed" says
            # much less than "0.2, the answer named no policy".
            if judged is not None:
                score = judged

            return EvalResult(
                case_name=case.name,
                passed=passed,
                score=score,
                output=output,
                tools_called=tools_called,
                iterations=agent_result.iterations,
                duration_ms=duration_ms,
                checks=checks,
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
