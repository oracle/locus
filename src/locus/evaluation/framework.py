# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Agent evaluation framework.

Test agents systematically with defined expectations:
- Expected tool usage patterns
- Output content requirements
- Iteration and performance budgets
- LLM-as-judge scoring
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """A single evaluation test case.

    Defines what to send to the agent and what to expect back.

    Example:
        >>> case = EvalCase(
        ...     name="weather_lookup",
        ...     prompt="What's the weather in NYC?",
        ...     expected_tools=["get_weather"],
        ...     expected_output_contains=["temperature", "New York"],
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

    def __init__(self, agent: Any) -> None:
        self.agent = agent

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

    def _run_case(self, case: EvalCase) -> EvalResult:
        """Run a single eval case."""
        start_time = time.perf_counter()
        checks: dict[str, bool] = {}

        try:
            agent_result = self.agent.run_sync(case.prompt)

            output = agent_result.message or ""
            output_lower = output.lower()
            iterations = agent_result.iterations

            # Collect tool names from execution
            tools_called = [te.tool_name for te in agent_result.tool_executions]

            duration_ms = (time.perf_counter() - start_time) * 1000

            # Check: expected tools
            if case.expected_tools:
                for tool_name in case.expected_tools:
                    key = f"tool_called:{tool_name}"
                    checks[key] = tool_name in tools_called

            # Check: output contains expected strings
            for expected in case.expected_output_contains:
                key = f"output_contains:{expected}"
                checks[key] = expected.lower() in output_lower

            # Check: output does NOT contain excluded strings
            for excluded in case.expected_output_not_contains:
                key = f"output_not_contains:{excluded}"
                checks[key] = excluded.lower() not in output_lower

            # Check: iteration budget
            if case.max_iterations is not None:
                checks["within_iteration_budget"] = iterations <= case.max_iterations

            # Check: duration budget
            if case.max_duration_ms is not None:
                checks["within_duration_budget"] = duration_ms <= case.max_duration_ms

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
