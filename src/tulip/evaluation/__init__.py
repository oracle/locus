# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Agent evaluation framework.

Provides systematic testing of agent quality:
- Define test cases with expected behaviors
- Run agents against test suites
- Score results and generate reports
- Grade free-form answers against a rubric with :class:`LLMJudge`
- Run the same suite against a ``StateGraph`` via :func:`as_eval_target`
"""

from tulip.evaluation.framework import (
    EvalCase,
    EvalReport,
    EvalResult,
    EvalRunner,
)
from tulip.evaluation.graph import GraphEvalTarget, as_eval_target
from tulip.evaluation.judge import LLMJudge, Verdict, check_trajectory


__all__ = [
    "GraphEvalTarget",
    "LLMJudge",
    "Verdict",
    "check_trajectory",
    "EvalCase",
    "EvalReport",
    "EvalResult",
    "EvalRunner",
    "as_eval_target",
]
