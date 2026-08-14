# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Agent evaluation framework.

Provides systematic testing of agent quality:
- Define test cases with expected behaviors
- Run agents against test suites
- Score results and generate reports
"""

from tulip.evaluation.framework import (
    EvalCase,
    EvalReport,
    EvalResult,
    EvalRunner,
)
from tulip.evaluation.judge import LLMJudge, Verdict, check_trajectory


__all__ = [
    "LLMJudge",
    "Verdict",
    "check_trajectory",
    "EvalCase",
    "EvalReport",
    "EvalResult",
    "EvalRunner",
]
