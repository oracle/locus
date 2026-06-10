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


__all__ = [
    "EvalCase",
    "EvalReport",
    "EvalResult",
    "EvalRunner",
]
