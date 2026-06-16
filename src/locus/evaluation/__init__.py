# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Agent evaluation framework.

Provides systematic testing of agent quality:
- Define test cases with expected behaviors
- Run agents against test suites
- Score results and generate reports
"""

from locus.evaluation.framework import (
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
