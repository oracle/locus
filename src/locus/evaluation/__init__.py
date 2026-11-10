# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Agent evaluation framework.

Provides systematic testing of agent quality:
- Define test cases with expected behaviors
- Run agents against test suites
- Score results and generate reports
- Grade free-form answers against a rubric with :class:`LLMJudge`
- Write the rubric as data with :class:`Rubric`: dimensions, weights and
  grade bands, with the composite and the letter computed by code
- Run the same suite against a ``StateGraph`` via :func:`as_eval_target`
"""

from locus.evaluation.framework import (
    EvalCase,
    EvalReport,
    EvalResult,
    EvalRunner,
)
from locus.evaluation.graph import GraphEvalTarget, as_eval_target
from locus.evaluation.judge import (
    JudgeProvenance,
    JudgeUnavailableError,
    LLMJudge,
    Verdict,
    check_trajectory,
    model_id_of,
)
from locus.evaluation.rubric import Cap, Dimension, Rubric


__all__ = [
    "Cap",
    "Dimension",
    "EvalCase",
    "EvalReport",
    "EvalResult",
    "EvalRunner",
    "GraphEvalTarget",
    "JudgeProvenance",
    "JudgeUnavailableError",
    "LLMJudge",
    "Rubric",
    "Verdict",
    "as_eval_target",
    "check_trajectory",
    "model_id_of",
]
