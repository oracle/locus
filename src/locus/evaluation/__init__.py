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
- Settle figures and claims against the run's own tool results with
  :class:`EvidenceLedger`, and assert what the control layer did
- Run the same suite against a ``StateGraph`` via :func:`as_eval_target`
"""

from locus.evaluation.evidence import (
    ClaimCheck,
    EvidenceLedger,
    EvidenceRow,
    ReceiptCheck,
    ToolCallExpectation,
    settle_claims,
    settle_receipts,
)
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
    "ClaimCheck",
    "Dimension",
    "EvalCase",
    "EvalReport",
    "EvalResult",
    "EvalRunner",
    "EvidenceLedger",
    "EvidenceRow",
    "GraphEvalTarget",
    "JudgeProvenance",
    "JudgeUnavailableError",
    "LLMJudge",
    "ReceiptCheck",
    "Rubric",
    "ToolCallExpectation",
    "Verdict",
    "as_eval_target",
    "check_trajectory",
    "model_id_of",
    "settle_claims",
    "settle_receipts",
]
