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
- Read how far to trust a verdict with :mod:`locus.evaluation.reliability`:
  repeats, judge panels, second opinions, kappa and calibration
- Load cases from a file, save and diff reports, and gate a build on one
  with :class:`EvalGate`; the same from the command line as ``locus eval``
- Run the same suite against a ``StateGraph`` via :func:`as_eval_target`
"""

from locus.evaluation.datasets import dump_cases, load_cases
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
    EvalSample,
    TagSummary,
    suite_fingerprint,
)
from locus.evaluation.graph import GraphEvalTarget, as_eval_target
from locus.evaluation.judge import (
    Comparison,
    JudgeProvenance,
    JudgeUnavailableError,
    LLMJudge,
    Verdict,
    check_trajectory,
    model_id_of,
)
from locus.evaluation.reliability import (
    CalibrationDrift,
    Consistency,
    Kappa,
    PanelAgreement,
    TrustDecision,
    TrustGates,
    bootstrap_ci,
    calibration_drift,
    cohen_kappa,
    consistency,
    panel_agreement,
    trust_decision,
    wilson_interval,
)
from locus.evaluation.report import CaseDiff, EvalGate, GateResult, ReportDiff, diff_reports
from locus.evaluation.rubric import Cap, Dimension, Rubric


__all__ = [
    "CalibrationDrift",
    "Cap",
    "CaseDiff",
    "ClaimCheck",
    "Comparison",
    "Consistency",
    "Dimension",
    "EvalCase",
    "EvalGate",
    "EvalReport",
    "EvalResult",
    "EvalRunner",
    "EvalSample",
    "EvidenceLedger",
    "EvidenceRow",
    "GateResult",
    "GraphEvalTarget",
    "JudgeProvenance",
    "JudgeUnavailableError",
    "Kappa",
    "LLMJudge",
    "PanelAgreement",
    "ReceiptCheck",
    "ReportDiff",
    "Rubric",
    "TagSummary",
    "ToolCallExpectation",
    "TrustDecision",
    "TrustGates",
    "Verdict",
    "as_eval_target",
    "bootstrap_ci",
    "calibration_drift",
    "check_trajectory",
    "cohen_kappa",
    "consistency",
    "diff_reports",
    "dump_cases",
    "load_cases",
    "model_id_of",
    "panel_agreement",
    "settle_claims",
    "settle_receipts",
    "suite_fingerprint",
    "trust_decision",
    "wilson_interval",
]
