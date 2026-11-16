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
- Score the path, not only the answer: :class:`TrajectoryMatch`,
  :class:`ToolSelection` and :class:`ArgumentCorrectness` read the evidence
  ledger with no model call
- Play a multi-turn case with a :class:`SimulatedUser`: a persona and a goal
  a model acts out against the agent, within a turn budget
- Grade a sample of live runs off the event bus with :class:`OnlineEvaluator`
- Keep reports in a store (:func:`report_store`: a directory, an Oracle AI
  Database table, an OCI or S3 bucket) and read the :func:`trend` across them
"""

from locus.evaluation.datasets import dump_cases, load_cases
from locus.evaluation.evaluators import (
    ArgumentCorrectness,
    ArgumentMismatch,
    EvaluatorProvenance,
    EvaluatorVerdict,
    ToolSelection,
    TrajectoryMatch,
)
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
from locus.evaluation.online import (
    OnlineBudget,
    OnlineEvaluator,
    RecordedRun,
    every_nth,
    fraction,
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
from locus.evaluation.simulator import (
    Conversation,
    ConversationTurn,
    SimulatedUser,
    SimulatorUnavailableError,
    UserSimulator,
)
from locus.evaluation.stores import (
    FileReportStore,
    OCIBucketReportStore,
    OracleReportStore,
    ReportStore,
    S3ReportStore,
    StoredReport,
    Trend,
    TrendPoint,
    report_store,
    trend,
)


__all__ = [
    "ArgumentCorrectness",
    "ArgumentMismatch",
    "CalibrationDrift",
    "Cap",
    "CaseDiff",
    "ClaimCheck",
    "Comparison",
    "Consistency",
    "Conversation",
    "ConversationTurn",
    "Dimension",
    "EvalCase",
    "EvalGate",
    "EvalReport",
    "EvalResult",
    "EvalRunner",
    "EvalSample",
    "EvaluatorProvenance",
    "EvaluatorVerdict",
    "EvidenceLedger",
    "EvidenceRow",
    "FileReportStore",
    "GateResult",
    "GraphEvalTarget",
    "JudgeProvenance",
    "JudgeUnavailableError",
    "Kappa",
    "LLMJudge",
    "OCIBucketReportStore",
    "OnlineBudget",
    "OnlineEvaluator",
    "OracleReportStore",
    "PanelAgreement",
    "ReceiptCheck",
    "RecordedRun",
    "ReportDiff",
    "ReportStore",
    "Rubric",
    "S3ReportStore",
    "SimulatedUser",
    "SimulatorUnavailableError",
    "StoredReport",
    "TagSummary",
    "ToolCallExpectation",
    "ToolSelection",
    "TrajectoryMatch",
    "Trend",
    "TrendPoint",
    "TrustDecision",
    "TrustGates",
    "UserSimulator",
    "Verdict",
    "as_eval_target",
    "bootstrap_ci",
    "calibration_drift",
    "check_trajectory",
    "cohen_kappa",
    "consistency",
    "diff_reports",
    "dump_cases",
    "every_nth",
    "fraction",
    "load_cases",
    "model_id_of",
    "panel_agreement",
    "report_store",
    "settle_claims",
    "settle_receipts",
    "suite_fingerprint",
    "trend",
    "trust_decision",
    "wilson_interval",
]
