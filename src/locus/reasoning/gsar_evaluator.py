# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""GSAR Algorithm-1 outer loop, packaged as ``GSAREvaluator``.

The evaluator wires together a :class:`BaseGSARJudge`, the scoring
layer from :mod:`locus.reasoning.gsar`, the three-tier decision
function, and a bounded replan budget ``K_max``. It hands the caller
back a :class:`GSARResult` carrying the final report, score, replan
count, and whether the loop exhausted its budget without proceeding
(the *degraded* flag from §5.3).

The evaluator is deliberately deployment-shaped: callers supply the
two side-effecting hooks the paper calls
``REGENERATE_SUMMARY(R, ε)`` and ``REVISE_PLAN_AND_DISPATCH(P, R, ε)``
as plain async callables. That keeps the framework reusable across
ReAct, orchestrator+specialist, or any custom multi-agent topology
without locus.reasoning forcing a specific Agent shape.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from locus.reasoning.gsar import (
    DEFAULT_CONTRADICTION_PENALTY,
    DEFAULT_K_MAX,
    DEFAULT_UNKNOWN_TYPE_WEIGHT,
    DEFAULT_WEIGHT_MAP,
    Decision,
    EvidenceType,
    GSARThresholds,
    Partition,
    decide,
    gsar_score,
)
from locus.reasoning.gsar_judge import JudgeOutput


# ---------------------------------------------------------------------------
# Side-effect callable shapes
# ---------------------------------------------------------------------------

# Returns the regenerated report-synthesis string.
RegenerateFn = Callable[[str, JudgeOutput], Awaitable[str]]

# Returns (revised report-synthesis, refreshed evidence corpus).
ReplanFn = Callable[[str, str, JudgeOutput], Awaitable[tuple[str, str]]]


class GSAREvaluation(BaseModel):
    """One iteration of the outer loop, frozen for the trajectory log."""

    iteration: int = Field(ge=0)
    score: float = Field(ge=0.0, le=1.0)
    decision: Decision
    judge_output: JudgeOutput
    report_synthesis: str
    evidence_corpus: str

    model_config = {"frozen": True}


class GSARResult(BaseModel):
    """Final result of running the Algorithm-1 outer loop."""

    final_report: str = Field(
        description="The synthesised report at loop exit (proceed or degraded)."
    )
    final_score: float = Field(ge=0.0, le=1.0)
    final_decision: Decision
    final_partition: Partition
    final_judge_output: JudgeOutput
    replans_used: int = Field(
        ge=0,
        description="``k`` — how many replan iterations the loop spent.",
    )
    regenerations_used: int = Field(ge=0)
    degraded: bool = Field(
        description=(
            "True when the loop exhausted ``K_max`` without reaching "
            "δ = proceed. The §5.3 'degraded-but-honest' contract: "
            "the report is returned with this flag rather than looped "
            "indefinitely or hallucinated grounded."
        )
    )
    trajectory: list[GSAREvaluation] = Field(
        default_factory=list,
        description=(
            "Every (iteration, score, decision, judge_output) tuple in "
            "loop order — matches the paper's recommended telemetry "
            "(§7 'Telemetry')."
        ),
    )

    model_config = {"frozen": True}


class GSAREvaluator(BaseModel):
    """The Algorithm-1 outer loop, plus inputs.

    Args:
        judge: Any :class:`BaseGSARJudge` implementation.
        regenerate_fn: Async ``(synthesis, judge_output) -> new_synthesis``.
            Called when ``δ = regenerate``. The paper's middle-tier
            recovery: rewrites the synthesis without touching the
            evidence corpus.
        replan_fn: Async ``(synthesis, evidence, judge_output) ->
            (new_synthesis, new_evidence)``. Called when ``δ = replan``
            *or* the judge abstained. The expensive branch: revise
            the plan, re-dispatch specialists, regenerate the corpus.
        weight_map: Override the Appendix-B reference weights.
        thresholds: Override the Appendix-B reference thresholds.
        contradiction_penalty: ``ρ`` from Eq. (2). Default 0.5.
        default_unknown_weight: Weight for evidence types missing from
            ``weight_map``.
        k_max: ``K_max`` replan budget. Default 2.
    """

    judge: Any = Field(description="A BaseGSARJudge instance.")
    regenerate_fn: Any = Field(description="Async (synthesis, judge_output) -> str.")
    replan_fn: Any = Field(
        description="Async (synthesis, evidence, judge_output) -> (str, str).",
    )

    weight_map: dict[EvidenceType, float] = Field(default_factory=lambda: dict(DEFAULT_WEIGHT_MAP))
    thresholds: GSARThresholds = Field(default_factory=GSARThresholds)
    contradiction_penalty: float = Field(default=DEFAULT_CONTRADICTION_PENALTY, ge=0.0, le=1.0)
    default_unknown_weight: float = Field(default=DEFAULT_UNKNOWN_TYPE_WEIGHT, ge=0.0, le=1.0)
    k_max: int = Field(default=DEFAULT_K_MAX, ge=0)

    model_config = {"arbitrary_types_allowed": True}

    async def evaluate(
        self,
        *,
        report_synthesis: str,
        evidence_corpus: str,
    ) -> GSARResult:
        """Run the Algorithm-1 outer loop to convergence or budget.

        Args:
            report_synthesis: The candidate report ``θ`` (the
                orchestrator's first-pass synthesis).
            evidence_corpus: The raw evidence ``E`` collected by the
                specialists.

        Returns:
            :class:`GSARResult` with the final state and full
            trajectory.
        """
        regenerate_fn: RegenerateFn = self.regenerate_fn
        replan_fn: ReplanFn = self.replan_fn

        synthesis = report_synthesis
        evidence = evidence_corpus
        replans = 0
        regenerations = 0
        degraded = False
        trajectory: list[GSAREvaluation] = []

        # --- iteration 0: judge the initial report -----------------------
        judge_output = await self.judge.judge(report_synthesis=synthesis, evidence_corpus=evidence)
        partition = judge_output.to_partition()
        score = gsar_score(
            partition,
            weight_map=self.weight_map,
            contradiction_penalty=self.contradiction_penalty,
            default_unknown=self.default_unknown_weight,
        )
        decision = (
            Decision.ABSTAIN
            if judge_output.abstained
            else decide(score, thresholds=self.thresholds)
        )
        trajectory.append(
            GSAREvaluation(
                iteration=0,
                score=score,
                decision=decision,
                judge_output=judge_output,
                report_synthesis=synthesis,
                evidence_corpus=evidence,
            )
        )

        # --- outer loop --------------------------------------------------
        # Per Algorithm 1 / §5.3: loop while δ ≠ proceed, treating
        # abstain identically to replan.
        while decision != Decision.PROCEED:
            if decision in (Decision.REPLAN, Decision.ABSTAIN):
                if replans >= self.k_max:
                    degraded = True
                    break
                synthesis, evidence = await replan_fn(synthesis, evidence, judge_output)
                replans += 1
            else:
                # δ = regenerate. Bounded separately (one attempt per
                # score evaluation in the reference impl); we let the
                # caller decide how many regenerate→regenerate hops to
                # tolerate by simply continuing the loop, but cap the
                # total iterations so a degenerate model that always
                # produces the same synthesis can't spin forever.
                synthesis = await regenerate_fn(synthesis, judge_output)
                regenerations += 1

            judge_output = await self.judge.judge(
                report_synthesis=synthesis, evidence_corpus=evidence
            )
            partition = judge_output.to_partition()
            score = gsar_score(
                partition,
                weight_map=self.weight_map,
                contradiction_penalty=self.contradiction_penalty,
                default_unknown=self.default_unknown_weight,
            )
            decision = (
                Decision.ABSTAIN
                if judge_output.abstained
                else decide(score, thresholds=self.thresholds)
            )
            trajectory.append(
                GSAREvaluation(
                    iteration=len(trajectory),
                    score=score,
                    decision=decision,
                    judge_output=judge_output,
                    report_synthesis=synthesis,
                    evidence_corpus=evidence,
                )
            )

            # Defensive cap on total iterations — guards against a
            # regenerate loop that never escalates to replan.
            if len(trajectory) > self.k_max + 8:
                degraded = True
                break

        return GSARResult(
            final_report=synthesis,
            final_score=score,
            final_decision=decision,
            final_partition=partition,
            final_judge_output=judge_output,
            replans_used=replans,
            regenerations_used=regenerations,
            degraded=degraded,
            trajectory=trajectory,
        )


__all__ = [
    "GSAREvaluation",
    "GSAREvaluator",
    "GSARResult",
    "RegenerateFn",
    "ReplanFn",
]
