# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for the GSAR Algorithm-1 outer loop.

Covers:

- Direct ``proceed`` from the judge skips the side-effect callables
  and returns a single-iteration trajectory.
- ``regenerate`` calls ``regenerate_fn`` exactly once per cycle, leaves
  the evidence corpus untouched, and re-judges.
- ``replan`` calls ``replan_fn``, replaces the evidence corpus, and
  decrements the budget.
- ``abstain`` is treated identically to ``replan``.
- Budget exhaustion sets ``degraded=True`` after exactly ``K_max``
  replans without ever reaching proceed.
- Trajectory log captures every iteration in order with the right
  decision labels.
"""

from __future__ import annotations

from typing import Any

import pytest

from locus.reasoning.gsar import (
    Claim,
    Decision,
    EvidenceType,
)
from locus.reasoning.gsar_evaluator import GSAREvaluator
from locus.reasoning.gsar_judge import JudgeOutput, safe_default_judge_output


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _ScriptedJudge:
    """Returns a queue of pre-baked JudgeOutputs."""

    def __init__(self, outputs: list[JudgeOutput]) -> None:
        self._queue = list(outputs)
        self.calls: list[tuple[str, str]] = []

    async def judge(
        self,
        *,
        report_synthesis: str,
        evidence_corpus: str,
        **_: Any,
    ) -> JudgeOutput:
        self.calls.append((report_synthesis, evidence_corpus))
        if not self._queue:
            raise AssertionError("scripted judge exhausted")
        if len(self._queue) == 1:
            return self._queue[0]
        return self._queue.pop(0)


def _grounded_payload(grounded: int = 2) -> JudgeOutput:
    """Build a payload that scores 1.0 (all grounded, no U/X/K)."""
    return JudgeOutput(
        grounding_score=1.0,
        is_grounded=True,
        grounded_claims=[
            Claim(text=f"g{i}", type=EvidenceType.TOOL_MATCH) for i in range(grounded)
        ],
    )


def _regenerate_band_payload() -> JudgeOutput:
    """Builds the Appendix-E partition (S ≈ 0.757, decision=regenerate)."""
    return JudgeOutput(
        grounding_score=0.0,  # ignored; we recompute from partition
        is_grounded=False,
        grounded_claims=[
            Claim(text="c1", type=EvidenceType.TOOL_MATCH),
            Claim(text="c2", type=EvidenceType.SPECIFIC_DATA),
        ],
        ungrounded_claims=[Claim(text="c3", type=EvidenceType.INFERENCE)],
        complementary_claims=[Claim(text="c4", type=EvidenceType.COMPLEMENTARY_FINDING)],
        contradicted_claims=[Claim(text="c5", type=EvidenceType.INFERENCE)],
    )


def _replan_band_payload() -> JudgeOutput:
    """All-ungrounded + contradicted → S well below 0.65 (replan)."""
    return JudgeOutput(
        grounding_score=0.0,
        is_grounded=False,
        ungrounded_claims=[
            Claim(text="u1", type=EvidenceType.INFERENCE),
            Claim(text="u2", type=EvidenceType.INFERENCE),
        ],
        contradicted_claims=[
            Claim(text="x1", type=EvidenceType.SPECIFIC_DATA),
        ],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProceedFastPath:
    @pytest.mark.asyncio
    async def test_direct_proceed_no_side_effects(self) -> None:
        regen_calls: list[Any] = []
        replan_calls: list[Any] = []

        async def regen(syn: str, jo: JudgeOutput) -> str:
            regen_calls.append((syn, jo))
            return syn

        async def replan(syn: str, ev: str, jo: JudgeOutput) -> tuple[str, str]:
            replan_calls.append((syn, ev, jo))
            return syn, ev

        evaluator = GSAREvaluator(
            judge=_ScriptedJudge([_grounded_payload()]),
            regenerate_fn=regen,
            replan_fn=replan,
        )
        result = await evaluator.evaluate(
            report_synthesis="initial report",
            evidence_corpus="evidence",
        )
        assert result.final_decision == Decision.PROCEED
        assert result.replans_used == 0
        assert result.regenerations_used == 0
        assert not result.degraded
        assert len(result.trajectory) == 1
        assert regen_calls == []
        assert replan_calls == []


class TestRegenerateBranch:
    @pytest.mark.asyncio
    async def test_regenerate_then_proceed(self) -> None:
        regen_calls = 0
        replan_calls = 0

        async def regen(syn: str, jo: JudgeOutput) -> str:
            nonlocal regen_calls
            regen_calls += 1
            return f"{syn}+regenerated"

        async def replan(syn: str, ev: str, jo: JudgeOutput) -> tuple[str, str]:
            nonlocal replan_calls
            replan_calls += 1
            return syn, ev

        evaluator = GSAREvaluator(
            judge=_ScriptedJudge([_regenerate_band_payload(), _grounded_payload()]),
            regenerate_fn=regen,
            replan_fn=replan,
        )
        result = await evaluator.evaluate(
            report_synthesis="initial",
            evidence_corpus="evidence",
        )
        assert result.final_decision == Decision.PROCEED
        assert result.regenerations_used == 1
        assert result.replans_used == 0
        assert regen_calls == 1
        assert replan_calls == 0
        assert "+regenerated" in result.final_report
        assert len(result.trajectory) == 2
        assert result.trajectory[0].decision == Decision.REGENERATE
        assert result.trajectory[1].decision == Decision.PROCEED


class TestReplanBranch:
    @pytest.mark.asyncio
    async def test_replan_then_proceed(self) -> None:
        evidence_versions: list[str] = []

        async def regen(syn: str, jo: JudgeOutput) -> str:  # pragma: no cover
            return syn

        async def replan(syn: str, ev: str, jo: JudgeOutput) -> tuple[str, str]:
            new_ev = f"{ev}+more"
            evidence_versions.append(new_ev)
            return f"{syn}+replanned", new_ev

        evaluator = GSAREvaluator(
            judge=_ScriptedJudge([_replan_band_payload(), _grounded_payload()]),
            regenerate_fn=regen,
            replan_fn=replan,
            k_max=2,
        )
        result = await evaluator.evaluate(
            report_synthesis="initial",
            evidence_corpus="evidence",
        )
        assert result.final_decision == Decision.PROCEED
        assert result.replans_used == 1
        assert result.regenerations_used == 0
        assert evidence_versions == ["evidence+more"]
        assert "replanned" in result.final_report


class TestAbstainTreatedAsReplan:
    @pytest.mark.asyncio
    async def test_abstain_dispatches_to_replan_fn(self) -> None:
        called = 0

        async def regen(syn: str, jo: JudgeOutput) -> str:  # pragma: no cover
            return syn

        async def replan(syn: str, ev: str, jo: JudgeOutput) -> tuple[str, str]:
            nonlocal called
            called += 1
            return f"{syn}+plan", f"{ev}+evid"

        evaluator = GSAREvaluator(
            judge=_ScriptedJudge(
                [
                    safe_default_judge_output("under-evidenced"),
                    _grounded_payload(),
                ]
            ),
            regenerate_fn=regen,
            replan_fn=replan,
            k_max=2,
        )
        result = await evaluator.evaluate(report_synthesis="r", evidence_corpus="e")
        assert called == 1
        assert result.final_decision == Decision.PROCEED
        assert result.trajectory[0].decision == Decision.ABSTAIN


class TestBudgetExhaustion:
    @pytest.mark.asyncio
    async def test_kmax_2_replans_then_degraded(self) -> None:
        replan_count = 0

        async def regen(syn: str, jo: JudgeOutput) -> str:  # pragma: no cover
            return syn

        async def replan(syn: str, ev: str, jo: JudgeOutput) -> tuple[str, str]:
            nonlocal replan_count
            replan_count += 1
            return syn, ev

        # Every judge call returns a replan-band payload — loop never escapes.
        bad = _replan_band_payload()
        evaluator = GSAREvaluator(
            judge=_ScriptedJudge([bad, bad, bad, bad, bad]),
            regenerate_fn=regen,
            replan_fn=replan,
            k_max=2,
        )
        result = await evaluator.evaluate(report_synthesis="r", evidence_corpus="e")

        assert result.degraded is True
        assert result.replans_used == 2
        # iteration 0 + two post-replan judges = 3 trajectory entries
        # (the budget-exhaustion break happens before the 4th judge call).
        assert len(result.trajectory) == 3
        assert all(t.decision == Decision.REPLAN for t in result.trajectory)


class TestTrajectoryFidelity:
    @pytest.mark.asyncio
    async def test_trajectory_records_every_iteration(self) -> None:
        async def regen(syn: str, jo: JudgeOutput) -> str:
            return f"{syn}+r"

        async def replan(syn: str, ev: str, jo: JudgeOutput) -> tuple[str, str]:
            return f"{syn}+p", f"{ev}+p"

        evaluator = GSAREvaluator(
            judge=_ScriptedJudge(
                [
                    _replan_band_payload(),  # iter 0: REPLAN
                    _regenerate_band_payload(),  # iter 1: REGENERATE
                    _grounded_payload(),  # iter 2: PROCEED
                ]
            ),
            regenerate_fn=regen,
            replan_fn=replan,
            k_max=3,
        )
        result = await evaluator.evaluate(report_synthesis="r", evidence_corpus="e")

        assert [t.decision for t in result.trajectory] == [
            Decision.REPLAN,
            Decision.REGENERATE,
            Decision.PROCEED,
        ]
        assert [t.iteration for t in result.trajectory] == [0, 1, 2]
        assert result.replans_used == 1
        assert result.regenerations_used == 1
        assert not result.degraded


class TestThresholdAndPenaltyOverrides:
    @pytest.mark.asyncio
    async def test_strict_thresholds_force_replan_on_appendix_e(self) -> None:
        # With proceed=0.9, the Appendix-E partition (~0.757) lands in
        # neither proceed nor regenerate — it falls into replan.
        from locus.reasoning.gsar import GSARThresholds

        async def regen(syn: str, jo: JudgeOutput) -> str:  # pragma: no cover
            return syn

        async def replan(syn: str, ev: str, jo: JudgeOutput) -> tuple[str, str]:
            # Return a grounded payload on the next judge call so we
            # exit cleanly.
            return syn, ev

        evaluator = GSAREvaluator(
            judge=_ScriptedJudge([_regenerate_band_payload(), _grounded_payload()]),
            regenerate_fn=regen,
            replan_fn=replan,
            thresholds=GSARThresholds(proceed=0.9, regenerate=0.85),
            k_max=2,
        )
        result = await evaluator.evaluate(report_synthesis="r", evidence_corpus="e")
        # First iteration should have been REPLAN (not REGENERATE) due
        # to the tighter thresholds.
        assert result.trajectory[0].decision == Decision.REPLAN
        assert result.replans_used == 1
        assert result.final_decision == Decision.PROCEED
