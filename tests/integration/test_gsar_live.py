# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Live integration tests for the GSAR layer.

These hit a real LLM judge and exercise the full Algorithm-1 outer
loop end-to-end:

- A clearly-grounded report should land at ``δ = proceed`` on the
  first iteration. Asserts the judge correctly recognises tool-typed
  evidence and the score function clears ``τ_proceed = 0.80``.
- A clearly-ungrounded / contradicted report should not land at
  ``proceed`` on the first iteration; the loop should escalate to
  ``replan`` (or, with a generous threshold, ``regenerate``).
- The trajectory log must be monotonically non-decreasing in
  ``iteration`` and present the right number of entries.

Activation: ``OPENAI_API_KEY`` (uses ``gpt-4o-mini`` as the judge).
Skipped automatically when the key isn't set.
"""

from __future__ import annotations

import pytest

from tests.integration.conftest import skip_without_openai


@skip_without_openai
@pytest.mark.asyncio
async def test_gsar_grounded_report_proceeds() -> None:
    from locus.models.native.openai import OpenAIModel
    from locus.reasoning.gsar import Decision
    from locus.reasoning.gsar_evaluator import GSAREvaluator
    from locus.reasoning.gsar_judge import JudgeOutput, StructuredOutputGSARJudge

    judge = StructuredOutputGSARJudge(
        model=OpenAIModel(model="gpt-4o-mini", max_tokens=2048),
    )

    report = (
        "CPU utilisation on host db-prod-1 reached 97% at 14:02 UTC. "
        "Request rate dropped to 12 RPS at the same time. "
        "These two observations indicate the spike is real."
    )
    evidence = (
        "[tool=query_metrics row=14:02:01] host=db-prod-1 cpu_pct=97.2\n"
        "[tool=query_metrics row=14:02:01] host=db-prod-1 rps=12.4\n"
        "[signal] alert_id=A-9912 fired_at=14:02:00 metric=cpu_pct severity=high\n"
    )

    async def regen(syn: str, jo: JudgeOutput) -> str:  # pragma: no cover
        raise AssertionError(
            f"unexpected regenerate on grounded report; jo={jo.model_dump_json()[:300]}"
        )

    async def replan(  # pragma: no cover
        syn: str, ev: str, jo: JudgeOutput
    ) -> tuple[str, str]:
        raise AssertionError(
            f"unexpected replan on grounded report; jo={jo.model_dump_json()[:300]}"
        )

    evaluator = GSAREvaluator(judge=judge, regenerate_fn=regen, replan_fn=replan)
    result = await evaluator.evaluate(report_synthesis=report, evidence_corpus=evidence)

    assert result.final_decision == Decision.PROCEED, (
        f"final={result.final_decision}, score={result.final_score:.3f}, "
        f"trajectory={[t.decision for t in result.trajectory]}"
    )
    assert result.final_score >= 0.80
    assert result.replans_used == 0
    assert result.regenerations_used == 0
    assert not result.degraded
    assert len(result.trajectory) == 1


@skip_without_openai
@pytest.mark.asyncio
async def test_gsar_ungrounded_report_does_not_proceed_first_iteration() -> None:
    from locus.models.native.openai import OpenAIModel
    from locus.reasoning.gsar import Decision
    from locus.reasoning.gsar_evaluator import GSAREvaluator
    from locus.reasoning.gsar_judge import JudgeOutput, StructuredOutputGSARJudge

    judge = StructuredOutputGSARJudge(
        model=OpenAIModel(model="gpt-4o-mini", max_tokens=2048),
    )

    # Report makes specific factual claims that the evidence does not
    # support. A well-functioning judge should partition the unsupported
    # claims into ungrounded (or contradicted), driving S below τ_proceed.
    report = (
        "The outage was caused by a failed power supply unit in rack 7B. "
        "The replacement was completed at 03:15 UTC. "
        "Customer-facing latency returned to baseline within 8 minutes."
    )
    evidence = (
        "[tool=query_metrics row=02:50:00] cluster=us-west-2a request_rate=0\n"
        "[signal] alert_id=A-1042 fired_at=02:48:12 metric=availability severity=critical\n"
    )

    # Cap replan to 1 so the test exits deterministically without
    # depending on whether the judge ever recovers.
    seen_decisions: list[Decision] = []

    async def regen(syn: str, jo: JudgeOutput) -> str:
        seen_decisions.append(Decision.REGENERATE)
        # Echo back unchanged — we want the loop to escalate.
        return syn

    async def replan(syn: str, ev: str, jo: JudgeOutput) -> tuple[str, str]:
        seen_decisions.append(Decision.REPLAN)
        return syn, ev

    evaluator = GSAREvaluator(
        judge=judge,
        regenerate_fn=regen,
        replan_fn=replan,
        k_max=1,
    )
    result = await evaluator.evaluate(report_synthesis=report, evidence_corpus=evidence)

    # The first-iteration decision must NOT be proceed for this report.
    assert result.trajectory[0].decision != Decision.PROCEED, (
        f"judge wrongly accepted ungrounded report at first iteration: "
        f"score={result.trajectory[0].score:.3f}"
    )
    # The loop should have spent at least one recovery action.
    assert len(seen_decisions) >= 1
    # Iteration counter must be sequential.
    assert [t.iteration for t in result.trajectory] == list(range(len(result.trajectory)))


@skip_without_openai
@pytest.mark.asyncio
async def test_gsar_judge_emits_partition_with_evidence_types() -> None:
    """Verify the live judge populates the partition with ``EvidenceType``s.

    Targets the most regression-prone part of the §6 contract: the
    judge has to map natural-language claims onto the eight-element
    evidence taxonomy, not just emit a binary verdict.
    """
    from locus.models.native.openai import OpenAIModel
    from locus.reasoning.gsar import EvidenceType
    from locus.reasoning.gsar_judge import StructuredOutputGSARJudge

    judge = StructuredOutputGSARJudge(
        model=OpenAIModel(model="gpt-4o-mini", max_tokens=2048),
    )

    out = await judge.judge(
        report_synthesis=(
            "CPU utilisation on host db-prod-1 reached 97% at 14:02 UTC. "
            "This likely indicates a runaway query."
        ),
        evidence_corpus=("[tool=query_metrics row=14:02:01] host=db-prod-1 cpu_pct=97.2\n"),
    )

    # Judge resolved (didn't abstain).
    assert not out.abstained, f"unexpected abstain: {out.abstain_reason}"
    # At least one claim landed in some bucket.
    partition = out.to_partition()
    assert partition.total_claims >= 1
    # Every emitted claim has a typed EvidenceType.
    for claim in partition.all_claims():
        assert isinstance(claim.type, EvidenceType)
    # The grounded "97%" claim should attract a tool-flavoured type.
    grounded_types = {c.type for c in partition.grounded}
    tool_flavoured = {
        EvidenceType.TOOL_MATCH,
        EvidenceType.SPECIFIC_DATA,
        EvidenceType.SIGNAL_MATCH,
    }
    assert grounded_types & tool_flavoured, (
        f"expected at least one grounded claim with tool-flavoured type, "
        f"got grounded={[(c.text, c.type) for c in partition.grounded]}"
    )


# ---------------------------------------------------------------------------
# End-to-end outer-loop dynamics — the paper's core claims live here.
# ---------------------------------------------------------------------------


@skip_without_openai
@pytest.mark.asyncio
async def test_gsar_recovery_then_proceed_live_cycle() -> None:
    """Loose synthesis with a contradicted claim → recovery → proceed.

    Paper's §5.2 contract on the recovery tiers: a report whose
    grounded core is solid but whose synthesis contains a refuted
    claim gets caught and the loop dispatches *some* recovery action
    (regenerate or replan, depending on how the judge weighs the
    contradiction); after the callback rewrites the synthesis to
    drop the contradiction, the loop converges to proceed.

    Real-world judge variance means we don't pin which tier fires
    on the first iteration; the load-bearing claim is that the loop
    *recovers* and *converges*. Discrete-tier dispatch is covered
    in the unit tests against a scripted judge.
    """
    from locus.models.native.openai import OpenAIModel
    from locus.reasoning.gsar import Decision, GSARThresholds
    from locus.reasoning.gsar_evaluator import GSAREvaluator
    from locus.reasoning.gsar_judge import JudgeOutput, StructuredOutputGSARJudge

    judge = StructuredOutputGSARJudge(
        model=OpenAIModel(model="gpt-4o-mini", max_tokens=2048),
    )

    # Mix grounded + a directly-contradicted claim. The contradicted
    # path is more reliable than the ungrounded path because the judge
    # has the explicit refuting row to point at — "rps held steady at
    # 4500" vs the tool output "rps=12.4". The regenerate sub-agent
    # drops the contradiction; the rewritten synthesis is pure tool_match
    # grounded → S = 1.0 → proceed.
    initial_report = (
        "CPU utilisation on host db-prod-1 reached 97% at 14:02 UTC. "
        "Request rate held steady at 4500 RPS throughout the spike. "
        "An alert fired at 14:02 UTC."
    )
    tightened_report = (
        "CPU utilisation on host db-prod-1 reached 97% at 14:02 UTC. An alert fired at 14:02 UTC."
    )
    evidence = (
        "[tool=query_metrics row=14:02:01] host=db-prod-1 cpu_pct=97.2\n"
        "[tool=query_metrics row=14:02:01] host=db-prod-1 rps=12.4\n"
        "[signal] alert_id=A-9912 fired_at=14:02:00 metric=cpu_pct severity=high\n"
    )

    regen_calls = 0
    replan_calls = 0

    async def regen(syn: str, jo: JudgeOutput) -> str:
        nonlocal regen_calls
        regen_calls += 1
        # In production, the regenerate sub-agent rewrites the synthesis
        # using ε; we simulate that here by returning the tightened
        # version that drops the unsupported claims.
        return tightened_report

    async def replan(syn: str, ev: str, jo: JudgeOutput) -> tuple[str, str]:
        nonlocal replan_calls
        replan_calls += 1
        # Replan should not be reached on this input.
        return tightened_report, ev

    # Default thresholds. The loop has up to 2 attempts to recover
    # before timing out — judges can land first iteration in either
    # regenerate (cheap) or replan (expensive) on contradicted-claim
    # input; both rewrite to a clean synthesis on the next pass.
    evaluator = GSAREvaluator(
        judge=judge,
        regenerate_fn=regen,
        replan_fn=replan,
        thresholds=GSARThresholds(),
        k_max=2,
    )
    result = await evaluator.evaluate(report_synthesis=initial_report, evidence_corpus=evidence)

    assert result.final_decision == Decision.PROCEED, (
        f"final={result.final_decision}, score={result.final_score:.3f}, "
        f"trajectory={[(t.decision, round(t.score, 3)) for t in result.trajectory]}"
    )
    # The test's premise is "first iteration not proceeding → recovery
    # fires → second iteration proceeds". Real-world judge variance
    # means the judge sometimes accepts the contradicted-claim-bearing
    # report on the first pass (false-positive on the contradiction);
    # in that case the loop goes straight to proceed without firing
    # any recovery, and the recovery-then-proceed claim is vacuously
    # true. Skip with a clear message so the run logs why; the unit
    # tests cover the discrete recovery branches deterministically.
    total_recovery_calls = regen_calls + replan_calls
    if total_recovery_calls == 0:
        pytest.skip(
            "live judge accepted the contradicted-claim report on the "
            "first iteration; recovery loop wasn't exercised. trajectory="
            f"{[(t.decision, round(t.score, 3)) for t in result.trajectory]}"
        )
    assert not result.degraded
    # Trajectory monotonicity: the last iteration's score must not be
    # lower than the first — recovery should be a non-regression.
    scores = [t.score for t in result.trajectory]
    assert scores[-1] >= scores[0] - 1e-9, f"score decreased across recovery cycle: {scores}"


@skip_without_openai
@pytest.mark.asyncio
async def test_gsar_replan_then_proceed_live_cycle() -> None:
    """Insufficient evidence → replan → proceed once evidence is added.

    Paper's §5.2 expensive branch: when the synthesis can't be fixed
    by rewriting alone, the orchestrator revises the plan and
    re-dispatches specialists. We simulate the dispatch by appending
    new tool outputs to the evidence corpus.
    """
    from locus.models.native.openai import OpenAIModel
    from locus.reasoning.gsar import Decision
    from locus.reasoning.gsar_evaluator import GSAREvaluator
    from locus.reasoning.gsar_judge import JudgeOutput, StructuredOutputGSARJudge

    judge = StructuredOutputGSARJudge(
        model=OpenAIModel(model="gpt-4o-mini", max_tokens=2048),
    )

    # Specific factual claims with literally no evidence corpus on the
    # first iteration → judge has nothing to ground against, must
    # partition into ungrounded (and may abstain). Either branch
    # dispatches to replan_fn under the §6 abstain == replan rule.
    report = (
        "CPU utilisation on host db-prod-1 reached 97% at 14:02 UTC. "
        "Request rate dropped to 12 RPS at the same time."
    )
    initial_evidence = "(no evidence collected yet)\n"
    fresh_tool_evidence = (
        "[signal] alert_id=A-9912 fired_at=14:02:00 metric=cpu_pct severity=high\n"
        "[tool=query_metrics row=14:02:01] host=db-prod-1 cpu_pct=97.2\n"
        "[tool=query_metrics row=14:02:01] host=db-prod-1 rps=12.4\n"
    )

    regen_calls = 0
    replan_calls = 0

    async def regen(syn: str, jo: JudgeOutput) -> str:
        nonlocal regen_calls
        regen_calls += 1
        # Echo unchanged — we want the loop to be the one that fixes this
        # by gathering fresh evidence on a subsequent replan.
        return syn

    async def replan(syn: str, ev: str, jo: JudgeOutput) -> tuple[str, str]:
        nonlocal replan_calls
        replan_calls += 1
        # Production: revise plan, re-dispatch specialists, get fresh evidence.
        return syn, fresh_tool_evidence

    evaluator = GSAREvaluator(
        judge=judge,
        regenerate_fn=regen,
        replan_fn=replan,
        k_max=2,
    )
    result = await evaluator.evaluate(report_synthesis=report, evidence_corpus=initial_evidence)

    # Eventually the loop should reach proceed once the evidence is fresh.
    # If the judge gets unusually picky we accept up to 2 replans; the test
    # is gated by k_max=2 so the loop terminates in any case.
    assert result.final_decision == Decision.PROCEED, (
        f"final={result.final_decision}, score={result.final_score:.3f}, "
        f"replans={result.replans_used}, regens={result.regenerations_used}, "
        f"trajectory={[(t.decision, round(t.score, 3)) for t in result.trajectory]}"
    )
    # At least one recovery action of *some* kind must have been taken —
    # the judge variance can land first iteration in either replan or
    # regenerate. What matters is that the loop recovered and the
    # evaluator dispatched to one of the two side-effect callbacks.
    assert (replan_calls + regen_calls) >= 1, (
        "no recovery callback was invoked despite first-iteration not proceeding"
    )
    assert not result.degraded


@skip_without_openai
@pytest.mark.asyncio
async def test_gsar_budget_exhaustion_sets_degraded_live() -> None:
    """Unsalvageable input → K_max replans → degraded=True.

    Paper's §5.3 contract: returning a degraded-but-honest report is
    preferable to looping indefinitely or hallucinating grounded.
    Drives the live judge with a no-op replan_fn, which leaves the
    bad evidence in place; the loop should exhaust ``K_max`` and
    return with the flag set.
    """
    from locus.models.native.openai import OpenAIModel
    from locus.reasoning.gsar_evaluator import GSAREvaluator
    from locus.reasoning.gsar_judge import JudgeOutput, StructuredOutputGSARJudge

    judge = StructuredOutputGSARJudge(
        model=OpenAIModel(model="gpt-4o-mini", max_tokens=2048),
    )

    # Specific factual claims, no supporting evidence at all. Judge
    # cannot ground these no matter how many times we re-call it.
    report = (
        "The outage was caused by a failed power supply unit in rack 7B. "
        "The PSU was replaced at 03:15 UTC by technician T-218. "
        "Customer-facing latency returned to baseline within 8 minutes."
    )
    evidence = "(no evidence available)\n"

    async def regen(syn: str, jo: JudgeOutput) -> str:
        return syn

    async def replan(syn: str, ev: str, jo: JudgeOutput) -> tuple[str, str]:
        # Deliberately do NOT add evidence — simulates a failed plan.
        return syn, ev

    evaluator = GSAREvaluator(
        judge=judge,
        regenerate_fn=regen,
        replan_fn=replan,
        k_max=2,
    )
    result = await evaluator.evaluate(report_synthesis=report, evidence_corpus=evidence)

    # The loop must terminate without hanging.
    assert result.degraded is True, (
        f"expected degraded=True after K_max replans without recovery, "
        f"got final={result.final_decision}, replans={result.replans_used}"
    )
    assert result.replans_used == 2
    # Three judge calls total: iteration 0 + 2 post-replan judges.
    assert len(result.trajectory) == 3


@skip_without_openai
@pytest.mark.asyncio
async def test_gsar_rho_zero_inflation_visible_live() -> None:
    """Property P5 in practice: dropping ρ inflates ``S`` on a real
    judge-produced partition.

    The §8.5 ablation. Run the judge once, then score the same
    partition twice — with default ρ=0.5 and with ρ=0. The ρ=0 score
    must be ≥ the ρ=0.5 score. The contradiction-non-suppression
    property is what prevents adversarial summarisers from boosting
    their score by silently dropping refuted claims.
    """
    from locus.models.native.openai import OpenAIModel
    from locus.reasoning.gsar import gsar_score
    from locus.reasoning.gsar_judge import StructuredOutputGSARJudge

    judge = StructuredOutputGSARJudge(
        model=OpenAIModel(model="gpt-4o-mini", max_tokens=2048),
    )

    # Mixed report: most claims grounded, but one factually wrong claim
    # that the judge should partition into contradicted.
    report = (
        "CPU utilisation on host db-prod-1 reached 97% at 14:02 UTC. "
        "Request rate held steady at 4500 RPS throughout the spike. "
        "An alert fired at 14:02 UTC."
    )
    evidence = (
        "[tool=query_metrics row=14:02:01] host=db-prod-1 cpu_pct=97.2\n"
        "[tool=query_metrics row=14:02:01] host=db-prod-1 rps=12.4\n"
        "[signal] alert_id=A-9912 fired_at=14:02:00 metric=cpu_pct severity=high\n"
    )

    out = await judge.judge(report_synthesis=report, evidence_corpus=evidence)
    partition = out.to_partition()
    parts = (
        ("grounded", len(partition.grounded)),
        ("ungrounded", len(partition.ungrounded)),
        ("contradicted", len(partition.contradicted)),
        ("complementary", len(partition.complementary)),
    )

    # Pre-conditions for the strict inflation inequality to be
    # observable (Property P5 from §4.2 / Appendix A):
    #   1. The judge identified at least one contradicted claim
    #      (W(X) > 0 — without it both ρ values yield identical S).
    #   2. The judge identified at least one grounded or complementary
    #      claim (W(G) + W(K) > 0 — without it the numerator is 0
    #      regardless of ρ, so both yield S=0). When the judge over-
    #      contradicts and leaves no positive mass, the math still
    #      holds (s_no_rho ≥ s_default) but inflation is degenerate.
    if not partition.contradicted:
        pytest.skip(
            f"judge produced no contradicted claim — W(X)=0 makes P5 "
            f"unobservable. partition={parts}"
        )
    if not (partition.grounded or partition.complementary):
        pytest.skip(
            f"judge produced no grounded/complementary claims — "
            f"W(G)+W(K)=0 collapses both ρ scores to 0. partition={parts}"
        )

    s_default = gsar_score(partition, contradiction_penalty=0.5)
    s_no_rho = gsar_score(partition, contradiction_penalty=0.0)

    # Weak inequality always holds under P5.
    assert s_no_rho >= s_default - 1e-9, (
        f"ρ=0 produced lower score than ρ=0.5 (violates P5): "
        f"s_default={s_default:.4f}, s_no_rho={s_no_rho:.4f}, partition={parts}"
    )
    # Strict inequality holds when both pre-conditions above are met.
    assert s_no_rho > s_default, (
        f"ρ=0 should strictly inflate when W(X) > 0 and W(G)+W(K) > 0: "
        f"s_default={s_default:.4f}, s_no_rho={s_no_rho:.4f}, partition={parts}"
    )


@skip_without_openai
@pytest.mark.asyncio
async def test_gsar_cross_judge_score_directional_agreement() -> None:
    """Two different OpenAI judges should agree on the *direction* of S
    between a grounded and an ungrounded report.

    Paper §11 / Table 10: the contradiction-penalty effect is
    judge-agnostic. The cheap proxy here: for the same pair of
    (grounded, ungrounded) reports, both judges must score the
    grounded report strictly higher than the ungrounded one. We
    don't pin the exact decision tier — judges legitimately disagree
    on tier under variance — but the score-ordering must be stable.
    """
    from locus.models.native.openai import OpenAIModel
    from locus.reasoning.gsar import gsar_score
    from locus.reasoning.gsar_judge import StructuredOutputGSARJudge

    j_mini = StructuredOutputGSARJudge(
        model=OpenAIModel(model="gpt-4o-mini", max_tokens=2048),
    )
    j_full = StructuredOutputGSARJudge(
        model=OpenAIModel(model="gpt-4o", max_tokens=2048),
    )

    grounded = {
        "report": (
            "CPU utilisation on host db-prod-1 reached 97% at 14:02 UTC. "
            "Request rate dropped to 12 RPS at the same time."
        ),
        "evidence": (
            "[tool=query_metrics row=14:02:01] host=db-prod-1 cpu_pct=97.2\n"
            "[tool=query_metrics row=14:02:01] host=db-prod-1 rps=12.4\n"
        ),
    }
    ungrounded = {
        "report": ("The outage was caused by a failed power supply at 03:15 UTC."),
        "evidence": "[signal] alert_id=A-1042 fired_at=02:48:12 metric=availability\n",
    }

    async def score_for(judge, payload: dict[str, str]) -> float:
        out = await judge.judge(
            report_synthesis=payload["report"],
            evidence_corpus=payload["evidence"],
        )
        if out.abstained:
            # Treat abstain as score 0 for directional comparison —
            # abstain on a grounded report would be a real failure;
            # abstain on the ungrounded report is fine.
            return 0.0
        return gsar_score(out.to_partition())

    s_mini_g = await score_for(j_mini, grounded)
    s_full_g = await score_for(j_full, grounded)
    s_mini_u = await score_for(j_mini, ungrounded)
    s_full_u = await score_for(j_full, ungrounded)

    # The judge-agnostic claim: each judge scores the grounded report
    # strictly higher than the ungrounded report. We don't compare
    # *across* judges — that would conflate model variance with the
    # mechanism we're testing.
    assert s_mini_g > s_mini_u, (
        f"gpt-4o-mini did not order grounded > ungrounded: "
        f"grounded={s_mini_g:.3f}, ungrounded={s_mini_u:.3f}"
    )
    assert s_full_g > s_full_u, (
        f"gpt-4o did not order grounded > ungrounded: "
        f"grounded={s_full_g:.3f}, ungrounded={s_full_u:.3f}"
    )
    # Sanity floor: the grounded report should clear the regenerate
    # threshold (0.65) on at least one of the two judges. If both fall
    # below, the report itself is too ambiguous and the test isn't
    # measuring what it claims to measure.
    assert max(s_mini_g, s_full_g) >= 0.65, (
        f"both judges scored grounded report below τ_regenerate: "
        f"mini={s_mini_g:.3f}, full={s_full_g:.3f}"
    )
