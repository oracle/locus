# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``locus.evaluation.reliability``: how far one verdict can be trusted.

Pure arithmetic over verdicts and grade pairs. Each test pins a reading
that a single verdict cannot give and a way the naive version misleads: a
coin-toss disagreement flagged as instability, raw agreement passing for
kappa, zero contradictions in twenty claims read as certainty, a judge
grading its own model trusted on the strength of one opinion.
"""

from __future__ import annotations

import pytest

from locus.evaluation import JudgeProvenance, Verdict
from locus.evaluation.reliability import (
    HoldReason,
    Kappa,
    ReliabilityGates,
    TrustGates,
    bootstrap_ci,
    calibration_drift,
    cohen_kappa,
    consistency,
    panel_agreement,
    publication_gates,
    trust_decision,
    wilson_interval,
)


def _v(
    score: float,
    grade: str | None = None,
    *,
    passed: bool | None = None,
    dims: dict[str, int] | None = None,
    self_graded: bool | None = None,
    abstained: bool = False,
) -> Verdict:
    return Verdict(
        passed=(score >= 0.6) if passed is None else passed,
        score=score,
        grade=grade,
        dimensions=dims or {},
        abstained=abstained,
        unparseable=abstained,
        provenance=JudgeProvenance(judge_model="j", self_graded=self_graded),
    )


# --------------------------------------------------------------------------
# consistency
# --------------------------------------------------------------------------


def test_repeated_readings_that_agree_are_stable() -> None:
    result = consistency([_v(0.82, "B"), _v(0.80, "B"), _v(0.84, "B")])
    assert result.samples == 3
    assert result.pass_rate == 1.0
    assert result.grade_distribution == {"B": 3}
    assert result.grade_agreement == 1.0
    assert result.score_stddev == pytest.approx(0.0163, abs=1e-3)
    assert not result.flagged_unstable


def test_a_letter_of_spread_flags_the_judge_not_the_agent() -> None:
    """F·41 / C·69 / B·76 on identical evidence: the measurement moved, not the thing."""
    result = consistency([_v(0.41, "F"), _v(0.69, "C"), _v(0.76, "B")])
    assert result.flagged_unstable
    assert result.grade_agreement == pytest.approx(1 / 3)
    assert result.pass_rate == pytest.approx(2 / 3)


def test_two_readings_disagreeing_is_a_coin_toss_not_a_finding() -> None:
    result = consistency([_v(0.41, "F"), _v(0.76, "B")])
    assert not result.flagged_unstable
    assert result.samples == 2


def test_abstained_readings_are_counted_apart() -> None:
    result = consistency([_v(0.8, "B"), _v(0.0, abstained=True), _v(0.8, "B")])
    assert result.samples == 2
    assert result.not_graded == 1
    assert result.pass_rate == 1.0


def test_dimension_spread_is_reported_per_key() -> None:
    result = consistency(
        [_v(0.8, "B", dims={"a": 80, "b": 50}), _v(0.8, "B", dims={"a": 80, "b": 90})]
    )
    assert result.dimension_stddev == {"a": 0.0, "b": 20.0}


def test_no_readings_is_an_empty_report() -> None:
    result = consistency([])
    assert result.samples == 0
    assert result.pass_rate is None
    assert result.score_mean is None
    assert not result.flagged_unstable


# --------------------------------------------------------------------------
# panel
# --------------------------------------------------------------------------


def test_a_panel_that_agrees_has_a_consensus_letter() -> None:
    panel = panel_agreement(
        [
            _v(0.82, "B", dims={"a": 80, "b": 84}),
            _v(0.80, "B", dims={"a": 78, "b": 82}),
            _v(0.84, "B", dims={"a": 82, "b": 86}),
        ]
    )
    assert panel.judges == 3
    assert panel.consensus_grade == "B"
    assert panel.grade_consensus_rate == 1.0
    assert panel.score_pearson == pytest.approx(1.0)
    assert panel.flagged_low_confidence is False


def test_a_split_panel_is_low_confidence() -> None:
    panel = panel_agreement([_v(0.41, "F"), _v(0.69, "C"), _v(0.76, "B")])
    assert panel.consensus_grade in {"F", "C", "B"}
    assert panel.grade_consensus_rate == pytest.approx(1 / 3)
    assert panel.flagged_low_confidence is True


def test_one_judge_is_not_a_panel() -> None:
    panel = panel_agreement([_v(0.8, "B")])
    assert panel.judges == 1
    assert panel.grade_consensus_rate is None
    assert panel.flagged_low_confidence is None
    assert panel.score_pearson is None


def test_constant_dimension_vectors_have_no_correlation() -> None:
    panel = panel_agreement(
        [_v(0.8, "B", dims={"a": 80, "b": 80}), _v(0.8, "B", dims={"a": 80, "b": 80})]
    )
    assert panel.score_pearson is None


# --------------------------------------------------------------------------
# trust
# --------------------------------------------------------------------------


def test_matching_letters_from_two_judges_are_trusted() -> None:
    decision = trust_decision(_v(0.82, "B"), _v(0.78, "B"))
    assert decision.decision == "trusted"
    assert decision.letter_distance == 0


def test_a_letter_apart_needs_calibration_and_two_apart_is_rejected() -> None:
    assert trust_decision(_v(0.82, "B"), _v(0.69, "C")).decision == "calibration_required"
    rejected = trust_decision(_v(0.82, "B"), _v(0.41, "F"))
    assert rejected.decision == "rejected"
    assert rejected.letter_distance == 3


def test_an_unchecked_primary_is_never_trusted() -> None:
    decision = trust_decision(_v(0.95, "A"), None)
    assert decision.decision == "calibration_required"
    assert decision.reason == "no independent opinion"


def test_a_self_graded_verdict_needs_calibration_whatever_the_second_opinion_says() -> None:
    decision = trust_decision(_v(0.82, "B", self_graded=True), _v(0.82, "B"))
    assert decision.decision == "calibration_required"
    assert decision.reason == "self-graded"


def test_an_abstained_primary_is_not_a_grade_to_trust() -> None:
    decision = trust_decision(_v(0.0, abstained=True), _v(0.8, "B"))
    assert decision.reason == "primary not graded"


def test_without_letters_the_pass_decision_and_score_distance_decide() -> None:
    assert trust_decision(_v(0.8), _v(0.75)).decision == "trusted"
    assert trust_decision(_v(0.8), _v(0.3)).decision == "rejected"
    widened = trust_decision(_v(0.95), _v(0.65), gates=TrustGates(score_delta_max=0.1))
    assert widened.decision == "calibration_required"
    assert widened.score_distance == pytest.approx(0.30)


def test_gates_are_adjustable() -> None:
    lenient = TrustGates(calibration_letter_delta=2, rejection_letter_delta=3)
    assert trust_decision(_v(0.82, "B"), _v(0.69, "C"), gates=lenient).decision == "trusted"


# --------------------------------------------------------------------------
# kappa and calibration
# --------------------------------------------------------------------------


def test_kappa_subtracts_chance_agreement() -> None:
    # 80% raw agreement, but both judges hand out B four times in five.
    pairs = (
        [("B", "B")] * 24
        + [("A", "A")] * 2
        + [("B", "A")] * 3
        + [("A", "B")] * 3
        + [("C", "C")] * 0
    )
    pairs += [("B", "B")] * 8  # 40 pairs, 34 agree = 85% raw
    result = cohen_kappa(pairs)
    assert result.pairs == 40
    assert result.kappa is not None
    assert result.kappa < 0.85
    assert result.kappa > 0.0


def test_kappa_is_undefined_on_a_small_sample() -> None:
    result = cohen_kappa([("B", "B")] * 10)
    assert result.kappa is None
    assert result.undefined_reason == "insufficient_sample"


def test_kappa_is_undefined_when_nobody_ever_varied() -> None:
    result = cohen_kappa([("B", "B")] * 40)
    assert result.kappa is None
    assert result.undefined_reason == "no_variance"


def test_perfect_varied_agreement_is_kappa_one() -> None:
    pairs = [("A", "A"), ("B", "B"), ("C", "C"), ("D", "D"), ("F", "F")] * 8
    assert cohen_kappa(pairs).kappa == 1.0


def test_calibration_drift_names_the_judges_lean() -> None:
    kinder = calibration_drift([("A", "B"), ("B", "C"), ("A", "A"), ("B", "D")])
    assert kinder.sample_size == 4
    assert kinder.agreement_rate == 0.25
    assert kinder.direction == "optimistic"
    assert kinder.mean_letter_drift == 1.0

    harsher = calibration_drift([("C", "B"), ("D", "B")])
    assert harsher.direction == "pessimistic"

    even = calibration_drift([("B", "B"), ("C", "B"), ("B", "C")])
    assert even.direction == "balanced"


def test_calibration_drift_drops_letters_it_does_not_know() -> None:
    result = calibration_drift([("B", "B"), ("?", "B"), ("A", "")])
    assert result.sample_size == 1
    assert calibration_drift([]).direction == "unknown"


# --------------------------------------------------------------------------
# intervals
# --------------------------------------------------------------------------


def test_wilson_does_not_read_zero_of_twenty_as_certainty() -> None:
    low, high = wilson_interval(0, 20)
    assert low == 0.0
    assert 0.15 < high < 0.17


def test_wilson_on_no_trials_is_no_information() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError, match="between 0 and trials"):
        wilson_interval(3, 2)


def test_bootstrap_is_reproducible_and_brackets_the_mean() -> None:
    values = [0.6, 0.7, 0.8, 0.9, 0.75, 0.65, 0.85]
    first = bootstrap_ci(values, seed=7, resamples=500)
    second = bootstrap_ci(values, seed=7, resamples=500)
    assert first == second
    assert first[0] <= sum(values) / len(values) <= first[1]
    with pytest.raises(ValueError, match="at least one value"):
        bootstrap_ci([], seed=1)
    with pytest.raises(ValueError, match="positive"):
        bootstrap_ci(values, seed=1, resamples=0)


# --------------------------------------------------------------------------
# publication gates
# --------------------------------------------------------------------------


def test_every_failing_gate_is_reported() -> None:
    reasons = publication_gates(
        ReliabilityGates(),
        samples=5,
        ci_low=0.6,
        ci_high=0.9,
        agreement=Kappa(kappa=0.2, pairs=40),
        contradiction_rate=0.5,
    )
    assert reasons == (
        HoldReason.LOW_SAMPLE,
        HoldReason.WIDE_CI,
        HoldReason.LOW_AGREEMENT,
        HoldReason.HIGH_CONTRADICTION,
    )


def test_a_cell_that_clears_every_gate_has_no_hold() -> None:
    reasons = publication_gates(
        ReliabilityGates(),
        samples=40,
        ci_low=0.78,
        ci_high=0.86,
        agreement=Kappa(kappa=0.7, pairs=40),
        contradiction_rate=0.05,
    )
    assert reasons == ()


def test_an_unavailable_kappa_holds_the_cell() -> None:
    assert HoldReason.LOW_AGREEMENT in publication_gates(
        ReliabilityGates(), samples=40, ci_low=0.8, ci_high=0.85, agreement=None
    )
    assert HoldReason.NO_GRADE_VARIANCE in publication_gates(
        ReliabilityGates(),
        samples=40,
        ci_low=0.8,
        ci_high=0.85,
        agreement=Kappa(pairs=40, undefined_reason="no_variance"),
    )


def test_a_width_exactly_on_the_gate_passes() -> None:
    reasons = publication_gates(
        ReliabilityGates(ci_max_width=0.1),
        samples=40,
        ci_low=0.774,
        ci_high=0.874,
        agreement=Kappa(kappa=0.9, pairs=40),
    )
    assert HoldReason.WIDE_CI not in reasons
