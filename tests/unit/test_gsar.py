# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Intensive unit tests for the GSAR scoring and decision layer.

Covers every claim the paper makes about ``S`` and ``δ``:

- Properties P1-P6 from §4.2 + Appendix A — bounded, monotonic in G,
  monotonic in X (modulated by ρ), complementary value, contradiction
  non-suppression, inference-observation asymmetry.
- Decision boundaries (Eq. 3) — exact-edge behaviour at τ_proceed and
  τ_regenerate, plus degenerate / out-of-range guards.
- Edge cases — empty partition (neutral 0.5), unknown evidence types,
  ρ=0 vs ρ=1 vs ρ=0.5, weight-map calibration overrides.
- Worked numerical example from Appendix E (S ≈ 0.757, δ = regenerate
  under reference thresholds).
- :func:`partition_weight` correctness across every evidence type in
  the default Appendix-B map.
"""

from __future__ import annotations

import math

import pytest

from locus.reasoning.gsar import (
    DEFAULT_CONTRADICTION_PENALTY,
    DEFAULT_TAU_PROCEED,
    DEFAULT_TAU_REGENERATE,
    DEFAULT_WEIGHT_MAP,
    NEUTRAL_SCORE_ON_EMPTY,
    Claim,
    Decision,
    EvidenceType,
    GSARThresholds,
    Partition,
    decide,
    gsar_score,
    partition_weight,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _c(text: str, etype: EvidenceType = EvidenceType.TOOL_MATCH) -> Claim:
    return Claim(text=text, type=etype)


# ---------------------------------------------------------------------------
# Section: defaults match the paper's Appendix B verbatim
# ---------------------------------------------------------------------------


class TestAppendixBDefaults:
    """Lock the reference instantiation against accidental drift."""

    def test_weight_map_values(self) -> None:
        assert DEFAULT_WEIGHT_MAP[EvidenceType.TOOL_MATCH] == 1.00
        assert DEFAULT_WEIGHT_MAP[EvidenceType.SPECIFIC_DATA] == 0.95
        assert DEFAULT_WEIGHT_MAP[EvidenceType.SIGNAL_MATCH] == 0.90
        assert DEFAULT_WEIGHT_MAP[EvidenceType.COMPLEMENTARY_FINDING] == 0.85
        assert DEFAULT_WEIGHT_MAP[EvidenceType.SYNTHESIS] == 0.80
        assert DEFAULT_WEIGHT_MAP[EvidenceType.NEG_EVIDENCE] == 0.70
        assert DEFAULT_WEIGHT_MAP[EvidenceType.INFERENCE] == 0.60
        assert DEFAULT_WEIGHT_MAP[EvidenceType.DOMAIN] == 0.60

    def test_threshold_defaults(self) -> None:
        assert DEFAULT_TAU_PROCEED == 0.80
        assert DEFAULT_TAU_REGENERATE == 0.65

    def test_contradiction_penalty(self) -> None:
        assert DEFAULT_CONTRADICTION_PENALTY == 0.5

    def test_taxonomy_size_matches_paper(self) -> None:
        assert len(EvidenceType) == 8


# ---------------------------------------------------------------------------
# Section: partition_weight (Equation 1)
# ---------------------------------------------------------------------------


class TestPartitionWeight:
    def test_empty_partition_weight_is_zero(self) -> None:
        assert partition_weight([]) == 0.0

    def test_sums_per_claim_weights(self) -> None:
        claims = [_c("a", EvidenceType.TOOL_MATCH), _c("b", EvidenceType.INFERENCE)]
        assert partition_weight(claims) == pytest.approx(1.0 + 0.6)

    def test_all_taxonomy_entries_summable(self) -> None:
        # Every type in the paper's taxonomy must round-trip through the
        # default map without falling to the unknown-type default.
        claims = [_c(t.value, t) for t in EvidenceType]
        expected = sum(DEFAULT_WEIGHT_MAP[t] for t in EvidenceType)
        assert partition_weight(claims) == pytest.approx(expected)

    def test_unknown_type_uses_default_unknown(self) -> None:
        # Build a sparse weight_map missing INFERENCE — verify default fires.
        claims = [_c("x", EvidenceType.INFERENCE)]
        out = partition_weight(
            claims,
            weight_map={EvidenceType.TOOL_MATCH: 1.0},
            default_unknown=0.123,
        )
        assert out == pytest.approx(0.123)


# ---------------------------------------------------------------------------
# Section: gsar_score (Equation 2) — basic shape + edge cases
# ---------------------------------------------------------------------------


class TestGSARScoreBasics:
    def test_empty_partition_returns_neutral(self) -> None:
        assert gsar_score(Partition()) == NEUTRAL_SCORE_ON_EMPTY

    def test_all_grounded_yields_one(self) -> None:
        p = Partition(grounded=[_c("a"), _c("b")])
        assert gsar_score(p) == pytest.approx(1.0)

    def test_all_ungrounded_yields_zero(self) -> None:
        p = Partition(ungrounded=[_c("a"), _c("b")])
        assert gsar_score(p) == pytest.approx(0.0)

    def test_all_contradicted_yields_zero(self) -> None:
        p = Partition(contradicted=[_c("a"), _c("b")])
        assert gsar_score(p) == pytest.approx(0.0)

    def test_only_complementary_yields_one(self) -> None:
        # K is in numerator and denominator with the same weight, so
        # an all-complementary partition scores 1.0 (the floor on
        # P4 — strict gain over no information).
        p = Partition(complementary=[_c("a", EvidenceType.COMPLEMENTARY_FINDING)])
        assert gsar_score(p) == pytest.approx(1.0)

    def test_score_in_unit_interval_for_random_partitions(self) -> None:
        # Property P1 sanity sweep across a Cartesian grid of partition
        # cardinalities and one weight per claim. Stays under 256
        # combinations so it runs in milliseconds.
        for ng in range(4):
            for nu in range(4):
                for nx in range(4):
                    for nk in range(4):
                        p = Partition(
                            grounded=[_c("g", EvidenceType.TOOL_MATCH)] * ng,
                            ungrounded=[_c("u", EvidenceType.INFERENCE)] * nu,
                            contradicted=[_c("x", EvidenceType.SPECIFIC_DATA)] * nx,
                            complementary=[_c("k", EvidenceType.COMPLEMENTARY_FINDING)] * nk,
                        )
                        s = gsar_score(p)
                        assert 0.0 <= s <= 1.0, (ng, nu, nx, nk, s)

    def test_rejects_invalid_rho(self) -> None:
        p = Partition(grounded=[_c("a")])
        with pytest.raises(ValueError):
            gsar_score(p, contradiction_penalty=-0.1)
        with pytest.raises(ValueError):
            gsar_score(p, contradiction_penalty=1.1)

    def test_zero_weight_partition_falls_back_to_neutral(self) -> None:
        # If all weights zero out, the framework treats it like the
        # empty partition rather than dividing by zero.
        p = Partition(grounded=[_c("g", EvidenceType.DOMAIN)])
        out = gsar_score(p, weight_map={EvidenceType.DOMAIN: 0.0})
        assert out == NEUTRAL_SCORE_ON_EMPTY


# ---------------------------------------------------------------------------
# Section: Properties P1-P6 from §4.2 / Appendix A
# ---------------------------------------------------------------------------


class TestPropertyP1Boundedness:
    """P1: S ∈ [0, 1] for every non-empty partition + every ρ ∈ [0, 1]."""

    @pytest.mark.parametrize("rho", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_bounded_across_rho(self, rho: float) -> None:
        # Mixed-type, mixed-bucket partition.
        p = Partition(
            grounded=[
                _c("g1", EvidenceType.TOOL_MATCH),
                _c("g2", EvidenceType.INFERENCE),
            ],
            ungrounded=[_c("u1", EvidenceType.SYNTHESIS)],
            contradicted=[
                _c("x1", EvidenceType.SPECIFIC_DATA),
                _c("x2", EvidenceType.DOMAIN),
            ],
            complementary=[_c("k1", EvidenceType.COMPLEMENTARY_FINDING)],
        )
        s = gsar_score(p, contradiction_penalty=rho)
        assert 0.0 <= s <= 1.0


class TestPropertyP2GroundedMonotonicity:
    """P2: moving c from U to G never decreases S; strictly increases
    unless w(type(c)) = 0."""

    def test_strictly_increases_when_weight_positive(self) -> None:
        c = _c("c", EvidenceType.INFERENCE)
        before = Partition(grounded=[_c("g")], ungrounded=[c])
        after = Partition(grounded=[_c("g"), c], ungrounded=[])
        assert gsar_score(after) > gsar_score(before)

    def test_no_decrease_under_zero_weight(self) -> None:
        c = _c("c", EvidenceType.DOMAIN)
        zw = dict(DEFAULT_WEIGHT_MAP) | {EvidenceType.DOMAIN: 0.0}
        before = Partition(grounded=[_c("g")], ungrounded=[c])
        after = Partition(grounded=[_c("g"), c], ungrounded=[])
        # Both should be 1.0 — only g contributes to numerator/denominator.
        assert gsar_score(before, weight_map=zw) == pytest.approx(1.0)
        assert gsar_score(after, weight_map=zw) == pytest.approx(1.0)


class TestPropertyP3ContradictionMonotonicity:
    """P3: adding to X never increases S; rate of decrease modulated by ρ."""

    def test_adding_contradiction_never_increases(self) -> None:
        base = Partition(grounded=[_c("g1"), _c("g2")])
        with_x = Partition(
            grounded=[_c("g1"), _c("g2")],
            contradicted=[_c("x", EvidenceType.SPECIFIC_DATA)],
        )
        assert gsar_score(with_x) <= gsar_score(base)

    def test_higher_rho_means_steeper_drop(self) -> None:
        p = Partition(
            grounded=[_c("g")],
            contradicted=[_c("x", EvidenceType.TOOL_MATCH)],
        )
        s_lo = gsar_score(p, contradiction_penalty=0.0)
        s_md = gsar_score(p, contradiction_penalty=0.5)
        s_hi = gsar_score(p, contradiction_penalty=1.0)
        # Adding X with ρ=0 leaves S unchanged; with ρ>0 strictly drops it.
        assert s_lo > s_md > s_hi
        assert s_lo == pytest.approx(1.0)


class TestPropertyP4ComplementaryValue:
    """P4: adding c to K with S ≤ 1 yields S' ≥ S; w(K-claim) ≤ w of a
    grounded claim of the same type."""

    def test_complementary_increases_score_when_below_one(self) -> None:
        p_base = Partition(
            grounded=[_c("g", EvidenceType.SYNTHESIS)],
            ungrounded=[_c("u", EvidenceType.INFERENCE)],
        )
        p_with_k = Partition(
            grounded=[_c("g", EvidenceType.SYNTHESIS)],
            ungrounded=[_c("u", EvidenceType.INFERENCE)],
            complementary=[_c("k", EvidenceType.COMPLEMENTARY_FINDING)],
        )
        s_base = gsar_score(p_base)
        s_k = gsar_score(p_with_k)
        assert s_base < 1.0
        assert s_k >= s_base
        assert s_k > s_base  # Strict, because s_base < 1.

    def test_complementary_at_score_one_stays_one(self) -> None:
        p = Partition(grounded=[_c("g")])
        assert gsar_score(p) == pytest.approx(1.0)
        p_k = Partition(
            grounded=[_c("g")],
            complementary=[_c("k", EvidenceType.COMPLEMENTARY_FINDING)],
        )
        assert gsar_score(p_k) == pytest.approx(1.0)


class TestPropertyP5ContradictionNonSuppression:
    """P5: removing X from C strictly inflates S (when W(X) > 0 and ρ > 0).

    This is the core anti-gaming property — keeping contradicted claims
    in the denominator prevents an adversarial summariser from boosting
    the score by silently dropping refuted claims.
    """

    def test_dropping_contradictions_inflates_score(self) -> None:
        with_x = Partition(
            grounded=[_c("g1"), _c("g2")],
            contradicted=[_c("x", EvidenceType.SPECIFIC_DATA)],
        )
        without_x = Partition(grounded=[_c("g1"), _c("g2")])
        # Same numerator, smaller denominator → strictly larger S.
        assert gsar_score(without_x) > gsar_score(with_x)

    def test_no_inflation_when_rho_is_zero(self) -> None:
        # ρ=0 reproduces the §8.5 ablation — contradictions don't pull
        # the denominator, so dropping them is score-neutral.
        with_x = Partition(
            grounded=[_c("g")],
            contradicted=[_c("x", EvidenceType.SPECIFIC_DATA)],
        )
        without_x = Partition(grounded=[_c("g")])
        assert gsar_score(with_x, contradiction_penalty=0.0) == pytest.approx(
            gsar_score(without_x, contradiction_penalty=0.0)
        )


class TestPropertyP6InferenceObservationAsymmetry:
    """P6: replacing a grounded tool_match with a grounded inference
    strictly decreases S (when W(U) + ρ·W(X) > 0)."""

    def test_inference_grounded_scores_lower_than_tool_match_grounded(self) -> None:
        # Mass in U or X is required for the inequality to be strict.
        p_tm = Partition(
            grounded=[_c("g", EvidenceType.TOOL_MATCH)],
            ungrounded=[_c("u", EvidenceType.INFERENCE)],
        )
        p_inf = Partition(
            grounded=[_c("g", EvidenceType.INFERENCE)],
            ungrounded=[_c("u", EvidenceType.INFERENCE)],
        )
        assert gsar_score(p_inf) < gsar_score(p_tm)

    def test_no_asymmetry_without_uncertain_mass(self) -> None:
        # If U = X = ∅, every grounded weighting yields S = 1 regardless
        # of evidence type — the asymmetry is invisible.
        p_tm = Partition(grounded=[_c("g", EvidenceType.TOOL_MATCH)])
        p_inf = Partition(grounded=[_c("g", EvidenceType.INFERENCE)])
        assert gsar_score(p_tm) == pytest.approx(1.0)
        assert gsar_score(p_inf) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Section: worked example from Appendix E (S ≈ 0.757)
# ---------------------------------------------------------------------------


class TestAppendixEWorkedExample:
    """Reproduce the numerical example in Appendix E of the paper."""

    def _example_partition(self) -> Partition:
        # c1 grounded tool_match (1.0)
        # c2 grounded specific_data (0.95)
        # c3 ungrounded inference (0.60)
        # c4 complementary complementary_finding (0.85)
        # c5 contradicted inference (0.60)
        return Partition(
            grounded=[
                _c("c1", EvidenceType.TOOL_MATCH),
                _c("c2", EvidenceType.SPECIFIC_DATA),
            ],
            ungrounded=[_c("c3", EvidenceType.INFERENCE)],
            complementary=[_c("c4", EvidenceType.COMPLEMENTARY_FINDING)],
            contradicted=[_c("c5", EvidenceType.INFERENCE)],
        )

    def test_appendix_e_numbers_match(self) -> None:
        p = self._example_partition()
        # W(G) = 1.0 + 0.95 = 1.95; W(U) = 0.60; W(X) = 0.60; W(K) = 0.85.
        # S = (1.95 + 0.85) / (1.95 + 0.60 + 0.5·0.60 + 0.85)
        #   = 2.80 / 3.70 ≈ 0.7567567...
        s = gsar_score(p, contradiction_penalty=0.5)
        assert s == pytest.approx(2.80 / 3.70, abs=1e-9)
        assert math.isclose(s, 0.7567567567, rel_tol=1e-6)

    def test_appendix_e_decision_is_regenerate(self) -> None:
        # 0.65 ≤ 0.757 < 0.80 → δ = regenerate (Eq. 3 + reference τ).
        s = gsar_score(self._example_partition(), contradiction_penalty=0.5)
        assert decide(s) == Decision.REGENERATE


# ---------------------------------------------------------------------------
# Section: decide() — Equation 3 boundary behaviour
# ---------------------------------------------------------------------------


class TestDecisionBoundaries:
    @pytest.mark.parametrize("score", [0.80, 0.85, 1.0])
    def test_proceed_at_or_above_tau_proceed(self, score: float) -> None:
        assert decide(score) == Decision.PROCEED

    @pytest.mark.parametrize("score", [0.65, 0.66, 0.799, 0.7999999])
    def test_regenerate_band(self, score: float) -> None:
        assert decide(score) == Decision.REGENERATE

    @pytest.mark.parametrize("score", [0.0, 0.1, 0.5, 0.6499999])
    def test_replan_below_regenerate(self, score: float) -> None:
        assert decide(score) == Decision.REPLAN

    def test_custom_thresholds(self) -> None:
        th = GSARThresholds(proceed=0.9, regenerate=0.5)
        assert decide(0.95, thresholds=th) == Decision.PROCEED
        assert decide(0.7, thresholds=th) == Decision.REGENERATE
        assert decide(0.4, thresholds=th) == Decision.REPLAN

    def test_thresholds_must_be_ordered(self) -> None:
        with pytest.raises(ValueError):
            GSARThresholds(proceed=0.5, regenerate=0.6)
        with pytest.raises(ValueError):
            GSARThresholds(proceed=0.5, regenerate=0.5)

    def test_score_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError):
            decide(-0.1)
        with pytest.raises(ValueError):
            decide(1.1)


# ---------------------------------------------------------------------------
# Section: serialization / round-trip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_partition_round_trips(self) -> None:
        p = Partition(
            grounded=[_c("g", EvidenceType.TOOL_MATCH)],
            ungrounded=[_c("u", EvidenceType.INFERENCE)],
        )
        dump = p.model_dump_json()
        reloaded = Partition.model_validate_json(dump)
        assert reloaded == p

    def test_thresholds_round_trip(self) -> None:
        th = GSARThresholds(proceed=0.9, regenerate=0.7)
        reloaded = GSARThresholds.model_validate_json(th.model_dump_json())
        assert reloaded == th

    def test_partition_is_frozen(self) -> None:
        p = Partition(grounded=[_c("g")])
        with pytest.raises((TypeError, ValueError)):
            p.grounded = []  # type: ignore[misc]

    def test_partition_total_claims(self) -> None:
        p = Partition(
            grounded=[_c("g")],
            ungrounded=[_c("u"), _c("u2")],
            contradicted=[],
            complementary=[_c("k")],
        )
        assert p.total_claims == 4
        assert len(p.all_claims()) == 4


# ---------------------------------------------------------------------------
# Section: ablations from §8.5 — reproduce the table directionally
# ---------------------------------------------------------------------------


class TestAblationDirections:
    """Sanity-check the ablation directions reported in Table 1.

    These don't reproduce the FEVER scores (we don't have the dataset
    in unit tests), but verify the *direction* of each ablation matches
    the paper's documented effect on a synthetic partition.
    """

    def _mixed_partition(self) -> Partition:
        return Partition(
            grounded=[
                _c("g1", EvidenceType.TOOL_MATCH),
                _c("g2", EvidenceType.SPECIFIC_DATA),
            ],
            ungrounded=[_c("u", EvidenceType.INFERENCE)],
            contradicted=[_c("x", EvidenceType.SPECIFIC_DATA)],
            complementary=[_c("k", EvidenceType.COMPLEMENTARY_FINDING)],
        )

    def test_uniform_weights_collapse_p6_signal(self) -> None:
        # With all weights = 1.0, swapping tool_match for inference in G
        # leaves S unchanged — the §8.5 "uniform weights" ablation that
        # makes the framework indistinguishable from FaithJudge on small
        # samples.
        uniform = dict.fromkeys(EvidenceType, 1.0)
        p_tm = Partition(
            grounded=[_c("g", EvidenceType.TOOL_MATCH)],
            ungrounded=[_c("u", EvidenceType.INFERENCE)],
        )
        p_inf = Partition(
            grounded=[_c("g", EvidenceType.INFERENCE)],
            ungrounded=[_c("u", EvidenceType.INFERENCE)],
        )
        s_tm = gsar_score(p_tm, weight_map=uniform)
        s_inf = gsar_score(p_inf, weight_map=uniform)
        assert s_tm == pytest.approx(s_inf)

    def test_no_complementary_class_removes_k_uplift(self) -> None:
        # Folding K into U (the §8.5 "no complementary class" ablation)
        # strictly lowers S vs the GSAR default on this partition.
        p_default = self._mixed_partition()
        p_no_k = Partition(
            grounded=p_default.grounded,
            ungrounded=[*p_default.ungrounded, *p_default.complementary],
            contradicted=p_default.contradicted,
        )
        assert gsar_score(p_no_k) < gsar_score(p_default)

    def test_rho_zero_inflates_score(self) -> None:
        # The §8.5 rho=0 ablation (P5 in action): S goes up when ρ=0.
        p = self._mixed_partition()
        assert gsar_score(p, contradiction_penalty=0.0) > gsar_score(p)
