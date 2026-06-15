# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Regression guard for the GSAR grounding decision benchmark.

Asserts the bundled labeled fixtures still separate cleanly at the reference
thresholds — a guard against regressions in `gsar_score` / `decide` /
`ground_finding`. Runs offline, no model.
"""

from __future__ import annotations

from tulip.reasoning.gsar import Decision, GSARThresholds, decide, gsar_score
from tulip.security.grounding_eval import GroundingBenchmark, bundled_cases, run_benchmark


def test_reference_thresholds_separate_cleanly() -> None:
    result = run_benchmark()
    assert result.total == len(bundled_cases())
    # Authored fixtures are clearly separated; the decision layer should nail them.
    assert result.precision >= 0.9
    assert result.recall >= 0.9
    assert result.f1 >= 0.9


def test_every_case_classified_correctly_at_reference() -> None:
    for case in bundled_cases():
        ships = decide(gsar_score(case.partition)) is Decision.PROCEED
        assert ships == case.should_ship, f"{case.name}: ships={ships} expected={case.should_ship}"


def test_sweep_is_monotone_in_the_expected_direction() -> None:
    bench = GroundingBenchmark()
    sweep = bench.sweep([0.5, 0.65, 0.8, 0.9])
    # Lower τ ships more → recall non-decreasing as τ falls; precision non-increasing.
    assert sweep[0.5].recall >= sweep[0.9].recall
    assert sweep[0.9].precision >= sweep[0.5].precision
    # The reference threshold is the sweet spot for these fixtures.
    assert sweep[0.8].precision == 1.0
    assert sweep[0.8].recall == 1.0


def test_threshold_override_changes_decision() -> None:
    # A permissive threshold ships more than a strict one.
    bench = GroundingBenchmark()
    permissive = bench.run(thresholds=GSARThresholds(proceed=0.5, regenerate=0.3))
    strict = bench.run(thresholds=GSARThresholds(proceed=0.95, regenerate=0.6))
    permissive_ships = permissive.true_positive + permissive.false_positive
    strict_ships = strict.true_positive + strict.false_positive
    assert permissive_ships >= strict_ships
