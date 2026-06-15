# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""A reproducible benchmark for the GSAR grounding decision.

Turns "validated GSAR" from a claim into a number you can run. A
:class:`GroundingBenchmark` scores a labeled set of evidence partitions
through :func:`~tulip.reasoning.gsar.gsar_score` +
:func:`~tulip.reasoning.gsar.decide` and reports how well the ship/abstain
decision matches ground truth — precision, recall, and F1 on "should this
finding ship", plus a threshold (``τ_proceed``) sweep.

Honest scope: the bundled fixtures are **authored** to characterise the
decision boundary (clear-grounding ships, ungrounded/contradicted abstains)
and to act as a **regression guard** on the scoring + thresholding logic.
This is not the external empirical validation of the GSAR method — that is
the arXiv paper (2604.23366). What this gives you is a CI-checked, runnable
number for the decision layer the SDK actually ships.

Run::

    python -m tulip.security.grounding_eval
"""

from __future__ import annotations

from dataclasses import dataclass

from tulip.reasoning.gsar import (
    Claim,
    Decision,
    EvidenceType,
    GSARThresholds,
    Partition,
    decide,
    gsar_score,
)


@dataclass(frozen=True)
class GroundingCase:
    """One labeled partition: the evidence and whether it *should* ship."""

    name: str
    partition: Partition
    should_ship: bool


def _c(text: str, etype: EvidenceType) -> Claim:
    """A claim with a single synthetic evidence ref (benchmark fixtures)."""
    return Claim(text=text, type=etype, evidence_refs=[f"fixture:{etype.value}"])


def _ship(name: str, partition: Partition) -> GroundingCase:
    """A labeled case that *should* ship a finding."""
    return GroundingCase(name=name, partition=partition, should_ship=True)


def _abstain(name: str, partition: Partition) -> GroundingCase:
    """A labeled case that *should* abstain."""
    return GroundingCase(name=name, partition=partition, should_ship=False)


def bundled_cases() -> list[GroundingCase]:
    """Authored, labeled partitions spanning the decision boundary.

    The positive cases are grounded by tool/signal/specific-data evidence;
    the negative cases are ungrounded (inference/domain only), contradicted,
    or grounded-but-outweighed. At the reference thresholds they separate
    cleanly — the benchmark exists to keep them separated as the scoring
    layer evolves.
    """
    tm, sd, sm = EvidenceType.TOOL_MATCH, EvidenceType.SPECIFIC_DATA, EvidenceType.SIGNAL_MATCH
    cf, sy = EvidenceType.COMPLEMENTARY_FINDING, EvidenceType.SYNTHESIS
    inf, dom = EvidenceType.INFERENCE, EvidenceType.DOMAIN

    return [
        # --- should ship: grounded evidence clears the bar ---
        _ship("tool_match_single", Partition(grounded=[_c("a", tm)])),
        _ship("tool_match_double", Partition(grounded=[_c("a", tm), _c("b", sd)])),
        _ship("signal_plus_specific", Partition(grounded=[_c("a", sm), _c("b", sd)])),
        _ship(
            "grounded_with_complementary",
            Partition(grounded=[_c("a", tm)], complementary=[_c("k", cf)]),
        ),
        _ship(
            "strong_overcomes_minor_contradiction",
            Partition(grounded=[_c("a", tm), _c("b", tm)], contradicted=[_c("x", inf)]),
        ),
        _ship("specific_data_only", Partition(grounded=[_c("a", sd)])),
        _ship(
            "mostly_grounded_tiny_inference",
            Partition(grounded=[_c("a", tm), _c("b", tm), _c("c", sd)], ungrounded=[_c("u", inf)]),
        ),
        # --- should abstain: ungrounded, contradicted, or outweighed ---
        _abstain("inference_only", Partition(ungrounded=[_c("u", inf)])),
        _abstain("domain_only", Partition(ungrounded=[_c("u", dom)])),
        _abstain(
            "contradicted_balanced",
            Partition(grounded=[_c("a", tm)], contradicted=[_c("x", tm)]),
        ),
        _abstain(
            "weak_grounded_heavy_ungrounded",
            Partition(grounded=[_c("a", sm)], ungrounded=[_c("u1", inf), _c("u2", inf)]),
        ),
        _abstain(
            "contradiction_dominant",
            Partition(grounded=[_c("a", sm)], contradicted=[_c("x1", tm), _c("x2", tm)]),
        ),
        _abstain(
            "single_grounded_single_inference",
            Partition(grounded=[_c("a", tm)], ungrounded=[_c("u", inf)]),
        ),
        _abstain(
            "synthesis_with_contradiction",
            Partition(grounded=[_c("a", sy)], contradicted=[_c("x", sd)]),
        ),
    ]


@dataclass(frozen=True)
class BenchmarkResult:
    """Confusion-matrix tallies + derived metrics for a benchmark run."""

    total: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        return (self.true_positive + self.true_negative) / self.total if self.total else 1.0

    def summary(self) -> str:
        return (
            f"n={self.total}  precision={self.precision:.3f}  recall={self.recall:.3f}  "
            f"f1={self.f1:.3f}  accuracy={self.accuracy:.3f}  "
            f"(tp={self.true_positive} fp={self.false_positive} "
            f"tn={self.true_negative} fn={self.false_negative})"
        )


class GroundingBenchmark:
    """Run a labeled grounding set through the GSAR decision and score it."""

    def __init__(self, cases: list[GroundingCase] | None = None) -> None:
        self.cases = cases if cases is not None else bundled_cases()

    def run(self, *, thresholds: GSARThresholds | None = None) -> BenchmarkResult:
        """Score every case; a case "ships" iff the decision is PROCEED."""
        tp = fp = tn = fn = 0
        for case in self.cases:
            score = gsar_score(case.partition)
            ships = decide(score, thresholds=thresholds) is Decision.PROCEED
            if case.should_ship and ships:
                tp += 1
            elif case.should_ship and not ships:
                fn += 1
            elif not case.should_ship and ships:
                fp += 1
            else:
                tn += 1
        return BenchmarkResult(
            total=len(self.cases),
            true_positive=tp,
            false_positive=fp,
            true_negative=tn,
            false_negative=fn,
        )

    def sweep(self, taus: list[float]) -> dict[float, BenchmarkResult]:
        """Run the benchmark at a series of ``τ_proceed`` thresholds."""
        out: dict[float, BenchmarkResult] = {}
        for tau in taus:
            regen = min(tau * 0.8, tau - 1e-6)
            out[tau] = self.run(thresholds=GSARThresholds(proceed=tau, regenerate=regen))
        return out


def run_benchmark(cases: list[GroundingCase] | None = None) -> BenchmarkResult:
    """Convenience: run the bundled (or given) benchmark at reference thresholds."""
    return GroundingBenchmark(cases).run()


__all__ = [
    "BenchmarkResult",
    "GroundingBenchmark",
    "GroundingCase",
    "bundled_cases",
    "run_benchmark",
]


if __name__ == "__main__":
    bench = GroundingBenchmark()
    print("GSAR grounding benchmark (authored fixtures)")
    print("  reference thresholds:", bench.run().summary())
    print("  τ_proceed sweep:")
    for tau, result in bench.sweep([0.5, 0.65, 0.8, 0.9]).items():
        print(f"    τ={tau:.2f}  precision={result.precision:.3f}  recall={result.recall:.3f}")
