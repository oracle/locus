# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""One verdict is one opinion. These functions say how far to trust it.

A judge grading the same answer twice can land a whole letter apart; two
judges of one answer disagree more often than their confidence suggests;
and a judge that agrees with a human 80% of the time may be agreeing by
luck when most answers land on the same two letters. None of that is
visible from a single :class:`~locus.evaluation.Verdict`. It takes repeats,
panels, an independent second opinion and human labels, and arithmetic
that is the same on every machine.

Everything here is a pure function over verdicts or grade pairs. No model
is called, no configuration is read; thresholds arrive as arguments so a
test can pin them and a report can say which ones it used.

- :func:`consistency`: the same case graded N times. Grade agreement, score
  spread, and a flag when the judge, not the agent, is what varies.
- :func:`panel_agreement`: N judges on one answer. Consensus letter, its
  share, and the average pairwise correlation of their dimension scores.
- :func:`trust_decision`: a primary verdict against an independent second
  opinion. ``trusted`` only when they corroborate; ``calibration_required``
  when nothing checked the primary, or it graded its own model's work;
  ``rejected`` when they conflict.
- :func:`cohen_kappa`: chance-corrected agreement between two judges over
  many cases. Undefined, not perfect, when neither judge ever varied.
- :func:`calibration_drift`: judge letters against human letters, and
  which way the judge leans.
- :func:`wilson_interval` and :func:`bootstrap_ci`: intervals that behave
  on small samples and bounded scores.
- :func:`publication_gates`: the reasons a cell of numbers should not yet
  be shown as a result.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from enum import StrEnum
from statistics import fmean, pstdev
from typing import Any, Literal

from pydantic import BaseModel, Field

from locus.evaluation.rubric import DEFAULT_GRADE_THRESHOLDS


__all__ = [
    "DEFAULT_GRADE_ORDER",
    "CalibrationDrift",
    "Consistency",
    "HoldReason",
    "Kappa",
    "PanelAgreement",
    "ReliabilityGates",
    "TrustDecision",
    "TrustGates",
    "bootstrap_ci",
    "calibration_drift",
    "cohen_kappa",
    "consistency",
    "panel_agreement",
    "publication_gates",
    "trust_decision",
    "wilson_interval",
]


#: Letters best first, from the default bands.
DEFAULT_GRADE_ORDER: tuple[str, ...] = tuple(
    sorted(DEFAULT_GRADE_THRESHOLDS, key=lambda letter: -DEFAULT_GRADE_THRESHOLDS[letter])
)

# Two-sided normal quantile for a 95% interval.
_Z_95 = 1.959964


# ---------------------------------------------------------------------------
# reading a verdict-shaped thing
# ---------------------------------------------------------------------------


def _field(verdict: Any, name: str) -> Any:
    if isinstance(verdict, Mapping):
        return verdict.get(name)
    return getattr(verdict, name, None)


def _graded(verdict: Any) -> bool:
    return not bool(_field(verdict, "abstained")) and not bool(_field(verdict, "unparseable"))


def _score(verdict: Any) -> float | None:
    value = _field(verdict, "score")
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _grade(verdict: Any) -> str | None:
    value = _field(verdict, "grade")
    return str(value).upper() if isinstance(value, str) and value else None


def _dimensions(verdict: Any) -> dict[str, float]:
    value = _field(verdict, "dimensions")
    if not isinstance(value, Mapping):
        return {}
    return {
        str(k): float(v)
        for k, v in value.items()
        if isinstance(v, int | float) and not isinstance(v, bool)
    }


def _self_graded(verdict: Any) -> bool:
    provenance = _field(verdict, "provenance")
    return bool(_field(provenance, "self_graded")) if provenance is not None else False


def _rank(grade: str | None, order: Sequence[str]) -> int | None:
    if grade is None:
        return None
    upper = grade.upper()
    return list(order).index(upper) if upper in order else None


def _stddev(values: Sequence[float]) -> float | None:
    return pstdev(values) if len(values) >= 2 else None


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = fmean(xs), fmean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / (sx * sy)


# ---------------------------------------------------------------------------
# consistency: one case, N readings
# ---------------------------------------------------------------------------


class Consistency(BaseModel):
    """How much the verdict on one case moved across repeated readings."""

    samples: int = Field(description="Readings that produced a verdict")
    not_graded: int = Field(default=0, description="Readings the judge abstained on")
    pass_rate: float | None = Field(
        default=None, description="Share of graded readings that passed"
    )
    score_mean: float | None = None
    score_stddev: float | None = None
    grade_distribution: dict[str, int] = Field(default_factory=dict)
    grade_agreement: float | None = Field(
        default=None, description="Share of graded readings on the modal letter"
    )
    dimension_stddev: dict[str, float] = Field(default_factory=dict)
    flagged_unstable: bool = Field(
        default=False,
        description="Enough readings, and they disagree: the judge is what varies",
    )


def consistency(
    samples: Sequence[Any],
    *,
    min_samples: int = 3,
    agreement_floor: float = 0.7,
    score_spread_max: float = 0.10,
) -> Consistency:
    """Summarise repeated verdicts on one case.

    Args:
        samples: Verdicts (or verdict-shaped mappings) of one case.
        min_samples: Fewer graded readings than this are never flagged: two
            readings disagreeing is a coin toss, not a finding.
        agreement_floor: Below this share on the modal letter the case is
            flagged unstable (when the verdicts carry letters).
        score_spread_max: Above this population standard deviation of the
            0-1 scores the case is flagged unstable.
    """
    graded = [v for v in samples if _graded(v)]
    scores = [s for s in (_score(v) for v in graded) if s is not None]
    grades = [g for g in (_grade(v) for v in graded) if g is not None]
    distribution = dict(Counter(grades))
    agreement = max(distribution.values()) / len(grades) if grades else None
    spread = _stddev(scores)

    buckets: dict[str, list[float]] = {}
    for verdict in graded:
        for key, value in _dimensions(verdict).items():
            buckets.setdefault(key, []).append(value)
    dimension_stddev = {
        key: round(deviation, 2)
        for key, values in sorted(buckets.items())
        if (deviation := _stddev(values)) is not None
    }

    passes = [bool(_field(v, "passed")) for v in graded]
    unstable = len(graded) >= min_samples and (
        (agreement is not None and agreement < agreement_floor)
        or (spread is not None and spread > score_spread_max)
    )
    return Consistency(
        samples=len(graded),
        not_graded=len(samples) - len(graded),
        pass_rate=sum(passes) / len(passes) if passes else None,
        score_mean=fmean(scores) if scores else None,
        score_stddev=spread,
        grade_distribution=distribution,
        grade_agreement=agreement,
        dimension_stddev=dimension_stddev,
        flagged_unstable=unstable,
    )


# ---------------------------------------------------------------------------
# panel: one answer, N judges
# ---------------------------------------------------------------------------


class PanelAgreement(BaseModel):
    """Inter-rater agreement of several judges on one answer."""

    judges: int = Field(description="Judges that produced a verdict")
    score_mean: float | None = None
    score_stddev: float | None = None
    score_pearson: float | None = Field(
        default=None,
        description="Average pairwise Pearson r over the judges' dimension-score vectors; "
        "None with fewer than two judges or constant vectors",
    )
    consensus_grade: str | None = Field(default=None, description="The modal letter")
    grade_consensus_rate: float | None = Field(
        default=None, description="Share of judges on the modal letter"
    )
    flagged_low_confidence: bool | None = Field(
        default=None,
        description="Weak consensus or a wide score spread; None with fewer than two judges",
    )


def panel_agreement(
    verdicts: Sequence[Any],
    *,
    consensus_floor: float = 0.7,
    score_spread_max: float = 0.10,
    grade_order: Sequence[str] = DEFAULT_GRADE_ORDER,
) -> PanelAgreement:
    """Roll N judges' verdicts on one answer into one agreement report."""
    graded = [v for v in verdicts if _graded(v)]
    scores = [s for s in (_score(v) for v in graded) if s is not None]
    spread = _stddev(scores)

    vectors: list[list[float]] = []
    keys: list[str] = []
    for verdict in graded:
        dims = _dimensions(verdict)
        if dims and not keys:
            keys = sorted(dims)
        if keys:
            vectors.append([dims.get(k, 0.0) for k in keys])
    correlations = [
        r
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
        if (r := _pearson(vectors[i], vectors[j])) is not None
    ]

    ranks = [r for r in (_rank(_grade(v), grade_order) for v in graded) if r is not None]
    consensus: str | None = None
    rate: float | None = None
    if ranks:
        modal_rank, count = Counter(ranks).most_common(1)[0]
        consensus = grade_order[modal_rank]
        rate = count / len(ranks) if len(ranks) >= 2 else None

    flagged: bool | None = None
    if len(graded) >= 2:
        flagged = (rate is not None and rate < consensus_floor) or (
            spread is not None and spread > score_spread_max
        )
    return PanelAgreement(
        judges=len(graded),
        score_mean=fmean(scores) if scores else None,
        score_stddev=spread,
        score_pearson=fmean(correlations) if correlations else None,
        consensus_grade=consensus,
        grade_consensus_rate=rate,
        flagged_low_confidence=flagged,
    )


# ---------------------------------------------------------------------------
# trust: a primary and an independent second opinion
# ---------------------------------------------------------------------------


class TrustGates(BaseModel):
    """Letter distances separating the three trust decisions."""

    calibration_letter_delta: int = Field(default=1, ge=1)
    rejection_letter_delta: int = Field(default=2, ge=1)
    score_delta_max: float = Field(
        default=0.10,
        ge=0.0,
        description="For verdicts without letters: the score distance two passing (or two "
        "failing) opinions may differ by and still corroborate",
    )


class TrustDecision(BaseModel):
    """Whether a grade may be acted on, and why."""

    decision: Literal["trusted", "calibration_required", "rejected"]
    reason: str
    letter_distance: int | None = None
    score_distance: float | None = None


def trust_decision(
    primary: Any,
    independent: Any | None,
    *,
    gates: TrustGates | None = None,
    grade_order: Sequence[str] = DEFAULT_GRADE_ORDER,
) -> TrustDecision:
    """Decide whether a primary verdict can be acted on.

    Only a corroborated grade is ``trusted``. A primary nobody checked, or
    one a model gave its own work, is ``calibration_required``; two opinions
    that conflict are ``rejected``. Letters decide when both verdicts carry
    them; otherwise the pass decision and the score distance do.
    """
    gates = gates or TrustGates()
    if not _graded(primary):
        return TrustDecision(decision="calibration_required", reason="primary not graded")
    if _self_graded(primary) or (independent is not None and _self_graded(independent)):
        return TrustDecision(decision="calibration_required", reason="self-graded")
    if independent is None or not _graded(independent):
        return TrustDecision(decision="calibration_required", reason="no independent opinion")

    primary_rank = _rank(_grade(primary), grade_order)
    independent_rank = _rank(_grade(independent), grade_order)
    if primary_rank is not None and independent_rank is not None:
        distance = abs(primary_rank - independent_rank)
        if distance >= gates.rejection_letter_delta:
            return TrustDecision(
                decision="rejected",
                reason="grades differ by two or more letters",
                letter_distance=distance,
            )
        if distance >= gates.calibration_letter_delta:
            return TrustDecision(
                decision="calibration_required",
                reason="grades differ by a letter",
                letter_distance=distance,
            )
        return TrustDecision(decision="trusted", reason="grades match", letter_distance=distance)

    primary_score, independent_score = _score(primary), _score(independent)
    score_distance = (
        abs(primary_score - independent_score)
        if primary_score is not None and independent_score is not None
        else None
    )
    if bool(_field(primary, "passed")) != bool(_field(independent, "passed")):
        return TrustDecision(
            decision="rejected",
            reason="one opinion passes and the other fails",
            score_distance=score_distance,
        )
    if score_distance is not None and score_distance > gates.score_delta_max:
        return TrustDecision(
            decision="calibration_required",
            reason="scores differ beyond the gate",
            score_distance=score_distance,
        )
    return TrustDecision(decision="trusted", reason="opinions agree", score_distance=score_distance)


# ---------------------------------------------------------------------------
# kappa and calibration: many cases
# ---------------------------------------------------------------------------


class Kappa(BaseModel):
    """Cohen's kappa over paired labels, or why it is undefined."""

    kappa: float | None = None
    pairs: int = 0
    undefined_reason: Literal["insufficient_sample", "no_variance"] | None = None


def cohen_kappa(pairs: Sequence[tuple[str, str]], *, min_pairs: int = 30) -> Kappa:
    """Chance-corrected agreement between two raters over many cases.

    Raw agreement overstates trust: with five letters and most answers landing
    on the same two, two judges collide often by luck alone. Kappa subtracts
    the expected collision rate. It is undefined on a small sample (a handful
    of pairs swings on one disagreement) and when both raters gave every case
    the same single label (agreement cannot be told from indifference).
    """
    clean = [(str(a).upper(), str(b).upper()) for a, b in pairs if a and b]
    count = len(clean)
    if count < min_pairs:
        return Kappa(pairs=count, undefined_reason="insufficient_sample")
    observed = sum(1 for a, b in clean if a == b) / count
    first = Counter(a for a, _ in clean)
    second = Counter(b for _, b in clean)
    chance = sum(
        (first[label] / count) * (second[label] / count) for label in set(first) | set(second)
    )
    if chance >= 1.0:
        return Kappa(pairs=count, undefined_reason="no_variance")
    return Kappa(kappa=round((observed - chance) / (1 - chance), 4), pairs=count)


class CalibrationDrift(BaseModel):
    """How far the judge's letters sit from human letters on the same cases."""

    sample_size: int = 0
    agreement_rate: float | None = None
    mean_letter_drift: float | None = Field(
        default=None,
        description="Mean of (human rank - judge rank): positive means the judge is kinder "
        "than the human, negative harsher",
    )
    abs_letter_drift: float | None = None
    direction: Literal["optimistic", "pessimistic", "balanced", "unknown"] = "unknown"


def calibration_drift(
    pairs: Sequence[tuple[str, str]],
    *,
    grade_order: Sequence[str] = DEFAULT_GRADE_ORDER,
    balanced_within: float = 0.2,
) -> CalibrationDrift:
    """Aggregate ``(judge_grade, human_grade)`` pairs into a bias report.

    Pairs whose letters the order does not know are dropped, not guessed at.
    """
    ranked = [
        (j, h)
        for j, h in (
            (_rank(judge, grade_order), _rank(human, grade_order)) for judge, human in pairs
        )
        if j is not None and h is not None
    ]
    if not ranked:
        return CalibrationDrift()
    size = len(ranked)
    agreement = sum(1 for j, h in ranked if j == h) / size
    mean_drift = sum(h - j for j, h in ranked) / size
    abs_drift = sum(abs(h - j) for j, h in ranked) / size
    if mean_drift > balanced_within:
        direction: Literal["optimistic", "pessimistic", "balanced"] = "optimistic"
    elif mean_drift < -balanced_within:
        direction = "pessimistic"
    else:
        direction = "balanced"
    return CalibrationDrift(
        sample_size=size,
        agreement_rate=round(agreement, 3),
        mean_letter_drift=round(mean_drift, 3),
        abs_letter_drift=round(abs_drift, 3),
        direction=direction,
    )


# ---------------------------------------------------------------------------
# intervals
# ---------------------------------------------------------------------------


def wilson_interval(successes: int, trials: int, *, z: float = _Z_95) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    The normal approximation builds its error from the observed proportion,
    so zero successes in twenty trials reads as the interval ``[0, 0]``:
    perfect certainty from any sample. Wilson derives the bound from each
    candidate rate instead, so zero in twenty reads as "somewhere under 16%".
    No trials is no information: ``(0, 1)``.
    """
    if successes < 0 or trials < 0 or successes > trials:
        raise ValueError("successes must be between 0 and trials")
    if trials == 0:
        return (0.0, 1.0)
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    margin = (z / denominator) * math.sqrt(
        proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    rank = math.ceil(percentile / 100 * len(sorted_values))
    return sorted_values[min(max(rank - 1, 0), len(sorted_values) - 1)]


def bootstrap_ci(
    values: Sequence[float],
    *,
    seed: int,
    resamples: int = 1000,
    lower_percentile: float = 2.5,
    upper_percentile: float = 97.5,
) -> tuple[float, float]:
    """Percentile-bootstrap confidence interval of the mean.

    Scores are bounded and cluster at band edges, so no normal assumption.
    The caller supplies the seed and a local generator is used, so the same
    values always give the same interval.
    """
    if not values:
        raise ValueError("bootstrap requires at least one value")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    rng = random.Random(seed)  # noqa: S311 — reproducible resampling, not security
    size = len(values)
    means = sorted(fmean(rng.choices(values, k=size)) for _ in range(resamples))
    return (_percentile(means, lower_percentile), _percentile(means, upper_percentile))


# ---------------------------------------------------------------------------
# publication gates
# ---------------------------------------------------------------------------


class HoldReason(StrEnum):
    """Why a cell of numbers is not yet a result."""

    LOW_SAMPLE = "low_sample"
    WIDE_CI = "wide_ci"
    NO_GRADE_VARIANCE = "no_grade_variance"
    LOW_AGREEMENT = "low_agreement"
    HIGH_CONTRADICTION = "high_contradiction"


class ReliabilityGates(BaseModel):
    """Thresholds a cell must clear before its numbers are shown as a result."""

    min_samples: int = Field(default=30, ge=1)
    ci_max_width: float = Field(default=0.10, ge=0.0, description="On the 0-1 score scale")
    min_agreement: float = Field(default=0.6, description="Minimum kappa between two judges")
    max_contradiction_rate: float = Field(default=0.1, ge=0.0, le=1.0)


def publication_gates(
    gates: ReliabilityGates,
    *,
    samples: int,
    ci_low: float,
    ci_high: float,
    agreement: Kappa | None = None,
    contradiction_rate: float | None = None,
) -> tuple[HoldReason, ...]:
    """Every gate the cell fails, in declaration order.

    All gates are evaluated: a cell can fail several at once and must say so.
    An unavailable kappa holds the cell: a figure only one judge has ever
    seen is one model's opinion, not a reliability claim.
    """
    reasons: list[HoldReason] = []
    if samples < gates.min_samples:
        reasons.append(HoldReason.LOW_SAMPLE)
    if round(ci_high - ci_low, 9) > gates.ci_max_width:
        reasons.append(HoldReason.WIDE_CI)
    if agreement is not None and agreement.undefined_reason == "no_variance":
        reasons.append(HoldReason.NO_GRADE_VARIANCE)
    elif agreement is None or agreement.kappa is None or agreement.kappa < gates.min_agreement:
        reasons.append(HoldReason.LOW_AGREEMENT)
    if contradiction_rate is not None and contradiction_rate > gates.max_contradiction_rate:
        reasons.append(HoldReason.HIGH_CONTRADICTION)
    return tuple(reasons)
