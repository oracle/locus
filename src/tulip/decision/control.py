# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Where a decision model meets the admission gate.

Two seams, neither of which lets a model grant anything:

- :class:`DecisionAdvisor` is a :class:`~tulip.control.policy.ControlAdvisor`.
  :func:`~tulip.control.policy.approve` consults it only when policy already
  allows, and folds its answer in as ``max(policy, model)`` — so an admit head
  can hold or deny an action the policy let through, and can never release one
  the policy held.
- :func:`verification_from_decision` turns yes/no safety answers into the
  :class:`~tulip.control.verification.VerificationResult` that
  ``ControlPolicy.require_verification_score`` weighs. A flagged answer is a
  fatal refutation, which ``approve()`` turns into a deny.

Thresholds belong to weights. A threshold chosen by certifying a head on held-out
data — a Neyman–Pearson bound of the form ``P(miss rate > alpha) <= delta`` —
holds for the weights it was measured on and for nothing else. Ship it with the
model, and recalibrate when the model changes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from tulip.control.policy import Action, ApprovalOutcome
from tulip.control.verification import Refutation, VerificationResult
from tulip.decision.fields import Choice, Decision
from tulip.decision.provider import DecisionProvider, decide_sync


__all__ = [
    "ADMIT_FIELD",
    "ADMIT_QUESTION",
    "DecisionAdvisor",
    "describe_action",
    "verification_from_decision",
]

_log = logging.getLogger(__name__)

_OUTCOMES = (ApprovalOutcome.ALLOW, ApprovalOutcome.REQUIRE_HUMAN, ApprovalOutcome.DENY)

#: The question an admit head is asked.
ADMIT_QUESTION = (
    "Under the policy, may this action proceed, does it need a person to approve it "
    "first, or is it forbidden?"
)

#: The admit field: allow / require_human / deny, in that order.
ADMIT_FIELD = Choice("admit", ADMIT_QUESTION, _OUTCOMES)


def describe_action(action: Action) -> str:
    """A plain rendering of an :class:`Action`, for a head that reads one."""
    parts = [f"action: {action.name}"]
    if action.asset:
        parts.append(f"asset: {action.asset}")
    parts.append(f"environment: {action.environment}")
    if action.kind:
        parts.append(f"kind: {action.kind}")
    parts.append(f"blast radius: {action.blast_radius}")
    if action.tags:
        parts.append(f"tags: {', '.join(sorted(action.tags))}")
    if action.cost_usd:
        parts.append(f"cost: ${action.cost_usd:,.2f}")
    return "\n".join(parts)


@dataclass
class DecisionAdvisor:
    """An admit head, as the reference monitor sees it.

    Pass it to ``approve(..., advisor=)``. It returns one of the three outcome
    strings or ``None`` for no opinion — and ``None`` on any failure, so a head
    that is down leaves exactly the decision policy alone would have made.

    Two ways to read the head:

    - **argmax** (default): the most probable outcome, if its probability is at
      least ``min_probability``.
    - **certified threshold** (``hold_at``): the risk ``1 - P(allow)`` against a
      threshold calibrated for these weights. At or above it, the more probable
      of ``require_human`` and ``deny``; below it, ``allow``.

    Args:
        provider: Serves the admit head.
        describe: Renders the action (and whatever policy text the head reads)
            as the input. Defaults to :func:`describe_action`.
        field: The admit field. Its options must be the three outcomes.
        min_probability: The argmax mode's floor.
        hold_at: The certified-threshold mode's risk threshold.
    """

    provider: DecisionProvider
    describe: Callable[[Action], str] = describe_action
    field: Choice = field(default=ADMIT_FIELD)
    min_probability: float = 0.0
    hold_at: float | None = None

    def __post_init__(self) -> None:
        if set(self.field.options) != set(_OUTCOMES):
            raise ValueError(f"an admit field's options must be exactly {_OUTCOMES}")
        if not 0.0 <= self.min_probability <= 1.0:
            raise ValueError("min_probability must lie in [0, 1]")
        if self.hold_at is not None and not 0.0 < self.hold_at <= 1.0:
            raise ValueError("hold_at must lie in (0, 1]")

    def advise(self, action: Action) -> str | None:
        """The head's outcome for ``action``, or ``None`` for no opinion."""
        try:
            decision = decide_sync(self.provider, self.describe(action), [self.field])
            answer = decision[self.field.name]
        except Exception:  # noqa: BLE001 - a failing advisor is an absent advisor
            _log.warning("decision advisor unavailable for %s", action.name, exc_info=True)
            return None
        if self.hold_at is not None:
            risk = 1.0 - answer.p(ApprovalOutcome.ALLOW)
            if risk < self.hold_at:
                return ApprovalOutcome.ALLOW
            human, deny = answer.p(ApprovalOutcome.REQUIRE_HUMAN), answer.p(ApprovalOutcome.DENY)
            return ApprovalOutcome.DENY if deny > human else ApprovalOutcome.REQUIRE_HUMAN
        if answer.probability < self.min_probability:
            return None
        return answer.label


def verification_from_decision(
    decision: Decision, thresholds: Mapping[str, float]
) -> VerificationResult:
    """Yes/no safety answers as a :class:`VerificationResult` for ``approve()``.

    Each named field must be a yes/no answer in ``decision``. One whose P(yes)
    reaches its threshold is a **fatal** refutation, so ``survives`` is False
    and ``approve()`` denies.

    ``confidence`` is ``1 - max(min(1, P(yes) / threshold))`` over the fields:
    1 when every head is sure the answer is no, 0 when any reaches its
    threshold, and in between proportionally to how close the nearest head came
    to its own threshold. Scaling by each threshold keeps heads with different
    thresholds comparable; ``ControlPolicy.require_verification_score`` then
    sets how close is close enough to want a person (0.8 holds anything past a
    fifth of a threshold).

    ``evidence_quality`` is the lowest :attr:`~tulip.decision.Answer.coverage`
    among the fields: how much of the model's probability was on the listed
    answers at all.
    """
    if not thresholds:
        raise ValueError("name at least one field and its threshold")
    refutations: list[Refutation] = []
    nearest = 0.0
    coverage = 1.0
    for name, threshold in thresholds.items():
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold for {name!r} must lie in (0, 1]; got {threshold!r}")
        answer = decision[name]
        p_yes = answer.p_yes
        nearest = max(nearest, min(1.0, p_yes / threshold))
        coverage = min(coverage, answer.coverage)
        if p_yes >= threshold:
            refutations.append(
                Refutation(
                    reason=f"{name}: P(yes) {p_yes:.3f} reaches its threshold {threshold:.3f}",
                    weight="fatal",
                )
            )
    return VerificationResult(
        survives=not refutations,
        confidence=1.0 - nearest,
        evidence_quality=coverage,
        refutations=refutations,
        notes=f"decision model {decision.model}",
    )
