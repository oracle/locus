# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Policy + approval — safe-before-action, the way a CISO reasons about it.

An agent that *finds* something still must not *act* on its own authority. A
`ControlPolicy` encodes the rules a security org actually has — how strong the
verification must be, how large a blast radius may auto-proceed, and which
actions always need a human — and :func:`approve` weighs a proposed
:class:`Action` against the **evidence** (the grounded `Evidence`), the
**verification** (the :class:`~tulip.security.verify.VerificationResult`), and the policy
to return an :class:`ApprovalDecision`: allow, require a human, or deny.

This is the "safe before action" half of the trust quartet (grounded · verified
· auditable · safe-before-action). The richer commercial approval/governance
engine builds on this open contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tulip.security.findings import Evidence
from tulip.security.taxonomy import Severity, severity_at_least
from tulip.security.verify import VerificationResult


class ApprovalOutcome:
    """Outcome labels (kept simple/stable as plain strings)."""

    ALLOW = "allow"
    REQUIRE_HUMAN = "require_human"
    DENY = "deny"


# Strength ordering — the strongest triggered outcome wins.
_ORDER = {ApprovalOutcome.ALLOW: 0, ApprovalOutcome.REQUIRE_HUMAN: 1, ApprovalOutcome.DENY: 2}


@dataclass(frozen=True)
class ControlPolicy:
    """The CISO knobs. Defaults are conservative — auto-allow only the safe path.

    - ``require_verification_score``: minimum :class:`VerificationResult` confidence to
      auto-allow; below it (or with no verdict) a human is required.
    - ``max_blast_radius``: most assets an action may affect to auto-allow.
    - ``require_human_for``: action labels (environment / kind / tag) that always
      need a human (default: anything in ``production``).
    - ``deny_for``: labels that are hard-denied outright.
    - ``min_severity``: don't act on findings below this band.
    """

    require_verification_score: float = 0.8
    max_blast_radius: int = 1
    require_human_for: frozenset[str] = field(default_factory=lambda: frozenset({"production"}))
    deny_for: frozenset[str] = field(default_factory=frozenset)
    min_severity: Severity = Severity.LOW


@dataclass(frozen=True)
class Action:
    """A proposed response action, with the attributes policy reasons over."""

    name: str
    asset: str = ""
    blast_radius: int = 1
    environment: str = "unknown"
    kind: str = ""
    tags: frozenset[str] = field(default_factory=frozenset)

    def labels(self) -> set[str]:
        """The environment / kind / tags as one label set for policy matching."""
        return {self.environment, self.kind, *self.tags} - {""}


@dataclass(frozen=True)
class ApprovalDecision:
    """The outcome of weighing an action against evidence, verification, and policy."""

    outcome: str
    reason: str
    action: Action
    checks: list[str] = field(default_factory=list)

    @property
    def allowed(self) -> bool:
        return self.outcome == ApprovalOutcome.ALLOW


def approve(
    action: Action,
    *,
    policy: ControlPolicy,
    finding: Evidence | None = None,
    verdict: VerificationResult | None = None,
) -> ApprovalDecision:
    """Decide whether ``action`` may proceed: allow / require_human / deny.

    Weighs every rule and returns the **strongest** triggered outcome (deny >
    require_human > allow), recording each check that fired so the decision is
    auditable.

    Args:
        action: The proposed action.
        policy: The governing :class:`ControlPolicy`.
        finding: The evidence the action responds to (optional).
        verdict: The :func:`~tulip.security.verify.verify` result (optional, but
            auto-allow needs one that clears the policy bar).

    Returns:
        An :class:`ApprovalDecision`.
    """
    triggered: list[tuple[str, str]] = []
    labels = action.labels()

    denied = labels & policy.deny_for
    if denied:
        triggered.append((ApprovalOutcome.DENY, f"labels {sorted(denied)} are denied by policy"))

    if finding is not None and not severity_at_least(finding.severity, policy.min_severity):
        triggered.append(
            (
                ApprovalOutcome.DENY,
                f"finding severity {finding.severity.value} is below the policy "
                f"minimum {policy.min_severity.value}",
            )
        )

    if policy.require_verification_score > 0:
        if verdict is None:
            triggered.append((ApprovalOutcome.REQUIRE_HUMAN, "no verification provided"))
        elif not verdict.survives:
            triggered.append((ApprovalOutcome.DENY, "the finding did not survive verification"))
        elif verdict.confidence < policy.require_verification_score:
            triggered.append(
                (
                    ApprovalOutcome.REQUIRE_HUMAN,
                    f"verification confidence {verdict.confidence:.2f} is below the bar "
                    f"{policy.require_verification_score:.2f}",
                )
            )

    if action.blast_radius > policy.max_blast_radius:
        triggered.append(
            (
                ApprovalOutcome.REQUIRE_HUMAN,
                f"blast radius {action.blast_radius} exceeds the maximum {policy.max_blast_radius}",
            )
        )

    needs_human = labels & policy.require_human_for
    if needs_human:
        triggered.append(
            (ApprovalOutcome.REQUIRE_HUMAN, f"labels {sorted(needs_human)} require human approval")
        )

    if not triggered:
        return ApprovalDecision(
            outcome=ApprovalOutcome.ALLOW,
            reason="all policy checks passed",
            action=action,
        )
    outcome = max((o for o, _ in triggered), key=lambda o: _ORDER[o])
    checks = [why for _, why in triggered]
    return ApprovalDecision(
        outcome=outcome,
        reason="; ".join(checks),
        action=action,
        checks=checks,
    )


__all__ = [
    "Action",
    "ApprovalDecision",
    "ApprovalOutcome",
    "ControlPolicy",
    "approve",
]
