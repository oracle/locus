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
from typing import Protocol

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

# The tag an action carries when its execution is sandboxed. Actions whose
# labels intersect ``ControlPolicy.require_sandbox_for`` are denied unless
# this tag is present.
SANDBOXED_TAG = "sandboxed"


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
    - ``require_sandbox_for``: labels whose actions must execute in a sandbox —
      an action matching one of these is denied unless it carries the
      :data:`SANDBOXED_TAG` tag. Enforced at the agent loop's tool seam by
      :class:`~tulip.tools.sandbox.SandboxEnforcerHook`.
    """

    require_verification_score: float = 0.8
    max_blast_radius: int = 1
    require_human_for: frozenset[str] = field(default_factory=lambda: frozenset({"production"}))
    deny_for: frozenset[str] = field(default_factory=frozenset)
    min_severity: Severity = Severity.LOW
    require_sandbox_for: frozenset[str] = field(default_factory=frozenset)


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


class ControlAdvisor(Protocol):
    """A trained control model, as the reference monitor sees it.

    Structural rather than an import: :func:`approve` must not depend on a model
    runtime, and a deployment with no control model has to be the ordinary case
    rather than a degraded one.

    :meth:`advise` returns one of the three outcome strings, or ``None`` for "no
    opinion". **Nothing it returns can widen authority** — that is enforced in
    :func:`approve`, in code, not in this docstring.
    """

    def advise(self, action: Action) -> str | None:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True)
class ApprovalDecision:
    """The outcome of weighing an action against evidence, verification, and policy."""

    outcome: str
    reason: str
    action: Action
    checks: list[str] = field(default_factory=list)
    #: What the deterministic policy decided, before any control model spoke.
    #: Always the ground truth of the decision: ``outcome`` may be *stronger*
    #: than this and can never be weaker, so an auditor comparing the two sees
    #: exactly what the model changed and in which direction.
    policy_outcome: str = ApprovalOutcome.ALLOW
    #: What the control model advised, or ``None`` if none was consulted, it
    #: declined, or it failed. Advisory only.
    model_outcome: str | None = None

    @property
    def allowed(self) -> bool:
        return self.outcome == ApprovalOutcome.ALLOW

    @property
    def escalated_by_model(self) -> bool:
        """Whether a control model made this decision stricter than policy alone."""
        return self.model_outcome is not None and _ORDER[self.outcome] > _ORDER[self.policy_outcome]


def approve(
    action: Action,
    *,
    policy: ControlPolicy,
    finding: Evidence | None = None,
    verdict: VerificationResult | None = None,
    advisor: ControlAdvisor | None = None,
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
        advisor: An optional trained control model. It may only raise the
            decision toward caution — see :func:`_combine`. Omitting it, or
            passing one that fails, yields exactly the decision policy alone
            would have made.

    Returns:
        An :class:`ApprovalDecision`.
    """
    triggered: list[tuple[str, str]] = []
    labels = action.labels()

    denied = labels & policy.deny_for
    if denied:
        triggered.append((ApprovalOutcome.DENY, f"labels {sorted(denied)} are denied by policy"))

    unsandboxed = labels & policy.require_sandbox_for
    if unsandboxed and SANDBOXED_TAG not in labels:
        triggered.append(
            (
                ApprovalOutcome.DENY,
                f"labels {sorted(unsandboxed)} require sandboxed execution "
                f"(the action carries no {SANDBOXED_TAG!r} tag)",
            )
        )

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
        policy_outcome = ApprovalOutcome.ALLOW
        # Empty, not seeded with a passing note: ``checks`` records what *fired*,
        # and a clean allow fires nothing. That is the existing contract and the
        # combiner must not quietly change it.
        checks: list[str] = []
    else:
        policy_outcome = max((o for o, _ in triggered), key=lambda o: _ORDER[o])
        checks = [why for _, why in triggered]

    outcome, model_outcome, advisory = _combine(action, policy_outcome, advisor)
    if advisory is not None:
        checks = [*checks, advisory]

    return ApprovalDecision(
        outcome=outcome,
        reason="; ".join(checks) if checks else "all policy checks passed",
        action=action,
        checks=checks,
        policy_outcome=policy_outcome,
        model_outcome=model_outcome,
    )


def _combine(
    action: Action, policy_outcome: str, advisor: ControlAdvisor | None
) -> tuple[str, str | None, str | None]:
    """Fold a control model's advice into a policy decision. **Never widens it.**

    The whole security argument for admitting a probabilistic component into a
    reference monitor lives in these few lines, so each constraint is stated and
    enforced here rather than assumed of the caller:

    1. **Consulted only on the allow branch.** If policy already holds or denies,
       the model is not asked. It could not change the answer — ``max()`` of an
       equal-or-lower value is a no-op — so asking would spend latency to learn
       nothing, and would put a model in the path of decisions it cannot affect.
    2. **``max()``, never ``min()``, never assignment.** The model raises a
       decision toward caution or leaves it alone. There is no code path by
       which it grants.
    3. **Anything unrecognised is no opinion.** An outcome outside the three
       known strings is discarded rather than coerced — a model that answers
       ``"ALLOW "`` or ``"probably fine"`` must not be able to reach a verdict by
       accident.
    4. **A failing advisor is an absent advisor.** Exceptions are swallowed on
       purpose. An advisor is not in the trusted computing base for
       authorization, so it must not be in it for availability either: one that
       crashes leaves exactly the decision policy alone would have made.

    Together these give three checkable properties. *Safety*: the effective
    outcome is always at least as strong as the policy outcome. *Degradation*:
    an absent, broken or silent advisor yields the policy decision unchanged.
    *Idempotent absence*: removing advisors entirely changes no authorization
    outcome, only precision.

    Returns ``(effective_outcome, model_outcome, audit_line)``.
    """
    if advisor is None or policy_outcome != ApprovalOutcome.ALLOW:
        return policy_outcome, None, None

    try:
        advice = advisor.advise(action)
    except Exception:  # noqa: BLE001 - see constraint 4 above
        return policy_outcome, None, "control model unavailable; policy decision stands"

    # `isinstance` before the membership test, not for tidiness: `x in _ORDER`
    # hashes `x`, and an advisor returning a list or a dict would raise
    # TypeError *inside the reference monitor*. A control model must not be able
    # to crash the thing that governs it, whatever it returns.
    if not isinstance(advice, str) or advice not in _ORDER:
        return policy_outcome, None, None

    if _ORDER[advice] <= _ORDER[policy_outcome]:
        # The model agreed, or was more permissive than policy. Either way the
        # policy decision stands unchanged -- recorded so a reviewer can see the
        # model was asked and did not move the outcome.
        return policy_outcome, advice, f"control model advised {advice}; no escalation"

    return advice, advice, f"control model escalated {policy_outcome} -> {advice}"


__all__ = [
    "SANDBOXED_TAG",
    "Action",
    "ApprovalDecision",
    "ApprovalOutcome",
    "ControlAdvisor",
    "ControlPolicy",
    "approve",
]
