# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Policy + approval — safe-before-action, the way a CISO reasons about it.

An agent that *finds* something still must not *act* on its own authority. A
`ControlPolicy` encodes the rules a security org actually has — how strong the
verification must be, how large a blast radius may auto-proceed, and which
actions always need a human — and :func:`approve` weighs a proposed
:class:`Action` against the **evidence** (the grounded `Evidence`), the
**verification** (the :class:`~locus.control.verification.VerificationResult`), and the policy
to return an :class:`ApprovalDecision`: allow, require a human, or deny.

This is the "safe before action" half of the trust quartet (grounded · verified
· auditable · safe-before-action). The richer commercial approval/governance
engine builds on this open contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

from locus.control.findings import Evidence
from locus.control.taxonomy import Severity, severity_at_least
from locus.control.verification import VerificationResult


if TYPE_CHECKING:
    import os


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
      :class:`~locus.tools.sandbox.SandboxEnforcerHook`.
    - ``version``: a label for this policy. When set it is bound into approval
      ids, so a decision made under one version is never redeemed under another.
    - ``require_human_over_usd``: an action whose ``cost_usd`` is above this needs a
      human.
    - ``spend_limit_usd``: an action that would take its scope's cumulative spend
      past this is denied — no approval overrides it. The caller supplies what the
      scope has spent (``spent_usd``), typically from a spend ledger.

    A policy is also a document. :meth:`load` reads one from a YAML or JSON
    file and :meth:`dump` writes one, so the rules can be reviewed and
    versioned as text; :meth:`content_hash` names the exact rules an admission
    was decided under and is recorded on every trail entry
    (see :mod:`locus.control.policy_file`).
    """

    require_verification_score: float = 0.8
    max_blast_radius: int = 1
    require_human_for: frozenset[str] = field(default_factory=lambda: frozenset({"production"}))
    deny_for: frozenset[str] = field(default_factory=frozenset)
    min_severity: Severity = Severity.LOW
    require_sandbox_for: frozenset[str] = field(default_factory=frozenset)
    version: str = ""
    require_human_over_usd: float | None = None
    spend_limit_usd: float | None = None

    @classmethod
    def load(cls, source: str | os.PathLike[str]) -> ControlPolicy:
        """The policy in a YAML or JSON file, or in a document passed as text.

        A path, or a string holding the document itself (one with a newline,
        or starting with ``{``). The format follows the file suffix, else the
        text. Unknown keys and badly typed values raise
        :class:`~locus.control.policy_file.PolicyFileError` naming the key.
        YAML needs the optional ``pyyaml`` package; JSON needs nothing.
        """
        from locus.control.policy_file import load_policy  # noqa: PLC0415 — avoids a cycle

        return load_policy(source)

    @classmethod
    def loads(cls, text: str, *, format: Literal["yaml", "json"] | None = None) -> ControlPolicy:  # noqa: A002
        """The policy in ``text``, a YAML or JSON document."""
        from locus.control.policy_file import loads_policy  # noqa: PLC0415 — avoids a cycle

        return loads_policy(text, format=format)

    def dump(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        format: Literal["yaml", "json"] | None = None,  # noqa: A002
    ) -> str:
        """This policy as a document, written to ``path`` when one is given.

        ``format`` defaults to the path's suffix, else YAML. Every rule is
        written, defaults included, so a reviewer reads the whole policy and
        ``ControlPolicy.load(policy.dump())`` gives back an equal policy.
        """
        from locus.control.policy_file import dump_policy  # noqa: PLC0415 — avoids a cycle

        return dump_policy(self, path, format=format)

    def content_hash(self) -> str:
        """SHA-256 over the rules, the same for equal policies however they were built."""
        from locus.control.policy_file import policy_hash  # noqa: PLC0415 — avoids a cycle

        return policy_hash(self)


class PolicySource(Protocol):
    """Something that hands out the policy to decide under *right now*.

    :func:`~locus.control.admission.admit` and
    :func:`~locus.control.gate.gate_tool` take one wherever they take a
    :class:`ControlPolicy`, and ask it once per admission, so a policy that
    changes between decisions (:class:`~locus.control.policy_file.PolicyWatcher`
    re-reading a file) is never half-swapped inside one.
    """

    def current(self) -> ControlPolicy:  # pragma: no cover - protocol
        """The policy in force now."""
        ...


def current_policy(policy: ControlPolicy | PolicySource) -> ControlPolicy:
    """``policy`` itself, or the one a :class:`PolicySource` holds now."""
    return policy if isinstance(policy, ControlPolicy) else policy.current()


@dataclass(frozen=True)
class Action:
    """A proposed response action, with the attributes policy reasons over."""

    name: str
    asset: str = ""
    blast_radius: int = 1
    environment: str = "unknown"
    kind: str = ""
    tags: frozenset[str] = field(default_factory=frozenset)
    #: What performing the action spends, in USD; weighed by the spend limits.
    cost_usd: float = 0.0

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
    spent_usd: float = 0.0,
) -> ApprovalDecision:
    """Decide whether ``action`` may proceed: allow / require_human / deny.

    Weighs every rule and returns the **strongest** triggered outcome (deny >
    require_human > allow), recording each check that fired so the decision is
    auditable.

    Args:
        action: The proposed action.
        policy: The governing :class:`ControlPolicy`.
        finding: The evidence the action responds to (optional).
        verdict: The :func:`~locus.control.verification.verify` result (optional, but
            auto-allow needs one that clears the policy bar).
        spent_usd: What the action's scope has already spent, for
            ``policy.spend_limit_usd``.
        advisor: An optional trained control model. It may only raise the
            decision toward caution — see :func:`_combine`. Omitting it, or
            passing one that fails, yields exactly the decision policy alone
            would have made.

    Returns:
        An :class:`ApprovalDecision`.
    """
    triggered = [
        (check.outcome, check.detail)
        for check in evaluate_rules(
            action, policy=policy, finding=finding, verdict=verdict, spent_usd=spent_usd
        )
        if check.outcome is not None
    ]

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


@dataclass(frozen=True)
class RuleCheck:
    """One policy rule weighed against an action: whether it fired, and why.

    ``rule`` is the :class:`ControlPolicy` field, ``outcome`` the result it
    contributed (``None`` when it did not fire) and ``detail`` the sentence an
    audit reader sees. :func:`approve` keeps the fired ones as its ``checks``;
    :func:`~locus.control.policy_lint.explain` shows all of them.
    """

    rule: str
    outcome: str | None
    detail: str

    @property
    def fired(self) -> bool:
        """Whether the rule contributed an outcome."""
        return self.outcome is not None


def evaluate_rules(
    action: Action,
    *,
    policy: ControlPolicy,
    finding: Evidence | None = None,
    verdict: VerificationResult | None = None,
    spent_usd: float = 0.0,
) -> list[RuleCheck]:
    """Every rule of ``policy`` weighed against ``action``, in the order they apply.

    The one evaluation :func:`approve` and ``locus policy explain`` share: the
    checks that fired, with the same wording, plus the rules that did not and
    why, so a decision path can be shown without a second reading of the rules.
    """
    labels = action.labels()
    checks: list[RuleCheck] = []

    def add(rule: str, outcome: str | None, detail: str) -> None:
        checks.append(RuleCheck(rule=rule, outcome=outcome, detail=detail))

    denied = labels & policy.deny_for
    if denied:
        add("deny_for", ApprovalOutcome.DENY, f"labels {sorted(denied)} are denied by policy")
    else:
        add("deny_for", None, "no label of the action is denied")

    unsandboxed = labels & policy.require_sandbox_for
    if unsandboxed and SANDBOXED_TAG not in labels:
        add(
            "require_sandbox_for",
            ApprovalOutcome.DENY,
            f"labels {sorted(unsandboxed)} require sandboxed execution "
            f"(the action carries no {SANDBOXED_TAG!r} tag)",
        )
    elif unsandboxed:
        add(
            "require_sandbox_for",
            None,
            f"labels {sorted(unsandboxed)} require sandboxed execution and the action is sandboxed",
        )
    else:
        add("require_sandbox_for", None, "no label of the action requires a sandbox")

    if finding is None:
        add("min_severity", None, "no finding supplied")
    elif not severity_at_least(finding.severity, policy.min_severity):
        add(
            "min_severity",
            ApprovalOutcome.DENY,
            f"finding severity {finding.severity.value} is below the policy "
            f"minimum {policy.min_severity.value}",
        )
    else:
        add(
            "min_severity",
            None,
            f"finding severity {finding.severity.value} meets the minimum "
            f"{policy.min_severity.value}",
        )

    checks.append(_verification_check(policy, verdict))

    if action.blast_radius > policy.max_blast_radius:
        add(
            "max_blast_radius",
            ApprovalOutcome.REQUIRE_HUMAN,
            f"blast radius {action.blast_radius} exceeds the maximum {policy.max_blast_radius}",
        )
    else:
        add(
            "max_blast_radius",
            None,
            f"blast radius {action.blast_radius} is within the maximum {policy.max_blast_radius}",
        )

    if policy.require_human_over_usd is None:
        add("require_human_over_usd", None, "not set")
    elif action.cost_usd > policy.require_human_over_usd:
        add(
            "require_human_over_usd",
            ApprovalOutcome.REQUIRE_HUMAN,
            f"cost ${action.cost_usd:,.2f} exceeds the per-action limit "
            f"${policy.require_human_over_usd:,.2f}",
        )
    else:
        add(
            "require_human_over_usd",
            None,
            f"cost ${action.cost_usd:,.2f} is within the per-action limit "
            f"${policy.require_human_over_usd:,.2f}",
        )

    if policy.spend_limit_usd is None:
        add("spend_limit_usd", None, "not set")
    elif spent_usd + action.cost_usd > policy.spend_limit_usd:
        add(
            "spend_limit_usd",
            ApprovalOutcome.DENY,
            f"spend ${spent_usd:,.2f} + ${action.cost_usd:,.2f} would exceed the spend "
            f"limit ${policy.spend_limit_usd:,.2f}",
        )
    else:
        add(
            "spend_limit_usd",
            None,
            f"spend ${spent_usd:,.2f} + ${action.cost_usd:,.2f} stays within the spend "
            f"limit ${policy.spend_limit_usd:,.2f}",
        )

    needs_human = labels & policy.require_human_for
    if needs_human:
        add(
            "require_human_for",
            ApprovalOutcome.REQUIRE_HUMAN,
            f"labels {sorted(needs_human)} require human approval",
        )
    else:
        add("require_human_for", None, "no label of the action requires human approval")

    return checks


def _verification_check(policy: ControlPolicy, verdict: VerificationResult | None) -> RuleCheck:
    rule = "require_verification_score"
    if policy.require_verification_score <= 0:
        return RuleCheck(rule, None, "the bar is 0: no verification required")
    if verdict is None:
        return RuleCheck(rule, ApprovalOutcome.REQUIRE_HUMAN, "no verification provided")
    if not verdict.survives:
        return RuleCheck(rule, ApprovalOutcome.DENY, "the finding did not survive verification")
    if verdict.confidence < policy.require_verification_score:
        return RuleCheck(
            rule,
            ApprovalOutcome.REQUIRE_HUMAN,
            f"verification confidence {verdict.confidence:.2f} is below the bar "
            f"{policy.require_verification_score:.2f}",
        )
    return RuleCheck(
        rule,
        None,
        f"verification confidence {verdict.confidence:.2f} meets the bar "
        f"{policy.require_verification_score:.2f}",
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
    "PolicySource",
    "RuleCheck",
    "approve",
    "current_policy",
    "evaluate_rules",
]
