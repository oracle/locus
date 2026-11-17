# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Lint a policy for gaps, and explain what it decides for one action.

A policy that passes review can still have holes a reader does not see:
a hold that a deny always pre-empts, an environment nothing in the policy
mentions, a kind of action that is never put in front of a person. The
``locus policy`` commands put those in front of a CI job and a reviewer::

    locus policy lint policy/refunds.yaml --environments production staging
    locus policy explain policy/refunds.yaml \\
        --action '{"name": "issue_refund", "kind": "payment", "blast_radius": 6}'

:func:`lint_policy` reports three kinds of finding:

- ``unreachable-rule``: a rule that can never decide an outcome. A label in
  ``require_human_for`` that ``deny_for`` also names, a per-action cost hold
  above the spend limit that denies first, a verification bar above 1.0.
- ``uncovered-environment``: a declared environment in which an ordinary
  action (one asset, no cost, no other label) clears the gate on its own.
- ``kind-never-held``: a declared kind of action that is never held for a
  person in any declared environment.

The last two need to know what the deployment has: the environments and
kinds come from the document's ``governs`` block or the command line. They
are found by weighing a probe action with :func:`~locus.control.policy.approve`
itself, so lint cannot disagree with the gate about what passes.

:func:`explain` runs the same evaluation the gate runs and shows every rule,
fired or not, with the outcome and whether a person can approve it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from locus.control.policy import (
    SANDBOXED_TAG,
    Action,
    ApprovalDecision,
    ApprovalOutcome,
    ControlPolicy,
    RuleCheck,
    approve,
    evaluate_rules,
)
from locus.control.policy_file import PolicyDocument, policy_hash
from locus.control.verification import VerificationResult


#: The finding kinds, in the order a report lists them.
UNREACHABLE_RULE = "unreachable-rule"
UNCOVERED_ENVIRONMENT = "uncovered-environment"
KIND_NEVER_HELD = "kind-never-held"

_PROBE = "lint-probe"
_ACTION_KEYS = ("name", "asset", "blast_radius", "environment", "kind", "tags", "cost_usd")


@dataclass(frozen=True)
class LintFinding:
    """One gap in a policy."""

    code: str
    message: str
    #: The :class:`ControlPolicy` field the finding is about, when it is one.
    rule: str | None = None
    #: The environment, kind or label the finding is about, when it is one.
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "rule": self.rule, "label": self.label, "message": self.message}


@dataclass(frozen=True)
class LintReport:
    """What :func:`lint_policy` found. ``ok`` when there are no findings."""

    findings: tuple[LintFinding, ...]
    #: Things worth knowing that are not gaps: what was not checked, what auto-allows.
    notes: tuple[str, ...] = ()
    policy_hash: str = ""
    source: str = ""

    @property
    def ok(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "source": self.source,
            "policy_hash": self.policy_hash,
            "findings": [finding.to_dict() for finding in self.findings],
            "notes": list(self.notes),
        }

    def render(self) -> str:
        """The report as text, one finding per line."""
        head = f"policy {self.source}" if self.source else "policy"
        if self.policy_hash:
            head += f" (sha256 {self.policy_hash[:12]})"
        lines = [head]
        if self.findings:
            lines.append(f"{len(self.findings)} finding(s):")
            lines.extend(f"  {finding.code}: {finding.message}" for finding in self.findings)
        else:
            lines.append("no findings")
        lines.extend(f"note: {note}" for note in self.notes)
        return "\n".join(lines)


def lint_document(
    document: PolicyDocument,
    *,
    environments: Iterable[str] = (),
    kinds: Iterable[str] = (),
) -> LintReport:
    """:func:`lint_policy` for a document, adding what it declares it governs."""
    report = lint_policy(
        document.policy,
        environments={*document.environments, *environments},
        kinds={*document.kinds, *kinds},
    )
    return LintReport(
        findings=report.findings,
        notes=report.notes,
        policy_hash=report.policy_hash,
        source=document.source,
    )


def lint_policy(
    policy: ControlPolicy,
    *,
    environments: Iterable[str] = (),
    kinds: Iterable[str] = (),
) -> LintReport:
    """Report the rules that cannot fire and the labels nothing holds.

    Args:
        policy: The policy to check.
        environments: The environments actions carry in this deployment.
            Each is checked for a rule that holds or denies an ordinary action
            there.
        kinds: The kinds of action in this deployment. Each is checked for a
            hold in at least one environment.
    """
    envs = sorted(set(environments))
    kind_list = sorted(set(kinds))
    findings = [
        *_unreachable(policy),
        *_uncovered_environments(policy, envs),
        *_kinds_never_held(policy, kind_list, envs),
    ]
    notes: list[str] = []
    if not envs and not kind_list:
        notes.append(
            "no environments or kinds declared: add a `governs:` block to the file or "
            "pass --environments / --kinds, and each is checked for a rule"
        )
    if approve(Action(name=_PROBE), policy=policy).allowed:
        notes.append(
            "an action with no environment, kind or tags, one asset and no cost is allowed: "
            "every gated tool must derive labels for this policy to see it"
        )
    return LintReport(findings=tuple(findings), notes=tuple(notes), policy_hash=policy_hash(policy))


def _unreachable(policy: ControlPolicy) -> list[LintFinding]:
    findings: list[LintFinding] = []
    for label in sorted(policy.require_human_for & policy.deny_for):
        findings.append(
            LintFinding(
                UNREACHABLE_RULE,
                f"label {label!r} in require_human_for is unreachable: deny_for denies it first",
                rule="require_human_for",
                label=label,
            )
        )
    for label in sorted(policy.require_sandbox_for & policy.deny_for):
        findings.append(
            LintFinding(
                UNREACHABLE_RULE,
                f"label {label!r} in require_sandbox_for is unreachable: deny_for denies it "
                "outright, sandboxed or not",
                rule="require_sandbox_for",
                label=label,
            )
        )
    if policy.require_sandbox_for and SANDBOXED_TAG in policy.deny_for:
        findings.append(
            LintFinding(
                UNREACHABLE_RULE,
                f"require_sandbox_for can never be satisfied: the {SANDBOXED_TAG!r} tag it "
                "requires is in deny_for",
                rule="require_sandbox_for",
                label=SANDBOXED_TAG,
            )
        )
    if policy.require_verification_score > 1.0:
        findings.append(
            LintFinding(
                UNREACHABLE_RULE,
                f"require_verification_score {policy.require_verification_score} is above 1.0: "
                "no verification confidence can meet it, so every action is held",
                rule="require_verification_score",
            )
        )
    if (
        policy.require_human_over_usd is not None
        and policy.spend_limit_usd is not None
        and policy.require_human_over_usd >= policy.spend_limit_usd
    ):
        findings.append(
            LintFinding(
                UNREACHABLE_RULE,
                f"require_human_over_usd {policy.require_human_over_usd:,.2f} is unreachable: "
                f"any action costing more already exceeds spend_limit_usd "
                f"{policy.spend_limit_usd:,.2f} and is denied",
                rule="require_human_over_usd",
            )
        )
    if policy.max_blast_radius < 1:
        findings.append(
            LintFinding(
                UNREACHABLE_RULE,
                f"max_blast_radius {policy.max_blast_radius} holds every action that touches "
                "an asset: the allow path is unreachable for them",
                rule="max_blast_radius",
            )
        )
    return findings


def _uncovered_environments(policy: ControlPolicy, envs: list[str]) -> list[LintFinding]:
    findings: list[LintFinding] = []
    for env in envs:
        if approve(Action(name=_PROBE, environment=env), policy=policy).allowed:
            findings.append(
                LintFinding(
                    UNCOVERED_ENVIRONMENT,
                    f"environment {env!r} has no rule: an action there with one asset, no "
                    "cost and no other label is allowed without a person",
                    label=env,
                )
            )
    return findings


def _kinds_never_held(
    policy: ControlPolicy, kinds: list[str], envs: list[str]
) -> list[LintFinding]:
    findings: list[LintFinding] = []
    for kind in kinds:
        if kind in policy.deny_for:
            continue  # never held because always denied, which is a rule
        probes = [Action(name=_PROBE, kind=kind, environment=env) for env in envs] or [
            Action(name=_PROBE, kind=kind)
        ]
        if all(approve(probe, policy=policy).allowed for probe in probes):
            where = (
                f"in every declared environment {envs}" if envs else "(no environments declared)"
            )
            findings.append(
                LintFinding(
                    KIND_NEVER_HELD,
                    f"kind {kind!r} is never held: a {kind} action with one asset, no cost "
                    f"and no other label is allowed {where}",
                    label=kind,
                )
            )
    return findings


# --------------------------------------------------------------------------- explain


@dataclass(frozen=True)
class Explanation:
    """The decision path for one action under one policy."""

    action: Action
    decision: ApprovalDecision
    checks: tuple[RuleCheck, ...]
    policy_hash: str
    source: str = ""
    version: str = ""

    @property
    def requires_approval(self) -> bool:
        """Whether a person can let this action run (a hold, not a deny)."""
        return self.decision.outcome == ApprovalOutcome.REQUIRE_HUMAN

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "policy_hash": self.policy_hash,
            "policy_version": self.version,
            "action": action_to_dict(self.action),
            "labels": sorted(self.action.labels()),
            "rules": [
                {"rule": check.rule, "outcome": check.outcome, "detail": check.detail}
                for check in self.checks
            ],
            "outcome": self.decision.outcome,
            "requires_approval": self.requires_approval,
            "reason": self.decision.reason,
        }

    def render(self) -> str:
        """The decision path as text."""
        action = self.action
        head = f"policy   {self.source or '<policy>'} (sha256 {self.policy_hash[:12]})"
        if self.version:
            head += f" version {self.version}"
        lines = [
            head,
            f"action   {action.name} environment={action.environment} kind={action.kind or '-'} "
            f"blast_radius={action.blast_radius} cost_usd={action.cost_usd:,.2f} "
            f"tags={sorted(action.tags)}",
            f"labels   {sorted(action.labels())}",
            "",
        ]
        width = max(len(check.rule) for check in self.checks)
        for check in self.checks:
            mark = check.outcome or "-"
            lines.append(f"  {check.rule.ljust(width)}  {mark.ljust(13)}  {check.detail}")
        lines.append("")
        lines.append(f"outcome  {self.decision.outcome}: {_meaning(self.decision.outcome)}")
        return "\n".join(lines)


def _meaning(outcome: str) -> str:
    if outcome == ApprovalOutcome.ALLOW:
        return "the action runs; no approval needed"
    if outcome == ApprovalOutcome.REQUIRE_HUMAN:
        return "held for a person; runs once someone approves (admit(..., approved_by=...))"
    return "denied; no approval overrides it"


def explain(
    policy: ControlPolicy,
    action: Action,
    *,
    verdict: VerificationResult | None = None,
    spent_usd: float = 0.0,
    source: str = "",
) -> Explanation:
    """Every rule weighed against ``action``, and the decision they add up to.

    The same evaluation :func:`~locus.control.policy.approve` makes, so the
    outcome here is the outcome the gate gives. No control model is
    consulted: this is the deterministic policy, which is what a reviewer
    asks about.
    """
    checks = evaluate_rules(action, policy=policy, verdict=verdict, spent_usd=spent_usd)
    decision = approve(action, policy=policy, verdict=verdict, spent_usd=spent_usd)
    return Explanation(
        action=action,
        decision=decision,
        checks=tuple(checks),
        policy_hash=policy_hash(policy),
        source=source,
        version=policy.version,
    )


def action_from_mapping(data: Mapping[str, Any]) -> Action:
    """An :class:`Action` from its JSON form.

    Raises ``ValueError`` for an unknown key or a missing name and
    ``TypeError`` for a value of the wrong type.
    """
    for key in data:
        if key not in _ACTION_KEYS:
            raise ValueError(f"unknown action key {key!r}; known keys are {list(_ACTION_KEYS)}")
    name = data.get("name")
    if not isinstance(name, str):
        raise TypeError("'name' must be a string")
    if not name:
        raise ValueError("an action needs a non-empty 'name'")
    tags = data.get("tags", [])
    if isinstance(tags, str) or not isinstance(tags, Iterable):
        raise TypeError("'tags' must be a list of strings")
    tags = list(tags)
    if not all(isinstance(tag, str) for tag in tags):
        raise TypeError("'tags' must be a list of strings")
    blast_radius = data.get("blast_radius", 1)
    if isinstance(blast_radius, bool) or not isinstance(blast_radius, int):
        raise TypeError("'blast_radius' must be an integer")
    cost = data.get("cost_usd", 0.0)
    if isinstance(cost, bool) or not isinstance(cost, int | float):
        raise TypeError("'cost_usd' must be a number")
    for key in ("asset", "environment", "kind"):
        if key in data and not isinstance(data[key], str):
            raise TypeError(f"'{key}' must be a string")
    return Action(
        name=name,
        asset=data.get("asset", ""),
        blast_radius=blast_radius,
        environment=data.get("environment", "unknown"),
        kind=data.get("kind", ""),
        tags=frozenset(tags),
        cost_usd=float(cost),
    )


def action_from_json(text: str) -> Action:
    """An :class:`Action` from a JSON object, as ``locus policy explain --action`` takes it."""
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"--action is not valid JSON: {exc}") from exc
    if not isinstance(data, Mapping):
        raise TypeError("--action must be a JSON object")
    return action_from_mapping(data)


def action_to_dict(action: Action) -> dict[str, Any]:
    """The JSON form of an :class:`Action`."""
    return {
        "name": action.name,
        "asset": action.asset,
        "blast_radius": action.blast_radius,
        "environment": action.environment,
        "kind": action.kind,
        "tags": sorted(action.tags),
        "cost_usd": action.cost_usd,
    }


__all__ = [
    "KIND_NEVER_HELD",
    "UNCOVERED_ENVIRONMENT",
    "UNREACHABLE_RULE",
    "Explanation",
    "LintFinding",
    "LintReport",
    "action_from_json",
    "action_from_mapping",
    "action_to_dict",
    "explain",
    "lint_document",
    "lint_policy",
]
