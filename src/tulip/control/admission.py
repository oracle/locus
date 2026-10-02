# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Admission control — the runtime's enforcement point for side-effecting actions.

A trust *library* offers grounding, verification, and policy as functions you may
call. A trust *runtime* makes them **mandatory**: a side-effecting action runs only
after it has cleared the chain — evidence → verification → policy → approval — and
the decision is recorded so execution is auditable.

:func:`admit` is that gate — the Kubernetes-admission-controller analog for agent
actions. ``perform`` fires only if :func:`~tulip.control.policy.approve` returns
ALLOW; otherwise it raises :class:`AdmissionError`. Either way the decision is
appended to the audit trail, so there is **no un-recorded path to a side effect**::

    from tulip.control import Action, AuditTrail, ControlPolicy, admit, verify

    trail = AuditTrail()
    verdict = await verify(finding)
    await admit(
        Action(name="disable_user", asset="mallory@corp", environment="production"),
        lambda: ctx.identity.disable("mallory@corp"),  # the side effect
        policy=ControlPolicy(),
        finding=finding,
        verdict=verdict,
        trail=trail,
    )
    # production label -> require_human -> AdmissionError, and the attempt is on the trail.

:func:`admit_sync` is the same gate for a side effect that is not a coroutine —
a tool body that writes a file or runs a process, called on a worker thread
with no event loop of its own. Both share one decision-and-record path, so the
two can never disagree about what was admitted or what was written down.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from tulip.control.audit import AuditTrail
from tulip.control.findings import Evidence
from tulip.control.policy import (
    Action,
    ApprovalDecision,
    ApprovalOutcome,
    ControlPolicy,
    approve,
)
from tulip.control.verification import VerificationResult


if TYPE_CHECKING:
    from tulip.control.spend import SpendLedger


T = TypeVar("T")


class AdmissionError(Exception):
    """A side-effecting action failed admission — it did not clear the trust chain.

    Carries the :class:`~tulip.control.policy.ApprovalDecision` so the caller can
    route a ``require_human`` hold to an approver or surface a ``deny`` reason.
    """

    def __init__(self, decision: ApprovalDecision) -> None:
        self.decision = decision
        super().__init__(
            f"action {decision.action.name!r} not admitted ({decision.outcome}): {decision.reason}"
        )


async def admit(
    action: Action,
    perform: Callable[[], Awaitable[T]],
    *,
    policy: ControlPolicy,
    finding: Evidence | None = None,
    verdict: VerificationResult | None = None,
    trail: AuditTrail | None = None,
    approved_by: str | None = None,
    ledger: SpendLedger | None = None,
    spend_scope: str = "default",
    context: Mapping[str, Any] | None = None,
) -> T:
    """Run ``perform`` only if ``action`` clears the trust chain; else reject.

    The mandatory gate that turns the composable chain into an enforced one:

    1. :func:`~tulip.control.policy.approve` weighs the action against the evidence
       (``finding``), the verification (``verdict``), and the ``policy``.
    2. The decision is recorded to ``trail`` (if given) — admitted or not — so no
       side effect is un-audited.
    3. On ALLOW, ``perform`` is awaited and its result returned. On require_human or
       deny, :class:`AdmissionError` is raised with the decision attached.

    Args:
        action: The proposed side-effecting action.
        perform: A zero-arg async callable that performs the side effect.
        policy: The governing :class:`~tulip.control.policy.ControlPolicy`.
        finding: The evidence the action responds to.
        verdict: The :func:`~tulip.control.verification.verify` result.
        trail: An :class:`~tulip.control.audit.AuditTrail` to record the decision on.
        approved_by: Who approved a ``require_human`` hold. With it, the held action
            runs and the approver is recorded on the trail. A ``deny`` still raises:
            no person approves past a denial.
        ledger: Where ``spend_scope``'s cumulative spend is read before the
            decision and ``action.cost_usd`` recorded after ``perform`` succeeds.
        spend_scope: The scope the spend counts against: a customer, a tenant.
        context: What the caller knows about this decision that the action
            does not carry — the rule that matched, the mode, who is acting, a
            summary of the arguments. Recorded on the trail under ``context``.

    Returns:
        Whatever ``perform`` returns.

    Raises:
        AdmissionError: if the action is not admitted (deny, or require_human
            without ``approved_by``).
    """
    _decide(
        action,
        policy=policy,
        finding=finding,
        verdict=verdict,
        trail=trail,
        approved_by=approved_by,
        ledger=ledger,
        spend_scope=spend_scope,
        context=context,
    )
    result = await perform()
    _spend(action, ledger, spend_scope)
    return result


def admit_sync(
    action: Action,
    perform: Callable[[], T],
    *,
    policy: ControlPolicy,
    finding: Evidence | None = None,
    verdict: VerificationResult | None = None,
    trail: AuditTrail | None = None,
    approved_by: str | None = None,
    ledger: SpendLedger | None = None,
    spend_scope: str = "default",
    context: Mapping[str, Any] | None = None,
) -> T:
    """:func:`admit` for a synchronous side effect.

    Same arguments, same decision, same record; ``perform`` is a plain
    zero-argument callable. For tool bodies that run on a worker thread, where
    there is no event loop to await :func:`admit` on.

    Raises:
        AdmissionError: if the action is not admitted.
    """
    _decide(
        action,
        policy=policy,
        finding=finding,
        verdict=verdict,
        trail=trail,
        approved_by=approved_by,
        ledger=ledger,
        spend_scope=spend_scope,
        context=context,
    )
    result = perform()
    _spend(action, ledger, spend_scope)
    return result


def _decide(  # noqa: PLR0913 — admit()'s arguments, minus the side effect
    action: Action,
    *,
    policy: ControlPolicy,
    finding: Evidence | None,
    verdict: VerificationResult | None,
    trail: AuditTrail | None,
    approved_by: str | None,
    ledger: SpendLedger | None,
    spend_scope: str,
    context: Mapping[str, Any] | None,
) -> ApprovalDecision:
    """Decide, record, and raise unless admitted. The one path both gates share."""
    spent = ledger.spent(spend_scope) if ledger is not None else 0.0
    decision = approve(action, policy=policy, finding=finding, verdict=verdict, spent_usd=spent)
    human = approved_by is not None and decision.outcome == ApprovalOutcome.REQUIRE_HUMAN
    if trail is not None:
        entry: dict[str, Any] = {
            "action": action.name,
            "asset": action.asset,
            "outcome": decision.outcome,
            "reason": decision.reason,
        }
        if ledger is not None:
            entry.update(
                {"cost_usd": action.cost_usd, "spent_usd": spent, "spend_scope": spend_scope}
            )
        if human:
            entry["approved_by"] = approved_by
        if context:
            entry["context"] = dict(context)
        trail.record("action-admission", entry)
    if not (decision.allowed or human):
        raise AdmissionError(decision)
    return decision


def _spend(action: Action, ledger: SpendLedger | None, spend_scope: str) -> None:
    # Only after it ran: a refused or failed action spends nothing.
    if ledger is not None and action.cost_usd:
        ledger.record(spend_scope, action.cost_usd, action=action.name)


__all__ = ["AdmissionError", "admit", "admit_sync"]
