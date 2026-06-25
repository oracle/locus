# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Admission control — the runtime's enforcement point for side-effecting actions.

A trust *library* offers grounding, verification, and policy as functions you may
call. A trust *runtime* makes them **mandatory**: a side-effecting action runs only
after it has cleared the chain — evidence → verification → policy → approval — and
the decision is recorded so execution is auditable.

:func:`admit` is that gate — the Kubernetes-admission-controller analog for agent
actions. ``perform`` fires only if :func:`~tulip.security.policy.approve` returns
ALLOW; otherwise it raises :class:`AdmissionError`. Either way the decision is
appended to the audit trail, so there is **no un-recorded path to a side effect**::

    from tulip.control import Action, ControlPolicy, admit, AuditTrail
    from tulip.security import verify

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
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from tulip.security.audit import AuditTrail
from tulip.security.findings import Evidence
from tulip.security.policy import Action, ApprovalDecision, ControlPolicy, approve
from tulip.security.verify import VerificationResult


T = TypeVar("T")


class AdmissionError(Exception):
    """A side-effecting action failed admission — it did not clear the trust chain.

    Carries the :class:`~tulip.security.policy.ApprovalDecision` so the caller can
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
) -> T:
    """Run ``perform`` only if ``action`` clears the trust chain; else reject.

    The mandatory gate that turns the composable chain into an enforced one:

    1. :func:`~tulip.security.policy.approve` weighs the action against the evidence
       (``finding``), the verification (``verdict``), and the ``policy``.
    2. The decision is recorded to ``trail`` (if given) — admitted or not — so no
       side effect is un-audited.
    3. On ALLOW, ``perform`` is awaited and its result returned. On require_human or
       deny, :class:`AdmissionError` is raised with the decision attached.

    Args:
        action: The proposed side-effecting action.
        perform: A zero-arg async callable that performs the side effect.
        policy: The governing :class:`~tulip.security.policy.ControlPolicy`.
        finding: The evidence the action responds to.
        verdict: The :func:`~tulip.security.verify.verify` result.
        trail: An :class:`~tulip.security.audit.AuditTrail` to record the decision on.

    Returns:
        Whatever ``perform`` returns.

    Raises:
        AdmissionError: if the action is not admitted (require_human or deny).
    """
    decision = approve(action, policy=policy, finding=finding, verdict=verdict)
    if trail is not None:
        trail.record(
            "action-admission",
            {
                "action": action.name,
                "asset": action.asset,
                "outcome": decision.outcome,
                "reason": decision.reason,
            },
        )
    if not decision.allowed:
        raise AdmissionError(decision)
    return await perform()


__all__ = ["AdmissionError", "admit"]
