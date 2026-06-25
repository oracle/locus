# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Tulip control runtime — let an agent act, on your terms.

The domain-neutral control core. Wrap any side-effecting :class:`Action` your
agent already takes — move money, change infrastructure, grant access, isolate a
host — and it runs only after it clears a :class:`ControlPolicy` you write
(:func:`admit`): low-risk actions proceed, the ones that matter hold for a human,
and every decision lands on a tamper-evident :class:`AuditTrail`. The gate is
code that runs *before* the action, not a rule in the prompt — so a model that is
fooled still cannot act outside your policy.

This is framework- and domain-agnostic. Security (SOC, EDR, identity) is the
first proven domain — see :mod:`tulip.security` for the grounded findings,
verification, red-team probes, and SOC tooling that build on this core — but the
control runtime applies to any agent that takes real actions.

    from tulip.control import Action, ControlPolicy, AuditTrail, admit, AdmissionError

    policy = ControlPolicy()        # conservative defaults: production → human
    trail = AuditTrail()

    refund = Action(
        name="refund", asset="cust:4821",
        blast_radius=1, kind="payment", environment="production",
    )
    try:
        await admit(refund, lambda: issue_refund("cust:4821"), policy=policy, trail=trail)
    except AdmissionError as e:
        print(e.decision.outcome)   # -> "require_human"; the refund did NOT run

``Evidence`` and ``VerificationResult`` (produced by the grounding/verification
layer in :mod:`tulip.security`) are re-exported here for typing the optional
evidence an admission decision can weigh.
"""

from tulip.security.admit import AdmissionError, admit
from tulip.security.audit import AuditRecord, AuditTrail
from tulip.security.findings import Evidence
from tulip.security.policy import (
    Action,
    ApprovalDecision,
    ApprovalOutcome,
    ControlPolicy,
    approve,
)
from tulip.security.secure import (
    AuditHook,
    GovernanceProfile,
    GovernedAgent,
    governed_agent,
)
from tulip.security.taxonomy import Severity
from tulip.security.verify import VerificationResult, verify


__all__ = [
    # Admission control — the runtime's enforcement point
    "admit",
    "AdmissionError",
    # Policy + approval — safe-before-action
    "Action",
    "ControlPolicy",
    "approve",
    "ApprovalDecision",
    "ApprovalOutcome",
    # Tamper-evident audit
    "AuditTrail",
    "AuditRecord",
    "AuditHook",
    # Governed-by-default agent wrapper
    "GovernedAgent",
    "governed_agent",
    "GovernanceProfile",
    # Optional evidence/verification an admission can weigh (home: tulip.security)
    "Evidence",
    "VerificationResult",
    "verify",
    "Severity",
]
