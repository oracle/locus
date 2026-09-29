# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Tulip control runtime — let an agent act, on your terms.

The domain-neutral control core. Wrap any side-effecting :class:`Action` your
agent already takes — move money, change infrastructure, grant access, isolate a
host — and it runs only after it clears a :class:`ControlPolicy` you write
(:func:`admit`): low-risk actions proceed, the ones that matter hold for a human,
and every decision lands on a tamper-evident :class:`AuditTrail`. The gate is
code that runs *before* the action, not a rule in the prompt — so a model that is
fooled still cannot act outside your policy.

This is framework- and domain-agnostic: the control runtime applies to any
agent that takes real actions. Domain tooling built on this core ships as
separate, opt-in distributions (the security-domain tooling is
``tulip-agents-security``, imported as :mod:`tulip_security`).

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

The grounding and verification layer an admission decision can weigh lives here
too: :func:`ground_finding` ships a typed :class:`Evidence` only when its claims
clear the GSAR threshold (otherwise an auditable :class:`Abstention`), and
:func:`verify` challenges a finding before it drives an action.
"""

from tulip.control.action import (
    UNDETERMINED_TAG,
    ActionSpec,
    DerivedLabels,
    action_from_labels,
    asset_from_args,
    default_action,
    derive_labels,
    resolve_action,
)
from tulip.control.admission import AdmissionError, admit
from tulip.control.approvals import (
    ApprovalAuthority,
    ApprovalAuthorityError,
    ApprovalRecord,
    ApprovalStore,
    ApproverRule,
    Delegation,
    FileApprovals,
    InMemoryApprovals,
    call_digest,
)
from tulip.control.audit import (
    AuditRecord,
    AuditSigner,
    AuditTrail,
    Ed25519Signer,
    verify_jsonl,
)
from tulip.control.findings import (
    Confidence,
    Evidence,
    FingerprintClassifier,
    FingerprintFinding,
    FingerprintVerdict,
    Indicator,
)
from tulip.control.gate import ApprovalBridge, gate_tool
from tulip.control.governed import (
    AuditHook,
    GovernanceProfile,
    GovernedAgent,
    governed_agent,
)
from tulip.control.grounded import (
    Abstention,
    GroundedFinding,
    ground_finding,
    ground_fingerprint,
    is_finding,
)
from tulip.control.policy import (
    SANDBOXED_TAG,
    Action,
    ApprovalDecision,
    ApprovalOutcome,
    ControlPolicy,
    approve,
)
from tulip.control.spend import FileSpendLedger, InMemorySpendLedger, SpendLedger
from tulip.control.taxonomy import (
    SEVERITY_ORDER,
    AtlasTechnique,
    IndicatorType,
    OwaspASI,
    OwaspLLM,
    Severity,
    TaxonomyTag,
    severity_at_least,
)
from tulip.control.verification import (
    AdversarialSkeptic,
    EvidenceQualitySkeptic,
    Refutation,
    Skeptic,
    VerificationResult,
    verify,
)


__all__ = [
    # Deriving the Action a policy is weighed against
    "ActionSpec",
    "action_from_labels",
    "asset_from_args",
    "default_action",
    "ApprovalBridge",
    "gate_tool",
    # Cumulative spend per scope, for the policy's spend limits
    "FileSpendLedger",
    "InMemorySpendLedger",
    "SpendLedger",
    # Held calls that wait for a person, across a restart
    "ApprovalRecord",
    "ApprovalStore",
    "FileApprovals",
    # Who may decide a held call
    "ApprovalAuthority",
    "ApprovalAuthorityError",
    "ApproverRule",
    "Delegation",
    "InMemoryApprovals",
    "call_digest",
    "resolve_action",
    # Argument-derived labels (declarative rules on a tool definition)
    "DerivedLabels",
    "derive_labels",
    "UNDETERMINED_TAG",
    # Admission control — the runtime's enforcement point
    "admit",
    "AdmissionError",
    # Policy + approval — safe-before-action
    "Action",
    "ControlPolicy",
    "approve",
    "ApprovalDecision",
    "ApprovalOutcome",
    "SANDBOXED_TAG",
    # Tamper-evident audit
    "AuditTrail",
    "AuditRecord",
    "AuditSigner",
    "Ed25519Signer",
    "verify_jsonl",
    "AuditHook",
    # Governed-by-default agent wrapper
    "GovernedAgent",
    "governed_agent",
    "GovernanceProfile",
    # Grounded findings — ship only what the evidence supports
    "Evidence",
    "Indicator",
    "Confidence",
    "Abstention",
    "GroundedFinding",
    "ground_finding",
    "ground_fingerprint",
    "is_finding",
    "FingerprintClassifier",
    "FingerprintFinding",
    "FingerprintVerdict",
    # Verification — the independent challenge a finding must survive
    "verify",
    "VerificationResult",
    "Refutation",
    "Skeptic",
    "EvidenceQualitySkeptic",
    "AdversarialSkeptic",
    # Severity + taxonomy tags a finding carries
    "Severity",
    "SEVERITY_ORDER",
    "severity_at_least",
    "IndicatorType",
    "AtlasTechnique",
    "OwaspLLM",
    "OwaspASI",
    "TaxonomyTag",
]
