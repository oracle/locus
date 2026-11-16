# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Locus control runtime — let an agent act, on your terms.

The domain-neutral control core. Wrap any side-effecting :class:`Action` your
agent already takes — move money, change infrastructure, grant access, isolate a
host — and it runs only after it clears a :class:`ControlPolicy` you write
(:func:`admit`): low-risk actions proceed, the ones that matter hold for a human,
and every decision lands on a tamper-evident :class:`AuditTrail`. The gate is
code that runs *before* the action, not a rule in the prompt — so a model that is
fooled still cannot act outside your policy.

This is framework- and domain-agnostic: the control runtime applies to any
agent that takes real actions.

    from locus.control import Action, ControlPolicy, AuditTrail, admit, AdmissionError

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

from locus.control.action import (
    UNDETERMINED_TAG,
    ActionSpec,
    DerivedLabels,
    action_from_labels,
    asset_from_args,
    default_action,
    derive_labels,
    resolve_action,
)
from locus.control.admission import AdmissionError, admit, admit_sync
from locus.control.approvals import (
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
from locus.control.approvals_oracle import OracleApprovals
from locus.control.approvals_postgres import PostgresApprovals
from locus.control.approvals_redis import RedisApprovals
from locus.control.approvals_spec import approval_store, store_spec
from locus.control.audit import (
    AuditRecord,
    AuditReport,
    AuditSigner,
    AuditTrail,
    Ed25519Signer,
    check_jsonl,
    verify_jsonl,
)
from locus.control.findings import (
    Confidence,
    Evidence,
    FingerprintClassifier,
    FingerprintFinding,
    FingerprintVerdict,
    Indicator,
)
from locus.control.gate import ApprovalBridge, gate_tool
from locus.control.governed import (
    AuditHook,
    GovernanceProfile,
    GovernedAgent,
    governed_agent,
)
from locus.control.grounded import (
    Abstention,
    GroundedFinding,
    ground_finding,
    ground_fingerprint,
    is_finding,
)
from locus.control.holds import PendingHold, holds_of, pending_holds
from locus.control.permissions import (
    VERDICT_POLICY,
    PermissionMatch,
    PermissionRule,
    PermissionRules,
    verdict_action,
    verdict_tag,
)
from locus.control.policy import (
    SANDBOXED_TAG,
    Action,
    ApprovalDecision,
    ApprovalOutcome,
    ControlPolicy,
    approve,
)
from locus.control.shell import ShellCommand, SimpleCommand, parse_command
from locus.control.spend import FileSpendLedger, InMemorySpendLedger, SpendLedger
from locus.control.taxonomy import (
    SEVERITY_ORDER,
    AtlasTechnique,
    IndicatorType,
    OwaspASI,
    OwaspLLM,
    Severity,
    TaxonomyTag,
    severity_at_least,
)
from locus.control.verification import (
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
    "OracleApprovals",
    "PostgresApprovals",
    "RedisApprovals",
    "approval_store",
    "store_spec",
    # What a paused thread's checkpoint says it waits on
    "PendingHold",
    "holds_of",
    "pending_holds",
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
    "admit_sync",
    "AdmissionError",
    # Operator-written allow / ask / deny rules, fed to admission as labels
    "PermissionRule",
    "PermissionRules",
    "PermissionMatch",
    "VERDICT_POLICY",
    "verdict_action",
    "verdict_tag",
    # What a shell line runs, for rules and gates that decide about commands
    "parse_command",
    "ShellCommand",
    "SimpleCommand",
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
    "AuditReport",
    "check_jsonl",
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
