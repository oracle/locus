# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""The 2.0 control-runtime rename.

``tulip.control`` is the canonical home for the domain-neutral control core;
``tulip.security`` keeps the grounded-evidence / red-team / SOC domain. The old
security-coded names (``ControlPolicy``, ``Evidence``, ``VerificationResult``,
``governed_agent``, ``GovernanceProfile``) were removed outright (no shims).
"""

import pytest


def test_control_exposes_the_core() -> None:
    from tulip.control import (  # noqa: F401
        Action,
        AdmissionError,
        ApprovalDecision,
        ApprovalOutcome,
        AuditHook,
        AuditRecord,
        AuditTrail,
        ControlPolicy,
        Evidence,
        GovernanceProfile,
        GovernedAgent,
        Severity,
        VerificationResult,
        admit,
        approve,
        governed_agent,
        verify,
    )


def test_security_no_longer_exports_control_core() -> None:
    import tulip.security as s

    # Pure control symbols moved to tulip.control (none of these are submodules).
    for name in (
        "Action",
        "ControlPolicy",
        "approve",
        "ApprovalDecision",
        "ApprovalOutcome",
        "AuditTrail",
        "AuditRecord",
        "AuditHook",
        "GovernedAgent",
        "governed_agent",
        "GovernanceProfile",
        "AdmissionError",
    ):
        assert not hasattr(s, name), f"tulip.security still exposes control symbol {name!r}"


def test_old_names_are_gone_everywhere() -> None:
    import tulip.control as c
    import tulip.security as s

    for old in (
        "SecurityPolicy",
        "Finding",
        "Verdict",
        "SecureAgent",
        "secure_agent",
        "SecurityProfile",
    ):
        assert not hasattr(s, old), f"old name {old!r} still on tulip.security"
        assert not hasattr(c, old), f"old name {old!r} leaked onto tulip.control"


def test_old_import_paths_raise() -> None:
    with pytest.raises(ImportError):
        from tulip.security import SecurityPolicy  # noqa: F401
    with pytest.raises(ImportError):
        # relocated to tulip.control — must not resolve from tulip.security
        from tulip.security import ControlPolicy  # noqa: F401


def test_security_keeps_its_domain() -> None:
    import tulip.security as s

    for name in (
        "Evidence",
        "verify",
        "VerificationResult",
        "red_team",
        "assure",
        "SecurityContext",
        "ground_finding",
        "Severity",
        "security_toolset",
        "Target",
    ):
        assert hasattr(s, name), f"tulip.security lost domain symbol {name!r}"


def test_evidence_and_verification_are_shared_objects() -> None:
    # Re-exported in both namespaces — must be the SAME object, not a copy.
    import tulip.control as c
    import tulip.security as s

    assert c.Evidence is s.Evidence
    assert c.VerificationResult is s.VerificationResult
    assert c.verify is s.verify
