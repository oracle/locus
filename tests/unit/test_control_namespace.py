# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The 2.0 control-runtime rename.

``locus.control`` is the canonical home for the domain-neutral control core,
including the grounding / verification layer. The security-domain tooling is no
longer part of Locus; ``locus.security`` is now only a compatibility shim (see
test_security_shim.py). The old security-coded names
(``SecurityPolicy``, ``Finding``, ``Verdict``, ``SecureAgent``, ``secure_agent``,
``SecurityProfile``) were removed outright (no shims).
"""

import pytest


def test_control_exposes_the_core() -> None:
    from locus.control import (  # noqa: F401
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
    import locus.security as s

    # Pure control symbols moved to locus.control (none of these are submodules).
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
        assert not hasattr(s, name), f"locus.security still exposes control symbol {name!r}"


def test_old_names_are_gone_everywhere() -> None:
    import locus.control as c
    import locus.security as s

    for old in (
        "SecurityPolicy",
        "Finding",
        "Verdict",
        "SecureAgent",
        "secure_agent",
        "SecurityProfile",
    ):
        assert not hasattr(s, old), f"old name {old!r} still on locus.security"
        assert not hasattr(c, old), f"old name {old!r} leaked onto locus.control"


def test_old_import_paths_raise() -> None:
    with pytest.raises(ImportError):
        from locus.security import SecurityPolicy  # noqa: F401
    with pytest.raises(ImportError):
        # relocated to locus.control — must not resolve from locus.security
        from locus.security import ControlPolicy  # noqa: F401


def test_evidence_and_verification_are_shared_objects() -> None:
    # Re-exported in both namespaces — must be the SAME object, not a copy.
    import locus.control as c
    import locus.security as s

    assert c.Evidence is s.Evidence
    assert c.VerificationResult is s.VerificationResult
    assert c.verify is s.verify
