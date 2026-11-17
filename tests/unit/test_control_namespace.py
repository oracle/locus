# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The 2.0 control-runtime rename.

``locus.control`` is the canonical home for the domain-neutral control core,
including the grounding / verification layer. The security-domain tooling is no
longer part of Locus, and the ``locus.security`` import path that forwarded to
``locus.control`` was removed in 6.0. The old security-coded names
(``SecurityPolicy``, ``Finding``, ``Verdict``, ``SecureAgent``, ``secure_agent``,
``SecurityProfile``) were removed outright (no shims).
"""

import importlib

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


def test_old_names_are_gone() -> None:
    import locus.control as c

    for old in (
        "SecurityPolicy",
        "Finding",
        "Verdict",
        "SecureAgent",
        "secure_agent",
        "SecurityProfile",
    ):
        assert not hasattr(c, old), f"old name {old!r} leaked onto locus.control"


def test_the_old_import_path_is_gone() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("locus.security")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("locus.security.policy")


def test_top_level_lazy_names_are_the_control_objects() -> None:
    import locus
    import locus.control as c

    assert locus.Evidence is c.Evidence
    assert locus.Severity is c.Severity
    assert locus.ground_finding is c.ground_finding
