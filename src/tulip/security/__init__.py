# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Security layer for Tulip — evidence-grounded findings.

The primitive that makes Tulip the cybersecurity agent SDK: a security
:class:`Finding` can only be produced from an evidence partition that
clears the GSAR grounding threshold. Ungrounded claims abstain instead of
shipping, so an agent's findings are trustworthy by construction.

Example usage:

    from tulip.security import Severity, ground_finding, is_finding
    from tulip.reasoning.gsar import Claim, EvidenceType, Partition

    partition = Partition(
        grounded=[
            Claim(
                text="TLS certificate on the endpoint expired",
                type=EvidenceType.TOOL_MATCH,
                evidence_refs=["tool:scan_endpoint:tls_expiry=2026-01-02"],
            ),
        ],
    )
    result = ground_finding(
        title="Expired TLS certificate on 192.0.2.10:443",
        description="The serving endpoint presents an expired certificate.",
        severity=Severity.HIGH,
        asset="192.0.2.10:443",
        remediation="Rotate the certificate and enforce automated renewal.",
        partition=partition,
    )
    if is_finding(result):
        print(result.title, result.gsar_score)
    else:
        print("withheld:", result.reason)
"""

from tulip.security.aws import (
    READONLY_PREFIXES,
    aws_services,
    describe_aws,
    describe_aws_tool,
    is_readonly_operation,
    use_aws,
    use_aws_tool,
)
from tulip.security.findings import (
    Confidence,
    Finding,
    FingerprintClassifier,
    FingerprintFinding,
    FingerprintVerdict,
    Indicator,
)
from tulip.security.grounded import (
    Abstention,
    GroundedFinding,
    ground_finding,
    ground_fingerprint,
    is_finding,
)
from tulip.security.soc import (
    PostureEvidence,
    PostureFinding,
    PostureReport,
    SecurityControls,
    create_soc_analyst,
    ground_report,
    submit_posture,
)
from tulip.security.taxonomy import (
    SEVERITY_ORDER,
    AtlasTechnique,
    IndicatorType,
    OwaspASI,
    OwaspLLM,
    Severity,
    TaxonomyTag,
    severity_at_least,
)


__all__ = [
    # Schemas
    "Confidence",
    "Finding",
    "Indicator",
    "FingerprintClassifier",
    "FingerprintFinding",
    "FingerprintVerdict",
    # Grounding bridge
    "Abstention",
    "GroundedFinding",
    "ground_finding",
    "ground_fingerprint",
    "is_finding",
    # Taxonomy
    "SEVERITY_ORDER",
    "AtlasTechnique",
    "IndicatorType",
    "OwaspASI",
    "OwaspLLM",
    "Severity",
    "TaxonomyTag",
    "severity_at_least",
    # AWS posture (spec-driven, read-only)
    "READONLY_PREFIXES",
    "aws_services",
    "describe_aws",
    "describe_aws_tool",
    "is_readonly_operation",
    "use_aws",
    "use_aws_tool",
    # SOC-analyst factory + grounded posture reporting
    "PostureEvidence",
    "PostureFinding",
    "PostureReport",
    "SecurityControls",
    "create_soc_analyst",
    "ground_report",
    "submit_posture",
]
