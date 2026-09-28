# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Tulip security-domain tooling — ``pip install tulip-agents-security``.

The security vertical built on the Tulip control core. The domain-neutral
runtime it stands on — the admission gate, policy, audit trail, governed agent,
and the GSAR grounding / verification layer — ships in ``tulip-agents`` as
:mod:`tulip.control`. This distribution adds the security-domain tooling on
top: the :class:`~tulip_security.adapter.SecurityAdapter` protocol + helper
toolkit, the conformance kit (:mod:`tulip_security.testing`), the
``SecurityContext`` facade, the AI red-team probes and job verbs
(``red_team`` / ``assure`` / ``monitor``), the SOC analyst factory, curated
IR playbooks, and a set of **bundled reference/offline adapters** (intel /
SIEM / EDR / scanner / fingerprint / AWS posture) so it runs standalone with
no credentials. A live vendor adapter is one you write in your own package
against the same contract, and pass in via ``security_toolset(extra=…)``.

The grounding vocabulary (``Evidence``, ``Severity``, ``ground_finding``, …)
is re-exported here from :mod:`tulip.control` so domain code has one import.

Example usage:

    from tulip_security import Severity, ground_finding, is_finding
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

from typing import Any

from tulip.control.findings import (
    Confidence,
    Evidence,
    FingerprintClassifier,
    FingerprintFinding,
    FingerprintVerdict,
    Indicator,
)
from tulip.control.grounded import (
    Abstention,
    GroundedFinding,
    ground_finding,
    ground_fingerprint,
    is_finding,
)
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
from tulip_security.adapter import (
    SecurityAdapter,
    ToolAdapter,
    as_json,
    env,
    indicator_type,
    inference_claim,
    tool_match,
)

# Agentic AI-security surface: a Target (the AI under assessment), the job
# verbs that act on it, the red-team probe library, and posture assessments.
from tulip_security.assess import guardrail_coverage
from tulip_security.aws import (
    READONLY_PREFIXES,
    aws_services,
    describe_aws,
    describe_aws_tool,
    is_readonly_operation,
    use_aws,
    use_aws_tool,
)
from tulip_security.context import (
    ActionsPort,
    CloudSource,
    EndpointSource,
    IdentitySource,
    LogSource,
    SecurityContext,
    ThreatIntelSource,
)
from tulip_security.edr import (
    fetch_host_timeline,
    fetch_host_timeline_tool,
    isolate_host,
    isolate_host_tool,
    list_detections,
    list_detections_tool,
)
from tulip_security.fingerprint import (
    FEATURE_KEYS,
    default_classifier,
    dispatch_timing_probe_reference,
    fingerprint_endpoint_tool,
    fingerprint_to_finding,
    measure_endpoint_timing,
)
from tulip_security.intel import (
    classify_indicator,
    enrich_indicator,
    enrich_indicator_tool,
    enrich_to_finding,
)
from tulip_security.jobs import assure, monitor, red_team
from tulip_security.playbooks import (
    all_playbooks,
    cloud_posture_audit,
    nist_800_61_ir,
    phishing_triage,
    ransomware_containment,
)
from tulip_security.redteam import (
    DirectPromptInjection,
    ExcessiveAgency,
    IndirectPromptInjection,
    Jailbreak,
    Probe,
    ProbeOutcome,
    SensitiveInformationDisclosure,
    UnsandboxedCodeExecution,
    all_probes,
    suite_probes,
)
from tulip_security.scanner import (
    scan_dependencies,
    scan_dependencies_tool,
    scan_endpoint,
    scan_endpoint_to_finding,
    scan_endpoint_tool,
)
from tulip_security.siem import query_siem, siem_query_tool
from tulip_security.soc import (
    PostureEvidence,
    PostureFinding,
    PostureReport,
    SecurityControls,
    create_soc_analyst,
    ground_report,
    submit_posture,
)
from tulip_security.target import Sender, Target


def security_toolset(
    *,
    threat_intel: bool = True,
    siem: bool = True,
    edr: bool = True,
    scanner: bool = True,
    fingerprint: bool = True,
    aws: bool = False,
    allow_containment: bool = False,
    extra: list[Any] | None = None,
) -> list[Any]:
    """Assemble the agent-ready security tool list.

    Returns the read-only SOC triage loop by default — IOC enrichment, SIEM
    search, EDR forensics, vuln/posture scanning, and inference
    fingerprinting — from the **bundled reference adapters**. Containment
    (``isolate_host``) and the AWS posture tools are opt-in (the latter needs
    ``boto3`` from the ``[aws]`` / ``[security]`` extra).

    ``extra`` merges tools from **external integrations** you imported
    explicitly (the LangChain model — no auto-discovery), e.g.::

        from acme_tulip.siem import acme_siem_tool  # your own package

        tools = security_toolset(siem=False, extra=[acme_siem_tool])

    Hand the result to ``Agent(tools=...)`` or ``create_soc_analyst(tools=...)``.
    """
    tools: list[Any] = []
    if threat_intel:
        tools.append(enrich_indicator_tool)
    if siem:
        tools.append(siem_query_tool)
    if edr:
        tools.extend([fetch_host_timeline_tool, list_detections_tool])
        if allow_containment:
            tools.append(isolate_host_tool)
    if scanner:
        tools.extend([scan_dependencies_tool, scan_endpoint_tool])
    if fingerprint:
        tools.append(fingerprint_endpoint_tool)
    if aws:
        tools.extend([describe_aws_tool, use_aws_tool])
    if extra:
        tools.extend(extra)
    return tools


__all__ = [
    # Integration contract + toolkit (the langchain-core boundary)
    "SecurityAdapter",
    "ToolAdapter",
    "as_json",
    "env",
    "indicator_type",
    "inference_claim",
    "tool_match",
    # Agent-ready toolset
    "security_toolset",
    # Threat-intel adapter
    "classify_indicator",
    "enrich_indicator",
    "enrich_indicator_tool",
    "enrich_to_finding",
    # SIEM adapter
    "query_siem",
    "siem_query_tool",
    # EDR adapter
    "fetch_host_timeline",
    "fetch_host_timeline_tool",
    "isolate_host",
    "isolate_host_tool",
    "list_detections",
    "list_detections_tool",
    # Vuln / posture scanner
    "scan_dependencies",
    "scan_dependencies_tool",
    "scan_endpoint",
    "scan_endpoint_to_finding",
    "scan_endpoint_tool",
    # Inference fingerprinting
    "FEATURE_KEYS",
    "default_classifier",
    "dispatch_timing_probe_reference",
    "fingerprint_endpoint_tool",
    "fingerprint_to_finding",
    "measure_endpoint_timing",
    # Curated IR/SOC playbooks
    "all_playbooks",
    "cloud_posture_audit",
    "nist_800_61_ir",
    "phishing_triage",
    "ransomware_containment",
    # Schemas
    "Confidence",
    "Evidence",
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
    # Agentic AI-security: the Target + the job verbs + the red-team probes
    "Target",
    "Sender",
    "red_team",
    "assure",
    "monitor",
    "guardrail_coverage",
    # Verification — independent challenge that prevents security hallucinations
    "verify",
    "VerificationResult",
    "Refutation",
    "Skeptic",
    "EvidenceQualitySkeptic",
    "AdversarialSkeptic",
    # SecurityContext — investigate by domain, not by vendor
    "SecurityContext",
    "LogSource",
    "EndpointSource",
    "IdentitySource",
    "CloudSource",
    "ThreatIntelSource",
    "ActionsPort",
    "Probe",
    "ProbeOutcome",
    "DirectPromptInjection",
    "IndirectPromptInjection",
    "Jailbreak",
    "ExcessiveAgency",
    "SensitiveInformationDisclosure",
    "UnsandboxedCodeExecution",
    "all_probes",
    "suite_probes",
]
