# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Curated, runnable IR/SOC playbooks wired to the security adapter tools.

Each function returns a fresh :class:`~tulip.playbooks.Playbook` whose steps
name tools from :func:`tulip.security.security_toolset` in
``expected_tools``. Hand one to ``Agent(playbook=...)`` (or
``create_soc_analyst``) and the :class:`~tulip.playbooks.PlaybookEnforcer`
pins the investigation to its steps in order, recording any violations.

The tool names referenced here (``query_siem``, ``enrich_indicator``,
``fetch_host_timeline``, ``list_detections``, ``isolate_host``,
``scan_dependencies``, ``scan_endpoint``, ``fingerprint_endpoint``,
``describe_aws``, ``use_aws``, ``submit_posture``) match the adapter tools so
a playbook and its toolset stay in lockstep.
"""

from __future__ import annotations

from tulip.playbooks import Playbook, PlaybookStep


def phishing_triage() -> Playbook:
    """Reported phishing → gather → enrich → scope → (optional) contain."""
    return Playbook(
        id="phishing_triage",
        name="Phishing triage",
        description="Triage a reported phishing message: pull events, enrich the "
        "indicators, scope affected hosts, and contain on confirmation.",
        steps=[
            PlaybookStep(
                id="gather",
                description="Pull the events around the reported message.",
                expected_tools=["query_siem"],
                hints=["Search the alert window for the sender/URL/attachment."],
            ),
            PlaybookStep(
                id="enrich",
                description="Enrich the indicators (sender domain, URL, attachment hash).",
                expected_tools=["enrich_indicator"],
                hints=["A newly-registered lookalike domain is a strong phishing signal."],
            ),
            PlaybookStep(
                id="scope",
                description="Check which hosts interacted with the indicators.",
                expected_tools=["fetch_host_timeline", "list_detections"],
            ),
            PlaybookStep(
                id="contain",
                description="Isolate confirmed-compromised hosts (only on confirmation).",
                expected_tools=["isolate_host"],
                required=False,
                hints=["Only contain when enrichment + telemetry agree it is malicious."],
            ),
        ],
        allow_extra_tools=True,
        tags=["soc", "phishing", "triage"],
    )


def nist_800_61_ir() -> Playbook:
    """NIST SP 800-61 incident-response lifecycle over the adapter tools."""
    return Playbook(
        id="nist_800_61_ir",
        name="Incident response (NIST 800-61)",
        description="Detection & analysis → containment → eradication & recovery → "
        "post-incident, mapped onto the SIEM/EDR/intel tools.",
        steps=[
            PlaybookStep(
                id="detect",
                description="Detection & analysis: gather telemetry and open detections.",
                expected_tools=["query_siem", "list_detections"],
            ),
            PlaybookStep(
                id="analyze",
                description="Enrich indicators and reconstruct the host timeline.",
                expected_tools=["enrich_indicator", "fetch_host_timeline"],
            ),
            PlaybookStep(
                id="contain",
                description="Containment: isolate the affected host(s).",
                expected_tools=["isolate_host"],
                hints=["Containment is a write — confirm scope before isolating."],
            ),
            PlaybookStep(
                id="recover",
                description="Eradication & recovery: verify the exposure is closed.",
                expected_tools=["scan_endpoint"],
                required=False,
            ),
            PlaybookStep(
                id="report",
                description="Post-incident: write the typed incident report.",
                expected_tools=[],
                hints=["Summarise root cause, scope, actions taken, and lessons."],
            ),
        ],
        allow_extra_tools=True,
        tags=["incident-response", "nist-800-61"],
    )


def ransomware_containment() -> Playbook:
    """Ransomware: confirm → contain fast → assess scope → preserve evidence."""
    return Playbook(
        id="ransomware_containment",
        name="Ransomware containment",
        description="Containment-first playbook: confirm the detection, isolate the "
        "host quickly, then assess lateral movement and preserve the timeline.",
        steps=[
            PlaybookStep(
                id="confirm",
                description="Confirm the detection on the host.",
                expected_tools=["list_detections", "fetch_host_timeline"],
            ),
            PlaybookStep(
                id="contain",
                description="Isolate the affected host(s) — speed matters.",
                expected_tools=["isolate_host"],
            ),
            PlaybookStep(
                id="assess_scope",
                description="Hunt for lateral movement from the host.",
                expected_tools=["query_siem"],
                hints=["Look for SMB / remote-service creation from the contained host."],
            ),
            PlaybookStep(
                id="preserve",
                description="Preserve the forensic timeline for the report.",
                expected_tools=["fetch_host_timeline"],
            ),
        ],
        allow_extra_tools=True,
        tags=["incident-response", "ransomware", "containment"],
    )


def cloud_posture_audit() -> Playbook:
    """Read-only cloud-posture audit: discover → gather evidence → submit."""
    return Playbook(
        id="cloud_posture_audit",
        name="Cloud posture audit",
        description="Map the account from the API spec, gather read-only evidence, "
        "and submit a grounded posture report.",
        steps=[
            PlaybookStep(
                id="discover",
                description="Discover the account's services and read-only operations.",
                expected_tools=["describe_aws"],
            ),
            PlaybookStep(
                id="gather",
                description="Gather evidence with read-only API calls.",
                expected_tools=["use_aws"],
                hints=["Cite the exact API fact behind every proposed finding."],
            ),
            PlaybookStep(
                id="report",
                description="Submit the grounded posture report.",
                expected_tools=["submit_posture"],
            ),
        ],
        allow_extra_tools=True,
        tags=["cloud", "posture", "audit"],
    )


def all_playbooks() -> dict[str, Playbook]:
    """Every bundled security playbook, keyed by id."""
    return {
        pb.id: pb
        for pb in (
            phishing_triage(),
            nist_800_61_ir(),
            ransomware_containment(),
            cloud_posture_audit(),
        )
    }


__all__ = [
    "all_playbooks",
    "cloud_posture_audit",
    "nist_800_61_ir",
    "phishing_triage",
    "ransomware_containment",
]
