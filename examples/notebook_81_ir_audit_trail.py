#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 81: Incident response with a tamper-evident audit chain.

Scenario
────────
A ransomware alert fires at 2 AM.  A ``SecureAgent`` runs the
``nist_800_61_r3`` IR playbook (detect → contain → eradicate → recover).
Every tool call — host isolation, SIEM query, IOC enrichment — is logged to
an immutable ``AuditTrail`` via ``AuditHook``.

After the response concludes, ``trail.verify()`` confirms the chain has not
been tampered with, and ``trail.export_jsonl()`` produces SIEM-ingestible,
legally defensible evidence for:
  - SOC 2 / ISO 27001 audit requirements (immutable log of containment actions)
  - NIS2 incident reporting (timeline of response decisions)
  - Legal hold (every action timestamped and hash-chained)

Why this is not a toy
─────────────────────
AI agents making containment decisions (isolating a host, blocking a domain)
without an immutable audit log are a liability.  If the agent is later
questioned — "why was this server taken offline?" — "the AI decided" is not
a defensible answer.  This demo shows that the SDK provides:

  1. ``SecurityProfile(audit=True)`` — all tool calls go through ``AuditHook``
  2. ``AuditTrail.verify()`` — cryptographic integrity before export
  3. ``AuditTrail.export_jsonl()`` — SIEM-ready output for legal hold

The ``nist_800_61_ir`` playbook pins the agent to the four NIST IR phases;
``security_toolset(allow_containment=True)`` opts in to write-capable tools
(``isolate_host``).  Without ``allow_containment=True``, containment tools
are absent and the agent operates read-only.

Run:
    python examples/notebook_81_ir_audit_trail.py

    # With a live provider (OpenAI):
    TULIP_MODEL_PROVIDER=openai OPENAI_API_KEY=sk-... \\
        python examples/notebook_81_ir_audit_trail.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


# Allow running from repo root: python examples/notebook_81_ir_audit_trail.py
sys.path.insert(0, str(Path(__file__).parent))
from config import get_model  # noqa: E402

from tulip.security import (
    AuditHook,
    AuditTrail,
    SecurityProfile,
    Severity,
    nist_800_61_ir,
    secure_agent,
    security_toolset,
)
from tulip.security.taxonomy import AtlasTechnique, OwaspASI


# ---------------------------------------------------------------------------
# Simulated ransomware incident context
# ---------------------------------------------------------------------------

_INCIDENT_BRIEF = """
RANSOMWARE ALERT — SEVERITY: CRITICAL

Timestamp : 2026-06-15T02:14:37Z
Alert ID  : INC-20260615-0042
Source    : CrowdStrike Falcon (EDR)

Summary:
  Host ACCT-SRV-01 (10.10.5.22) is exhibiting ransomware behaviour:
  - vssadmin.exe deleting shadow copies
  - Rapid mass file encryption (*.docx, *.xlsx) in \\\\FILESERVER\\shared
  - Outbound C2 beacon to 198.51.100.88 (unclassified)
  - Lateral movement attempt to PAYROLL-DB-01 (blocked at firewall)

Affected assets:
  - ACCT-SRV-01 (10.10.5.22) — confirmed infected
  - FILESERVER (10.10.5.30)  — shares encrypted
  - PAYROLL-DB-01 (10.10.5.100) — attempted, blocked

Initial IOCs:
  - Process : vssadmin.exe, ransomware_payload.exe (SHA256: a3f1...)
  - Network  : 198.51.100.88 (C2), 198.51.100.99 (exfil endpoint suspected)
  - File     : \\WINDOWS\\Temp\\ransomware_payload.exe

Requested action: Follow NIST SP 800-61 IR procedure.
"""


# ---------------------------------------------------------------------------
# Model — uses get_model() from config.py (MockModel by default, or live
# provider via TULIP_MODEL_PROVIDER env var)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# IR runner
# ---------------------------------------------------------------------------


async def run_incident_response(incident: str) -> None:
    trail = AuditTrail()
    # Input guardrails are off here because the incident brief arrives from
    # the SIEM (trusted internal source), not from an untrusted user.
    # Output guardrails (PII, injection in model output) remain on via grounding.
    profile = SecurityProfile(grounding=True, guardrails=False, audit=True)
    hook = AuditHook(trail)

    playbook = nist_800_61_ir()
    tools = security_toolset(
        siem=True,
        edr=True,
        threat_intel=True,
        scanner=False,
        fingerprint=False,
        aws=False,
        allow_containment=True,  # opt-in: enables isolate_host
    )

    agent = secure_agent(
        model=get_model(),
        tools=tools,
        system_prompt=(
            "You are a senior incident responder. Follow the NIST SP 800-61 "
            "playbook exactly. Document every action and cite your evidence."
        ),
        profile=profile,
        audit_trail=trail,
        hooks=[hook],
    )

    trail.record(
        "incident-start",
        {
            "incident_id": "INC-20260615-0042",
            "severity": Severity.CRITICAL.value,
            "taxonomy": [str(AtlasTechnique.EXTERNAL_HARMS), str(OwaspASI.CASCADING_FAILURES)],
        },
    )

    print("== Incident Response: NIST SP 800-61 playbook ==")
    print(f"Playbook: {playbook.id} ({len(playbook.steps)} phases)\n")

    # Run through each playbook phase
    for step in playbook.steps:
        print(f"--- Phase: {step.id} ---")
        prompt = f"{incident}\n\nExecute phase: {step.id}\n{step.description}"
        result = agent.run_sync(prompt)
        text = result.text if hasattr(result, "text") else str(result)
        print(text[:400] + ("..." if len(text) > 400 else ""))
        trail.record(f"phase-{step.id}", {"output_preview": str(result)[:200]})
        print()

    trail.record("incident-end", {"incident_id": "INC-20260615-0042", "outcome": "contained"})

    # Verify chain integrity before export
    ok = trail.verify()
    assert ok, "CRITICAL: Audit trail integrity check failed — chain has been tampered with!"

    print("== Audit trail ==")
    records = trail.records()
    print(f"   {len(records)} records  |  head: {trail.head[:16]}...  |  integrity: OK\n")

    # Export JSONL for SIEM ingest / legal hold
    jsonl = trail.export_jsonl()
    lines = jsonl.strip().split("\n")
    print(f"   JSONL export: {len(lines)} lines")
    print("   (Ingest to SIEM / archive for SOC 2, ISO 27001, NIS2 evidence)")
    print()

    # Show first and last records as sample
    import json

    first = json.loads(lines[0])
    last = json.loads(lines[-1])
    print(f"   First record: seq={first['seq']}  type={first['event_type']}  ts={first['ts']}")
    print(f"   Last record : seq={last['seq']}  type={last['event_type']}  ts={last['ts']}")
    print(f"   Chain link  : last.prev_hash = {last['prev_hash'][:16]}...")


if __name__ == "__main__":
    asyncio.run(run_incident_response(_INCIDENT_BRIEF))
