#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 81: Incident response with a tamper-evident audit chain.

Scenario
────────
A production outage pages the on-call at 2 AM.  A ``GovernedAgent`` runs the
``sre_incident_runbook`` (detect → triage → mitigate → recover).  Every tool
call — metrics query, log tail, deploy rollback — is logged to an immutable
``AuditTrail`` via ``AuditHook``.

After the incident concludes, ``trail.verify()`` confirms the chain has not
been tampered with, and ``trail.export_jsonl()`` produces a portable,
defensible record of the response for:
  - SOC 2 / ISO 27001 change management (immutable log of mitigation actions)
  - Blameless postmortem (timeline of response decisions)
  - Compliance / audit hold (every action timestamped and hash-chained)

Why this is not a toy
─────────────────────
AI agents making remediation decisions (rolling back a deploy, draining a
node) without an immutable audit log are a liability.  If the agent is later
questioned — "why was this release rolled back at 2 AM?" — "the AI decided" is
not a defensible answer.  This demo shows that the SDK provides:

  1. ``GovernanceProfile(audit=True)`` — all tool calls go through ``AuditHook``
  2. ``AuditTrail.verify()`` — cryptographic integrity before export
  3. ``AuditTrail.export_jsonl()`` — portable output for postmortem / audit hold

The local ``sre_incident_runbook`` pins the agent to the four response phases;
``ops_toolset(allow_mitigation=True)`` opts in to write-capable tools
(``rollback_deploy``).  Without ``allow_mitigation=True``, mitigation tools are
absent and the agent operates read-only.

Run:
    python examples/notebook_81_ir_audit_trail.py

    # With a live provider (OpenAI):
    TULIP_MODEL_PROVIDER=openai OPENAI_API_KEY=sk-... \\
        python examples/notebook_81_ir_audit_trail.py
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path


# Allow running from repo root: python examples/notebook_81_ir_audit_trail.py
sys.path.insert(0, str(Path(__file__).parent))
from config import get_model  # noqa: E402

from tulip.control import (
    AuditHook,
    AuditTrail,
    GovernanceProfile,
    Severity,
    governed_agent,
)
from tulip.tools import tool


# ---------------------------------------------------------------------------
# Simulated production outage context
# ---------------------------------------------------------------------------

_INCIDENT_BRIEF = """
PRODUCTION OUTAGE — SEVERITY: CRITICAL (SEV-1)

Timestamp : 2026-06-15T02:14:37Z
Alert ID  : INC-20260615-0042
Source    : Prometheus Alertmanager (on-call paged via PagerDuty)

Summary:
  Service checkout-api (prod, us-east-1) is degrading hard:
  - p99 latency 4.8s (SLO 300ms), error rate 37% and climbing
  - 5xx surge began ~6 min after deploy of checkout-api:1.9.0
  - DB connection pool exhausted on orders-db-primary
  - Cascading 503s into cart-api and payments-api (shared upstream)

Affected services:
  - checkout-api (prod) — confirmed degraded, suspected bad release
  - orders-db-primary  — connection pool saturated
  - cart-api, payments-api — collateral 503s from upstream timeouts

Initial signals:
  - Deploy   : checkout-api 1.8.2 -> 1.9.0 at 02:08:11Z (canary skipped)
  - Metric   : db_pool_in_use{db="orders-db-primary"} = 100/100
  - Log      : "FATAL: remaining connection slots are reserved" (x3.2k/min)

Requested action: Follow the SRE incident runbook.
"""


# ---------------------------------------------------------------------------
# A tiny local runbook — the four SRE response phases.
#
# This stands in for whatever runbook engine you already use (a wiki page, a
# PagerDuty workflow, an internal playbook service). Each phase is just an id
# plus a one-line description the agent is asked to execute. Nothing here is
# security-specific; it is a plain dataclass so the example stays self-contained.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase:
    id: str
    description: str


@dataclass(frozen=True)
class Runbook:
    id: str
    steps: list[Phase] = field(default_factory=list)


def sre_incident_runbook() -> Runbook:
    return Runbook(
        id="sre_incident_runbook",
        steps=[
            Phase("detect", "Confirm the alert is real and scope the blast radius."),
            Phase("triage", "Find the proximate cause; correlate the deploy with the metrics."),
            Phase("mitigate", "Stop the bleeding — roll back the bad release or shed load."),
            Phase("recover", "Verify SLOs have recovered and the system is stable."),
        ],
    )


# ---------------------------------------------------------------------------
# Ops toolset. Read-only telemetry tools are always available; write-capable
# mitigation tools (rollback_deploy) are gated behind allow_mitigation, mirror-
# ing how you would never hand an unattended agent a destructive verb by default.
#
# Every tool here is a deterministic offline stub returning canned data, so the
# notebook runs with no cluster, no creds, and no network.
# ---------------------------------------------------------------------------


@tool
def query_metrics(service: str) -> str:
    """Query the metrics backend (Prometheus) for a service's golden signals."""
    table = {
        "checkout-api": "p99=4.8s err=37% rps=1200 db_pool_in_use=100/100 (degraded)",
        "orders-db-primary": "connections=100/100 wait_queue=812 (pool exhausted)",
        "cart-api": "p99=2.1s err=11% (collateral 503s from upstream)",
    }
    return table.get(service, f"{service}: within SLO, no anomalies")


@tool
def tail_logs(service: str) -> str:
    """Tail the most recent error logs for a service."""
    table = {
        "checkout-api": "FATAL: remaining connection slots are reserved (x3.2k/min) since 02:08Z",
        "orders-db-primary": "too many clients already; max_connections=100 reached",
    }
    return table.get(service, f"{service}: no errors in the last 5 minutes")


@tool
def describe_deploy(service: str) -> str:
    """Describe the current and previous rollout for a service."""
    table = {
        "checkout-api": (
            "current=checkout-api:1.9.0 (rolled 02:08Z, canary skipped); "
            "previous=checkout-api:1.8.2 (stable 6d); ready 12/12"
        ),
    }
    return table.get(service, f"{service}: no recent rollouts")


def rollback_deploy_tool() -> object:
    """Write-capable mitigation tool — only handed to the agent on opt-in."""

    @tool
    def rollback_deploy(service: str, to_revision: str) -> str:
        """Roll a service back to a known-good revision (mitigation action)."""
        # Real version: `kubectl rollout undo deploy/<service> --to-revision=…`.
        return f"rolled back {service} to {to_revision}; new pods healthy 12/12"

    return rollback_deploy


def ops_toolset(*, allow_mitigation: bool = False) -> list[object]:
    """Assemble the ops toolset. Read-only by default; mitigation on opt-in."""
    tools: list[object] = [query_metrics, tail_logs, describe_deploy]
    if allow_mitigation:
        tools.append(rollback_deploy_tool())
    return tools


# ---------------------------------------------------------------------------
# Model — uses get_model() from config.py (MockModel by default, or live
# provider via TULIP_MODEL_PROVIDER env var)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Incident-response runner
# ---------------------------------------------------------------------------


async def run_incident_response(incident: str) -> None:
    trail = AuditTrail()
    # Input guardrails are off here because the incident brief arrives from
    # Alertmanager (trusted internal source), not from an untrusted user.
    # Output guardrails (PII, injection in model output) remain on via grounding.
    profile = GovernanceProfile(grounding=True, guardrails=False, audit=True)
    hook = AuditHook(trail)

    runbook = sre_incident_runbook()
    tools = ops_toolset(allow_mitigation=True)  # opt-in: enables rollback_deploy

    agent = governed_agent(
        model=get_model(),
        tools=tools,
        system_prompt=(
            "You are a senior site-reliability engineer on call. Follow the SRE "
            "incident runbook exactly. Document every action and cite your evidence."
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
            "services": ["checkout-api", "orders-db-primary", "cart-api", "payments-api"],
        },
    )

    print("== Incident Response: SRE incident runbook ==")
    print(f"Runbook: {runbook.id} ({len(runbook.steps)} phases)\n")

    # Run through each runbook phase
    for step in runbook.steps:
        print(f"--- Phase: {step.id} ---")
        prompt = f"{incident}\n\nExecute phase: {step.id}\n{step.description}"
        result = await agent.agent.arun(prompt)
        text = result.text if hasattr(result, "text") else str(result)
        print(text[:400] + ("..." if len(text) > 400 else ""))
        trail.record(f"phase-{step.id}", {"output_preview": str(result)[:200]})
        print()

    trail.record("incident-end", {"incident_id": "INC-20260615-0042", "outcome": "recovered"})

    # Verify chain integrity before export
    ok = trail.verify()
    assert ok, "CRITICAL: Audit trail integrity check failed — chain has been tampered with!"

    print("== Audit trail ==")
    records = trail.records()
    print(f"   {len(records)} records  |  head: {trail.head[:16]}...  |  integrity: OK\n")

    # Export JSONL for postmortem / audit hold
    jsonl = trail.export_jsonl()
    lines = jsonl.strip().split("\n")
    print(f"   JSONL export: {len(lines)} lines")
    print("   (Archive for SOC 2, ISO 27001 change-management, postmortem evidence)")
    print()

    # Show first and last records as sample
    import json

    first = json.loads(lines[0])
    last = json.loads(lines[-1])
    print(f"   First record: seq={first['seq']}  type={first['event_type']}  ts={first['ts']}")
    print(f"   Last record : seq={last['seq']}  type={last['event_type']}  ts={last['ts']}")
    print(f"   Chain link  : last.prev_hash = {last['prev_hash'][:16]}...")


async def main() -> None:
    await run_incident_response(_INCIDENT_BRIEF)


if __name__ == "__main__":
    asyncio.run(main())
