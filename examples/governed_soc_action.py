#!/usr/bin/env python
"""Governed SOC action — what a bare LLM+wrapper structurally CANNOT do.

A frontier model can *say* "isolate the prod host." It cannot, on its own:
  (1) be *prevented* from doing it by policy (risk-band + human approval), and
  (2) leave a *tamper-evident, replayable* record proving every decision.

That is Tulip's moat: not intelligence (rented from the model), but CONTROL —
admission gates + an enforced audit chain. This demo is deliberately offline /
model-free to prove the value is the SDK's, not the model's.

  grounded finding -> verify -> propose action -> admit() gate -> execute|HOLD
                                                              -> tamper-evident audit
"""
from __future__ import annotations

import asyncio
from dataclasses import replace

from tulip.reasoning.gsar import Partition
from tulip.security import (
    Action,
    AdmissionError,
    AuditTrail,
    SecurityPolicy,
    Severity,
    admit,
    ground_finding,
    is_finding,
    tool_match,
    verify,
)


def grounded_finding():
    """A real, tool-backed finding (2 refs so verification stays high)."""
    part = Partition(grounded=[
        tool_match("VirusTotal: 198.51.100.5 flagged malicious by 7/90 engines", "tool:vt:malicious=7"),
        tool_match("EDR: ws-7 beaconed to 198.51.100.5 42x in 10m", "tool:edr:beacon_count=42"),
    ])
    return ground_finding(
        title="Malicious C2 beaconing to 198.51.100.5",
        description="Host ws-7 is beaconing to a known-malicious IP.",
        severity=Severity.HIGH, asset="ws-7",
        remediation="Contain ws-7 and block the IP.", partition=part,
    )


async def main() -> None:
    policy = SecurityPolicy()  # defaults: verify>=0.80, blast<=1, production->human
    trail = AuditTrail()

    f = grounded_finding()
    assert is_finding(f)
    verdict = await verify(f)
    print(f"finding: grounded (gsar={f.gsar_score:.2f}) | verified (survives={verdict.survives}, "
          f"confidence={verdict.confidence:.2f})\n")

    executed: dict[str, bool] = {}

    async def do(action_name: str, asset: str):
        executed[action_name] = True
        return {"executed": action_name, "asset": asset}

    # A) Contained, non-prod, blast=1 -> auto-ALLOW (the safe path runs).
    a = Action(name="block_ip", asset="198.51.100.5", blast_radius=1, environment="staging", kind="network-block")
    try:
        res = await admit(a, lambda: do(a.name, a.asset), policy=policy, finding=f, verdict=verdict, trail=trail)
        print(f"[A] block_ip (staging, blast 1) -> ALLOWED, executed: {res}")
    except AdmissionError as e:
        print(f"[A] block_ip -> {e.decision.outcome}: {e.decision.reason}")

    # B) Isolate a PRODUCTION host, blast=50 -> REQUIRE_HUMAN -> HELD (NOT executed).
    b = Action(name="isolate_host", asset="prod-db-01", blast_radius=50, environment="production", kind="containment")
    try:
        await admit(b, lambda: do(b.name, b.asset), policy=policy, finding=f, verdict=verdict, trail=trail)
        print("[B] isolate_host EXECUTED  <-- should NOT happen")
    except AdmissionError as e:
        print(f"[B] isolate_host (production, blast 50) -> HELD ({e.decision.outcome}): {e.decision.reason}")
        print(f"    >>> prod host actually isolated? {executed.get('isolate_host', False)}  "
              "— the agent could NOT take the dangerous action without a human")

    # C) Hard-denied label (irreversible) -> DENY.
    deny_policy = SecurityPolicy(deny_for=frozenset({"irreversible"}))
    c = Action(name="wipe_disk", asset="ws-7", environment="staging", tags=frozenset({"irreversible"}))
    try:
        await admit(c, lambda: do(c.name, c.asset), policy=deny_policy, finding=f, verdict=verdict, trail=trail)
    except AdmissionError as e:
        print(f"[C] wipe_disk -> DENIED: {e.decision.reason}")

    # The audit chain: tamper-evident + replayable.
    print(f"\naudit: {len(trail)} decisions recorded | chain intact: {trail.verify()}")
    recs = trail.records()
    recs[1] = replace(recs[1], payload={**recs[1].payload, "outcome": "allow"})  # forge "HELD"->"allow"
    forged = AuditTrail.from_records(recs)
    print(f"forge one decision (require_human -> allow) and re-check: chain intact: {forged.verify()}  "
          "<-- tampering detected")
    print("\n--- replayable audit trail (JSONL, SIEM-ready) ---")
    print(trail.export_jsonl())
    print("\nMoat in one line: a bare LLM would have just called isolate_host on prod.")
    print("Tulip held it for a human and left a record you can't forge.")


if __name__ == "__main__":
    asyncio.run(main())
