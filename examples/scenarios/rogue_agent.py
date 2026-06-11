# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Rogue agents — an agent acting outside its declared mandate.

Threat: an agent (compromised, mis-prompted, or simply over-eager) starts
doing things outside its job — a triage agent reaching into payroll, a
research agent calling a containment tool. The danger is the *deviation*,
and if no one is watching, it goes unnoticed.

Defense (SDK primitive + pattern): declare each agent's mandate (its
allowed tool scope), enforce it with a guardrail allowlist, AND emit every
out-of-mandate attempt to the audit trail (Tulip's hooks/EventBus) so a
human can review the deviation. Enforcement stops the action; the audit
record is how you *detect* a rogue agent over time.

Taxonomy: OWASP ASI10 (Rogue Agents).
"""

from __future__ import annotations


# Each agent's mandate: the tool scope it is authorised to use.
_MANDATE: dict[str, frozenset[str]] = {
    "triage": frozenset({"enrich_indicator", "query_siem", "get_alert"}),
    "research": frozenset({"web_search", "fetch_doc", "summarize"}),
}


def check_mandate(agent: str, tool: str, audit: list[dict[str, str]]) -> bool:
    """Enforce the agent's mandate and record every decision for review."""
    allowed = tool in _MANDATE.get(agent, frozenset())
    audit.append(
        {"agent": agent, "tool": tool, "decision": "allow" if allowed else "DENY-OUT-OF-MANDATE"}
    )
    return allowed


def main() -> None:
    print("Scenario: rogue agents  [ASI10]\n")

    audit: list[dict[str, str]] = []
    calls = [
        ("triage", "query_siem"),  # in mandate
        ("triage", "wire_transfer"),  # rogue: out of mandate
        ("research", "fetch_doc"),  # in mandate
        ("research", "disable_account"),  # rogue: out of mandate
    ]
    for agent, tool in calls:
        ok = check_mandate(agent, tool, audit)
        print(f"  [{'ALLOW' if ok else 'BLOCK'}] {agent} → {tool}")

    print("\nAudit trail (what a reviewer sees):")
    for entry in audit:
        flag = "  ⚠ " if entry["decision"].startswith("DENY") else "    "
        print(f"{flag}{entry['agent']:>8} · {entry['tool']:<16} · {entry['decision']}")

    deviations = [e for e in audit if e["decision"].startswith("DENY")]
    print(
        f"\n{len(deviations)} out-of-mandate attempt(s) flagged for review — that's how you spot a rogue agent."
    )


if __name__ == "__main__":
    main()
