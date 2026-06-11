# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Insecure inter-agent communication.

Threat: in a multi-agent system, one agent accepts a task or a "finding"
from an unauthenticated peer, or ships data to an untrusted destination
agent. A spoofed peer injects work; a compromised peer exfiltrates.

Defense (SDK primitive + pattern): Tulip's A2A server (tulip.a2a.A2AServer)
authenticates every peer with a bearer token and binds loopback-only when
unauthenticated. On top of that, gate inbound peers and outbound
destinations against a trusted-identity allowlist before acting.

Taxonomy: OWASP ASI07 (Insecure Inter-Agent Communication) · MITRE ATLAS
AML.T0086 (Exfiltration via AI Agent Tool Invocation).
"""

from __future__ import annotations


# The agents this node will exchange work with — its trust boundary.
_TRUSTED_PEERS = frozenset({"soc-orchestrator", "intel-mesh", "ir-commander"})


def accept_from(peer_id: str) -> bool:
    """Accept inbound work only from an authenticated, allowlisted peer."""
    return peer_id in _TRUSTED_PEERS


def send_to(peer_id: str) -> bool:
    """Ship data only to an allowlisted destination agent."""
    return peer_id in _TRUSTED_PEERS


def main() -> None:
    print("Scenario: insecure inter-agent communication  [ASI07 · AML.T0086]\n")

    print("Inbound peer messages (after A2A bearer auth, gate on identity):")
    for peer in ("soc-orchestrator", "unknown-agent-7f3a", "intel-mesh"):
        verdict = "ACCEPT" if accept_from(peer) else "REJECT"
        print(f"  [{verdict}] task from {peer}")

    print("\nOutbound data (exfiltration guard — where may findings go?):")
    for peer in ("ir-commander", "external-pastebin-bot"):
        verdict = "SEND" if send_to(peer) else "REJECT"
        print(f"  [{verdict}] forward findings to {peer}")

    print("\nA2AServer authenticates the peer; the allowlist decides who is in the mesh.")


if __name__ == "__main__":
    main()
