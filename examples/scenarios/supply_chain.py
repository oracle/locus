# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Supply-chain trust: untrusted tools, MCP servers, and model artifacts.

Threat: an agent mounts an MCP server, loads a tool, or runs a model
pulled from an unverified source. A poisoned tool exfiltrates on every
call; a backdoored model misbehaves on a trigger phrase. The compromise
enters through what the agent *depends on*, not through its prompt.

Defense (pattern, with SDK taxonomy): gate every dependency through a
provenance allowlist before it is wired in — only trusted publishers /
pinned digests are admitted, and the decision is recorded as a typed
finding. (Tulip's MCP bridge and tool registry are the wiring points; the
allowlist is the policy you put in front of them.)

Taxonomy: OWASP LLM03 (Supply Chain) · OWASP ASI04 (Agentic Supply Chain) ·
MITRE ATLAS AML.T0110 (AI Agent Tool Poisoning) · AML.T0018 (Backdoor ML
Model).
"""

from __future__ import annotations


# Trusted publishers and pinned model digests — the allowlist the agent
# checks before mounting anything. In production this is signed metadata.
_TRUSTED_PUBLISHERS = frozenset({"tuliplabs", "first-party"})
_PINNED_MODELS = {
    "clusiana-3b-v1": "sha256:9f86d0...",  # the digest you trained and signed
}


def admit_tool(name: str, publisher: str) -> bool:
    """Admit an MCP tool / plugin only from a trusted publisher."""
    return publisher in _TRUSTED_PUBLISHERS


def admit_model(name: str, digest: str) -> bool:
    """Admit a model only when its digest matches the pinned, signed value."""
    return _PINNED_MODELS.get(name) == digest


def main() -> None:
    print("Scenario: supply-chain trust  [LLM03 · ASI04 · AML.T0110 · AML.T0018]\n")

    print("MCP tools / plugins (publisher allowlist):")
    candidates = [
        ("indicator_pivot", "tuliplabs"),  # first-party, trusted
        ("free-osint-scraper", "anon-marketplace"),  # unknown publisher
        ("totally_safe_shell", "anon-marketplace"),  # poisoned tool
    ]
    for name, publisher in candidates:
        verdict = "MOUNT" if admit_tool(name, publisher) else "REJECT"
        print(f"  [{verdict}] {name}  (publisher={publisher})")

    print("\nModel artifacts (pinned digest):")
    artifacts = [
        ("clusiana-3b-v1", "sha256:9f86d0..."),  # matches pinned digest
        ("clusiana-3b-v1", "sha256:deadbeef"),  # tampered / backdoored build
        ("randweights-7b", "sha256:abc123"),  # unpinned, unknown provenance
    ]
    for name, digest in artifacts:
        verdict = "LOAD" if admit_model(name, digest) else "REJECT"
        print(f"  [{verdict}] {name}  (digest={digest})")

    print("\nProvenance is checked before the dependency is wired in — not after it runs.")


if __name__ == "__main__":
    main()
