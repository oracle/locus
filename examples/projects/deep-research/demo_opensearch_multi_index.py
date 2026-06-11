#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Multi-index OpenSearch Deep Research demo.

Two OpenSearch indices (threat intel + CVE/vulnerability), the agent routes
between them via `create_deepagent(datastores={...})`.

Reads OpenSearch credentials from env vars and uses OpenAI embeddings.

Run:
    export OPENSEARCH_ENDPOINT=https://<your-opensearch-host>:9200
    export OPENSEARCH_USERNAME=<your-username>
    export OPENSEARCH_PASSWORD='...'
    export OPENSEARCH_VERIFY_CERTS=true
    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/projects/deep-research/demo_opensearch_multi_index.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

from tulip.deepagent import create_deepagent
from tulip.models import get_model
from tulip.rag import OpenAIEmbeddings, RAGRetriever
from tulip.rag.stores.base import Document
from tulip.rag.stores.opensearch import OpenSearchVectorStore


# Benign, education-level threat-intelligence facts (no live IOCs; RFC 5737
# documentation ranges only).
INTEL_CORPUS = [
    "Cobalt Strike beacons commonly use malleable C2 profiles to blend with normal web traffic.",
    "ATT&CK technique T1059.001 covers adversary use of PowerShell for execution.",
    "Living-off-the-land binaries (LOLBins) such as certutil are abused to download payloads.",
    "Domain generation algorithms (DGAs) produce many candidate C2 domains to evade blocklists.",
    "Credential dumping from LSASS memory maps to ATT&CK technique T1003.001.",
    "Fast-flux DNS rotates the IPs behind a domain rapidly to frustrate takedown.",
    "Beaconing traffic shows a regular periodicity that stands out from human-driven sessions.",
    "RFC 5737 reserves 198.51.100.0/24 for documentation; it is never routed on the public internet.",
    "Encoded PowerShell (-enc) is frequently used to obfuscate malicious command lines.",
    "Lateral movement via SMB and remote service creation maps to ATT&CK technique T1021.",
    "Phishing remains the most commonly reported initial-access vector in intrusions.",
    "MITRE ATT&CK orders tactics from Reconnaissance through Impact across the intrusion lifecycle.",
]

# Benign, education-level vulnerability facts.
CVE_CORPUS = [
    "CVSS v3.1 scores range from 0.0 to 10.0, with 9.0-10.0 classified as Critical.",
    "A CVE identifier uniquely names a publicly disclosed vulnerability, e.g. CVE-2021-44228.",
    "Log4Shell (CVE-2021-44228) allowed remote code execution via JNDI lookups in Log4j.",
    "An SSRF vulnerability lets an attacker coerce a server into making unintended requests.",
    "Path-traversal flaws use ../ sequences to read files outside the intended directory.",
    "The EPSS score estimates the probability that a vulnerability will be exploited in the wild.",
    "Patching cadence and asset exposure drive real-world risk more than CVSS alone.",
    "A vulnerability is exploitable only when a reachable, affected code path is exposed.",
    "Deserialization of untrusted data can lead to remote code execution (CWE-502).",
    "Default credentials on internet-facing services are a recurring source of compromise.",
    "Coordinated disclosure gives a vendor time to ship a fix before details go public.",
    "SQL injection (CWE-89) arises when untrusted input is concatenated into a query.",
]


def _parse_hosts(endpoint: str) -> list[str]:
    """Return the endpoint as the single-host list of URL strings tulip expects."""
    return [endpoint]


async def _ingest(
    store: OpenSearchVectorStore, embedder: OpenAIEmbeddings, texts: list[str], prefix: str
) -> None:
    """Embed + bulk-add docs to one OpenSearch index."""
    embs = await embedder.embed_documents(texts)
    docs = [
        Document(
            id=f"{prefix}-{i:02d}",
            content=text,
            embedding=e.embedding,
            metadata={"domain": prefix},
        )
        for i, (text, e) in enumerate(zip(texts, embs, strict=True))
    ]
    await store.add_batch(docs)


async def main() -> int:
    endpoint = os.environ.get(
        "OPENSEARCH_ENDPOINT",
        "https://<your-opensearch-host>:9200",
    )
    username = os.environ["OPENSEARCH_USERNAME"]
    password = os.environ["OPENSEARCH_PASSWORD"]
    verify_certs = os.environ.get("OPENSEARCH_VERIFY_CERTS", "true").lower() == "true"

    model_id = os.environ.get("TULIP_RESEARCH_MODEL", "anthropic:claude-sonnet-4-6")

    run_id = uuid.uuid4().hex[:8].lower()
    intel_index = f"tulip_intel_{run_id}"
    cve_index = f"tulip_cve_{run_id}"
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("MULTI-INDEX OPENSEARCH DEEP RESEARCH — TULIP PORT")
    print("=" * 70)
    print(f"  OpenSearch  : {endpoint} (user={username})")
    print(f"  Indices     : {intel_index}, {cve_index}")
    print(f"  Model       : {model_id}")
    print()

    print("[1/4] Embedding corpora with text-embedding-3-small …")
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    probe = await embedder.embed_query("probe")
    dim = len(probe.embedding)
    print(f"       dim={dim}")

    common_kwargs = {
        "hosts": _parse_hosts(endpoint),
        "http_auth": (username, password),
        "use_ssl": endpoint.startswith("https"),
        "dimension": dim,
        "verify_certs": verify_certs,
    }

    intel_store = OpenSearchVectorStore(index_name=intel_index, **common_kwargs)
    cve_store = OpenSearchVectorStore(index_name=cve_index, **common_kwargs)

    print("\n[2/4] Seeding OpenSearch indices …")
    try:
        await _ingest(intel_store, embedder, INTEL_CORPUS, "intel")
        await _ingest(cve_store, embedder, CVE_CORPUS, "cve")
        # OpenSearch default refresh interval is 1s; force a refresh so the
        # docs are visible to the next search. The underlying client is
        # AsyncOpenSearch, so its `indices.refresh` is a coroutine.
        for s, name in [(intel_store, intel_index), (cve_store, cve_index)]:
            client = getattr(s, "_client", None) or getattr(s, "client", None)
            if client is not None:
                await client.indices.refresh(index=name)
        print(f"       threat-intel: {len(INTEL_CORPUS)} docs  (count={await intel_store.count()})")
        print(f"       cve         : {len(CVE_CORPUS)} docs  (count={await cve_store.count()})")

        print("\n[3/4] Building deepagent with datastores={intel, cve} …")
        intel_retriever = RAGRetriever(embedder=embedder, store=intel_store)
        cve_retriever = RAGRetriever(embedder=embedder, store=cve_store)
        chat = get_model(model_id)
        agent = create_deepagent(
            model=chat,
            system_prompt=(
                "You are a security research assistant with access to two "
                "OpenSearch indices (threat intel + CVE/vulnerability). Route "
                "each search to the right index based on the topic. Cite "
                "document ids (intel-NN, cve-NN)."
            ),
            tools=[],
            datastores={
                "intel": {
                    "retriever": intel_retriever,
                    "description": "threat intelligence: malware C2, ATT&CK techniques, intrusion TTPs, indicators",
                    "top_k": 4,
                },
                "cve": {
                    "retriever": cve_retriever,
                    "description": "vulnerability knowledge: CVEs, CVSS/EPSS scoring, CWE classes, exploitation concepts",
                    "top_k": 4,
                },
            },
            max_output_tokens=4096,
            max_iterations=8,
            reflexion=False,
            grounding=False,
        )

        print("\n[4/4] Running cross-domain prompt …")
        print("-" * 70)
        prompt = (
            "Using only the two indices: (a) summarize how adversaries hide "
            "command-and-control traffic, drawing on the threat-intel index, "
            "and (b) list two distinct vulnerability concepts from the CVE "
            "index. Keep each section short (3-5 bullets). Cite document ids "
            "(intel-NN / cve-NN)."
        )
        # `agent.run(...)` returns an async generator of events. We can't
        # use `agent.run_sync()` here because we're already inside an
        # event loop with live AsyncOpenSearch clients — `run_sync`
        # spawns a new thread + loop and the OpenSearch clients (bound to
        # *this* loop) silently return empty results when invoked from
        # the agent's tool calls running in the other loop.
        from tulip.core.events import (
            TerminateEvent,
            ToolCompleteEvent,
            ToolStartEvent,
        )

        t0 = time.time()
        text = ""
        # Correlate ToolStartEvent.arguments <-> ToolCompleteEvent by tool_call_id
        args_by_id: dict[str, dict] = {}
        tool_records: list[tuple[str, dict, int]] = []
        async for event in agent.run(prompt):
            if isinstance(event, ToolStartEvent):
                args_by_id[event.tool_call_id] = event.arguments or {}
            elif isinstance(event, ToolCompleteEvent):
                tool_records.append(
                    (
                        event.tool_name,
                        args_by_id.get(event.tool_call_id, {}),
                        len(str(event.result or "")),
                    )
                )
            elif isinstance(event, TerminateEvent):
                text = event.final_message or ""
        elapsed = time.time() - t0

        intel_calls = sum(1 for n, _, _ in tool_records if n == "search_intel")
        cve_calls = sum(1 for n, _, _ in tool_records if n == "search_cve")

        print(f"\nTool calls    : {len(tool_records)} (intel={intel_calls}, cve={cve_calls})")
        for name, args, n in tool_records:
            q = args.get("query", args) if isinstance(args, dict) else args
            print(f"  - {name}({q!r}) -> {n} chars")
        print(f"Time          : {elapsed:.1f}s")
        print(f"\n--- Response ---\n{text}")

        out_path = out_dir / "opensearch_multi_index_report.md"
        out_path.write_text(text)
        print(f"\nReport saved to: {out_path}")
        if intel_calls > 0 and cve_calls > 0:
            print("\nROUTING CHECK: agent hit BOTH indices — PASS")
        else:
            print(f"\nROUTING CHECK: intel={intel_calls}, cve={cve_calls} — partial")
    finally:
        # Drop the indices so we leave the cluster clean. AsyncOpenSearch
        # exposes `indices.exists` / `indices.delete` as coroutines.
        for s, name in [(intel_store, intel_index), (cve_store, cve_index)]:
            try:
                client = getattr(s, "_client", None) or getattr(s, "client", None)
                if client is not None and await client.indices.exists(index=name):
                    await client.indices.delete(index=name)
                    print(f"Dropped index: {name}")
            except Exception as exc:
                print(f"Cleanup warning for {name}: {exc}")
            try:
                await s.close()
            except BaseException:
                pass
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(130)
