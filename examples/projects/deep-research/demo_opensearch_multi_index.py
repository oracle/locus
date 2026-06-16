#!/usr/bin/env python3
# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Locus port of the OpenSearch multi-index Deep Research gist.

Mirrors `gist 92b9a515/opensearch_multi_index_huggingface_example.py`:
two OpenSearch indices (medical + news), agent routes between them
via `create_deepagent(datastores={...})`.

Locus equivalents:
- `from langchain_oci.datastores import OpenSearch`       -> `from locus.rag.stores.opensearch import OpenSearchVectorStore`
- `create_deep_research_agent(datastores={...})`           -> `create_deepagent(datastores={...})`
- Hugging Face `pubmed_qa` + `ag_news`                     -> inline 12-sentence corpora (no HF deps)

Reads OpenSearch credentials from the locus-side env vars
(or the user's eolas dev cluster defaults — set via flags).

Run:
    export OPENSEARCH_ENDPOINT=https://<your-opensearch-host>:9200
    export OPENSEARCH_USERNAME=<your-username>
    export OPENSEARCH_PASSWORD='...'
    export OPENSEARCH_VERIFY_CERTS=true
    export OCI_PROFILE=DEFAULT
    export OCI_AUTH_TYPE=api_key
    export OCI_COMPARTMENT=ocid1.tenancy.oc1..<your-tenancy>
    .venv/bin/python examples/projects/deep-research/demo_opensearch_multi_index.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

from locus.deepagent import create_deepagent
from locus.models import get_model
from locus.rag import OCIEmbeddings, RAGRetriever
from locus.rag.stores.base import Document
from locus.rag.stores.opensearch import OpenSearchVectorStore


MEDICAL_CORPUS = [
    "Iron is primarily absorbed in the duodenum and proximal jejunum of the small intestine.",
    "Hepcidin, produced by the liver, is the master regulator of systemic iron homeostasis.",
    "Iron deficiency anemia is the most common nutritional deficiency worldwide.",
    "Hereditary hemochromatosis is most commonly caused by C282Y homozygosity in the HFE gene.",
    "Transferrin saturation below 16% suggests iron deficiency; above 45% raises concern for iron overload.",
    "Ferritin is the primary iron storage protein and an acute-phase reactant.",
    "First-line treatment for iron deficiency anemia is oral ferrous sulfate.",
    "Phlebotomy is the first-line treatment for hereditary hemochromatosis.",
    "Anemia of chronic disease is driven by elevated hepcidin in inflammatory states.",
    "Reticulocyte hemoglobin content (CHr) is an early functional marker of iron deficiency.",
    "MRI T2* relaxometry is the gold standard for non-invasive iron quantification.",
    "Iron-refractory iron deficiency anemia (IRIDA) is caused by TMPRSS6 mutations.",
]

NEWS_CORPUS = [
    "Markets closed mixed on Friday as tech stocks rallied while energy shares declined.",
    "The central bank held interest rates steady, citing persistent inflation in services.",
    "A major airline announced new transatlantic routes opening next quarter.",
    "Local elections saw record turnout in three coastal districts, officials reported.",
    "A new infrastructure bill cleared the lower chamber by a 215-204 margin.",
    "The national weather service issued advisories for severe thunderstorms across the plains.",
    "An automaker recalled 120,000 SUVs over a brake-line manufacturing defect.",
    "Box office returns this weekend were dominated by an animated sequel.",
    "Tech regulators proposed new rules on cross-border data transfers for cloud providers.",
    "Universities reported a 6% rise in international graduate applications this cycle.",
    "A diplomatic delegation arrived in the capital ahead of next week's trade talks.",
    "Sports authorities approved a new playoff format starting next season.",
]


def _parse_hosts(endpoint: str) -> list[str]:
    """Return the endpoint as the single-host list of URL strings locus expects."""
    return [endpoint]


async def _ingest(
    store: OpenSearchVectorStore, embedder: OCIEmbeddings, texts: list[str], prefix: str
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

    oci_profile = os.environ.get("OCI_PROFILE", "DEFAULT")
    auth_type = os.environ.get("OCI_AUTH_TYPE", "api_key")
    compartment = os.environ.get(
        "OCI_COMPARTMENT",
        "ocid1.tenancy.oc1..<your-tenancy>",
    )
    genai_endpoint = os.environ.get(
        "OCI_ENDPOINT",
        "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    )
    region = os.environ.get("OCI_REGION", "us-chicago-1")
    model_id = os.environ.get("OCI_RESEARCH_MODEL", "oci:openai.gpt-4o-mini")

    run_id = uuid.uuid4().hex[:8].lower()
    med_index = f"locus_med_{run_id}"
    news_index = f"locus_news_{run_id}"
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("MULTI-INDEX OPENSEARCH DEEP RESEARCH — LOCUS PORT")
    print("=" * 70)
    print(f"  OpenSearch  : {endpoint} (user={username})")
    print(f"  Indices     : {med_index}, {news_index}")
    print(f"  Model       : {model_id}")
    print()

    print("[1/4] Embedding corpora with cohere.embed-v4.0 …")
    embedder = OCIEmbeddings(
        model_id="cohere.embed-v4.0",
        compartment_id=compartment,
        profile_name=oci_profile,
        auth_type=auth_type,
        service_endpoint=genai_endpoint,
    )
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

    med_store = OpenSearchVectorStore(index_name=med_index, **common_kwargs)
    news_store = OpenSearchVectorStore(index_name=news_index, **common_kwargs)

    print(f"\n[2/4] Seeding OpenSearch indices …")
    try:
        await _ingest(med_store, embedder, MEDICAL_CORPUS, "med")
        await _ingest(news_store, embedder, NEWS_CORPUS, "news")
        # OpenSearch default refresh interval is 1s; force a refresh so the
        # docs are visible to the next search. The underlying client is
        # AsyncOpenSearch, so its `indices.refresh` is a coroutine.
        for s, name in [(med_store, med_index), (news_store, news_index)]:
            client = getattr(s, "_client", None) or getattr(s, "client", None)
            if client is not None:
                await client.indices.refresh(index=name)
        print(f"       medical: {len(MEDICAL_CORPUS)} docs  (count={await med_store.count()})")
        print(f"       news   : {len(NEWS_CORPUS)} docs  (count={await news_store.count()})")

        print(f"\n[3/4] Building deepagent with datastores={{medical, news}} …")
        med_retriever = RAGRetriever(embedder=embedder, store=med_store)
        news_retriever = RAGRetriever(embedder=embedder, store=news_store)
        chat = get_model(model_id, profile=oci_profile, compartment_id=compartment, region=region)
        agent = create_deepagent(
            model=chat,
            system_prompt=(
                "You are a research assistant with access to two OpenSearch "
                "indices (medical + news). Route each search at the right "
                "index based on the topic. Cite document ids (med-NN, news-NN)."
            ),
            tools=[],
            datastores={
                "medical": {
                    "retriever": med_retriever,
                    "description": "clinical/hematology knowledge: iron metabolism, anemia, hemochromatosis, diagnostics, treatment",
                    "top_k": 4,
                },
                "news": {
                    "retriever": news_retriever,
                    "description": "general news headlines: markets, politics, weather, transportation, sports",
                    "top_k": 4,
                },
            },
            max_output_tokens=4096,
            max_iterations=8,
            reflexion=False,
            grounding=False,
        )

        print(f"\n[4/4] Running cross-domain prompt …")
        print("-" * 70)
        prompt = (
            "Using only the two indices: (a) summarize the key regulators "
            "of iron homeostasis from the medical index, and (b) list two "
            "distinct items from the news index. Keep each section short "
            "(3-5 bullets). Cite document ids (med-NN / news-NN)."
        )
        # `agent.run(...)` returns an async generator of events. We can't
        # use `agent.run_sync()` here because we're already inside an
        # event loop with live AsyncOpenSearch clients — `run_sync`
        # spawns a new thread + loop and the OpenSearch clients (bound to
        # *this* loop) silently return empty results when invoked from
        # the agent's tool calls running in the other loop.
        from locus.core.events import (
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

        med_calls = sum(1 for n, _, _ in tool_records if n == "search_medical")
        news_calls = sum(1 for n, _, _ in tool_records if n == "search_news")

        print(f"\nTool calls    : {len(tool_records)} (medical={med_calls}, news={news_calls})")
        for name, args, n in tool_records:
            q = args.get("query", args) if isinstance(args, dict) else args
            print(f"  - {name}({q!r}) -> {n} chars")
        print(f"Time          : {elapsed:.1f}s")
        print(f"\n--- Response ---\n{text}")

        out_path = out_dir / "opensearch_multi_index_report.md"
        out_path.write_text(text)
        print(f"\nReport saved to: {out_path}")
        if med_calls > 0 and news_calls > 0:
            print("\nROUTING CHECK: agent hit BOTH indices — PASS")
        else:
            print(f"\nROUTING CHECK: med={med_calls}, news={news_calls} — partial")
    finally:
        # Drop the indices so we leave the cluster clean. AsyncOpenSearch
        # exposes `indices.exists` / `indices.delete` as coroutines.
        for s, name in [(med_store, med_index), (news_store, news_index)]:
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
