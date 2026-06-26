# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Notebook 38: RAG basics — a cloud Well-Architected knowledge base.

Retrieval-Augmented Generation (RAG) grounds an agent's answers in your
own documents. For a cloud-platform team that means the AWS
Well-Architected best-practice catalogue, mapped to your internal
runbooks — not whatever the model half-remembers about ``REL`` or
``COST`` ids. This notebook builds the Index that the cloud-ops agent
(STRATUS, notebook 40) reads from. The pipeline has four steps:

- **Embed** — turn text into vectors with ``OpenAIEmbeddings``
  (``text-embedding-3-small``, 1536 dims).
- **Store** — persist the vectors in a vector store. This notebook uses
  an in-memory ``QdrantVectorStore`` so it runs with no external
  service; swap in ``QdrantVectorStore`` / ``PgVectorStore`` /
  ``OpenSearchVectorStore`` for a durable backend.
- **Search** — find the closest vectors by cosine distance.
- **Generate** — feed the retrieved chunks to the LLM as grounded
  context. (This notebook focuses on steps 1–3; notebook 40 wires it
  into an agent.)

The corpus is Well-Architected best practices (``REL-xx``, ``COST-xx``,
``SEC-xx``, ``OPS-xx``); the retrieval quality of this Index directly
bounds how grounded STRATUS's answers can be. A low-coverage or stale
Index leaves the ops agent guessing instead of citing your runbooks —
the failure mode notebook 40 hardens against.

Run it:
    # Embeddings need an OpenAI api key:
    export OPENAI_API_KEY=sk-...
    python examples/notebook_38_rag_basics.py

    # Offline (skips the live demo cleanly when the key is missing):
    python examples/notebook_38_rag_basics.py
"""

import asyncio
import math
import os
import sys

from tulip.rag import OpenAIEmbeddings, QdrantVectorStore, RAGRetriever
from tulip.rag.stores.base import Document


def _missing_env() -> list[str]:
    return [name for name in ("OPENAI_API_KEY",) if not os.environ.get(name)]


def _get_embedder() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small")


def _get_store(dimension: int = 1536) -> QdrantVectorStore:
    return QdrantVectorStore(location=":memory:", dimension=dimension)


# =============================================================================
# Step 1: Embeddings — vectors that capture meaning, not just keywords.
# =============================================================================


async def understand_embeddings():
    print("=" * 60)
    print("Step 1: Embeddings (OpenAI · text-embedding-3-small)")
    print("=" * 60)

    embedder = _get_embedder()
    print(f"Embedder: {embedder.__class__.__name__}")
    print(f"Embedding dimension: {embedder.config.dimension}")

    texts = [
        "Auto Scaling groups add instances when average CPU exceeds the target",
        "Horizontal pod autoscaling adds replicas when observed CPU rises",
        "IAM policies grant least-privilege access to cloud resources",
    ]
    results = await embedder.embed_batch(texts)

    for i, result in enumerate(results):
        preview = result.embedding[:5]
        print(f"\n'{texts[i]}'")
        print(f"  First 5 dims: {[round(x, 4) for x in preview]}")
        print(f"  Total dims:   {len(result.embedding)}")

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb)

    sim_01 = cosine(results[0].embedding, results[1].embedding)
    sim_02 = cosine(results[0].embedding, results[2].embedding)
    print("\nCosine similarity:")
    print(f"  'Auto Scaling' vs 'Pod autoscaling': {sim_01:.4f}")
    print(f"  'Auto Scaling' vs 'IAM policies':    {sim_02:.4f}")


# =============================================================================
# Step 2: QdrantVectorStore (in-memory) — store vectors, query by cosine distance.
# =============================================================================


async def using_vector_store():
    print("\n" + "=" * 60)
    print("Step 2: QdrantVectorStore (in-memory, cosine)")
    print("=" * 60)

    embedder = _get_embedder()
    store = _get_store(dimension=embedder.config.dimension)
    print(f"Created QdrantVectorStore dim={store.config.dimension}")

    # AWS Well-Architected best-practice summaries — the operational
    # backbone the ops agent cites. Ids follow the pillar-prefixed form.
    docs_text = [
        "REL-04 Auto Scaling: scale compute horizontally on demand metrics so "
        "capacity tracks load and unhealthy instances are replaced automatically.",
        "REL-07 Multi-AZ Deployment: spread workloads across availability zones "
        "so a single zone failure does not take the whole service down.",
        "COST-03 Rightsizing: match instance types to measured utilization to "
        "stop paying for idle compute and memory headroom you never use.",
        "SEC-02 Least-Privilege IAM: grant the minimum permissions a workload "
        "needs and rotate long-lived credentials on a schedule.",
        "OPS-05 Infrastructure as Code: define resources in version-controlled "
        "templates for repeatable, reviewable, auditable provisioning.",
    ]

    print("\nEmbedding and inserting Well-Architected best practices…")
    for i, text in enumerate(docs_text):
        result = await embedder.embed(text)
        await store.add(
            Document(
                id=f"practice_{i}",
                content=text,
                embedding=result.embedding,
                metadata={"source": "well_architected_kb", "index": i},
            )
        )
        print(f"  inserted: {text[:50]}…")

    print("\nSearching for 'surviving a datacenter zone outage'…")
    q = await embedder.embed("surviving a datacenter zone outage")
    hits = await store.search(query_embedding=q.embedding, limit=3)
    for i, hit in enumerate(hits, start=1):
        print(f"  #{i}  score={hit.score:.4f}  {hit.document.content}")

    print(f"\nTotal rows in store: {await store.count()}")


# =============================================================================
# Step 3: RAGRetriever — one object that handles chunking, embedding, and
#         storage for you.
# =============================================================================


async def using_rag_retriever():
    print("\n" + "=" * 60)
    print("Step 3: RAGRetriever")
    print("=" * 60)

    embedder = _get_embedder()
    store = _get_store(dimension=embedder.config.dimension)

    retriever = RAGRetriever(
        embedder=embedder,
        store=store,
        chunk_size=500,
        chunk_overlap=50,
    )
    print("Created RAGRetriever (chunk_size=500, chunk_overlap=50)")

    knowledge_base = [
        """
        REL-07 Multi-AZ Deployment: run at least two instances of every tier
        in separate availability zones behind a load balancer, and put the
        database in a Multi-AZ configuration with automatic failover. Runbook:
        when a zone is reported unhealthy, drain its targets, confirm the
        replica has taken the primary role, and let Auto Scaling backfill
        capacity in the surviving zones.
        """,
        """
        A vector store keeps embeddings in a structure that supports fast
        nearest-neighbour search. Tulip ships in-memory, pgvector,
        OpenSearch, Qdrant, and Chroma stores behind one interface —
        the same API whether the Index holds five Well-Architected
        practices or the full framework plus every team runbook.
        """,
        """
        REL-04 Auto Scaling: a scaling policy adds instances when a target
        metric (average CPU, request count per target, or queue depth) stays
        above its threshold for the configured period, and removes them when
        it falls below. Runbook: if instances flap, widen the cooldown window
        and switch from a simple step policy to target-tracking.
        """,
    ]

    for doc in knowledge_base:
        ids = await retriever.add_document(doc.strip())
        print(f"  inserted practice note → {len(ids)} chunk(s)")

    print("\nQuerying: 'How do I keep a service up when a zone fails?'")
    result = await retriever.retrieve("How do I keep a service up when a zone fails?", limit=2)
    for i, doc_result in enumerate(result.documents, 1):
        print(f"\n  result {i} (score={doc_result.score:.4f}):")
        print(f"  {doc_result.document.content[:200]}…")

    print("\nUsing retrieve_text() for the same Index on a different question:")
    text = await retriever.retrieve_text("How does autoscaling decide to add capacity?", limit=2)
    print(f"\n{text[:300]}…")


# =============================================================================
# Step 4: Metadata-tagged retrieval — store Well-Architected pillar/practice
#         tags alongside the vector and use them to narrow results beyond
#         similarity alone.
# =============================================================================


async def rag_with_metadata():
    print("\n" + "=" * 60)
    print("Step 4: RAG with metadata")
    print("=" * 60)

    embedder = _get_embedder()
    store = _get_store(dimension=embedder.config.dimension)
    retriever = RAGRetriever(embedder=embedder, store=store)

    documents = [
        (
            "Auto Scaling groups add instances when average CPU exceeds the target.",
            {"pillar": "reliability", "practice": "REL-04"},
        ),
        (
            "Run every tier across multiple availability zones with automatic failover.",
            {"pillar": "reliability", "practice": "REL-07"},
        ),
        (
            "Rightsize instances to measured utilization to stop paying for idle compute.",
            {"pillar": "cost-optimization", "practice": "COST-03"},
        ),
        (
            "Grant workloads least-privilege IAM and rotate long-lived credentials.",
            {"pillar": "security", "practice": "SEC-02"},
        ),
        (
            "Define all infrastructure in version-controlled templates for repeatability.",
            {"pillar": "operational-excellence", "practice": "OPS-05"},
        ),
    ]

    for content, metadata in documents:
        await retriever.add_document(content, metadata=metadata)
        print(f"  inserted: {content[:40]}… {metadata}")

    print("\nQuerying 'scaling compute out when load increases'…")
    result = await retriever.retrieve("scaling compute out when load increases", limit=3)
    for r in result.documents:
        print(f"  score={r.score:.4f}  {r.document.content[:60]}…")
        print(f"    metadata: {r.document.metadata}")


# =============================================================================
# Main
# =============================================================================


async def main():
    missing = _missing_env()
    if missing:
        print("\n--- Notebook 38: RAG basics (Well-Architected Index) ---")
        print(
            "Required environment variables not set; skipping the live "
            "demo so this file still runs cleanly in CI.\n"
        )
        for name in missing:
            print(f"  - {name}")
        print("\nSet OPENAI_API_KEY (for embeddings), then re-run.")
        return

    await understand_embeddings()
    await using_vector_store()
    await using_rag_retriever()
    await rag_with_metadata()

    print("\n" + "=" * 60)
    print("Notebook 38 complete.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
