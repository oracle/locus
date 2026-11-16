# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Notebook 52: RAG providers — a payments runbook KB on any vector store.

A payments operations team's runbook corpus — decline-code playbooks,
chargeback procedures, refund SLAs — outlives any single backend choice.
Production RAG is two pluggable pieces, both behind one Locus interface:

- **Embeddings** — ``OpenAIEmbeddings`` (``text-embedding-3-small`` /
  ``-3-large``), ``CohereEmbeddings`` (Cohere's direct API),
  ``OCIEmbeddings`` (Cohere embedding models served by OCI Generative AI)
  or ``OracleInDBEmbeddings`` (ONNX models running inside Oracle Database
  26ai).
- **Vector store** — ``InMemoryVectorStore`` for demos and tests, or a
  durable backend: ``PgVectorStore``, ``OpenSearchVectorStore``,
  ``QdrantVectorStore``, ``ChromaVectorStore``, ``OracleVectorStore``
  (Oracle Database 26ai), ``MilvusVectorStore``, ``WeaviateVectorStore``,
  ``ElasticsearchVectorStore``, ``RedisVectorStore``,
  ``MongoDBAtlasVectorStore`` or ``PineconeVectorStore``. Swapping is a
  one-line change; the retrieve/add API is identical.

Choice of embedding model and distance metric is an operations
decision, not just a tuning knob: it sets retrieval precision, and weak
retrieval is what routes an agent to the wrong runbook and a wrong
answer to a merchant. What each part covers (all against the same
runbook corpus):

- Part 1 — embedding-model selection (two models of different width).
- Part 2 — three ``distance_metric`` strings (COSINE / DOT / EUCLIDEAN) and
  what the store does with them.
- Part 3 — Qdrant in-memory store as a drop-in for ``InMemoryVectorStore``.
- Part 4 — batch ingest, ``count()``, ``clear()``.
- Part 5 — Milvus Lite: a file-backed embedded store (no server, no account)
  with BM25 hybrid search behind the retriever's ``hybrid=True`` switch.
- Part 6 — ``LLMReranker``: the chat model grades the over-fetched hits.

Run it:
    # Embeddings need a provider: an OpenAI API key, or an OCI profile
    # (LOCUS_EMBEDDING_PROVIDER=oci picks OCI Generative AI explicitly).
    # Without either the notebook prints what's missing and exits cleanly.
    export OPENAI_API_KEY=sk-...
    python examples/notebook_52_rag_providers.py
"""

import asyncio
import os
import shutil
import sys
import tempfile

from config import get_embedder, get_embedding_provider, get_model

from locus.rag import (
    InMemoryVectorStore,
    LLMReranker,
    MilvusVectorStore,
    QdrantVectorStore,
    RAGRetriever,
)
from locus.rag.stores.base import Document


def _missing_env() -> list[str]:
    if get_embedding_provider() == "oci":
        return (
            []
            if os.environ.get("OCI_PROFILE") or os.environ.get("LOCUS_OCI_PROFILE")
            else ["OCI_PROFILE"]
        )
    return [name for name in ("OPENAI_API_KEY",) if not os.environ.get(name)]


def _embedding_models() -> list[str]:
    """Two embedding models of different width for Part 1; the first is used everywhere else."""
    configured = [
        m
        for m in (
            os.environ.get("LOCUS_EMBEDDING_MODEL"),
            os.environ.get("LOCUS_EMBEDDING_MODEL_B"),
        )
        if m
    ]
    if configured:
        return configured
    if get_embedding_provider() == "oci":
        return ["cohere.embed-v4.0", "openai.text-embedding-3-small"]
    return ["text-embedding-3-small", "text-embedding-3-large"]


def _store(*, dimension: int, distance: str = "COSINE") -> InMemoryVectorStore:
    return InMemoryVectorStore(dimension=dimension, distance_metric=distance)


# A small payments-operations runbook corpus. All reason codes and
# merchant ids are fictitious; amounts and accounts are illustrative only.
CORPUS = [
    "Reason code 4853 (cardholder dispute): merchandise not received. Request "
    "proof of delivery and tracking; represent within 30 days or accept the "
    "chargeback.",
    "Decline code 51 (insufficient funds): the issuer rejected the auth for a "
    "low balance. Safe to retry with a smaller amount or after payday; do not "
    "force-post.",
    "Decline code 05 (do not honor): a generic issuer refusal. Do not retry "
    "blindly; route the customer to update their card or contact their bank.",
    "Refund SLA: card refunds settle in 5-10 business days; the funds leave the "
    "merchant account immediately but appear on the statement after the issuer "
    "posts them.",
    "Chargeback reason 10.4 (fraud, card-absent): the cardholder denies the "
    "transaction. Submit AVS match, 3-D Secure result, and prior order history "
    "as compelling evidence.",
    "ACH return R01 (insufficient funds): the bank account had no balance to "
    "cover the debit. Reattempt once after 2 business days, then dun the "
    "customer if it fails again.",
]


# =============================================================================
# Part 1: small vs large embedding models against the same runbook corpus.
# =============================================================================


async def part1_embedding_models():
    print("=" * 60)
    print("Part 1: two embedding models on the same corpus")
    print("=" * 60)

    for model in _embedding_models():
        embedder = get_embedder(model=model)
        store = _store(dimension=embedder.config.dimension)
        retriever = RAGRetriever(embedder=embedder, store=store)
        print(f"\n  {model} → dim={embedder.config.dimension}")
        await retriever.add_documents(CORPUS)
        hits = await retriever.retrieve("customer says their package never arrived", limit=2)
        for i, h in enumerate(hits.documents, start=1):
            print(f"    #{i} score={h.score:.4f} {h.document.content[:70]}…")


# =============================================================================
# Part 2: Distance metric variants — the metric is a property of the store.
#         InMemoryVectorStore matches "cosine" and "euclidean" exactly and
#         scores any other name, including the upper-case ones below, as a
#         dot product; OracleVectorStore is the backend that spells the
#         metrics COSINE / DOT / EUCLIDEAN.
# =============================================================================


async def part2_distance_metrics():
    print("\n" + "=" * 60)
    print("Part 2: Distance metric variants on the same corpus")
    print("=" * 60)

    embedder = get_embedder(model=_embedding_models()[0])
    query = "issuer declined the card for not enough money"

    # The metric is a property of the store, not the vectors: embed the
    # corpus and the query once, then load the same Documents into three
    # stores so the only thing that differs between the rows is the metric.
    results = await embedder.embed_batch(CORPUS)
    docs = [
        Document(id=f"runbook_{i}", content=text, embedding=r.embedding)
        for i, (text, r) in enumerate(zip(CORPUS, results, strict=True))
    ]
    q = await embedder.embed(query)

    for metric in ["COSINE", "DOT", "EUCLIDEAN"]:
        store = _store(dimension=embedder.config.dimension, distance=metric)
        await store.add_batch(docs)
        hits = await store.search(query_embedding=q.embedding, limit=2)
        top = hits[0]
        print(f"  {metric}: top score={top.score:.4f}  → {top.document.content[:60]}…")


# =============================================================================
# Part 3: Swap the backend — Qdrant in-memory is a drop-in for the
#         in-memory store. Same RAGRetriever API, durable backend in prod.
# =============================================================================


async def part3_swap_backend():
    print("\n" + "=" * 60)
    print("Part 3: QdrantVectorStore (location=':memory:')")
    print("=" * 60)

    embedder = get_embedder(model=_embedding_models()[0])
    try:
        store = QdrantVectorStore(
            location=":memory:",
            dimension=embedder.config.dimension,
        )
    except ImportError:
        # qdrant-client isn't installed here (e.g. a browser/WASM runtime or a
        # slim install). The swap is a one-liner precisely because the API is
        # identical to the InMemoryVectorStore shown above — install the extra
        # to run it: pip install "locus-sdk[qdrant]".
        print("  QdrantVectorStore needs the qdrant extra (pip install 'locus-sdk[qdrant]').")
        print("  Same RAGRetriever/store API as InMemoryVectorStore above — nothing else changes.")
        return

    for i, text in enumerate(CORPUS[:4]):
        emb = await embedder.embed(text)
        await store.add(
            Document(
                id=f"runbook_{i}",
                content=text,
                embedding=emb.embedding,
                metadata={"channel": "card" if i % 2 == 0 else "bank"},
            )
        )

    q = await embedder.embed("how do I handle a cardholder disputing a charge?")
    hits = await store.search(query_embedding=q.embedding, limit=3)
    print(f"  Searched {await store.count()} runbooks in the Qdrant store:")
    for i, hit in enumerate(hits, start=1):
        print(f"    #{i} score={hit.score:.4f}  {hit.document.content[:70]}…")


# =============================================================================
# Part 4: Batch lifecycle — add_documents, count(), clear().
# =============================================================================


async def part4_batch():
    print("\n" + "=" * 60)
    print("Part 4: Batch ingest + count + clear")
    print("=" * 60)

    embedder = get_embedder(model=_embedding_models()[0])
    store = _store(dimension=embedder.config.dimension)
    retriever = RAGRetriever(embedder=embedder, store=store)

    await retriever.add_documents(CORPUS)
    print(f"  After add_documents: runbooks = {await store.count()}")
    await store.clear()
    print(f"  After clear():       runbooks = {await store.count()}")


# =============================================================================
# Part 5: Milvus Lite — a file on disk is the whole database. No server, no
#         account. ``enable_hybrid=True`` adds a BM25 sparse field so the
#         retriever's ``hybrid=True`` switch blends keyword and vector
#         relevance; the same switch on a store without hybrid search falls
#         back to vector-only and logs it once.
# =============================================================================


async def part5_milvus_lite_hybrid():
    print("\n" + "=" * 60)
    print("Part 5: MilvusVectorStore (Milvus Lite) + hybrid=True")
    print("=" * 60)

    embedder = get_embedder(model=_embedding_models()[0])
    workdir = tempfile.mkdtemp(prefix="locus-nb52-")
    try:
        try:
            store = MilvusVectorStore(
                uri=os.path.join(workdir, "runbooks.db"),
                dimension=embedder.config.dimension,
                enable_hybrid=True,
            )
        except ImportError:
            print("  MilvusVectorStore needs the milvus extra (pip install 'locus-sdk[milvus]').")
            print("  Same RAGRetriever/store API as the stores above — nothing else changes.")
            return

        retriever = RAGRetriever(embedder=embedder, store=store, hybrid=True, hybrid_alpha=0.5)
        await retriever.add_documents(CORPUS)
        print(f"  Milvus Lite file: {store!r}; hybrid supported: {retriever.hybrid_supported}")

        # An exact code is a keyword problem: "05" carries little meaning for
        # an embedding model but is decisive for BM25.
        query = "what should the agent do with decline code 05?"
        for hybrid in (False, True):
            hits = await retriever.retrieve(query, limit=2, hybrid=hybrid)
            label = "hybrid (alpha=0.5)" if hybrid else "vector only       "
            print(f"\n  {label}:")
            for i, h in enumerate(hits.documents, start=1):
                print(f"    #{i} score={h.score:.4f} {h.document.content[:70]}…")
        await store.close()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# =============================================================================
# Part 6: LLM reranker — the chat model the agent already uses grades the
#         over-fetched candidates. No rerank endpoint, one completion per
#         batch of up to 20 passages; the retriever's score stays on
#         ``.distance`` so both signals can be compared.
# =============================================================================


async def part6_llm_reranker():
    print("\n" + "=" * 60)
    print("Part 6: LLMReranker on the retriever")
    print("=" * 60)

    embedder = get_embedder(model=_embedding_models()[0])
    store = _store(dimension=embedder.config.dimension)
    retriever = RAGRetriever(
        embedder=embedder,
        store=store,
        reranker=LLMReranker(get_model(), top_n=2),
        rerank_candidate_pool=len(CORPUS),
    )
    await retriever.add_documents(CORPUS)

    hits = await retriever.retrieve("how long until a refund shows on the statement?", limit=2)
    for i, h in enumerate(hits.documents, start=1):
        print(
            f"  #{i} grade={h.score:.2f} (vector score {h.distance:.4f}) {h.document.content[:60]}…"
        )


async def main():
    missing = _missing_env()
    if missing:
        print("\n--- Notebook 52: RAG providers (payments runbook KB) ---")
        print(
            "Required environment variables not set; skipping the live "
            "demo so this file still runs cleanly in CI.\n"
        )
        for name in missing:
            print(f"  - {name}")
        print(
            "\nSet OPENAI_API_KEY, or an OCI profile with LOCUS_EMBEDDING_PROVIDER=oci "
            "(for embeddings), then re-run."
        )
        return

    await part1_embedding_models()
    await part2_distance_metrics()
    await part3_swap_backend()
    await part4_batch()
    await part5_milvus_lite_hybrid()
    await part6_llm_reranker()

    print("\n" + "=" * 60)
    print("Notebook 52 complete.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
