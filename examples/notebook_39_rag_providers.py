# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Notebook 39: RAG providers — a payments runbook KB on any vector store.

A payments operations team's runbook corpus outlives any single backend
choice. The same corpus that feeds the dispute assistant (notebook 38)
also carries decline-code playbooks and chargeback procedures, and
production RAG is two pluggable pieces, both behind one Tulip interface:

- **Embeddings** — ``OpenAIEmbeddings`` (``text-embedding-3-small`` /
  ``-3-large``) or ``CohereEmbeddings`` (Cohere's direct API).
- **Vector store** — ``InMemoryVectorStore`` for demos and tests, or a
  durable backend: ``PgVectorStore``, ``OpenSearchVectorStore``,
  ``QdrantVectorStore``, ``ChromaVectorStore``. Swapping is a one-line
  change; the retrieve/add API is identical.

Choice of embedding model and distance metric is an operations
decision, not just a tuning knob: it sets retrieval precision, and weak
retrieval is what routes an agent to the wrong runbook and a wrong
answer to a merchant. What each part covers (all against the same
runbook corpus):

- Part 1 — embedding-model selection (small vs large dimensions).
- Part 2 — distance metric choices (COSINE / DOT / EUCLIDEAN).
- Part 3 — Qdrant in-memory store as a drop-in for InMemoryVectorStore.
- Part 4 — batch ingest, ``count()``, ``clear()``.

Run it:
    export OPENAI_API_KEY=sk-...
    python examples/notebook_39_rag_providers.py

    # Offline (skips the live demo cleanly when the key is missing):
    python examples/notebook_39_rag_providers.py
"""

import asyncio
import os
import sys

from tulip.rag import (
    InMemoryVectorStore,
    OpenAIEmbeddings,
    QdrantVectorStore,
    RAGRetriever,
)
from tulip.rag.stores.base import Document


def _missing_env() -> list[str]:
    return [name for name in ("OPENAI_API_KEY",) if not os.environ.get(name)]


def _embedder(model: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=model)


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
    print("Part 1: OpenAIEmbeddings — small vs large")
    print("=" * 60)

    for model in ["text-embedding-3-small", "text-embedding-3-large"]:
        embedder = _embedder(model)
        store = _store(dimension=embedder.config.dimension)
        retriever = RAGRetriever(embedder=embedder, store=store)
        print(f"\n  {model} → dim={embedder.config.dimension}")
        await retriever.add_documents(CORPUS)
        hits = await retriever.retrieve("customer says their package never arrived", limit=2)
        for i, h in enumerate(hits.documents, start=1):
            print(f"    #{i} score={h.score:.4f} {h.document.content[:70]}…")


# =============================================================================
# Part 2: Distance metric variants — COSINE is the default; DOT and
#         EUCLIDEAN are alternative shapes the store supports.
# =============================================================================


async def part2_distance_metrics():
    print("\n" + "=" * 60)
    print("Part 2: Distance metric variants on the same corpus")
    print("=" * 60)

    embedder = _embedder("text-embedding-3-small")
    query = "issuer declined the card for not enough money"

    for metric in ["COSINE", "DOT", "EUCLIDEAN"]:
        store = _store(dimension=embedder.config.dimension, distance=metric)
        retriever = RAGRetriever(embedder=embedder, store=store)
        await retriever.add_documents(CORPUS)
        hits = await retriever.retrieve(query, limit=2)
        top = hits.documents[0]
        print(f"  {metric}: top score={top.score:.4f}  → {top.document.content[:60]}…")


# =============================================================================
# Part 3: Swap the backend — Qdrant in-memory is a drop-in for the
#         in-memory store. Same RAGRetriever API, durable backend in prod.
# =============================================================================


async def part3_swap_backend():
    print("\n" + "=" * 60)
    print("Part 3: QdrantVectorStore (location=':memory:')")
    print("=" * 60)

    embedder = _embedder("text-embedding-3-small")
    try:
        store = QdrantVectorStore(
            location=":memory:",
            dimension=embedder.config.dimension,
        )
    except ImportError:
        # qdrant-client isn't installed here (e.g. a browser/WASM runtime or a
        # slim install). The swap is a one-liner precisely because the API is
        # identical to the InMemoryVectorStore shown above — install the extra
        # to run it: pip install "tulip-agents[qdrant]".
        print("  QdrantVectorStore needs the qdrant extra (pip install 'tulip-agents[qdrant]').")
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

    embedder = _embedder("text-embedding-3-small")
    store = _store(dimension=embedder.config.dimension)
    retriever = RAGRetriever(embedder=embedder, store=store)

    await retriever.add_documents(CORPUS)
    print(f"  After add_documents: runbooks = {await store.count()}")
    await store.clear()
    print(f"  After clear():       runbooks = {await store.count()}")


# =============================================================================
# Main
# =============================================================================


async def main():
    missing = _missing_env()
    if missing:
        print("\n--- Notebook 39: RAG providers (payments runbook KB) ---")
        print(
            "Required environment variables not set; skipping the live "
            "demo so this file still runs cleanly in CI.\n"
        )
        for name in missing:
            print(f"  - {name}")
        print("\nSet OPENAI_API_KEY (for embeddings), then re-run.")
        return

    await part1_embedding_models()
    await part2_distance_metrics()
    await part3_swap_backend()
    await part4_batch()

    print("\n" + "=" * 60)
    print("Notebook 39 complete.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
