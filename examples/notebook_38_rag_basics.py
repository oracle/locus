# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Notebook 38: RAG basics — a MITRE ATLAS technique knowledge base.

Retrieval-Augmented Generation (RAG) grounds an agent's answers in your
own documents. For an AI-security team that means the MITRE ATLAS
adversarial-ML technique catalogue, mapped to internal detections — not
whatever the model half-remembers about AML.Txxxx ids. This notebook
builds the Index that the threat-intel agent (AUGUR, notebook 40) reads
from. The pipeline has four steps:

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

The corpus is MITRE ATLAS techniques (``AML.Txxxx``); the retrieval
quality of this Index directly bounds how grounded AUGUR's answers can
be. A poisoned or low-coverage Index maps to OWASP LLM08 (Vector &
Embedding Weaknesses) — covered in notebook 40.

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
        "Prompt injection overrides an LLM's instructions via crafted input",
        "Indirect prompt injection hides instructions in content the model retrieves",
        "Disk snapshots are rotated nightly for backup retention",
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
    print(f"  'Prompt injection' vs 'Indirect injection': {sim_01:.4f}")
    print(f"  'Prompt injection' vs 'Backups':            {sim_02:.4f}")


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

    # MITRE ATLAS technique summaries — the AI-security analogue of the
    # ATT&CK matrix. Ids are the canonical AML.Txxxx.
    docs_text = [
        "AML.T0051 LLM Prompt Injection: adversaries craft input that overrides "
        "the model's instructions, directly or indirectly via retrieved content.",
        "AML.T0054 LLM Jailbreak: adversaries bypass model safety controls to "
        "elicit restricted behaviour.",
        "AML.T0020 Poison Training Data: adversaries tamper with training or "
        "fine-tuning data to bias or backdoor a model.",
        "AML.T0024 Exfiltration via AI Inference API: adversaries probe an "
        "inference endpoint to extract model parameters or memorized data.",
        "AML.T0110 AI Agent Tool Poisoning: adversaries corrupt a tool's output "
        "or schema so the agent acts on attacker-controlled data.",
    ]

    print("\nEmbedding and inserting ATLAS technique summaries…")
    for i, text in enumerate(docs_text):
        result = await embedder.embed(text)
        await store.add(
            Document(
                id=f"technique_{i}",
                content=text,
                embedding=result.embedding,
                metadata={"source": "atlas_kb", "index": i},
            )
        )
        print(f"  inserted: {text[:50]}…")

    print("\nSearching for 'hidden instructions in retrieved documents'…")
    q = await embedder.embed("hidden instructions in retrieved documents")
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
        AML.T0051 LLM Prompt Injection: adversaries supply input that the model
        treats as instructions, overriding the system prompt. Indirect variants
        plant the payload in documents the RAG layer will retrieve. Detection:
        flag retrieved chunks containing imperative phrases addressed to the
        model, and isolate untrusted tool output from the instruction channel.
        """,
        """
        A vector store keeps embeddings in a structure that supports fast
        nearest-neighbour search. Tulip ships in-memory, pgvector,
        OpenSearch, Qdrant, and Chroma stores behind one interface —
        the same API whether the Index holds five ATLAS techniques or the
        full matrix plus the OWASP LLM Top 10.
        """,
        """
        AML.T0024 Exfiltration via AI Inference API: adversaries query an
        inference endpoint to reconstruct parameters or extract memorized
        training data. Detection: rate-limit per-principal inference volume and
        alert on query patterns consistent with model-extraction sweeps.
        """,
    ]

    for doc in knowledge_base:
        ids = await retriever.add_document(doc.strip())
        print(f"  inserted technique note → {len(ids)} chunk(s)")

    print("\nQuerying: 'How do attackers smuggle instructions into a model?'")
    result = await retriever.retrieve(
        "How do attackers smuggle instructions into a model?", limit=2
    )
    for i, doc_result in enumerate(result.documents, 1):
        print(f"\n  result {i} (score={doc_result.score:.4f}):")
        print(f"  {doc_result.document.content[:200]}…")

    print("\nUsing retrieve_text() for the same Index on a different question:")
    text = await retriever.retrieve_text("How is model extraction detected?", limit=2)
    print(f"\n{text[:300]}…")


# =============================================================================
# Step 4: Metadata-tagged retrieval — store ATLAS tactic/technique tags
#         alongside the vector and use them to narrow results beyond
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
            "Crafted input overrides the system prompt to change the agent's goal.",
            {"tactic": "ml-attack-staging", "technique": "AML.T0051"},
        ),
        (
            "A jailbreak prompt elicits behaviour the model's controls forbid.",
            {"tactic": "defense-evasion", "technique": "AML.T0054"},
        ),
        (
            "Adversaries poison fine-tuning data to plant a trigger phrase.",
            {"tactic": "resource-development", "technique": "AML.T0020"},
        ),
        (
            "Repeated targeted queries reconstruct model parameters over the API.",
            {"tactic": "exfiltration", "technique": "AML.T0024"},
        ),
        (
            "A poisoned tool schema redirects the agent to attacker infrastructure.",
            {"tactic": "ml-attack-staging", "technique": "AML.T0110"},
        ),
    ]

    for content, metadata in documents:
        await retriever.add_document(content, metadata=metadata)
        print(f"  inserted: {content[:40]}… {metadata}")

    print("\nQuerying 'overriding the model's instructions'…")
    result = await retriever.retrieve("overriding the model's instructions", limit=3)
    for r in result.documents:
        print(f"  score={r.score:.4f}  {r.document.content[:60]}…")
        print(f"    metadata: {r.document.metadata}")


# =============================================================================
# Main
# =============================================================================


async def main():
    missing = _missing_env()
    if missing:
        print("\n--- Notebook 38: RAG basics (MITRE ATLAS Index) ---")
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
