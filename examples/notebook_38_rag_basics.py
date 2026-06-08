# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL
"""Notebook 38: RAG basics.

Retrieval-Augmented Generation (RAG) grounds an agent's answers in your
own documents. The pipeline has four steps:

- **Embed** — turn text into vectors with ``OpenAIEmbeddings``
  (``text-embedding-3-small``, 1536 dims).
- **Store** — persist the vectors in a vector store. This notebook uses
  the bundled ``InMemoryVectorStore`` so it runs with no external
  service; swap in ``QdrantVectorStore`` / ``PgVectorStore`` /
  ``OpenSearchVectorStore`` for a durable backend.
- **Search** — find the closest vectors by cosine distance.
- **Generate** — feed the retrieved chunks to the LLM as grounded
  context. (This notebook focuses on steps 1–3; notebook 40 wires it
  into an agent.)

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

from tulip.rag import InMemoryVectorStore, OpenAIEmbeddings, RAGRetriever
from tulip.rag.stores.base import Document


def _missing_env() -> list[str]:
    return [name for name in ("OPENAI_API_KEY",) if not os.environ.get(name)]


def _get_embedder() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small")


def _get_store(dimension: int = 1536) -> InMemoryVectorStore:
    return InMemoryVectorStore(dimension=dimension)


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
        "Python is a programming language",
        "Python is used for machine learning",
        "Cats are fluffy animals",
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
    print(f"  'Python programming' vs 'Python ML': {sim_01:.4f}")
    print(f"  'Python programming' vs 'Cats':     {sim_02:.4f}")


# =============================================================================
# Step 2: InMemoryVectorStore — store vectors, query by cosine distance.
# =============================================================================


async def using_vector_store():
    print("\n" + "=" * 60)
    print("Step 2: InMemoryVectorStore (COSINE)")
    print("=" * 60)

    embedder = _get_embedder()
    store = _get_store(dimension=embedder.config.dimension)
    print(f"Created InMemoryVectorStore dim={store.config.dimension}")

    docs_text = [
        "Python is great for data science and machine learning.",
        "JavaScript is the language of the web browser.",
        "A relational database stores data in tables with rows and columns.",
        "PostgreSQL is a popular open-source database.",
        "Docker containers package applications with dependencies.",
    ]

    print("\nEmbedding and inserting documents…")
    for i, text in enumerate(docs_text):
        result = await embedder.embed(text)
        await store.add(
            Document(
                id=f"doc_{i}",
                content=text,
                embedding=result.embedding,
                metadata={"source": "notebook", "index": i},
            )
        )
        print(f"  inserted: {text[:50]}…")

    print("\nSearching for 'database systems'…")
    q = await embedder.embed("database systems")
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
        Python was created by Guido van Rossum and first released in 1991.
        It emphasizes code readability with its notable use of significant
        indentation. Python is dynamically typed and garbage-collected.
        """,
        """
        A vector store keeps embeddings in a structure that supports fast
        nearest-neighbour search. Tulip ships in-memory, pgvector,
        OpenSearch, Qdrant, and Chroma stores behind one interface.
        """,
        """
        Machine learning is a subset of artificial intelligence (AI) that
        provides systems the ability to automatically learn and improve
        from experience without being explicitly programmed.
        """,
    ]

    for doc in knowledge_base:
        ids = await retriever.add_document(doc.strip())
        print(f"  inserted document → {len(ids)} chunk(s)")

    print("\nQuerying: 'When was Python created?'")
    result = await retriever.retrieve("When was Python created?", limit=2)
    for i, doc_result in enumerate(result.documents, 1):
        print(f"\n  result {i} (score={doc_result.score:.4f}):")
        print(f"  {doc_result.document.content[:200]}…")

    print("\nUsing retrieve_text() for the same query on a different prompt:")
    text = await retriever.retrieve_text("What is a vector store?", limit=2)
    print(f"\n{text[:300]}…")


# =============================================================================
# Step 4: Metadata-tagged retrieval — store arbitrary JSON alongside the
#         vector and use it to narrow results beyond similarity alone.
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
            "Python supports async/await syntax for concurrency.",
            {"category": "programming", "language": "python"},
        ),
        (
            "Use pip to install Python packages.",
            {"category": "programming", "language": "python"},
        ),
        (
            "JavaScript uses async/await for async operations.",
            {"category": "programming", "language": "javascript"},
        ),
        (
            "Set up a relational database with these steps.",
            {"category": "database", "type": "relational"},
        ),
        (
            "PostgreSQL is an open-source database.",
            {"category": "database", "type": "postgresql"},
        ),
    ]

    for content, metadata in documents:
        await retriever.add_document(content, metadata=metadata)
        print(f"  inserted: {content[:40]}… {metadata}")

    print("\nQuerying 'async programming'…")
    result = await retriever.retrieve("async programming", limit=3)
    for r in result.documents:
        print(f"  score={r.score:.4f}  {r.document.content[:60]}…")
        print(f"    metadata: {r.document.metadata}")


# =============================================================================
# Main
# =============================================================================


async def main():
    missing = _missing_env()
    if missing:
        print("\n--- Notebook 38: RAG basics ---")
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
