# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Notebook 39: RAG basics on Oracle Database 26ai.

Retrieval-Augmented Generation (RAG) grounds an agent's answers in your
own documents. The pipeline has four steps:

- **Embed** — turn text into vectors with ``OCIEmbeddings`` (Cohere V3
  English, 1024 dims, on the OCI GenAI inference endpoint).
- **Store** — persist the vectors in ``OracleVectorStore``, backed by
  Oracle Database 26ai's native ``VECTOR(N, FLOAT32)`` column.
- **Search** — find the closest vectors with the ``VECTOR_DISTANCE``
  SQL function. The native type and operator are 26ai differentiators
  — no extension required.
- **Generate** — feed the retrieved chunks to the LLM as grounded
  context. (This notebook focuses on steps 1–3; notebook 40 wires it
  into an agent.)

This notebook drives the whole pipeline against a real 26ai instance.
There is no in-memory or local fallback — provision an Autonomous
Database 26ai and set the env vars below before running.

Run it:
    # OCI GenAI is the default for embeddings — auto-detected from ~/.oci/config.
    python examples/notebook_44_rag_basics.py

    # Offline (skips the live demo cleanly when env vars are missing):
    LOCUS_MODEL_PROVIDER=mock python examples/notebook_44_rag_basics.py

Prerequisites:
    export ORACLE_DSN=mydb_low                   # tnsnames alias
    export ORACLE_USER=locus_app                 # least-privileged user
    export ORACLE_PASSWORD='<app-password>'
    export ORACLE_WALLET=~/.oci/wallets/mydb
    export ORACLE_WALLET_PASSWORD='<wallet-pw>'  # if encrypted
    export OCI_PROFILE=<your-profile>
    export OCI_REGION=us-chicago-1
    export OCI_COMPARTMENT=ocid1.compartment.oc1..…
"""

import asyncio
import math
import os
import sys

from locus.rag import OCIEmbeddings, OracleVectorStore, RAGRetriever
from locus.rag.stores.base import Document


_REQUIRED_ENV = (
    "ORACLE_DSN",
    "ORACLE_USER",
    "ORACLE_PASSWORD",
    "ORACLE_WALLET",
    "OCI_COMPARTMENT",
)


def _missing_env() -> list[str]:
    return [name for name in _REQUIRED_ENV if not os.environ.get(name)]


def _get_embedder() -> OCIEmbeddings:
    region = os.environ.get("LOCUS_OCI_REGION") or os.environ.get("OCI_REGION", "us-chicago-1")
    return OCIEmbeddings(
        model_id="cohere.embed-english-v3.0",
        profile_name=os.environ.get("LOCUS_OCI_PROFILE")
        or os.environ.get("OCI_PROFILE", "DEFAULT"),
        auth_type=os.environ.get("LOCUS_OCI_AUTH_TYPE")
        or os.environ.get("OCI_AUTH_TYPE", "api_key"),
        compartment_id=os.environ["OCI_COMPARTMENT"],
        service_endpoint=(f"https://inference.generativeai.{region}.oci.oraclecloud.com"),
    )


def _get_store(table_suffix: str, dimension: int = 1024) -> OracleVectorStore:
    return OracleVectorStore(
        dsn=os.environ["ORACLE_DSN"],
        user=os.environ["ORACLE_USER"],
        password=os.environ["ORACLE_PASSWORD"],
        wallet_location=os.path.expanduser(os.environ["ORACLE_WALLET"]),
        wallet_password=os.environ.get("ORACLE_WALLET_PASSWORD", ""),
        table_name=f"locus_notebook_39_{table_suffix}",
        dimension=dimension,
        distance_metric="COSINE",
    )


# =============================================================================
# Step 1: Embeddings — vectors that capture meaning, not just keywords.
# =============================================================================


async def understand_embeddings():
    print("=" * 60)
    print("Step 1: Embeddings (OCI GenAI · Cohere V3, 1024-dim)")
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
# Step 2: OracleVectorStore — write to a native VECTOR column, query with
#         VECTOR_DISTANCE.
# =============================================================================


async def using_vector_store():
    print("\n" + "=" * 60)
    print("Step 2: OracleVectorStore (native VECTOR, COSINE)")
    print("=" * 60)

    embedder = _get_embedder()
    store = _get_store("manual", dimension=embedder.config.dimension)
    print(f"Created OracleVectorStore table=locus_notebook_39_manual dim={store.config.dimension}")

    docs_text = [
        "Python is great for data science and machine learning.",
        "JavaScript is the language of the web browser.",
        "Oracle Database is an enterprise relational database.",
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

    print(f"\nTotal rows in table: {await store.count()}")


# =============================================================================
# Step 3: RAGRetriever — one object that handles chunking, embedding, and
#         storage for you.
# =============================================================================


async def using_rag_retriever():
    print("\n" + "=" * 60)
    print("Step 3: RAGRetriever over Oracle 26ai")
    print("=" * 60)

    embedder = _get_embedder()
    store = _get_store("retriever", dimension=embedder.config.dimension)

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
        Oracle Cloud Infrastructure (OCI) is a cloud computing service
        offered by Oracle Corporation. It provides servers, storage,
        network, applications and services through a global network of
        Oracle Corporation managed data centers.
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
    text = await retriever.retrieve_text("What is Oracle Cloud?", limit=2)
    print(f"\n{text[:300]}…")


# =============================================================================
# Step 4: Metadata-tagged retrieval — store arbitrary JSON alongside the
#         vector and use it to narrow results beyond similarity alone.
# =============================================================================


async def rag_with_metadata():
    print("\n" + "=" * 60)
    print("Step 4: RAG with metadata in Oracle 26ai")
    print("=" * 60)

    embedder = _get_embedder()
    store = _get_store("metadata", dimension=embedder.config.dimension)
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
            "Set up Oracle Database with these steps.",
            {"category": "database", "type": "oracle"},
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
        print("\n--- Notebook 39: RAG basics on Oracle 26ai ---")
        print(
            "Required environment variables not set; skipping the live "
            "demo so this file still runs cleanly in CI.\n"
        )
        for name in missing:
            print(f"  - {name}")
        print(
            "\nProvision an Autonomous Database 26ai + an OCI GenAI "
            "compartment, then set the variables above and re-run."
        )
        return

    await understand_embeddings()
    await using_vector_store()
    await using_rag_retriever()
    await rag_with_metadata()

    print("\n" + "=" * 60)
    print("Notebook 39 complete — every vector now lives in Oracle 26ai.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
