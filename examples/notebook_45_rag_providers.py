# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Notebook 40: RAG providers — choose embeddings and tune the Oracle 26ai store.

Production RAG on OCI is two pieces. ``OracleVectorStore`` is the
default backend across this notebook series — backed by Oracle
Database 26ai with its native ``VECTOR(N, FLOAT32)`` column and the
``VECTOR_DISTANCE`` SQL function. Other backends (pgvector,
OpenSearch) are valid alternatives; the Locus interface is identical.

- **Embeddings** — ``OCIEmbeddings`` on the OCI GenAI inference endpoint.
  Cohere V3 for English (1024 dims), Cohere V4 for multilingual.
- **Vector store** — ``OracleVectorStore`` against an Autonomous
  Database 26ai. Every section in this notebook talks to your ADB.

What each part covers:

- Part 1 — embedding-model selection (Cohere V3 vs V4 dimensions).
- Part 2 — distance metric choices (COSINE / DOT / EUCLIDEAN).
- Part 3 — attaching to an existing langchain_oracledb-style table via
  column-name overrides.
- Part 4 — batch ingest, ``count()``, ``clear()``.

Run it:
    # OCI GenAI is the default for embeddings — auto-detected from ~/.oci/config.
    python examples/notebook_45_rag_providers.py

    # Offline (skips the live demo cleanly when env vars are missing):
    LOCUS_MODEL_PROVIDER=mock python examples/notebook_45_rag_providers.py

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


def _embedder(model_id: str) -> OCIEmbeddings:
    region = os.environ.get("LOCUS_OCI_REGION") or os.environ.get("OCI_REGION", "us-chicago-1")
    return OCIEmbeddings(
        model_id=model_id,
        profile_name=os.environ.get("LOCUS_OCI_PROFILE")
        or os.environ.get("OCI_PROFILE", "DEFAULT"),
        auth_type=os.environ.get("LOCUS_OCI_AUTH_TYPE")
        or os.environ.get("OCI_AUTH_TYPE", "api_key"),
        compartment_id=os.environ["OCI_COMPARTMENT"],
        service_endpoint=(f"https://inference.generativeai.{region}.oci.oraclecloud.com"),
    )


def _store(
    *,
    table_suffix: str,
    dimension: int,
    distance: str = "COSINE",
    **overrides,
) -> OracleVectorStore:
    base = {
        "dsn": os.environ["ORACLE_DSN"],
        "user": os.environ["ORACLE_USER"],
        "password": os.environ["ORACLE_PASSWORD"],
        "wallet_location": os.path.expanduser(os.environ["ORACLE_WALLET"]),
        "wallet_password": os.environ.get("ORACLE_WALLET_PASSWORD", ""),
        "table_name": f"locus_notebook_45_{table_suffix}",
        "dimension": dimension,
        "distance_metric": distance,
    }
    base.update(overrides)
    return OracleVectorStore(**base)


CORPUS = [
    "Oracle 26ai supports HNSW and IVF vector indexes on VECTOR columns.",
    "VECTOR_DISTANCE returns either COSINE, DOT, or EUCLIDEAN distance.",
    "Autonomous Database wallets bundle a tnsnames.ora alias per service.",
    "Locus exposes Oracle through OracleVectorStore over python-oracledb.",
    "Cohere V4 multilingual embeddings produce 1024-dim vectors.",
    "Cohere V3 English embeddings also produce 1024-dim vectors.",
]


# =============================================================================
# Part 1: Cohere V3 (English) vs V4 (multilingual) against the same corpus.
# =============================================================================


async def part1_embedding_models():
    print("=" * 60)
    print("Part 1: OCIEmbeddings — Cohere V3 (english) vs V4 (multilingual)")
    print("=" * 60)

    for model_id, suffix in [
        ("cohere.embed-english-v3.0", "embed_v3"),
        ("cohere.embed-v4.0", "embed_v4"),
    ]:
        embedder = _embedder(model_id)
        store = _store(table_suffix=suffix, dimension=embedder.config.dimension)
        retriever = RAGRetriever(embedder=embedder, store=store)
        print(f"\n  {model_id} → dim={embedder.config.dimension}")
        await retriever.add_documents(CORPUS)
        hits = await retriever.retrieve("vector index types in Oracle 26ai", limit=2)
        for i, h in enumerate(hits.documents, start=1):
            print(f"    #{i} score={h.score:.4f} {h.document.content[:70]}…")


# =============================================================================
# Part 2: Distance metric variants. COSINE is the default — DOT and
#         EUCLIDEAN are alternative shapes ``VECTOR_DISTANCE`` supports.
# =============================================================================


async def part2_distance_metrics():
    print("\n" + "=" * 60)
    print("Part 2: Distance metric variants on the same corpus")
    print("=" * 60)

    embedder = _embedder("cohere.embed-english-v3.0")
    query = "vector index types in Oracle 26ai"

    for metric, suffix in [
        ("COSINE", "metric_cos"),
        ("DOT", "metric_dot"),
        ("EUCLIDEAN", "metric_euc"),
    ]:
        store = _store(
            table_suffix=suffix,
            dimension=embedder.config.dimension,
            distance=metric,
        )
        retriever = RAGRetriever(embedder=embedder, store=store)
        await retriever.add_documents(CORPUS)
        hits = await retriever.retrieve(query, limit=2)
        top = hits.documents[0]
        print(f"  {metric}: top score={top.score:.4f}  → {top.document.content[:60]}…")


# =============================================================================
# Part 3: Attach to a foreign-schema table — point OracleVectorStore at
#         a table written by another tool (langchain_oracledb here) by
#         overriding column names.
# =============================================================================


async def part3_foreign_schema():
    print("\n" + "=" * 60)
    print("Part 3: Attach to an existing langchain_oracledb-style table")
    print("=" * 60)

    embedder = _embedder("cohere.embed-english-v3.0")
    foreign = _store(
        table_suffix="foreign",
        dimension=embedder.config.dimension,
        # langchain_oracledb's column names differ from the Locus defaults.
        content_column="text",
        created_at_column=None,
        auto_create_table=True,
    )

    for i, text in enumerate(CORPUS[:4]):
        emb = await embedder.embed(text)
        await foreign.add(
            Document(
                id=f"foreign_{i}",
                content=text,
                embedding=emb.embedding,
                metadata={"shard": i % 2},
            )
        )

    q = await embedder.embed("how do Oracle vector indexes work?")
    hits = await foreign.search(query_embedding=q.embedding, limit=3)
    print(f"  Searched {await foreign.count()} rows in the foreign-schema table:")
    for i, hit in enumerate(hits, start=1):
        print(f"    #{i} score={hit.score:.4f}  {hit.document.content[:70]}…")


# =============================================================================
# Part 4: Batch lifecycle — add_documents, count(), clear().
# =============================================================================


async def part4_batch():
    print("\n" + "=" * 60)
    print("Part 4: Batch ingest + count + clear")
    print("=" * 60)

    embedder = _embedder("cohere.embed-english-v3.0")
    store = _store(table_suffix="batch", dimension=embedder.config.dimension)
    retriever = RAGRetriever(embedder=embedder, store=store)

    await retriever.add_documents(CORPUS)
    print(f"  After add_documents: rows = {await store.count()}")
    await store.clear()
    print(f"  After clear():       rows = {await store.count()}")


# =============================================================================
# Main
# =============================================================================


async def main():
    missing = _missing_env()
    if missing:
        print("\n--- Notebook 40: RAG providers on Oracle 26ai ---")
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

    await part1_embedding_models()
    await part2_distance_metrics()
    await part3_foreign_schema()
    await part4_batch()

    print("\n" + "=" * 60)
    print("Notebook 40 complete — every variant runs on Oracle 26ai.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
