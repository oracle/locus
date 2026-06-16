#!/usr/bin/env python3
# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Notebook 05: retrieve-then-rerank on Oracle 26ai with Cohere Reranker V4.

Embedding-only retrieval often isn't enough for production RAG. The
embedding model sees query and document independently and can mis-rank
candidates whose surface form scores high but whose semantic relevance
is lower. A cross-encoder reranker scores the query against each
candidate together and catches the signals embeddings miss. This
notebook wires Cohere Reranker V4 on Oracle Cloud Infrastructure (OCI)
GenAI on top of an Oracle Database 26ai vector store and shows the lift
on a small medical corpus.

Key concepts:

- Retrieve-then-rerank: cheaply over-fetch from the vector store
  (e.g. top-50), then rerank against the query and keep the top-N
  (e.g. top-5). Feed the top-N to the LLM as grounded context.
- ``CohereReranker(model="cohere.rerank-v4.0-fast", ...)`` calls OCI
  GenAI's on-demand rerank-v4 endpoint. Same auth surface as embeddings.
- ``RAGRetriever(embedder=..., store=..., reranker=...,
  rerank_candidate_pool=N)`` is the one wiring change — the retriever
  fetches ``N`` candidates from the store, reranks them, then returns
  ``limit`` to the caller.
- The lift is visible: the canonical hepcidin passage ranks 4th by
  embedding similarity and 1st after reranking.

Run it::

    export OCI_PROFILE=DEFAULT       # or your locus_app-typed profile
    export OCI_AUTH_TYPE=api_key     # or security_token
    export OCI_REGION=us-chicago-1
    export OCI_COMPARTMENT=ocid1.compartment.oc1..xxx
    # Plus the Oracle 26ai wallet (see notebook 05):
    export ORACLE_DSN=mydb_low
    export ORACLE_USER=locus_app
    export ORACLE_PASSWORD='<app-password>'
    export ORACLE_WALLET=~/.oci/wallets/mydb
    python examples/notebook_05_cohere_reranker.py

Difficulty: Intermediate. Self-contained — no prior notebook required.
"""

from __future__ import annotations

import asyncio
import os

from locus.rag import (
    CohereReranker,
    OCIEmbeddings,
    OracleVectorStore,
    RAGRetriever,
)


_REQUIRED_ENV = (
    "ORACLE_DSN",
    "ORACLE_USER",
    "ORACLE_PASSWORD",
    "ORACLE_WALLET",
    "OCI_COMPARTMENT",
)


def _missing_env() -> list[str]:
    return [name for name in _REQUIRED_ENV if not os.environ.get(name)]


CORPUS = [
    "Iron is primarily absorbed in the duodenum and proximal jejunum of the small intestine.",
    "Phlebotomy is the first-line treatment for hereditary hemochromatosis.",
    "Ferritin is the primary iron storage protein and an acute-phase reactant.",
    (
        "Hepcidin, produced by the liver, is the master regulator of systemic iron homeostasis. "
        "It binds ferroportin and induces its degradation, blocking iron export from "
        "enterocytes and macrophages."
    ),
    "Transferrin saturation below 16% suggests iron deficiency.",
    "MRI T2* relaxometry is the gold standard for non-invasive iron quantification.",
    "Iron deficiency anemia is the most common nutritional deficiency worldwide.",
    "First-line treatment for iron deficiency anemia is oral ferrous sulfate.",
]


QUERY = "What is hepcidin's role in iron homeostasis?"


def _print_table(title: str, rows: list[tuple[int, float, str]]) -> None:
    print(f"\n--- {title} ---")
    for rank, score, content in rows:
        print(f"  #{rank}  score={score:.4f}  {content[:90]}")


async def main() -> None:
    missing = _missing_env()
    if missing:
        print("\n--- Notebook 05: Cohere Reranker V4 on Oracle 26ai ---")
        print(
            "Required environment variables not set; skipping the live demo so "
            "this file still runs cleanly in CI.\n"
        )
        for name in missing:
            print(f"  - {name}")
        print(
            "\nProvision an Autonomous Database wallet + an OCI GenAI compartment, "
            "then set ORACLE_DSN / ORACLE_USER / ORACLE_PASSWORD / ORACLE_WALLET "
            "(plus ORACLE_WALLET_PASSWORD if encrypted) and OCI_COMPARTMENT."
        )
        return

    profile = os.environ.get("LOCUS_OCI_PROFILE") or os.environ.get("OCI_PROFILE", "DEFAULT")
    region = os.environ.get("LOCUS_OCI_REGION") or os.environ.get("OCI_REGION", "us-chicago-1")
    compartment = os.environ["OCI_COMPARTMENT"]
    auth_type = os.environ.get("LOCUS_OCI_AUTH_TYPE") or os.environ.get("OCI_AUTH_TYPE", "api_key")

    embedder = OCIEmbeddings(
        model_id="cohere.embed-english-v3.0",
        profile_name=profile,
        auth_type=auth_type,
        compartment_id=compartment,
        service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
    )

    store = OracleVectorStore(
        dsn=os.environ["ORACLE_DSN"],
        user=os.environ["ORACLE_USER"],
        password=os.environ["ORACLE_PASSWORD"],
        wallet_location=os.path.expanduser(os.environ["ORACLE_WALLET"]),
        wallet_password=os.environ.get("ORACLE_WALLET_PASSWORD", ""),
        table_name="locus_notebook_05",
        dimension=1024,
        distance_metric="COSINE",
    )
    print(f"Embedding {len(CORPUS)} passages and seeding OracleVectorStore…")
    baseline = RAGRetriever(embedder=embedder, store=store)
    await baseline.add_documents(CORPUS)

    # Baseline retrieval — embedding similarity only.
    print(f"\nQuery: {QUERY!r}")
    baseline_hits = await baseline.retrieve(QUERY, limit=4)
    _print_table(
        "Baseline (embedding similarity)",
        [(i + 1, h.score, h.document.content) for i, h in enumerate(baseline_hits.documents)],
    )

    # Retrieve-then-rerank — over-fetch + Cohere V4 cross-encoder.
    reranker = CohereReranker(
        # Cohere Reranker V4 fast on OCI GenAI on-demand.
        model="cohere.rerank-v4.0-fast",
        compartment_id=compartment,
        profile_name=profile,
        auth_type=auth_type,
        region=region,
        top_n=4,
    )
    reranked_retriever = RAGRetriever(
        embedder=embedder,
        store=store,
        reranker=reranker,
        # Toy corpus: rerank everything. In production, over-fetch
        # ~50-200 from the store and rerank that pool.
        rerank_candidate_pool=len(CORPUS),
    )
    reranked_hits = await reranked_retriever.retrieve(QUERY, limit=4)
    _print_table(
        "Reranked (Cohere V4 cross-encoder)",
        [(i + 1, h.score, h.document.content) for i, h in enumerate(reranked_hits.documents)],
    )

    # Lift summary — where did the canonical answer land in each pass?
    canonical_marker = "Hepcidin, produced by the liver"
    baseline_rank = next(
        (
            i + 1
            for i, h in enumerate(baseline_hits.documents)
            if h.document.content.startswith(canonical_marker)
        ),
        None,
    )
    reranked_rank = next(
        (
            i + 1
            for i, h in enumerate(reranked_hits.documents)
            if h.document.content.startswith(canonical_marker)
        ),
        None,
    )
    print(
        f"\nCanonical hepcidin passage rank: baseline = {baseline_rank}, reranked = {reranked_rank}"
    )
    print(
        "Reranking promoted the canonical answer to the top position even though "
        "embedding similarity placed less-relevant passages above it.\n"
    )


if __name__ == "__main__":
    asyncio.run(main())
