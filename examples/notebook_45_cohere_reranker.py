# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Notebook 45: retrieve-then-rerank on Oracle 26ai with Cohere Reranker V4.

Embedding-only retrieval is often not enough for production RAG. The
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
- ``OCICohereReranker(model="cohere.rerank-v4.0-fast", ...)`` calls OCI
  GenAI's on-demand rerank-v4 endpoint. Same auth surface as embeddings.
- ``RAGRetriever(embedder=..., store=..., reranker=...,
  rerank_candidate_pool=N)`` is the one wiring change — the retriever
  fetches ``N`` candidates from the store, reranks them, then returns
  ``limit`` to the caller.
- The run prints where the canonical hepcidin passage lands in each
  pass (``None`` when it falls outside the four baseline hits), so any
  lift from reranking shows as a change of rank.

Run it:
    export OCI_PROFILE=DEFAULT       # or your locus_app-typed profile
    export OCI_AUTH_TYPE=api_key     # or security_token
    export OCI_REGION=us-chicago-1
    export OCI_COMPARTMENT=ocid1.compartment.oc1..xxx
    # Plus the Oracle 26ai connection (see notebook 38):
    export ORACLE_DSN=mydb_low
    export ORACLE_USER=locus_app
    export ORACLE_PASSWORD='<app-password>'
    export ORACLE_WALLET=~/.oci/wallets/mydb     # mTLS only; a TLS-only database has none
    export ORACLE_WALLET_PASSWORD='<wallet-pw>'  # if the wallet is encrypted
    # TLS-only (no wallet): a connect string in ORACLE_DSN, or an alias + ORACLE_CONFIG_DIR
    python examples/notebook_45_cohere_reranker.py

Difficulty: Intermediate. Self-contained — no prior notebook required.
"""

from __future__ import annotations

import asyncio
import os

from locus.rag import (
    OCICohereReranker,
    OCIEmbeddings,
    OracleVectorStore,
    RAGRetriever,
)


_REQUIRED_ENV = (
    "ORACLE_DSN",
    "ORACLE_USER",
    "ORACLE_PASSWORD",
    "OCI_COMPARTMENT",
)


def _missing_env() -> list[str]:
    missing = [name for name in _REQUIRED_ENV if not os.environ.get(name)]
    # A tnsnames alias in ORACLE_DSN needs a directory to resolve it: the
    # mTLS wallet, or ORACLE_CONFIG_DIR / TNS_ADMIN for a TLS-only database.
    # A connect descriptor or an Easy Connect address (host:port/service)
    # needs neither, and an unset DSN is already reported above.
    dsn = os.environ.get("ORACLE_DSN", "")
    if (
        dsn
        and "(" not in dsn
        and "/" not in dsn
        and ":" not in dsn
        and not any(
            os.environ.get(name) for name in ("ORACLE_WALLET", "ORACLE_CONFIG_DIR", "TNS_ADMIN")
        )
    ):
        missing.append("ORACLE_WALLET (mTLS) or ORACLE_CONFIG_DIR (TLS-only alias)")
    return missing


def _oracle_connection() -> dict[str, str]:
    """dsn / user / password, plus the TNS directory or wallet the environment names."""
    kwargs = {
        "dsn": os.environ["ORACLE_DSN"],
        "user": os.environ["ORACLE_USER"],
        "password": os.environ["ORACLE_PASSWORD"],
    }
    if os.environ.get("ORACLE_CONFIG_DIR"):
        kwargs["config_dir"] = os.path.expanduser(os.environ["ORACLE_CONFIG_DIR"])
    if os.environ.get("ORACLE_WALLET"):  # mTLS only; a TLS-only database has no wallet
        kwargs["wallet_location"] = os.path.expanduser(os.environ["ORACLE_WALLET"])
        kwargs["wallet_password"] = os.environ.get("ORACLE_WALLET_PASSWORD", "")
    return kwargs


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
        print("\n--- Notebook 45: Cohere Reranker V4 on Oracle 26ai ---")
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
        model_id="cohere.embed-v4.0",
        profile_name=profile,
        auth_type=auth_type,
        compartment_id=compartment,
        service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
    )

    store = OracleVectorStore(
        **_oracle_connection(),
        table_name="locus_notebook_45",
        dimension=1536,  # embed-v4.0 width
        distance_metric="COSINE",
    )
    try:
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
        reranker = OCICohereReranker(
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
            f"\nCanonical hepcidin passage rank: baseline = {baseline_rank}, "
            f"reranked = {reranked_rank}"
        )
        if reranked_rank == 1 and baseline_rank != 1:
            print(
                "Reranking promoted the canonical answer to the top position even though "
                "embedding similarity placed less-relevant passages above it.\n"
            )
        else:
            print(
                "No lift to report on this run: the canonical passage ranked "
                f"{baseline_rank} by embedding similarity and {reranked_rank} after reranking.\n"
            )

    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
