#!/usr/bin/env python3
# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Notebook 41: a Locus agent that retrieves over Oracle Database 26ai.

The Locus agent runtime is the protagonist here. Oracle 26ai is the
durable substrate — embeddings live in a native ``VECTOR(N, FLOAT32)``
column, similarity is the ``VECTOR_DISTANCE`` SQL function — but what
this notebook shows is the *agent* deciding when to search the corpus,
firing a real tool call, and grounding its reply in what came back.

Key concepts:

- ``OracleVectorStore`` opens an async pool against an Autonomous
  Database wallet, auto-creates a ``VECTOR(1024, FLOAT32)`` table on
  first use, and serves cosine similarity via ``VECTOR_DISTANCE``.
- ``OCIEmbeddings`` produces 1024-dim Cohere V3 vectors on OCI GenAI;
  ``RAGRetriever`` glues the two together.
- ``create_rag_tool(retriever, ...)`` returns a ``@tool``-decorated
  callable. Hand it to an ``Agent`` and the model picks when to call
  it — Locus surfaces every retrieval as ``ToolStartEvent`` /
  ``ToolCompleteEvent`` in the event stream, so you can see the agent
  driving the database.
- The same retriever, the same tool, the same agent loop. Swap the
  store import to OpenSearch, pgvector, or in-memory and nothing else
  changes.

Run it::

    # Database side — Autonomous Database wallet (TLS) + credentials.
    export ORACLE_DSN=mydb_low                  # tnsnames alias in the wallet
    export ORACLE_USER=locus_app                # least-privileged app schema
    export ORACLE_PASSWORD='<app-password>'
    export ORACLE_WALLET=~/.oci/wallets/mydb    # directory with tnsnames.ora
    export ORACLE_WALLET_PASSWORD='<wallet-pw>' # if the wallet is encrypted

    # Embedding + model side — OCI GenAI in us-chicago-1.
    export OCI_PROFILE=DEFAULT
    export OCI_REGION=us-chicago-1
    export OCI_COMPARTMENT=ocid1.compartment.oc1..xxx
    export OCI_AUTH_TYPE=api_key                # or security_token

    python examples/notebook_06_oracle_26ai_rag.py

If ``ORACLE_DSN`` / ``ORACLE_PASSWORD`` / ``OCI_COMPARTMENT`` aren't set
the notebook prints the wiring snippet and exits cleanly — no
traceback, no half-initialised state.

**Schema hygiene.** This notebook uses ``auto_create_table=True`` so the
demo provisions the table on first run. For production, create the
table out-of-band as a least-privileged app schema owner and set
``auto_create_table=False`` — see ``docs/concepts/rag.md`` for the
``CREATE USER locus_app`` and ``CREATE VECTOR INDEX`` DDL.

Difficulty: Intermediate. Self-contained — no prior notebook required.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from config import get_model, print_config

from locus.agent import Agent
from locus.rag import OCIEmbeddings, OracleVectorStore, RAGRetriever, create_rag_tool


_REQUIRED_ENV = (
    "ORACLE_DSN",
    "ORACLE_USER",
    "ORACLE_PASSWORD",
    "ORACLE_WALLET",
    "OCI_COMPARTMENT",
)


CORPUS = [
    "Oracle 26ai introduces a native VECTOR(N, FLOAT32) column type "
    "with HNSW and IVF index organisations.",
    "VECTOR_DISTANCE(embedding, :query, COSINE) returns the cosine "
    "distance between a stored vector and a query vector.",
    "Autonomous Database wallets bundle a tnsnames.ora alias per "
    "consumer group — _low, _medium, _high, _tp, _tpurgent.",
    "Locus exposes Oracle 26ai through OracleVectorStore; the same "
    "RAGRetriever drives every backend uniformly.",
    "A vector index on a VECTOR column is built asynchronously after "
    "the first INSERT, so the first query may be slower than steady-state.",
    "PostgreSQL stores embeddings through the pgvector extension, not a native type.",
]


QUERIES = [
    "How does Oracle 26ai store embeddings under the hood?",
    "What index organisations are supported for the VECTOR column?",
    "Why might the very first similarity query feel slower than later ones?",
]


def _missing_env() -> list[str]:
    return [name for name in _REQUIRED_ENV if not os.environ.get(name)]


def _print_skip_banner(missing: list[str]) -> None:
    print("\n--- Notebook 06: Oracle 26ai RAG agent ---")
    print(
        "Required environment variables not set; skipping the live demo so "
        "this file still runs cleanly in CI.\n"
    )
    print("Missing:")
    for name in missing:
        print(f"  - {name}")
    print(
        "\nProvision an Autonomous Database, drop its wallet under "
        "$ORACLE_WALLET, then set the variables above and re-run."
    )
    print("\nMinimal wiring (what the live path below builds):")
    print(
        """
    from locus.agent import Agent
    from locus.rag import (
        OCIEmbeddings, OracleVectorStore, RAGRetriever, create_rag_tool,
    )

    retriever = RAGRetriever(
        embedder=OCIEmbeddings(model_id="cohere.embed-english-v3.0", ...),
        store=OracleVectorStore(dsn=..., user=..., dimension=1024, ...),
    )
    await retriever.add_documents(corpus)

    knowledge = create_rag_tool(retriever, name="search_oracle_26ai_corpus")
    agent = Agent(model=model, tools=[knowledge], system_prompt="...")

    async for ev in agent.run("How does Oracle 26ai store embeddings?"):
        print(ev.event_type)
        """.rstrip()
    )


async def _seed_corpus(retriever: RAGRetriever) -> None:
    print(f"\nSeeding {len(CORPUS)} passages into the VECTOR(1024) table…")
    await retriever.add_documents(CORPUS)


async def _drive_agent(agent: Agent, prompt: str) -> None:
    """Run the agent and print every relevant lifecycle event.

    The point of this loop is to make the *agent's* decision-making
    visible: when it plans a tool call, the ``ToolStartEvent`` /
    ``ToolCompleteEvent`` pair show the Oracle 26ai retrieval happening
    in real time. The retriever itself is just the substrate the agent
    chose to reach for.
    """
    print(f"\n--- User: {prompt} ---")
    final = ""
    async for event in agent.run(prompt):
        et = event.event_type
        if et == "tool_start":
            args = getattr(event, "arguments", None) or {}
            query = args.get("query") if isinstance(args, dict) else None
            print(f"  [agent → Oracle 26ai] tool_start: {event.tool_name}(query={query!r})")
        elif et == "tool_complete":
            # ToolCompleteEvent.result is always a string (JSON-encoded for
            # dict/list return values). Decode it back so we can show the
            # agent's view of the retrieval.
            raw = getattr(event, "result", "") or ""
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {}
            hits = payload.get("results", []) if isinstance(payload, dict) else []
            print(f"  [agent ← Oracle 26ai] tool_complete: {len(hits)} hit(s)")
            for i, hit in enumerate(hits, start=1):
                snippet = (hit.get("content") or "").replace("\n", " ")[:80]
                print(f"      #{i}  score={hit.get('score')!r}  {snippet}")
        elif et == "terminate":
            final = getattr(event, "final_message", "") or ""
    if final:
        print(f"\nAgent: {final.strip()}")


async def main() -> None:
    missing = _missing_env()
    if missing:
        _print_skip_banner(missing)
        return

    print_config()

    profile = os.environ.get("LOCUS_OCI_PROFILE") or os.environ.get("OCI_PROFILE", "DEFAULT")
    region = os.environ.get("LOCUS_OCI_REGION") or os.environ.get("OCI_REGION", "us-chicago-1")
    compartment = os.environ["OCI_COMPARTMENT"]
    auth_type = os.environ.get("LOCUS_OCI_AUTH_TYPE") or os.environ.get("OCI_AUTH_TYPE", "api_key")

    embedder = OCIEmbeddings(
        # 1024-dim Cohere V3 on OCI GenAI — matches the VECTOR(1024, FLOAT32) column.
        model_id="cohere.embed-english-v3.0",
        profile_name=profile,
        auth_type=auth_type,
        compartment_id=compartment,
        service_endpoint=(f"https://inference.generativeai.{region}.oci.oraclecloud.com"),
    )

    print("Opening pool against Oracle 26ai…")
    store = OracleVectorStore(
        dsn=os.environ["ORACLE_DSN"],
        user=os.environ["ORACLE_USER"],
        password=os.environ["ORACLE_PASSWORD"],
        wallet_location=os.path.expanduser(os.environ["ORACLE_WALLET"]),
        wallet_password=os.environ.get("ORACLE_WALLET_PASSWORD", ""),
        # Demo defaults. For production, pre-create the table and pass
        # auto_create_table=False so the runtime user only needs DML.
        table_name="locus_notebook_06",
        dimension=1024,
        distance_metric="COSINE",
    )

    retriever = RAGRetriever(embedder=embedder, store=store)
    await _seed_corpus(retriever)

    # The agent only ever sees the tool surface — never the retriever
    # directly. ``create_rag_tool`` returns a Locus ``@tool`` that wraps
    # ``retriever.retrieve(...)`` and tags the result with metadata,
    # scores, and a security note (retrieved content is untrusted data).
    knowledge_tool = create_rag_tool(
        retriever,
        name="search_oracle_26ai_corpus",
        description=(
            "Search the Oracle 26ai vector knowledge base. Returns up to N "
            "passages with cosine-similarity scores against a 1024-dim "
            "Cohere V3 query embedding. Call this whenever the answer might "
            "depend on Oracle 26ai's vector storage, index types, or wallet "
            "details — do not guess."
        ),
        limit=3,
        threshold=0.0,
    )

    agent = Agent(
        model=get_model(max_tokens=300),
        tools=[knowledge_tool],
        system_prompt=(
            "You are an Oracle 26ai field engineer. When a question touches "
            "Oracle 26ai vector storage, index types, or Autonomous Database "
            "wallets, call search_oracle_26ai_corpus first and ground your "
            "answer in the returned passages. Cite the passage you used. "
            "Never invent SQL syntax — quote what the retrieval returned."
        ),
        max_iterations=4,
    )

    for query in QUERIES:
        await _drive_agent(agent, query)

    print(
        "\nOne agent. One Oracle 26ai pool. Every reply above was the "
        "agent deciding to issue VECTOR_DISTANCE / COSINE on its own — "
        "the retriever, the tool, and the model are all Locus primitives."
    )

    await store.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
