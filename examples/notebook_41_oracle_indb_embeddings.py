# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Notebook 41: a Locus agent doing data-residency-aware RAG in-database.

Oracle 23ai / 26ai can host ONNX embedding models *inside* the
database via ``DBMS_VECTOR.LOAD_ONNX_MODEL``. When the embedding model
lives in the DB, the application ships text into the server, the
database produces the vector locally, and the caller never sees the
text leave Oracle's process. Pair that with ``OracleVectorStore`` and
the entire embedding + similarity-search half of the RAG pipeline runs
inside the database — the canonical pattern when data residency rules
forbid sending corpus text to OCI GenAI / OpenAI / a third-party
endpoint.

This notebook puts a Locus ``Agent`` on top of that stack so the
agent's retrieval calls land on Oracle 26ai. The agent picks the
query, fires a tool, and the database does the embedding + the
cosine search. The retrieved passages then ground the agent's answer.

(Caveat: the *LLM* the agent uses for reasoning still sees the
question + the retrieved passages, so this is residency for embeddings
and search, not for the chat prompt. Pair with an in-house OCI model
in the same compartment if you need end-to-end residency.)

Key concepts:

- ``OracleInDBEmbeddings(model_name=..., dimension=...)`` binds the
  embedder to an in-DB ONNX model. ``embed(text)`` calls
  ``DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDING`` server-side.
- ``OracleVectorStore(dimension=...)`` is the persistence side of the
  pair. Same wallet, same pool kwargs.
- ``RAGRetriever(embedder=indb_embedder, store=store)`` glues them
  together. ``create_rag_tool(retriever)`` exposes the pair as a
  single Locus ``@tool``.
- ``Agent(tools=[tool], ...)`` lets the model decide when to search.
  Every search round-trips through the database — no external
  embedding endpoint.

Prerequisites
-------------

The ONNX model must be loaded into the database before this notebook
can run a query. On Autonomous Database the one-time path is a single
call against Oracle's prebuilt model in Object Storage (the URL is in the
Oracle Machine Learning blog post "Pre-built Embedding Generation model
for Oracle Database 26ai"; it is a pre-authenticated URL, so no
credential)::

    BEGIN
        DBMS_VECTOR.LOAD_ONNX_MODEL_CLOUD(
            model_name => 'ALL_MINILM_L12_V2',
            credential => NULL,
            uri        => 'https://.../all_MiniLM_L12_v2.onnx');
    END;
    /

On-premises, copy the file to a directory object and load it from there::

    BEGIN
        DBMS_VECTOR.LOAD_ONNX_MODEL(
            directory  => 'DM_DUMP',
            file_name  => 'all_MiniLM_L12_v2.onnx',
            model_name => 'ALL_MINILM_L12_V2');
    END;
    /

And the schema running this notebook needs the grants::

    GRANT EXECUTE ON DBMS_VECTOR_CHAIN TO locus_app;
    GRANT SELECT ON MINING MODEL ALL_MINILM_L12_V2 TO locus_app;

The notebook accepts ``OCI_INDB_MODEL`` as an env override for the
model name (default ``ALL_MINILM_L12_V2``) and ``OCI_INDB_DIM`` for its
output width (default 384) — change both together. If the model isn't loaded
in the target DB, the live path catches the resulting ``ORA-…``,
prints a skip banner, and exits cleanly.

Run it:
    export ORACLE_DSN=mydb_low                   # tnsnames alias in the wallet
    export ORACLE_USER=locus_app                 # least-privileged app schema
    export ORACLE_PASSWORD='<app-password>'
    export ORACLE_WALLET=~/.oci/wallets/mydb     # directory holding tnsnames.ora
    export ORACLE_WALLET_PASSWORD='<wallet-pw>'  # if the wallet is encrypted
    export OCI_INDB_MODEL=ALL_MINILM_L12_V2      # optional — defaults to this
    export OCI_INDB_DIM=384                      # optional — must match the model

    python examples/notebook_41_oracle_indb_embeddings.py

Difficulty: Intermediate. Self-contained — no prior notebook required.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from config import get_model, print_config

from locus.agent import Agent
from locus.rag import OracleVectorStore, RAGRetriever, create_rag_tool
from locus.rag.embeddings.oracle_indb import OracleInDBEmbeddings


_REQUIRED_ENV = (
    "ORACLE_DSN",
    "ORACLE_USER",
    "ORACLE_PASSWORD",
    "ORACLE_WALLET",
)


_DEFAULT_MODEL = "ALL_MINILM_L12_V2"
# 384 is the all-MiniLM-L12-v2 output width; a different OCI_INDB_MODEL needs
# OCI_INDB_DIM set to its width, or the VECTOR column and the embedder disagree.
_DEFAULT_DIM = 384


CORPUS = [
    "Oracle Database 26ai exposes server-side embedding via "
    "DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDING and UTL_TO_EMBEDDINGS.",
    "DBMS_VECTOR.LOAD_ONNX_MODEL imports an ONNX model into the database "
    "as a mining model. The text never leaves the DB at embedding time.",
    "The ALL_MINILM_L12_V2 ONNX model emits 384-dim sentence embeddings "
    "and is the default model in 26ai data-residency demos.",
    "Pair OracleInDBEmbeddings with OracleVectorStore(dimension=384) and "
    "the whole embedding + similarity-search half of the pipeline stays "
    "inside Oracle 26ai.",
    "VECTOR_DISTANCE(col, :query, COSINE) is the SQL function that scores "
    "stored vectors against the in-DB-generated query vector.",
]


QUERIES = (
    "How does Oracle 26ai keep embedding text inside the database?",
    "Which ONNX model does the 26ai data-residency demo default to, and what "
    "dimension does it emit?",
)


def _missing_env() -> list[str]:
    return [name for name in _REQUIRED_ENV if not os.environ.get(name)]


def _print_skip_banner(missing: list[str]) -> None:
    print("\n--- Notebook 41: OracleInDBEmbeddings agent ---")
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
    print(
        "\nPrerequisite: the ONNX model must be loaded into the DB once and the "
        "schema granted access — see the Prerequisites section of this file."
    )
    print("\nMinimal wiring (what the live path below builds):")
    print(
        """
    from locus.agent import Agent
    from locus.rag import OracleVectorStore, RAGRetriever, create_rag_tool
    from locus.rag.embeddings.oracle_indb import OracleInDBEmbeddings

    embedder = OracleInDBEmbeddings(model_name="ALL_MINILM_L12_V2",
                                    dimension=384, dsn=..., ...)
    store = OracleVectorStore(dimension=384, ...)
    retriever = RAGRetriever(embedder=embedder, store=store)
    await retriever.add_documents(corpus)
    tool = create_rag_tool(retriever, name="search_indb_corpus")
    agent = Agent(model=model, tools=[tool], ...)
    agent.run_sync("...")
        """.rstrip()
    )


def _print_model_skip_banner(model: str, exc: Exception) -> None:
    print("\n--- Notebook 41: OracleInDBEmbeddings agent ---")
    print(
        f"The ONNX model {model!r} isn't loaded (or accessible) in the "
        "target database, so the live path can't run.\n"
    )
    print(f"Driver / DB reported:\n  {type(exc).__name__}: {exc}")
    print(
        "\nLoad the model and grant access as described in the Prerequisites "
        "section of this file, then re-run. Or point OCI_INDB_MODEL (and "
        "OCI_INDB_DIM) at a model that's already loaded in your DB."
    )


def _is_missing_model_error(exc: BaseException) -> bool:
    """Heuristic: did the DB tell us the ONNX model isn't there?"""
    text = str(exc)
    return any(
        marker in text
        for marker in (
            "ORA-40284",  # mining model does not exist
            "ORA-04063",  # the model's wrapper object is invalid / has errors
            "ORA-20000",  # raise_application_error envelopes DBMS_VECTOR_CHAIN
            "ORA-20100",  # wraps around a model it cannot find or execute
        )
    )


async def _drive_agent(agent: Agent, prompt: str) -> None:
    print(f"\n--- User: {prompt} ---")
    final = ""
    async for event in agent.run(prompt):
        et = event.event_type
        if et == "tool_start":
            args = getattr(event, "arguments", None) or {}
            query = args.get("query") if isinstance(args, dict) else None
            print(
                f"  [agent → 26ai] tool_start: {event.tool_name}"
                f"(query={query!r})  ← embedding + search both run in-DB"
            )
        elif et == "tool_complete":
            raw = getattr(event, "result", "") or ""
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {}
            hits = payload.get("results", []) if isinstance(payload, dict) else []
            print(f"  [agent ← 26ai] tool_complete: {len(hits)} hit(s)")
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
    model_name = os.environ.get("OCI_INDB_MODEL", _DEFAULT_MODEL)
    dimension = int(os.environ.get("OCI_INDB_DIM", _DEFAULT_DIM))

    embedder = OracleInDBEmbeddings(
        model_name=model_name,
        dimension=dimension,
        dsn=os.environ["ORACLE_DSN"],
        user=os.environ["ORACLE_USER"],
        password=os.environ["ORACLE_PASSWORD"],
        wallet_location=os.path.expanduser(os.environ["ORACLE_WALLET"]),
        wallet_password=os.environ.get("ORACLE_WALLET_PASSWORD", ""),
    )

    # Quick smoke test of the in-DB embedder before we start spinning up
    # an Agent — surfaces ORA-* on the missing-model path with a clean
    # banner instead of failing inside the agent loop.
    try:
        smoke = await embedder.embed("smoke test")
    except Exception as exc:  # noqa: BLE001 - detect ORA-* and skip cleanly.
        if _is_missing_model_error(exc):
            _print_model_skip_banner(model_name, exc)
            await embedder.close()
            return
        await embedder.close()
        raise
    print(f"\nIn-DB embedder ready (model={model_name}, dim={len(smoke.embedding)}).")

    store = OracleVectorStore(
        dsn=os.environ["ORACLE_DSN"],
        user=os.environ["ORACLE_USER"],
        password=os.environ["ORACLE_PASSWORD"],
        wallet_location=os.path.expanduser(os.environ["ORACLE_WALLET"]),
        wallet_password=os.environ.get("ORACLE_WALLET_PASSWORD", ""),
        table_name="locus_notebook_41",
        dimension=dimension,
        distance_metric="COSINE",
    )

    retriever = RAGRetriever(embedder=embedder, store=store)

    print(f"Seeding {len(CORPUS)} passages — every embedding runs in-DB…")
    await retriever.add_documents(CORPUS)

    knowledge_tool = create_rag_tool(
        retriever,
        name="search_indb_corpus",
        description=(
            "Search the data-residency knowledge base. Embedding generation "
            "runs inside Oracle 26ai via DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDING, "
            "and similarity search runs as VECTOR_DISTANCE / COSINE in the "
            "same database — no text leaves the DB at retrieval time. Use "
            "this whenever the question depends on 26ai's in-database "
            "embedding capabilities."
        ),
        limit=3,
        threshold=0.0,
    )

    agent = Agent(
        model=get_model(max_tokens=300),
        tools=[knowledge_tool],
        system_prompt=(
            "You are a data-residency architect. When a question touches "
            "in-database embedding or the in-DB ONNX model, call "
            "search_indb_corpus first and ground your answer in the "
            "returned passages. Always note that the embedding/search "
            "round-trip stayed inside Oracle 26ai. Do not invent figures."
        ),
        max_iterations=4,
    )

    try:
        for query in QUERIES:
            await _drive_agent(agent, query)

        print(
            "\nEvery retrieval above embedded the query and ran "
            "VECTOR_DISTANCE inside Oracle 26ai. The agent only sees the "
            "tool's result — no external embedding endpoint was called."
        )
    finally:
        await store.close()
        await embedder.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
