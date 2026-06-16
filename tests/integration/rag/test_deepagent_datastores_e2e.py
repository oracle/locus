# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""End-to-end integration test for ``create_deepagent(datastores=...)``.

Builds a real Oracle ADB + (optionally) OpenSearch retriever, wires them
into ``create_deepagent``, and lets a real OCI GenAI chat model perform a
single tool-calling research turn.

This is the locus equivalent of the gist::

    create_deep_research_agent(
        datastores={"medical": adb_store, "logs": opensearch_store},
        ...
    )

Skips automatically if any of:
- ADB env vars (ADB_DSN / ADB_PASSWORD / ADB_WALLET_LOCATION) are missing
- OCI env vars (OCI_PROFILE / OCI_COMPARTMENT) are missing
- The configured OCI account can't reach a chat-capable model

Run with the same env block used by the other integration tests::

    OCI_PROFILE=BOAT-OC1 OCI_AUTH_TYPE=security_token \\
    OCI_COMPARTMENT=ocid1.compartment.oc1..xxx \\
    OCI_ENDPOINT=https://inference.generativeai.us-chicago-1.oci.oraclecloud.com \\
    ADB_DSN=deepresearch_low ADB_PASSWORD=... ADB_WALLET_LOCATION=~/.oci/wallets/deepresearch \\
    uv run pytest tests/integration/rag/test_deepagent_datastores_e2e.py -v
"""

from __future__ import annotations

import os
import uuid

import pytest


pytestmark = pytest.mark.asyncio


try:  # noqa: SIM105
    import oracledb  # noqa: F401
except ImportError:
    pytest.skip("oracledb not installed", allow_module_level=True)


SAMPLE_DOCS = [
    "Iron is absorbed in the duodenum and proximal jejunum. Heme iron from "
    "animal sources is absorbed more efficiently than non-heme iron from plants.",
    "Hepcidin is the master regulator of iron homeostasis, produced by the "
    "liver. It blocks iron export from enterocytes and macrophages by degrading "
    "ferroportin.",
    "Transferrin binds two ferric ions per molecule. Saturation below 16% "
    "suggests iron deficiency.",
    "Ferritin is the primary iron storage protein, sequestering up to 4500 "
    "iron atoms per molecule. Serum ferritin reflects total body iron stores.",
    "Hereditary hemochromatosis is caused by mutations in the HFE gene (C282Y "
    "homozygosity), leading to excessive intestinal iron absorption.",
]


def _drop_table_if_exists(adb_config, table_name: str) -> None:
    conn = oracledb.connect(
        user=adb_config["user"],
        password=adb_config["password"],
        dsn=adb_config["dsn"],
        config_dir=adb_config["wallet_location"],
        wallet_location=adb_config["wallet_location"],
        wallet_password=adb_config["wallet_password"],
    )
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(f"DROP TABLE {table_name} PURGE")
                conn.commit()
            except oracledb.DatabaseError as exc:  # pragma: no cover
                if "ORA-00942" not in str(exc):
                    raise
    finally:
        conn.close()


def _seed_locus_native_table_sync(
    adb_config, table_name: str, dim: int, embeddings: list[list[float]], docs: list[str]
) -> None:
    """Create + seed a locus-native schema table using a SYNC oracledb
    connection. Keeping seeding out of the test's asyncio loop avoids a
    pool-vs-loop mismatch when ``agent.run_sync`` later spins up its own
    event loop to execute the search tool from inside the agent."""
    conn = oracledb.connect(
        user=adb_config["user"],
        password=adb_config["password"],
        dsn=adb_config["dsn"],
        config_dir=adb_config["wallet_location"],
        wallet_location=adb_config["wallet_location"],
        wallet_password=adb_config["wallet_password"],
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE {table_name} (
                    id VARCHAR2(255) PRIMARY KEY,
                    content CLOB,
                    embedding VECTOR({dim}, FLOAT32),
                    metadata CLOB CHECK (metadata IS JSON),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP
                )
                """
            )
            import json as _json

            for i, (text, emb) in enumerate(zip(docs, embeddings, strict=True)):
                vec = "[" + ",".join(str(f) for f in emb) + "]"
                cur.execute(
                    f"""INSERT INTO {table_name} (id, content, embedding, metadata) VALUES (:id, :c, TO_VECTOR(:e), :m)""",  # noqa: S608
                    {
                        "id": f"doc-{i:02d}",
                        "c": text,
                        "e": vec,
                        "m": _json.dumps({"source": "test"}),
                    },
                )
            conn.commit()
    finally:
        conn.close()


async def test_deepagent_with_adb_datastore_e2e(oracle_adb_config, oci_config) -> None:
    """Full gist shape: real ADB retriever → create_deepagent → tool call → memo."""
    from locus.deepagent.factory import create_deepagent
    from locus.models import get_model
    from locus.rag import OCIEmbeddings, RAGRetriever
    from locus.rag.stores.oracle import OracleVectorStore

    embedder = OCIEmbeddings(
        model_id="cohere.embed-v4.0",
        compartment_id=oci_config["compartment_id"],
        profile_name=oci_config["profile_name"],
        auth_type=oci_config["auth_type"],
        service_endpoint=oci_config["service_endpoint"],
    )
    # Force a probe so the embedder's reported dim matches the v4 reality.
    probe = await embedder.embed_query("probe")
    dim = len(probe.embedding)

    table_name = f"LOCUS_E2E_{uuid.uuid4().hex[:8].upper()}"
    try:
        # Seed via SYNC oracledb so the OracleVectorStore's async pool gets
        # created lazily inside the agent's own event loop.
        embs = await embedder.embed_documents(SAMPLE_DOCS)
        raw = [e.embedding for e in embs]
        _seed_locus_native_table_sync(oracle_adb_config, table_name, dim, raw, SAMPLE_DOCS)

        store = OracleVectorStore(
            **oracle_adb_config,
            table_name=table_name,
            dimension=dim,
            auto_create_table=False,  # we already created it
        )
        retriever = RAGRetriever(embedder=embedder, store=store)

        # Build the chat model and the deepagent
        chat = get_model(
            "oci:openai.gpt-4o-mini",  # reliable tool caller for CI
            profile=os.environ["OCI_PROFILE"],
            compartment_id=oci_config["compartment_id"],
            region=os.environ.get("OCI_REGION", "us-chicago-1"),
        )

        agent = create_deepagent(
            model=chat,
            system_prompt=(
                "You are a medical research assistant. When asked, call the "
                "search_medical tool first, then write a short answer citing "
                "at least one returned document id."
            ),
            tools=[],
            datastores={
                "medical": {
                    "retriever": retriever,
                    "description": ("iron metabolism, anemia, hemochromatosis, transport proteins"),
                    "top_k": 3,
                },
            },
            max_output_tokens=1024,
            max_iterations=4,
            reflexion=False,
            grounding=False,
        )

        result = agent.run_sync("What does the medical literature say about iron absorption?")

        # The agent must have called the auto-wired tool. This is the
        # contract under test — that ``datastores=`` actually wires a tool
        # the model can see and invoke. Whether the model then chooses to
        # quote retrieved text vs paraphrase from priors is a model-policy
        # question, not a wiring question.
        tool_calls = [
            t.tool_name
            for t in result.tool_executions  # type: ignore[attr-defined]
        ]
        assert "search_medical" in tool_calls, (
            f"agent did not call search_medical; tool history: {tool_calls!r}"
        )

        # And the first ``search_medical`` execution must have completed
        # without error (proves the retriever is reading from the ADB and
        # not exploding mid-query). Whether the model then cites the
        # returned content vs paraphrases is a model-policy concern.
        searches = [
            t
            for t in result.tool_executions  # type: ignore[attr-defined]
            if t.tool_name == "search_medical"
        ]
        first_search = searches[0]
        assert first_search.error is None, f"search_medical errored: {first_search.error!r}"
        assert first_search.result is not None, "search_medical returned no result"
        # And the result string must mention at least one of our seeded
        # documents (proves the ADB round-trip actually happened).
        assert "doc-0" in first_search.result or "duodenum" in first_search.result, (
            f"search_medical didn't surface seeded doc; "
            f"raw result (first 300): {first_search.result[:300]!r}"
        )

        # And the agent did emit an answer (non-empty text).
        text = getattr(result, "text", "") or ""
        assert len(text) > 50, f"agent answer too short: {text!r}"

        # The agent's sync wrapper can leave the pool in a state where
        # ``close()`` either raises DPY-1005 (connections still busy) or
        # CancelledError (close was scheduled on a loop that's been torn
        # down). The DROP in ``finally`` opens a fresh sync connection, so
        # this teardown is purely cosmetic — swallow whatever it raises,
        # including CancelledError (which is a BaseException, not Exception).
        try:
            await store.close()
        except BaseException:  # noqa: BLE001 - cleanup-only swallow
            pass
    finally:
        _drop_table_if_exists(oracle_adb_config, table_name)
