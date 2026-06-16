# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Live integration tests for OracleVectorStore against an Oracle Autonomous
Database (ADB).

Two distinct flavours are covered:

1. **Locus-native schema** — the table is created by ``_ensure_table`` with
   columns ``(id, content, embedding, metadata, created_at)``. Verifies
   the standard add → search → get → delete round-trip.

2. **OracleVS-compat schema** — a table written by a foreign ingestion
   pipeline (the column layout produced by
   ``langchain_oracledb.vectorstores.OracleVS``: ``id``, ``text``,
   ``embedding``, ``metadata``, no ``created_at``). Verifies the
   ``content_column``/``auto_create_table`` overrides let locus read it
   without re-ingestion.

Run with::

    ADB_DSN=deepresearch_low \\
    ADB_PASSWORD=<pw> \\
    ADB_WALLET_LOCATION=~/.oci/wallets/deepresearch \\
    OCI_PROFILE=API_FREE_TIER \\
    OCI_AUTH_TYPE=api_key \\
    OCI_COMPARTMENT=ocid1.tenancy.oc1..xxx \\
    OCI_ENDPOINT=https://inference.generativeai.us-chicago-1.oci.oraclecloud.com \\
    uv run pytest tests/integration/rag/test_oracle_adb_store.py -v
"""

from __future__ import annotations

import json
import uuid

import pytest


pytestmark = pytest.mark.asyncio


# Skip the whole module unless ``oracledb`` is importable.
try:  # noqa: SIM105
    import oracledb  # noqa: F401
except ImportError:
    pytest.skip("oracledb not installed", allow_module_level=True)


SAMPLE_DOCS = [
    {
        "id": "doc-iron-1",
        "content": (
            "Iron is absorbed in the duodenum and proximal jejunum. Heme iron "
            "from animal sources is absorbed more efficiently than non-heme iron."
        ),
        "source": "review/2024",
    },
    {
        "id": "doc-iron-2",
        "content": (
            "Hepcidin is the master regulator of iron homeostasis, produced by "
            "the liver. It degrades ferroportin to block iron export from "
            "enterocytes and macrophages."
        ),
        "source": "review/2024",
    },
    {
        "id": "doc-iron-3",
        "content": (
            "Transferrin binds two ferric ions per molecule. Saturation below "
            "16% indicates iron deficiency."
        ),
        "source": "review/2024",
    },
]


@pytest.fixture
def adb_embedder(oci_config):
    """Cohere v4 embeddings for the integration test corpus."""
    from locus.rag import OCIEmbeddings

    return OCIEmbeddings(
        model_id="cohere.embed-v4.0",
        compartment_id=oci_config["compartment_id"],
        profile_name=oci_config["profile_name"],
        auth_type=oci_config["auth_type"],
        service_endpoint=oci_config["service_endpoint"],
    )


@pytest.fixture
def adb_table_name() -> str:
    """A unique table per test run so concurrent runs don't collide."""
    # Oracle identifier max 30 chars; the suffix keeps room.
    return f"LOCUS_IT_{uuid.uuid4().hex[:8].upper()}"


def _drop_table_if_exists(adb_config, table_name: str) -> None:
    """Best-effort cleanup; ignores ORA-00942 (table doesn't exist)."""
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
            except oracledb.DatabaseError as exc:  # pragma: no cover - cleanup
                if "ORA-00942" not in str(exc):
                    raise
    finally:
        conn.close()


# =============================================================================
# Locus-native schema round-trip
# =============================================================================


async def test_locus_native_schema_roundtrip(
    oracle_adb_config, adb_embedder, adb_table_name
) -> None:
    """Insert via locus, search/get back, cleanup."""
    from locus.rag import RAGRetriever
    from locus.rag.stores.base import Document
    from locus.rag.stores.oracle import OracleVectorStore

    # Warm up the embedder so .config.dimension reflects v4 (1536).
    probe = await adb_embedder.embed_query("probe")
    dim = len(probe.embedding)
    assert dim == 1536, f"expected 1536-d cohere.embed-v4.0, got {dim}"

    try:
        store = OracleVectorStore(
            **oracle_adb_config,
            table_name=adb_table_name,
            dimension=dim,
        )
        retriever = RAGRetriever(embedder=adb_embedder, store=store)

        # Embed + add three docs
        contents = [d["content"] for d in SAMPLE_DOCS]
        embeddings = await adb_embedder.embed_documents(contents)
        docs = [
            Document(
                id=meta["id"],
                content=meta["content"],
                embedding=emb.embedding,
                metadata={"source": meta["source"]},
            )
            for meta, emb in zip(SAMPLE_DOCS, embeddings, strict=True)
        ]
        ids = await store.add_batch(docs)
        assert set(ids) == {d["id"] for d in SAMPLE_DOCS}
        assert await store.count() == len(SAMPLE_DOCS)

        # Search semantically
        results = await retriever.retrieve(query="iron absorption", limit=2)
        assert results.total_results == 2
        # Top result should be the absorption document, not the transferrin one.
        top_ids = [r.document.id for r in results.documents]
        assert "doc-iron-1" in top_ids

        # Get one back explicitly
        fetched = await store.get("doc-iron-2")
        assert fetched is not None
        assert "Hepcidin" in fetched.content
        assert fetched.metadata.get("source") == "review/2024"
        assert len(fetched.embedding) == dim

        # Delete one
        assert await store.delete("doc-iron-1") is True
        assert await store.count() == len(SAMPLE_DOCS) - 1

        await store.close()
    finally:
        _drop_table_if_exists(oracle_adb_config, adb_table_name)


# =============================================================================
# OracleVS-compat schema (foreign-table read path)
# =============================================================================


def _create_oraclevs_style_table(
    adb_config, table_name: str, embeddings: list[list[float]], docs: list[dict]
) -> None:
    """Create + seed a table that matches what
    ``langchain_oracledb.vectorstores.OracleVS`` would write: a CLOB ``text``
    column instead of ``content``, no ``created_at``, document id stored
    inside the ``metadata`` JSON."""
    dim = len(embeddings[0])
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
                    text CLOB,
                    embedding VECTOR({dim}, FLOAT32),
                    metadata CLOB CHECK (metadata IS JSON)
                )
                """
            )
            for doc, emb in zip(docs, embeddings, strict=True):
                vec = "[" + ",".join(str(f) for f in emb) + "]"
                cur.execute(
                    f"""INSERT INTO {table_name} (id, text, embedding, metadata) VALUES (:id, :text, TO_VECTOR(:emb), :meta)""",  # noqa: S608
                    {
                        "id": doc["id"],
                        "text": doc["content"],
                        "emb": vec,
                        "meta": json.dumps({"id": doc["id"], "source": doc["source"]}),
                    },
                )
            conn.commit()
    finally:
        conn.close()


async def test_oraclevs_compat_schema_read_only(
    oracle_adb_config, adb_embedder, adb_table_name
) -> None:
    """Read a foreign-schema table without re-ingestion via column overrides."""
    from locus.rag.stores.oracle import OracleVectorStore

    # Warm up embeddings + seed the OracleVS-shaped table directly.
    contents = [d["content"] for d in SAMPLE_DOCS]
    embeddings = await adb_embedder.embed_documents(contents)
    raw_embeddings = [e.embedding for e in embeddings]
    dim = len(raw_embeddings[0])

    try:
        _create_oraclevs_style_table(oracle_adb_config, adb_table_name, raw_embeddings, SAMPLE_DOCS)

        # Attach with the OracleVS-compat overrides; CRITICAL: no
        # ``auto_create_table`` race, no ``created_at`` references.
        store = OracleVectorStore(
            **oracle_adb_config,
            table_name=adb_table_name,
            content_column="text",
            created_at_column=None,
            auto_create_table=False,
            dimension=dim,
        )

        # The table already has 3 rows
        assert await store.count() == len(SAMPLE_DOCS)

        # Semantic search returns docs back through the ``text`` column path
        query_emb = await adb_embedder.embed_query("iron absorption")
        results = await store.search(query_emb.embedding, limit=3)
        assert len(results) == 3
        # Top match should be the absorption doc; we don't assert score
        # numerics because cosine values vary across model deployments.
        contents_back = [r.document.content for r in results]
        assert any("duodenum" in c for c in contents_back)

        # get() can target the metadata.id-based path? In OracleVS the doc id
        # is *also* the primary key here (we set it that way in the helper),
        # so a primary-key SELECT works.
        got = await store.get("doc-iron-2")
        assert got is not None
        assert "Hepcidin" in got.content
        assert got.metadata.get("source") == "review/2024"

        await store.close()
    finally:
        _drop_table_if_exists(oracle_adb_config, adb_table_name)
