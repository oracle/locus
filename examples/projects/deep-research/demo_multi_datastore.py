#!/usr/bin/env python3
# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Locus port of the multi-datastore Deep Research gist.

Mirrors `gist 92b9a515/adb_multi_store_huggingface_example.py`:
two ADB vector tables (medical + news) on a shared connection, with a
single `create_deepagent(datastores={"medical": ..., "news": ...})`
call. The agent must route each search at the correct store based on
the per-store description.

Locus equivalents:
- `from langchain_oci.datastores import ADB`           -> two `OracleVectorStore`s
                                                           on the same wallet/DSN
                                                           with distinct table names
- `create_deep_research_agent(datastores={...})`        -> `create_deepagent(datastores={...})`

Differences from the gist:
- The gist pulls 25 rows each from `pubmed_qa` + `ag_news` on Hugging Face;
  here we inline two small disjoint corpora so the demo has no network
  dependency beyond OCI itself.
- Single ADB (`deepresearch`) with two distinct table names per the
  user's "use only deepresearch" instruction.

Run:
    export OCI_PROFILE=DEFAULT
    export OCI_AUTH_TYPE=api_key
    export OCI_COMPARTMENT=ocid1.tenancy.oc1..xxx
    export ADB_DSN=<your-adb-tns>
    export ADB_PASSWORD=$(cat <your-adb-password-file>)
    export ADB_WALLET_LOCATION=~/.oci/wallets/<your-adb>
    .venv/bin/python examples/projects/deep-research/demo_multi_datastore.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

import oracledb

from locus.deepagent import create_deepagent
from locus.models import get_model
from locus.rag import OCIEmbeddings, RAGRetriever
from locus.rag.stores.oracle import OracleVectorStore


MEDICAL_CORPUS = [
    "Iron is primarily absorbed in the duodenum and proximal jejunum of the small intestine.",
    "Hepcidin, produced by the liver, is the master regulator of systemic iron homeostasis.",
    "Iron deficiency anemia is the most common nutritional deficiency worldwide, affecting ~1.2 billion people.",
    "Hereditary hemochromatosis is most commonly caused by C282Y homozygosity in the HFE gene.",
    "Transferrin saturation below 16% suggests iron deficiency; above 45% raises concern for iron overload.",
    "Ferritin is the primary iron storage protein and an acute-phase reactant elevated in inflammation.",
    "First-line treatment for iron deficiency anemia is oral ferrous sulfate; IV iron for malabsorption.",
    "Phlebotomy is the first-line treatment for hereditary hemochromatosis.",
    "Anemia of chronic disease is driven by elevated hepcidin in chronic inflammatory states.",
    "Reticulocyte hemoglobin content (CHr) is an early functional marker of iron deficiency.",
    "MRI T2* relaxometry is the gold standard for non-invasive iron quantification in liver/heart.",
    "Iron-refractory iron deficiency anemia (IRIDA) is caused by TMPRSS6 mutations.",
]


NEWS_CORPUS = [
    "Markets closed mixed on Friday as tech stocks rallied while energy shares declined.",
    "The central bank held interest rates steady, citing persistent inflation in services.",
    "A major airline announced new transatlantic routes opening next quarter.",
    "Local elections saw record turnout in three coastal districts, officials reported.",
    "A new infrastructure bill cleared the lower chamber by a 215-204 margin.",
    "The national weather service issued advisories for severe thunderstorms across the plains.",
    "An automaker recalled 120,000 SUVs over a brake-line manufacturing defect.",
    "Box office returns this weekend were dominated by an animated sequel.",
    "Tech regulators proposed new rules on cross-border data transfers for cloud providers.",
    "Universities reported a 6% rise in international graduate applications this cycle.",
    "A diplomatic delegation arrived in the capital ahead of next week's trade talks.",
    "Sports authorities approved a new playoff format starting next season.",
]


def _drop_tables(adb_cfg: dict, names: list[str]) -> None:
    conn = oracledb.connect(
        user=adb_cfg["user"],
        password=adb_cfg["password"],
        dsn=adb_cfg["dsn"],
        config_dir=adb_cfg["wallet_location"],
        wallet_location=adb_cfg["wallet_location"],
        wallet_password=adb_cfg["wallet_password"],
    )
    try:
        with conn.cursor() as cur:
            for name in names:
                try:
                    cur.execute(f"DROP TABLE {name} PURGE")
                    conn.commit()
                except oracledb.DatabaseError as exc:
                    if "ORA-00942" not in str(exc):
                        raise
    finally:
        conn.close()


def _seed_table_sync(
    adb_cfg: dict,
    table_name: str,
    dim: int,
    embeddings: list[list[float]],
    docs: list[str],
    domain: str,
) -> None:
    conn = oracledb.connect(
        user=adb_cfg["user"],
        password=adb_cfg["password"],
        dsn=adb_cfg["dsn"],
        config_dir=adb_cfg["wallet_location"],
        wallet_location=adb_cfg["wallet_location"],
        wallet_password=adb_cfg["wallet_password"],
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
            for i, (text, emb) in enumerate(zip(docs, embeddings, strict=True)):
                vec = "[" + ",".join(str(f) for f in emb) + "]"
                cur.execute(
                    f"INSERT INTO {table_name} (id, content, embedding, metadata) VALUES (:id, :c, TO_VECTOR(:e), :m)",  # noqa: S608
                    {
                        "id": f"{domain}-{i:02d}",
                        "c": text,
                        "e": vec,
                        "m": json.dumps({"domain": domain}),
                    },
                )
            conn.commit()
    finally:
        conn.close()


async def main() -> None:
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    adb_cfg = {
        "dsn": os.environ["ADB_DSN"],
        # Default to a least-priv app schema (locus_app), not ADMIN.
        # See docs/concepts/rag.md for the CREATE USER / GRANT script.
        "user": os.environ.get("ADB_USER", "locus_app"),
        "password": os.environ["ADB_PASSWORD"],
        "wallet_location": os.path.expanduser(os.environ["ADB_WALLET_LOCATION"]),
        "wallet_password": os.environ.get("ADB_WALLET_PASSWORD", os.environ["ADB_PASSWORD"]),
    }
    oci_profile = os.environ.get("OCI_PROFILE", "DEFAULT")
    auth_type = os.environ.get("OCI_AUTH_TYPE", "api_key")
    compartment = os.environ.get(
        "OCI_COMPARTMENT",
        "ocid1.tenancy.oc1..<your-tenancy>",
    )
    endpoint = os.environ.get(
        "OCI_ENDPOINT",
        "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    )
    region = os.environ.get("OCI_REGION", "us-chicago-1")
    model_id = os.environ.get("OCI_RESEARCH_MODEL", "oci:openai.gpt-4o-mini")
    max_out = int(os.environ.get("MAX_OUTPUT_TOKENS", "4096"))

    run_id = uuid.uuid4().hex[:8].upper()
    med_table = f"LOCUS_MED_{run_id}"
    news_table = f"LOCUS_NEWS_{run_id}"
    print("=" * 70)
    print("MULTI-DATASTORE DEEP RESEARCH — LOCUS PORT")
    print("=" * 70)
    print(f"\n[1/4] Embedding two disjoint corpora with cohere.embed-v4.0 …")

    embedder = OCIEmbeddings(
        model_id="cohere.embed-v4.0",
        compartment_id=compartment,
        profile_name=oci_profile,
        auth_type=auth_type,
        service_endpoint=endpoint,
    )
    med_embs = await embedder.embed_documents(MEDICAL_CORPUS)
    news_embs = await embedder.embed_documents(NEWS_CORPUS)
    med_raw = [e.embedding for e in med_embs]
    news_raw = [e.embedding for e in news_embs]
    dim = len(med_raw[0])
    print(f"       medical: {len(med_raw)} × {dim}-d")
    print(f"       news   : {len(news_raw)} × {dim}-d")

    try:
        print(f"\n[2/4] Seeding ADB tables ({med_table}, {news_table}) …")
        _seed_table_sync(adb_cfg, med_table, dim, med_raw, MEDICAL_CORPUS, "med")
        _seed_table_sync(adb_cfg, news_table, dim, news_raw, NEWS_CORPUS, "news")

        med_store = OracleVectorStore(
            **adb_cfg,
            table_name=med_table,
            dimension=dim,
            auto_create_table=False,
        )
        news_store = OracleVectorStore(
            **adb_cfg,
            table_name=news_table,
            dimension=dim,
            auto_create_table=False,
        )
        med_retriever = RAGRetriever(embedder=embedder, store=med_store)
        news_retriever = RAGRetriever(embedder=embedder, store=news_store)

        print(f"\n[3/4] Building deepagent with datastores={{medical, news}} …")
        chat = get_model(model_id, profile=oci_profile, compartment_id=compartment, region=region)
        agent = create_deepagent(
            model=chat,
            system_prompt=(
                "You are a research assistant with access to two datastores. "
                "When asked a question, pick the datastore whose description "
                "best matches the topic (or call both when the question spans "
                "both domains). Cite the doc ids (e.g. med-03, news-07) you "
                "draw evidence from. Do not invent facts outside the retrieved "
                "documents."
            ),
            tools=[],
            datastores={
                "medical": {
                    "retriever": med_retriever,
                    "description": (
                        "clinical and hematology knowledge: iron metabolism, "
                        "anemia, hemochromatosis, diagnostics, treatment"
                    ),
                    "top_k": 4,
                },
                "news": {
                    "retriever": news_retriever,
                    "description": (
                        "general news headlines: markets, politics, weather, "
                        "transportation, sports, education"
                    ),
                    "top_k": 4,
                },
            },
            max_output_tokens=max_out,
            max_iterations=8,
            reflexion=False,
            grounding=False,
        )

        print(f"\n[4/4] Running cross-domain prompt …")
        print("-" * 70)
        prompt = (
            "Using only the two datastores: (a) summarize the key regulators "
            "of iron homeostasis from the medical datastore, and (b) list two "
            "distinct items from the news datastore. Keep each section short "
            "(3-5 bullets). Cite document ids (med-NN / news-NN)."
        )
        t0 = time.time()
        result = agent.run_sync(prompt)
        elapsed = time.time() - t0

        text = getattr(result, "text", "") or ""
        tool_execs = list(result.tool_executions or ())  # type: ignore[arg-type]
        metrics = getattr(result, "metrics", None)

        med_calls = sum(1 for t in tool_execs if t.tool_name == "search_medical")
        news_calls = sum(1 for t in tool_execs if t.tool_name == "search_news")

        print(f"\nTotal tool calls : {len(tool_execs)}  (medical={med_calls}, news={news_calls})")
        for t in tool_execs:
            args = (
                t.arguments.get("query", t.arguments)
                if isinstance(t.arguments, dict)
                else t.arguments
            )
            print(f"  - {t.tool_name}({args!r}) -> {len(t.result or '')} chars")
        if metrics:
            print(f"Iterations       : {metrics.iterations}")
            print(
                f"Tokens           : prompt={metrics.prompt_tokens} "
                f"completion={metrics.completion_tokens} total={metrics.total_tokens}"
            )
        print(f"Time             : {elapsed:.1f}s")
        print()
        print("--- Response ---")
        print(text)

        out_path = out_dir / "multi_datastore_report.md"
        out_path.write_text(text)
        print(f"\nReport saved to: {out_path}")

        if med_calls > 0 and news_calls > 0:
            print("\nROUTING CHECK: agent hit BOTH datastores — PASS")
        else:
            print(
                f"\nROUTING CHECK: agent only hit medical={med_calls}, news={news_calls} — partial"
            )

        try:
            await med_store.close()
            await news_store.close()
        except BaseException:
            pass
    finally:
        _drop_tables(adb_cfg, [med_table, news_table])


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
