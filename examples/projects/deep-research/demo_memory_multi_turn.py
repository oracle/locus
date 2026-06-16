#!/usr/bin/env python3
# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Locus port of the Memory-Aware Deep Research gist.

Mirrors `gist 15ab302e/memory_aware_adb_65k.py`:
two-turn conversation backed by an ADB datastore, with each turn's final
response captured into an `InMemoryStore` under a deterministic key.

Locus equivalents:
- `from langgraph.checkpoint.memory import MemorySaver`     -> two consecutive
                                                              `agent.run_sync()` calls
                                                              with turn 1's memo inlined into turn 2's
                                                              prompt (full-history multi-turn would
                                                              use a locus DeltaCheckpointer; out of
                                                              scope here).
- `from langgraph.store.memory import InMemoryStore`         -> `from locus.memory import InMemoryStore`
- `from langchain_oci import OCIGenAIEmbeddings,
   create_deep_research_agent`                              -> `from locus.rag import OCIEmbeddings, RAGRetriever`
                                                              + `from locus.deepagent import create_deepagent`
- `langchain_oci.datastores.ADB`                            -> `from locus.rag.stores.oracle import OracleVectorStore`

The `capture_response()` helper preserves the gist's cookbook pattern
verbatim: deterministic SHA-256 key over the query, namespace
``("research_sessions",)``, plus structured metadata.

Run:
    export OCI_PROFILE=DEFAULT     # or DEFAULT (when its session is fresh)
    export OCI_AUTH_TYPE=api_key         # or security_token for DEFAULT
    export OCI_COMPARTMENT=ocid1.tenancy.oc1..xxx
    export ADB_DSN=<your-adb-tns>
    export ADB_PASSWORD=$(cat <your-adb-password-file>)
    export ADB_WALLET_LOCATION=~/.oci/wallets/<your-adb>
    .venv/bin/python examples/projects/deep-research/demo_memory_multi_turn.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

import oracledb

from locus.deepagent import create_deepagent
from locus.memory import InMemoryStore
from locus.models import get_model
from locus.rag import OCIEmbeddings, RAGRetriever
from locus.rag.stores.oracle import OracleVectorStore


# Same 45-doc iron-metabolism corpus as the iron_metabolism demo —
# small enough for fast ingest, dense enough to support two distinct
# turn topics (memo + diagnostic markers follow-up).
CORPUS = [
    "Iron is primarily absorbed in the duodenum and proximal jejunum.",
    "Heme iron from animal sources is absorbed at 15-35%, while non-heme iron from plants is absorbed at only 2-20%.",
    "Vitamin C (ascorbic acid) enhances non-heme iron absorption by reducing ferric (Fe3+) to ferrous (Fe2+) iron.",
    "Phytates in legumes, tannins in tea, and calcium in dairy products all inhibit non-heme iron absorption.",
    "DMT1 (divalent metal transporter 1) is the main apical transporter for non-heme iron in enterocytes.",
    "Heme iron is absorbed via the heme carrier protein HCP1, then broken down intracellularly by heme oxygenase to release iron.",
    "Ferroportin is the only known cellular iron exporter, located on the basolateral membrane of enterocytes and macrophages.",
    "Transferrin is the main iron transport protein in plasma, binding two ferric ions per molecule.",
    "Transferrin saturation below 16% suggests iron deficiency; saturation above 45% raises concern for iron overload.",
    "Iron is delivered to cells via transferrin receptor 1 (TfR1), which is upregulated in iron-deficient and rapidly dividing cells.",
    "Soluble transferrin receptor (sTfR) is elevated in iron deficiency but unaffected by inflammation, helping distinguish IDA from ACD.",
    "Ferritin is the primary iron storage protein, sequestering up to 4500 iron atoms per molecule.",
    "Serum ferritin reflects total body iron stores but is an acute-phase reactant elevated by inflammation, infection, and malignancy.",
    "Hemosiderin is an insoluble form of iron storage found in macrophages, accumulating in iron-overload states.",
    "The average adult body contains 3-4 grams of iron; about 65% is in hemoglobin, 25% in storage, 10% in myoglobin and enzymes.",
    "Hepcidin, produced by the liver, is the master regulator of iron homeostasis.",
    "Hepcidin binds ferroportin and induces its internalization and degradation, blocking iron export from enterocytes and macrophages.",
    "Hepcidin is upregulated by iron repletion (via BMP6/SMAD signaling) and by inflammation (via IL-6/STAT3 signaling).",
    "Hepcidin is suppressed by iron deficiency, hypoxia, and increased erythropoietic demand (via erythroferrone from erythroblasts).",
    "Erythroferrone, secreted by EPO-stimulated erythroblasts, suppresses hepcidin to mobilize iron for erythropoiesis.",
    "Matriptase-2 (TMPRSS6) cleaves hemojuvelin to suppress hepcidin transcription; loss-of-function mutations cause IRIDA.",
    "Iron deficiency anemia is the most common nutritional deficiency worldwide, affecting an estimated 1.2 billion people.",
    "Classic lab findings in IDA include low ferritin (<30 ng/mL), low transferrin saturation, high TIBC, and microcytic hypochromic RBCs.",
    "Pica (ice, dirt, starch craving) and restless legs syndrome are non-hematologic clinical clues to iron deficiency.",
    "First-line treatment for IDA is oral ferrous sulfate 325 mg three times daily, ideally on an empty stomach with vitamin C.",
    "Alternate-day oral iron dosing improves fractional absorption by avoiding hepcidin elevation triggered by daily doses.",
    "IV iron (ferric carboxymaltose, iron sucrose) is indicated for malabsorption, intolerance, chronic blood loss, or rapid replacement before surgery.",
    "Hereditary hemochromatosis is most commonly caused by C282Y homozygosity in the HFE gene.",
    "HFE-hemochromatosis impairs hepcidin sensing of body iron stores, leading to inappropriately low hepcidin and excessive intestinal iron absorption.",
    "Iron overload in hemochromatosis damages the liver (cirrhosis, HCC), heart (cardiomyopathy, arrhythmias), pancreas (diabetes), joints, and skin.",
    "Diagnosis of HH typically involves transferrin saturation >45%, elevated ferritin, and HFE genetic testing.",
    "Phlebotomy is the first-line treatment for hereditary hemochromatosis, removing roughly 200-250 mg of iron per unit of blood.",
    "Therapeutic phlebotomy target in HH is ferritin below 50 ng/mL with transferrin saturation below 50%.",
    "Iron chelators (deferoxamine, deferasirox, deferiprone) are used when phlebotomy is contraindicated and in transfusion iron overload.",
    "Anemia of chronic disease (anemia of inflammation) is caused by elevated hepcidin in chronic inflammatory states, sequestering iron in macrophages.",
    "ACD typically presents as normocytic normochromic anemia with elevated ferritin, low serum iron, low TIBC, and normal or low transferrin saturation.",
    "Treating the underlying inflammation is the cornerstone of ACD management; iron supplementation alone is largely ineffective due to hepcidin block.",
    "Iron-refractory iron deficiency anemia (IRIDA) is an autosomal recessive disorder caused by TMPRSS6 mutations.",
    "In IRIDA, defective matriptase-2 leaves hepcidin inappropriately elevated, so oral iron is poorly absorbed; IV iron is the mainstay of treatment.",
    "Iron requirements increase from 18 mg/day in non-pregnant women to 27 mg/day during pregnancy due to fetal demand and plasma volume expansion.",
    "Maternal iron deficiency in pregnancy is associated with preterm birth, low birth weight, and impaired infant neurodevelopment.",
    "Vegetarian and vegan diets typically require 1.8x the iron RDA due to lower bioavailability of non-heme iron.",
    "MRI T2* relaxometry is the gold standard for non-invasive quantification of liver and cardiac iron load in iron-overload disorders.",
    "Bone marrow iron staining with Prussian blue remains the historical gold standard for assessing iron stores but is rarely needed clinically.",
    "Reticulocyte hemoglobin content (CHr) drops within days of iron deficiency, providing an early functional marker before MCV changes.",
]


async def capture_response(
    store: InMemoryStore,
    query: str,
    response: str,
    tool_calls: int,
    message_count: int,
    namespace: tuple[str, ...] = ("research_sessions",),
) -> None:
    """Cookbook pattern: deterministic SHA-256 key, structured value.

    Identical contract to the gist's `capture_response()` — only the
    InMemoryStore implementation under the hood changes (locus vs langgraph).
    Locus's InMemoryStore is async, hence the `await store.put(...)`.
    """
    key = hashlib.sha256(query.encode()).hexdigest()[:16]
    await store.put(
        namespace,
        key=key,
        value={
            "query": query,
            "response": response,
            "tool_calls": tool_calls,
            "message_count": message_count,
        },
    )


def _drop_table_if_exists(adb_cfg: dict, table_name: str) -> None:
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
            try:
                cur.execute(f"DROP TABLE {table_name} PURGE")
                conn.commit()
            except oracledb.DatabaseError as exc:
                if "ORA-00942" not in str(exc):
                    raise
    finally:
        conn.close()


def _seed_table_sync(
    adb_cfg: dict, table_name: str, dim: int, embeddings: list[list[float]], docs: list[str]
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
                        "id": f"doc-{i:02d}",
                        "c": text,
                        "e": vec,
                        "m": json.dumps({"topic": "iron_metabolism"}),
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
    # gpt-4o-mini is the current reliable tool-caller on OCI; gist used Gemini Pro
    # but that path is intermittently broken on OCI for tool-augmented prompts.
    model_id = os.environ.get("OCI_RESEARCH_MODEL", "oci:openai.gpt-4o-mini")
    max_out = int(os.environ.get("MAX_OUTPUT_TOKENS", "8192"))

    table_name = f"LOCUS_MEM_{uuid.uuid4().hex[:8].upper()}"
    print("=" * 70)
    print("MEMORY-AWARE DEEP RESEARCH AGENT — LOCUS PORT")
    print("=" * 70)

    print("\n[1/5] Connecting to ADB datastore...")
    embedder = OCIEmbeddings(
        model_id="cohere.embed-v4.0",
        compartment_id=compartment,
        profile_name=oci_profile,
        auth_type=auth_type,
        service_endpoint=endpoint,
    )
    embs = await embedder.embed_documents(CORPUS)
    raw_embs = [e.embedding for e in embs]
    dim = len(raw_embs[0])
    _seed_table_sync(adb_cfg, table_name, dim, raw_embs, CORPUS)
    print(f"   ADB table  : {table_name} ({adb_cfg['dsn']})")
    print(f"   Corpus     : {len(CORPUS)} sentences, dim={dim}")
    print(f"   Embedding  : cohere.embed-v4.0")

    try:
        # ----- 2. Create memory stack (cookbook pattern) -----
        print("\n[2/5] Setting up memory stack...")
        memory_store = InMemoryStore()
        print("   InMemoryStore: for captured responses")
        print(f"   (Multi-turn handled by inlining turn-1 response into turn-2 prompt;")
        print(f"    locus also offers DeltaCheckpointer for true thread-id resume.)")

        # ----- 3. Create agent -----
        print("\n[3/5] Creating Deep Research Agent...")
        print(f"   Model           : {model_id}")
        print(f"   Max output tokens: {max_out:,}")

        store = OracleVectorStore(
            **adb_cfg,
            table_name=table_name,
            dimension=dim,
            auto_create_table=False,
        )
        retriever = RAGRetriever(embedder=embedder, store=store)

        chat = get_model(model_id, profile=oci_profile, compartment_id=compartment, region=region)

        def build_agent(extra_context: str = "") -> object:
            sysprompt = (
                "You are a medical research assistant with access to a datastore "
                "of clinical knowledge. When asked to write a research memo, "
                "search the datastore thoroughly, gather evidence from multiple "
                "documents, and write a comprehensive, well-structured report "
                "with citations of the form (doc-NN) to source documents."
            )
            if extra_context:
                sysprompt = f"{sysprompt}\n\n# Earlier conversation\n\n{extra_context}"
            return create_deepagent(
                model=chat,
                system_prompt=sysprompt,
                tools=[],
                datastores={
                    "medical": {
                        "retriever": retriever,
                        "description": (
                            "medical questions and answers, clinical knowledge on iron "
                            "metabolism, anemia, hemochromatosis, diagnostics, treatment"
                        ),
                        "top_k": 6,
                    }
                },
                max_output_tokens=max_out,
                max_iterations=8,
                reflexion=False,
                grounding=False,
            )

        print("   Agent ready.")

        # ----- 4. Turn 1 — long-form memo -----
        print("\n[4/5] Turn 1 — research memo")
        print("-" * 70)
        query1 = (
            "Write a comprehensive research memo on iron metabolism and its "
            "clinical significance. Search the datastore for all relevant "
            "evidence. Cover: iron absorption, transport, storage, disorders "
            "(deficiency and overload), diagnostic markers, and treatment. "
            "Include (doc-NN) citations."
        )
        print(f"Query: {query1}\n")
        agent1 = build_agent()
        t0 = time.time()
        result1 = agent1.run_sync(query1)
        elapsed1 = time.time() - t0
        text1 = getattr(result1, "text", "") or ""
        tool_execs1 = list(result1.tool_executions or ())  # type: ignore[arg-type]
        metrics1 = getattr(result1, "metrics", None)

        await capture_response(
            memory_store,
            query=query1,
            response=text1,
            tool_calls=len(tool_execs1),
            message_count=getattr(metrics1, "iterations", 0) or 0,
        )
        print(f"Response length   : {len(text1):,} chars")
        print(f"Tool calls        : {len(tool_execs1)}")
        print(f"Iterations        : {getattr(metrics1, 'iterations', None)}")
        print(f"Time              : {elapsed1:.1f}s")
        print(f"\n--- Response preview (first 1500 chars) ---")
        print(text1[:1500])
        if len(text1) > 1500:
            print(f"... ({len(text1) - 1500:,} more chars)")

        # ----- 5. Turn 2 — follow-up using turn-1 context -----
        print("\n" + "-" * 70)
        print("[5/5] Turn 2 — follow-up (uses turn-1 context)")
        print("-" * 70)
        query2 = (
            "Based on the research memo you just wrote, what are the most "
            "important diagnostic markers a clinician should order? "
            "Summarize as a quick-reference markdown table with columns: "
            "Marker | What it suggests | Caveat."
        )
        print(f"Query: {query2}\n")
        # Inline turn-1's memo as 'earlier conversation' context for turn 2
        agent2 = build_agent(extra_context=f"## Turn 1 memo (yours)\n\n{text1}")
        t1 = time.time()
        result2 = agent2.run_sync(query2)
        elapsed2 = time.time() - t1
        text2 = getattr(result2, "text", "") or ""
        tool_execs2 = list(result2.tool_executions or ())  # type: ignore[arg-type]
        metrics2 = getattr(result2, "metrics", None)

        await capture_response(
            memory_store,
            query=query2,
            response=text2,
            tool_calls=len(tool_execs2),
            message_count=getattr(metrics2, "iterations", 0) or 0,
        )
        print(f"Response length     : {len(text2):,} chars")
        print(f"Tool calls (turn 2) : {len(tool_execs2)}")
        print(f"Iterations          : {getattr(metrics2, 'iterations', None)}")
        print(f"Time                : {elapsed2:.1f}s")
        print(f"\n--- Response ---\n{text2}")

        # ----- Summary of captured store -----
        print("\n" + "=" * 70)
        print("MEMORY STORE SUMMARY")
        print("=" * 70)
        items = await memory_store.search(("research_sessions",), limit=10)
        print(f"\nCaptured responses in store: {len(items)}")
        for i, item in enumerate(items, 1):
            v = item.value
            print(f"\n  [{i}] Query        : {v['query'][:80]}...")
            print(f"      Response len : {len(v['response']):,} chars")
            print(f"      Tool calls   : {v['tool_calls']}")
            print(f"      Iterations   : {v['message_count']}")
        print(f"\nTotal time: {elapsed1 + elapsed2:.1f}s")

        # Persist artifacts
        memo_path = out_dir / "memory_multi_turn_memo.md"
        memo_path.write_text(text1 + "\n\n---\n\n## Follow-up\n\n" + text2)
        print(f"\nMemo saved to: {memo_path}")

        try:
            await store.close()
        except BaseException:
            pass
    finally:
        _drop_table_if_exists(adb_cfg, table_name)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
