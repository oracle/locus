#!/usr/bin/env python3
# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""End-to-end smoke test of locus.deepagent.create_deepagent(datastores=...).

Mirrors the shape of langchain-oci's `create_deep_research_agent()` gist
but stays entirely on locus primitives:

    OCIEmbeddings + InMemoryVectorStore + RAGRetriever
        |
        v
    create_deepagent(datastores={"medical": retriever}, max_output_tokens=...)
        |
        v
    agent.run_sync("write a memo on iron metabolism")

Validates:
- `cohere.embed-v4.0` auto-detects to 1536 dims (no enum entry needed).
- `datastores=` auto-wires a `search_medical` tool + datastore description
  block in the system prompt.
- `max_output_tokens=` lands on the per-completion request.

Run:
    OCI_COMPARTMENT_ID=... uv run python examples/projects/deep-research/demo_smoke.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from locus.deepagent import create_deepagent
from locus.rag import OCIEmbeddings, RAGRetriever
from locus.rag.stores.memory import InMemoryVectorStore


SAMPLE_DOCS = [
    "Iron is absorbed in the duodenum and proximal jejunum. Heme iron from "
    "animal sources is absorbed more efficiently than non-heme iron from plants.",
    "Hepcidin is the master regulator of iron homeostasis, produced by the liver "
    "in response to iron stores and inflammation. It blocks iron export from "
    "enterocytes and macrophages by degrading ferroportin.",
    "Iron deficiency anemia is the most common nutritional deficiency worldwide, "
    "affecting an estimated 1.2 billion people. Diagnostic markers include low "
    "ferritin, low transferrin saturation, and microcytic hypochromic RBCs.",
    "Hereditary hemochromatosis is caused by mutations in the HFE gene (most "
    "commonly C282Y homozygosity), leading to excessive intestinal iron "
    "absorption and tissue iron overload affecting the liver, heart, and pancreas.",
    "Treatment for iron deficiency typically begins with oral ferrous sulfate "
    "325mg three times daily. IV iron is indicated for malabsorption, "
    "intolerance, or rapid replacement needs.",
    "Transferrin is the main iron transport protein in plasma, binding two ferric "
    "ions per molecule. Transferrin saturation below 16% suggests iron deficiency.",
    "Ferritin is the primary iron storage protein, sequestering up to 4500 iron "
    "atoms per molecule. Serum ferritin reflects total body iron stores but is "
    "an acute-phase reactant elevated by inflammation.",
    "Phlebotomy is the first-line treatment for hereditary hemochromatosis, "
    "removing roughly 200-250mg of iron per unit of blood. Therapeutic target is "
    "ferritin <50 ng/mL.",
    "Anemia of chronic disease results from elevated hepcidin in inflammatory "
    "states, sequestering iron in macrophages. Typically presents as normocytic "
    "with elevated ferritin and low transferrin saturation.",
    "Iron-refractory iron deficiency anemia (IRIDA) is caused by mutations in "
    "TMPRSS6 leading to inappropriately elevated hepcidin and poor response to "
    "oral iron supplementation. IV iron is the mainstay of treatment.",
]


async def main() -> None:
    compartment_id = os.environ.get(
        "OCI_COMPARTMENT_ID",
        # Your OCI compartment with GenAI service access
        "ocid1.compartment.oc1..<your-compartment>",
    )
    profile_name = os.environ.get("OCI_PROFILE", "DEFAULT")
    auth_type = os.environ.get("OCI_AUTH_TYPE", "security_token")
    service_endpoint = os.environ.get(
        "OCI_SERVICE_ENDPOINT",
        "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    )

    print("[1/4] Embeddings: cohere.embed-v4.0 (auto-detected dimension)")
    embedder = OCIEmbeddings(
        model_id="cohere.embed-v4.0",
        compartment_id=compartment_id,
        profile_name=profile_name,
        auth_type=auth_type,
        service_endpoint=service_endpoint,
    )

    print("[2/4] InMemoryVectorStore (10 sample docs on iron metabolism)")
    # Trigger one probe call so we know the dimension before constructing
    # the store; in production OracleVectorStore reads dim from config.
    probe = await embedder.embed_query("probe")
    store = InMemoryVectorStore(dimension=len(probe.embedding))

    retriever = RAGRetriever(embedder=embedder, store=store)
    await retriever.add_documents(SAMPLE_DOCS)
    print(f"      stored {len(SAMPLE_DOCS)} docs at dim={len(probe.embedding)}")

    print("[3/4] create_deepagent(datastores={...}, max_output_tokens=2048)")
    from locus.models import get_model

    # OCIChatCompletionsModel: profile= XOR auth_type=. We use profile= which
    # auto-reads the security_token_file from ~/.oci/config[DEFAULT].
    chat_model = get_model(
        "oci:openai.gpt-4o-mini",  # reliable tool caller for smoke
        profile=profile_name,
        compartment_id=compartment_id,
        region="us-chicago-1",
    )

    agent = create_deepagent(
        model=chat_model,
        system_prompt=(
            "You are a medical research assistant. When asked about a topic, "
            "search the medical datastore for evidence, then write a concise "
            "memo with bullet-pointed findings. Cite document indices when "
            "they support a claim."
        ),
        tools=[],
        datastores={
            "medical": {
                "retriever": retriever,
                "description": (
                    "iron metabolism, anemia, hemochromatosis, iron transport "
                    "proteins, diagnostics, and treatment"
                ),
                "top_k": 4,
            },
        },
        max_output_tokens=2048,
        max_iterations=8,
        reflexion=False,  # keep the smoke run small
        grounding=False,
    )

    print("[4/4] Running: 'short memo on iron metabolism'\n" + "-" * 70)
    result = agent.run_sync(
        "Write a short memo on iron metabolism. Search the medical datastore "
        "first; cite at least three documents."
    )
    # Locus AgentResult: print the final answer
    print(getattr(result, "answer", None) or getattr(result, "output", None) or result)
    print("-" * 70)
    print("DONE.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
