#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke test of tulip.deepagent.create_deepagent(datastores=...).

Stays entirely on tulip primitives:

    OpenAIEmbeddings + QdrantVectorStore + RAGRetriever
        |
        v
    create_deepagent(datastores={"medical": retriever}, max_output_tokens=...)
        |
        v
    agent.run_sync("write a memo on iron metabolism")

Validates:
- `text-embedding-3-small` auto-detects its dimension (no enum entry needed).
- `datastores=` auto-wires a `search_medical` tool + datastore description
  block in the system prompt.
- `max_output_tokens=` lands on the per-completion request.

Run:
    OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-ant-... \\
    python examples/projects/deep-research/demo_smoke.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from tulip.deepagent import create_deepagent
from tulip.rag import OpenAIEmbeddings, RAGRetriever
from tulip.rag.stores.qdrant import QdrantVectorStore


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
    print("[1/4] Embeddings: text-embedding-3-small (auto-detected dimension)")
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    print("[2/4] QdrantVectorStore in-memory (10 sample docs on iron metabolism)")
    # Trigger one probe call so we know the dimension before constructing
    # the store.
    probe = await embedder.embed_query("probe")
    store = QdrantVectorStore(location=":memory:", dimension=len(probe.embedding))

    retriever = RAGRetriever(embedder=embedder, store=store)
    await retriever.add_documents(SAMPLE_DOCS)
    print(f"      stored {len(SAMPLE_DOCS)} docs at dim={len(probe.embedding)}")

    print("[3/4] create_deepagent(datastores={...}, max_output_tokens=2048)")
    from tulip.models import get_model

    chat_model = get_model(
        "anthropic:claude-sonnet-4-6",  # reliable tool caller for smoke
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
    # Tulip AgentResult: print the final answer
    print(getattr(result, "answer", None) or getattr(result, "output", None) or result)
    print("-" * 70)
    print("DONE.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
