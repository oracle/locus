# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Notebook 53: RAG agents — ATLAS, an on-call SRE copilot over the Index.

Once you have a vector store full of runbooks (notebook 51 / 52), the
next step is to let an agent reach into it — so an incident answer cites
your internal runbooks instead of model memory. ATLAS is the platform
team's on-call copilot; it reads the Index. ``RAGRetriever.as_tool()``
turns the retriever into an ordinary Locus tool that ATLAS picks up
alongside any other ``@tool`` you define.

- ``retriever.as_tool(name, description)`` — convert a retriever into a
  callable tool for the agent.
- Single-tool Q&A copilot against an internal runbook KB.
- Mixed tool set — runbook search alongside a calculator and a date tool
  for rollout math.
- Streaming events from the agent while it searches and answers.
- An **answer-grounding gate**: before ATLAS ships remediation advice,
  the GSAR scorer partitions its claims into grounded vs. ungrounded and
  ``decide()`` says whether to proceed. When the answer is anchored in a
  retrieved runbook chunk, it proceeds; when it is speculation, the same
  call replans rather than guess — a hallucinated fix by construction
  withheld.

Chunk-size, prompt-design, and metadata-filter notes for ops corpora are
in ``docs/concepts/rag.md``.

Backend: the pure-Python ``InMemoryVectorStore`` keeps the demo dependency-free
and runnable anywhere (even in a browser). Swap ``_make_store`` for any other
Locus vector store implementation (pgvector, OpenSearch, Qdrant, Chroma) for a
durable backend.

Run it:
    export OPENAI_API_KEY=sk-...
    python examples/notebook_53_rag_agents.py

    # Offline (the embedding-backed sections skip cleanly when the key is
    # missing; the answer-grounding gate is pure-Python and always runs):
    python examples/notebook_53_rag_agents.py
"""

import asyncio
import json
import os
from datetime import datetime

from config import get_embedder, get_model, safe_math_eval

from locus.agent import Agent
from locus.core.events import ThinkEvent, ToolCompleteEvent, ToolStartEvent
from locus.rag import InMemoryVectorStore, RAGRetriever
from locus.reasoning.gsar import Claim, Decision, EvidenceType, Partition, decide, gsar_score
from locus.tools import tool


def _missing_env() -> list[str]:
    return [name for name in ("OPENAI_API_KEY",) if not os.environ.get(name)]


def _make_store(dim: int) -> InMemoryVectorStore:
    # A fresh store per section so sections don't stomp on each other.
    return InMemoryVectorStore(dimension=dim, distance_metric="cosine")


# =============================================================================
# Step 1: RAGRetriever.as_tool() — turn a retriever into a normal agent tool.
# =============================================================================


async def rag_as_tool():
    print("=" * 60)
    print("Notebook 53: RAG as a Tool")
    print("=" * 60)

    embedder = get_embedder()
    store = _make_store(dim=embedder.config.dimension)
    retriever = RAGRetriever(embedder=embedder, store=store)

    # Internal runbook one-liners. All runbook ids and services are fictitious.
    knowledge = [
        "RB-014 covers the checkout-api p99 latency regression after the 2.4.1 deploy; roll back via Argo.",
        "RB-015 documents orders-db connection-pool exhaustion under peak load.",
        "RB-016: the nightly billing batch OOMs when the dataset exceeds 8 GB; raise the memory limit.",
        "RB-017: gateway ingress 502s trace to a misconfigured readiness probe.",
        "RB-018: rotate the registry pull-secret after the expired-credential incident.",
    ]

    print("Building the Index (internal runbooks)...")
    await retriever.add_documents(knowledge)
    print(f"  Added {len(knowledge)} runbooks")

    search_tool = retriever.as_tool(
        name="search_runbooks",
        description="Search the internal SRE runbook knowledge base.",
    )

    print(f"\nCreated tool: {search_tool.name}")
    print(f"Description: {search_tool.description}")

    print("\n" + "-" * 40)
    print("Testing tool directly...")

    result = await search_tool("Which runbook covers the checkout latency regression?")

    print("\nQuery: 'Which runbook covers the checkout latency regression?'")
    print(f"Results found: {result['total']}")
    for i, doc in enumerate(result["results"], 1):
        print(f"  {i}. Score: {doc['score']:.4f}")
        print(f"     {doc['content'][:60]}...")


# =============================================================================
# Step 2: A small Q&A copilot that grounds answers in internal runbooks.
# =============================================================================


async def simple_rag_agent():
    print("\n" + "=" * 60)
    print("Notebook 53: ATLAS — On-Call SRE Copilot")
    print("=" * 60)

    embedder = get_embedder()
    store = _make_store(dim=embedder.config.dimension)
    retriever = RAGRetriever(embedder=embedder, store=store)

    runbook_docs = [
        """
        RB-014 / INC-2026-014 is a p99 latency regression in the
        checkout-api service, opened in 2026. The 2.4.0 deploy introduced
        an N+1 query in the pricing module. Customer-facing p99 climbed
        from 180ms to 1.4s.
        """,
        """
        Affected versions: checkout-api 2.4.0. Versions 2.3.x and earlier
        are not affected because the pricing module was rewritten in 2.4.0.
        Fixed in 2.4.1.
        """,
        """
        Remediation: roll forward to checkout-api 2.4.1 or later. Interim
        mitigation: roll back to 2.3.7 via Argo Rollouts and enable the
        pricing query cache at the gateway.
        """,
        """
        Detection: watch p99 latency on the checkout-api Grafana board.
        The SLO alert fires above 800ms for 5 minutes. Regressed pods
        first appeared in us-ashburn-1 since 2026-01-15; treat them as the
        canary cohort to drain first.
        """,
    ]

    print("Building runbook knowledge base...")
    await retriever.add_documents(runbook_docs)

    search_tool = retriever.as_tool(
        name="search_runbook_docs",
        description="Search internal runbook documentation for affected versions, remediation, detection guidance, and rollout cohorts.",
    )

    agent = Agent(
        model=get_model(max_tokens=512),
        tools=[search_tool],
        system_prompt="""You are ATLAS, the platform team's on-call SRE
copilot. You answer over the Index — the internal runbook knowledge base.

When responders ask questions:
1. Use the search_runbook_docs tool to find relevant runbook content
2. Answer based on the search results only
3. Be concise and accurate
4. If the runbooks don't cover it, say so — never guess

Always cite the runbook text you relied on.""",
        max_iterations=3,
    )

    questions = [
        "Which checkout-api versions are affected by the latency regression?",
        "What is the recommended remediation?",
    ]

    for question in questions:
        print("\n" + "-" * 40)
        print(f"Responder: {question}")
        result = await agent.arun(question)
        print(f"Copilot: {result.message}")


# =============================================================================
# Step 3: Mixed tool set — runbook search + calculator + date.
# =============================================================================


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Example: calculate('350 / 1200 * 100')"""
    try:
        return f"Result: {safe_math_eval(expression)}"
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        return f"Error: {e}"


@tool
def get_current_date() -> str:
    """Get the current date and time."""
    return f"Current date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"


async def multi_tool_rag_agent():
    print("\n" + "=" * 60)
    print("Notebook 53: ATLAS — Multi-Tool Rollout Copilot")
    print("=" * 60)

    embedder = get_embedder()
    store = _make_store(dim=embedder.config.dimension)
    retriever = RAGRetriever(embedder=embedder, store=store)

    rollout_docs = [
        "The fleet runs 1200 checkout-api pods across three regions.",
        "As of the last rollout, 350 pods remain on the regressed version 2.4.0.",
        "Runbook RB-014 was opened on 2026-01-20.",
        "Argo Rollouts drains roughly 50 pods per canary step per day.",
        "The pricing query-cache mitigation is active on 100% of user-facing pods.",
    ]

    await retriever.add_documents(rollout_docs)

    search_tool = retriever.as_tool(
        name="search_rollout_data",
        description="Search rollout data for runbook RB-014 including pod counts, canary progress, and mitigations.",
    )

    agent = Agent(
        model=get_model(max_tokens=512),
        tools=[search_tool, calculate, get_current_date],
        system_prompt="""You are a release-engineering analyst assistant.

You have access to:
- search_rollout_data: Search fleet rollout documentation
- calculate: Perform mathematical calculations
- get_current_date: Get current date

Use tools as needed to answer questions accurately.""",
        max_iterations=5,
    )

    queries = [
        "How many pods are still on the regressed checkout-api version?",
        "What percentage of the fleet is still regressed?",
        "How many days has RB-014 been open as of today?",
    ]

    for query in queries:
        print("\n" + "-" * 40)
        print(f"Responder: {query}")
        result = await agent.arun(query)
        print(f"Copilot: {result.message}")


# =============================================================================
# Step 4: Streaming — print tool/think events as they fire.
# =============================================================================


async def rag_with_streaming():
    print("\n" + "=" * 60)
    print("Notebook 53: ATLAS with Streaming")
    print("=" * 60)

    embedder = get_embedder()
    store = _make_store(dim=embedder.config.dimension)
    retriever = RAGRetriever(embedder=embedder, store=store)

    docs = [
        "The AURORA migration moves stateful services onto the new us-ashburn-1 cluster.",
        "AURORA cutover proceeds region by region inside the weekly freeze window.",
        "In scope so far are the gateway and cache tiers; no customer-facing impact observed.",
    ]
    await retriever.add_documents(docs)

    search_tool = retriever.as_tool(
        name="search_changelog", description="Search change and rollout notes"
    )

    agent = Agent(
        model=get_model(max_tokens=512),
        tools=[search_tool],
        system_prompt="Search the change notes and answer the responder's question.",
        max_iterations=2,
    )

    print("Streaming copilot response...\n")

    async for event in agent.run("Which tiers does the AURORA migration affect?"):
        if isinstance(event, ToolStartEvent):
            print(f"[Tool] Searching: {event.tool_name}...")
        elif isinstance(event, ToolCompleteEvent):
            # The retriever tool returns a dict; the loop hands it to the
            # model (and to us) as its JSON text, so decode it to count hits.
            payload = json.loads(event.result) if event.result else {}
            print(f"[Tool] Found {payload.get('total', 0)} result(s)")
        elif isinstance(event, ThinkEvent):
            reasoning = event.reasoning or ""
            print(f"[Agent] {reasoning[:100]}...")


# =============================================================================
# Step 5: Answer-grounding gate — before ATLAS ships a remediation answer,
#         the GSAR scorer partitions its claims into grounded vs. ungrounded
#         and decide() picks proceed / regenerate / replan. A fix anchored
#         in a retrieved runbook chunk proceeds; a hunch with no retrieved
#         text replans rather than guess. Pure-Python — no embeddings or
#         model needed, so it runs offline.
# =============================================================================


def answer_grounding_gate():
    print("\n" + "=" * 60)
    print("Notebook 53: Answer-grounding gate (GSAR decide)")
    print("=" * 60)

    # ATLAS drafted a remediation answer for INC-2026-014. Before it ships,
    # we partition the answer's claims by where the evidence came from. A
    # claim that points at a retrieved runbook chunk is grounded; a claim
    # the model just inferred is ungrounded.
    question = "What is the remediation for the checkout-api latency regression?"
    print(f"Responder: {question}\n")

    # Grounded case: every claim in the drafted answer traces to a retrieved
    # runbook chunk and the deploy log that pinned the root cause — concrete,
    # traceable evidence, so the answer ships.
    grounded = Partition(
        grounded=[
            Claim(
                text="RB-014 says to roll forward to checkout-api 2.4.1 to clear the regression.",
                type=EvidenceType.TOOL_MATCH,
                evidence_refs=["retriever:search_runbook_docs:doc=RB-014:offset=0"],
            ),
            Claim(
                text="The interim mitigation is to roll back to 2.3.7 via Argo Rollouts.",
                type=EvidenceType.SPECIFIC_DATA,
                evidence_refs=["retriever:search_runbook_docs:doc=RB-014:offset=2"],
            ),
            Claim(
                text="The 2.4.0 deploy introduced the N+1 query in the pricing module.",
                type=EvidenceType.SIGNAL_MATCH,
                evidence_refs=["deploy_log:service=checkout-api:rev=2.4.0"],
            ),
        ],
    )
    score = gsar_score(grounded)
    decision = decide(score)
    print("Grounded case (answer cites retrieved runbook chunks + deploy log):")
    print(f"  S={score:.4f}  decision={decision.value}")
    if decision is Decision.PROCEED:
        print("  SHIP  — roll forward to 2.4.1; interim roll back to 2.3.7 via Argo.")
        print(f"    evidence: {[r for c in grounded.grounded for r in c.evidence_refs]}")
    else:
        print("  HOLD  — answer not grounded enough to ship.")

    # Speculative case: a hunch that the same fix applies to the unrelated
    # orders-db incident, with no retrieved runbook text to point at. Under
    # GSAR this is an ungrounded claim — decide() replans rather than ship a
    # fabricated remediation.
    speculative = Partition(
        ungrounded=[
            Claim(
                text="The same 2.4.1 rollback probably fixes the orders-db pool exhaustion too.",
                type=EvidenceType.INFERENCE,
            ),
        ],
    )
    maybe_score = gsar_score(speculative)
    maybe_decision = decide(maybe_score)
    print("\nSpeculative case (a hunch, no retrieved evidence):")
    print(f"  S={maybe_score:.4f}  decision={maybe_decision.value}")
    if maybe_decision is Decision.PROCEED:
        print(f"  SHIP  — {speculative.ungrounded[0].text}")
    else:
        print("  HOLD  — replanning: cross-incident claim has no runbook support.")


async def main():
    missing = _missing_env()
    if missing:
        print("\n--- Notebook 53: ATLAS on-call SRE copilot ---")
        print(
            "Embedding credentials not set; skipping the retrieval sections "
            "so this file still runs cleanly in CI. The answer-grounding gate "
            "below is pure-Python and runs anyway.\n"
        )
        for name in missing:
            print(f"  - {name}")
        print("\nSet OPENAI_API_KEY (for embeddings) to run ATLAS over the Index.")
    else:
        await rag_as_tool()
        await simple_rag_agent()
        await multi_tool_rag_agent()
        await rag_with_streaming()

    # Always run — the grounding gate needs no embeddings or model.
    answer_grounding_gate()

    print("\n" + "=" * 60)
    print("Done. Next: notebook 54 — reasoning patterns.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
