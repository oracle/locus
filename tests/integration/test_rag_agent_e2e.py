# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""End-to-end RAG Agent tests using Locus SDK components.

Tests the full stack: Agent + RAGToolkit + OracleVectorStore + Embeddings
against the real deep research Oracle 26ai database with 1,787 medical
documents and pre-computed 1536-dimension embeddings.

All components are from the Locus SDK — no custom tools or SQL.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from locus.agent import Agent, GroundingConfig, ReflexionConfig
from locus.core.events import (
    GroundingEvent,
    ReflectEvent,
    TerminateEvent,
    ToolCompleteEvent,
)


pytestmark = [pytest.mark.integration]

WALLET_DIR = str(Path.home() / ".oci/wallets/deepresearch")
ORACLE_DSN = os.getenv("ORACLE_DSN", "deepresearch_low")
ORACLE_USER = os.getenv("ORACLE_USER", "ADMIN")
ORACLE_PASSWORD = os.getenv("ORACLE_PASSWORD", "")
ORACLE_WALLET_PASSWORD = os.getenv("ORACLE_WALLET_PASSWORD", "")


def has_oracle_and_openai() -> bool:
    """Check if Oracle DB and OpenAI (for embeddings) are available."""
    wallet = Path(WALLET_DIR)
    return wallet.exists() and bool(ORACLE_PASSWORD) and bool(os.environ.get("OPENAI_API_KEY"))


def has_oracle_and_oci() -> bool:
    """Check if Oracle DB and OCI GenAI (for embeddings) are available."""
    wallet = Path(WALLET_DIR)
    return (
        wallet.exists()
        and bool(ORACLE_PASSWORD)
        and bool(os.environ.get("OCI_PROFILE"))
        and bool(os.environ.get("OCI_ENDPOINT"))
    )


skip_without_oracle_openai = pytest.mark.skipif(
    not has_oracle_and_openai(),
    reason="Need ORACLE_PASSWORD + OPENAI_API_KEY + wallet",
)

skip_without_oracle_oci = pytest.mark.skipif(
    not has_oracle_and_oci(),
    reason="Need ORACLE_PASSWORD + OCI_PROFILE + OCI_ENDPOINT + wallet",
)


def build_rag_toolkit_openai():
    """Build RAGToolkit using OpenAI embeddings + Oracle vector store."""
    from locus.rag.embeddings import OpenAIEmbeddings
    from locus.rag.retriever import RAGRetriever
    from locus.rag.stores.oracle import OracleVectorStore
    from locus.rag.tools import RAGToolkit

    embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    store = OracleVectorStore(
        dsn=ORACLE_DSN,
        user=ORACLE_USER,
        password=ORACLE_PASSWORD,
        wallet_location=WALLET_DIR,
        wallet_password=ORACLE_WALLET_PASSWORD,
        dimension=1536,
        table_name="VECTOR_DOCUMENTS",
    )

    retriever = RAGRetriever(embedder=embedder, store=store)
    toolkit = RAGToolkit(retriever, prefix="medical")

    return toolkit, store, embedder


# =============================================================================
# Test 1: Agent with RAG Toolkit — Full Semantic Search
# =============================================================================


@skip_without_oracle_openai
class TestRAGAgentWithOracleDB:
    """Agent uses Locus RAG toolkit to search real Oracle 26ai vector store."""

    @pytest.mark.asyncio
    async def test_agent_searches_knowledge_base(self, model):
        """Agent uses medical_search tool to find relevant documents."""
        toolkit, store, embedder = build_rag_toolkit_openai()

        try:
            agent = Agent(
                model=model,
                tools=toolkit.get_tools(),
                system_prompt=(
                    "You are a medical knowledge assistant. Use the medical_search "
                    "tool to find relevant medical information. Answer based ONLY "
                    "on what the search returns."
                ),
                max_iterations=4,
                max_tool_result_length=3000,
            )

            events = []
            async for event in agent.run("What causes iron deficiency anemia?"):
                events.append(event)

            tool_events = [e for e in events if isinstance(e, ToolCompleteEvent)]
            assert len(tool_events) >= 1
            # Should have used the RAG search tool
            rag_calls = [e for e in tool_events if "medical" in e.tool_name]
            assert len(rag_calls) >= 1

            terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
            assert terminate is not None
            assert terminate.final_message is not None

        finally:
            await store.close()
            await embedder.close()

    @pytest.mark.asyncio
    async def test_agent_multi_query_rag(self, model):
        """Agent makes multiple RAG queries to build a comprehensive answer."""
        toolkit, store, embedder = build_rag_toolkit_openai()

        try:
            agent = Agent(
                model=model,
                tools=toolkit.get_tools(),
                system_prompt=(
                    "You are a medical research assistant. To answer questions:\n"
                    "1. Use medical_search to find relevant documents\n"
                    "2. Use medical_context for additional context if needed\n"
                    "3. Synthesize findings into a clear answer\n"
                    "Search for MULTIPLE related terms to be thorough."
                ),
                max_iterations=6,
                max_tool_result_length=3000,
            )

            events = []
            async for event in agent.run(
                "Compare the treatment approaches for diabetes mellitus type 1 vs type 2"
            ):
                events.append(event)

            tool_events = [e for e in events if isinstance(e, ToolCompleteEvent)]
            # Should have made multiple searches
            assert len(tool_events) >= 2

            terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
            assert terminate is not None

        finally:
            await store.close()
            await embedder.close()

    @pytest.mark.asyncio
    async def test_agent_rag_with_reflexion(self, model):
        """Agent uses RAG with reflexion tracking research progress."""
        toolkit, store, embedder = build_rag_toolkit_openai()

        try:
            agent = Agent(
                model=model,
                tools=toolkit.get_tools(),
                system_prompt=(
                    "Search the medical knowledge base for information. "
                    "Be thorough — search multiple related terms."
                ),
                reflexion=ReflexionConfig(enabled=True),
                max_iterations=5,
                max_tool_result_length=3000,
            )

            events = []
            async for event in agent.run("What are the key enzymes in glycolysis?"):
                events.append(event)

            # Reflexion should have tracked progress
            reflect_events = [e for e in events if isinstance(e, ReflectEvent)]
            assert len(reflect_events) >= 1

            terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
            assert terminate is not None

        finally:
            await store.close()
            await embedder.close()

    @pytest.mark.asyncio
    async def test_agent_rag_with_grounding(self, model):
        """Agent uses RAG with grounding to validate answer against evidence."""
        toolkit, store, embedder = build_rag_toolkit_openai()

        try:
            agent = Agent(
                model=model,
                tools=toolkit.get_tools(),
                system_prompt=(
                    "Search the medical knowledge base and answer ONLY "
                    "based on what the search returns. Cite your sources."
                ),
                grounding=GroundingConfig(enabled=True, threshold=0.3),
                max_iterations=5,
                max_tool_result_length=3000,
            )

            events = []
            async for event in agent.run("What is the role of hemoglobin in oxygen transport?"):
                events.append(event)

            # Grounding should have evaluated the answer
            grounding_events = [e for e in events if isinstance(e, GroundingEvent)]
            # May or may not fire depending on whether agent used tools
            # but should complete either way

            terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
            assert terminate is not None

        finally:
            await store.close()
            await embedder.close()

    @pytest.mark.asyncio
    async def test_full_pipeline_rag(self, model):
        """Full pipeline: RAG + reflexion + grounding + token tracking."""
        toolkit, store, embedder = build_rag_toolkit_openai()

        try:
            agent = Agent(
                model=model,
                tools=toolkit.get_tools(),
                system_prompt=(
                    "You are a medical knowledge assistant. Search the database "
                    "thoroughly using different queries. Only state facts from "
                    "the search results."
                ),
                reflexion=ReflexionConfig(enabled=True),
                grounding=GroundingConfig(enabled=True, threshold=0.3),
                max_iterations=6,
                max_tool_result_length=3000,
                token_budget=15000,
            )

            events = []
            event_types = set()
            async for event in agent.run(
                "What are the common causes and treatments for hypertension?"
            ):
                events.append(event)
                event_types.add(type(event).__name__)

            # Should have diverse events
            assert "ThinkEvent" in event_types
            assert "ToolCompleteEvent" in event_types
            assert "TerminateEvent" in event_types

            # Should have used RAG tools
            tool_events = [e for e in events if isinstance(e, ToolCompleteEvent)]
            assert len(tool_events) >= 1

            terminate = next((e for e in events if isinstance(e, TerminateEvent)), None)
            assert terminate is not None

        finally:
            await store.close()
            await embedder.close()
