# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL
"""Notebook 40: RAG agents — wire a retriever into an agent's tool set.

Once you have a vector store full of documents (notebook 38 / 39), the
next step is to let an agent reach into it. ``RAGRetriever.as_tool()``
turns the retriever into an ordinary Tulip tool that the agent picks
up alongside any other ``@tool`` you define.

- ``retriever.as_tool(name, description)`` — convert a retriever into a
  callable tool for the agent.
- Single-tool Q&A agent against a product knowledge base.
- Mixed tool set — RAG search alongside a calculator and a date tool.
- Streaming events from the agent while it searches and answers.
- Best-practice notes on chunk size, prompt design, and metadata
  filters.

Backend: ``InMemoryVectorStore`` keeps the demo dependency-free. Swap
``_make_store`` for any other Tulip vector store implementation
(pgvector, OpenSearch, Qdrant, Chroma) for a durable backend.

Run it:
    export OPENAI_API_KEY=sk-...
    python examples/notebook_40_rag_agents.py

    # Offline (skips the live demo cleanly when the key is missing):
    python examples/notebook_40_rag_agents.py
"""

import ast
import asyncio
import operator as _op
import os
import sys

from tulip.rag import InMemoryVectorStore


def _missing_env() -> list[str]:
    return [name for name in ("OPENAI_API_KEY",) if not os.environ.get(name)]


def _make_store(suffix: str, dim: int) -> InMemoryVectorStore:
    # One store per section so sections don't stomp on each other.
    return InMemoryVectorStore(dimension=dim, distance_metric="COSINE")


_SAFE_MATH_BIN_OPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.FloorDiv: _op.floordiv,
    ast.Mod: _op.mod,
    ast.Pow: _op.pow,
}
_SAFE_MATH_UNARY_OPS = {ast.USub: _op.neg, ast.UAdd: _op.pos}


def _safe_math_eval(expression: str) -> float:
    # AST-only arithmetic — disallows names, calls, attribute access, etc.
    # so the calculator tool can't be turned into a sandbox escape.
    tree = ast.parse(expression, mode="eval")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_MATH_BIN_OPS:
            return _SAFE_MATH_BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_MATH_UNARY_OPS:
            return _SAFE_MATH_UNARY_OPS[type(node.op)](_eval(node.operand))
        raise ValueError("Unsupported expression")

    return _eval(tree)


# =============================================================================
# Step 1: RAGRetriever.as_tool() — turn a retriever into a normal agent tool.
# =============================================================================


async def rag_as_tool():
    print("=" * 60)
    print("Notebook 40: RAG as a Tool")
    print("=" * 60)

    from tulip.rag import RAGRetriever

    embedder = get_embedder()
    if not embedder:
        return

    store = _make_store(suffix="as_tool", dim=embedder.config.dimension)
    retriever = RAGRetriever(embedder=embedder, store=store)

    knowledge = [
        "Tulip is a Python framework for building AI agents.",
        "Tulip supports multiple LLM providers including OpenAI and Anthropic.",
        "Agents in Tulip can use tools to interact with external systems.",
        "RAG in Tulip enables agents to search through documents.",
        "Tulip uses async/await for efficient concurrent operations.",
    ]

    print("Building knowledge base...")
    await retriever.add_documents(knowledge)
    print(f"  Added {len(knowledge)} documents")

    search_tool = retriever.as_tool(
        name="search_knowledge",
        description="Search the knowledge base for information about Tulip.",
    )

    print(f"\nCreated tool: {search_tool.name}")
    print(f"Description: {search_tool.description}")

    print("\n" + "-" * 40)
    print("Testing tool directly...")

    result = await search_tool("What LLM providers does Tulip support?")

    print("\nQuery: 'What LLM providers does Tulip support?'")
    print(f"Results found: {result['total']}")
    for i, doc in enumerate(result["results"], 1):
        print(f"  {i}. Score: {doc['score']:.4f}")
        print(f"     {doc['content'][:60]}...")


# =============================================================================
# Step 2: A small Q&A agent that grounds answers in product docs.
# =============================================================================


async def simple_rag_agent():
    print("\n" + "=" * 60)
    print("Notebook 40: Simple RAG Agent")
    print("=" * 60)

    from tulip.agent import Agent
    from tulip.rag import RAGRetriever

    embedder = get_embedder()
    model = get_model()
    if not embedder or not model:
        return

    store = _make_store(suffix="simple", dim=embedder.config.dimension)
    retriever = RAGRetriever(embedder=embedder, store=store)

    product_docs = [
        """
        ProductX is an enterprise data platform launched in 2024.
        It supports real-time data processing at scale.
        ProductX can handle up to 1 million events per second.
        """,
        """
        ProductX pricing starts at $500/month for the starter plan.
        The enterprise plan costs $5000/month and includes support.
        All plans include a 30-day free trial.
        """,
        """
        ProductX integrates with popular tools like Kafka, Spark, and Flink.
        It provides REST APIs for custom integrations.
        SDKs are available for Python, Java, and Go.
        """,
        """
        ProductX requires a minimum of 4 CPU cores and 8GB RAM.
        For production, we recommend 16 cores and 32GB RAM.
        Cloud deployment is available on AWS, GCP, and Azure.
        """,
    ]

    print("Building product knowledge base...")
    await retriever.add_documents(product_docs)

    search_tool = retriever.as_tool(
        name="search_product_docs",
        description="Search ProductX documentation for information about features, pricing, requirements, and integrations.",
    )

    agent = Agent(
        model=model,
        tools=[search_tool],
        system_prompt="""You are a helpful product assistant for ProductX.

When users ask questions:
1. Use the search_product_docs tool to find relevant information
2. Answer based on the search results
3. Be concise and accurate
4. If you can't find the answer, say so

Always cite information from the documentation.""",
        max_iterations=3,
    )

    questions = [
        "How much does ProductX cost?",
        "What are the system requirements?",
    ]

    for question in questions:
        print("\n" + "-" * 40)
        print(f"User: {question}")
        result = agent.run_sync(question)
        print(f"Agent: {result.message}")


# =============================================================================
# Step 3: Mixed tool set — RAG search + calculator + date.
# =============================================================================


async def multi_tool_rag_agent():
    print("\n" + "=" * 60)
    print("Notebook 40: Multi-Tool RAG Agent")
    print("=" * 60)

    from datetime import datetime

    from tulip.agent import Agent
    from tulip.rag import RAGRetriever
    from tulip.tools import tool

    embedder = get_embedder()
    model = get_model()
    if not embedder or not model:
        return

    store = _make_store(suffix="multi_tool", dim=embedder.config.dimension)
    retriever = RAGRetriever(embedder=embedder, store=store)

    finance_docs = [
        "Company ABC reported revenue of $10.5 billion in Q3 2024.",
        "Company ABC has 15,000 employees worldwide.",
        "Company ABC stock price is currently $150 per share.",
        "Company ABC was founded in 2010 in San Francisco.",
        "Company ABC expects 15% revenue growth in 2025.",
    ]

    await retriever.add_documents(finance_docs)

    @tool
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression. Example: calculate('150 * 1000')"""
        try:
            return f"Result: {_safe_math_eval(expression)}"
        except (ValueError, SyntaxError, ZeroDivisionError) as e:
            return f"Error: {e}"

    @tool
    def get_current_date() -> str:
        """Get the current date and time."""
        return f"Current date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    search_tool = retriever.as_tool(
        name="search_company_info",
        description="Search for information about Company ABC including financials, employees, and history.",
    )

    agent = Agent(
        model=model,
        tools=[search_tool, calculate, get_current_date],
        system_prompt="""You are a financial analyst assistant.

You have access to:
- search_company_info: Search company documentation
- calculate: Perform mathematical calculations
- get_current_date: Get current date

Use tools as needed to answer questions accurately.""",
        max_iterations=5,
    )

    queries = [
        "What is Company ABC's revenue?",
        "If the stock price doubles, what would it be?",
        "How old is Company ABC today?",
    ]

    for query in queries:
        print("\n" + "-" * 40)
        print(f"User: {query}")
        result = agent.run_sync(query)
        print(f"Agent: {result.message}")


# =============================================================================
# Step 4: Streaming — print tool/think events as they fire.
# =============================================================================


async def rag_with_streaming():
    print("\n" + "=" * 60)
    print("Notebook 40: RAG with Streaming")
    print("=" * 60)

    from tulip.agent import Agent
    from tulip.core.events import ThinkEvent, ToolCompleteEvent, ToolStartEvent
    from tulip.rag import RAGRetriever

    embedder = get_embedder()
    model = get_model()
    if not embedder or not model:
        return

    store = _make_store(suffix="streaming", dim=embedder.config.dimension)
    retriever = RAGRetriever(embedder=embedder, store=store)

    docs = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models learn patterns from data.",
        "Neural networks are inspired by biological neurons.",
    ]
    await retriever.add_documents(docs)

    search_tool = retriever.as_tool(name="search", description="Search documents")

    agent = Agent(
        model=model,
        tools=[search_tool],
        system_prompt="Search for information and provide helpful answers.",
        max_iterations=2,
    )

    print("Streaming agent response...\n")

    async for event in agent.run("What do neural networks do?"):
        if isinstance(event, ToolStartEvent):
            print(f"[Tool] Searching: {event.tool_name}...")
        elif isinstance(event, ToolCompleteEvent):
            print(f"[Tool] Found {len(event.result.get('results', []))} results")
        elif isinstance(event, ThinkEvent):
            print(f"[Agent] {event.reasoning[:100]}...")


# =============================================================================
# Step 5: Best-practice notes — chunking, prompt design, metadata filters.
# =============================================================================


async def rag_best_practices():
    print("\n" + "=" * 60)
    print("Notebook 40: RAG Best Practices")
    print("=" * 60)

    print("""
Best Practices for RAG Agents:

1. CHUNK SIZE MATTERS
   - Too small: Lose context
   - Too large: Dilute relevance
   - Recommended: 500-1000 characters with 50-100 overlap

2. QUALITY OVER QUANTITY
   - Clean your documents before indexing
   - Remove boilerplate, headers, footers
   - Keep source metadata for citations

3. PROMPT ENGINEERING
   - Tell the agent when to search
   - Instruct it to cite sources
   - Handle "not found" gracefully

4. HYBRID APPROACHES
   - Combine keyword + semantic search
   - Use metadata filters to narrow scope
   - Rerank results for better precision

5. EVALUATION
   - Test with real user questions
   - Measure retrieval relevance
   - Track answer quality over time

6. PRODUCTION CONSIDERATIONS
   - Use persistent vector stores (pgvector, OpenSearch, Qdrant, Chroma)
   - Implement caching for embeddings
   - Monitor latency and costs
""")

    # Example of good prompt engineering
    print("-" * 40)
    print("Example System Prompt for RAG Agent:")
    print("-" * 40)
    print("""
You are a helpful assistant with access to a knowledge base.

INSTRUCTIONS:
1. When asked a question, ALWAYS search the knowledge base first
2. Base your answers ONLY on the search results
3. If search returns no relevant results, say "I couldn't find information about that"
4. Quote relevant passages when helpful
5. If multiple documents are relevant, synthesize the information

RESPONSE FORMAT:
- Start with a direct answer
- Provide supporting details from the documents
- End with "Source: [document reference]" if applicable
""")


# =============================================================================
# Helpers — picks the embedder and model implementation based on env.
# =============================================================================


def get_embedder():
    """Pick an embedder from whichever credentials are present."""
    if os.environ.get("OPENAI_API_KEY"):
        from tulip.rag.embeddings import OpenAIEmbeddings

        return OpenAIEmbeddings(model="text-embedding-3-small")

    if os.environ.get("COHERE_API_KEY"):
        from tulip.rag.embeddings import CohereEmbeddings

        return CohereEmbeddings(model="embed-english-v3.0")

    print("No embedding credentials found")
    return None


def get_model():
    """LLM model from the shared notebook config — honours every env var."""
    from config import get_model as _get_model

    return _get_model(max_tokens=512)


# =============================================================================
# Main
# =============================================================================


async def main():
    missing = _missing_env()
    if missing:
        print("\n--- Notebook 40: RAG agents ---")
        print(
            "Required environment variables not set; skipping the live "
            "demo so this file still runs cleanly in CI.\n"
        )
        for name in missing:
            print(f"  - {name}")
        print("\nSet OPENAI_API_KEY (for embeddings), then re-run.")
        return

    await rag_as_tool()
    await simple_rag_agent()
    await multi_tool_rag_agent()
    await rag_with_streaming()
    await rag_best_practices()

    print("\n" + "=" * 60)
    print("Done. Next: notebook 41 — MCP integration.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
