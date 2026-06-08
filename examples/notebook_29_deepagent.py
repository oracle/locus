#!/usr/bin/env python3
# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Notebook 29: DeepAgent — a research-shaped Agent factory with RAG grounding.

``create_deepagent`` bundles the configuration patterns for deep research
into one call: reflexion + grounding on by default, a typed termination
algebra, plus opt-in filesystem scratchspace, todo tracking, subagent
spawning, and datastore auto-wiring. The result is a plain ``tulip.Agent``
— every hook, checkpointer, and observability primitive in the SDK
attaches normally.

- Typed termination: ``(ToolCalled('submit') & ConfidenceMet(0.85))
  | TokenLimit(N) | MaxIterations(M)`` — composable and testable
  without running a model.
- Filesystem-as-memory: opt in to ``write_file`` / ``read_file`` for
  scratchpad notes that persist across iterations without bloating
  context.
- Todo tracking: ``write_todos`` / ``read_todos`` backed by a
  ``TodoState`` the caller can inspect after the run.
- Subagent dispatch: ``SubAgentDef`` + ``task(...)`` for one-shot
  delegated investigations whose trajectories never reach the parent's
  context window.
- ``datastores={name: {retriever, description, top_k}}``: auto-wire a
  ``search_<name>`` tool from any ``RAGRetriever`` and prepend a routing
  block to the system prompt. Part 5 wires an ``InMemoryVectorStore`` +
  ``OpenAIEmbeddings`` retriever and gracefully skips when no embedding
  key is set.

Run it:
    python examples/notebook_29_deepagent.py

The default provider is the bundled mock model; set TULIP_MODEL_PROVIDER
to openai / anthropic for a live model. Set
``TULIP_MODEL_PROVIDER=mock`` for offline runs.

Prerequisites:
- Notebook 06 (Agent basics).
- Notebook 15 (typed termination) — the algebra DeepAgent uses internally.
- For Part 5 only: ``OPENAI_API_KEY`` for embeddings. Absent it, Part 5
  exits cleanly and the rest still runs.
"""

from __future__ import annotations

import asyncio

from config import get_model
from pydantic import BaseModel, Field

from tulip.deepagent import (
    SubAgentDef,
    TodoState,
    create_deepagent,
    make_todo_tools,
)
from tulip.observability import get_event_bus, run_context
from tulip.tools import tool


# =============================================================================
# Shared domain — a tiny module catalogue the agent will research
# =============================================================================

_MODULE_CATALOGUE = {
    "tulip.router": {
        "description": "Meta-orchestration layer — GoalFrame extraction, protocol registry, policy gate, cognitive compiler.",
        "public_api": [
            "Router",
            "GoalFrame",
            "TaskType",
            "ProtocolRegistry",
            "PolicyGate",
            "CognitiveCompiler",
        ],
        "since": "0.2.0",
    },
    "tulip.observability": {
        "description": "In-process SSE pub/sub bus — EventBus, run_context, canonical EV_* constants.",
        "public_api": [
            "EventBus",
            "EventBusHook",
            "run_context",
            "get_event_bus",
            "emit",
            "emit_sync",
        ],
        "since": "0.2.0",
    },
    "tulip.deepagent": {
        "description": "Research-shaped agent factory: create_deepagent, filesystem tools, todos, subagents.",
        "public_api": [
            "create_deepagent",
            "SubAgentDef",
            "TodoState",
            "make_filesystem_tools",
            "make_todo_tools",
        ],
        "since": "0.2.0",
    },
}


@tool
def list_modules() -> list[str]:
    """List all modules available in the tulip catalogue."""
    return list(_MODULE_CATALOGUE.keys())


@tool
def inspect_module(name: str) -> dict:
    """Return description, public API, and version for a module.

    Args:
        name: Module dotted name, e.g. ``tulip.router``.

    Returns:
        Dict with ``description``, ``public_api``, and ``since``.
    """
    if name not in _MODULE_CATALOGUE:
        return {"error": f"module '{name}' not found"}
    return _MODULE_CATALOGUE[name]


@tool
def count_public_symbols(name: str) -> int:
    """Return the number of public symbols exported by a module.

    Args:
        name: Module dotted name.
    """
    entry = _MODULE_CATALOGUE.get(name)
    if not entry:
        return 0
    return len(entry["public_api"])


# =============================================================================
# Typed output — what every Part submits when confidence is high enough
# =============================================================================


class ModuleReport(BaseModel):
    module: str = Field(description="Dotted module name researched.")
    summary: str = Field(description="2-3 sentence summary of what the module does.")
    public_symbols: list[str] = Field(description="All public symbols in the module.")
    available_since: str = Field(description="Version the module was introduced.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the report (0–1).")


@tool
def submit_research(report: ModuleReport) -> str:
    """Submit the completed research report. Call when confidence ≥ 0.85.

    Args:
        report: The completed ``ModuleReport``.
    """
    return f"submitted: {report.module} ({report.confidence:.0%} confidence)"


# =============================================================================
# Part 1 — minimal create_deepagent
# =============================================================================


async def part1_basic() -> None:
    """Reflexion + grounding on, typed termination, nothing else."""
    print("\n--- Part 1: basic create_deepagent ---")

    agent = create_deepagent(
        model=get_model(),
        tools=[list_modules, inspect_module, count_public_symbols, submit_research],
        system_prompt=(
            "You are a tulip module researcher. "
            "Use list_modules, inspect_module, and count_public_symbols to gather facts. "
            "Submit a complete ModuleReport via submit_research once you reach ≥ 0.85 confidence."
        ),
        output_schema=ModuleReport,
        submit_tool="submit_research",
        min_confidence=0.85,
        max_iterations=12,
    )

    result = agent.run_sync("Research the tulip.observability module.")
    print("protocol terminated:", result.stop_reason)
    if result.parsed:
        rpt: ModuleReport = result.parsed  # type: ignore[assignment]
        print(f"module:    {rpt.module}")
        print(f"symbols:   {', '.join(rpt.public_symbols[:4])} …")
        print(f"confidence:{rpt.confidence:.0%}")


# =============================================================================
# Part 2 — filesystem scratchpad + todos
# =============================================================================


async def part2_filesystem_and_todos() -> None:
    """Enable filesystem tools for scratchpad notes and todos for tracking."""
    print("\n--- Part 2: filesystem scratchspace + todos ---")

    todo_state = TodoState()

    agent = create_deepagent(
        model=get_model(),
        tools=[list_modules, inspect_module, count_public_symbols, submit_research],
        system_prompt=(
            "You are a tulip module researcher. "
            "Use write_file to take scratchpad notes as you gather facts. "
            "Use write_todos to track which modules you've checked. "
            "Submit when you have a complete report with ≥ 0.85 confidence."
        ),
        output_schema=ModuleReport,
        submit_tool="submit_research",
        min_confidence=0.85,
        max_iterations=16,
        enable_filesystem=True,
        enable_todos=True,
        todo_state=todo_state,
    )

    result = agent.run_sync("Research all three modules in the catalogue.")
    print("terminated:", result.stop_reason)
    print("todos after run:")
    for todo in todo_state.snapshot():
        print(f"  [{todo.status}] {todo.content[:60]}")


# =============================================================================
# Part 3 — subagent dispatch
# =============================================================================


async def part3_subagents() -> None:
    """Delegate to a focused subagent; only its final answer reaches the parent."""
    print("\n--- Part 3: subagent dispatch ---")

    # The subagent only carries one tool — focused, cheap, easy to test.
    symbol_analyst = SubAgentDef(
        name="symbol_analyst",
        description="Deep-dives on a single module's public API.",
        system_prompt="Inspect the given module and return a plain list of its public symbols.",
        tools=[inspect_module],
        max_iterations=4,
    )

    agent = create_deepagent(
        model=get_model(),
        tools=[list_modules, submit_research],
        system_prompt=(
            "Use list_modules to discover modules, then delegate symbol analysis "
            "to the symbol_analyst subagent via the task() tool. "
            "Submit a ModuleReport for tulip.router once you have the symbol list."
        ),
        output_schema=ModuleReport,
        submit_tool="submit_research",
        min_confidence=0.8,
        max_iterations=12,
        subagents=[symbol_analyst],
    )

    result = agent.run_sync("Research tulip.router using the symbol_analyst subagent.")
    print("terminated:", result.stop_reason)
    if result.parsed:
        rpt: ModuleReport = result.parsed  # type: ignore[assignment]
        print(f"symbols from subagent: {rpt.public_symbols}")


# =============================================================================
# Part 4 — observe deepagent.* events on the SSE bus
# =============================================================================


async def part4_observability() -> None:
    """Subscribe to deepagent.* events: subagent.*, fs.*, todo.*."""
    print("\n--- Part 4: deepagent.* SSE events ---")

    todo_state = TodoState()
    symbol_analyst = SubAgentDef(
        name="symbol_analyst",
        description="Inspect one module.",
        system_prompt="Inspect the given module and list its public symbols.",
        tools=[inspect_module],
        max_iterations=4,
    )

    agent = create_deepagent(
        model=get_model(),
        tools=[list_modules, submit_research],
        system_prompt=(
            "Use list_modules, delegate symbol analysis via task(), "
            "write scratchpad notes, track progress with todos. "
            "Submit a report for tulip.deepagent."
        ),
        output_schema=ModuleReport,
        submit_tool="submit_research",
        min_confidence=0.8,
        max_iterations=14,
        enable_filesystem=True,
        enable_todos=True,
        todo_state=todo_state,
        subagents=[symbol_analyst],
    )

    deepagent_events: list[str] = []

    async def _collect(rid: str) -> None:
        async for ev in get_event_bus().subscribe(rid):
            if ev.event_type.startswith("deepagent."):
                deepagent_events.append(ev.event_type)

    async with run_context() as rid:
        collector = asyncio.create_task(_collect(rid))
        result = agent.run_sync("Research tulip.deepagent module.")
        await asyncio.sleep(0.1)
        collector.cancel()

    print("deepagent.* events seen:")
    for ev_type in sorted(set(deepagent_events)):
        count = deepagent_events.count(ev_type)
        print(f"  {ev_type} × {count}")

    print("terminated:", result.stop_reason)


# =============================================================================
# Part 5 — auto-wired `search_<name>` tools against a vector store
# =============================================================================


async def part5_datastores() -> None:
    """Pass ``datastores={name: {retriever, description, top_k}}`` and the
    factory appends a ``search_<name>`` tool plus a per-store routing block
    in the system prompt. The agent then picks the right store per query.

    This Part requires an embedding key (``OPENAI_API_KEY``). Without it,
    Part 5 exits cleanly and the earlier parts still run.
    """
    import os

    required = ("OPENAI_API_KEY",)
    missing = [n for n in required if not os.environ.get(n)]
    if missing:
        print("\n[multi_datastore_routing] skipped — missing env vars:")
        for n in missing:
            print(f"  - {n}")
        return

    from tulip.rag import InMemoryVectorStore, OpenAIEmbeddings, RAGRetriever

    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    probe = await embedder.embed_query("probe")
    store = InMemoryVectorStore(
        dimension=len(probe.embedding),
        distance_metric="COSINE",
    )
    retriever = RAGRetriever(embedder=embedder, store=store)
    await retriever.add_documents(
        [
            "Hepcidin is the master regulator of iron homeostasis.",
            "Ferritin is the primary iron storage protein.",
            "Transferrin saturation below 16% suggests iron deficiency.",
            "Phlebotomy is first-line treatment for hereditary hemochromatosis.",
        ]
    )

    agent = create_deepagent(
        model=get_model(),
        tools=[],
        system_prompt=(
            "You are a medical research assistant. When asked a hematology "
            "question, call search_medical first, then answer briefly with "
            "(doc-NN) citations."
        ),
        datastores={
            "medical": {
                "retriever": retriever,
                "description": "iron metabolism, anemia, hemochromatosis",
                "top_k": 3,
            }
        },
        reflexion=False,
        grounding=False,
        max_iterations=4,
    )

    result = agent.run_sync("What regulates iron homeostasis? Cite the retrieved doc.")
    print("part 5 response:", (result.text or "")[:300])
    print("part 5 tool calls:", len(result.tool_executions or ()))


async def main() -> None:
    await part1_basic()
    await part2_filesystem_and_todos()
    await part3_subagents()
    await part4_observability()
    await part5_datastores()


if __name__ == "__main__":
    asyncio.run(main())
