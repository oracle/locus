#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 29: DeepAgent — scoped attack-surface recon grounded in prior notes.

``create_deepagent`` bundles the configuration patterns for deep, methodical
investigation into one call: reflexion + grounding on by default, a typed
termination algebra, plus opt-in filesystem scratchspace, todo tracking,
subagent spawning, and datastore auto-wiring. Here the agent runs a scoped,
authorized attack-surface review over an in-scope asset inventory — the
external enumeration step of an engagement (MITRE ATT&CK TA0043
Reconnaissance, T1595 Active Scanning), bounded to assets the rules of
engagement permit. The result is a plain ``tulip.Agent`` — every hook,
checkpointer, and observability primitive in the SDK attaches normally.

- Typed termination: ``(ToolCalled('submit') & ConfidenceMet(0.85))
  | TokenLimit(N) | MaxIterations(M)`` — composable and testable
  without running a model.
- Filesystem-as-memory: opt in to ``write_file`` / ``read_file`` for
  scratchpad notes that persist across iterations without bloating
  context.
- Todo tracking: ``write_todos`` / ``read_todos`` backed by a
  ``TodoState`` the caller can inspect after the run.
- Subagent dispatch: ``SubAgentDef`` + ``task(...)`` for one-shot
  delegated reviews whose trajectories never reach the parent's
  context window.
- ``datastores={name: {retriever, description, top_k}}``: auto-wire a
  ``search_<name>`` tool from any ``RAGRetriever`` and prepend a routing
  block to the system prompt. Part 5 wires an in-memory ``QdrantVectorStore`` +
  ``OpenAIEmbeddings`` retriever over prior recon notes and gracefully
  skips when no embedding key is set.

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
# Shared domain — the in-scope asset inventory the agent will review.
# Scope discipline matters: the agent enumerates exposure on assets named in
# the rules of engagement and refuses anything outside that set.
# =============================================================================

_ASSET_INVENTORY = {
    "vpn.corp.example": {
        "description": "Employee VPN gateway — TLS portal plus OpenVPN endpoint, MFA enforced.",
        "exposed_services": [
            "443/https",
            "1194/openvpn",
        ],
        "last_review": "2026-03",
    },
    "mail.corp.example": {
        "description": "Mail gateway — inbound MX with SPF/DKIM/DMARC enforced, webmail disabled.",
        "exposed_services": [
            "25/smtp",
            "443/https",
        ],
        "last_review": "2026-01",
    },
    "shop.corp.example": {
        "description": "Customer storefront — behind the WAF, rate-limited login, no admin panel.",
        "exposed_services": [
            "443/https",
        ],
        "last_review": "2026-05",
    },
}


@tool
def list_assets() -> list[str]:
    """List all in-scope assets in the engagement inventory."""
    return list(_ASSET_INVENTORY.keys())


@tool
def inspect_asset(name: str) -> dict:
    """Return description, exposed services, and last review date for an asset.

    Args:
        name: Asset hostname, e.g. ``vpn.corp.example``.

    Returns:
        Dict with ``description``, ``exposed_services``, and ``last_review``.
    """
    if name not in _ASSET_INVENTORY:
        return {"error": f"asset '{name}' is not in the engagement scope"}
    return _ASSET_INVENTORY[name]


@tool
def count_exposed_services(name: str) -> int:
    """Return the number of externally exposed services on an asset.

    Args:
        name: Asset hostname.
    """
    entry = _ASSET_INVENTORY.get(name)
    if not entry:
        return 0
    return len(entry["exposed_services"])


# =============================================================================
# Typed output — what every Part submits when confidence is high enough
# =============================================================================


class AssetReport(BaseModel):
    asset: str = Field(description="Hostname of the asset reviewed.")
    summary: str = Field(description="2-3 sentence summary of the asset's exposure.")
    exposed_services: list[str] = Field(description="All externally exposed services.")
    last_review: str = Field(description="Date of the asset's last security review.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the report (0–1).")


@tool
def submit_recon(report: AssetReport) -> str:
    """Submit the completed recon report. Call when confidence ≥ 0.85.

    Args:
        report: The completed ``AssetReport``.
    """
    return f"submitted: {report.asset} ({report.confidence:.0%} confidence)"


# =============================================================================
# Part 1 — minimal create_deepagent
# =============================================================================


async def part1_basic() -> None:
    """Reflexion + grounding on, typed termination, nothing else."""
    print("\n--- Part 1: basic create_deepagent ---")

    agent = create_deepagent(
        model=get_model(),
        tools=[list_assets, inspect_asset, count_exposed_services, submit_recon],
        system_prompt=(
            "You are an attack-surface recon analyst on a scoped, authorized engagement. "
            "Use list_assets, inspect_asset, and count_exposed_services to gather facts "
            "about in-scope assets only. "
            "Submit a complete AssetReport via submit_recon once you reach ≥ 0.85 confidence."
        ),
        output_schema=AssetReport,
        submit_tool="submit_recon",
        min_confidence=0.85,
        max_iterations=12,
    )

    result = agent.run_sync("Review the exposure of vpn.corp.example.")
    print("protocol terminated:", result.stop_reason)
    if result.parsed:
        rpt: AssetReport = result.parsed  # type: ignore[assignment]
        print(f"asset:     {rpt.asset}")
        print(f"services:  {', '.join(rpt.exposed_services[:4])} …")
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
        tools=[list_assets, inspect_asset, count_exposed_services, submit_recon],
        system_prompt=(
            "You are an attack-surface recon analyst on a scoped, authorized engagement. "
            "Use write_file to keep scratchpad notes as you review each asset. "
            "Use write_todos to track which assets you've covered. "
            "Submit when you have a complete report with ≥ 0.85 confidence."
        ),
        output_schema=AssetReport,
        submit_tool="submit_recon",
        min_confidence=0.85,
        max_iterations=16,
        enable_filesystem=True,
        enable_todos=True,
        todo_state=todo_state,
    )

    result = agent.run_sync("Review all three assets in the engagement scope.")
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
    service_analyst = SubAgentDef(
        name="service_analyst",
        description="Deep-dives on a single asset's exposed services.",
        system_prompt="Inspect the given asset and return a plain list of its exposed services.",
        tools=[inspect_asset],
        max_iterations=4,
    )

    agent = create_deepagent(
        model=get_model(),
        tools=[list_assets, submit_recon],
        system_prompt=(
            "Use list_assets to discover in-scope assets, then delegate service analysis "
            "to the service_analyst subagent via the task() tool. "
            "Submit an AssetReport for vpn.corp.example once you have the service list."
        ),
        output_schema=AssetReport,
        submit_tool="submit_recon",
        min_confidence=0.8,
        max_iterations=12,
        subagents=[service_analyst],
    )

    result = agent.run_sync("Review vpn.corp.example using the service_analyst subagent.")
    print("terminated:", result.stop_reason)
    if result.parsed:
        rpt: AssetReport = result.parsed  # type: ignore[assignment]
        print(f"services from subagent: {rpt.exposed_services}")


# =============================================================================
# Part 4 — observe deepagent.* events on the SSE bus
# =============================================================================


async def part4_observability() -> None:
    """Subscribe to deepagent.* events: subagent.*, fs.*, todo.*."""
    print("\n--- Part 4: deepagent.* SSE events ---")

    todo_state = TodoState()
    service_analyst = SubAgentDef(
        name="service_analyst",
        description="Inspect one asset.",
        system_prompt="Inspect the given asset and list its exposed services.",
        tools=[inspect_asset],
        max_iterations=4,
    )

    agent = create_deepagent(
        model=get_model(),
        tools=[list_assets, submit_recon],
        system_prompt=(
            "Use list_assets, delegate service analysis via task(), "
            "write scratchpad notes, track progress with todos. "
            "Submit a report for mail.corp.example."
        ),
        output_schema=AssetReport,
        submit_tool="submit_recon",
        min_confidence=0.8,
        max_iterations=14,
        enable_filesystem=True,
        enable_todos=True,
        todo_state=todo_state,
        subagents=[service_analyst],
    )

    deepagent_events: list[str] = []

    async def _collect(rid: str) -> None:
        async for ev in get_event_bus().subscribe(rid):
            if ev.event_type.startswith("deepagent."):
                deepagent_events.append(ev.event_type)

    async with run_context() as rid:
        collector = asyncio.create_task(_collect(rid))
        result = agent.run_sync("Review the mail.corp.example asset.")
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
    in the system prompt. The agent then grounds its answers in the prior
    recon notes instead of guessing.

    This Part requires an embedding key (``OPENAI_API_KEY``). Without it,
    Part 5 exits cleanly and the earlier parts still run.
    """
    import os

    required = ("OPENAI_API_KEY",)
    missing = [n for n in required if not os.environ.get(n)]
    if missing:
        print("\n[recon_notes_datastore] skipped — missing env vars:")
        for n in missing:
            print(f"  - {n}")
        return

    from tulip.rag import OpenAIEmbeddings, QdrantVectorStore, RAGRetriever

    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    probe = await embedder.embed_query("probe")
    store = QdrantVectorStore(
        dimension=len(probe.embedding),
        location=":memory:",
        distance_metric="cosine",
    )
    retriever = RAGRetriever(embedder=embedder, store=store)
    await retriever.add_documents(
        [
            "vpn.corp.example: only 443 and 1194 exposed after the 2025 hardening pass; "
            "TLS certificate renewed 2026-03.",
            "mail.corp.example: SPF, DKIM and DMARC enforced since 2025-11; webmail disabled.",
            "shop.corp.example sits behind the WAF; login endpoint rate-limited after the "
            "2025 credential-stuffing attempt.",
            "test.corp.example was decommissioned in 2026-01 — any new sighting is a finding.",
        ]
    )

    agent = create_deepagent(
        model=get_model(),
        tools=[],
        system_prompt=(
            "You are an attack-surface recon analyst. When asked about an asset's "
            "history, call search_recon_notes first, then answer briefly with "
            "(doc-NN) citations."
        ),
        datastores={
            "recon_notes": {
                "retriever": retriever,
                "description": "prior recon notes: exposed services, hardening history, "
                "decommissioned hosts",
                "top_k": 3,
            }
        },
        reflexion=False,
        grounding=False,
        max_iterations=4,
    )

    result = agent.run_sync(
        "What do prior recon notes say about vpn.corp.example? Cite the retrieved doc."
    )
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
