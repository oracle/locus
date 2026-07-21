#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 29: DeepAgent — scoped service-fleet reliability review grounded in prior notes.

``create_deepagent`` bundles the configuration patterns for deep, methodical
investigation into one call: reflexion + grounding on by default, a typed
termination algebra, plus opt-in filesystem scratchspace, todo tracking,
subagent spawning, and datastore auto-wiring. Here the agent runs a scoped
reliability review over a service-fleet inventory — the pre-release readiness
step of an on-call rotation, bounded to the services a deploy window actually
touches. The result is a plain ``tulip.Agent`` — every hook, checkpointer,
and observability primitive in the SDK attaches normally.

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
  ``OpenAIEmbeddings`` retriever over prior incident notes and gracefully
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
# Shared domain — the service-fleet inventory the agent will review.
# Scope discipline matters: the agent inspects services inside the deploy
# window and refuses anything outside that set.
# =============================================================================

_SERVICE_INVENTORY = {
    "api-gateway": {
        "description": "Public edge API gateway — terminates TLS, fans out to upstream services, autoscaled 4-12 replicas.",
        "active_alerts": [
            "p99-latency-warn",
            "upstream-5xx-warn",
        ],
        "last_deploy": "2026-06-22",
    },
    "payments-worker": {
        "description": "Async payments settlement worker — consumes the ledger queue, idempotent retries, PCI scope.",
        "active_alerts": [
            "queue-depth-warn",
        ],
        "last_deploy": "2026-06-18",
    },
    "web-frontend": {
        "description": "Server-rendered web frontend — behind the CDN, blue/green rollout, no open alerts this week.",
        "active_alerts": [],
        "last_deploy": "2026-06-24",
    },
}


@tool
def list_services() -> list[str]:
    """List all services in the current deploy-window inventory."""
    return list(_SERVICE_INVENTORY.keys())


@tool
def inspect_service(name: str) -> dict:
    """Return description, active alerts, and last deploy date for a service.

    Args:
        name: Service name, e.g. ``api-gateway``.

    Returns:
        Dict with ``description``, ``active_alerts``, and ``last_deploy``.
    """
    if name not in _SERVICE_INVENTORY:
        return {"error": f"service '{name}' is not in the deploy window"}
    return _SERVICE_INVENTORY[name]


@tool
def count_active_alerts(name: str) -> int:
    """Return the number of firing alerts on a service.

    Args:
        name: Service name.
    """
    entry = _SERVICE_INVENTORY.get(name)
    if not entry:
        return 0
    return len(entry["active_alerts"])


# =============================================================================
# Typed output — what every Part submits when confidence is high enough
# =============================================================================


class ServiceReport(BaseModel):
    service: str = Field(description="Name of the service reviewed.")
    summary: str = Field(description="2-3 sentence summary of the service's reliability posture.")
    active_alerts: list[str] = Field(description="All currently firing alerts.")
    last_deploy: str = Field(description="Date of the service's last deploy.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the report (0–1).")


@tool
def submit_review(report: ServiceReport) -> str:
    """Submit the completed reliability review. Call when confidence ≥ 0.85.

    Args:
        report: The completed ``ServiceReport``.
    """
    return f"submitted: {report.service} ({report.confidence:.0%} confidence)"


# =============================================================================
# Part 1 — minimal create_deepagent
# =============================================================================


async def part1_basic() -> None:
    """Reflexion + grounding on, typed termination, nothing else."""
    print("\n--- Part 1: basic create_deepagent ---")

    agent = create_deepagent(
        model=get_model(),
        tools=[list_services, inspect_service, count_active_alerts, submit_review],
        system_prompt=(
            "You are a site-reliability engineer running a pre-release readiness review. "
            "Use list_services, inspect_service, and count_active_alerts to gather facts "
            "about services inside the deploy window only. "
            "Submit a complete ServiceReport via submit_review once you reach ≥ 0.85 confidence."
        ),
        output_schema=ServiceReport,
        submit_tool="submit_review",
        min_confidence=0.85,
        max_iterations=12,
    )

    result = await agent.arun("Review the reliability posture of api-gateway.")
    print("protocol terminated:", result.stop_reason)
    if result.parsed:
        rpt: ServiceReport = result.parsed  # type: ignore[assignment]
        print(f"service:   {rpt.service}")
        print(f"alerts:    {', '.join(rpt.active_alerts[:4])} …")
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
        tools=[list_services, inspect_service, count_active_alerts, submit_review],
        system_prompt=(
            "You are a site-reliability engineer running a pre-release readiness review. "
            "Use write_file to keep scratchpad notes as you review each service. "
            "Use write_todos to track which services you've covered. "
            "Submit when you have a complete report with ≥ 0.85 confidence."
        ),
        output_schema=ServiceReport,
        submit_tool="submit_review",
        min_confidence=0.85,
        max_iterations=16,
        enable_filesystem=True,
        enable_todos=True,
        todo_state=todo_state,
    )

    result = await agent.arun("Review all three services in the deploy window.")
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
    alert_analyst = SubAgentDef(
        name="alert_analyst",
        description="Deep-dives on a single service's firing alerts.",
        system_prompt="Inspect the given service and return a plain list of its active alerts.",
        tools=[inspect_service],
        max_iterations=4,
    )

    agent = create_deepagent(
        model=get_model(),
        tools=[list_services, submit_review],
        system_prompt=(
            "Use list_services to discover services in the deploy window, then delegate "
            "alert analysis to the alert_analyst subagent via the task() tool. "
            "Submit a ServiceReport for api-gateway once you have the alert list."
        ),
        output_schema=ServiceReport,
        submit_tool="submit_review",
        min_confidence=0.8,
        max_iterations=12,
        subagents=[alert_analyst],
    )

    result = await agent.arun("Review api-gateway using the alert_analyst subagent.")
    print("terminated:", result.stop_reason)
    if result.parsed:
        rpt: ServiceReport = result.parsed  # type: ignore[assignment]
        print(f"alerts from subagent: {rpt.active_alerts}")


# =============================================================================
# Part 4 — observe deepagent.* events on the SSE bus
# =============================================================================


async def part4_observability() -> None:
    """Subscribe to deepagent.* events: subagent.*, fs.*, todo.*."""
    print("\n--- Part 4: deepagent.* SSE events ---")

    todo_state = TodoState()
    alert_analyst = SubAgentDef(
        name="alert_analyst",
        description="Inspect one service.",
        system_prompt="Inspect the given service and list its active alerts.",
        tools=[inspect_service],
        max_iterations=4,
    )

    agent = create_deepagent(
        model=get_model(),
        tools=[list_services, submit_review],
        system_prompt=(
            "Use list_services, delegate alert analysis via task(), "
            "write scratchpad notes, track progress with todos. "
            "Submit a report for payments-worker."
        ),
        output_schema=ServiceReport,
        submit_tool="submit_review",
        min_confidence=0.8,
        max_iterations=14,
        enable_filesystem=True,
        enable_todos=True,
        todo_state=todo_state,
        subagents=[alert_analyst],
    )

    deepagent_events: list[str] = []

    async def _collect(rid: str) -> None:
        async for ev in get_event_bus().subscribe(rid):
            if ev.event_type.startswith("deepagent."):
                deepagent_events.append(ev.event_type)

    async with run_context() as rid:
        collector = asyncio.create_task(_collect(rid))
        result = await agent.arun("Review the payments-worker service.")
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
    incident notes instead of guessing.

    This Part requires an embedding key (``OPENAI_API_KEY``). Without it,
    Part 5 exits cleanly and the earlier parts still run.
    """
    import os

    required = ("OPENAI_API_KEY",)
    missing = [n for n in required if not os.environ.get(n)]
    if missing:
        print("\n[incident_notes_datastore] skipped — missing env vars:")
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
            "api-gateway: p99 latency regressed after the 2026-05 connection-pool change; "
            "fixed by raising the upstream keep-alive limit, alert auto-resolved 2026-06-01.",
            "payments-worker: queue-depth alert fired during the 2026-04 ledger backfill; "
            "scaling consumers to 8 drained it; idempotency kept retries safe.",
            "web-frontend: blue/green rollout has had zero rollbacks since the 2026-03 "
            "CDN cache-key fix.",
            "billing-cron was decommissioned 2026-01 — any new alert from it is a finding.",
        ]
    )

    agent = create_deepagent(
        model=get_model(),
        tools=[],
        system_prompt=(
            "You are a site-reliability engineer. When asked about a service's "
            "incident history, call search_incident_notes first, then answer briefly "
            "with (doc-NN) citations."
        ),
        datastores={
            "incident_notes": {
                "retriever": retriever,
                "description": "prior incident notes: alert history, remediations, "
                "decommissioned services",
                "top_k": 3,
            }
        },
        reflexion=False,
        grounding=False,
        max_iterations=4,
    )

    result = await agent.arun(
        "What do prior incident notes say about api-gateway? Cite the retrieved doc."
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
