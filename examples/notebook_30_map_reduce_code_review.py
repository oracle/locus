#!/usr/bin/env python3
# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Notebook 31: map-reduce code review — scatter-gather with the Send primitive.

Three source files, three reviewer roles (security, performance, style)
= nine reviewer agents running in parallel, then one synthesizer
collapses everything into a single report.

- ``Send(node, payload, metadata)`` is a first-class graph primitive.
  The splitter node returns a list of Sends; the executor fans them out
  and runs them concurrently. No queues, no manual ``asyncio.gather``.
- Each reviewer is a distinct ``Agent`` with its own role-specific
  system prompt — the graph orchestrates them, not a hand-rolled loop.
- The synthesizer reads each Send's output back from merged state and
  emits a single Markdown report.
- The whole pipeline is one ``StateGraph.execute`` call. Streaming,
  cancellation, checkpointing, and GSAR judgment attach for free.

Run it:
    .venv/bin/python examples/notebook_36_map_reduce_code_review.py

The default provider is the bundled mock model. Set TULIP_MODEL_PROVIDER=openai
(or anthropic) and the matching credentials to use a live model. Set
``TULIP_MODEL_PROVIDER=mock`` for offline runs.

Prerequisites:
- Notebook 17 (basic graph).
- Notebook 25 (Swarm) for the dynamic-claim counterpoint.
"""

from __future__ import annotations

import asyncio
from typing import Any

from config import get_model

from tulip.agent import Agent, AgentConfig
from tulip.core.send import Send
from tulip.multiagent.graph import END, START, StateGraph


# ----------------------------------------------------------------------------
# Three independent files to review in parallel
# ----------------------------------------------------------------------------

SAMPLE_FILES = {
    "auth.py": """
def login(username, password):
    if password == "admin123":
        return {"role": "admin", "token": username}
    return {"role": "user", "token": username}
""",
    "billing.py": """
def calculate_total(items, tax_rate):
    total = 0
    for item in items:
        total = total + item.price
    return total * (1 + tax_rate)
""",
    "search.py": """
def search(query, db):
    sql = f"SELECT * FROM products WHERE name LIKE '%{query}%'"
    return db.execute(sql).fetchall()
""",
}


# ----------------------------------------------------------------------------
# Reviewer roles — each runs as its own Agent with a role-specific prompt
# ----------------------------------------------------------------------------

REVIEWER_ROLES = {
    "security": (
        "You are a senior application-security reviewer. Read the code and "
        "list every concrete security issue you can prove with the snippet "
        "(SQL injection, hardcoded credentials, missing auth, unvalidated "
        "input, etc.). Be specific. Do not invent issues. End with a single "
        "line: SEVERITY=<low|medium|high|critical>."
    ),
    "performance": (
        "You are a performance-engineering reviewer. List concrete inefficiencies "
        "in the code (N+1 patterns, accidental quadratic loops, redundant work, "
        "missing batching, blocking I/O, etc.). End with: SEVERITY=<...>."
    ),
    "style": (
        "You are a Python style/idiom reviewer focused on readability, naming, "
        "and modern Python idioms. Be terse. End with: SEVERITY=<...>."
    ),
}


def _make_reviewer(role: str, model: Any) -> Agent:
    return Agent(
        config=AgentConfig(
            agent_id=f"reviewer-{role}",
            model=model,
            system_prompt=REVIEWER_ROLES[role],
            # One model call is enough for static review of a short snippet.
            max_iterations=2,
            max_tokens=400,
        )
    )


# ----------------------------------------------------------------------------
# Graph nodes
# ----------------------------------------------------------------------------


async def split_files(state: dict[str, Any]) -> list[Send]:
    """Fan out: one Send per (file, role) — 3 × 3 = 9 reviewers, all concurrent."""
    files: dict[str, str] = state["files"]
    roles = list(REVIEWER_ROLES)
    return [
        Send(
            node="review_one",
            payload={"file_name": fname, "code": code, "role": role},
            metadata={"file": fname, "role": role},
        )
        for fname in files
        for code, _ in [(files[fname], None)]
        for role in roles
    ]


async def review_one(state: dict[str, Any]) -> dict[str, Any]:
    """One reviewer Agent against one (file, role).

    Uses ``async for event in agent.run(...)`` instead of ``run_sync()``
    so the 9 instances run truly in parallel inside the graph's
    ``asyncio.gather`` — ``run_sync`` would serialise them on a shared
    thread-pool worker.
    """
    from tulip.core.events import TerminateEvent

    role: str = state["role"]
    file_name: str = state["file_name"]
    code: str = state["code"]
    model = state["__model__"]
    agent = _make_reviewer(role, model)
    prompt = (
        f"File: {file_name}\n"
        f"Role: {role}\n\n"
        f"```python\n{code}\n```\n\n"
        f"Review the code as the {role} reviewer."
    )
    final_msg: str = ""
    iterations = 0
    async for event in agent.run(prompt):
        if isinstance(event, TerminateEvent):
            final_msg = event.final_message or ""
            iterations = event.iterations_used
    return {
        "review": {
            "file": file_name,
            "role": role,
            "comments": final_msg.strip(),
            "iterations": iterations,
        }
    }


async def synthesize(state: dict[str, Any]) -> dict[str, Any]:
    """Reduce: walk the merged state, collect every ``review`` payload, render."""
    reviews = [v["review"] for v in state.values() if isinstance(v, dict) and "review" in v]
    by_file: dict[str, list[dict[str, Any]]] = {}
    for r in reviews:
        by_file.setdefault(r["file"], []).append(r)

    lines = ["# Code review report", ""]
    for fname in sorted(by_file):
        lines.append(f"## {fname}")
        for r in sorted(by_file[fname], key=lambda x: x["role"]):
            lines.append(f"### {r['role']}")
            lines.append(r["comments"])
            lines.append("")
    return {"report": "\n".join(lines), "review_count": len(reviews)}


# ----------------------------------------------------------------------------
# Build the graph
# ----------------------------------------------------------------------------


def build_review_graph(model: Any) -> StateGraph:
    """Wire the three nodes: split → review_one (parallel) → synthesize → END.

    The model is threaded through state under ``__model__`` rather than
    captured by closure so the graph stays picklable for checkpointing.
    """
    graph = StateGraph(name="code-review-crew")
    graph.add_node("split", split_files)
    graph.add_node("review_one", review_one)
    graph.add_node("synthesize", synthesize)

    graph.add_edge(START, "split")
    # No explicit edge "split → review_one" — the Sends from ``split``
    # carry their own routing. Once every Send finishes, control returns
    # to ``split``'s adjacency, which points at ``synthesize``.
    graph.add_edge("split", "synthesize")
    graph.add_edge("synthesize", END)
    return graph


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------


async def main() -> None:
    print("Notebook 31: map-reduce code review — scatter-gather with Send")
    print("=" * 60)

    model = get_model()
    graph = build_review_graph(model)

    initial = {"files": SAMPLE_FILES, "__model__": model}

    print(
        f"\nFanning out {len(SAMPLE_FILES)} files × {len(REVIEWER_ROLES)} roles "
        f"= {len(SAMPLE_FILES) * len(REVIEWER_ROLES)} reviewer agents in parallel...\n"
    )

    result = await graph.execute(initial)

    print(
        f"Graph completed in {result.duration_ms:.0f} ms across "
        f"{result.iterations} graph iteration(s)"
    )
    print(f"Reviews collected: {result.final_state.get('review_count', 0)}")
    print()
    print(result.final_state.get("report", "(no report)"))


if __name__ == "__main__":
    asyncio.run(main())
