#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 30: secure code review at scale — scatter-gather with Send.

Three source files, three security lenses (injection, authz, secrets)
= nine reviewer agents running in parallel, then one synthesizer
collapses everything into a single findings report. Each lens maps to
a CWE class — injection (CWE-89 / CWE-78), broken access control
(CWE-862 / CWE-639), and hardcoded credentials (CWE-798) — so findings
land in vocabulary a SAST pipeline or triage queue already speaks.

- ``Send(node, payload, metadata)`` is a first-class graph primitive.
  The splitter node returns a list of Sends; the executor fans them out
  and runs them concurrently. No queues, no manual ``asyncio.gather``.
- Each reviewer is a distinct ``Agent`` with its own lens-specific
  system prompt — the graph orchestrates them, not a hand-rolled loop.
- The synthesizer reads each Send's output back from merged state and
  emits a single Markdown findings report.
- The whole pipeline is one ``StateGraph.execute`` call. Streaming,
  cancellation, checkpointing, and GSAR judgment attach for free.

Run it:
    .venv/bin/python examples/notebook_30_map_reduce_code_review.py

The default provider is the bundled mock model. Set TULIP_MODEL_PROVIDER=openai
(or anthropic) and the matching credentials to use a live model. Set
``TULIP_MODEL_PROVIDER=mock`` for offline runs.

Prerequisites:
- Notebook 16 (basic graph).
- Notebook 24 (Swarm) for the dynamic-claim counterpoint.
"""

from __future__ import annotations

import asyncio
from typing import Any

from config import get_model

from tulip.agent import Agent, AgentConfig
from tulip.core.send import Send
from tulip.multiagent.graph import END, START, StateGraph


# ----------------------------------------------------------------------------
# Three independent files to review in parallel. Each snippet carries a
# deliberately planted, provable weakness so reviewers have something
# concrete to cite — no real secrets: the AWS key below is the documented
# example value from AWS docs, not a live credential.
#   auth.py   — hardcoded credential + role from untrusted input (CWE-798)
#   export.py — missing ownership check + hardcoded API key (CWE-862, CWE-798)
#   search.py — SQL injection via string interpolation (CWE-89)
# ----------------------------------------------------------------------------

SAMPLE_FILES = {
    "auth.py": """
def login(username, password):
    if password == "admin123":
        return {"role": "admin", "token": username}
    return {"role": "user", "token": username}
""",
    "export.py": """
def export_report(user, report_id, db):
    # TODO: check report ownership
    report = db.get_report(report_id)
    return report.render(api_key="AKIAIOSFODNN7EXAMPLE")
""",
    "search.py": """
def search(query, db):
    sql = f"SELECT * FROM products WHERE name LIKE '%{query}%'"
    return db.execute(sql).fetchall()
""",
}


# ----------------------------------------------------------------------------
# Review lenses — each runs as its own Agent with a lens-specific prompt
# ----------------------------------------------------------------------------

REVIEWER_ROLES = {
    "injection": (
        "You are an application-security reviewer hunting injection bugs "
        "(CWE-89 SQL injection, CWE-78 OS command injection). "
        "Read the code and list every injection issue you can prove with "
        "the snippet — unsanitized string interpolation into a query or a "
        "shell command, untrusted input reaching an interpreter. Be "
        "specific and quote the offending line. Do not invent issues you "
        "cannot point to in the code. End with a single "
        "line: SEVERITY=<low|medium|high|critical>."
    ),
    "authz": (
        "You are a broken-access-control reviewer (CWE-862 missing "
        "authorization, CWE-639 IDOR). List every authorization "
        "issue you can prove in the code: missing ownership checks, role "
        "decisions from untrusted input, insecure direct object references, "
        "privilege-escalation paths. Quote the offending line. Do not "
        "invent issues. End with: SEVERITY=<...>."
    ),
    "secrets": (
        "You are a secrets-hygiene reviewer (CWE-798 hardcoded credentials) "
        "focused on API keys, tokens, and passwords embedded in source. "
        "Quote the offending literal. Be terse. End with: SEVERITY=<...>."
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
    """Fan out: one Send per (file, lens) — 3 × 3 = 9 reviewers, all concurrent."""
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
    """One reviewer Agent against one (file, lens).

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
        f"Lens: {role}\n\n"
        f"```python\n{code}\n```\n\n"
        f"Review the code through the {role} lens."
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

    lines = ["# Secure code review — findings report", ""]
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
    graph = StateGraph(name="secure-review-crew")
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
    print("Notebook 30: secure code review at scale — scatter-gather with Send")
    print("=" * 60)

    model = get_model()
    graph = build_review_graph(model)

    initial = {"files": SAMPLE_FILES, "__model__": model}

    print(
        f"\nFanning out {len(SAMPLE_FILES)} files × {len(REVIEWER_ROLES)} lenses "
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
