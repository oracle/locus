#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 32: supervisor + critic — drafting with a refinement loop.

A researcher gathers notes, a writer drafts a response, a critic either
approves or sends it back for revision. The loop caps at two revisions
to bound runtime.

- The control flow is a ``StateGraph`` with conditional edges — no
  hand-rolled ``while True`` plus message passing.
- Each role is its own ``Agent`` with a role-specific system prompt.
  No agent can see the others' internal state; they communicate only
  through graph state keys (``notes``, ``draft``, ``revision_request``).
- ``stream(mode=StreamMode.NODES)`` emits one event per node
  completion, so a UI can show "Researcher done / Writer working /
  Critic rejected — revising…" with no extra code.
- ``execute(...)`` returns the authoritative final state plus a
  ``GraphResult`` with timing and iteration metrics.

```text
START → research → write → critique → END (approve)
                     ↑         │
                     └── revise (cap: 2)
```

Run it:
    .venv/bin/python examples/notebook_37_supervisor_critic_loop.py

The default provider is the bundled mock model. Set TULIP_MODEL_PROVIDER=openai
(or anthropic) and the matching credentials to use a live model. Set
``TULIP_MODEL_PROVIDER=mock`` for offline runs.

Prerequisites:
- Notebook 17 (basic graph).
- Notebook 26 (agent handoff) for an alternative shape.
"""

from __future__ import annotations

import asyncio
from typing import Any

from config import get_model

from tulip.agent import Agent, AgentConfig
from tulip.core.events import TerminateEvent
from tulip.multiagent.graph import END, START, StateGraph


# ---------------------------------------------------------------------------
# Each role is a real Agent with a role-specific system prompt
# ---------------------------------------------------------------------------


def _make_agent(role: str, system_prompt: str, model: Any, max_iterations: int = 2) -> Agent:
    return Agent(
        config=AgentConfig(
            agent_id=f"agent-{role}",
            model=model,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            max_tokens=400,
        )
    )


SUPERVISOR_PROMPT = (
    "You are a project supervisor. Given the task and the current state, "
    "decide whether the Researcher, Writer, or Critic should run next. "
    "Respond with ONE word: research, write, or critique."
)

RESEARCHER_PROMPT = (
    "You are a research specialist. Given a topic, return 3–5 concise factual "
    "notes that a writer can use. No opinions. Bullet points only."
)

WRITER_PROMPT = (
    "You are a technical writer. Given research notes (and optionally a critic's "
    "revision request), produce a concise 1–2 paragraph response. Plain prose."
)

CRITIC_PROMPT = (
    "You are a strict editor. Read the draft and decide if it's publishable. "
    "If yes, respond with exactly: APPROVE. "
    "If not, respond with: REVISE: <one-line specific instruction>."
)


# ---------------------------------------------------------------------------
# Drive an Agent inside a graph node and return the final text
# ---------------------------------------------------------------------------


async def _run_agent(agent: Agent, prompt: str) -> str:
    final = ""
    async for event in agent.run(prompt):
        if isinstance(event, TerminateEvent):
            final = event.final_message or ""
    return final.strip()


# ---------------------------------------------------------------------------
# Graph nodes — one per role
# ---------------------------------------------------------------------------


async def research_node(state: dict[str, Any]) -> dict[str, Any]:
    agent = _make_agent("researcher", RESEARCHER_PROMPT, state["__model__"])
    notes = await _run_agent(agent, f"Topic: {state['topic']}")
    return {"notes": notes}


async def write_node(state: dict[str, Any]) -> dict[str, Any]:
    agent = _make_agent("writer", WRITER_PROMPT, state["__model__"])
    revision = state.get("revision_request", "")
    prompt = f"Topic: {state['topic']}\nResearch notes:\n{state.get('notes', '')}\n"
    if revision:
        prompt += f"\nCritic feedback (apply this): {revision}\n"
    prompt += "\nWrite the final response now."

    draft = await _run_agent(agent, prompt)
    revisions_done = state.get("revisions_done", 0) + (1 if revision else 0)
    return {"draft": draft, "revisions_done": revisions_done}


async def critique_node(state: dict[str, Any]) -> dict[str, Any]:
    agent = _make_agent("critic", CRITIC_PROMPT, state["__model__"])
    verdict = await _run_agent(agent, f"Draft to review:\n{state.get('draft', '')}")
    approved = verdict.strip().upper().startswith("APPROVE")
    revision_request = "" if approved else verdict
    return {
        "approved": approved,
        "revision_request": revision_request,
        "critic_verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Conditional routing — accept, or send back to the writer (capped)
# ---------------------------------------------------------------------------


def route_after_critique(state: dict[str, Any]) -> str:
    if state.get("approved"):
        return "done"
    if state.get("revisions_done", 0) >= 2:
        return "done"
    return "revise"


# ---------------------------------------------------------------------------
# Wire it: research → write → critique → (revise → write | done → END)
# ---------------------------------------------------------------------------


def build_supervisor_graph() -> StateGraph:
    graph = StateGraph(name="supervisor-critic-loop")
    graph.add_node("research", research_node)
    graph.add_node("write", write_node)
    graph.add_node("critique", critique_node)

    graph.add_edge(START, "research")
    graph.add_edge("research", "write")
    graph.add_edge("write", "critique")
    graph.add_conditional_edges(
        "critique",
        route_after_critique,
        targets={"revise": "write", "done": END},
    )
    return graph


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def main() -> None:
    print("Notebook 32: supervisor + critic — drafting with a refinement loop")
    print("=" * 60)

    model = get_model()
    graph = build_supervisor_graph()

    initial = {
        "topic": "Why structured logging beats plain prints in production",
        "__model__": model,
    }

    print(f"\nTopic: {initial['topic']!r}\n")

    # Stream node-completion events for live UI feedback, then call
    # execute() for the authoritative final state with metrics.
    from tulip.multiagent.graph import StreamMode

    async for event in graph.stream(initial, mode=StreamMode.NODES):
        if event.node_id:
            print(f"  ✓ {event.node_id}", flush=True)

    final = await graph.execute(initial)
    final_state = final.final_state

    print()
    print(f"Revisions:    {final_state.get('revisions_done', 0)}")
    verdict = final_state.get("critic_verdict") or "(unknown)"
    print(f"Critic:       {verdict[:80]}")
    print(f"Total tokens: ~{final.duration_ms:.0f} ms across {final.iterations} graph iterations")
    print()
    print("Final draft:")
    print("-" * 60)
    print(final_state.get("draft", "(no draft)"))


if __name__ == "__main__":
    asyncio.run(main())
