#!/usr/bin/env python3
# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Notebook 33: adversarial debate with a structured-output judge.

PRO and CON take turns arguing a resolution. After N rounds a Judge
reads the full transcript and emits a typed ``Verdict`` — winner,
confidence, key points, reasoning — that downstream systems (tickets,
audit logs, databases) can consume directly.

- Each turn is a ``Turn(side, round, text)`` Pydantic model. The
  transcript is a ``list[Turn]`` accumulated in graph state.
- The Judge uses Tulip's ``output_schema=Verdict``, so the result is a
  populated Pydantic object — no JSON parsing in the caller.
- If the configured model can't honor the schema, the judge node raises
  rather than fabricating a verdict from raw text.
- A ``check_structured_output_capable()`` guard at the top short-
  circuits the notebook when the model can't produce JSON (mock,
  Cohere R-series).

Run it:
    .venv/bin/python examples/notebook_38_debate_with_judge.py

The default provider is the bundled mock model. Set TULIP_MODEL_PROVIDER=openai
(or anthropic) and the matching credentials to use a live model that
supports constrained decoding (e.g. ``openai:gpt-4o``). Set
``TULIP_MODEL_PROVIDER=mock`` and the notebook exits cleanly with
setup instructions.

Prerequisites:
- Notebook 14 (structured output).
- Notebook 17 (basic graph).
"""

from __future__ import annotations

import asyncio
from typing import Any

from config import get_model
from pydantic import BaseModel, Field

from tulip.agent import Agent, AgentConfig
from tulip.core.events import TerminateEvent
from tulip.multiagent.graph import END, START, StateGraph


# ---------------------------------------------------------------------------
# Typed shapes — Turn for the transcript, Verdict for the final ruling
# ---------------------------------------------------------------------------


class Turn(BaseModel):
    """One turn of the debate."""

    side: str  # "pro" | "con"
    round: int
    text: str


class Verdict(BaseModel):
    """Judge's structured ruling."""

    winner: str = Field(description="'pro', 'con', or 'tie'")
    confidence: float = Field(ge=0.0, le=1.0, description="0..1 confidence in the call")
    key_points: list[str] = Field(description="The 2–4 strongest arguments that drove the decision")
    reasoning: str = Field(description="One-paragraph rationale")


PRO_PROMPT = (
    "You are arguing the FOR side. Be specific, cite reasoning, and "
    "directly rebut the OPPOSITION's most recent point if any. Three "
    "sentences max."
)
CON_PROMPT = (
    "You are arguing the AGAINST side. Be specific, cite reasoning, and "
    "directly rebut the FOR side's most recent point if any. Three "
    "sentences max."
)
JUDGE_PROMPT = (
    "You are an impartial debate judge. Read the full transcript and "
    "emit a Verdict object. Pick a winner only if one side clearly "
    "outargued the other; otherwise return 'tie'."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(role: str, prompt: str, model: Any) -> Agent:
    return Agent(
        config=AgentConfig(
            agent_id=f"debate-{role}",
            model=model,
            system_prompt=prompt,
            max_iterations=2,
            # Reasoning-class models (gpt-5.x, o-series) burn thinking
            # tokens before any output; 2000 leaves room for both the
            # thinking budget and a short debater turn.
            max_tokens=2000,
        )
    )


async def _argue(agent: Agent, transcript: list[Turn], topic: str, side: str, rnd: int) -> str:
    history = "\n".join(f"[{t.side.upper()} r{t.round}] {t.text}" for t in transcript)
    prompt = (
        f"Topic: {topic}\n\n"
        f"Transcript so far:\n{history or '(no turns yet)'}\n\n"
        f"You are arguing the {side.upper()} side, round {rnd}. Make your point now."
    )
    final = ""
    async for event in agent.run(prompt):
        if isinstance(event, TerminateEvent):
            final = event.final_message or ""
    return final.strip()


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


async def pro_turn(state: dict[str, Any]) -> dict[str, Any]:
    agent = _make_agent("pro", PRO_PROMPT, state["__model__"])
    rnd = state.get("round", 0)
    text = await _argue(agent, state.get("transcript", []), state["topic"], "pro", rnd)
    return {"transcript": state.get("transcript", []) + [Turn(side="pro", round=rnd, text=text)]}


async def con_turn(state: dict[str, Any]) -> dict[str, Any]:
    agent = _make_agent("con", CON_PROMPT, state["__model__"])
    rnd = state.get("round", 0)
    text = await _argue(agent, state.get("transcript", []), state["topic"], "con", rnd)
    return {
        "transcript": state.get("transcript", []) + [Turn(side="con", round=rnd, text=text)],
        "round": rnd + 1,
    }


async def judge_turn(state: dict[str, Any]) -> dict[str, Any]:
    """Judge with ``output_schema=Verdict`` — the result is a typed Pydantic object.

    If the configured model can't honor the JSON schema we raise rather
    than fabricating a verdict from free text.
    """
    import asyncio as _asyncio

    agent = Agent(
        config=AgentConfig(
            agent_id="judge",
            model=state["__model__"],
            system_prompt=JUDGE_PROMPT,
            output_schema=Verdict,
            max_iterations=2,
            # 4000 tokens covers both the reasoning-model thinking
            # budget and the JSON verdict payload.
            max_tokens=4000,
        )
    )
    transcript_text = "\n".join(
        f"[{t.side.upper()} r{t.round}] {t.text}" for t in state["transcript"]
    )
    prompt = f"Topic: {state['topic']}\n\nTranscript:\n{transcript_text}\n\nNow emit your Verdict."
    # run_sync returns the parsed object. We're already inside an asyncio
    # loop driving the graph, so hop the call onto a worker thread.
    last_exc: BaseException | None = None
    final = None
    for attempt in range(3):
        try:
            final = await _asyncio.to_thread(agent.run_sync, prompt)
            break
        # Retry covers transient provider flakiness.
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            await _asyncio.sleep(0.5 * (attempt + 1))
    if final is None:
        raise RuntimeError(f"Judge failed after 3 attempts. Last error: {last_exc!r}") from last_exc
    if final.parsed is None:
        raise RuntimeError(
            "Judge returned no parsed Verdict. The configured model could not "
            "honor the JSON schema. Use a stronger model (e.g. openai.gpt-4o, "
            "openai.gpt-5, anthropic.claude-3-5-sonnet) for notebook 32. "
            f"Raw output: {final.message!r}"
        )
    return {"verdict": final.parsed}


# ---------------------------------------------------------------------------
# Routing — N rounds of pro/con, then judge once
# ---------------------------------------------------------------------------


N_ROUNDS = 2


def route_after_con(state: dict[str, Any]) -> str:
    if state.get("round", 0) >= N_ROUNDS:
        return "judge"
    return "pro"


def build_debate_graph() -> StateGraph:
    graph = StateGraph(name="debate-with-judge")
    graph.add_node("pro", pro_turn)
    graph.add_node("con", con_turn)
    graph.add_node("judge", judge_turn)
    graph.add_edge(START, "pro")
    graph.add_edge("pro", "con")
    graph.add_conditional_edges("con", route_after_con, targets={"pro": "pro", "judge": "judge"})
    graph.add_edge("judge", END)
    return graph


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def main() -> None:
    from config import check_structured_output_capable

    # Short-circuits the notebook when the model can't produce JSON.
    check_structured_output_capable()
    print("Notebook 33: adversarial debate with a structured-output judge")
    print("=" * 60)

    model = get_model()
    graph = build_debate_graph()
    initial = {
        "topic": (
            "Resolved: A 30-engineer SaaS team running a 250k-LOC Python "
            "monolith on a single Postgres + Redis stack should split the "
            "monolith into microservices over the next 12 months, given a "
            "current weekly deploy cadence and ~2 outages per quarter "
            "traceable to coupling between billing and provisioning code."
        ),
        "transcript": [],
        "round": 0,
        "__model__": model,
    }

    print(f"\nTopic: {initial['topic']!r}\n")
    print(f"Running {N_ROUNDS} rounds of PRO vs CON, then judge…\n")

    result = await graph.execute(initial)
    failed = [
        (nid, nr.error) for nid, nr in result.node_results.items() if nr.status.value == "failed"
    ]
    if failed:
        for nid, err in failed:
            print(f"\n  ✗ node {nid} FAILED: {err}")
        raise RuntimeError(f"graph had {len(failed)} failed node(s); see above")
    transcript: list[Turn] = result.final_state.get("transcript", [])
    verdict: Verdict = result.final_state["verdict"]

    print(f"Total turns: {len(transcript)}")
    print()
    for t in transcript:
        print(f"  [{t.side.upper()} r{t.round}] {t.text}")

    print()
    print("Verdict:")
    print("-" * 60)
    print(f"  Winner:     {verdict.winner}")
    print(f"  Confidence: {verdict.confidence:.2f}")
    print("  Key points:")
    for p in verdict.key_points:
        print(f"    - {p}")
    print(f"  Reasoning:  {verdict.reasoning}")


if __name__ == "__main__":
    asyncio.run(main())
