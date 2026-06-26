#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 32: incident adjudication debate — real incident vs noise.

An INCIDENT advocate and a NOISE advocate take turns arguing over a
cloud-monitoring alert — here a possible production reliability incident
(a p99 latency spike paired with autoscaling churn). After N rounds a
Judge reads the full transcript and emits a typed ``Verdict`` — call,
confidence, key points, reasoning — that downstream systems (incident
tooling, audit logs, on-call paging) can consume directly. Two-advocate
adjudication is a cheap way to surface the disconfirming evidence a
single-pass triage agent tends to skip.

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
    .venv/bin/python examples/notebook_32_debate_with_judge.py

The default provider is the bundled mock model. Set TULIP_MODEL_PROVIDER=openai
(or anthropic) and the matching credentials to use a live model that
supports constrained decoding (e.g. ``openai:gpt-4o``). Set
``TULIP_MODEL_PROVIDER=mock`` and the notebook exits cleanly with
setup instructions.

Prerequisites:
- Notebook 35 (structured output).
- Notebook 16 (basic graph).
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
    """One turn of the adjudication."""

    side: str  # "incident" | "noise"
    round: int
    text: str


class Verdict(BaseModel):
    """Judge's structured ruling on the alert."""

    call: str = Field(description="'incident', 'noise', or 'inconclusive'")
    confidence: float = Field(ge=0.0, le=1.0, description="0..1 confidence in the call")
    key_points: list[str] = Field(description="The 2–4 strongest arguments that drove the decision")
    reasoning: str = Field(description="One-paragraph rationale")


INCIDENT_PROMPT = (
    "You are the INCIDENT advocate in a cloud-reliability adjudication. "
    "Argue that the alert is a real production incident worth paging "
    "on-call. Cite the alert telemetry, and directly rebut the noise "
    "side's most recent point if any. Three sentences max."
)
NOISE_PROMPT = (
    "You are the NOISE advocate in a cloud-reliability adjudication. "
    "Argue that the alert has a benign explanation and does not warrant "
    "paging. Cite the alert telemetry, and directly rebut the incident "
    "side's most recent point if any. Three sentences max."
)
JUDGE_PROMPT = (
    "You are an impartial SRE on-call lead. Read the full adjudication "
    "transcript and emit a Verdict object. Call incident or noise only "
    "if one side clearly outargued the other; otherwise return "
    "'inconclusive'."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(role: str, prompt: str, model: Any) -> Agent:
    return Agent(
        config=AgentConfig(
            agent_id=f"adjudicate-{role}",
            model=model,
            system_prompt=prompt,
            max_iterations=2,
            # Reasoning-class models (gpt-5.x, o-series) burn thinking
            # tokens before any output; 2000 leaves room for both the
            # thinking budget and a short advocate turn.
            max_tokens=2000,
        )
    )


async def _argue(agent: Agent, transcript: list[Turn], alert: str, side: str, rnd: int) -> str:
    history = "\n".join(f"[{t.side.upper()} r{t.round}] {t.text}" for t in transcript)
    prompt = (
        f"Alert under adjudication: {alert}\n\n"
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


async def incident_turn(state: dict[str, Any]) -> dict[str, Any]:
    agent = _make_agent("incident", INCIDENT_PROMPT, state["__model__"])
    rnd = state.get("round", 0)
    text = await _argue(agent, state.get("transcript", []), state["alert"], "incident", rnd)
    return {
        "transcript": state.get("transcript", []) + [Turn(side="incident", round=rnd, text=text)]
    }


async def noise_turn(state: dict[str, Any]) -> dict[str, Any]:
    agent = _make_agent("noise", NOISE_PROMPT, state["__model__"])
    rnd = state.get("round", 0)
    text = await _argue(agent, state.get("transcript", []), state["alert"], "noise", rnd)
    return {
        "transcript": state.get("transcript", []) + [Turn(side="noise", round=rnd, text=text)],
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
    prompt = f"Alert: {state['alert']}\n\nTranscript:\n{transcript_text}\n\nNow emit your Verdict."
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
# Routing — N rounds of incident/noise, then judge once
# ---------------------------------------------------------------------------


N_ROUNDS = 2


def route_after_noise(state: dict[str, Any]) -> str:
    if state.get("round", 0) >= N_ROUNDS:
        return "judge"
    return "incident"


def build_debate_graph() -> StateGraph:
    graph = StateGraph(name="incident-adjudication")
    graph.add_node("incident", incident_turn)
    graph.add_node("noise", noise_turn)
    graph.add_node("judge", judge_turn)
    graph.add_edge(START, "incident")
    graph.add_edge("incident", "noise")
    graph.add_conditional_edges(
        "noise", route_after_noise, targets={"incident": "incident", "judge": "judge"}
    )
    graph.add_edge("judge", END)
    return graph


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def main() -> None:
    from config import check_structured_output_capable

    # Short-circuits the notebook when the model can't produce JSON.
    check_structured_output_capable()
    print("Notebook 32: incident adjudication debate — incident vs noise with a typed judge")
    print("=" * 60)

    model = get_model()
    graph = build_debate_graph()
    initial = {
        "alert": (
            "Alert CLOUD-4189 (suspected reliability incident): service "
            "checkout-api p99 latency rose from 120ms to 2.4s over 8 minutes; "
            "the HPA scaled the deployment from 6 to 20 pods; one node in the "
            "autoscaling group reported MemoryPressure. The error rate held "
            "flat at 0.1%, a scheduled nightly reindex job is running on the "
            "same cluster, and no recent deploy preceded the spike."
        ),
        "transcript": [],
        "round": 0,
        "__model__": model,
    }

    print(f"\nAlert: {initial['alert']!r}\n")
    print(f"Running {N_ROUNDS} rounds of incident vs noise, then judge…\n")

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
    print(f"  Call:       {verdict.call}")
    print(f"  Confidence: {verdict.confidence:.2f}")
    print("  Key points:")
    for p in verdict.key_points:
        print(f"    - {p}")
    print(f"  Reasoning:  {verdict.reasoning}")


if __name__ == "__main__":
    asyncio.run(main())
