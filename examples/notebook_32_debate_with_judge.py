#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 32: alert adjudication debate — true positive vs false positive.

A TRUE-POSITIVE advocate and a FALSE-POSITIVE advocate take turns
arguing over a SOC alert — here a possible DNS-based command-and-control
beacon (MITRE ATT&CK T1071.004 Application Layer Protocol: DNS). After N
rounds a Judge reads the full transcript and emits a typed ``Verdict`` —
call, confidence, key points, reasoning — that downstream systems (SOAR
cases, audit logs, ticketing) can consume directly. Two-advocate
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

    side: str  # "tp" | "fp"
    round: int
    text: str


class Verdict(BaseModel):
    """Judge's structured ruling on the alert."""

    call: str = Field(description="'true_positive', 'false_positive', or 'inconclusive'")
    confidence: float = Field(ge=0.0, le=1.0, description="0..1 confidence in the call")
    key_points: list[str] = Field(description="The 2–4 strongest arguments that drove the decision")
    reasoning: str = Field(description="One-paragraph rationale")


TP_PROMPT = (
    "You are the TRUE-POSITIVE advocate in a SOC adjudication. Argue that "
    "the alert is real and worth escalating. Cite the alert telemetry, and "
    "directly rebut the false-positive side's most recent point if any. "
    "Three sentences max."
)
FP_PROMPT = (
    "You are the FALSE-POSITIVE advocate in a SOC adjudication. Argue that "
    "the alert has a benign explanation. Cite the alert telemetry, and "
    "directly rebut the true-positive side's most recent point if any. "
    "Three sentences max."
)
JUDGE_PROMPT = (
    "You are an impartial SOC shift lead. Read the full adjudication "
    "transcript and emit a Verdict object. Call true_positive or "
    "false_positive only if one side clearly outargued the other; "
    "otherwise return 'inconclusive'."
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


async def tp_turn(state: dict[str, Any]) -> dict[str, Any]:
    agent = _make_agent("tp", TP_PROMPT, state["__model__"])
    rnd = state.get("round", 0)
    text = await _argue(agent, state.get("transcript", []), state["alert"], "tp", rnd)
    return {"transcript": state.get("transcript", []) + [Turn(side="tp", round=rnd, text=text)]}


async def fp_turn(state: dict[str, Any]) -> dict[str, Any]:
    agent = _make_agent("fp", FP_PROMPT, state["__model__"])
    rnd = state.get("round", 0)
    text = await _argue(agent, state.get("transcript", []), state["alert"], "fp", rnd)
    return {
        "transcript": state.get("transcript", []) + [Turn(side="fp", round=rnd, text=text)],
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
# Routing — N rounds of tp/fp, then judge once
# ---------------------------------------------------------------------------


N_ROUNDS = 2


def route_after_fp(state: dict[str, Any]) -> str:
    if state.get("round", 0) >= N_ROUNDS:
        return "judge"
    return "tp"


def build_debate_graph() -> StateGraph:
    graph = StateGraph(name="alert-adjudication")
    graph.add_node("tp", tp_turn)
    graph.add_node("fp", fp_turn)
    graph.add_node("judge", judge_turn)
    graph.add_edge(START, "tp")
    graph.add_edge("tp", "fp")
    graph.add_conditional_edges("fp", route_after_fp, targets={"tp": "tp", "judge": "judge"})
    graph.add_edge("judge", END)
    return graph


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def main() -> None:
    from config import check_structured_output_capable

    # Short-circuits the notebook when the model can't produce JSON.
    check_structured_output_capable()
    print("Notebook 32: alert adjudication debate — TP vs FP with a typed judge")
    print("=" * 60)

    model = get_model()
    graph = build_debate_graph()
    initial = {
        "alert": (
            "Alert SOC-4189 (suspected DNS C2, ATT&CK T1071.004): workstation "
            "WS-0231 issued 412 DNS queries to subdomains of "
            "updates.evil.example over 35 minutes at fixed 5-second intervals. "
            "The domain was registered 12 days ago; the proxy shows no related "
            "HTTP traffic; the user was logged in and an inventory-sync agent "
            "is also installed on this host."
        ),
        "transcript": [],
        "round": 0,
        "__model__": model,
    }

    print(f"\nAlert: {initial['alert']!r}\n")
    print(f"Running {N_ROUNDS} rounds of TP vs FP, then judge…\n")

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
