# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL
"""Notebook 38: GSAR — typed grounding for hallucination detection.

GSAR (Grounded Structured Answer Reasoning) is the Tulip layer from
`Federico A. Tulip (2026), arXiv:2604.23366 <https://arxiv.org/abs/2604.23366>`_.
It partitions an answer's claims into four buckets, scores them
against evidence, and decides whether to proceed, regenerate, or
replan.

- The four-way partition — grounded / ungrounded / contradicted /
  complementary — as a Pydantic type.
- Equation (2): the evidence-typed weighted grounding score ``S``.
- Equation (3): the three-tier ``{proceed, regenerate, replan}``
  decision with the Appendix-B reference thresholds
  (``τ_proceed=0.80``, ``τ_regenerate=0.65``).
- Algorithm 1: a bounded outer loop with a ``K_max`` replan budget,
  driven by an LLM-as-judge and two side-effect callables.

Run it:
    # The bundled mock model is the default; set TULIP_MODEL_PROVIDER for a live provider.
    TULIP_MODEL_ID=openai.gpt-4.1 python examples/notebook_43_gsar_typed_grounding.py

    # Offline:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_43_gsar_typed_grounding.py

Prerequisites:
- An OpenAI or Anthropic API key, or set ``TULIP_MODEL_PROVIDER`` to
  ``openai`` / ``anthropic`` / ``mock``.
- Part 4 (Algorithm 1) needs a model that supports constrained JSON
  decoding for the structured-output judge.
"""

from __future__ import annotations

import asyncio
import time

from config import get_model

from tulip.agent import Agent
from tulip.reasoning.gsar import (
    DEFAULT_WEIGHT_MAP,
    Claim,
    Decision,
    EvidenceType,
    GSARThresholds,
    Partition,
    decide,
    gsar_score,
)


def _llm_call(
    prompt: str, *, system: str = "Reply in one short sentence.", max_tokens: int = 80
) -> str:
    agent = Agent(model=get_model(max_tokens=max_tokens), system_prompt=system)
    t0 = time.perf_counter()
    res = agent.run_sync(prompt)
    dt = time.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · {res.metrics.prompt_tokens}→{res.metrics.completion_tokens} tokens]"
    )
    return res.message.strip()


# =============================================================================
# Part 1: The four-way claim partition + the Appendix-B weight table.
# =============================================================================


def example_partition_and_weights() -> None:
    print("=== Part 1: Partition + Appendix-B weights ===\n")
    print(
        f"AI rationale: {_llm_call('In one sentence, why does GSAR partition claims into grounded/ungrounded/contradicted/complementary?')}"
    )

    partition = Partition(
        grounded=[
            Claim(text="CPU at 97% on db-prod-1", type=EvidenceType.TOOL_MATCH),
            Claim(text="Request rate dropped to 12 RPS", type=EvidenceType.SPECIFIC_DATA),
        ],
        ungrounded=[
            Claim(text="A runaway query is the cause", type=EvidenceType.INFERENCE),
        ],
        complementary=[
            Claim(
                text="Region-wide network event also plausible",
                type=EvidenceType.COMPLEMENTARY_FINDING,
            ),
        ],
        contradicted=[
            Claim(text="The saturation was transient", type=EvidenceType.INFERENCE),
        ],
    )
    print(
        f"Partition: |G|={len(partition.grounded)}, "
        f"|U|={len(partition.ungrounded)}, "
        f"|X|={len(partition.contradicted)}, "
        f"|K|={len(partition.complementary)}, "
        f"total={partition.total_claims}"
    )
    print()
    print("Reference weights (Appendix B):")
    for etype, weight in sorted(DEFAULT_WEIGHT_MAP.items(), key=lambda kv: -kv[1]):
        print(f"  {etype.value:24s} {weight:.2f}")


# =============================================================================
# Part 2: Score (Equation 2) and decision (Equation 3) — Appendix-E worked
#         example reproduced end to end.
# =============================================================================


def example_score_and_decision() -> None:
    print("\n=== Part 2: Score and decision ===\n")
    print(
        f"AI rationale: {_llm_call('In one sentence, what does the GSAR S-score (Eq. 2) measure?')}"
    )

    partition = Partition(
        grounded=[
            Claim(text="c1", type=EvidenceType.TOOL_MATCH),
            Claim(text="c2", type=EvidenceType.SPECIFIC_DATA),
        ],
        ungrounded=[Claim(text="c3", type=EvidenceType.INFERENCE)],
        complementary=[Claim(text="c4", type=EvidenceType.COMPLEMENTARY_FINDING)],
        contradicted=[Claim(text="c5", type=EvidenceType.INFERENCE)],
    )
    s = gsar_score(partition, contradiction_penalty=0.5)
    d = decide(s)

    print(f"S = {s:.4f}  (paper Appendix E: ≈0.757)")
    print(f"δ(S) = {d.value}  (paper Appendix E under reference thresholds)")
    print()
    print("Score breakdown:")
    print(f"  W(G) + W(K) = numerator = 1.00 + 0.95 + 0.85 = 2.80")
    print(f"  W(U) + ρ·W(X) = 0.60 + 0.5·0.60 = 0.90")
    print(f"  S = 2.80 / (2.80 + 0.90) = 2.80 / 3.70 = {2.80 / 3.70:.4f}")


# =============================================================================
# Part 3: Threshold sensitivity — how the decision shifts when you
#         re-calibrate τ_proceed / τ_regenerate for production.
# =============================================================================


def example_threshold_sensitivity() -> None:
    print("\n=== Part 3: Threshold sensitivity ===\n")
    print(
        f"AI rationale: {_llm_call('In one sentence, why might production tighten GSAR thresholds vs research defaults?')}"
    )

    base = Partition(
        grounded=[Claim(text="g", type=EvidenceType.TOOL_MATCH)],
        ungrounded=[Claim(text="u", type=EvidenceType.INFERENCE)],
    )
    s = gsar_score(base)
    print(f"Score: {s:.4f}\n")

    profiles = {
        "default (0.80 / 0.65)": GSARThresholds(),
        "lenient (0.70 / 0.50)": GSARThresholds(proceed=0.70, regenerate=0.50),
        "strict (0.95 / 0.85)": GSARThresholds(proceed=0.95, regenerate=0.85),
    }
    for name, th in profiles.items():
        print(f"  {name:30s} → δ = {decide(s, thresholds=th).value}")


# =============================================================================
# Part 4: Algorithm 1 outer loop — bounded replan budget, LLM judge,
#         regenerate / replan callables wired in.
# =============================================================================


async def example_outer_loop() -> None:
    print("\n=== Part 4: Algorithm-1 outer loop ===\n")

    from tulip.reasoning.gsar_evaluator import GSAREvaluator
    from tulip.reasoning.gsar_judge import JudgeOutput, StructuredOutputGSARJudge

    judge = StructuredOutputGSARJudge(model=get_model(max_tokens=2048))

    report = (
        "CPU utilisation on db-prod-1 reached 97% at 14:02 UTC. "
        "The request rate dropped to 12 RPS at the same time. "
        "Both observations are consistent with the alert that fired."
    )
    evidence = (
        "[tool=query_metrics row=14:02:01] host=db-prod-1 cpu_pct=97.2\n"
        "[tool=query_metrics row=14:02:01] host=db-prod-1 rps=12.4\n"
        "[signal] alert_id=A-9912 fired_at=14:02:00 metric=cpu_pct severity=high\n"
    )

    # Stub callbacks — never exercised because the test report is fully grounded.
    async def regen(syn: str, jo: JudgeOutput) -> str:  # pragma: no cover
        return syn

    async def replan(syn: str, ev: str, jo: JudgeOutput) -> tuple[str, str]:
        return syn, ev

    evaluator = GSAREvaluator(judge=judge, regenerate_fn=regen, replan_fn=replan)
    result = await evaluator.evaluate(report_synthesis=report, evidence_corpus=evidence)

    print(f"final_decision: {result.final_decision.value}")
    print(f"final_score:    {result.final_score:.4f}")
    print(f"replans_used:   {result.replans_used}")
    print(f"degraded:       {result.degraded}")
    print()
    print("Trajectory:")
    for entry in result.trajectory:
        print(f"  iter={entry.iteration}  score={entry.score:.4f}  decision={entry.decision.value}")


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    example_partition_and_weights()
    example_score_and_decision()
    example_threshold_sensitivity()
    asyncio.run(example_outer_loop())
