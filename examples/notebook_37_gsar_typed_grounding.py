# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Notebook 37: GSAR grounded findings — a finding ships, or the agent abstains.

This is the file that earns the SDK its claim: a security finding an
agent emits is only as trustworthy as the evidence behind every
sentence in it. An ungrounded vulnerability claim is a false positive
*by construction* — so Tulip will not let you produce one.

The primitive is :func:`tulip.security.ground_finding`. You hand it a
candidate finding plus a GSAR partition of its claims; it scores the
partition, and either returns a typed :class:`~tulip.security.Finding`
(the finding ships) or an :class:`~tulip.security.Abstention` (the
finding is withheld, with an audit record of why). There is no public
path that constructs a ``Finding`` without clearing the grounding bar.

GSAR — Grounding-Stratified Adaptive Replanning (``tulip.reasoning.gsar``,
from `arXiv:2604.23366 (2026) <https://arxiv.org/abs/2604.23366>`_) — is
the scoring layer underneath. Every claim is partitioned into grounded /
ungrounded / contradicted / complementary buckets and weighted by
provenance: a scanner row (``tool_match``) outranks a typed step output
(``specific_data``), which outranks a model-internal inference, which
outranks a domain prior. That ordering is the same discipline a SOC lead
applies to a junior analyst's write-up — "show me the log line" — encoded
as a typed, auditable function.

Key ideas:
- ``ground_finding(...)`` returns ``Finding | Abstention``. A finding
  backed by scanner rows ships; an "it's probably exploitable" hunch with
  no evidence abstains. ``is_finding(result)`` narrows the union.
- The four-way partition (Eq. 1) and the evidence-typed grounding score
  ``S`` (Eq. 2): tool output beats inference beats domain knowledge, so
  the evidence hierarchy is explicit, not vibes.
- The three-tier ``{proceed, regenerate, replan}`` decision (Eq. 3) with
  the Appendix-B reference thresholds (``τ_proceed=0.80``,
  ``τ_regenerate=0.65``). Below proceed, no finding is emitted.
- Algorithm 1: a bounded outer loop with a ``K_max`` replan budget,
  driven by an LLM-as-judge — "go back and get me proof, twice, then
  escalate", as code.

Findings here carry MITRE ATLAS (``AML.Txxxx``) / OWASP LLM
(``LLM01``–``LLM10``) tags so they drop into a SIEM or compliance report
without a translation layer.

Run it:
    # The bundled mock model is the default; set TULIP_MODEL_PROVIDER for a live provider.
    TULIP_MODEL_ID=openai.gpt-4.1 python examples/notebook_37_gsar_typed_grounding.py

    # Offline:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_37_gsar_typed_grounding.py

Prerequisites:
- An OpenAI or Anthropic API key, or set ``TULIP_MODEL_PROVIDER`` to
  ``openai`` / ``anthropic`` / ``mock``.
- Part 5 (Algorithm 1) needs a model that supports constrained JSON
  decoding for the structured-output judge; under mock it abstains
  deterministically, which is itself a correct GSAR outcome.

Provider notes:
- ``ground_finding`` and the GSAR scoring layer are pure Python: the
  admit/abstain decisions in Parts 1–4 are deterministic and identical
  across providers. Only the judge in Part 5 calls a model.
"""

from __future__ import annotations

import asyncio
import time

from config import get_model

from tulip.agent import Agent
from tulip.reasoning.gsar import (
    DEFAULT_WEIGHT_MAP,
    Claim,
    EvidenceType,
    GSARThresholds,
    Partition,
    decide,
    gsar_score,
)
from tulip.security import (
    AtlasTechnique,
    Finding,
    Indicator,
    IndicatorType,
    OwaspLLM,
    Severity,
    ground_finding,
    is_finding,
)


def _llm_call(
    prompt: str, *, system: str = "Reply in one short sentence.", max_tokens: int = 80
) -> str:
    """Fire one model call and print a timing/token banner."""
    agent = Agent(model=get_model(max_tokens=max_tokens), system_prompt=system)
    t0 = time.perf_counter()
    res = agent.run_sync(prompt)
    dt = time.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · "
        f"{res.metrics.prompt_tokens}→{res.metrics.completion_tokens} tokens]"
    )
    return res.message.strip()


def _report(result) -> None:
    """Print the outcome of a ``ground_finding`` call — Finding or Abstention."""
    if is_finding(result):
        print(f"  SHIPPED  Finding(S={result.gsar_score:.4f}, severity={result.severity.value})")
        print(f"           {result.title}")
        if result.taxonomy:
            print(f"           taxonomy: {', '.join(t.value for t in result.taxonomy)}")
        print(f"           evidence_refs: {len(result.evidence_refs)} ref(s)")
    else:
        print(f"  WITHHELD Abstention(S={result.gsar_score:.4f}, δ={result.decision.value})")
        print(f"           candidate: {result.candidate_title}")
        print(f"           reason: {result.reason}")


# =============================================================================
# Part 1: A grounded finding ships.
#         Every sentence traces to a scanner row, a TLS-handshake field, or
#         the originating signal. One model-internal aside lands in U but does
#         not sink the score. ground_finding() returns a typed Finding.
# =============================================================================


def example_grounded_ships() -> None:
    print("=== Part 1: A grounded finding ships ===\n")
    print(
        "AI rationale: "
        + _llm_call(
            "In one sentence, why should a security finding be withheld unless every "
            "claim in it traces back to evidence?"
        )
    )
    print()

    # Each claim carries its provenance type and the opaque evidence ref it
    # was lifted from — the audit trail a SOC reviews after the fact.
    partition = Partition(
        grounded=[
            Claim(
                text="TLS certificate on 192.0.2.10:443 expired 2026-05-30",
                type=EvidenceType.TOOL_MATCH,
                evidence_refs=["tool:tls_scan:host=192.0.2.10:not_after=2026-05-30"],
            ),
            Claim(
                text="Endpoint negotiates TLS 1.0 with a deprecated cipher suite",
                type=EvidenceType.TOOL_MATCH,
                evidence_refs=["tool:tls_scan:host=192.0.2.10:proto=TLSv1.0"],
            ),
            Claim(
                text="Finding F-2209 fired on the cert_expired check at 14:02",
                type=EvidenceType.SIGNAL_MATCH,
                evidence_refs=["signal:F-2209:check=cert_expired:fired_at=14:02:00"],
            ),
        ],
        complementary=[
            Claim(
                text="The same certificate chain is also served on 198.51.100.7",
                type=EvidenceType.COMPLEMENTARY_FINDING,
                evidence_refs=["tool:tls_scan:host=198.51.100.7:cert_fp=match"],
            ),
        ],
        ungrounded=[
            # A reasonable guess with no evidence behind it — GSAR keeps it in
            # U so it can't masquerade as fact, but one weak claim among strong
            # ones doesn't drag the score under the bar.
            Claim(
                text="The renewal ticket was probably ignored by operators",
                type=EvidenceType.INFERENCE,
            ),
        ],
    )

    result = ground_finding(
        title="Expired TLS certificate and TLS 1.0 negotiation on 192.0.2.10:443",
        description=(
            "The serving endpoint presents a certificate that expired on "
            "2026-05-30 and negotiates TLS 1.0 with a deprecated cipher suite. "
            "The cert_expired detection (F-2209) fired against it."
        ),
        severity=Severity.HIGH,
        asset="192.0.2.10:443",
        remediation="Rotate the certificate, disable TLS 1.0, and enforce automated renewal.",
        partition=partition,
        indicators=[Indicator(type=IndicatorType.ENDPOINT, value="192.0.2.10:443")],
        taxonomy=[OwaspLLM.MISINFORMATION],  # an ungrounded report would BE LLM09
    )
    _report(result)

    if is_finding(result):
        # W(G) = 1.00 + 1.00 + 0.90 = 2.90 ; W(K) = 0.85 ; W(U) = 0.60.
        # S = (W(G) + W(K)) / (W(G) + W(U) + W(K)) = 3.75 / 4.35 ≈ 0.8621 ≥ τ_proceed.
        print()
        print("  Score arithmetic (Eq. 2, Appendix-B weights):")
        print("    W(G) = 1.00 + 1.00 + 0.90 = 2.90   W(K) = 0.85   W(U) = 0.60")
        print(f"    S = (2.90 + 0.85) / (2.90 + 0.60 + 0.85) = 3.75 / 4.35 = {3.75 / 4.35:.4f}")


# =============================================================================
# Part 2: An ungrounded finding never ships — it abstains.
#         A suspected indirect prompt injection (LLM01 / AML.T0051) backed only
#         by a hunch and a textbook prior. No tool row, no signal field. GSAR
#         scores it at the floor and ground_finding() returns an Abstention.
#         This is the thesis: an ungrounded finding is a false positive by
#         construction, and the API will not hand you one.
# =============================================================================


def example_ungrounded_abstains() -> None:
    print("\n=== Part 2: An ungrounded finding abstains ===\n")
    print(
        "AI rationale: "
        + _llm_call(
            "In one sentence, why is 'the agent was probably prompt-injected' a false "
            "positive until you can point to the injected content in a tool output?"
        )
    )
    print()

    partition = Partition(
        # Nothing grounded: the analyst suspects an indirect prompt injection
        # in a retrieved document but has not produced the offending span.
        ungrounded=[
            Claim(
                text="The retrieval agent was prompt-injected via a poisoned document",
                type=EvidenceType.INFERENCE,
            ),
            Claim(
                text="Indirect injections commonly hide in retrieved web content",
                type=EvidenceType.DOMAIN,
            ),
        ],
    )

    result = ground_finding(
        title="Suspected indirect prompt injection in the retrieval pipeline",
        description=(
            "The analyst suspects the retrieval agent followed an instruction "
            "planted in a retrieved document, but the injected span has not been "
            "located in any tool output."
        ),
        severity=Severity.HIGH,
        asset="augur-rag-index",
        remediation="Locate the injected span; quarantine the source document.",
        partition=partition,
        # LLM01 Prompt Injection / AML.T0051 — the technique this WOULD be if grounded.
        taxonomy=[OwaspLLM.PROMPT_INJECTION, AtlasTechnique.PROMPT_INJECTION],
    )
    _report(result)
    print()
    print("  The candidate is kept as an audit record, not discarded: a SOC can")
    print("  review what the agent declined to assert and re-open it with evidence.")


# =============================================================================
# Part 3: Evidence that refutes a claim is worse than evidence that's missing.
#         A contradicted claim pulls the score down via the ρ penalty and is
#         called out explicitly in the abstention reason — so a finding that
#         the evidence actively disputes is never quietly shipped.
# =============================================================================


def example_contradiction_withholds() -> None:
    print("\n=== Part 3: Contradicted evidence withholds the finding ===\n")
    print(
        "AI rationale: "
        + _llm_call(
            "In one sentence, why must a finding be withheld when one of its claims is "
            "directly contradicted by the evidence, even if other claims hold?"
        )
    )
    print()

    partition = Partition(
        grounded=[
            Claim(
                text="Host 192.0.2.20 exposes an admin port to the internet",
                type=EvidenceType.TOOL_MATCH,
                evidence_refs=["tool:portscan:host=192.0.2.20:8443=open"],
            ),
        ],
        contradicted=[
            # The analyst asserted the service is unpatched; the inventory row
            # shows the current build. The evidence refutes the claim.
            Claim(
                text="The admin service is running an end-of-life build",
                type=EvidenceType.SPECIFIC_DATA,
            ),
        ],
    )

    result = ground_finding(
        title="Internet-exposed admin port on 192.0.2.20",
        description="An admin port is reachable from the internet; the build claim is disputed.",
        severity=Severity.MEDIUM,
        asset="192.0.2.20:8443",
        remediation="Restrict the admin port to management networks; re-verify the build.",
        partition=partition,
    )
    _report(result)
    print()
    s = gsar_score(partition)
    print(f"  S = {s:.4f}: the ρ-weighted contradiction (Eq. 2) drops it below τ_proceed.")
    print("  Re-investigate the disputed claim before this finding can ship.")


# =============================================================================
# Part 4: Threshold sensitivity — a SOC re-calibrates the bar for its risk
#         appetite. Auto-filed tickets warrant a stricter τ_proceed than a
#         human-in-the-loop queue. Same finding, different proceed/withhold.
# =============================================================================


def example_threshold_recalibration() -> None:
    print("\n=== Part 4: Re-calibrating the proceed bar ===\n")
    print(
        "AI rationale: "
        + _llm_call(
            "In one sentence, why would a SOC that auto-files tickets raise the GSAR "
            "proceed threshold above the research default?"
        )
    )
    print()

    # A borderline finding: two grounded scanner observations, one ungrounded
    # inference. S lands between the lenient and default proceed bars, so the
    # same finding ships for a human queue but is held back from auto-filing.
    partition = Partition(
        grounded=[
            Claim(
                text="Port 443 is open on 192.0.2.30",
                type=EvidenceType.TOOL_MATCH,
                evidence_refs=["tool:portscan:host=192.0.2.30:443=open"],
            ),
            Claim(
                text="The endpoint returns an HTTP 200 with a login form",
                type=EvidenceType.SPECIFIC_DATA,
                evidence_refs=["tool:http_probe:host=192.0.2.30:443:status=200"],
            ),
        ],
        ungrounded=[
            Claim(text="The service behind it is likely vulnerable", type=EvidenceType.INFERENCE),
        ],
    )
    s = gsar_score(partition)
    print(f"  Finding score S = {s:.4f}\n")

    print("  Reference evidence weights (Appendix B) — the hierarchy, made explicit:")
    for etype, weight in sorted(DEFAULT_WEIGHT_MAP.items(), key=lambda kv: -kv[1]):
        print(f"    {etype.value:24s} {weight:.2f}")
    print()

    profiles = {
        "human queue (0.70 / 0.50)": GSARThresholds(proceed=0.70, regenerate=0.50),
        "research default (0.80 / 0.65)": GSARThresholds(),
        "auto-file (0.95 / 0.85)": GSARThresholds(proceed=0.95, regenerate=0.85),
    }
    print("  Same finding, different risk appetites:")
    for name, th in profiles.items():
        result = ground_finding(
            title="Open port on 192.0.2.30",
            description="Port 443 open; service vulnerability unconfirmed.",
            severity=Severity.LOW,
            asset="192.0.2.30:443",
            remediation="Fingerprint the service and confirm before escalating.",
            partition=partition,
            thresholds=th,
        )
        verdict = "ships" if is_finding(result) else f"withheld ({result.decision.value})"
        print(f"    {name:32s} δ = {decide(s, thresholds=th).value:10s} → {verdict}")


# =============================================================================
# Part 5: Algorithm-1 outer loop — bounded replan budget, an LLM judge, and
#         regenerate / replan callables. The judge partitions the report
#         against the evidence corpus; the loop proceeds only when the
#         synthesis clears the bar, else regenerates, replans up to K_max, or
#         returns degraded-but-honest. The grounded finding from Part 1, end
#         to end through the loop the framework is built around.
# =============================================================================


async def example_outer_loop() -> None:
    print("\n=== Part 5: Algorithm-1 outer loop (LLM judge) ===\n")

    from tulip.reasoning.gsar_evaluator import GSAREvaluator
    from tulip.reasoning.gsar_judge import JudgeOutput, StructuredOutputGSARJudge

    judge = StructuredOutputGSARJudge(model=get_model(max_tokens=2048))

    report = (
        "The TLS certificate on 192.0.2.10:443 expired on 2026-05-30. "
        "The endpoint negotiates TLS 1.0 with a deprecated cipher suite. "
        "Detection F-2209 fired on the cert_expired check at 14:02."
    )
    evidence = (
        "[tool=tls_scan] host=192.0.2.10 port=443 cert_expired=true not_after=2026-05-30\n"
        "[tool=tls_scan] host=192.0.2.10 port=443 proto=TLSv1.0 cipher=deprecated\n"
        "[signal] finding_id=F-2209 fired_at=14:02:00 check=cert_expired severity=high\n"
    )

    # Recovery hooks. On a live judge that proceeds on iteration 0 these are
    # never called; under mock the judge abstains, so replan_fn fires up to
    # K_max and the loop returns degraded-but-honest — itself a correct GSAR
    # outcome (it refuses to fabricate grounding it can't get).
    async def regenerate(synthesis: str, judge_output: JudgeOutput) -> str:
        return synthesis

    async def replan(synthesis: str, ev: str, judge_output: JudgeOutput) -> tuple[str, str]:
        return synthesis, ev

    evaluator = GSAREvaluator(judge=judge, regenerate_fn=regenerate, replan_fn=replan)
    result = await evaluator.evaluate(report_synthesis=report, evidence_corpus=evidence)

    print(f"  final_decision: {result.final_decision.value}")
    print(f"  final_score:    {result.final_score:.4f}")
    print(f"  replans_used:   {result.replans_used}")
    print(f"  degraded:       {result.degraded}")
    print()
    print("  Trajectory:")
    for entry in result.trajectory:
        print(
            f"    iter={entry.iteration}  score={entry.score:.4f}  "
            f"decision={entry.decision.value}"
        )

    # The loop never silently ships an ungrounded report: a degraded result is
    # flagged, not hidden. Convert a proceed into a typed Finding for the queue.
    print()
    if result.final_decision.value == "proceed":
        shipped = Finding(
            title="Expired TLS certificate and TLS 1.0 negotiation on 192.0.2.10:443",
            description=report,
            severity=Severity.HIGH,
            asset="192.0.2.10:443",
            remediation="Rotate the certificate; disable TLS 1.0.",
            gsar_score=result.final_score,
            evidence_refs=["tool:tls_scan:not_after=2026-05-30", "signal:F-2209"],
        )
        print(f"  Loop proceeded → Finding ready for the queue (S={shipped.gsar_score:.4f}).")
    else:
        print("  Loop did not proceed → no finding emitted; degraded result flagged for review.")


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    example_grounded_ships()
    example_ungrounded_abstains()
    example_contradiction_withholds()
    example_threshold_recalibration()
    asyncio.run(example_outer_loop())
