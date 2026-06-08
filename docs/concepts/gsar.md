# GSAR — typed grounding

Imagine an incident-response agent that pulls three log lines, two
metric points, and one alert. Every fact it cites is real. But the
synthesis says *"the outage was caused by a config push,"* and the
evidence only supports *"a config push happened thirty seconds before
the outage."* Vanilla `Agent(grounding=True)` — a single LLM-as-judge
scalar over the answer as a whole — often misses this. Each *claim* is
grounded; the *conclusion* over-reaches.

**GSAR** (Grounding-Stratified Adaptive Replanning, from
[Federico A. Tulip (2026), arXiv:2604.23366](https://arxiv.org/abs/2604.23366))
is the upgrade. It
breaks the synthesis into claims, partitions them four ways, scores
the partition with per-evidence-type weights, and picks one of three
responses: `proceed` if the synthesis holds up, `regenerate` if the
evidence is fine but the wording is loose (rewrites without re-running
tools), or `replan` if the evidence itself is missing or contradicted.
The math is small (one equation), the integration is one Pydantic
type, and six monotonicity / adversarial-robustness properties are
formally provable.

Use GSAR for high-stakes pipelines — operational incidents, regulated
diagnostics, anything where the "evidence fine, conclusion loose"
failure mode is a real cost. Use vanilla `grounding=True` for
everything else; binary verdicts are cheaper and good enough most
of the time.

## What it adds

| | Vanilla grounding | GSAR |
|---|---|---|
| Output | `is_grounded ∈ {true, false}` + scalar `s ∈ [0, 1]` | Four-way partition `G ⊔ U ⊔ X ⊔ K`, scalar `S`, abstain channel |
| Evidence weighting | Uniform | Per-type weights `w: T → [0, 1]` (tool_match weighted higher than inference) |
| Recovery | Binary `{stop, replan}` | Three-tier `{proceed, regenerate, replan}` — middle tier rewrites the synthesis without re-running expensive tools |
| Adversarial robustness | Score inflates if a contradicted claim is silently dropped | Asymmetric contradiction penalty `ρ` keeps `X` in the denominator |
| Budget | Implicit | Explicit `K_max` replan budget, degraded flag on exhaustion |

## The score

For a claim partition — `G` grounded, `U` ungrounded, `X` contradicted,
`K` complementary — an evidence-type weight map `w`, and a
contradiction penalty `ρ ∈ [0, 1]`, the GSAR score is:

$$
S = \frac{W(\mathcal G) + W(\mathcal K)}{W(\mathcal G) + W(\mathcal U) + \rho \cdot W(\mathcal X) + W(\mathcal K)}
$$

where `W(P) = Σ_{c ∈ P} w(type(c))`. On the empty partition `S = 0.5`
(epistemic indifference). The score lives in `[0, 1]`; six monotonicity
and adversarial-robustness properties are proven in Appendix A of the
paper and locked under unit tests in `tests/unit/test_gsar.py`.

## The decision

```
δ(s) = proceed     if s ≥ τ_proceed
δ(s) = regenerate  if τ_regenerate ≤ s < τ_proceed
δ(s) = replan      if s < τ_regenerate
```

The reference thresholds are `τ_proceed = 0.80`, `τ_regenerate = 0.65`
(Appendix B); the paper recommends per-deployment recalibration on a
small (100–200) human-graded held-out set. The `regenerate` tier is
the critical middle band — it rewrites the synthesis without
re-dispatching the specialists, catching the "evidence is fine,
synthesis is loose" mode that dominates real production logs.

## Wiring

```python
from tulip.models.native.openai import OpenAIModel
from tulip.reasoning.gsar import GSARThresholds
from tulip.reasoning.gsar_evaluator import GSAREvaluator
from tulip.reasoning.gsar_judge import JudgeOutput, StructuredOutputGSARJudge

judge = StructuredOutputGSARJudge(
    model=OpenAIModel(model="gpt-4o-mini", max_tokens=2048),
)

async def regenerate(synthesis: str, judge_output: JudgeOutput) -> str:
    """Cheap branch: rewrite synthesis from existing evidence."""
    ...

async def replan(synthesis: str, evidence: str, jo: JudgeOutput) -> tuple[str, str]:
    """Expensive branch: revise plan, re-dispatch specialists."""
    ...

evaluator = GSAREvaluator(
    judge=judge,
    regenerate_fn=regenerate,
    replan_fn=replan,
    thresholds=GSARThresholds(),     # Appendix-B defaults
    contradiction_penalty=0.5,        # ρ
    k_max=2,                          # bounded replan budget
)

result = await evaluator.evaluate(
    report_synthesis=initial_report,
    evidence_corpus=evidence,
)
# result.final_report, result.final_score, result.final_decision,
# result.trajectory  (every iteration logged for audit)
# result.degraded    (True when the budget exhausted without proceed)
```

The evaluator runs Algorithm 1 from the paper to convergence
(`δ = proceed`) or budget exhaustion (`degraded = True`, returning a
"degraded but honest" report rather than looping indefinitely or
silently shipping un-grounded claims).

## Evidence taxonomy

The default `EvidenceType` enum mirrors the paper's reference
instantiation (Appendix B). Tool-side annotations populate it; in
production you'd map your tool taxonomy onto these.

| Evidence type | When to use it | Default weight |
|---|---|---|
| `tool_match` | Claim directly traceable to a tool output row | 1.00 |
| `specific_data` | Cites a structured field of a step output | 0.95 |
| `signal_match` | References the originating alert / signal | 0.90 |
| `complementary_finding` | Non-redundant alternative perspective | 0.85 |
| `synthesis` | Cross-specialist combination | 0.80 |
| `neg_evidence` | Absence-of-signal observation | 0.70 |
| `inference` | Model-internal inference, no tool support | 0.60 |
| `domain` | Textbook / runbook fact | 0.60 |

## When to use GSAR vs vanilla `grounding=True`

- **Vanilla** is right for most tasks. Binary verdict, one scalar,
  cheap. If you're not in a regulated / safety-critical setting, start
  here.
- **GSAR** is right when (a) the cost of a wrong "ship it" decision
  outweighs the extra LLM judge call, (b) you want auditable per-claim
  evidence-type provenance in your checkpoint stream, or (c) your
  synthesis layer can plausibly be looser than the underlying evidence
  (the regenerate tier earns its keep here).

## Source and tests

- [`src/tulip/reasoning/gsar.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/src/tulip/reasoning/gsar.py)
  — Pydantic types, `gsar_score`, `decide`, defaults from Appendix B.
- [`src/tulip/reasoning/gsar_judge.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/src/tulip/reasoning/gsar_judge.py)
  — `BaseGSARJudge` Protocol, `JudgeOutput` schema (Appendix C),
  `StructuredOutputGSARJudge` reference implementation.
- [`src/tulip/reasoning/gsar_evaluator.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/src/tulip/reasoning/gsar_evaluator.py)
  — Algorithm-1 outer loop with `K_max` budget.
- [`tests/unit/test_gsar.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/tests/unit/test_gsar.py)
  — 54 tests verifying properties P1–P6 + Appendix-E worked example.
- [`tests/unit/test_gsar_judge.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/tests/unit/test_gsar_judge.py)
  — schema validation + structured-output fallback chain.
- [`tests/unit/test_gsar_evaluator.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/tests/unit/test_gsar_evaluator.py)
  — outer loop, abstain handling, budget exhaustion.
- [`tests/integration/test_gsar_live.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/tests/integration/test_gsar_live.py)
  — live LLM judge driving the full loop.
- [`examples/notebook_37_gsar_typed_grounding.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/examples/notebook_37_gsar_typed_grounding.py)
  — a runnable walkthrough of the four parts.
- Paper: [Federico A. Tulip (2026), arXiv:2604.23366](https://arxiv.org/abs/2604.23366).
