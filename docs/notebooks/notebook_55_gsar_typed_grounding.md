# GSAR typed grounding

GSAR (Grounding-Stratified Adaptive Replanning) is the typed-grounding
layer of the SDK, described on the [GSAR — typed grounding](../concepts/gsar.md)
concept page and in [arXiv:2604.23366 (2026)](https://arxiv.org/abs/2604.23366).
It partitions an answer's claims into four buckets (grounded, ungrounded,
contradicted and complementary), weights every claim by the type of evidence
behind it, and compares the resulting score `S` with two thresholds to decide
whether to proceed, regenerate or replan. In Parts 1 to 4 of this notebook the
caller supplies the partition; in Part 5 a large language model (LLM) acting
as judge produces it.

The notebook's headline primitive, `ground_finding`, applies that decision to
a candidate security finding: it either returns a typed `Evidence` (the
finding ships) or an `Abstention` (the finding is withheld, with an audit
record of why).

What this notebook covers:

- `ground_finding(...)`: hand it a candidate finding plus a GSAR partition of
  its claims; it returns a typed `Evidence` (the finding ships) or an
  `Abstention` (the finding is withheld, with an audit record of the decision,
  the score, the would-be title and the reason). `ground_finding` returns
  `Evidence` only on a `proceed` decision; `Evidence` is still an ordinary
  Pydantic model, so code that constructs one directly skips this check.
  `is_finding(result)` narrows the union.
- The four-way partition (grounded / ungrounded / contradicted /
  complementary) as a Pydantic type.
- Equation (2): the evidence-typed weighted grounding score `S`, a single
  score for how well the answer is backed by evidence.
- Equation (3): the three-tier `{proceed, regenerate, replan}` decision (ship
  the answer, rewrite it or rework the plan) with the Appendix-B reference
  thresholds (`τ_proceed=0.80`, `τ_regenerate=0.65`).
- Algorithm 1: a bounded outer loop with a `K_max` replan budget, driven by an
  LLM judge and two recovery callables, `regenerate_fn` and `replan_fn`.
  `decide()` never returns `abstain`; that signal comes from the judge. The
  loop treats a judge `abstain` like `replan` (it calls `replan_fn`) but
  records it as `abstain` in the trajectory, so an observer can tell "the
  judge could not decide" from "this needs a fresh investigation". This is
  separate from the `Abstention` record `ground_finding` returns, whose
  decision is always `regenerate` or `replan`.

## Prerequisites

- An OCI profile in `~/.oci/config` (picked up automatically), or
  `LOCUS_MODEL_PROVIDER` set to `openai` or `anthropic` with the matching
  key, or `mock` for an offline run.
- Part 5 (Algorithm 1) needs a model whose reply parses into the judge's
  `JudgeOutput` schema. `StructuredOutputGSARJudge` requests it through
  `response_format` constrained decoding and falls back to its safe default,
  which abstains, when the reply does not parse.

The drawing follows the loop that `example_outer_loop` runs through
`GSAREvaluator.evaluate`. The Transport Layer Security (TLS) report and its
evidence corpus (two scanner rows and a fired detection) go to the judge,
which partitions the claims into the four buckets; the partition is scored
and `S` is compared with the two thresholds. `proceed` ends the loop with the
final report, and `regenerate` rewrites the synthesis from the same evidence
and returns to the judge. `replan`, or a judge that abstains, goes to the
budget check: while fewer than `K_max` replans have been spent, `replan_fn`
is asked for a new synthesis and evidence corpus; otherwise the loop ends
with the result flagged degraded. Parts 1 to 4 call the score and decide
steps in the middle of the drawing directly, through `ground_finding`,
`gsar_score` and `decide`:

{{ locus_diagram gsar-decision }}

## Key lines

`example_grounded_ships` builds the `Partition` the gate scores: three
grounded `Claim`s whose `evidence_refs` trace to TLS scan rows and a fired
detection, one complementary observation and one ungrounded inference. It
hands the partition to `ground_finding` with the finding's title, severity,
asset, remediation and
[OWASP LLM](../concepts/threat-scenarios.md#owasp-llm-top-10-2025) tag; the
call returns an `Evidence` because `S` clears `τ_proceed`, and `is_finding`
narrows the union before the score arithmetic is printed:

{{ locus_source examples/notebook_55_gsar_typed_grounding.py::example_grounded_ships }}

`example_threshold_recalibration` scores one borderline partition with
`gsar_score`, lists the Appendix-B weights in `DEFAULT_WEIGHT_MAP` that rank a
tool match above an inference, and then calls `ground_finding` three times
with different `GSARThresholds`. The same finding ships for a human queue, is
withheld for regeneration at the research default and withheld for replanning
at the auto-file bar, with `decide` showing the three-tier decision behind
each verdict:

{{ locus_source examples/notebook_55_gsar_typed_grounding.py::example_threshold_recalibration }}

`example_outer_loop` is Algorithm 1: a `StructuredOutputGSARJudge` over the
model `examples/config.py` selects, two recovery callables for `regenerate_fn`
and `replan_fn` (identity functions here, so the loop's own control flow is
what runs), and one `GSAREvaluator.evaluate` call over the TLS report and its
evidence corpus. The result carries the final decision and score, the replans
used, the `degraded` flag and the per-iteration trajectory, and only a
`proceed` is converted into an `Evidence` for the queue:

{{ locus_source examples/notebook_55_gsar_typed_grounding.py::example_outer_loop }}

## Run it

```bash
python examples/notebook_55_gsar_typed_grounding.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one. `ground_finding` and
the scoring layer are pure Python, so the admit and withhold decisions in
Parts 1 to 4 are deterministic and identical across providers. Only the judge
in Part 5 calls a model. The mock model answers in prose rather than the JSON
the judge requests, so the judge falls back to its safe default (`abstain`,
`S = 0.5`) on every round; the loop spends its replan budget and returns a
degraded result rather than an ungrounded finding, which is the contract the
evaluator exists to keep.

To run offline regardless of what `~/.oci/config` holds:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_55_gsar_typed_grounding.py
```

## Output

Running it offline — no credentials, bundled mock model — prints the admit/withhold trail: a grounded TLS finding shipping with its score arithmetic, an ungrounded and a contradicted candidate withheld with their reasons, the Appendix-B evidence-weight table, the same finding under three threshold profiles, and the Algorithm-1 loop abstaining through its replan budget to a degraded result:

```text
=== Part 1: A grounded finding ships ===

  SHIPPED  Evidence(S=0.8621, severity=high)
           Expired TLS certificate and TLS 1.0 negotiation on 192.0.2.10:443
           taxonomy: LLM09
           evidence_refs: 4 ref(s)

  Score arithmetic (Eq. 2, Appendix-B weights):
    W(G) = 1.00 + 1.00 + 0.90 = 2.90   W(K) = 0.85   W(U) = 0.60
    S = (2.90 + 0.85) / (2.90 + 0.60 + 0.85) = 3.75 / 4.35 = 0.8621

=== Part 2: An ungrounded finding abstains ===

  WITHHELD Abstention(S=0.0000, δ=replan)
           candidate: Suspected indirect prompt injection in the retrieval pipeline
           reason: withheld (replan): grounding below the proceed threshold (2 ungrounded of 2 claims)

  The candidate is kept as an audit record, not discarded: a SOC can
  review what the agent declined to assert and re-open it with evidence.

=== Part 3: Contradicted evidence withholds the finding ===

  WITHHELD Abstention(S=0.6780, δ=regenerate)
           candidate: Internet-exposed admin port on 192.0.2.20
           reason: withheld (regenerate): 1 claim(s) contradicted by evidence

  S = 0.6780: the ρ-weighted contradiction (Eq. 2) drops it below τ_proceed.
  Re-investigate the disputed claim before this finding can ship.

=== Part 4: Re-calibrating the proceed bar ===

  Evidence score S = 0.7647

  Reference evidence weights (Appendix B) — the hierarchy, made explicit:
    tool_match               1.00
    specific_data            0.95
    signal_match             0.90
    complementary_finding    0.85
    synthesis                0.80
    neg_evidence             0.70
    inference                0.60
    domain                   0.60

  Same finding, different risk appetites:
    human queue (0.70 / 0.50)        δ = proceed    → ships
    research default (0.80 / 0.65)   δ = regenerate → withheld (regenerate)
    auto-file (0.95 / 0.85)          δ = replan     → withheld (replan)

=== Part 5: Algorithm-1 outer loop (LLM judge) ===

  final_decision: abstain
  final_score:    0.5000
  replans_used:   2
  degraded:       True

  Trajectory:
    iter=0  score=0.5000  decision=abstain
    iter=1  score=0.5000  decision=abstain
    iter=2  score=0.5000  decision=abstain

  Loop did not proceed → no finding emitted; degraded result flagged for review.
```
<!-- notebook-output:end -->

## See also

- [GSAR — typed grounding](../concepts/gsar.md) — the score, the decision
  function and the evidence taxonomy behind this notebook.

## Source

??? note "Full source — `examples/notebook_55_gsar_typed_grounding.py`"

    ````python
    --8<-- "examples/notebook_55_gsar_typed_grounding.py"
    ````
