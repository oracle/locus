# GSAR Typed Grounding

GSAR (typed grounding) is the Locus layer from
[arXiv:2604.23366 (2026)](https://arxiv.org/abs/2604.23366).
It partitions an answer's claims into four buckets, scores them
against evidence, and decides whether to proceed, regenerate, or
replan. The notebook's headline primitive, `ground_finding`, uses
that decision to either ship a grounded finding or abstain with an
audit record of why.

- `ground_finding(...)`: hand it a candidate finding plus a GSAR
  partition of its claims; it returns a typed `Evidence` (the finding
  ships) or an `Abstention` (the finding is withheld, with an audit
  record of the decision, the score, the would-be title, and why).
  `is_finding(result)` narrows the union. It returns `Evidence` only
  on a `proceed` decision; `Evidence` is still an ordinary Pydantic
  model, so code that constructs one directly skips this check.
- The four-way partition (grounded / ungrounded / contradicted /
  complementary) as a Pydantic type.
- Equation (2): the evidence-typed weighted grounding score `S` — a
  single score for how well the answer is backed by evidence.
- Equation (3): the three-tier `{proceed, regenerate, replan}`
  decision — whether to ship the answer, retry it, or rework the
  plan — with the Appendix-B reference thresholds
  (`τ_proceed=0.80`, `τ_regenerate=0.65`). `decide()` never returns
  `abstain`: that signal comes from the LLM judge. The Algorithm-1
  loop treats a judge `abstain` like `replan` (it calls `replan_fn`)
  but records it as `abstain` in the trajectory, so an observer can
  tell "the judge couldn't decide" from "this needs a fresh
  investigation". This is separate from the `Abstention` record
  `ground_finding` returns, whose decision is always `regenerate` or
  `replan`.
- Algorithm 1: a bounded outer loop with a `K_max` replan budget,
  driven by an LLM-as-judge and two side-effect callables.

## Run it

The bundled mock model is the default; set `LOCUS_MODEL_PROVIDER` for a live provider:

```bash
LOCUS_MODEL_PROVIDER=openai LOCUS_MODEL_ID=gpt-4o python examples/notebook_55_gsar_typed_grounding.py
```

Offline:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_55_gsar_typed_grounding.py
```

## Prerequisites

- An OCI profile in `~/.oci/config` (picked up automatically), or
  `LOCUS_MODEL_PROVIDER` set to `openai` / `anthropic` / `mock` with the
  matching key.
- Part 5 (Algorithm 1) needs a model that supports constrained JSON
  decoding for the structured-output judge.

## Source

````python
--8<-- "examples/notebook_55_gsar_typed_grounding.py"
````
