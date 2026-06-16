# GSAR Typed Grounding

GSAR (Grounded Structured Answer Reasoning) is the Locus layer from
[Federico A. Kamelhar (2026), arXiv:2604.23366](https://arxiv.org/abs/2604.23366).
It partitions an answer's claims into four buckets, scores them
against evidence, and decides whether to proceed, regenerate, or
replan.

- The four-way partition (grounded / ungrounded / contradicted /
  complementary) as a Pydantic type.
- Equation (2): the evidence-typed weighted grounding score `S`.
- Equation (3): the three-tier `{proceed, regenerate, replan}`
  decision with the Appendix-B reference thresholds
  (`τ_proceed=0.80`, `τ_regenerate=0.65`).
- Algorithm 1: a bounded outer loop with a `K_max` replan budget,
  driven by an LLM-as-judge and two side-effect callables.

## Run it

OCI GenAI is the default — the notebook harness picks it when `~/.oci/config` exists (set `OCI_PROFILE` to choose a non-default profile):

```bash
LOCUS_MODEL_ID=openai.gpt-4.1 python examples/notebook_37_gsar_typed_grounding.py
```

Offline:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_37_gsar_typed_grounding.py
```

## Prerequisites

- An OCI profile with GenAI access, or `LOCUS_MODEL_PROVIDER` set to
  `openai` / `anthropic` / `mock`.
- Part 4 (Algorithm 1) needs a model that supports constrained JSON
  decoding for the structured-output judge.

## Source

```python
--8<-- "examples/notebook_37_gsar_typed_grounding.py"
```
