# Debate with Judge

An INCIDENT advocate and a NOISE advocate take turns arguing over a
cloud-monitoring alert. After N rounds a Judge reads the full transcript
and emits a typed `Verdict` — call, confidence, key points, reasoning —
that downstream systems (incident tooling, audit logs, on-call paging)
can consume directly.

This notebook covers:

- `Turn(side, round, text)` accumulated into a `list[Turn]` in graph
  state — the transcript.
- `output_schema=Verdict` on the judge Agent, so `result.parsed` is a
  populated Pydantic object, not a JSON string.
- The judge node raises rather than fabricating a verdict if the
  configured model can't honor the schema.
- `check_structured_output_capable()` short-circuits the notebook with
  setup guidance when running under the mock model or a model without
  constrained-decoding support.

```text
incident r0 → noise r0 → incident r1 → noise r1 → ... → judge → END
```

The judge's `Verdict.call` lands on `incident`, `noise`, or
`inconclusive` — it only picks a side when one advocate clearly
outargued the other, which keeps a two-agent debate from manufacturing
false certainty about an ambiguous alert.

## Prerequisites

- Structured output.
- Basic graph.

## Run

```bash
python examples/notebook_26_debate_with_judge.py
```

This notebook needs a model that supports constrained JSON decoding. Without an OCI profile or `LOCUS_MODEL_PROVIDER`, the harness falls back to the bundled mock model and the notebook exits cleanly with setup instructions.

## Source

````python
--8<-- "examples/notebook_26_debate_with_judge.py"
````
