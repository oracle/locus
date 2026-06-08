# Debate with Judge

PRO and CON take turns arguing a resolution. After N rounds a Judge
reads the full transcript and emits a typed `Verdict` — winner,
confidence, key points, reasoning — that downstream systems (tickets,
audit logs, databases) can consume directly.

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
PRO r0 → CON r0 → PRO r1 → CON r1 → ... → judge → END
```

## Prerequisites

- Notebook 14 (structured output).
- Notebook 17 (basic graph).

## Run

```bash
python examples/notebook_32_debate_with_judge.py
```

The default provider is the bundled mock model. For this notebook pick a
provider that supports constrained JSON decoding (e.g. `openai:gpt-4o`).
Under `TULIP_MODEL_PROVIDER=mock` the notebook exits cleanly with setup
instructions.

## Source

```python
--8<-- "examples/notebook_32_debate_with_judge.py"
```
