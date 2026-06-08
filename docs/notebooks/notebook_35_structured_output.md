# Structured Output

Get a Pydantic object back from a model call instead of a string you
have to re-parse. Every part below fires a real model call and prints
a `[model call: X.XXs · prompt→completion tokens]` banner.

- `extract_json` / `parse_structured` — pull JSON out of a model reply
  and validate it against a Pydantic schema (a typed model the LLM
  must produce JSON for).
- `create_schema_prompt` / `create_output_instructions` — emit the
  schema-aware system prompt the model needs to comply.
- `Agent(output_schema=YourModel)` — constrained decoding plus a
  prompted-JSON fallback; the parsed Pydantic object lands on
  `result.parsed`.
- `StructuredOutputError` for strict-mode failures.

## Run it

The bundled mock model is the default; set `TULIP_MODEL_PROVIDER` for a live provider:

```bash
TULIP_MODEL_ID=openai.gpt-4.1 python examples/notebook_35_structured_output.py
```

Offline:

```bash
TULIP_MODEL_PROVIDER=mock python examples/notebook_35_structured_output.py
```

## Prerequisites

- `TULIP_MODEL_PROVIDER` pointed at `openai` / `anthropic` /
  `mock`, with the matching credentials.
- A model that supports constrained JSON decoding for Part 8. The
  `check_structured_output_capable()` helper exits cleanly under mock.

## Source

```python
--8<-- "examples/notebook_35_structured_output.py"
```
