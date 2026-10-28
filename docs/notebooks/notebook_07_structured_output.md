# Structured Output

Get a Pydantic object back from a model call instead of a string you
have to re-parse — a typed ticket update the rest of the help desk can
route on. Every part below fires a real model call and prints a
`[model call: X.XXs · prompt→completion tokens]` banner.

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

OCI Generative AI with an OCI profile, otherwise the bundled mock model; set `LOCUS_MODEL_PROVIDER` to force a provider:

```bash
LOCUS_MODEL_PROVIDER=openai LOCUS_MODEL_ID=gpt-4o python examples/notebook_07_structured_output.py
```

Offline:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_07_structured_output.py
```

## Prerequisites

- `LOCUS_MODEL_PROVIDER` pointed at `openai` / `anthropic` /
  `mock`, with the matching credentials.
- A model that supports constrained JSON decoding for Part 8. The
  `check_structured_output_capable()` helper exits cleanly under mock.

## Source

````python
--8<-- "examples/notebook_07_structured_output.py"
````
