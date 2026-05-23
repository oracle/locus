# Termination Conditions

Every agent loop needs to know when to stop. Locus ships small
predicates you compose with `|` (OR) and `&` (AND) to describe the exit
condition exactly. This notebook also covers two related conveniences:
`output_key` and a callable `system_prompt`.

What you'll learn:

- Termination predicates: `MaxIterations`, `TextMention`, `TokenLimit`,
  `TimeLimit`, `ConfidenceMet`, plus `CustomCondition(callable)`.
- Combining with `|` and `&` — and inspecting the result by calling
  `.check(state)` directly.
- `output_key="answer"` to drop the final message into
  `result.state.metadata["answer"]` so downstream agents don't have to
  parse prose.
- A callable `system_prompt(ctx)` that reads `ctx["metadata"]` and
  returns different instructions per run.

Run it:

```
.venv/bin/python examples/notebook_15_termination.py
```

Uses the OCI Generative AI default provider (canonical id:
`openai.gpt-4.1` or `meta.llama-3.3-70b-instruct`). For offline runs set
`LOCUS_MODEL_PROVIDER=mock`; OpenAI, Anthropic and Ollama also work.

## Source

```python
--8<-- "examples/notebook_15_termination.py"
```
