# Termination Conditions

Every agent loop needs to know when to stop. A customer-support agent
that keeps talking to itself wastes budget and leaves the customer
waiting, so a ticket-resolution loop must end after a fixed number of
turns, a time limit, or as soon as the agent declares the issue
resolved. Locus ships small predicates you compose with `|` (OR) and
`&` (AND) to describe the exit condition exactly. This notebook also
covers two related conveniences: `output_key` and a callable
`system_prompt`.

What you'll learn:

- Termination predicates: `MaxIterations`, `TextMention`, `TokenLimit`,
  `TimeLimit`, `ConfidenceMet`, plus `CustomCondition(callable)`.
- Combining with `|` and `&` — and inspecting the result by calling
  `.check(state)` directly (e.g. `MaxIterations(5) |
  TextMention("TICKET_RESOLVED")`).
- `output_key="resolution"` to drop the final message into
  `result.state.metadata["resolution"]` so downstream agents don't have
  to parse prose.
- A callable `system_prompt(ctx)` that reads `ctx["metadata"]` and
  returns different instructions per run — a tier-1 agent versus a
  billing specialist writing for customers.

Run it:

```
.venv/bin/python examples/notebook_08_termination.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

## Source

````python
--8<-- "examples/notebook_08_termination.py"
````
