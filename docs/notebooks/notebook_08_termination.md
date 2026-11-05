# Termination Conditions

Every agent loop needs to know when to stop. A customer-support agent
that keeps talking to itself wastes budget and leaves the customer
waiting, so a ticket-resolution loop must end after a fixed number of
turns, a time limit, or as soon as the agent declares the issue
resolved. The SDK ships small predicates you compose with `|` (OR) and
`&` (AND) to describe the exit condition exactly. This notebook also
covers two related conveniences: `output_key` and a callable
`system_prompt`.

What this notebook covers:

- Termination predicates: `MaxIterations`, `TextMention`, `TokenLimit`,
  `TimeLimit`, `ConfidenceMet`, plus `CustomCondition(callable)`.
- Combining with `|` and `&` — and inspecting the result by calling
  `.check(state)` directly (e.g. `MaxIterations(5) |
  TextMention("TICKET_RESOLVED")`).
- `output_key="resolution"` to store the final message at
  `result.state.metadata["resolution"]` so downstream agents do not
  have to parse prose.
- A callable `system_prompt(ctx)` that reads `ctx["metadata"]` and
  returns different instructions per run — a tier-1 agent versus a
  billing specialist writing for customers.

Each `|` or `&` builds one two-child node, so the support loop's
`MaxIterations(5) | TextMention("TICKET_RESOLVED")` is a tree of the
same kind as the refund condition drawn here,
`(ToolCalled("issue_refund") & ConfidenceMet(0.9)) | MaxIterations(10)`
— the agent loop evaluates the root at the start of every iteration,
before the model call (Think in the diagram), and when the root answers
stop the run ends with a `TerminateEvent` carrying its reason, which
`result.stop_reason` reports in normalized form
(`text_mention:TICKET_RESOLVED` becomes `complete`,
`tool_called:issue_refund` becomes `terminal_tool`):

{{ locus_diagram termination-algebra }}

## Key lines

The OR and AND compositions are probed directly with `.check(state)`,
and the same `MaxIterations(5) | TextMention("TICKET_RESOLVED")` is then
attached to a triage agent whose `result.stop_reason` is the normalized
reason — `complete` here, because under the default
`completion_mode="auto"` a reply with no tool calls ends the run before
the condition is checked again, and a `text_mention:*` reason would
normalize to `complete` as well:

{{ locus_source examples/notebook_08_termination.py::example_termination }}

`output_key="resolution"` writes the final message into
`result.state.metadata["resolution"]`, where the next agent can read it
without parsing prose:

{{ locus_source examples/notebook_08_termination.py::example_output_key }}

A callable `system_prompt` reads the per-run `metadata` and returns
different instructions for a tier-1 agent and for a billing specialist
writing for customers:

{{ locus_source examples/notebook_08_termination.py::example_dynamic_prompt }}

## Run it

```bash
python examples/notebook_08_termination.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

## Output

Running it offline — no credentials, bundled mock model — prints the `(stop, reason)` each predicate and composition returns (`max_iterations`, `text_mention:TICKET_RESOLVED`, `max_iterations AND token_budget`, `confidence_met`), so the OR and AND semantics are visible before the agent runs, then the agent's normalized stop reason, the resolution read back from `state.metadata['resolution']` and the two replies the callable system prompt produced:

```text
============================================================
Notebook 08: Bounded Resolution — When the Support Loop Stops
============================================================

Model Provider: mock
Using mock model (no API calls)

=== Part 1: Composable Termination ===

MaxIterations(5) | TextMention('TICKET_RESOLVED')
  Turn 6: stop=True, reason=max_iterations
  Message 'TICKET_RESOLVED': stop=True, reason=text_mention:TICKET_RESOLVED

MaxIterations(3) & TokenLimit(1000)
  Iterations met, tokens not: stop=False
  Both met: stop=True, reason=max_iterations AND token_budget

CustomCondition: (True, 'reply_budget_exhausted')
ConfidenceMet(0.9) at confidence 0.92: (True, 'confidence_met')
TimeLimit(3600) on a fresh run: (False, None)

Agent reply: This is a mock response for testing purposes.
Stop reason: complete

=== Part 2: output_key ===

Response: This is a mock response for testing purposes.
State metadata['resolution']: This is a mock response for testing purposes.
Downstream agents read state.metadata['resolution'] directly — no parsing.

=== Part 3: Dynamic System Prompt ===

Tier-1 agent: This is a mock response for testing purposes.
Billing specialist (for customers): Possible friendly fraud: delivery was signed for at the billing address and the account has prior un

============================================================
Next: Notebook 09 — Audit hooks: every agent action on the record
============================================================
```
<!-- notebook-output:end -->

## Source

??? note "Full source — `examples/notebook_08_termination.py`"

    ````python
    --8<-- "examples/notebook_08_termination.py"
    ````
