# Reasoning Patterns

Walks the Locus reasoning toolkit one piece at a time against a
data-privacy postmortem: a privacy-team exercise seeded a synthetic PII
canary (a fake SSN) into a data export and the redaction pipeline never
flagged it. The parts that call a model print `[model call: X.XXs ·
prompt→completion tokens]` so you can see the round-trip.

The thesis the reasoning layer enforces: an ungrounded claim in a
postmortem is how a false root cause gets written into a runbook, so each
claim is scored against tool evidence before it is allowed to stand.

- `@tool` + `Agent(tools=...)` — let the agent call real Python functions
  over the redaction pipeline's logs (`read_redaction_logs`) and DLP
  stats (`query_dlp`).
- `Agent(reflexion=True)` and `Reflector` — Reflexion is a self-critique
  loop; the agent inspects its own trajectory and decides whether it's
  making progress or stuck.
- `Agent(output_schema=...)` — typed JSON for postmortem claims and event
  timelines.
- `GroundingEvaluator` — score each claim against tool evidence and
  decide whether to replan.
- `CausalChain` / `build_causal_chain` — build and walk a cause/effect
  graph from the canary seed to PII shipped in the clear.

## Run it

The bundled mock model is the default; set `LOCUS_MODEL_PROVIDER` for a live provider:

```bash
LOCUS_MODEL_PROVIDER=openai LOCUS_MODEL_ID=gpt-4o python examples/notebook_54_reasoning_patterns.py
```

Offline:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_54_reasoning_patterns.py
```

## Prerequisites

- An OCI profile in `~/.oci/config` (picked up automatically), or
  `LOCUS_MODEL_PROVIDER` set to `openai` / `anthropic` / `mock` with the
  matching key.
- A model that supports constrained JSON decoding for the
  `output_schema=` parts. The `check_structured_output_capable()` helper
  exits cleanly under mock.

## Source

````python
--8<-- "examples/notebook_54_reasoning_patterns.py"
````
