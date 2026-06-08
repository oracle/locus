# Reasoning Patterns

Walks the Tulip reasoning toolkit one piece at a time. Each part fires
a real model call and prints `[model call: X.XXs · prompt→completion
tokens]` so you can see the round-trip.

- `@tool` + `Agent(tools=...)` — let the agent call real Python functions.
- `Agent(reflexion=True)` and `Reflector` — Reflexion is a self-critique
  loop; the agent inspects its own trajectory and decides whether it's
  making progress or stuck.
- `Agent(output_schema=...)` — typed JSON for claims and event timelines.
- `GroundingEvaluator` — score each claim against tool evidence and
  decide whether to replan.
- `CausalChain` / `build_causal_chain` — build and walk a cause/effect
  graph.

## Run it

The bundled mock model is the default; set `TULIP_MODEL_PROVIDER` for a live provider:

```bash
TULIP_MODEL_ID=openai.gpt-4.1 python examples/notebook_36_reasoning_patterns.py
```

Offline:

```bash
TULIP_MODEL_PROVIDER=mock python examples/notebook_36_reasoning_patterns.py
```

## Prerequisites

- An OpenAI or Anthropic API key, or `TULIP_MODEL_PROVIDER` set to
  `openai` / `anthropic` / `mock`.
- A model that supports constrained JSON decoding for the
  `output_schema=` parts. The `check_structured_output_capable()` helper
  exits cleanly under mock or Cohere R-series.

## Source

```python
--8<-- "examples/notebook_36_reasoning_patterns.py"
```
