# Research Workflow

The end-of-series capstone: a research-shaped pipeline that strings six
node primitives into a single `StateGraph` and streams every step.
Mirrors the production pattern used by specialist research agents —
gather evidence, infer causality, summarise, judge the summary's
grounding, and recover when the score is low.

## What you learn

- Composing a research workflow with `create_research_workflow`.
- The two-tier recovery loop: cheap `regenerate_summary` on the first
  grounding miss, then a full `replan + execute` on subsequent misses.
- Streaming `research.*` SSE events live, the same way you would stream
  any `Agent` run.
- Reading the final state — summary, structured output, grounding score,
  causal hypothesis + confidence.

## Prerequisites

Notebook 64 builds on the agent loop (notebook 08), tools (09),
streaming events (11), graphs (16), DeepAgent (29), and SSE
observability (53). Read those first if any of the pieces look
unfamiliar.

## Run it

```bash
# Default: the bundled mock model. Set TULIP_MODEL_PROVIDER=openai
# (or anthropic ) and the matching credentials for a live model.
python examples/notebook_69_research_workflow.py

# Offline, no credentials:
TULIP_MODEL_PROVIDER=mock python examples/notebook_69_research_workflow.py
```

## Source

```python
--8<-- "examples/notebook_69_research_workflow.py"
```
