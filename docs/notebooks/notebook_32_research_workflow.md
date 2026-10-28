# Research Workflow

The end-of-series capstone: a research-shaped pipeline that strings six
node primitives into a single `StateGraph` and streams every step. Here
a customer-support analyst investigates the known-issue knowledge base —
gather evidence from KB tools, infer the root cause, summarise, judge the
summary's grounding, and recover when the score is low. An ungrounded
claim is a hallucinated answer, so it never reaches the customer reply.

## What you learn

- Composing a research workflow with `create_research_workflow`.
- The two-tier recovery loop: cheap `regenerate_summary` on the first
  grounding miss, then a full `replan + execute` on subsequent misses.
- Streaming `research.*` SSE events live, the same way you would stream
  any `Agent` run.
- Reading the final state — summary, structured output, grounding score,
  causal hypothesis + confidence.

## Prerequisites

This workflow builds on the agent loop, tools, streaming events,
graphs, DeepAgent, and SSE observability. Read those first if any of
the pieces look unfamiliar.

## Run it

```bash
# OCI Generative AI with an OCI profile, otherwise the bundled mock model;
# set LOCUS_MODEL_PROVIDER to force a provider.
python examples/notebook_32_research_workflow.py

# Offline, no credentials:
LOCUS_MODEL_PROVIDER=mock python examples/notebook_32_research_workflow.py
```

## Source

````python
--8<-- "examples/notebook_32_research_workflow.py"
````
