# Research workflow

The capstone of the multi-agent shapes: a research-shaped pipeline that strings six node primitives into a single `StateGraph` and streams every step. Here a customer-support analyst investigates the known-issue knowledge base — gathering evidence from knowledge-base tools, inferring the root cause, summarizing, scoring the summary's grounding against the evidence, and reworking it when the score is low.

An ungrounded claim is a hallucinated answer, so a summary below the threshold is sent back for rework while the recovery budgets last rather than straight into the customer reply; the final state carries the last grounding score next to the summary, so the caller can tell when the budgets ran out first.

What this notebook covers:

- Composing a research workflow with `create_research_workflow`.
- The two-tier recovery loop: regenerate the summary first, then replan
  and re-execute.
- Streaming the run's `research.*` events from the in-process event bus
  as they happen, through the same
  [in-process SSE](../concepts/observability.md#in-process-sse-eventbus)
  (server-sent events) subscription that carries any `Agent` run's
  `agent.*` events.
- Reading the final state: summary, structured output, grounding score,
  causal hypothesis and confidence.

## Prerequisites

- [Basic agent](notebook_02_basic_agent.md) and
  [Agent with tools](notebook_03_agent_with_tools.md).
- [Streaming events](notebook_06_agent_streaming.md).
- [Basic graph](notebook_12_basic_graph.md) and
  [Conditional routing](notebook_13_conditional_routing.md).
- [DeepAgent](notebook_31_deepagent.md).
- [Observability basics](notebook_60_observability_basics.md) and
  [SSE streaming](notebook_64_sse_streaming.md).

Part 1 runs the graph in the drawing against the knowledge-base tools:

- Execute gathers evidence in an agent tool loop.
- Causal inference builds a root-cause chain from that evidence (on by default; with `causal_inference=False` Execute goes straight to Summarize).
- Summarize distills the evidence into the summary and, when the model returns one, the typed `SupportAssessment`.
- Grounding eval scores the summary's claims against the evidence.

Below `grounding_threshold` the router goes to Regenerate, a rewrite without re-running tools, while regenerations are left, then to Replan, a prompt focused on the ungrounded claims and sent back to Execute, while replans are left. Once both are used up the run ends:

{{ locus_diagram deepagent-workflow }}

## Key lines

`SupportAssessment` is the workflow's structured output: Summarize and
Regenerate parse the model's reply into it when `output_schema` is set,
and Part 1 reads it back from the final state when the model fills it
in:

{{ locus_source examples/notebook_32_research_workflow.py::SupportAssessment }}

Part 1 composes the workflow with `create_research_workflow`, where
`grounding_threshold`, `max_regenerations` and `max_replans` set the
recovery budget, subscribes to the run's `research.*` events through
`run_context` and the event bus while `workflow.execute` runs, and reads
the grounding score, causal hypothesis and recovery counts from
`result.final_state`:

{{ locus_source examples/notebook_32_research_workflow.py::part1_factory_with_sse }}

Part 2 builds a smaller graph by hand from the same node primitives —
execute, summarize and grounding eval, with no causal step — and wires
`route_after_grounding` in as the conditional edge. With both recovery
budgets at zero the router only ever returns `END`; the `regenerate` and
`replan` labels are mapped to `END` as well so that every label the
router can return resolves to a target, since `compile()` does not check
the map and an unmapped label is used as a node id at run time:

{{ locus_source examples/notebook_32_research_workflow.py::part2_custom_graph }}

## Run it

```bash
python examples/notebook_32_research_workflow.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

## See also

- [DeepAgent](../concepts/deepagent.md#create_research_workflow-stategraph-with-quality-loop)
  — `create_research_workflow` and its parameters, next to the
  single-loop `create_deepagent`.
- [SSE event catalogue](../concepts/sse-events.md#research-research-workflow-nodes)
  — the `research.*` events and the fields each one carries.

## Source

??? note "Full source — `examples/notebook_32_research_workflow.py`"

    ````python
    --8<-- "examples/notebook_32_research_workflow.py"
    ````
