# DeepAgent

`create_deepagent` bundles the configuration patterns for deep, methodical
investigation into one call: reflexion and grounding on by default, a typed
termination algebra, plus opt-in filesystem scratchspace, todo tracking,
subagent spawning, and datastore auto-wiring. The factory is a convenience
layer only: the result is a plain `locus.Agent` with no DeepAgent-specific
surface once it is built, so every hook, checkpointer, and observability
primitive attaches normally.

Here the agent runs a scoped service-fleet reliability review — the
pre-release readiness step of an on-call rotation, bounded to the services
inside the deploy window. The shared fixture is a small inventory
(`api-gateway`, `payments-worker`, and `web-frontend`, each with a
description, its firing alerts, and a last-deploy date) read by three tools,
`list_services`, `inspect_service`, and `count_active_alerts`; the scope is
enforced in code, since `inspect_service` refuses any name outside the deploy
window. The review is submitted as a structured `ServiceReport` through the
typed `submit_review` tool.

Termination is a typed expression rather than a flag: for Part 1 the factory
builds `(ToolCalled("submit_review", require_success=True) &
ConfidenceMet(0.85)) | MaxIterations(12)`, and a `TokenLimit` term joins the
chain only when `total_token_budget` is passed, which this example never
does. The expression can be unit-tested without a model.

What this notebook covers:

1. Basic `create_deepagent` with a typed submit tool and `ServiceReport` as
   the output schema.
2. Filesystem-as-memory: `write_file` / `read_file` for scratchpad notes
   that persist across iterations without bloating context.
3. Todo tracking: `write_todos` / `read_todos` backed by a `TodoState` the
   caller can inspect after the run.
4. Subagent dispatch: `SubAgentDef` + `task(...)` — one-shot delegated
   investigations whose trajectories never reach the parent's context.
5. The `deepagent.*` events (`subagent.spawned` / `completed`, `fs.read` /
   `write`, `todo.added` / `completed`) published on the in-process event
   bus and listed in the [SSE event catalogue](../concepts/sse-events.md)
   (SSE: server-sent events).
6. Retrieval-Augmented Generation (RAG) grounding:
   `datastores={name: {retriever, description, top_k}}` auto-wires a
   `search_<name>` tool from any `RAGRetriever` and prepends a routing block
   to the system prompt.

## Prerequisites

- [Agent basics](notebook_02_basic_agent.md).
- [Termination conditions](notebook_08_termination.md) — the algebra
  `create_deepagent` composes.
- For Part 5 only: `OPENAI_API_KEY` for embeddings.

The `locus.deepagent` package also ships the research pattern as an explicit graph, `create_research_workflow`, and the drawing shows that shape. Execute is the agent tool loop that gathers evidence, Causal inference builds a causal chain from it, Summarize distills the evidence into a summary, and Grounding eval scores the summary's claims against the evidence with the same `GroundingEvaluator` class that `create_deepagent` configures when `grounding=True`.

Below the threshold the summary is regenerated without re-running tools while regenerations are left, then replanned with a prompt focused on the ungrounded claims, and the run ends once the score clears the threshold or both budgets are used up:

{{ locus_diagram deepagent-workflow }}

## Key lines

`part1_basic` is the whole pattern: `create_deepagent` takes the inventory tools and the typed `submit_review` tool, enforces `ServiceReport` as the output schema, and keeps reflexion and grounding on.

The run ends once `submit_review` has returned without error and the agent's confidence — the figure the reflexion pass maintains on the agent state, not the `confidence` field inside the report — has reached `min_confidence`, with `max_iterations` as the backstop.

An output schema also switches the loop to explicit completion, in which the grounding check — run only before a final answer that calls no tool — is never reached, so the exit rests on the submit tool and the confidence gate. `arun` runs the loop and `result.parsed` is the final answer coerced into `ServiceReport`:

{{ locus_source examples/notebook_31_deepagent.py::part1_basic }}

`part3_subagents` declares a single-tool `SubAgentDef` and passes it as
`subagents`; the parent reaches it through the injected `task()` tool, only
the subagent's final answer comes back into the parent's context, the system
prompt names `alert_analyst` because the factory does not add the subagent
catalogue to the prompt, and `min_confidence` drops to 0.8 here and in
Part 4:

{{ locus_source examples/notebook_31_deepagent.py::part3_subagents }}

`part5_datastores` passes a `datastores` entry, and the factory auto-wires a
`search_incident_notes` tool from the `RAGRetriever` and prepends a routing
block to the system prompt, so the agent can answer from the prior incident
notes; the function checks for `OPENAI_API_KEY` first and skips cleanly
without it:

{{ locus_source examples/notebook_31_deepagent.py::part5_datastores }}

## Run it

```bash
python examples/notebook_31_deepagent.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

Part 5 additionally requires `OPENAI_API_KEY`, which the shared
`get_embedder()` uses for `OpenAIEmbeddings`, and the `qdrant` extra. Without
the key the script prints a skip notice for that part and returns; without
the extra it prints a skip notice naming `pip install "locus-sdk[qdrant]"`.
Parts 1 to 4 run either way.

## See also

- [DeepAgent](../concepts/deepagent.md) — the factory's capability layers,
  the `deepagent.*` event table, and `create_research_workflow`.
- [Research workflow](notebook_32_research_workflow.md) — the research
  pattern as an explicit `StateGraph` with the grounding quality loop.
- [`examples/projects/deep-research`](https://github.com/oracle/locus/tree/main/examples/projects/deep-research)
  — multi-backend ports (in-memory and OpenSearch).

## Source

??? note "Full source — `examples/notebook_31_deepagent.py`"

    ````python
    --8<-- "examples/notebook_31_deepagent.py"
    ````
