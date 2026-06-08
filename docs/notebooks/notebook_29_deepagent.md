# DeepAgent

`create_deepagent` bundles the configuration patterns for deep research
into one call: reflexion + grounding on by default, a typed termination
algebra, plus opt-in filesystem scratchspace, todo tracking, subagent
spawning, and datastore auto-wiring. The result is a plain
`tulip.Agent` — every hook, checkpointer, and observability primitive
attaches normally.

This notebook covers:

1. Basic `create_deepagent` with a typed submit tool — the agent loops
   with tools, self-corrects via reflexion, grounds claims against tool
   results, and submits a structured `ModuleReport`.
2. Filesystem-as-memory: `write_file` / `read_file` for scratchpad
   notes that persist across iterations without bloating context.
3. Todo tracking: `write_todos` / `read_todos` backed by a `TodoState`
   the caller can inspect after the run.
4. Subagent dispatch: `SubAgentDef` + `task(...)` — one-shot delegated
   investigations whose trajectories never reach the parent's context.
5. `deepagent.*` SSE events: `subagent.spawned/completed`, `fs.*`,
   `todo.*`.
6. **RAG grounding** — `datastores={name: {retriever, description, top_k}}`
   auto-wires a `search_<name>` tool from any `RAGRetriever` and
   prepends a routing block to the system prompt. The path exercised
   here is `InMemoryVectorStore` + `OpenAIEmbeddings`; absent an
   embedding key, Part 5 exits cleanly.

The factory is convenience-only: the returned Agent has nothing
"DeepAgent-specific" once it's built. Typed termination reads like a
sentence — `(ToolCalled("submit") & ConfidenceMet(0.85))
| TokenLimit(80_000)` — and can be unit-tested without a model.

## Prerequisites

- Notebook 06 (Agent basics).
- Notebook 15 (typed termination).
- For Part 5 only: `OPENAI_API_KEY` for embeddings.

## Run

```bash
python examples/notebook_29_deepagent.py
```

The default provider is the bundled mock model. Set
`TULIP_MODEL_PROVIDER` (openai / anthropic) and credentials to
use a live model. Keep `TULIP_MODEL_PROVIDER=mock` for offline runs.

Multi-backend ports (in-memory + OpenSearch) live in
[`examples/projects/deep-research`](https://github.com/tuliplabs-ai/sdk-python/tree/main/examples/projects/deep-research).

## Source

```python
--8<-- "examples/notebook_29_deepagent.py"
```
