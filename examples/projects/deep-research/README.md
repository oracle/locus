# deep-research — Tulip SDK deep-research examples

A runnable suite of `create_deepagent(datastores=...)` examples covering
the retrieval backends the SDK supports: **in-memory** and **OpenSearch**.
Each demo is built 1:1 on the SDK's primitives — no langchain or
deepagents imports.

```
                    ┌────────────────────────────────┐
                    │  create_deepagent(             │
                    │    datastores={"intel": ...},  │
                    │    max_output_tokens=4096,     │
                    │  )                             │
                    │  + auto-wired search_<name>    │
                    │    tools per datastore         │
                    └─────────────┬──────────────────┘
                                  │
       ┌──────────────┬───────────┼────────────┐
       ▼              ▼           ▼            ▼
  InMemoryVector   OpenSearch  Qdrant /     any custom
  Store            k-NN index  pgvector /   RAGRetriever
                               Chroma
```

## Demos

| Demo | Backend |
|---|---|
| [`demo_hello_world.py`](demo_hello_world.py) | none — just `@tool` functions |
| [`demo_smoke.py`](demo_smoke.py) | InMemoryVectorStore |
| [`demo_opensearch_multi_index.py`](demo_opensearch_multi_index.py) | Two OpenSearch indices (threat intel + CVE) |

## Quick start (InMemory smoke — no DB required)

```bash
export OPENAI_API_KEY=sk-...          # embeddings
export ANTHROPIC_API_KEY=sk-ant-...   # chat model
python examples/projects/deep-research/demo_smoke.py
```

Uses OpenAI for embeddings (`text-embedding-3-small`) and Anthropic for
chat completions; auto-wires the `search_intel` tool from the
in-memory `RAGRetriever`. Expects 1 tool call + a short memo on 10
inline threat-intel sentences.

## OpenSearch multi-index replay

```bash
export OPENSEARCH_ENDPOINT=https://<your-opensearch-host>:9200
export OPENSEARCH_USERNAME=<your-username>
export OPENSEARCH_PASSWORD='...'
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export TULIP_RESEARCH_MODEL=anthropic:claude-sonnet-4-6

python examples/projects/deep-research/demo_opensearch_multi_index.py
```

Embeds two corpora with `text-embedding-3-small`, seeds two OpenSearch
indices, then routes each search at the right index.

## Gotchas surfaced during these examples

1. **`tulip.memory.InMemoryStore` is async.** Its `put`/`search`/`get`
   are coroutines. Use `await store.put(...)`.
2. **`OpenSearchVectorStore._client` is `AsyncOpenSearch`.** When you
   need to drive the underlying client directly (refresh, exists,
   delete), `await` every call. Sync-style calls silently no-op.
3. **From inside an async `def` use `async for event in agent.run(...)`,
   not `agent.run_sync(...)`.** `run_sync` spawns a new thread + event
   loop; any aiohttp/AsyncOpenSearch client created on the caller's
   loop becomes unusable from the agent's tool calls and returns
   silent empty results.
4. **Some model providers JSON-encode floats as strings.** Some models
   send `"min_score": "0.5"` rather than `0.5` for tool args; the SDK
   coerces this defensively in `RAGRetriever.retrieve`. If you
   implement a custom store, coerce at the search boundary too.
