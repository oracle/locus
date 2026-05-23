# deep-research — Oracle Generative AI – Multi-Agent Locus SDK ports of the langchain-oci deep-research gists

A runnable suite of seven `create_deepagent(datastores=...)` examples
covering every retrieval backend the SDK supports: **in-memory**, **Oracle
Autonomous Database** (vector), **OCI Object Storage** (tool-based
retrieval), and **OpenSearch**. Each demo mirrors a published
langchain-oci deep-research gist 1:1 on the SDK's primitives — no langchain
or deepagents imports.

```
                    ┌────────────────────────────────┐
                    │  create_deepagent(             │
                    │    datastores={"medical": ...},│
                    │    max_output_tokens=65536,    │
                    │  )                             │
                    │  + auto-wired search_<name>    │
                    │    tools per datastore         │
                    └─────────────┬──────────────────┘
                                  │
       ┌──────────────┬───────────┼────────────┬────────────────┐
       ▼              ▼           ▼            ▼                ▼
  InMemoryVector   ADB 26ai   OpenSearch   OCI Object       any custom
  Store            Vector     k-NN index   Storage @tools   RAGRetriever
                   Search                  (list/read/
                                            search)
```

## Demos

| Demo | Backend | Mirrors |
|---|---|---|
| [`demo_hello_world.py`](demo_hello_world.py) | none — just `@tool` functions | [gist 2453c40e/04](https://gist.github.com/fede-kamel/2453c40eb044c453ff5b87885c460dad) |
| [`demo_smoke.py`](demo_smoke.py) | InMemoryVectorStore | shape of [gist 15ab302e](https://gist.github.com/fede-kamel/15ab302e6b4d155f192555a6a6e33cd0) |
| [`demo_iron_metabolism.py`](demo_iron_metabolism.py) | Oracle Autonomous DB (241-doc corpus) | [gist 15ab302e](https://gist.github.com/fede-kamel/15ab302e6b4d155f192555a6a6e33cd0) |
| [`demo_memory_multi_turn.py`](demo_memory_multi_turn.py) | ADB + `locus.memory.InMemoryStore` for response capture | [gist 15ab302e](https://gist.github.com/fede-kamel/15ab302e6b4d155f192555a6a6e33cd0) (multi-turn variant) |
| [`demo_multi_datastore.py`](demo_multi_datastore.py) | Two ADB tables (medical + news) | [gist 92b9a515](https://gist.github.com/fede-kamel/92b9a515155f4332f794d491602b6e79) (ADB variant) |
| [`demo_opensearch_multi_index.py`](demo_opensearch_multi_index.py) | Two OpenSearch indices (medical + news) | [gist 92b9a515](https://gist.github.com/fede-kamel/92b9a515155f4332f794d491602b6e79) (OpenSearch variant) |
| [`demo_object_storage.py`](demo_object_storage.py) | OCI Object Storage via `@tool`-wrapped SDK (`list_bucket_objects`, `read_bucket_object`, `search_bucket_data`) | [gist 00cb5682](https://gist.github.com/fede-kamel/00cb568227912735e9717ddc12b649c4) |

The Python-published gists with the verified SDK ports live at:

- [demo_hello_world](https://gist.github.com/fede-kamel/87f1dc5cabd8b434b5ff0e432804c553)
- [demo_iron_metabolism (full replay)](https://gist.github.com/fede-kamel/27a6fc50e0df3854e427ecf4715bc005)
- [demo_memory_multi_turn](https://gist.github.com/fede-kamel/36c238d9e549ef8b59b4c7f65b1039a5)
- [demo_multi_datastore](https://gist.github.com/fede-kamel/2300629bd73d0abab17e92a9722362a0)
- [demo_object_storage](https://gist.github.com/fede-kamel/886293f25576b1589d76a3c18bc3bc4e)
- [demo_opensearch_multi_index](https://gist.github.com/fede-kamel/eb9c8ef0ef59cf55c0f67cb8ed80feb1)

## SDK translations from langchain-oci

| langchain-oci | locus |
|---|---|
| `from langchain_oci import create_deep_research_agent` | `from locus.deepagent import create_deepagent` |
| `from langchain_oci import OCIGenAIEmbeddings` | `from locus.rag import OCIEmbeddings` |
| `from langchain_oci import ChatOCIGenAI` | `from locus.models import get_model("oci:...")` |
| `from langchain_oci.datastores import ADB` | `from locus.rag.stores.oracle import OracleVectorStore` + `RAGRetriever` |
| `from langchain_oci.datastores import OpenSearch` | `from locus.rag.stores.opensearch import OpenSearchVectorStore` |
| `from langchain_core.tools import tool` | `from locus.tools import tool` |
| `from langgraph.store.memory import InMemoryStore` | `from locus.memory import InMemoryStore` |
| `agent.invoke({"messages": [HumanMessage(...)]})` | `agent.run_sync("...")` *or* `async for event in agent.run("...")` |

## Quick start (InMemory smoke — no DB required)

```bash
export OCI_PROFILE=DEFAULT
export OCI_COMPARTMENT=ocid1.tenancy.oc1..<your-tenancy>
python examples/projects/deep-research/demo_smoke.py
```

Hits OCI GenAI for both embeddings (`cohere.embed-v4.0`) and chat
completions; auto-wires the `search_medical` tool from the in-memory
`RAGRetriever`. Expects 1 tool call + a short memo on 10 inline
sentences.

## Full ADB replay (45+ → 241-doc corpus, 60+ char-citations)

```bash
export OCI_PROFILE=DEFAULT
export OCI_COMPARTMENT=ocid1.tenancy.oc1..<your-tenancy>
export ADB_DSN=<your-adb-tns>
# ADB_USER defaults to ``locus_app`` (a least-privileged app schema).
# Provision it once as ADMIN — see docs/concepts/rag.md for the
# CREATE USER / GRANT script. Override only if you've named your
# app schema differently.
export ADB_USER=locus_app
export ADB_PASSWORD=<your-locus_app-password>
export ADB_WALLET_LOCATION=~/.oci/wallets/<your-adb>
export OCI_RESEARCH_MODEL=oci:openai.gpt-5.1
export MAX_OUTPUT_TOKENS=65536

python examples/projects/deep-research/demo_iron_metabolism.py
```

Embeds the corpus with `cohere.embed-v4.0`, seeds a `LOCUS_IRON_<uuid>`
vector table in your Oracle ADB, runs `gpt-5.1` with a 65,536-token
output budget, prints every retrieved snippet as evidence, and drops
the table on exit. A successful run looks like:

```
   tool calls         : 20
   memo               : 5166 words, 38,688 chars
   citations (unique) : 65 / 241 docs in corpus
   retrieved (total)  : 58,442 chars across 20 search calls
```

## Gotchas surfaced during the port

The OpenSearch + GPT-5.1 path surfaced three SDK-specific behaviors
worth pinning:

1. **`locus.memory.InMemoryStore` is async.** Its `put`/`search`/`get`
   are coroutines — unlike langgraph's sync `InMemoryStore`. Use
   `await store.put(...)`.
2. **`OpenSearchVectorStore._client` is `AsyncOpenSearch`.** When you
   need to drive the underlying client directly (refresh, exists,
   delete), `await` every call. Sync-style calls silently no-op.
3. **From inside an async `def` use `async for event in agent.run(...)`,
   not `agent.run_sync(...)`.** `run_sync` spawns a new thread + event
   loop; any aiohttp/AsyncOpenSearch client created on the caller's
   loop becomes unusable from the agent's tool calls and returns
   silent empty results.
4. **Some model providers JSON-encode floats as strings.** GPT-5.x
   sends `"min_score": "0.5"` rather than `0.5` for tool args; the SDK
   coerces this defensively in `RAGRetriever.retrieve` and
   `OracleVectorStore.search`. If you implement a custom store,
   coerce at the search boundary too.

## Model selection note

The OCI GenAI Gemini 2.5 Pro endpoint has been returning empty
completions for tool-augmented prompts intermittently (May 2026).
`gpt-5.1` (or `gpt-4o-mini` for lighter runs) is the most reliable
tool-caller on the OCI path right now and is the default in every
demo. `MAX_OUTPUT_TOKENS=65536` is supported on `gpt-5.1`.
