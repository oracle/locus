# RAG agents

Once your runbooks are in a vector store (the [RAG basics](notebook_51_rag_basics.md)
and [RAG providers](notebook_52_rag_providers.md) notebooks), the next step is to
let an agent reach into it, so that an incident answer cites your runbooks instead
of model memory. This is Retrieval-Augmented Generation (RAG) as an agent tool:
`RAGRetriever.as_tool()` turns the retriever into an ordinary tool that the
agent picks up alongside any other `@tool` you define. ATLAS is the platform
team's on-call Site Reliability Engineering (SRE) copilot; it reads the Index,
the internal runbook knowledge base.

Backend: the pure-Python `InMemoryVectorStore` keeps the vectors in process,
so the example needs no database. Swap `_make_store` for any other vector store (`OracleVectorStore` for Oracle Database 26ai, `PgVectorStore`,
`OpenSearchVectorStore`, `QdrantVectorStore`, `ChromaVectorStore`) for a
durable backend.

What this notebook covers:

- `retriever.as_tool(name, description)`: converting a retriever into a
  callable tool for the agent.
- A single-tool question-and-answer copilot over an internal runbook
  knowledge base.
- A mixed tool set: runbook search alongside a calculator and a date tool
  for rollout arithmetic.
- Streaming: the example prints `ToolStartEvent`, `ToolCompleteEvent` and
  `ThinkEvent` as the agent searches and answers (the
  [Events](../concepts/events.md) page describes every event type).
- An **answer-grounding gate**: before ATLAS ships remediation advice, the
  drafted answer's claims are placed in a [GSAR](../concepts/gsar.md)
  (typed grounding) `Partition` according to the evidence behind them;
  `gsar_score()` scores the partition and `decide()` returns proceed,
  regenerate or replan. A fix anchored in a retrieved runbook chunk
  proceeds; for a claim with no retrieved evidence the gate replans rather
  than let the agent guess.

## Prerequisites

```bash
export OPENAI_API_KEY=sk-...
```

In this notebook the ingest lane is only `add_documents()`: there is no
loader, and each runbook string is shorter than the retriever's `chunk_size`
(1000 characters by default), so `OpenAIEmbeddings` embeds it whole and the
retriever writes it to the `InMemoryVectorStore`. The retrieve lane is the
subject: ATLAS calls the search tool that `retriever.as_tool()` built, the
tool runs `RAGRetriever.retrieve()`, which embeds the responder's question
with the same embedder and searches the store, and because no reranker is
configured the ranked chunks come back as they are, each with content, score
and metadata, as the text the agent answers from:

{{ locus_diagram rag-pipeline }}

## Key lines

`RAGRetriever` pairs the embedder with the store, `add_documents()` fills the
Index, and `as_tool(name, description)` returns a tool. The example
calls that tool directly, before any agent is involved, to show the result
shape it hands back: a `results` list of chunks, each with `content`, `score`,
`metadata` and `id`, alongside `total`:

{{ locus_source examples/notebook_53_rag_agents.py::rag_as_tool }}

ATLAS is an ordinary `Agent` whose only tool is the retriever: the system
prompt tells it to answer from search results alone and to say when the
runbooks do not cover a question, and each `arun()` call is one responder
question:

{{ locus_source examples/notebook_53_rag_agents.py::simple_rag_agent }}

The grounding gate needs no embeddings or model. The example places the
drafted answer's claims in a `Partition` according to the evidence behind
them: grounded when a claim points at a retrieved runbook chunk or the deploy
log, ungrounded when the model inferred it. `gsar_score()` scores the
partition and `decide()` returns proceed for the remediation that cites
retrieved runbook chunks and replan for the cross-incident claim that cites
none:

{{ locus_source examples/notebook_53_rag_agents.py::answer_grounding_gate }}

## Run it

```bash
python examples/notebook_53_rag_agents.py
```

Embeddings need `OPENAI_API_KEY`: `get_embedder()` in `examples/config.py`
builds `OpenAIEmbeddings` with `text-embedding-3-small` unless
`LOCUS_EMBEDDING_MODEL` names another model, and `OPENAI_BASE_URL` points it
at a self-hosted endpoint. Without the key the script prints one skip notice,
leaves out the four retrieval sections, still runs the answer-grounding gate,
which is pure Python, and exits cleanly. The chat model behind ATLAS comes from
`get_model()`: OCI Generative AI when `~/.oci/config` has a usable profile,
otherwise the bundled mock model. Set `LOCUS_MODEL_PROVIDER` (`oci`, `openai`,
`anthropic` or `mock`) to force one.

## Output

Running it offline — no credentials, bundled mock model — prints one skip notice for the four retrieval sections, which need OPENAI_API_KEY, then the answer-grounding gate's decision trail: S=1.0 and proceed for the remediation that cites retrieved runbook chunks, with its evidence references, and S=0.0 and replan for the unsupported cross-incident claim:

```text

--- Notebook 53: ATLAS on-call SRE copilot ---
Embedding credentials not set; skipping the retrieval sections so this file still runs cleanly in CI. The answer-grounding gate below is pure-Python and runs anyway.

  - OPENAI_API_KEY

Set OPENAI_API_KEY (for embeddings) to run ATLAS over the Index.

============================================================
Notebook 53: Answer-grounding gate (GSAR decide)
============================================================
Responder: What is the remediation for the checkout-api latency regression?

Grounded case (answer cites retrieved runbook chunks + deploy log):
  S=1.0000  decision=proceed
  SHIP  — roll forward to 2.4.1; interim roll back to 2.3.7 via Argo.
    evidence: ['retriever:search_runbook_docs:doc=RB-014:offset=0', 'retriever:search_runbook_docs:doc=RB-014:offset=2', 'deploy_log:service=checkout-api:rev=2.4.0']

Speculative case (a hunch, no retrieved evidence):
  S=0.0000  decision=replan
  HOLD  — replanning: cross-incident claim has no runbook support.

============================================================
Done. Next: notebook 54 — reasoning patterns.
============================================================
```
<!-- notebook-output:end -->

## See also

- [RAG basics](notebook_51_rag_basics.md) and
  [RAG providers](notebook_52_rag_providers.md) — building the Index this
  agent reads.
- [GSAR typed grounding](notebook_55_gsar_typed_grounding.md) — the four-way
  partition, the score and the decision loop in full.
- [Concepts: RAG](../concepts/rag.md#retrieval-is-vector-similarity-only) —
  chunk-size and metadata-filter levers for operations corpora, and
  reranking.
- [Concepts: GSAR](../concepts/gsar.md) — when to use typed grounding over
  `grounding=True`.

## Source

??? note "Full source — `examples/notebook_53_rag_agents.py`"

    ````python
    --8<-- "examples/notebook_53_rag_agents.py"
    ````
