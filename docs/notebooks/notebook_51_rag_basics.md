# RAG basics

Retrieval-Augmented Generation (RAG) grounds an agent's answers in your
own documents rather than in what the model recalls. For a cloud-platform
team that means the cloud Well-Architected best-practice catalogue
(`REL-xx`, `COST-xx`, `SEC-xx`, `OPS-xx`) mapped to its runbooks. This
notebook builds that Index over the bundled in-memory vector store, first
by hand and then through `RAGRetriever`; it is the Index that ATLAS, the
cloud-ops agent of [notebook 53](notebook_53_rag_agents.md), reads from.

The pipeline has four stages: embed the text into vectors, store the
vectors, search them by similarity, and generate an answer from the hits.
The script covers the first three; generation is notebook 53's subject.
Retrieval quality bounds how grounded the agent's answers can be: a
low-coverage or stale Index leaves it guessing instead of citing a
runbook.

What this notebook covers:

- Embeddings: `OpenAIEmbeddings` (`text-embedding-3-small`, 1536
  dimensions) and pairwise cosine similarity across three operations
  sentences.
- The store driven by hand: `Document`, `add_batch()` and `search()` on
  the pure-Python `InMemoryVectorStore`, which keeps vectors in process
  and ranks them by cosine similarity; swap in `OracleVectorStore`,
  `PgVectorStore`, `OpenSearchVectorStore`, `QdrantVectorStore` or
  `ChromaVectorStore` for a durable backend.
- `RAGRetriever`: `add_document()`, `retrieve()` and `retrieve_text()`
  over the same embedder and store.
- Metadata-filtered retrieval: every chunk tagged with its pillar and
  practice id, and `metadata_filter` narrowing the search to one pillar.

## Prerequisites

```bash
export OPENAI_API_KEY=sk-...
```

The drawing shows the two lanes of the retrieval pipeline, which share
one embedder and one vector store. This notebook drives both lanes
directly over the in-memory store, with no loader, no reranker and no
agent in front: step 2 runs the ingest lane by hand with `embed_batch()`
and `add_batch()`, and step 3 hands it to `RAGRetriever.add_document()`,
which would also split any text longer than `chunk_size`.
`RAGRetriever.retrieve()` is the retrieve lane, embedding each question
with the same embedder, searching the store and returning a
`RetrievalResult` whose documents carry content, score and metadata; the
Agent and Search tool at the head of that lane are what notebook 53 adds:

{{ locus_diagram rag-pipeline }}

## Key lines

Step 2 walks the ingest lane by hand: one `embed_batch()` call embeds all
five practice summaries, each becomes a `Document` carrying its vector
and a `source` tag, one `add_batch()` writes them to the
`InMemoryVectorStore`, and `store.search()` ranks them by cosine
similarity against the embedded question:

{{ locus_source examples/notebook_51_rag_basics.py::using_vector_store }}

Step 3 replaces that hand-driven ingest with
`RAGRetriever(embedder=..., store=..., chunk_size=500, chunk_overlap=50)`:
`add_document()` embeds and stores each of the three notes in one call,
splitting only text longer than `chunk_size` (these notes are shorter, so
each is stored as one chunk); `retrieve(query, limit=2)` returns a
`RetrievalResult` with a score per document; and `retrieve_text()` runs
the same search and returns the hits as one string, each wrapped in
`<retrieved_document>` markers so a prompt can treat retrieved text as
data rather than instructions:

{{ locus_source examples/notebook_51_rag_basics.py::using_rag_retriever }}

Step 4 stores the pillar and practice id as metadata on every chunk and
runs the same cost-and-capacity question twice: once unfiltered, where a
reliability practice that merely sounds similar can outrank the cost
answer, and once with `metadata_filter={"pillar": "cost-optimization"}`,
which `retrieve()` passes to the store and the in-memory store applies
before scoring, so only that pillar is ranked:

{{ locus_source examples/notebook_51_rag_basics.py::rag_with_metadata }}

## Run it

```bash
python examples/notebook_51_rag_basics.py
```

The script needs an OpenAI embeddings key and no chat model, so
`get_model()` and `LOCUS_MODEL_PROVIDER` play no part here.
`get_embedder()` in `examples/config.py` builds `OpenAIEmbeddings`, which
reads the `OPENAI_API_KEY` set under Prerequisites. `LOCUS_EMBEDDING_MODEL`
(default `text-embedding-3-small`) selects the model id, and
`OPENAI_BASE_URL` points the client at a self-hosted OpenAI-compatible
embedding endpoint instead of api.openai.com; the script still checks
that `OPENAI_API_KEY` is set either way. Without the key it prints a skip
notice naming the variable and exits cleanly before any embedding call;
the script has no offline run.

## See also

- [RAG providers](notebook_52_rag_providers.md) — choosing the embedding
  model and distance metric, and swapping in another vector store.
- [RAG agents](notebook_53_rag_agents.md) — `RAGRetriever.as_tool()`
  gives ATLAS a search tool over this Index.
- [Concepts: RAG](../concepts/rag.md) — the pipeline end to end, and when
  to turn on [hybrid search](../concepts/rag.md#hybrid-search-vector-plus-lexical),
  with `metadata_filter` as one of the precision levers.
- [API: RAG](../api/rag.md)

## Source

??? note "Full source — `examples/notebook_51_rag_basics.py`"

    ````python
    --8<-- "examples/notebook_51_rag_basics.py"
    ````
