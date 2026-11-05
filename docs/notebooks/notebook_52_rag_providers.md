# RAG providers

Production [retrieval-augmented generation (RAG)](../concepts/rag.md) is
two pluggable pieces, both behind one interface.

- **Embeddings** — `OpenAIEmbeddings` (`text-embedding-3-small` /
  `-3-large`), `CohereEmbeddings` (Cohere's direct API), `OCIEmbeddings`
  (Cohere embedding models served by OCI Generative AI) or
  `OracleInDBEmbeddings` (ONNX models running inside Oracle Database
  26ai).
- **Vector store** — `InMemoryVectorStore` for demos, or a durable
  backend: `PgVectorStore`, `OpenSearchVectorStore`,
  `QdrantVectorStore`, `ChromaVectorStore` or `OracleVectorStore`
  (Oracle Database 26ai). Swapping is a one-line change; the
  retrieve/add API is identical.

The corpus here is a payments-operations runbook — decline codes,
dispute reason codes, refund service-level agreements (SLAs) and
Automated Clearing House (ACH) returns — the same knowledge a support
agent relies on to answer a merchant. Choosing the embedding model and
distance metric is an operations decision: it sets retrieval precision,
and weak retrieval routes the agent to the wrong runbook.

What this notebook covers:

- Part 1 — embedding-model selection: `text-embedding-3-small` against
  `text-embedding-3-large` on the same corpus.
- Part 2 — three `InMemoryVectorStore` instances built with
  `distance_metric` set to `COSINE`, `DOT` and `EUCLIDEAN`, and what the
  store does with those names.
- Part 3 — `QdrantVectorStore(location=":memory:")` as a drop-in for
  `InMemoryVectorStore`.
- Part 4 — batch ingest with `add_documents`, then `count()` and
  `clear()`.

## Prerequisites

```bash
export OPENAI_API_KEY=sk-...
```

The drawing shows the two lanes of the retrieval pipeline,
`RAGRetriever.add_documents` for ingest and `retrieve()` for retrieval,
sharing one embedder and one vector store; this notebook keeps the lanes
fixed and changes what sits in those two boxes (Part 1 swaps the embedder
between the two `OpenAIEmbeddings` models, Part 2 changes the store's
`distance_metric`, Part 3 replaces `InMemoryVectorStore` with
`QdrantVectorStore`), Parts 2 to 4 also call `add_batch()`, `add()`,
`search()`, `count()` and `clear()` on the store directly, and no agent
search tool or reranker is in the path, so the ranked hits return as
they are:

{{ locus_diagram rag-pipeline }}

## Key lines

Part 1 builds the same `RAGRetriever` twice, once per `OpenAIEmbeddings`
model, sizes each `InMemoryVectorStore` from `embedder.config.dimension`
(1536 and 3072) and runs both lanes with `add_documents` and `retrieve`;
the only thing that changes between the two passes is the embedding
model:

{{ locus_source examples/notebook_52_rag_providers.py::part1_embedding_models }}

Part 2 shows that the distance metric belongs to the store, not the
vectors: the corpus and the query are embedded once, and the same
`Document` objects are loaded into three stores that differ only in
`distance_metric`. `InMemoryVectorStore.search` compares that string
without folding case and matches only `cosine` (the default) and
`euclidean`; every other value is scored as a dot product, so the
upper-case `COSINE`, `DOT` and `EUCLIDEAN` this example passes, which
are the spellings `OracleVectorStore` validates, all take the
dot-product branch and the three rows print the same score:

{{ locus_source examples/notebook_52_rag_providers.py::part2_distance_metrics }}

Part 3 is the backend swap: `QdrantVectorStore(location=":memory:")`
takes the place of `InMemoryVectorStore` and the `add`, `search` and
`count` calls are unchanged. When `qdrant-client` is not installed the
constructor raises `ImportError`, the part says so and returns; the
extra is `locus-sdk[qdrant]`:

{{ locus_source examples/notebook_52_rag_providers.py::part3_swap_backend }}

## Run it

```bash
python examples/notebook_52_rag_providers.py
```

The script uses no chat model, so `get_model()` and
`LOCUS_MODEL_PROVIDER` play no part here; its only provider call is the
embedder. `get_embedder()` in `examples/config.py` builds
`OpenAIEmbeddings`, and because the script names its models explicitly
(`text-embedding-3-small` and `text-embedding-3-large`),
`LOCUS_EMBEDDING_MODEL` has no effect either; it needs the
`OPENAI_API_KEY` set under Prerequisites and has no mock fallback.
Without the key it prints a skip notice naming the variable and exits
cleanly before any embedding call, so none of the four parts has an
offline run. Part 3 also needs the `qdrant-client` package
(`pip install "locus-sdk[qdrant]"`); when it is missing, that part
prints a notice and the remaining parts still run.

## See also

- [Notebook 51 — RAG basics (the same pipeline over one in-memory store)](notebook_51_rag_basics.md)
- [Notebook 53 — RAG agents (the retriever as an agent tool)](notebook_53_rag_agents.md)
- [Notebook 38 — Oracle 26ai RAG (`OracleVectorStore` as the backend)](notebook_38_oracle_26ai_rag.md)
- [Concepts — RAG](../concepts/rag.md)

## Source

??? note "Full source — `examples/notebook_52_rag_providers.py`"

    ````python
    --8<-- "examples/notebook_52_rag_providers.py"
    ````
