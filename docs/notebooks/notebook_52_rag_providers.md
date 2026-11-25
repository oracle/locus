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
  `QdrantVectorStore`, `ChromaVectorStore`, `OracleVectorStore`
  (Oracle Database 26ai), `MilvusVectorStore`, `WeaviateVectorStore`,
  `ElasticsearchVectorStore`, `RedisVectorStore`,
  `MongoDBAtlasVectorStore` or `PineconeVectorStore`. Swapping is a
  one-line change; the retrieve/add API is identical.

The corpus here is a payments-operations runbook — decline codes,
dispute reason codes, refund service-level agreements (SLAs) and
Automated Clearing House (ACH) returns — the same knowledge a support
agent relies on to answer a merchant. Choosing the embedding model and
distance metric is an operations decision: it sets retrieval precision,
and weak retrieval routes the agent to the wrong runbook.

What this notebook covers:

- Part 1 — embedding-model selection: two models of different width on
  the same corpus (`text-embedding-3-small` against
  `text-embedding-3-large` on OpenAI; `cohere.embed-v4.0` against
  `openai.text-embedding-3-small` on OCI Generative AI).
- Part 2 — three `InMemoryVectorStore` instances built with
  `distance_metric` set to `COSINE`, `DOT` and `EUCLIDEAN`, and what the
  store does with those names.
- Part 3 — `QdrantVectorStore(location=":memory:")` as a drop-in for
  `InMemoryVectorStore`.
- Part 4 — batch ingest with `add_documents`, then `count()` and
  `clear()`.
- Part 5 — `MilvusVectorStore` on Milvus Lite: the whole database is one
  file under a temporary directory, no server and no account, created
  with `enable_hybrid=True` so `RAGRetriever(hybrid=True)` blends BM25
  with the vector leg.
- Part 6 — `LLMReranker`: the notebook's chat model grades the
  over-fetched hits and the retriever trims to `top_n`.

## Prerequisites

Embeddings need a provider. Either an OpenAI key:

```bash
export OPENAI_API_KEY=sk-...
```

or an OCI profile, in which case `get_embedder()` builds `OCIEmbeddings`
(`cohere.embed-v4.0`) with the same `OCI_*` variables the chat models use:

```bash
export LOCUS_EMBEDDING_PROVIDER=oci
export OCI_PROFILE=DEFAULT OCI_REGION=us-chicago-1 OCI_COMPARTMENT_ID=ocid1.compartment...
```

Part 5 needs the Milvus extra (`pip install "locus-sdk[milvus]"`) and
Part 6 a chat model (`get_model()` from `examples/config.py`, so
`LOCUS_MODEL_PROVIDER` / `LOCUS_MODEL_ID` apply); the parts say so and
return when either is missing.

The drawing shows the two lanes of the retrieval pipeline, `RAGRetriever.add_documents` for ingest and `retrieve()` for retrieval, sharing one embedder and one vector store. This notebook keeps the lanes fixed and changes what sits in those two boxes:

- Part 1 swaps the embedder between two embedding models.
- Part 2 changes the store's `distance_metric`.
- Parts 3 and 5 replace `InMemoryVectorStore` with `QdrantVectorStore` and `MilvusVectorStore`.
- Parts 2 to 4 also call `add_batch()`, `add()`, `search()`, `count()` and `clear()` on the store directly.

No agent search tool is in the path. Parts 1 to 5 return the ranked hits as they are; Part 6 puts a reranker between the store and the result:

{{ locus_diagram rag-pipeline }}

## Key lines

Part 1 builds the same `RAGRetriever` twice, once per embedding model
from `_embedding_models()` (`LOCUS_EMBEDDING_MODEL` and
`LOCUS_EMBEDDING_MODEL_B` when set, otherwise the provider's default
pair), sizes each `InMemoryVectorStore` from `embedder.config.dimension`
and runs both lanes with `add_documents` and `retrieve`; the only thing
that changes between the two passes is the embedding model:

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

Part 5 is the keyless store: `MilvusVectorStore(uri=…/runbooks.db, enable_hybrid=True)` runs Milvus Lite from a file in a temporary directory and adds a BM25 sparse field next to the dense vector. The retriever is built with `hybrid=True, hybrid_alpha=0.5`, and the same query runs twice through `retrieve(..., hybrid=False)` and `retrieve(..., hybrid=True)`.

The query names an exact decline code, which carries little meaning for an embedding model and is decisive for BM25, so the blended scores differ from the vector-only ones (and the order can, on a larger corpus). `retriever.hybrid_supported` reports whether the store can blend; on a store without hybrid search the same switch falls back to vector-only search and logs it once:

{{ locus_source examples/notebook_52_rag_providers.py::part5_milvus_lite_hybrid }}

Part 6 wires `LLMReranker(get_model(), top_n=2)` into the retriever with
`rerank_candidate_pool=len(CORPUS)`, so every runbook is over-fetched and
graded by the chat model in one completion. Each hit's `.score` is the
model's grade (0 to 1) and `.distance` keeps the vector score the store
returned, so both signals print side by side:

{{ locus_source examples/notebook_52_rag_providers.py::part6_llm_reranker }}

## Run it

```bash
python examples/notebook_52_rag_providers.py
```

Parts 1 to 5 call only the embedder; Part 6 also calls the chat model from `get_model()`, so `LOCUS_MODEL_PROVIDER` and `LOCUS_MODEL_ID` apply there. `get_embedder()` in `examples/config.py` picks the embedding provider from `LOCUS_EMBEDDING_PROVIDER`, or auto-detects OpenAI when `OPENAI_API_KEY` is set and OCI when only an OCI profile is; the model ids come from `LOCUS_EMBEDDING_MODEL` / `LOCUS_EMBEDDING_MODEL_B` or the provider's default pair.

There is no mock fallback for embeddings: without a provider the script prints a skip notice naming the missing variable and exits cleanly before any embedding call.

Part 3 needs the `qdrant-client` package (`pip install "locus-sdk[qdrant]"`) and Part 5 the `pymilvus` package with Milvus Lite (`pip install "locus-sdk[milvus]"`); when one is missing, that part prints a notice and the remaining parts still run.

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
