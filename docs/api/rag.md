# RAG

Retrieval-augmented generation (RAG): the SDK's RAG stack is built
around pluggable embedders and vector stores behind one interface. The `BaseVectorStore` / `BaseEmbedding`
contracts are identical across backends so you can swap stores with a
one-line import change.

For the concepts, start with [RAG](../concepts/rag.md).

## Retriever

The unified interface — combines an embedder and a store into the
`retrieve()` call that returns ranked, optionally reranked results.

::: locus.rag.retriever.RAGRetriever
::: locus.rag.retriever.RetrievalResult

## Embeddings

`OCIEmbeddings` runs Cohere's `embed-*` models on OCI Generative AI;
`OpenAIEmbeddings` covers the `text-embedding-3-*` family;
`CohereEmbeddings` wraps Cohere's direct embeddings API. `OracleInDBEmbeddings`
(below) embeds *inside* Oracle Database 26ai.

### Providers

::: locus.rag.embeddings.oci.OCIEmbeddings
::: locus.rag.embeddings.openai.OpenAIEmbeddings
::: locus.rag.embeddings.cohere.CohereEmbeddings

### Embeddings base contract

::: locus.rag.embeddings.base.BaseEmbedding
::: locus.rag.embeddings.base.EmbeddingConfig
::: locus.rag.embeddings.base.EmbeddingProvider
::: locus.rag.embeddings.base.EmbeddingResult

## Vector stores

### Backends

::: locus.rag.stores.oracle.OracleVectorStore
::: locus.rag.stores.memory.InMemoryVectorStore
::: locus.rag.stores.pgvector.PgVectorStore
::: locus.rag.stores.opensearch.OpenSearchVectorStore
::: locus.rag.stores.qdrant.QdrantVectorStore
::: locus.rag.stores.chroma.ChromaVectorStore
::: locus.rag.stores.milvus.MilvusVectorStore
::: locus.rag.stores.weaviate.WeaviateVectorStore
::: locus.rag.stores.elasticsearch.ElasticsearchVectorStore
::: locus.rag.stores.redis.RedisVectorStore
::: locus.rag.stores.mongodb.MongoDBAtlasVectorStore
::: locus.rag.stores.pinecone.PineconeVectorStore
::: locus.rag.stores.heatwave.HeatWaveVectorStore

### Oracle Database 26ai

The Oracle-native path. `OracleVectorStore` opens an async pool against
an Autonomous Database wallet, auto-creates a `VECTOR(N, FLOAT32)`
table on first use, and serves cosine / dot / Euclidean similarity via
`VECTOR_DISTANCE`. `OracleInDBEmbeddings` and `OracleInDBChunker` run
the embedding and chunking entirely inside the database via
`DBMS_VECTOR_CHAIN`, eliminating the round-trip to an external embed
service. `OracleADBLoader` ingests documents directly from ADB tables.
Sync companions (`OracleSyncVectorStore`, `OracleSyncInDBEmbeddings`,
`OracleSyncInDBChunker`, `OracleSyncADBLoader`) wrap the same classes for
synchronous code. All need the `oracledb` extra.

::: locus.rag.embeddings.oracle_indb.OracleInDBEmbeddings
::: locus.rag.chunkers.oracle_indb.OracleInDBChunker
::: locus.rag.loaders.oracle.OracleADBLoader

### MySQL HeatWave

`HeatWaveVectorStore` keeps documents in one InnoDB table: a `VECTOR(n)`
column for the embedding, a `JSON` column for the metadata and a `FULLTEXT`
index on the content. It talks plain MySQL through Connector/Python (the
`heatwave` extra — the same package as the `mysql` checkpoint extra) and
probes the server once to decide how to search:

| | HeatWave MySQL (MySQL HeatWave on OCI, MySQL AI) | Community / Commercial MySQL 9+ |
|---|---|---|
| `DISTANCE()` | Yes — similarity is computed and ordered on the server. | No (`ERROR 1305`) — the store fetches up to `candidate_limit` rows (after the metadata filter) and ranks them client-side. |
| `server_side_distance` | `True` | `False` |
| Vector index | A HeatWave cluster accelerates `DISTANCE()` once the table is loaded (`await store.load_into_heatwave()`). | None; every search is a bounded scan. |
| Metrics | `COSINE`, `DOT`, `EUCLIDEAN`. `DISTANCE(..., 'DOT')` returns the dot product itself, so the store negates it: `SearchResult.distance` is lower-is-closer on every metric. | The same three, computed the same way. |
| Metadata filters | Equality on `JSON_EXTRACT(metadata, path) = CAST(value AS JSON)` with the path and the value bound as parameters, on both. | |
| Hybrid search | `MATCH ... AGAINST` in natural language mode on the `FULLTEXT` index, blended with the vector leg client-side, on both. | |

The outcome is one `INFO` log line per store instance and the
`server_side_distance` property; pass `server_side_distance=True` or
`False` to skip the probe. The community path exists so the same code runs
against a laptop MySQL; it is a full scan and suits small corpora only — the
credible "AI on MySQL" retrieval path is HeatWave. The shared store
conformance suite runs against either server behind `HEATWAVE_DSN`
(`tests/integration/rag/test_store_conformance_live.py`).

```python
from locus.rag.stores import HeatWaveVectorStore

store = HeatWaveVectorStore(
    host="10.0.1.5", database="kb", user="locus", password="...", dimension=1024
)
await store.ensure_ready()
print(store.server_side_distance)  # True on HeatWave MySQL
```

### Vector store base contract

::: locus.rag.stores.base.BaseVectorStore
::: locus.rag.stores.base.VectorStore
::: locus.rag.stores.base.VectorStoreConfig
::: locus.rag.stores.base.Document
::: locus.rag.stores.base.SearchResult
::: locus.rag.stores.base.HybridSearchable
::: locus.rag.stores.base.supports_hybrid_search

## Reranker

Re-score candidates after the initial vector search. The `Reranker`
Protocol lets you plug in any scorer; six implementations ship:
`CrossEncoderReranker` (local, offline), `CohereReranker` (Cohere's
direct rerank API), `OCICohereReranker` (Cohere `rerank-v3.5` /
`rerank-v4.0` served by OCI Generative AI), `LLMReranker` (any
registered chat model grades the candidates), `JinaReranker` and
`VoyageReranker` (hosted rerank APIs).

::: locus.rag.reranker.base.Reranker
::: locus.rag.reranker.cross_encoder.CrossEncoderReranker
::: locus.rag.reranker.cohere.CohereReranker
::: locus.rag.reranker.cohere_oci.OCICohereReranker
::: locus.rag.reranker.llm.LLMReranker
::: locus.rag.reranker.jina.JinaReranker
::: locus.rag.reranker.voyage.VoyageReranker

## Multimodal processing

Convert non-text inputs (PDF text + OCR, image OCR, audio
transcription) into the same `Document` shape the retriever consumes.

::: locus.rag.multimodal.ContentType
::: locus.rag.multimodal.MultimodalProcessor
::: locus.rag.multimodal.ProcessedContent
::: locus.rag.multimodal.process_content

## Tool wiring

Expose a retriever as an agent tool so the model can call it like any
other function. Use `create_rag_tool` for a one-shot retrieval call
and `create_rag_context_tool` when you want the agent to inject
retrieved context into its own response.

::: locus.rag.tools.RAGToolkit
::: locus.rag.tools.create_rag_tool
::: locus.rag.tools.create_rag_context_tool
