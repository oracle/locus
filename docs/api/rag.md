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

### Vector store base contract

::: locus.rag.stores.base.BaseVectorStore
::: locus.rag.stores.base.VectorStore
::: locus.rag.stores.base.VectorStoreConfig
::: locus.rag.stores.base.Document
::: locus.rag.stores.base.SearchResult

## Reranker

Re-score candidates after the initial vector search. The `Reranker`
Protocol lets you plug in any scorer; three implementations ship:
`CrossEncoderReranker` (local, offline), `CohereReranker` (Cohere's
direct rerank API) and `OCICohereReranker` (Cohere `rerank-v3.5` /
`rerank-v4.0` served by OCI Generative AI).

::: locus.rag.reranker.base.Reranker
::: locus.rag.reranker.cross_encoder.CrossEncoderReranker
::: locus.rag.reranker.cohere.CohereReranker
::: locus.rag.reranker.cohere_oci.OCICohereReranker

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
