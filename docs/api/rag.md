# RAG

Locus's RAG stack is built around **Oracle Database 26ai** as the default
backend. The 26ai engine ships native `VECTOR(N, FLOAT32)` columns, the
`VECTOR_DISTANCE` SQL function, and the in-database
`DBMS_VECTOR_CHAIN` primitives for embedding generation and chunking —
so the full retrieve-augment loop can live inside the database next to
the source data, without copying vectors out to a sidecar service.

OpenSearch, pgvector, and the in-memory store are also supported as
alternates; the `BaseVectorStore` / `BaseEmbedding` contracts are
identical across backends so you can swap stores with a one-line import
change.

## Oracle 26ai

The recommended path. `OracleVectorStore` opens an async pool against
an Autonomous Database wallet, auto-creates a `VECTOR(N, FLOAT32)`
table on first use, and serves cosine / dot / Euclidean similarity via
`VECTOR_DISTANCE`. `OracleInDBEmbeddings` and `OracleInDBChunker` run
the embedding and chunking entirely inside the database via
`DBMS_VECTOR_CHAIN`, eliminating the round-trip to an external embed
service. `OracleADBLoader` ingests documents directly from ADB tables.

::: locus.rag.stores.oracle.OracleVectorStore
::: locus.rag.embeddings.oracle_indb.OracleInDBEmbeddings
::: locus.rag.chunkers.oracle_indb.OracleInDBChunker
::: locus.rag.loaders.oracle.OracleADBLoader

## Retriever

The unified interface — combines an embedder and a store into the
`retrieve()` call that returns ranked, optionally reranked results.

::: locus.rag.retriever.RAGRetriever
::: locus.rag.retriever.RetrievalResult

## Embeddings

`OCIEmbeddings` is the default external embedder against OCI GenAI's
Cohere V3 (1024 dims, English) and V4 (multilingual) endpoints.
`OpenAIEmbeddings` covers the `text-embedding-3-*` family. For
production on Oracle 26ai prefer `OracleInDBEmbeddings` (above) so the
vectors stay in the database.

### Providers

::: locus.rag.embeddings.oci.OCIEmbeddings
::: locus.rag.embeddings.openai.OpenAIEmbeddings

### Embeddings base contract

::: locus.rag.embeddings.base.BaseEmbedding
::: locus.rag.embeddings.base.EmbeddingConfig
::: locus.rag.embeddings.base.EmbeddingProvider
::: locus.rag.embeddings.base.EmbeddingResult

## Vector stores

### Alternate backends

Use these when you can't run on Oracle 26ai — otherwise prefer the
`OracleVectorStore` block above.

::: locus.rag.stores.opensearch.OpenSearchVectorStore
::: locus.rag.stores.pgvector.PgVectorStore
::: locus.rag.stores.memory.InMemoryVectorStore

### Vector store base contract

::: locus.rag.stores.base.BaseVectorStore
::: locus.rag.stores.base.VectorStore
::: locus.rag.stores.base.VectorStoreConfig
::: locus.rag.stores.base.Document
::: locus.rag.stores.base.SearchResult

## Reranker

Re-score candidates after the initial vector search. Cohere V4 rerank
is the production default; the `Reranker` Protocol lets you plug in any
scorer.

::: locus.rag.reranker.base.Reranker
::: locus.rag.reranker.cohere_oci.CohereReranker

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
