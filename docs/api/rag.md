# RAG

Tulip's RAG stack is built around pluggable embedders and vector
stores behind one interface. The `BaseVectorStore` / `BaseEmbedding`
contracts are identical across backends so you can swap stores with a
one-line import change.

## Retriever

The unified interface — combines an embedder and a store into the
`retrieve()` call that returns ranked, optionally reranked results.

::: tulip.rag.retriever.RAGRetriever
::: tulip.rag.retriever.RetrievalResult

## Embeddings

`OpenAIEmbeddings` covers the `text-embedding-3-*` family;
`CohereEmbeddings` wraps Cohere's direct embeddings API.

### Providers

::: tulip.rag.embeddings.openai.OpenAIEmbeddings
::: tulip.rag.embeddings.cohere.CohereEmbeddings

### Embeddings base contract

::: tulip.rag.embeddings.base.BaseEmbedding
::: tulip.rag.embeddings.base.EmbeddingConfig
::: tulip.rag.embeddings.base.EmbeddingProvider
::: tulip.rag.embeddings.base.EmbeddingResult

## Vector stores

### Backends

::: tulip.rag.stores.memory.InMemoryVectorStore
::: tulip.rag.stores.pgvector.PgVectorStore
::: tulip.rag.stores.opensearch.OpenSearchVectorStore
::: tulip.rag.stores.qdrant.QdrantVectorStore
::: tulip.rag.stores.chroma.ChromaVectorStore

### Vector store base contract

::: tulip.rag.stores.base.BaseVectorStore
::: tulip.rag.stores.base.VectorStore
::: tulip.rag.stores.base.VectorStoreConfig
::: tulip.rag.stores.base.Document
::: tulip.rag.stores.base.SearchResult

## Reranker

Re-score candidates after the initial vector search. The `Reranker`
Protocol lets you plug in any scorer; two implementations ship:
`CrossEncoderReranker` (local, offline) and `CohereReranker` (Cohere's
direct rerank API).

::: tulip.rag.reranker.base.Reranker
::: tulip.rag.reranker.cross_encoder.CrossEncoderReranker
::: tulip.rag.reranker.cohere.CohereReranker

## Multimodal processing

Convert non-text inputs (PDF text + OCR, image OCR, audio
transcription) into the same `Document` shape the retriever consumes.

::: tulip.rag.multimodal.ContentType
::: tulip.rag.multimodal.MultimodalProcessor
::: tulip.rag.multimodal.ProcessedContent
::: tulip.rag.multimodal.process_content

## Tool wiring

Expose a retriever as an agent tool so the model can call it like any
other function. Use `create_rag_tool` for a one-shot retrieval call
and `create_rag_context_tool` when you want the agent to inject
retrieved context into its own response.

::: tulip.rag.tools.RAGToolkit
::: tulip.rag.tools.create_rag_tool
::: tulip.rag.tools.create_rag_context_tool
