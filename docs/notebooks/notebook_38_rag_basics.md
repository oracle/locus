# RAG Basics

Retrieval-Augmented Generation grounds an agent's answers in your own
documents. This notebook drives the four-step pipeline against the
bundled in-memory vector store.

- **Embed** — `OpenAIEmbeddings` (`text-embedding-3-small`, 1536 dims).
- **Store** — `InMemoryVectorStore` keeps vectors in process; swap in
  `PgVectorStore` / `OpenSearchVectorStore` / `QdrantVectorStore` /
  `ChromaVectorStore` for a durable backend.
- **Search** — nearest-neighbour by cosine distance.
- **Retrieve** — `RAGRetriever` wraps embed + chunk + store behind one
  call.

## Run it

Embeddings need an OpenAI api key:

```bash
export OPENAI_API_KEY=sk-...
python examples/notebook_38_rag_basics.py
```

Offline (skips the live demo cleanly when the key is missing):

```bash
python examples/notebook_38_rag_basics.py
```

## Prerequisites

```bash
export OPENAI_API_KEY=sk-...
```

## Source

```python
--8<-- "examples/notebook_38_rag_basics.py"
```
