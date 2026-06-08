# RAG Providers

Production RAG is two pluggable pieces, both behind one Tulip
interface.

- **Embeddings** — `OpenAIEmbeddings` (`text-embedding-3-small` /
  `-3-large`) or `CohereEmbeddings` (Cohere's direct API).
- **Vector store** — `InMemoryVectorStore` for demos, or a durable
  backend: `PgVectorStore`, `OpenSearchVectorStore`,
  `QdrantVectorStore`, `ChromaVectorStore`. Swapping is one line; the
  retrieve/add API is identical.

What the four parts cover:

- Part 1 — embedding-model selection (small vs large dimensions).
- Part 2 — distance metric choices (COSINE / DOT / EUCLIDEAN).
- Part 3 — Qdrant in-memory store as a drop-in for InMemoryVectorStore.
- Part 4 — batch ingest, `count()`, `clear()`.

## Run it

Embeddings need an OpenAI api key:

```bash
export OPENAI_API_KEY=sk-...
python examples/notebook_39_rag_providers.py
```

Offline (skips the live demo cleanly when the key is missing):

```bash
python examples/notebook_39_rag_providers.py
```

## Prerequisites

```bash
export OPENAI_API_KEY=sk-...
```

## Source

```python
--8<-- "examples/notebook_39_rag_providers.py"
```
