# RAG Agents

Once you have documents in a vector store (notebook 38 / 39), the next
step is to let an agent reach into it. `RAGRetriever.as_tool()` turns
the retriever into an ordinary Tulip tool the agent picks up alongside
any other `@tool` you define.

- `retriever.as_tool(name, description)` — convert a retriever into a
  callable tool for the agent.
- Single-tool Q&A agent against a product knowledge base.
- Mixed tool set — RAG search alongside a calculator and a date tool.
- Streaming events from the agent while it searches and answers.
- Best-practice notes on chunk size, prompt design, and metadata
  filters.

Backend: `InMemoryVectorStore` keeps the demo dependency-free. Swap
`_make_store` for any other Tulip vector store (pgvector, OpenSearch,
Qdrant, Chroma) for a durable backend.

## Run it

Embeddings need an OpenAI api key:

```bash
export OPENAI_API_KEY=sk-...
python examples/notebook_40_rag_agents.py
```

Offline (skips the live demo cleanly when the key is missing):

```bash
python examples/notebook_40_rag_agents.py
```

## Prerequisites

```bash
export OPENAI_API_KEY=sk-...
```

## Source

```python
--8<-- "examples/notebook_40_rag_agents.py"
```
