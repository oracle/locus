# RAG Agents

Once you have documents in a vector store (notebook 38 / 39), the next
step is to let an agent reach into it. `RAGRetriever.as_tool()` turns
the retriever into an ordinary Locus tool the agent picks up alongside
any other `@tool` you define.

- `retriever.as_tool(name, description)` — convert a retriever into a
  callable tool for the agent.
- Single-tool Q&A agent against a product knowledge base.
- Mixed tool set — RAG search alongside a calculator and a date tool.
- Streaming events from the agent while it searches and answers.
- Best-practice notes on chunk size, prompt design, and metadata
  filters.

Backend: `OracleVectorStore` is the default — Oracle Database 26ai's
native `VECTOR` column and `VECTOR_DISTANCE` SQL function. Swap
`_oracle_store` for any other Locus vector store (OpenSearch,
pgvector, in-memory).

## Run it

OCI GenAI is the default (auto-detected from `~/.oci/config`):

```bash
LOCUS_MODEL_ID=openai.gpt-4.1 python examples/notebook_46_rag_agents.py
```

Offline (skips the live demo cleanly when env vars are missing):

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_46_rag_agents.py
```

## Prerequisites

```bash
export ORACLE_DSN=mydb_low                   # tnsnames alias
export ORACLE_USER=locus_app
export ORACLE_PASSWORD='<app-password>'
export ORACLE_WALLET=~/.oci/wallets/mydb
export ORACLE_WALLET_PASSWORD='<wallet-pw>'  # if encrypted
export OCI_COMPARTMENT=ocid1.compartment.oc1..…
```

## Source

```python
--8<-- "examples/notebook_46_rag_agents.py"
```
