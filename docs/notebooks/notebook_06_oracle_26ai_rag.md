# Oracle 26ai RAG agent

A Locus `Agent` that retrieves over Oracle Database 26ai. The agent
runtime is the protagonist; Oracle 26ai is the durable substrate.
Embeddings live in a real `VECTOR(N, FLOAT32)` column, similarity is
the `VECTOR_DISTANCE` SQL function, and the index is `CREATE VECTOR
INDEX ... ORGANIZATION NEIGHBOR PARTITIONS WITH DISTANCE COSINE` — but
what this notebook shows is the *agent* deciding when to search,
firing a real tool call, and grounding its reply in what came back.

## What this covers

- `OCIEmbeddings` producing 1024-dim Cohere V3 vectors on OCI GenAI.
- `OracleVectorStore` opening an async pool against an Autonomous
  Database wallet, auto-creating the table with a native
  `VECTOR(1024, FLOAT32)` column on first use, and serving cosine
  similarity queries via `VECTOR_DISTANCE`.
- `create_rag_tool(retriever, ...)` turning the `RAGRetriever` into a
  Locus `@tool` and handing it to an `Agent`. Each retrieval surfaces
  as `ToolStartEvent` / `ToolCompleteEvent` so the agent's
  decision-making is visible.
- The same agent, the same tool — swap the store import to OpenSearch,
  pgvector, or in-memory and nothing else changes.

## Prerequisites

```bash
# Database side — Autonomous Database wallet (TLS) + credentials.
export ORACLE_DSN=mydb_low                   # tnsnames alias in the wallet
export ORACLE_USER=locus_app                 # least-privileged app schema
export ORACLE_PASSWORD='<app-password>'
export ORACLE_WALLET=~/.oci/wallets/mydb     # directory holding tnsnames.ora
export ORACLE_WALLET_PASSWORD='<wallet-pw>'  # if wallet is encrypted

# Embedding side — OCI GenAI on us-chicago-1.
export OCI_PROFILE=<your-profile>
export OCI_REGION=us-chicago-1
export OCI_COMPARTMENT=ocid1.compartment.oc1..…
export OCI_AUTH_TYPE=api_key                 # or security_token
```

If `ORACLE_DSN` / `ORACLE_PASSWORD` / `OCI_COMPARTMENT` aren't set the
notebook prints the wiring snippet and exits cleanly — no traceback,
no half-initialised state.

## Run

```bash
python examples/notebook_06_oracle_26ai_rag.py
```

## Schema hygiene

This notebook uses `auto_create_table=True` so the demo provisions the
table on first run. For production, create the table out-of-band as a
least-privileged app schema owner and set `auto_create_table=False` so
the runtime user is restricted to DML only — see the
[Production setup section in the RAG concepts page](../concepts/rag.md#3-wire-the-retriever)
for the `CREATE USER locus_app` / `CREATE VECTOR INDEX` DDL.

## See also

- [Concepts — RAG (Oracle 26ai walkthrough)](../concepts/rag.md)
- [Notebook 05 — Cohere Reranker V4 over Oracle 26ai](notebook_05_cohere_reranker.md)
- [Oracle Database 26ai — documentation](https://docs.oracle.com/en/database/oracle/oracle-database/23/index.html)
- [Oracle AI Database 26ai — AI Vector Search User's Guide](https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/index.html)
- [Oracle Autonomous Database — documentation](https://docs.oracle.com/en/cloud/paas/autonomous-database/index.html)
- [python-oracledb driver](https://python-oracledb.readthedocs.io/en/latest/index.html)
- [OCI Generative AI — documentation hub](https://docs.oracle.com/en-us/iaas/Content/generative-ai/home.htm)

## Source

```python
--8<-- "examples/notebook_06_oracle_26ai_rag.py"
```
