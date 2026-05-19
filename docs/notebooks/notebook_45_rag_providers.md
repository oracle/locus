# RAG Providers

Production RAG on OCI is two pieces. `OracleVectorStore` is the
default backend across this notebook series — Oracle Database 26ai
with its native `VECTOR(N, FLOAT32)` column and the `VECTOR_DISTANCE`
SQL function. Other backends (pgvector, OpenSearch) are valid
alternatives; the Locus interface is identical.

- **Embeddings** — `OCIEmbeddings` on the OCI GenAI inference endpoint.
  Cohere V3 for English (1024 dims), Cohere V4 for multilingual.
- **Vector store** — `OracleVectorStore` against an Autonomous Database
  26ai. Every section talks to your ADB.

What the four parts cover:

- Part 1 — embedding-model selection (Cohere V3 vs V4 dimensions).
- Part 2 — distance metric choices (COSINE / DOT / EUCLIDEAN).
- Part 3 — attaching to an existing langchain_oracledb-style table via
  column-name overrides.
- Part 4 — batch ingest, `count()`, `clear()`.

## Run it

OCI GenAI is the default for embeddings (auto-detected from
`~/.oci/config`):

```bash
python examples/notebook_45_rag_providers.py
```

Offline (skips the live demo cleanly when env vars are missing):

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_45_rag_providers.py
```

## Prerequisites

```bash
export ORACLE_DSN=mydb_low                   # tnsnames alias
export ORACLE_USER=locus_app                 # least-privileged user
export ORACLE_PASSWORD='<app-password>'
export ORACLE_WALLET=~/.oci/wallets/mydb
export ORACLE_WALLET_PASSWORD='<wallet-pw>'  # if encrypted
export OCI_PROFILE=<your-profile>
export OCI_REGION=us-chicago-1
export OCI_COMPARTMENT=ocid1.compartment.oc1..…
```

## Source

```python
--8<-- "examples/notebook_45_rag_providers.py"
```
