# RAG Basics

Retrieval-Augmented Generation grounds an agent's answers in your own
documents. This notebook drives the four-step pipeline against a real
26ai instance.

- **Embed** — `OCIEmbeddings` (Cohere V3 English, 1024 dims) on the OCI
  GenAI inference endpoint.
- **Store** — `OracleVectorStore` writes to 26ai's native
  `VECTOR(N, FLOAT32)` column — no extension required.
- **Search** — `VECTOR_DISTANCE` is a first-class SQL function in 26ai.
  The native type plus operator are differentiators.
- **Retrieve** — `RAGRetriever` wraps embed + chunk + store behind one
  call.

## Run it

OCI GenAI is the default for embeddings (auto-detected from
`~/.oci/config`):

```bash
python examples/notebook_44_rag_basics.py
```

Offline (skips the live demo cleanly when env vars are missing):

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_44_rag_basics.py
```

## Prerequisites

Provision an Autonomous Database 26ai and an OCI GenAI compartment,
then export:

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
--8<-- "examples/notebook_44_rag_basics.py"
```
