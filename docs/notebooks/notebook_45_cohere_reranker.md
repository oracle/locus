# Cohere Reranker V4

Embedding-only retrieval is often not enough for production
retrieval-augmented generation (RAG). The embedding model sees query and
document independently and can mis-rank candidates whose surface form
scores high but whose semantic relevance is lower. A **cross-encoder
reranker** scores the query and each candidate together and catches the
signals embeddings miss.

This notebook wires Cohere Reranker V4 on Oracle Cloud Infrastructure
(OCI) Generative AI on top of an Oracle Database 26ai vector store and
compares two retrievers over the same store: a baseline that ranks by
embedding similarity alone, and a second that over-fetches the candidate
pool and lets the reranker rescore it. Retrieve-then-rerank has four
steps: embed the corpus once into the vector store; at query time,
over-fetch a wide candidate set cheaply from the store; have the reranker
rescore each candidate against the query and keep the top-N; then feed the
top-N to the model as grounded context. This example performs the first
three steps and prints the ranks; it has no chat model.
[Notebook 38](notebook_38_oracle_26ai_rag.md) shows an agent consuming
hits from the same kind of store.

The run reports where the canonical hepcidin passage lands in each pass,
so any lift from reranking shows as a change of rank; when the passage
falls outside the four baseline hits, its baseline position prints as
`None`.

What this notebook covers:

- `OCIEmbeddings` producing 1536-dimension Cohere `embed-v4.0` vectors on
  OCI Generative AI, shared by both retrievers.
- `OracleVectorStore` connecting through an Autonomous Database wallet and
  creating the `locus_notebook_45` table, with its `VECTOR(1536, FLOAT32)`
  column and Hierarchical Navigable Small World (HNSW) vector index, on
  first use (the store default `auto_create_table=True`).
- A baseline `RAGRetriever(embedder=..., store=...)` whose `retrieve()`
  returns the rows in the store's cosine-similarity order.
- `OCICohereReranker(model="cohere.rerank-v4.0-fast", top_n=4, ...)` wired
  through `RAGRetriever(reranker=..., rerank_candidate_pool=len(CORPUS))`,
  so `retrieve()` over-fetches the candidate pool from the store, sends it
  to the OCI Generative AI `rerank_text` call and keeps the top `limit`
  hits.
- A rank report that locates the canonical passage in both result lists.

## Prerequisites

```bash
# OCI Generative AI side — embeddings and reranker.
export OCI_PROFILE=<your-profile>
export OCI_AUTH_TYPE=api_key               # or security_token
export OCI_REGION=us-chicago-1
export OCI_COMPARTMENT=ocid1.compartment.oc1..…

# Oracle Database 26ai side — see notebook 38 for wallet setup.
export ORACLE_DSN=mydb_low
export ORACLE_USER=locus_app
export ORACLE_PASSWORD='<app-password>'
export ORACLE_WALLET=~/.oci/wallets/mydb     # mTLS only; a TLS-only database has none
export ORACLE_WALLET_PASSWORD='<wallet-pw>'  # if the wallet is encrypted
# TLS-only (no wallet): a connect string in ORACLE_DSN, or an alias + ORACLE_CONFIG_DIR
```

Both lanes of the drawing run here against one `OCIEmbeddings` embedder
and one `OracleVectorStore`, driven directly rather than through an
agent's search tool: `RAGRetriever.add_documents()` is the ingest lane,
embedding the eight passages and writing them to the
`VECTOR(1536, FLOAT32)` table, and `RAGRetriever.retrieve()` is the
retrieve lane, where the query is embedded with the same embedder, Oracle
Database 26ai ranks the rows with `VECTOR_DISTANCE` and, in the second
retriever only, the ranked hits pass through `OCICohereReranker` before
the `RetrievalResult` returns:

{{ locus_diagram rag-pipeline }}

## Key lines

The example keeps the comparison in `main`. Read it from
`baseline = RAGRetriever(embedder=embedder, store=store)`, which retrieves
by embedding similarity alone, to `reranked_retriever`, which adds
`reranker=OCICohereReranker(model="cohere.rerank-v4.0-fast", top_n=4, ...)`
and `rerank_candidate_pool=len(CORPUS)`, so that `retrieve(QUERY, limit=4)`
fetches as many hits from the store as the corpus has passages, has the
cross-encoder rescore them and returns the top four. The closing block
reports the canonical hepcidin passage's position in each list; the two
`score` columns are on different scales, the store's similarity
`1 - distance / 2` over the cosine distance for the baseline and the
reranker's `relevance_score` after reranking:

{{ locus_source examples/notebook_45_cohere_reranker.py::main }}

## Run it

```bash
python examples/notebook_45_cohere_reranker.py
```

The script needs two live services and no chat model, so
`examples/config.py` and `LOCUS_MODEL_PROVIDER` play no part here.
Oracle Database 26ai is reached
through `ORACLE_DSN`, `ORACLE_USER` and `ORACLE_PASSWORD` from Prerequisites,
plus `ORACLE_WALLET` for mutual TLS (`ORACLE_WALLET_PASSWORD` when the wallet
is encrypted) or `ORACLE_CONFIG_DIR` when a TLS-only database is named by a
tnsnames alias; each run appends the eight passages to a `locus_notebook_45`
table in that schema, created on first use. Both the Cohere `embed-v4.0`
embeddings and the `rerank-v4.0-fast` reranker come from OCI Generative
AI, which needs `OCI_COMPARTMENT`; `OCI_PROFILE`, `OCI_REGION` and
`OCI_AUTH_TYPE` default to `DEFAULT`, `us-chicago-1` and `api_key`, and
the `LOCUS_OCI_*` equivalents take precedence when set. If any of the
five required variables is missing, the script prints a skip notice
listing them and exits cleanly without opening a connection.

## See also

- [Notebook 38 — Oracle 26ai RAG (the store this notebook builds on)](notebook_38_oracle_26ai_rag.md)
- [Concepts — RAG](../concepts/rag.md)
- [Concepts — Oracle Database 26ai](../concepts/oracle-26ai.md)
- [OCI Generative AI — documentation hub](https://docs.oracle.com/en-us/iaas/Content/generative-ai/home.htm)
- [Oracle AI Database 26ai — AI Vector Search overview](https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/overview-ai-vector-search.html)

## Source

??? note "Full source — `examples/notebook_45_cohere_reranker.py`"

    ````python
    --8<-- "examples/notebook_45_cohere_reranker.py"
    ````
