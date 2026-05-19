# Cohere Reranker V4

Embedding-only retrieval often isn't enough for production RAG. The
embedding model sees query and document independently and can mis-rank
candidates whose surface form scores high but whose semantic relevance
is lower. A **cross-encoder reranker** scores the query and each
candidate together and catches the signals embeddings miss.

This notebook wires Cohere Reranker V4 on Oracle Cloud Infrastructure
(OCI) Generative AI on top of an Oracle Database 26ai vector store. The
pattern:

1. Embed the corpus once into the 26ai vector store.
2. At query time, cheaply over-fetch a wide candidate set (e.g. top-50)
   from the embedding store.
3. Have the reranker rescore each candidate against the query and
   return the top-N (e.g. top-5).
4. Feed the top-N to the LLM as grounded context.

The demo runs over a small medical corpus where the canonical hepcidin
passage ranks 4th by embedding similarity. The reranker promotes it
to 1st.

## Prerequisites

```bash
# OCI GenAI side — embeddings and reranker.
export OCI_PROFILE=<your-profile>          # api_key or security_token
export OCI_REGION=us-chicago-1
export OCI_COMPARTMENT=ocid1.compartment.oc1..…

# Oracle 26ai side — see notebook 05 for wallet setup.
export ORACLE_DSN=mydb_low
export ORACLE_USER=locus_app
export ORACLE_PASSWORD='<app-password>'
export ORACLE_WALLET=~/.oci/wallets/mydb
```

## Run

```bash
python examples/notebook_05_cohere_reranker.py
```

## See also

- [Notebook 05 — Oracle 26ai RAG (the store this notebook builds on)](notebook_06_oracle_26ai_rag.md)
- [Concepts — RAG](../concepts/rag.md)
- [OCI Generative AI — documentation hub](https://docs.oracle.com/en-us/iaas/Content/generative-ai/home.htm)
- [Oracle AI Database 26ai — AI Vector Search overview](https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/overview-ai-vector-search.html)

## Source

```python
--8<-- "examples/notebook_05_cohere_reranker.py"
```
