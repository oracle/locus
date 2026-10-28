# Oracle in-DB embeddings

Oracle Database 26ai can run the entire embedding and search step
without leaving the database. An ONNX model loaded once via
`DBMS_VECTOR.LOAD_ONNX_MODEL` produces vectors server-side through
`DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDING`, and `VECTOR_DISTANCE` does the
cosine search in a second SQL statement against the same database.
Text sent for embedding never leaves the database: the data-residency
guarantee is that the data never crossed the boundary, not that it
was encrypted on the way out.

`OracleInDBEmbeddings` is the Locus adapter. Pair it with
`OracleVectorStore` (sharing one wallet, one pool kwargs envelope)
and you have a complete in-DB retrieval pipeline. One agent
retrieval, from the tool call to the ranked rows:

{{ locus_diagram indb-embeddings }}

To drive the pipeline the notebook hands a `RAGRetriever(embedder=
OracleInDBEmbeddings, store=OracleVectorStore)` to a Locus `Agent` via
`create_rag_tool(retriever)`. Each retrieval is two SQL statements,
both executed inside the database: the query text goes in once, and
only the vector and the ranked rows come back — no external embedding
endpoint, no Python embedding library. The agent is the harness; the
in-DB embedder and `VECTOR_DISTANCE` are the subject.

## What this covers

- `OracleInDBEmbeddings(model_name="ALL_MINILM_L12_V2", dimension=384,
  ...)` calling `DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDING` server-side.
- `OracleVectorStore(dimension=384, ...)` as the persistence side of
  the pair, with a native `VECTOR(384, FLOAT32)` column and cosine
  `VECTOR_DISTANCE` search.
- `DBMS_VECTOR.LOAD_ONNX_MODEL` as the one-time bootstrap to register
  the ONNX model inside the database.
- `RAGRetriever(embedder=indb, store=store)` combining the two, then
  exposed to an `Agent` via `create_rag_tool(retriever)` — each
  agent retrieval runs two SQL statements inside 26ai, one to embed
  the query and one to search.

## Prerequisites

```bash
# Autonomous Database wallet (TLS) + credentials.
export ORACLE_DSN=mydb_low                   # tnsnames alias in the wallet
export ORACLE_USER=locus_app                 # least-privileged app schema
export ORACLE_PASSWORD='<app-password>'
export ORACLE_WALLET=~/.oci/wallets/mydb     # directory holding tnsnames.ora
export ORACLE_WALLET_PASSWORD='<wallet-pw>'  # if wallet is encrypted
export OCI_INDB_MODEL=ALL_MINILM_L12_V2      # optional, defaults to this
```

The ONNX model must be loaded into the DB once, out-of-band:

```sql
BEGIN
    DBMS_VECTOR.LOAD_ONNX_MODEL(
        directory  => 'DM_DUMP',
        file_name  => 'all_MiniLM_L12_v2.onnx',
        model_name => 'ALL_MINILM_L12_V2');
END;
/

GRANT EXECUTE ON DBMS_VECTOR_CHAIN TO locus_app;
GRANT SELECT ON MINING MODEL ALL_MINILM_L12_V2 TO locus_app;
```

If any of `ORACLE_DSN`, `ORACLE_USER`, `ORACLE_PASSWORD` or
`ORACLE_WALLET` is unset, the notebook prints the wiring snippet and
exits cleanly. If the model is not loaded, the notebook detects
`ORA-40284` / `ORA-20100` (and the related `ORA-04063` / `ORA-20000`)
from the initial embedding call and prints a skip notice instead of a
traceback.

## Run

```bash
python examples/notebook_41_oracle_indb_embeddings.py
```

## See also

- [Notebook 38 — Oracle 26ai RAG](notebook_38_oracle_26ai_rag.md)
- [Notebook 39 — Oracle ADB document loader](notebook_39_oracle_adb_loader.md)
- [Notebook 40 — Oracle in-DB chunker](notebook_40_oracle_indb_chunker.md)
- [Oracle AI Database 26ai — AI Vector Search User's Guide](https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/index.html)
- [`DBMS_VECTOR_CHAIN` PL/SQL package reference](https://docs.oracle.com/en/database/oracle/oracle-database/26/arpls/dbms_vector_chain1.html)
- [python-oracledb driver](https://python-oracledb.readthedocs.io/en/latest/index.html)

## Source

```python
--8<-- "examples/notebook_41_oracle_indb_embeddings.py"
```
