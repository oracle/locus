# Oracle In-DB Embeddings — `DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDING`

Oracle Database 26ai can run the **entire** embed + search step
without leaving the database. An ONNX model loaded once via
`DBMS_VECTOR.LOAD_ONNX_MODEL` produces vectors server-side through
`DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDING`, and `VECTOR_DISTANCE` does the
cosine search in the same SQL round-trip. Text-in-flight never leaves
the database at embedding time — the data-residency story is "the
data never crossed the boundary," not "we encrypted it on the way
out."

`OracleInDBEmbeddings` is the Locus adapter. Pair it with
`OracleVectorStore` (sharing one wallet, one pool kwargs envelope)
and you have a complete in-DB retrieval pipeline:

```text
text  ──▶  UTL_TO_EMBEDDING (server-side ONNX)  ──▶  VECTOR(384, FLOAT32)
                                                            │
                                                            ▼
                                                VECTOR_DISTANCE COSINE
```

To drive the pipeline the notebook hands a `RAGRetriever(embedder=
OracleInDBEmbeddings, store=OracleVectorStore)` to a Locus `Agent` via
`create_rag_tool(retriever)`. Each retrieval is one SQL round-trip —
no external embedding endpoint, no Python embedding library. The
agent is the harness; the in-DB embedder + `VECTOR_DISTANCE` are the
subject.

## What this covers

- `OracleInDBEmbeddings(model_name="ALL_MINILM_L12_V2", dimension=384,
  ...)` calling `DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDING` server-side.
- `OracleVectorStore(dimension=384, ...)` as the persistence side of
  the pair, with a native `VECTOR(384, FLOAT32)` column and cosine
  `VECTOR_DISTANCE` search.
- `DBMS_VECTOR.LOAD_ONNX_MODEL` as the one-time bootstrap to register
  the ONNX model inside the database.
- `RAGRetriever(embedder=indb, store=store)` glued together, then
  exposed to an `Agent` via `create_rag_tool(retriever)` — each
  agent retrieval = one SQL round-trip that embeds + searches inside
  26ai.

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
GRANT MINING MODEL SELECT ON ALL_MINILM_L12_V2 TO locus_app;
```

If `ORACLE_DSN` / `ORACLE_PASSWORD` / `ORACLE_WALLET` aren't set the
notebook prints the wiring snippet and exits cleanly. If the model
isn't loaded the notebook catches `ORA-29024` / `ORA-20100` / similar
from the SQL call and prints a friendly skip — no traceback in either
case.

## Run

```bash
python examples/notebook_44_oracle_indb_embeddings.py
```

## See also

- [Notebook 06 — Oracle 26ai RAG](notebook_41_oracle_26ai_rag.md)
- [Notebook 08 — Oracle ADB document loader](notebook_42_oracle_adb_loader.md)
- [Notebook 09 — Oracle in-DB chunker](notebook_43_oracle_indb_chunker.md)
- [Oracle AI Database 26ai — AI Vector Search User's Guide](https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/index.html)
- [`DBMS_VECTOR_CHAIN` PL/SQL package reference](https://docs.oracle.com/en/database/oracle/oracle-database/23/arpls/dbms_vector_chain1.html)
- [python-oracledb driver](https://python-oracledb.readthedocs.io/en/latest/index.html)

## Source

```python
--8<-- "examples/notebook_44_oracle_indb_embeddings.py"
```
