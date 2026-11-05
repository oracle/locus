# Oracle in-DB embeddings

Oracle Database 26ai can run the embedding and the similarity search of
a retrieval-augmented generation (RAG) pipeline without the text leaving
the database. An Open Neural Network Exchange (ONNX) model, loaded once
with `DBMS_VECTOR.LOAD_ONNX_MODEL_CLOUD` (Autonomous Database) or
`DBMS_VECTOR.LOAD_ONNX_MODEL` (on-premises), produces vectors
server-side through `DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDING`, and
`VECTOR_DISTANCE` ranks the stored vectors in a second SQL statement
against the same database. The text to embed travels from the
application to the database and nowhere else: no embedding endpoint
outside the database sees it. The guarantee covers embedding and
search; the question and the retrieved passages still go to whichever
chat model the agent is configured with, so pair the pipeline with an
OCI Generative AI model in the same region when end-to-end residency is
required.

`OracleInDBEmbeddings` is the adapter. Pair it with
`OracleVectorStore`, sharing one wallet and one set of connection
settings, and the result is a complete in-database retrieval pipeline.
To drive it the notebook builds a `RAGRetriever` over
`OracleInDBEmbeddings` and `OracleVectorStore` and hands it to an `Agent` through `create_rag_tool(retriever, ..., limit=3)`. Each
retrieval is two SQL statements, both executed inside the database: the
query text goes in once, and only its vector and the ranked rows come
back; there is no external embedding endpoint and no Python embedding
library. The agent is the harness; the in-database embedder and
`VECTOR_DISTANCE` are the subject.

What this notebook covers:

- `OracleInDBEmbeddings(model_name="ALL_MINILM_L12_V2", dimension=384,
  ...)` calling `DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDING` server-side.
- `OracleVectorStore(dimension=384, ...)` as the persistence side of
  the pair, with a native `VECTOR(384, FLOAT32)` column and cosine
  `VECTOR_DISTANCE` search.
- `DBMS_VECTOR.LOAD_ONNX_MODEL_CLOUD` (Autonomous Database) or
  `DBMS_VECTOR.LOAD_ONNX_MODEL` (on-premises) as the one-time step that
  registers the ONNX model inside the database.
- `RAGRetriever(embedder=embedder, store=store)` combining the two,
  exposed to an `Agent` via `create_rag_tool(retriever, ..., limit=3)`:
  each agent retrieval runs two SQL statements inside Oracle Database
  26ai, one to embed the query and one to search.

## Prerequisites

```bash
# Autonomous Database wallet (TLS) + credentials.
export ORACLE_DSN=mydb_low                   # tnsnames alias in the wallet
export ORACLE_USER=locus_app                 # least-privileged app schema
export ORACLE_PASSWORD='<app-password>'
export ORACLE_WALLET=~/.oci/wallets/mydb     # directory holding tnsnames.ora
export ORACLE_WALLET_PASSWORD='<wallet-pw>'  # if wallet is encrypted
export OCI_INDB_MODEL=ALL_MINILM_L12_V2      # optional, defaults to this
export OCI_INDB_DIM=384                      # optional, must match the model
```

The ONNX model must be loaded into the database once, out of band. On
Autonomous Database (ADB), Oracle's pre-built all-MiniLM-L12-v2 model
(384 dimensions) loads directly from Object Storage with
`DBMS_VECTOR.LOAD_ONNX_MODEL_CLOUD`; the download URL is in the Oracle
Machine Learning post [Pre-built Embedding Generation model for Oracle
Database 26ai](https://blogs.oracle.com/machinelearning+selectai/use-our-prebuilt-onnx-model-now-available-for-embedding-generation-in-oracle-database-23ai),
and because it is a pre-authenticated URL the credential is `NULL`:

```sql
BEGIN
    DBMS_VECTOR.LOAD_ONNX_MODEL_CLOUD(
        model_name => 'ALL_MINILM_L12_V2',
        credential => NULL,   -- pre-authenticated URL
        uri        => '<URL of all_MiniLM_L12_v2.onnx from the Oracle Machine Learning blog>');
END;
/
```

On-premises, copy the file to a directory object and load it from there
with `DBMS_VECTOR.LOAD_ONNX_MODEL`:

```sql
BEGIN
    DBMS_VECTOR.LOAD_ONNX_MODEL(
        directory  => 'DM_DUMP',
        file_name  => 'all_MiniLM_L12_v2.onnx',
        model_name => 'ALL_MINILM_L12_V2');
END;
/
```

In either case the application schema then needs `EXECUTE` on
`DBMS_VECTOR_CHAIN` and `SELECT` on the model:

```sql
GRANT EXECUTE ON DBMS_VECTOR_CHAIN TO locus_app;
GRANT SELECT ON MINING MODEL ALL_MINILM_L12_V2 TO locus_app;
```

If any of `ORACLE_DSN`, `ORACLE_USER`, `ORACLE_PASSWORD` or
`ORACLE_WALLET` is unset, the notebook prints the wiring snippet and
exits cleanly. If the model is not loaded, the notebook detects
`ORA-40284` / `ORA-20100` (and the related `ORA-04063` / `ORA-20000`)
from the initial embedding call and prints a skip notice instead of a
traceback.

The drawing follows one agent retrieval across the two lanes, from the
tool call in your process to the ranked rows coming back out of the
database:

{{ locus_diagram indb-embeddings }}

## Key lines

The wiring is in `main`, after the environment check: `OracleInDBEmbeddings`
and `OracleVectorStore` take the same `dsn`, user and wallet; a single
`embed("smoke test")` probes the in-database model before the store,
the retriever and the agent are built; `RAGRetriever(embedder=embedder,
store=store)` joins the two; `add_documents(CORPUS)` seeds the
`locus_notebook_41` table with every passage embedded inside the
database; and `create_rag_tool(retriever, name="search_indb_corpus",
..., limit=3)` is the one tool the `Agent` receives:

{{ locus_source examples/notebook_41_oracle_indb_embeddings.py::main }}

`_drive_agent` reads the agent's event stream: `tool_start` is the query
text crossing into Oracle Database 26ai for `UTL_TO_EMBEDDING` and the
`VECTOR_DISTANCE` search, `tool_complete` brings the ranked hits back
with their scores in `result`, and `terminate` carries the agent's final
answer in `final_message`:

{{ locus_source examples/notebook_41_oracle_indb_embeddings.py::_drive_agent }}

## Run it

```bash
python examples/notebook_41_oracle_indb_embeddings.py
```

The script needs an Oracle Database 26ai reachable through the four
variables in Prerequisites: `ORACLE_DSN`, `ORACLE_USER`,
`ORACLE_PASSWORD` and `ORACLE_WALLET` (`ORACLE_WALLET_PASSWORD` only
when the wallet is encrypted), with the ONNX model loaded and granted
as shown above; `OCI_INDB_MODEL` and `OCI_INDB_DIM` override the model
name and its output width, and must change together. When any of the
four is unset it prints the missing names and the minimal wiring, then
exits cleanly without opening a connection. When the model is not
loaded, the first embedding call raises an `ORA-` error that the
script recognizes and reports as a skip notice. With the database in
place it creates the `locus_notebook_41` table in that schema on first
use and does not drop it, so each run adds the five passages again.

The agent's model comes from `examples/config.py`: OCI Generative AI
when `~/.oci/config` has a usable profile, otherwise the bundled mock
model. Set `LOCUS_MODEL_PROVIDER` (`oci`, `openai`, `anthropic` or
`mock`) to force one. Under the mock model the first turn of each run
still issues a synthesized tool call, so the embedding and the search do
run in the database, while the closing answer is placeholder text rather
than a reading of the returned passages.

## See also

- [Concepts — Oracle Database 26ai, Retrieval](../concepts/oracle-26ai.md#retrieval)
- [Notebook 38 — Oracle 26ai RAG](notebook_38_oracle_26ai_rag.md)
- [Notebook 39 — Oracle ADB loader](notebook_39_oracle_adb_loader.md)
- [Notebook 40 — Oracle in-DB chunker](notebook_40_oracle_indb_chunker.md)
- [Oracle AI Database 26ai — AI Vector Search User's Guide](https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/index.html)
- [`DBMS_VECTOR` PL/SQL package reference (`LOAD_ONNX_MODEL`, `LOAD_ONNX_MODEL_CLOUD`)](https://docs.oracle.com/en/database/oracle/oracle-database/26/arpls/dbms_vector1.html)
- [`DBMS_VECTOR_CHAIN` PL/SQL package reference](https://docs.oracle.com/en/database/oracle/oracle-database/26/arpls/dbms_vector_chain1.html)
- [python-oracledb driver](https://python-oracledb.readthedocs.io/en/latest/index.html)

## Source

??? note "Full source — `examples/notebook_41_oracle_indb_embeddings.py`"

    ````python
    --8<-- "examples/notebook_41_oracle_indb_embeddings.py"
    ````
