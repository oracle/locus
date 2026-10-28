# Oracle In-DB Chunker — `DBMS_VECTOR_CHAIN.UTL_TO_CHUNKS`

Oracle Database 26ai ships **server-side chunking** as a first-class
PL/SQL primitive. `DBMS_VECTOR_CHAIN.UTL_TO_CHUNKS` tokenises and
segments text inside the database — the source CLOBs never travel
back to Python, only the resulting chunk records. For corpora that
already live in the database this collapses the first stage of any
RAG pipeline into a single SQL call.

`OracleInDBChunker` is the Locus wrapper. Two shapes:

- **`chunk_text(passage)`** — chunks a single string the application
  passed in. Good for ad-hoc material the user supplied.
- **`chunk_column(table, column, ...)`** — streams chunks out of
  every row of a CLOB column. Good for ingesting an entire table
  without round-tripping the documents through the application.

To exercise both shapes the notebook drives the chunker from a Locus
`Agent`. Each chunker call is wrapped as a `@tool`; the agent picks
which one to invoke from the prompt, and the SQL fires inside a
`ToolStartEvent` / `ToolCompleteEvent` pair so you can see exactly
when the database does the work. The agent is the harness; the
PL/SQL chunker is the subject.

## What this covers

- `OracleInDBChunker(...).chunk_text(passage)` returning chunk records
  for a single string.
- `OracleInDBChunker(...).chunk_column(table, column)` streaming
  chunks out of an entire CLOB column without round-tripping the
  source documents.
- `@tool chunk_paragraph(text)` / `@tool chunk_table_rows(limit)` —
  exposing each shape as an agent tool.
- One connection envelope (DSN, user, wallet) shared with the loader
  (notebook 39) and the vector store (notebook 38).

## Prerequisites

```bash
# Autonomous Database wallet (TLS) + credentials.
export ORACLE_DSN=mydb_low                   # tnsnames alias in the wallet
export ORACLE_USER=locus_app                 # least-privileged app schema
export ORACLE_PASSWORD='<app-password>'
export ORACLE_WALLET=~/.oci/wallets/mydb     # directory holding tnsnames.ora
export ORACLE_WALLET_PASSWORD='<wallet-pw>'  # if wallet is encrypted
```

One-time SQL prereq (out-of-band, as the schema owner):

```sql
GRANT EXECUTE ON DBMS_VECTOR_CHAIN TO locus_app;
```

If `ORACLE_DSN` / `ORACLE_PASSWORD` / `ORACLE_WALLET` aren't set the
notebook prints the wiring snippet and exits cleanly — no traceback,
no half-initialised state.

## Run

```bash
python examples/notebook_40_oracle_indb_chunker.py
```

## See also

- [Notebook 02 — Oracle 26ai RAG](notebook_38_oracle_26ai_rag.md)
- [Notebook 05 — Oracle ADB document loader](notebook_39_oracle_adb_loader.md)
- [Notebook 37 — Oracle in-DB embeddings](notebook_41_oracle_indb_embeddings.md)
- [Oracle AI Database 26ai — AI Vector Search User's Guide](https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/index.html)
- [`DBMS_VECTOR_CHAIN` PL/SQL package reference](https://docs.oracle.com/en/database/oracle/oracle-database/23/arpls/dbms_vector_chain1.html)
- [python-oracledb driver](https://python-oracledb.readthedocs.io/en/latest/index.html)

## Source

```python
--8<-- "examples/notebook_40_oracle_indb_chunker.py"
```
