# Oracle ADB Document Loader

The first link of any Oracle-native RAG pipeline is **getting the
source rows out of the database**. `OracleADBLoader` is an
async-pooled loader that turns a parameter-bound ADB `SELECT` into a
stream of `Document` objects ready for chunking, embedding, or direct
agent consumption.

Two production-shape details earn their keep:

- **Parameter-bound queries by construction.** The loader's
  `bind_params=` is the only path user input reaches the database;
  the SQL template itself is fixed at construction time. SQL
  injection through the query text isn't possible.
- **Async connection pool.** One pool, kwargs-shared with the rest of
  the 26ai surface (vector store from notebook 06, in-DB chunker from
  09, in-DB embeddings from 10, checkpointer from 07) — same wallet,
  same DSN, one connection envelope for the whole stack.

To show the loader doing real work, the notebook wraps it as a `@tool`
on a Locus `Agent`. The agent decides which topic to pull from a
disposable `locus_notebook_08_articles` table, the loader's `SELECT`
fires inside a `ToolStartEvent` / `ToolCompleteEvent` pair, and the
agent grounds its summary in the returned rows. The agent is the
harness; the loader is the subject.

## What this covers

- `OracleADBLoader(query="SELECT ...", bind_params={...})` with the
  query template fixed at construction — only `bind_params` accepts
  user input.
- `await loader.load()` returning the full row set in one round-trip;
  switch to `lazy_load()` for streaming into a UI.
- One connection envelope (DSN, user, wallet) shared with the vector
  store, in-DB chunker, in-DB embedder, and checkpointer.
- A `@tool fetch_articles_by_topic(topic)` wrapping the loader so an
  agent can decide what to fetch.

The notebook is self-contained: it creates a disposable demo table
`locus_notebook_08_articles`, populates a few rows, hands the loader
to an `Agent` as a tool, asks the agent two domain questions, and
drops the table on the way out.

## Prerequisites

```bash
# Autonomous Database wallet (TLS) + credentials.
export ORACLE_DSN=mydb_low                   # tnsnames alias in the wallet
export ORACLE_USER=locus_app                 # least-privileged app schema
export ORACLE_PASSWORD='<app-password>'
export ORACLE_WALLET=~/.oci/wallets/mydb     # directory holding tnsnames.ora
export ORACLE_WALLET_PASSWORD='<wallet-pw>'  # if wallet is encrypted
```

If `ORACLE_DSN` / `ORACLE_PASSWORD` / `ORACLE_WALLET` aren't set the
notebook prints the wiring snippet and exits cleanly — no traceback,
no half-initialised state.

## Run

```bash
python examples/notebook_42_oracle_adb_loader.py
```

## See also

- [Notebook 06 — Oracle 26ai RAG](notebook_41_oracle_26ai_rag.md)
- [Notebook 09 — Oracle in-DB chunker](notebook_43_oracle_indb_chunker.md)
- [Notebook 10 — Oracle in-DB embeddings](notebook_44_oracle_indb_embeddings.md)
- [Oracle Autonomous Database — documentation](https://docs.oracle.com/en/cloud/paas/autonomous-database/index.html)
- [Oracle AI Database 26ai — AI Vector Search User's Guide](https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/index.html)
- [python-oracledb driver](https://python-oracledb.readthedocs.io/en/latest/index.html)

## Source

```python
--8<-- "examples/notebook_42_oracle_adb_loader.py"
```
