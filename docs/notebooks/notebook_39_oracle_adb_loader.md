# Oracle ADB loader

The first step of an Oracle-native
[retrieval-augmented generation (RAG)](../concepts/rag.md) pipeline is
getting the source rows out of the database. `OracleADBLoader` runs a
`SELECT` against Oracle Autonomous Database (ADB) over its own small async
connection pool and yields one `Document` per row, ready for chunking,
embedding or direct use by an agent. This notebook puts the loader behind
an `@tool` so that an `Agent` decides which rows to fetch.

Two properties shape how the loader is used. The SQL text is fixed when
the loader is constructed and is handed to the driver verbatim, so the
caller owns its shape and must keep user input out of it; in this
notebook the agent's `topic` reaches the database only through
`bind_params`. The loader's connection arguments — `dsn`, `user`,
`password`, `wallet_location`, `wallet_password` — are the same ones the
vector store (notebook 38), in-DB chunker (40), in-DB embeddings (41) and
checkpointer (42) accept, so one configuration block serves the whole
stack even though each class opens its own pool.

The agent decides which topic to fetch from a disposable
`locus_notebook_39_articles` table, the loader's `SELECT` runs inside a
`ToolStartEvent` / `ToolCompleteEvent` pair, and the agent grounds its
summary in the returned rows. The agent is the harness; the loader is the
subject. The notebook creates the table, populates a few rows, asks the
agent two questions and drops the table before exiting.

What this notebook covers:

- `OracleADBLoader(sql="SELECT ...", bind_params={...})` with the query
  text fixed at construction; only `bind_params` carries the agent's
  input.
- `await loader.load()` returning the full row set in one call; switch to
  `lazy_load()` to stream rows as the driver fetches them in
  `fetch_arraysize` batches.
- One set of connection arguments — DSN (data source name), user,
  wallet — accepted by the loader, the vector store, the in-DB chunker,
  the in-DB embedder and the checkpointer alike.
- An `@tool fetch_articles_by_topic(topic)` wrapping the loader so an
  agent can decide what to fetch.

## Prerequisites

```bash
# Autonomous Database wallet (TLS) + credentials.
export ORACLE_DSN=mydb_low                   # tnsnames alias in the wallet
export ORACLE_USER=locus_app                 # least-privileged app schema
export ORACLE_PASSWORD='<app-password>'
export ORACLE_WALLET=~/.oci/wallets/mydb     # directory holding tnsnames.ora
export ORACLE_WALLET_PASSWORD='<wallet-pw>'  # if wallet is encrypted
```

In the drawing below the loader sits in the vectors group, reading rows
from your tables as `Document` objects on the same Oracle Database 26ai
that the vector store, in-DB chunker and in-DB embedder of notebooks 38,
40 and 41 use; the drawing has it run at ingest, outside any agent tool
call, whereas this notebook puts it one `@tool` call away from the agent:

{{ locus_diagram oracle-26ai-primitives }}

## Key lines

The connection arguments — DSN, user, password, wallet — are defined once
and read from the environment by the demo-table setup and by every loader
the tool creates:

{{ locus_source examples/notebook_39_oracle_adb_loader.py::_conn_kwargs }}

`fetch_articles_by_topic` is the `@tool` the agent sees: each call builds
a fresh `OracleADBLoader` whose SQL text is fixed and whose `topic`
arrives through `bind_params`, maps the returned `Document` objects to
plain rows, and closes that loader's pool:

{{ locus_source examples/notebook_39_oracle_adb_loader.py::fetch_articles_by_topic }}

The run loop iterates `agent.run(prompt)` and reads the `tool_start` /
`tool_complete` pair around each database call — the topic the agent
chose on the way out, the row count and slugs decoded from the JSON
string in `result` on the way back — before printing the grounded answer
carried by the `terminate` event:

{{ locus_source examples/notebook_39_oracle_adb_loader.py::_drive_agent }}

## Run it

```bash
python examples/notebook_39_oracle_adb_loader.py
```

The script needs an Autonomous Database: `ORACLE_DSN`, `ORACLE_USER`,
`ORACLE_PASSWORD` and `ORACLE_WALLET` must all be set, plus
`ORACLE_WALLET_PASSWORD` when the wallet is encrypted. When any of the
four is missing it prints the missing names and the minimal wiring
snippet, then exits cleanly without opening a connection. The agent's
model comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model;
set `LOCUS_MODEL_PROVIDER` (`oci`, `openai`, `anthropic` or `mock`) to
force one. Under the mock model the first turn of each run still issues a
synthesised tool call, so the parameter-bound `SELECT` does run against
the database; the synthesised `topic` is a single capitalised word taken
from the prompt rather than one of the stored labels, so the query
returns no rows, and the closing answer is placeholder text rather than a
reading of the result.

## See also

- [Concepts — Oracle Database 26ai](../concepts/oracle-26ai.md)
- [API reference — RAG, Oracle Database 26ai](../api/rag.md#oracle-database-26ai)
- [Notebook 38 — Oracle 26ai RAG](notebook_38_oracle_26ai_rag.md)
- [Notebook 40 — Oracle in-DB chunker](notebook_40_oracle_indb_chunker.md)
- [Notebook 41 — Oracle in-DB embeddings](notebook_41_oracle_indb_embeddings.md)
- [Notebook 42 — Oracle 26ai checkpointer](notebook_42_oracle_26ai_checkpointer.md)
- [Oracle Autonomous Database — documentation](https://docs.oracle.com/en/cloud/paas/autonomous-database/index.html)
- [Oracle AI Database 26ai — AI Vector Search User's Guide](https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/index.html)
- [python-oracledb driver](https://python-oracledb.readthedocs.io/en/latest/index.html)

## Source

??? note "Full source — `examples/notebook_39_oracle_adb_loader.py`"

    ````python
    --8<-- "examples/notebook_39_oracle_adb_loader.py"
    ````
