# Oracle in-DB chunker

Oracle Database 26ai ships **server-side chunking** as a PL/SQL
primitive. `DBMS_VECTOR_CHAIN.UTL_TO_CHUNKS` tokenises and segments
text inside the database; for text that already sits in a CLOB column
the source never travels back to Python, only the resulting chunk
records do. For such a corpus this collapses the chunking stage of a
retrieval-augmented generation (RAG) pipeline into a single SQL
statement.

`OracleInDBChunker` is the wrapper. Two shapes:

- **`chunk_text(passage)`** — chunks a single string the application
  passed in. Suited to ad-hoc material the user supplied.
- **`chunk_column(table_name=..., text_column=...)`** — streams chunks
  out of every row of a CLOB column. Suited to ingesting an entire
  table without round-tripping the documents through the application.

To exercise both shapes the notebook drives the chunker from an `Agent`. Each chunker call is wrapped as an `@tool`; the agent picks
which one to invoke from the prompt, and the SQL runs between the
`ToolStartEvent` and `ToolCompleteEvent` the agent emits around the
call, so the moment the database does the work is visible in the event
stream. The agent is the harness; the PL/SQL chunker is the subject.

What this notebook covers:

- `OracleInDBChunker(...).chunk_text(passage)` returning chunk records
  for a single string.
- `OracleInDBChunker(...).chunk_column(table_name=..., text_column=...)`
  streaming chunks out of an entire CLOB column without round-tripping
  the source documents.
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

One-time SQL prerequisite (out-of-band, as the schema owner):

```sql
GRANT EXECUTE ON DBMS_VECTOR_CHAIN TO locus_app;
```

In the drawing, `OracleInDBChunker` sits in the vectors group of the
primitives the SDK runs on one Oracle Database 26ai, beside the loader
that reads rows (notebook 39) and the embeddings and vector store that
take the chunks on (notebooks 41 and 38), on the ingest path called
from your code; this notebook moves that same `UTL_TO_CHUNKS` call
behind two `@tool` functions so the `Agent` on the left reaches it
directly:

{{ locus_diagram oracle-26ai-primitives }}

## Key lines

`chunk_paragraph` wraps `chunker.chunk_text`: the passage is passed as a
bind variable to one `SELECT` over
`TABLE(DBMS_VECTOR_CHAIN.UTL_TO_CHUNKS(:text, JSON(:params)))`, segmented
inside the database at the chunker's twenty-word window, and the tool
returns the chunk count, the passage length and a preview of the first
chunk:

{{ locus_source examples/notebook_40_oracle_indb_chunker.py::main.chunk_paragraph }}

`chunk_table_rows` wraps the streaming `chunker.chunk_column` over the
demo CLOB table: only chunk records come back, never the source CLOBs,
and `limit` caps rows rather than chunks:

{{ locus_source examples/notebook_40_oracle_indb_chunker.py::main.chunk_table_rows }}

`_drive_agent` reads the event stream from `agent.run(prompt)`:
`tool_start` is the call crossing into the database with its arguments,
`tool_complete` brings the chunk statistics back, and `terminate`
carries the agent's final message; the `Agent` that binds both tools,
with its ingest-planning system prompt and `max_iterations=4`, is in
`main()` in the full source below:

{{ locus_source examples/notebook_40_oracle_indb_chunker.py::_drive_agent }}

## Run it

```bash
python examples/notebook_40_oracle_indb_chunker.py
```

The script needs an Oracle Database 26ai reachable through the four
variables in Prerequisites — `ORACLE_DSN`, `ORACLE_USER`,
`ORACLE_PASSWORD` and `ORACLE_WALLET`; `ORACLE_WALLET_PASSWORD` is
optional and matters only for an encrypted wallet — and `EXECUTE` on
`DBMS_VECTOR_CHAIN` for that user. When any of the four is unset it
prints a skip notice naming the missing variables, together with the
minimal wiring, and exits cleanly with status 0 before opening a
connection. With them set it creates a one-row demo table,
`locus_notebook_40_docs`, in that schema, runs the two prompts and drops
the table on exit.

Model choice for the agent comes from `examples/config.py`: OCI
Generative AI when `~/.oci/config` has a usable profile, otherwise the
bundled mock model. Set `LOCUS_MODEL_PROVIDER` (`oci`, `openai`,
`anthropic` or `mock`) to force one. Under the mock model each prompt
still issues one synthesised tool call, so `UTL_TO_CHUNKS` does run in
the database; but neither tool's name appears as a word in either
prompt, so the mock's selection falls back to the first bound tool and
both prompts call `chunk_paragraph`, with the prompt's last capitalised
word as the passage. `chunk_table_rows` is reached only by a live
model, the agent's final message is the mock's placeholder text rather
than a reading of the returned statistics, and the closing paragraph
the script prints is fixed text written for a live run.

## See also

- [Concepts — Oracle Database 26ai, Retrieval](../concepts/oracle-26ai.md#retrieval)
- [Concepts — Events: `ToolStartEvent` and `ToolCompleteEvent`](../concepts/events.md#toolstartevent)
- [Notebook 38 — Oracle 26ai RAG](notebook_38_oracle_26ai_rag.md)
- [Notebook 39 — Oracle ADB loader](notebook_39_oracle_adb_loader.md)
- [Notebook 41 — Oracle in-DB embeddings](notebook_41_oracle_indb_embeddings.md)
- [Oracle AI Database 26ai — AI Vector Search User's Guide](https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/index.html)
- [`DBMS_VECTOR_CHAIN` PL/SQL package reference](https://docs.oracle.com/en/database/oracle/oracle-database/26/arpls/dbms_vector_chain1.html)
- [python-oracledb driver](https://python-oracledb.readthedocs.io/en/latest/index.html)

## Source

??? note "Full source — `examples/notebook_40_oracle_indb_chunker.py`"

    ````python
    --8<-- "examples/notebook_40_oracle_indb_chunker.py"
    ````
