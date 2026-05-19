# Oracle 26ai cross-thread memory agent

A Locus `Agent` with cross-thread long-term memory backed by
`OracleStore`. Where the [Oracle 26ai checkpointer](notebook_07_oracle_26ai_checkpointer.md)
persists *per-thread* state, `OracleStore` persists *cross-thread*
facts. The Locus agent reaches into it through
`LLMMemoryManager`, which extracts memories at the end of a session
and injects them into the system prompt at the start of the next one.
The agent in this notebook learns a fact in thread A and a brand-new
agent in thread B answers a question that depends on that fact —
without ever seeing thread A's messages.

## What this covers

- `OracleStore` implementing the `BaseStore` contract on top of
  Oracle 26ai. Plain K/V here; pass `dimension=N` for vector mode.
- `LLMMemoryManager(store=oracle_store, namespace_prefix=("users",
  user_id), extract_fn=...)` as the agent-facing surface. Hand it to
  `Agent(memory_manager=...)`.
- The agent loop calling `manager.extract(messages)` at session end
  and `manager.retrieve()` at session start — visible by inspecting
  the store between the two threads.
- A deterministic regex-based extractor for the demo; production
  deployments wire an LLM-backed extractor here.

## Prerequisites

The store, the [checkpointer](notebook_07_oracle_26ai_checkpointer.md),
the [versioned saver](notebook_12_oracle_versioned_saver.md), and the
[vector store](notebook_06_oracle_26ai_rag.md) can all share a single
Autonomous Database:

```bash
export ORACLE_DSN=mydb_low                   # tnsnames alias in the wallet
export ORACLE_USER=locus_app                 # least-privileged app schema
export ORACLE_PASSWORD='<app-password>'
export ORACLE_WALLET=~/.oci/wallets/mydb     # directory holding tnsnames.ora
export ORACLE_WALLET_PASSWORD='<wallet-pw>'  # if wallet is encrypted
```

If `ORACLE_DSN` / `ORACLE_PASSWORD` aren't set the notebook prints the
wiring snippet and exits cleanly — no traceback, no half-initialised
state.

## Run

```bash
python examples/notebook_11_oracle_store.py
```

## Schema hygiene

The notebook writes to `locus_notebook_11_store` (overridable via
`table_name=`) and drops the table at the end so the demo is
re-runnable. For production, pre-create the table out-of-band as a
least-privileged app schema owner and pass `auto_create_table=False`
so the runtime user runs with `INSERT / SELECT / UPDATE / DELETE`
only.

## See also

- [Notebook 07 — Oracle 26ai checkpointer](notebook_07_oracle_26ai_checkpointer.md)
- [Notebook 12 — Oracle 26ai versioned checkpoint saver](notebook_12_oracle_versioned_saver.md)
- [Oracle Database 26ai — documentation](https://docs.oracle.com/en/database/oracle/oracle-database/23/index.html)
- [Oracle AI Database 26ai — AI Vector Search User's Guide](https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/index.html)
- [python-oracledb driver](https://python-oracledb.readthedocs.io/en/latest/index.html)

## Source

```python
--8<-- "examples/notebook_11_oracle_store.py"
```
