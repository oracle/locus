# Oracle 26ai cross-thread memory agent

An `Agent` with cross-thread long-term memory backed by
`OracleStore`. Where the [Oracle 26ai checkpointer](notebook_42_oracle_26ai_checkpointer.md)
persists *per-thread* state, `OracleStore` persists *cross-thread*
facts. The agent reaches it through `LLMMemoryManager`, which extracts
memories when a session ends and, at the start of the next one, recalls
them and inserts them as a system message immediately after the user's
turn. The agent in this notebook learns a fact in thread A and a new
agent in thread B answers a question that depends on that fact, without
ever seeing thread A's messages.

What this notebook covers:

- `OracleStore` implementing the `BaseStore` contract on top of
  Oracle Database 26ai. Plain key/value here; pass `dimension=N` for
  vector mode.
- `LLMMemoryManager(store=oracle_store, namespace_prefix=("users",
  user_id), extract_fn=...)` as the agent-facing surface. Hand it to
  `Agent(memory_manager=...)`.
- The agent loop calling the manager's `on_session_start` (recall and
  inject) before the first model call and `on_session_end` (extract and
  save) when the run finishes — visible by inspecting the store between
  the two threads.
- A deterministic regex-based extractor for the example; production
  deployments wire an LLM-backed extractor here.

## Prerequisites

The store, the [checkpointer](notebook_42_oracle_26ai_checkpointer.md),
the [versioned saver](notebook_43_oracle_versioned_saver.md), and the
[vector store](notebook_38_oracle_26ai_rag.md) can all share a single
Autonomous Database:

```bash
export ORACLE_DSN=mydb_low                   # tnsnames alias in the wallet
export ORACLE_USER=locus_app                 # least-privileged app schema
export ORACLE_PASSWORD='<app-password>'
export ORACLE_WALLET=~/.oci/wallets/mydb     # directory holding tnsnames.ora
export ORACLE_WALLET_PASSWORD='<wallet-pw>'  # if wallet is encrypted
```

The notebook writes to the table `locus_notebook_37_store` (set with
`table_name=`) and drops it at the end so the example can be run again.
For production, pre-create the table out-of-band as the schema owner
and pass `auto_create_table=False`, so the runtime user needs
`INSERT`, `SELECT`, `UPDATE` and `DELETE` only.

The drawing places this notebook in the Memory group: `OracleStore`
is a `BaseStore` of namespaced key/value rows, and the agent reaches
it through `LLMMemoryManager(store=…)` passed as `memory_manager=`,
while the checkpointer in the Threads group keeps each thread's state
in the same database, next to the data the agent acts on:

{{ locus_diagram oracle-26ai-primitives }}

## Key lines

The store is a plain key/value `OracleStore` on the wallet connection
from Prerequisites; `dimension=None` keeps it out of vector mode,
because the manager recalls through `store.search()`, a `LIKE`
substring match against the new user turn followed by a recency pass
with `query=None`, and neither needs an embedding:

{{ locus_source examples/notebook_37_oracle_store.py::_build_store }}

`LLMMemoryManager` is the surface the agent sees: the store, a
`namespace_prefix` that scopes every row to one user, and an explicit
extractor, here a small regex function so the example is deterministic:

{{ locus_source examples/notebook_37_oracle_store.py::_build_manager }}

A new agent on a new thread shares nothing with thread A except
`memory_manager=manager`; at session start the manager recalls the
facts thread A wrote and inserts them as a system message after the
new prompt, so the model answers without ever seeing thread A's
messages:

{{ locus_source examples/notebook_37_oracle_store.py::thread_b }}

## Run it

```bash
python examples/notebook_37_oracle_store.py
```

The script needs an Oracle Database reachable through the four
variables in Prerequisites: `ORACLE_DSN`, `ORACLE_USER`,
`ORACLE_PASSWORD` and `ORACLE_WALLET` (`ORACLE_WALLET_PASSWORD` only
when the wallet is encrypted). When any of the four is unset it prints
the missing names and the minimal wiring, then exits cleanly without
opening a connection. With the database in place, the agent's model
comes from `examples/config.py`: OCI Generative AI when `~/.oci/config`
has a usable profile, otherwise the bundled mock model. Set
`LOCUS_MODEL_PROVIDER` (`oci`, `openai`, `anthropic` or `mock`) to
force one.

## See also

- [Oracle 26ai checkpointer](notebook_42_oracle_26ai_checkpointer.md)
- [Oracle 26ai versioned checkpoint agent](notebook_43_oracle_versioned_saver.md)
- [Oracle agent memory](notebook_44_oracle_agent_memory.md)
- [Concepts — Long-term memory](../concepts/memory-manager.md)
- [Concepts — Oracle Database 26ai](../concepts/oracle-26ai.md)
- [Oracle Database 26ai — documentation](https://docs.oracle.com/en/database/oracle/oracle-database/26/index.html)
- [Oracle Database 26ai — AI Vector Search User's Guide](https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/index.html)
- [Oracle Autonomous Database — documentation](https://docs.oracle.com/en/cloud/paas/autonomous-database/index.html)
- [python-oracledb driver](https://python-oracledb.readthedocs.io/en/latest/index.html)

## Source

??? note "Full source — `examples/notebook_37_oracle_store.py`"

    ````python
    --8<-- "examples/notebook_37_oracle_store.py"
    ````
