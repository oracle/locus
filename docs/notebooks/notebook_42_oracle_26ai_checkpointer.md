# Oracle 26ai checkpointer

In-memory and on-disk checkpointers suit local development: when the
process restarts, the conversation is gone. This notebook wires the SDK's
`oracle_checkpointer` into an `Agent` so the agent loop itself writes
`AgentState` to Oracle Database 26ai on every iteration. A second
`Agent` instance, built fresh to simulate a different process, loads
the same `thread_id` and the conversation continues. The application
never calls `cp.save`; the agent loop does.

What this notebook covers:

- `oracle_checkpointer(...)` plugged into `Agent(checkpointer=...)`:
  the agent loop persists `AgentState` after every iteration.
- `agent.run_sync(prompt, thread_id="...")` loading the thread's
  newest checkpoint before the first model call, then saving after
  each iteration and when the turn ends.
- Per-iteration saves (`checkpoint_every_n_iterations=1`, already the
  default for this adapter on a thread): a process killed mid-turn loses
  at most the iteration in flight, and `Agent.continue_turn(thread_id)`
  picks the turn up from the last save.
- A new `Agent` object resuming the same `thread_id` and recalling
  turn 1 without being told what was said.
- `cp.list_threads()` on the agent's own checkpointer, the enumeration
  an admin dashboard would use. The adapter forwards its default
  pattern `*` unchanged to the Oracle backend's `LIKE` filter, which
  matches no row, so the live run prints an empty list;
  `cp.list_threads(pattern="%")` returns the stored keys.

## Prerequisites

The checkpointer and the vector store from notebook 38 can share a
single Autonomous Database:

```bash
export ORACLE_DSN=mydb_low                   # tnsnames alias in the wallet
export ORACLE_USER=locus_app                 # least-privileged app schema
export ORACLE_PASSWORD='<app-password>'
export ORACLE_WALLET=~/.oci/wallets/mydb     # directory holding tnsnames.ora
export ORACLE_WALLET_PASSWORD='<wallet-pw>'  # if wallet is encrypted
```

The notebook writes to the table `locus_notebook_42` (set with
`table_name=`) and leaves the thread in place, so a second run of the
script continues it rather than starting over. `OracleBackend` creates
the table on first use when the connected user does not already own
one by that name; for production, create it ahead of time in your
deployment script, so the runtime user needs no `CREATE TABLE`
privilege.

The drawing places `oracle_checkpointer()` in Shape B, a
`StorageBackendAdapter` over `OracleBackend` that keys every save as
`thread_id:checkpoint_id`, one record per key. Two parts of the drawing
are what the two sessions rely on: in the run panel at the top right,
the loop saves after every iteration and when the run ends; the arrow
from `BaseCheckpointer` up into `Agent` is the second `Agent`'s
`run_sync(prompt, thread_id=…)` loading the thread's newest checkpoint
before its first model call:

{{ locus_diagram checkpointer-shapes }}

## Key lines

The factory call is the only Oracle-specific wiring: the DSN (data
source name), user, password and wallet from Prerequisites and the
table the thread lives in; what comes back is a
`StorageBackendAdapter`, the shape `Agent(checkpointer=…)` expects:

{{ locus_source examples/notebook_42_oracle_26ai_checkpointer.py::_build_checkpointer }}

Each session builds its own `Agent` and its own connection pool, so
the two share nothing in memory; `checkpointer=` hands the adapter to
the loop, and `checkpoint_every_n_iterations=1` makes the
per-iteration save explicit: on a thread, the loop already saves every
iteration when its checkpointer can delete a single save, which this
adapter can:

{{ locus_source examples/notebook_42_oracle_26ai_checkpointer.py::_build_agent }}

The second session is a different `Agent` object on the same
`thread_id`; `run_sync` loads the saved state from Oracle Database
before its first model call, so the agent answers a follow-up about
turn 1 although this process never saw it:

{{ locus_source examples/notebook_42_oracle_26ai_checkpointer.py::second_session }}

## Run it

```bash
python examples/notebook_42_oracle_26ai_checkpointer.py
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

- [Concepts — Checkpointers](../concepts/checkpointers.md)
- [Notebook 38 — Oracle 26ai retrieval-augmented generation (RAG) agent, which shares the same wallet](notebook_38_oracle_26ai_rag.md)
- [Oracle Database 26ai — documentation](https://docs.oracle.com/en/database/oracle/oracle-database/26/index.html)
- [Oracle Autonomous Database — documentation](https://docs.oracle.com/en/cloud/paas/autonomous-database/index.html)
- [python-oracledb driver](https://python-oracledb.readthedocs.io/en/latest/index.html)

## Source

??? note "Full source — `examples/notebook_42_oracle_26ai_checkpointer.py`"

    ````python
    --8<-- "examples/notebook_42_oracle_26ai_checkpointer.py"
    ````
