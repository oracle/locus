# Oracle 26ai Checkpointer agent

In-memory and on-disk checkpointers are fine for local development —
when the process restarts, the conversation is gone. This notebook
wires Locus's `oracle_checkpointer` into an `Agent` so the agent loop
itself writes `AgentState` to Oracle 26ai on every iteration. A second
`Agent` instance — built fresh, simulating a different process — loads
the same `thread_id` and the conversation continues uninterrupted.
The user never calls `cp.save` — the agent does that for them.

## What this covers

- `oracle_checkpointer(...)` plugged into `Agent(checkpointer=...)`.
  The agent loop persists `AgentState` after every iteration.
- `agent.run_sync(prompt, thread_id="...")` loading state for the
  thread before the first model call and saving after each iteration.
- `checkpoint_every_n_iterations=1` — a mid-loop crash recovers to
  the last completed iteration, not the start of the turn.
- A brand-new `Agent` object resuming the same `thread_id` and
  recalling turn 1 without being told what was said.
- `cp.list_threads()` as the admin view dashboards reach for.

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

If `ORACLE_DSN` / `ORACLE_PASSWORD` aren't set the notebook prints the
wiring snippet and exits cleanly — no traceback, no half-initialised
state.

## Run

```bash
python examples/notebook_42_oracle_26ai_checkpointer.py
```

## Schema hygiene

This notebook writes to `locus_notebook_42` (overridable via
`table_name=`). For production, pre-create the table out-of-band as a
least-privileged app schema owner and run the application user with
`INSERT / SELECT / UPDATE / DELETE` only — see the
[checkpointer concepts page](../concepts/checkpointers.md) for the DDL
and the rotation guidance.

## See also

- [Concepts — Checkpointers & Store](../concepts/checkpointers.md)
- [Notebook 45 — Oracle 26ai RAG (shares the same wallet)](notebook_38_oracle_26ai_rag.md)
- [Oracle Database 26ai — documentation](https://docs.oracle.com/en/database/oracle/oracle-database/23/index.html)
- [Oracle Autonomous Database — documentation](https://docs.oracle.com/en/cloud/paas/autonomous-database/index.html)
- [python-oracledb driver](https://python-oracledb.readthedocs.io/en/latest/index.html)

## Source

```python
--8<-- "examples/notebook_42_oracle_26ai_checkpointer.py"
```
