# Oracle 26ai versioned checkpoint agent

A Locus `Agent` driving the versioned `OracleCheckpointSaver`. The
agent runs three turns; after each one, the post-turn `AgentState`
gets serialised via `AgentState.to_checkpoint` and written to the
saver with a monotonic `checkpoint_id` and a parent link. The result:
an auditable lineage of the agent's life inside Oracle 26ai, queryable
by `list_checkpoints` / `get(checkpoint_id=...)`.

Contrast with the simpler `oracle_checkpointer` from notebook 07:
that one MERGEs a single row per thread (latest wins). Reach for
`OracleCheckpointSaver` when you need history (time-travel, fork) or
intra-step durability (`put_writes` / `get_writes`). This notebook
runs both side-by-side — the single-row checkpointer for live thread
continuity, the versioned saver for the audit trail.

## What this covers

- `Agent(checkpointer=oracle_checkpointer(...))` providing per-turn
  state continuity for the live thread.
- After each `agent.run_sync(prompt, thread_id=...)`,
  `saver.put(thread_id, checkpoint_id=vN, parent_checkpoint_id=vN-1,
  checkpoint_data=result.state.to_checkpoint())` appending a new
  versioned row.
- `saver.list_checkpoints(thread_id=...)` showing the lineage; each
  row carries the `metadata` (turn number, stop reason).
- `saver.get(thread_id, checkpoint_id="v1")` plus
  `AgentState.from_checkpoint` to rehydrate the historical state for
  branching.
- `saver.put_writes(...)` for idempotent intra-step pending writes
  against the agent's latest checkpoint.

## Prerequisites

```bash
export ORACLE_DSN=mydb_low
export ORACLE_USER=locus_app
export ORACLE_PASSWORD='<app-password>'
export ORACLE_WALLET=~/.oci/wallets/mydb
export ORACLE_WALLET_PASSWORD='<wallet-pw>'   # if encrypted
```

If those env vars aren't set the notebook prints a skip-banner and
exits cleanly — no traceback.

## Run

```bash
python examples/notebook_54_oracle_versioned_saver.py
```

## See also

- [Oracle 26ai checkpointer (single-row)](notebook_53_oracle_26ai_checkpointer.md)
- [Concepts → Checkpointers & Store](../concepts/checkpointers.md)
- [Oracle Database 26ai — documentation](https://docs.oracle.com/en/database/oracle/oracle-database/23/index.html)
- [Oracle Autonomous Database — documentation](https://docs.oracle.com/en/cloud/paas/autonomous-database/index.html)
- [python-oracledb driver](https://python-oracledb.readthedocs.io/en/latest/index.html)

## Source

```python
--8<-- "examples/notebook_54_oracle_versioned_saver.py"
```
