# Oracle 26ai versioned checkpoint agent

An `Agent` paired with the versioned `OracleCheckpointSaver`. The
agent runs three turns; after each one, the post-turn `AgentState` is
serialised with `AgentState.to_checkpoint` and written to the saver
with a monotonic `checkpoint_id` and a parent link. The result is an
auditable lineage of the agent's state inside Oracle Database 26ai,
queryable with `list_checkpoints` and `get(checkpoint_id=...)`.

Contrast this with `oracle_checkpointer` from notebook 42. That factory
returns a `StorageBackendAdapter` over `OracleBackend`: a
`BaseCheckpointer` the agent loop drives itself, saving at the loop's
save points under generated ids and `MERGE`-upserting one row per key
it writes: `thread_id:checkpoint_id` for each save, a `thread_id:latest`
copy of the newest save that the next turn loads, and a
`thread_id:_checkpoints` index. `OracleCheckpointSaver` is not a
`BaseCheckpointer`: application code drives it with `put()`, choosing
each `checkpoint_id`, its `parent_checkpoint_id` link and the row's
`metadata`, and a second table holds intra-step pending writes
(`put_writes` / `get_writes`). This notebook runs both side by side:
the loop-driven checkpointer for live thread continuity, the versioned
saver for the audit trail.

What this notebook covers:

- `Agent(checkpointer=oracle_checkpointer(...), checkpoint_every_n_iterations=1)`
  providing state continuity for the live thread: each run on the same
  `thread_id` loads the thread's newest checkpoint before its first
  model call.
- After each `await agent.arun(prompt, thread_id=...)`,
  `saver.put(thread_id=..., checkpoint_id=cid, parent_checkpoint_id=parent,
  checkpoint_data=state.to_checkpoint(), metadata=...)` appending a new
  versioned row, with `cid` running `v1`, `v2`, `v3` and `parent` the
  previous id.
- `saver.list_checkpoints(thread_id=...)` listing the lineage newest
  first; each row carries the `metadata` written with it (turn number,
  message count, stop reason).
- `saver.get(thread_id=..., checkpoint_id="v1")` plus
  `AgentState.from_checkpoint` to rehydrate the historical state an
  alternate branch could start from.
- `saver.put_writes(...)` and `saver.get_writes(...)` for idempotent
  intra-step pending writes against the agent's latest checkpoint, `v3`.

## Prerequisites

```bash
export ORACLE_DSN=mydb_low
export ORACLE_USER=locus_app
export ORACLE_PASSWORD='<app-password>'
export ORACLE_WALLET=~/.oci/wallets/mydb
export ORACLE_WALLET_PASSWORD='<wallet-pw>'   # if encrypted
```

The script creates three tables on first use when the connected user
does not already own them: `locus_notebook_43_checkpoints` and
`locus_notebook_43_writes` for the saver (`auto_create_table=True`, the
default) and `locus_notebook_43_live` for the loop-driven checkpointer.
When it finishes it drops the two saver tables and deletes the live
thread's rows, so a second run starts clean. For production, create
the saver's tables ahead of time and pass `auto_create_table=False`,
so the runtime user needs no `CREATE TABLE` privilege.

In the drawing below, the agent's `oracle_checkpointer(...)` is Shape B,
a storage backend behind an adapter that the loop loads from and saves
into at the marked points; `OracleCheckpointSaver` sits outside that
contract, so every versioned row in this notebook is written by the
notebook's own `saver.put(...)` call, not by the loop:

{{ locus_diagram checkpointer-shapes }}

## Key lines

The agent runs three turns on the live thread with `await agent.arun(...)`,
the awaitable form of `run_sync`; after each one the post-turn
`AgentState` in `result.state` is serialised with `to_checkpoint()` and
written to the saver under a fresh `checkpoint_id` linked to the previous
one, with the turn number, message count and stop reason as `metadata`:

{{ locus_source examples/notebook_43_oracle_versioned_saver.py::run_agent_with_versioned_history }}

`saver.get(thread_id=..., checkpoint_id="v1")` reads one historical row
back; its `checkpoint` field is the dictionary `to_checkpoint()`
produced, and `AgentState.from_checkpoint` turns it into a state an
alternate-branch agent could start from:

{{ locus_source examples/notebook_43_oracle_versioned_saver.py::rewind_to_v1 }}

`put_writes` stores intra-step pending writes against the latest
checkpoint; a retry for the same
`(thread_id, checkpoint_ns, checkpoint_id, task_id)` tuple deletes the
earlier set and inserts the new one under a single commit, so it is safe
to call from a tool that may run twice:

{{ locus_source examples/notebook_43_oracle_versioned_saver.py::write_pending }}

## Run it

```bash
python examples/notebook_43_oracle_versioned_saver.py
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

- [Notebook 42 — Oracle 26ai checkpointer (the loop-driven checkpointer this notebook pairs with)](notebook_42_oracle_26ai_checkpointer.md)
- [Concepts — Checkpointers & Store](../concepts/checkpointers.md)
- [Concepts — Oracle Database 26ai](../concepts/oracle-26ai.md)
- [Oracle Database 26ai — documentation](https://docs.oracle.com/en/database/oracle/oracle-database/26/index.html)
- [Oracle Autonomous Database — documentation](https://docs.oracle.com/en/cloud/paas/autonomous-database/index.html)
- [python-oracledb driver](https://python-oracledb.readthedocs.io/en/latest/index.html)

## Source

??? note "Full source — `examples/notebook_43_oracle_versioned_saver.py`"

    ````python
    --8<-- "examples/notebook_43_oracle_versioned_saver.py"
    ````
