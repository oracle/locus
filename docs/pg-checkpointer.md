# PgCheckpointer and message retention

Conversation threads in Postgres for a server that serves many companies, and
retention that works per message, so a thread used every day stops growing.

## Local first, the same retention everywhere

| | Local default (no infrastructure) | Postgres (multi-tenant) |
| --- | --- | --- |
| Checkpointer | `MemoryCheckpointer`, `FileCheckpointer` | `PgCheckpointer` |
| Retention on save | `RetainedCheckpointer(inner, max_age=...)` | `PgCheckpointer(message_retention=...)` |
| Idle threads | `vacuum` where the backend has it | `purge_messages()`, `vacuum()` |
| Tenants | one process, one user | a `tenant` column with Row-Level Security |

Nothing here needs a cloud account. `PgCheckpointer` runs against any
Postgres (a laptop, a container, a managed cluster); the generic
`postgresql_checkpointer()` adapter is still there for a single-tenant table.

## PgCheckpointer

```python
from datetime import timedelta
from tulip.memory.backends import PgCheckpointer

checkpointer = PgCheckpointer(
    dsn,
    table="agent_threads",
    keep_checkpoints=1,                     # one row per thread
    message_retention=timedelta(days=30),   # exchanges older than this go at each save
)
agent = Agent(model=model, checkpointer=checkpointer)

with checkpointer.tenant_scope("acme"):
    await agent.arun("hello", thread_id="support/42")
```

- **One tenant never sees another's threads.** Every row carries its tenant.
  Row-Level Security admits only the tenant pinned for the transaction
  (`tulip.tenant`, the setting `PgMemory` uses), on read and on write, for the
  table owner too (`FORCE`). Every query also filters on the tenant. Connect as
  an application role that is not a superuser and has no `BYPASSRLS`: those
  skip Row-Level Security.
- **Which tenant.** A `tenant_scope()` block (context-local, so concurrent runs
  keep their own), else `tenant_of(thread_id)` for thread ids that carry the
  tenant, else `tenant=` (`"default"`). A `tenant_of` that fails, fails the call.
- **One statement per save.** The checkpoint is upserted and the thread pruned
  to `keep_checkpoints` in one statement, with the tenant pinned in the same
  transaction. The generic adapter makes four round trips per save.
- **No `CREATE` needed when the table exists.** The schema is probed first and
  only what is missing is created, so an application role with DML only works
  against a table a migration made. `PgCheckpointer.ddl()` returns those
  statements; `create_schema=False` never runs DDL. A table with another layout
  (the generic adapter's) is refused with a message, never written to.
- **Housekeeping per tenant.** `vacuum(days)` deletes threads nobody has saved
  to for that long; `purge_messages(older_than)` rewrites, in place, the
  checkpoints that still hold older exchanges (for idle threads, which no save
  trims); `forget_tenant()` deletes everything of the tenant in scope.

## Message retention

When a message is first checkpointed it is stamped with the time, in
`Message.metadata["tulip_at"]`, which no provider is sent. Trimming drops whole
exchanges (a user message and everything up to the next one) that began before
the cut-off, so a tool result never outlives its call. System messages (the
prompt, a compaction summary) stay. A message without a stamp is kept: its age
is unknown, and retention never guesses.

`tulip.memory.retention` has the pieces for any other backend:
`stamp_messages`, `trim_messages`, `oldest_message_time`, and
`RetainedCheckpointer`, which wraps any checkpointer.
