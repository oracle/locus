# Checkpointers

A checkpointer is the contract for **persisting agent state** between
runs. Pass one to `Agent(checkpointer=...)` and the agent saves
`AgentState` at three points: when the run finishes or raises; before
it pauses on an interrupt; and, on a checkpointer that can delete a
single checkpoint, after each tool-calling iteration.
`checkpoint_every_n_iterations` sets that last interval, and the turn's
final save deletes the iteration saves it supersedes. The
[durability mode](#durability-modes) decides whether those iteration
saves are awaited, written in the background, or skipped. Resume a
thread by re-running with the same `thread_id`: same code, same case
context, different process, different day.

This is the continuity story for production agents. A case outlives a
process: a refund case opens Friday, the service redeploys over the
weekend, and Monday's run picks up the same conversation — every
lookup, every approval, carried forward. Without a checkpointer the
agent forgets what it already did and re-litigates the case from
scratch. With one, the same `thread_id` round-trips through restarts,
across containers, and across regions, so the record of *what was done
and why* survives.

The same story on an on-call rotation: a case is worked at 02:00, the
runner restarts, the on-call hands off, and the next shift must pick up the
*same* case — every lookup, every action taken, the record carried forward.

{{ locus_model_tabs }}

```python
from locus.agent import Agent
from locus.memory.backends import S3Backend
from locus.tools import tool


@tool
def lookup_ticket(ticket_id: str) -> str:
    """Return the current record for a support ticket."""
    return TICKETS[ticket_id]


agent = Agent(
    model="{{ locus_model }}",
    tools=[lookup_ticket],
    checkpointer=S3Backend(
        bucket="case-checkpoints",
        endpoint_url="https://<namespace>.compat.objectstorage.<region>.oraclecloud.com",
    ),
)

# 02:00 — the case is worked, the runner crashes mid-loop
agent.run_sync("Work ticket T-4821 and record what you did.", thread_id="case-4821")

# 08:00 — next shift, different process, same thread_id: the saved case state is loaded back
agent.run_sync("What did we do on this ticket, and on what evidence?", thread_id="case-4821")
```

!!! note "What a checkpoint is — and isn't"
    A checkpoint is the agent's `AgentState` serialised to plain,
    mutable JSON. Each save the agent makes mints a new checkpoint id and
    writes a new record — a new `<checkpoint_id>.json` for
    `FileCheckpointer`, a new object plus metadata for `S3Backend`, a new
    `<thread_id>:<checkpoint_id>` entry for the `redis_checkpointer` /
    `postgresql_checkpointer` / `mysql_checkpointer` factories. Only
    bookkeeping is overwritten: the `<thread_id>/_latest` pointer object
    for `S3Backend`, and for the factories a `<thread_id>:latest` copy
    plus the `<thread_id>:_checkpoints` index. Earlier checkpoints stay
    until you delete them or a retention rule (TTL, lifecycle policy,
    `vacuum`) expires them. What you do not get is signing or any
    integrity guarantee on the stored bytes: anything with write access
    to the store can alter a checkpoint in place. It gives you
    **durability and resumability**, not a tamper-evident audit log. For
    a tamper-evident record of decisions, route those through the
    [`AuditTrail`](control-layer.md) hash chain instead.

## Durability modes

Every mode writes the run's final state and the state it pauses on
before control returns. `durability` decides how the **iteration**
checkpoints — the saves after each tool-calling iteration — are made.
Set it on the agent (`Agent(durability=...)`) or per run
(`agent.run(..., durability=...)`, also on `arun` and `run_sync`); a
`StateGraph` takes it on `GraphConfig.durability` for its per-step
saves.

| Mode | Iteration checkpoints | Crash mid-run loses | Use when |
|---|---|---|---|
| `"sync"` (default) | Written and awaited before the next model call. | At most the iteration in flight. | Tool calls cost money or cannot be undone, and the model call dominates latency anyway. |
| `"async"` | Scheduled on the event loop while the next model call runs; every pending write is awaited before the final save. | The writes still in flight. | Many short iterations on a remote checkpointer; the thread's history still fills in. |
| `"exit"` | None. One save at the end of the run (and before a pause). | Everything since the previous checkpoint. | Short runs, cheap tools, or a backend billed per write. |

The two cheaper modes have a window the default does not: a process
killed after a tool returned but before the run's next save leaves no
checkpoint of that call, and a resumed run would issue it again. The
[tool-effect ledger](idempotency.md#the-tool-effect-ledger) closes it:
with `durability="async"` or `"exit"` the executor records every
executed call in the thread's checkpointer the moment it returns (one
small write per call, under the sidecar thread `<thread_id>~ledger`),
and `agent.continue_turn(thread_id)` or
`agent.resume(invocation_id=..., thread_id=...)` replays the recorded
results instead of running the tools again. `Agent(tool_ledger=True)`
turns the ledger on for `"sync"` too, closing its smaller window at the
same per-call cost; `tool_ledger=False` turns it off everywhere.

```python
agent = Agent(model=..., tools=[...], checkpointer=cp, durability="async")

# The caller keeps the id; a crashed run is picked up by it.
agent.run_sync("Work ticket T-4821.", thread_id="case-4821", invocation_id="job-77")
# ... the process dies mid-run; a new one continues the same turn:
async for event in agent.resume(invocation_id="job-77", thread_id="case-4821"):
    ...
```

### Overhead

Measured with `scripts/bench_durability.py` on a scripted agent — ten
iterations, one tool call each, a 2 KB tool result, model and tool cost
near zero — so the figures are the checkpoint and ledger traffic alone,
per iteration. Median of eleven runs on one machine, PostgreSQL 17 on
localhost; your network to the database sets the real scale.

| Checkpointer | Mode | `tool_ledger=False` | `tool_ledger=True` |
|---|---|---|---|
| `MemoryCheckpointer` | `sync` | 0.5 ms | 0.5 ms |
| | `async` | 0.5 ms | 0.5 ms |
| | `exit` | 0.4 ms | 0.4 ms |
| `postgresql_checkpointer()` (adapter) | `sync` | 6.0 ms | 5.6 ms |
| | `async` | 2.6 ms | 4.4 ms |
| | `exit` | 0.6 ms | 2.7 ms |
| `PgCheckpointer` | `sync` | 2.2 ms | 3.0 ms |
| | `async` | 1.7 ms | 2.4 ms |
| | `exit` | 0.5 ms | 1.5 ms |

How to read it. The left column is the cost of the checkpoint schedule
alone: `sync` pays the awaited write every iteration, `async` hides most
of it behind the next model call, `exit` pays nothing until the end. The
right column adds one ledger record per tool call — about 2 ms through
the adapter, whose every save is four statements (the record, a `latest`
row, an index read and write) whatever the payload, and about 1 ms on
`PgCheckpointer`, whose save is one upsert. So on PostgreSQL
`async` with the ledger costs what `sync` without it costs through the
adapter, and `exit` with the ledger costs less than `async` without it;
on `PgCheckpointer` every combination is cheaper than the adapter's
default. In memory nothing registers. The ledger also writes a manifest
once per turn and deletes the sidecar once at the end. With a real model
call of a second or more per iteration, none of these numbers is visible;
they matter for tight tool loops on a remote database and for backends
billed per write.

## Time travel

Every checkpoint is a restore point, and every checkpointer exposes the
same three operations over them: **list** a thread's checkpoints with
what each one is, **fork** a new thread from one of them, and **run or
resume from** one of them in place. By default a thread has one
checkpoint per turn, plus one for each pause and each
`agent.update_state`; `Agent(keep_iteration_checkpoints=True)` keeps a
restore point after every tool-calling iteration as well.

```python
from locus.agent import Agent
from locus.memory import CheckpointInfo
from locus.memory.backends import MemoryCheckpointer

cp = MemoryCheckpointer()
agent = Agent(model="{{ locus_model }}", checkpointer=cp)
agent.run_sync("Open ticket T-4821.", thread_id="case-4821")
agent.run_sync("Refund the customer in full.", thread_id="case-4821")


async def what_if() -> None:
    history: list[CheckpointInfo] = await cp.list("case-4821")
    for info in history:  # newest first
        print(info.checkpoint_id, info.iteration, info.size, info.parent_checkpoint_id)
    opened = history[-1]

    # A new thread that stops where the ticket was opened; the original is untouched.
    alt = await cp.fork("case-4821", opened.checkpoint_id, new_thread_id="case-4821-partial")
    await agent.arun("Refund half and escalate.", thread_id=alt)

    # Or branch the thread itself: this turn descends from `opened`, and the
    # refund turn stays in the history beside it.
    await agent.arun("Hold the refund.", thread_id="case-4821", from_checkpoint=opened.checkpoint_id)
```

`cp.list(thread_id, limit=10)` returns `CheckpointInfo` records, newest
first: `checkpoint_id`, `thread_id`, `created_at`, the `iteration` the
state had reached, the `size` of the checkpoint in bytes as that backend
stores it (the serialised state as compact JSON for most backends, the
stored row's text for `PgCheckpointer`'s `jsonb` — compare sizes within
one backend, not across them), the `parent_checkpoint_id` it descends
from, `forked_from` (the thread the parent lives in, for a fork's first
checkpoint) and the `metadata` it was saved with. The parent is
recorded in the checkpoint's metadata when the writer knew it was not
the thread's previous checkpoint — a fork's first checkpoint, an
`update_state` of an older checkpoint, the first lasting save of a run
started `from_checkpoint=` — and is the previous checkpoint everywhere
else, since a thread's history is otherwise a line. Only a thread's
first checkpoint has no parent: with a `limit`, the oldest checkpoint
described still names the one before it.

`cp.fork(thread_id, checkpoint_id=None, *, new_thread_id=None)` saves the
state at that checkpoint (the latest by default) as the first checkpoint
of a new thread (`<thread_id>-fork-<8 hex>` unless named) and returns
the new thread's id. `Agent.fork` and `StateGraph.afork` call it. A
fork takes the thread's
[tool-effect ledger](idempotency.md#the-tool-effect-ledger) with it
**up to the forked checkpoint's iteration and no further**: the records
of calls the forked state already holds are copied, re-keyed under the
new thread, and the records of what the parent did past the fork point
are not, so a run on the fork — even one continuing the same
`invocation_id` — replays nothing from beyond the fork.

`agent.run(prompt, thread_id=..., from_checkpoint=...)` (also `arun`,
`run_sync`) starts the turn from the thread's state at that checkpoint
instead of its latest; `agent.resume(response, thread_id=...,
from_checkpoint=...)` continues a *paused* turn from the checkpoint it
paused on, discarding any later answer — the way to answer a question
differently. Both need `thread_id` and a checkpointer, raise
`LookupError` for a checkpoint the thread does not have, and record the
checkpoint as the parent of the run's first lasting save — the run's
later saves follow on from that one, so the branch is a chain in `list`
and in the drawing below, not a star. The checkpoints written after the
branch point stay, and the checkpoint the run started from is never
deleted, even when it is an iteration save of a turn that did not
finish; nothing is deleted.

`draw_mermaid_history(cp, *thread_ids)`
(`locus.multiagent.visualize`; `StateGraph.adraw_mermaid_history` for a
graph's threads) renders the histories as one Mermaid flowchart: a lane
per thread, oldest checkpoint first, an arrow from each checkpoint to the
ones that descend from it, and — when a thread and its fork are both
given — a dashed `fork` arrow from the checkpoint the fork was taken at,
so the fork reads as a branch of its parent. Node ids are the thread and
checkpoint ids made safe for Mermaid; one that needed a character
replaced carries a short hash of the original, so `a-b` and `a_b` never
share a node, and labels show the ids as they are.

```text
graph LR
    subgraph thread_case_4821_dd05c8c4["case-4821"]
        direction LR
        cp_case_4821_dd05c8c4__a1["a1…<br/>iteration 1"]
        cp_case_4821_dd05c8c4__b2["b2…<br/>iteration 1"]
    end
    subgraph thread_case_4821_partial_dfe2003b["case-4821-partial"]
        direction LR
        cp_case_4821_partial_dfe2003b__f0["f0…<br/>iteration 1<br/>fork"]
        cp_case_4821_partial_dfe2003b__c3["c3…<br/>iteration 1"]
    end
    cp_case_4821_dd05c8c4__a1 --> cp_case_4821_dd05c8c4__b2
    cp_case_4821_partial_dfe2003b__f0 --> cp_case_4821_partial_dfe2003b__c3
    cp_case_4821_dd05c8c4__a1 -.->|fork| cp_case_4821_partial_dfe2003b__f0
```

### Per backend

Every backend already keeps each checkpoint as its own record, so
history is complete everywhere; what differs is what `list` has to read
and how the history is bounded. A thread's storage grows with the
square of the conversation's length — each checkpoint holds the whole
conversation — so a long-lived thread wants a bound, a retention rule,
or `message_retention` on `PgCheckpointer`. No bound ever trims the
tool-effect ledger's sidecar thread (`<thread_id>~ledger`): its records
are one per executed call, and dropping one would run that call again
on a replay.

| Backend | History | What `list` reads | Bound | Storage per checkpoint |
|---|---|---|---|---|
| `MemoryCheckpointer` | every save, for the life of the process | the in-memory records; no state is loaded | `max_checkpoints_per_thread` | the serialised state, in memory |
| `FileCheckpointer` | one `<checkpoint_id>.json` per save | the head of each file (the step fields and metadata are written before the state) | `max_checkpoints_per_thread` | one file |
| `redis_checkpointer()`, `postgresql_checkpointer()`, `mysql_checkpointer()`, `opensearch_checkpointer()`, `oracle_checkpointer()` (the adapter) | one `<thread_id>:<checkpoint_id>` key per save, plus the `<thread_id>:_checkpoints` index | one read of the index key | `max_checkpoints_per_thread` on the factory | one key, plus the shared `latest` copy and index entry |
| `PgCheckpointer` | one row per save | one query over the row's own columns; the state column is not returned (`size` is the row's `jsonb` as text) | `keep_checkpoints` | one row |
| `S3Backend`, `OCIBucketBackend` | one `<checkpoint_id>.json` and `.meta.json` per save | one listing, then one GET of the small `.meta.json` per checkpoint described; the listing's modification times are coarse (S3 keeps them to the second), so checkpoints written within one second are ordered by the exact write time in their `.meta.json`, read for those alone | `max_checkpoints_per_thread`, or a bucket lifecycle rule | two objects |
| `OracleCheckpointSaver` (the reference: one row per checkpoint with `parent_checkpoint_id` kept on the row) | one row per `put` | one query; the rows carry their data | none built in (`delete_thread`) | one row |
| `HTTPCheckpointer`, your own `BaseCheckpointer` subclass | whatever the store keeps | the base default: each state is loaded to read its iteration and size, `get_metadata` for the rest when the backend has it | none unless you add one | the store's |

Through the adapter factories, each save or removal changes the index in
one operation the store makes atomic — a Redis hash field (`HSET` /
`HDEL`), one `UPDATE` of the index row's JSON array on PostgreSQL and
MySQL, one scripted `_update` of the index document on OpenSearch, a
rewrite of the index row under its row lock (`SELECT ... FOR UPDATE`) on
Oracle — so two processes saving to the same thread at once both keep
every entry. An index an earlier release wrote on Redis is read as before
and moved into the hash on the thread's next save or removal; reading
those strings ends in 7.0.0, and a process still on the earlier release
lists only the checkpoints it wrote itself after the move, so upgrade the
processes that share one Redis store together. A backend of your own
without those operations updates the index under a per-process lock:
safe within one process, not between processes.

A bounded backend deletes the surplus from the oldest end after every
save — several saves within one second included — so `fork` and `list`
reach back only as far as the bound; the oldest checkpoint kept reports
no parent.

## Picking a backend

| Situation | Backend |
|---|---|
| Unit tests, single-process REPL | `MemoryCheckpointer` |
| Local development, single machine | `FileCheckpointer` |
| Multi-worker deployment, fast access, TTLs | `redis_checkpointer()` (wraps `RedisBackend`) |
| Teams running Postgres that want SQL queries on metadata | `postgresql_checkpointer()` (wraps `PostgreSQLBackend`) |
| Teams running MySQL that want official Connector/Python async access | `mysql_checkpointer()` (wraps `MySQLBackend`) |
| Need full-text search across past runs | `opensearch_checkpointer()` (wraps `OpenSearchBackend`) |
| Object storage (S3 / MinIO / R2), serverless, lifecycle policies | `S3Backend` |
| Oracle Database 26ai / Autonomous Database — durable threads next to your data | `oracle_checkpointer()` (wraps `OracleBackend`); `OracleCheckpointSaver` for a versioned history your code drives |
| OCI Object Storage | `OCIBucketBackend` |
| Already have a checkpoint service over HTTP | `HTTPCheckpointer` |

Production default: object storage — `OCIBucketBackend` on OCI Object
Storage, `S3Backend` for S3-compatible stores (S3, MinIO, R2, or the
OCI S3-compatibility endpoint the examples use). No database to run, no
Redis to scale; lifecycle policies handle retention and IAM handles
access.

## Getting started

### Oracle Database 26ai: `oracle_checkpointer()`

Durable threads in the same database as the data the agent acts on. The
factory wraps `OracleBackend` in the adapter and returns a ready
checkpointer. `OracleCheckpointSaver` is a separate, versioned history
store, not a drop-in for `Agent(checkpointer=...)`: it is not a
`BaseCheckpointer`, and application code drives it with `put()` /
`get()` after each run, keeping one row per checkpoint with a parent
link (see [Oracle Database 26ai](oracle-26ai.md) and
[notebook 43](../notebooks/notebook_43_oracle_versioned_saver.md)).
Passing it as `Agent(checkpointer=...)` raises `TypeError` at
construction, naming `oracle_checkpointer()`.

```python
import os

from locus.agent import Agent
from locus.memory.backends import oracle_checkpointer

cp = oracle_checkpointer(
    dsn=os.environ["ORACLE_DSN"],
    user=os.environ["ORACLE_USER"],
    password=os.environ["ORACLE_PASSWORD"],
    wallet_location=os.environ["ORACLE_WALLET"],        # Autonomous Database wallet
    wallet_password=os.environ.get("ORACLE_WALLET_PASSWORD", ""),
    table_name="locus_checkpoints",
)
agent = Agent(model="oci:openai.gpt-5.5", checkpointer=cp)
agent.run_sync("My name is Alice.", thread_id="t1")
agent.run_sync("What's my name?", thread_id="t1")   # resumes from the database
```

### Local: `FileCheckpointer`

```python
from locus.memory.backends.file import FileCheckpointer

agent = Agent(
    model=...,
    tools=[...],
    checkpointer=FileCheckpointer(base_dir="./threads"),
)
```

One directory per `thread_id` — one per case — with a JSON file per
checkpoint.
Zero dependencies; the on-disk case files are grep-able when you need to
inspect a thread's saved state by hand.

### Production: `S3Backend`

```python
from locus.memory.backends import S3Backend

agent = Agent(
    model=...,
    tools=[...],
    # S3Backend implements BaseCheckpointer directly — no adapter needed.
    checkpointer=S3Backend(
        bucket="agent-thread-checkpoints",
        # or a MinIO / R2 endpoint
        endpoint_url="https://<namespace>.compat.objectstorage.<region>.oraclecloud.com",
        prefix="prod/",
    ),
)
```

S3-compatible object storage (S3 / MinIO / Cloudflare R2) with
bucket-level lifecycle rules (retain closed cases for your audit
window, then expire), region replication, and IAM-controlled access.
Workers across processes / pods see the same cases.

### Postgres: `postgresql_checkpointer`

```python
from locus.memory.backends import postgresql_checkpointer

agent = Agent(
    model=...,
    tools=[...],
    checkpointer=postgresql_checkpointer(
        dsn="postgresql://user:pass@host:5432/locus",
        schema_name="locus_threads",        # default "public"
        table_name="locus_checkpoints",     # default "checkpoints"
    ),
)
```

Tables auto-created on first save. Index on `thread_id` plus a JSONB
column for ad-hoc metadata queries.

### MySQL: `mysql_checkpointer`

```python
from locus.memory.backends import mysql_checkpointer

agent = Agent(
    model=...,
    tools=[...],
    checkpointer=mysql_checkpointer(
        dsn="mysql://user:pass@host:3306/locus",
        table_name="locus_threads",
    ),
)
```

Tables auto-created on first save. Uses the official
`mysql-connector-python` asyncio API, MySQL `JSON` columns, and
`JSON_CONTAINS` metadata queries.

### Redis: `redis_checkpointer`

```python
from locus.memory.backends import redis_checkpointer

agent = Agent(
    model=...,
    tools=[...],
    checkpointer=redis_checkpointer(
        url="redis://host:6379/0",
        ttl_seconds=86_400,        # auto-expire after 24h
    ),
)
```

Fastest reads, optional TTL for short-lived threads that
auto-expire once the case is closed.

### SQL: `PostgreSQLBackend` / `MySQLBackend`

If your stack is already on PostgreSQL or MySQL, the SDK ships storage
backends so agent state can live alongside your app data. The backend
upserts one row per key. Through the `*_checkpointer()` factories above,
that key is `<thread_id>:<checkpoint_id>`, so each save adds a row, and a
`<thread_id>:latest` row and a `<thread_id>:_checkpoints` index row are
updated in place. `list_threads` and `vacuum` work over those rows, and
`search_data` matches against the JSON `data` column. These are
**storage backends**, not native checkpointers, so go
through the `postgresql_checkpointer` / `mysql_checkpointer` factories
(which wrap the backend in a `StorageBackendAdapter`) — see the
[Postgres block above](#postgres-postgresql_checkpointer) for the call.
`mysql_checkpointer` mirrors the same shape on the official MySQL
Connector/Python asyncio driver.

## Two checkpointer shapes — how to wire them

The SDK has **two** kinds of checkpointer implementations and you need
to wire them differently. Both sit behind the `BaseCheckpointer`
contract the loop calls — loading the thread's newest checkpoint before
the first model call, saving at its three save points — as the drawing
shows:

{{ locus_diagram checkpointer-shapes }}

1. **Native checkpointers** implement `BaseCheckpointer` directly and
   accept `AgentState`:
    - `MemoryCheckpointer`, `FileCheckpointer`, `HTTPCheckpointer`,
      `S3Backend` and `OCIBucketBackend` (despite the `Backend` name,
      both subclass `BaseCheckpointer`), and
      [`PgCheckpointer`](../pg-checkpointer.md).
    - Pass straight to `Agent(checkpointer=...)`.

2. **Storage backends** expose a simpler dict-shaped `save(thread_id,
   data)` interface and need adapter wrapping:
    - `RedisBackend`, `PostgreSQLBackend`, `MySQLBackend`,
      `OpenSearchBackend`, `OracleBackend`.
    - Wrap with the matching factory — `redis_checkpointer()`,
      `postgresql_checkpointer()`, `mysql_checkpointer()`,
      `opensearch_checkpointer()`, `oracle_checkpointer()`; there is
      **no `.as_checkpointer()` method**.

```python
# WRONG — passing a storage backend directly fails: the agent calls
# save(state, thread_id) but RedisBackend.save expects (thread_id, data)
from locus.memory.backends.redis import RedisBackend
agent = Agent(..., checkpointer=RedisBackend(url="..."))   # ✗

# RIGHT — use the factory
from locus.memory.backends import redis_checkpointer
agent = Agent(..., checkpointer=redis_checkpointer(url="..."))  # ✓
```

The `*_checkpointer()` factory wraps the storage backend in a
`StorageBackendAdapter` that translates the agent's `save(state,
thread_id)` calls into the backend's `save(thread_id, dict)` shape.
(`s3_checkpointer()` and `oci_bucket_checkpointer()` are the exceptions:
they return the native `S3Backend` / `OCIBucketBackend` unchanged, since
both already implement `BaseCheckpointer`.)

## Capabilities — feature detection

Each backend advertises which optional operations it supports, so
your code can do the right thing at runtime:

```python
caps = checkpointer.capabilities

if caps.search:
    hits = await checkpointer.search("error handling")

if caps.branching:
    await checkpointer.copy_thread("main", "experiment")

if caps.vacuum:
    await checkpointer.vacuum(older_than_days=30)

if caps.list_threads:
    threads = await checkpointer.list_threads()
```

| Capability | What it adds |
|---|---|
| `search` | Full-text search across all stored checkpoints. |
| `metadata_query` | Query by metadata fields (tags, agent_id, etc). |
| `vacuum` | Delete checkpoints older than a threshold. |
| `branching` | Copy / fork a thread (useful for what-if experiments). |
| `ttl` | Time-to-live / auto-expiration. |
| `list_threads` | Enumerate stored thread IDs. |
| `list_with_metadata` | List threads with their latest metadata. |
| `persistent_checkpoint_ids` | Checkpoint IDs survive restart. |

## Building your own

Subclass `BaseCheckpointer` and implement the three abstract methods —
`save`, `load`, `list_checkpoints` — and override `delete` / `exists`
if the defaults don't fit. Advertise your `capabilities`. Pass the
instance directly to `Agent(checkpointer=...)` — no glue needed.

See [how-to/custom-checkpointer](../how-to/custom-checkpointer.md)
for a worked example.

## Cross-thread store

Checkpointers persist *one case's* state. The companion abstraction —
`BaseStore` — persists key-value data **across** cases: a team
playbook, a pricing-exception cache, anything that should outlive a
single case.

```python
import asyncio
from locus.memory.store import InMemoryStore   # tests / REPL


async def main():
    store = InMemoryStore()
    namespace = ("locus_memory", "support")
    await store.put(
        namespace,
        "refund_policy",
        {"content": "Refunds over $500 need manager approval"},
    )
    hit = await store.get(namespace, "refund_policy")
    hits = await store.search(namespace, query=None, limit=5)
    # -> [StoreItem(namespace=("locus_memory", "support"),
    #               key="refund_policy", value={...}, ...)]


asyncio.run(main())
```

The interface is `put / get / delete / list_keys`, keyed on a
`(namespace, key)` pair, plus optional `search` and
`search_by_embedding` that backends advertise through
`store.capabilities`. The [`LLMMemoryManager`](memory-manager.md) builds on this
to give an agent a long-term memory layer; you can also use the store
directly for anything cross-case that doesn't need LLM extraction
(product catalogues, customer allowlists, rate-limit counters).

### The built-in store: `InMemoryStore`

`InMemoryStore` (`locus.memory.store`) is the in-process
implementation — namespaces and keys live in a dict, and it is the store
the examples pass to [`LLMMemoryManager`](memory-manager.md), which
requires a `store` argument rather than defaulting to one. For a durable
cross-thread store the SDK ships `HolographicStore` (SQLite + FTS5, no
external infrastructure; give it a file `path`, since the default is
`:memory:`) and `PgMemory` (Postgres + pgvector with per-tenant
row-level security), both in `locus.memory.store_backends`; subclass
`BaseStore` only if neither fits — see [Memory API](../api/memory.md).

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `TypeError` on save (e.g. `save()` got an unexpected positional/`AgentState` where a dict was expected) | Storage backend (`RedisBackend` etc.) passed directly: its `save(thread_id, data)` signature doesn't match the agent's `save(state, thread_id)`. Wrap it with the matching `*_checkpointer()` factory. |
| Open cases forgotten between deployments | `FileCheckpointer` directory inside an ephemeral container — the saved state dies with the pod. Mount a volume, or move to `S3Backend`. |
| Two replicas show different case state for the same thread | The checkpointer isn't shared between replicas, so one worker doesn't know the refund was already issued. `FileCheckpointer` is per-host; switch to a centralised backend (Redis, Postgres, MySQL, S3). |
| Slow first save | Some backends auto-create schema on first call. Pre-create in your deployment script if startup latency matters. |

## Source

- [`locus.memory.backends`](https://github.com/oracle/locus/tree/main/src/locus/memory/backends) — every backend, plus `StorageBackendAdapter` and the `redis_checkpointer` / `postgresql_checkpointer` / `mysql_checkpointer` / `opensearch_checkpointer` / `oracle_checkpointer` / `s3_checkpointer` / `oci_bucket_checkpointer` factory functions.

## See also

- [State](state.md) — what `AgentState` actually contains.
- [Conversation management](conversation-management.md) — higher-level patterns built on checkpointers.
- [Idempotency](idempotency.md) — in-run deduplication of identical tool calls, and the tool-effect ledger a resumed run replays from.
- [How-to: custom checkpointer](../how-to/custom-checkpointer.md) — write your own backend.
