# Checkpointers

A checkpointer is the contract for **persisting agent state** between
runs. Pass one to `Agent(checkpointer=...)` and the agent saves
`AgentState` after every iteration; resume a conversation by re-running
with the same `thread_id`. Same code, same context, different process,
different day.

This is the durability story for production agents. Without a
checkpointer your agent forgets every conversation when the process
exits. With one, the same `thread_id` round-trips through restarts,
across containers, and across regions.

```python
from locus.agent import Agent
from locus.memory.backends import oci_bucket_checkpointer

agent = Agent(
    model="oci:openai.gpt-5.5",
    tools=[search, summarise],
    checkpointer=oci_bucket_checkpointer(
        bucket_name="my-app-checkpoints",
        namespace="my-tenancy-namespace",
    ),
)

# Day 1
agent.run_sync("I'm planning a trip to Tokyo.", thread_id="user-c42")

# Day 2 — different process, same thread_id, conversation continues
agent.run_sync("What were we discussing?", thread_id="user-c42")
```

## Picking a backend

| Situation | Backend |
|---|---|
| Unit tests, single-process REPL | `MemoryCheckpointer` |
| Local development, single machine | `FileCheckpointer` |
| Multi-worker deployment, fast access, TTLs | `redis_checkpointer` |
| Postgres shop, want SQL queries on metadata | `postgresql_checkpointer` |
| Need full-text search across past runs | `opensearch_checkpointer` |
| Oracle Database shop, want JSON queries | `oracle_checkpointer` |
| **[Oracle 26ai][oracle-db] with versioned history + pending writes** | `OracleCheckpointSaver` — LangGraph-shape, native |
| **[OCI Object Storage][oci-objectstorage], serverless, lifecycle policies** | `oci_bucket_checkpointer` — the day-1 OCI path |
| Already have a checkpoint service over HTTP | `HTTPCheckpointer` |

Default on OCI: `oci_bucket_checkpointer`. No DB to run,
no Redis to scale, lifecycle policies handle retention, IAM handles
auth.

## Getting started

### Local: `FileCheckpointer`

```python
from locus.memory.backends.file import FileCheckpointer

agent = Agent(
    model=...,
    tools=[...],
    checkpointer=FileCheckpointer(directory="./threads"),
)
```

One JSON file per `thread_id` in the directory. Zero dependencies,
plays well with `git stash` for "save my agent state" workflows.

### Production: `oci_bucket_checkpointer`

```python
from locus.memory.backends import oci_bucket_checkpointer

agent = Agent(
    model=...,
    tools=[...],
    checkpointer=oci_bucket_checkpointer(
        bucket_name="my-app-checkpoints",
        namespace="my-tenancy-namespace",
        compartment_id="ocid1.compartment...",
        prefix="prod/",
    ),
)
```

OCI Object Storage with bucket-level lifecycle rules ("delete threads
older than 90 days"), region replication, and IAM-controlled access.
Workers across processes / pods see the same threads.

### Postgres: `postgresql_checkpointer`

```python
from locus.memory.backends import postgresql_checkpointer

agent = Agent(
    model=...,
    tools=[...],
    checkpointer=postgresql_checkpointer(
        dsn="postgresql://user:pass@host:5432/locus",
        schema="locus_threads",
    ),
)
```

Tables auto-created on first save. Index on `thread_id` plus a JSONB
column for ad-hoc metadata queries.

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

Fastest reads, optional TTL for ephemeral conversations.

### Oracle 26ai: `oracle_checkpointer` + `OracleCheckpointSaver`

If your stack is already on **[Oracle Autonomous Database 26ai][adb]**,
Oracle Generative AI – Multi-Agent Locus SDK ships two native
checkpointers — one to colocate agent state with the rest of your app
data, the other to give LangGraph-style versioned history in the same
database.

#### Single-row per thread — `oracle_checkpointer`

The default option. One row per `thread_id` (MERGE-upsert). Survives
restarts, lives in the same schema as your business data, supports
`list_threads` / `vacuum` / `search` over a `CLOB IS JSON` column.

```python
from locus.memory.backends import oracle_checkpointer

agent = Agent(
    model=...,
    tools=[...],
    checkpointer=oracle_checkpointer(
        dsn="mydb_low",                      # tnsnames alias from the wallet
        user="locus_app",                    # least-privileged app schema
        password=os.environ["ORACLE_PW"],
        wallet_location="~/.oci/wallets/mydb",
        wallet_password=os.environ["WALLET_PW"],
        table_name="locus_checkpoints",
    ),
)
```

CLOB columns are pinned via `setinputsizes(... = DB_TYPE_CLOB)` on every
write — large agent states won't trip ORA-01461.

#### Versioned history + pending writes — `OracleCheckpointSaver`

Direct equivalent of `langgraph-oracledb.OracleSaver`. Two tables —
one row per `(thread_id, checkpoint_ns, checkpoint_id)` for full
graph-step history plus a `<table>_writes` table for intra-step
durability. Zero `langchain` / `langgraph` dependency.

```python
from locus.memory.backends import OracleCheckpointSaver

saver = OracleCheckpointSaver(
    dsn="mydb_low",
    user="locus_app",
    password=os.environ["ORACLE_PW"],
    wallet_location="~/.oci/wallets/mydb",
    table_name="locus",                      # creates locus_checkpoints + locus_writes
)
await saver.put(thread_id="t1", checkpoint_id="v1", checkpoint_data={"step": 1})
await saver.put_writes(
    thread_id="t1", checkpoint_id="v1",
    task_id="node-a",
    writes=[("channel-x", "pending-write")],
)
latest = await saver.get(thread_id="t1")               # most-recent checkpoint
history = await saver.list_checkpoints(thread_id="t1")  # walk lineage
```

Notebook walkthrough: [Notebook 07 — Oracle 26ai
checkpointer](../notebooks/notebook_53_oracle_26ai_checkpointer.md).

## Two checkpointer shapes — the gotcha to know

The SDK has **two** kinds of checkpointer implementations and you need
to wire them differently:

1. **Native checkpointers** implement `BaseCheckpointer` directly and
   accept `AgentState`:
   - `MemoryCheckpointer`, `FileCheckpointer`, `HTTPCheckpointer`,
     `OCIBucketBackend`.
   - Pass straight to `Agent(checkpointer=...)`.

2. **Storage backends** expose a simpler dict-shaped interface and
   need adapter wrapping:
   - `RedisBackend`, `PostgreSQLBackend`, `OpenSearchBackend`,
     `OracleBackend`.
   - Use the factory function: `redis_checkpointer(...)`,
     `postgresql_checkpointer(...)`, etc.

```python
# WRONG — passing a storage backend directly will fail at save time
from locus.memory.backends.redis import RedisBackend
agent = Agent(..., checkpointer=RedisBackend(url="..."))   # ✗

# RIGHT — use the factory
from locus.memory.backends import redis_checkpointer
agent = Agent(..., checkpointer=redis_checkpointer(url="..."))  # ✓
```

The `*_checkpointer()` factory wraps the storage backend in a
`StorageBackendAdapter` that translates the agent's `save(state,
thread_id)` calls into the backend's `save(thread_id, dict)` shape.

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
| `branching` | Copy / fork a thread (great for "what-if" experiments). |
| `ttl` | Time-to-live / auto-expiration. |
| `list_threads` | Enumerate stored thread IDs. |
| `list_with_metadata` | List threads with their latest metadata. |
| `persistent_checkpoint_ids` | Checkpoint IDs survive restart. |

## Building your own

Subclass `BaseCheckpointer`, implement `save`, `load`,
`list_checkpoints`, `exists`, `delete`. Advertise your capabilities.
Pass the instance directly to `Agent(checkpointer=...)` — no glue
needed.

See [how-to/custom-checkpointer](../how-to/custom-checkpointer.md)
for a worked example.

## Cross-thread store

Checkpointers persist *one thread's* state. The companion abstraction —
`BaseStore` — persists key-value data **across** threads: a per-user
profile, long-term memory, anything that should outlive a single
conversation.

```python
from locus.memory.store import InMemoryStore   # tests / REPL

store = InMemoryStore()
store.put(("locus_memory", "user"), "role", {"content": "Senior Python engineer"})
hit = store.get(("locus_memory", "user"), "role")
```

The interface is `put / get / list / delete` keyed on a `(namespace,
key)` pair. The [`LLMMemoryManager`](memory-manager.md) builds on this
to give an agent a long-term memory layer; you can also use the store
directly for anything cross-thread that doesn't need LLM extraction
(API tokens, user preferences, rate-limit counters).

### Production: `OracleStore` on Oracle 26ai

The SDK ships a native `BaseStore` implementation backed by Oracle 26ai —
the equivalent of `langgraph-oracledb.OracleStore`, with **zero**
langchain/langgraph dependency. Namespaces persist as primary keys on
a single table; optional vector search runs natively against an
embedding column.

```python
from locus.memory.store_backends import OracleStore

store = OracleStore(
    dsn="mydb_low",
    user="locus_app",
    password=os.environ["ORACLE_PW"],
    wallet_location="~/.oci/wallets/mydb",
    table_name="locus_store",
    dimension=1024,           # set for vector search; omit for plain K/V
)
await store.put(("memory", "u42"), "fact-1", {"note": "user likes cats"})
await store.put_with_embedding(
    ("memory", "u42"), "fact-2", {"note": "user dislikes mornings"},
    embedding=[...],          # 1024-dim vector
)
hits = await store.search_by_embedding(("memory", "u42"), query=[...], limit=5)
```

Same connection envelope as `oracle_checkpointer` — one wallet, one
schema, both checkpoint state and long-term memory live in Oracle.

## Common gotchas

| Symptom | Likely cause |
|---|---|
| `AttributeError: 'RedisBackend' has no attribute 'save'` (with `state` arg) | Storage backend passed without the adapter. Use `redis_checkpointer(...)` factory instead. |
| Threads forgotten between deployments | `FileCheckpointer` directory inside an ephemeral container. Mount a volume, or move to `oci_bucket_checkpointer`. |
| Two replicas show different conversation state for the same thread | The checkpointer isn't shared between replicas. `FileCheckpointer` is per-host; switch to a centralised backend (Redis, Postgres, OCI bucket). |
| Slow first save | Some backends auto-create schema on first call. Pre-create in your deployment script if startup latency matters. |

## Source

- [`locus.memory.backends`](https://github.com/oracle-samples/locus/tree/main/src/locus/memory/backends) — every backend, plus `StorageBackendAdapter` and the `*_checkpointer()` factories.

## See also

- [State](state.md) — what `AgentState` actually contains.
- [Conversation management](conversation-management.md) — higher-level patterns built on checkpointers.
- [Idempotency](idempotency.md) — replay-safe side effects when a checkpoint resume re-issues a tool call.
- [How-to: custom checkpointer](../how-to/custom-checkpointer.md) — write your own backend.

### Oracle reference docs

- [OCI Object Storage — overview](https://docs.oracle.com/iaas/Content/Object/Concepts/objectstorageoverview.htm)
  — buckets, namespaces, IAM policies. Backs the `oci_bucket_checkpointer()` factory.
- [Oracle AI Database 26ai — AI Vector Search User's Guide](https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/index.html)
  — referenced by the Oracle 26ai checkpointer + vector store.

[oracle-db]: https://docs.oracle.com/en/database/oracle/oracle-database/23/index.html
[adb]: https://docs.oracle.com/en/cloud/paas/autonomous-database/index.html
[oci-objectstorage]: https://docs.oracle.com/en-us/iaas/Content/Object/Concepts/objectstorageoverview.htm
