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
from tulip.agent import Agent
from tulip.memory.backends import S3Backend

agent = Agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search, summarise],
    checkpointer=S3Backend(
        bucket="my-app-checkpoints",
        endpoint_url="https://s3.amazonaws.com",
    ).as_checkpointer(),
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
| Multi-worker deployment, fast access, TTLs | `RedisBackend` |
| Postgres shop, want SQL queries on metadata | `PostgreSQLBackend` |
| MySQL shop, want official Connector/Python async access | `MySQLBackend` |
| Need full-text search across past runs | `OpenSearchBackend` |
| **Object storage (S3 / MinIO / R2), serverless, lifecycle policies** | `S3Backend` |
| Already have a checkpoint service over HTTP | `HTTPCheckpointer` |

Production default: `S3Backend`. No DB to run, no Redis to scale,
lifecycle policies handle retention, IAM handles auth.

## Getting started

### Local: `FileCheckpointer`

```python
from tulip.memory.backends.file import FileCheckpointer

agent = Agent(
    model=...,
    tools=[...],
    checkpointer=FileCheckpointer(directory="./threads"),
)
```

One JSON file per `thread_id` in the directory. Zero dependencies,
plays well with `git stash` for "save my agent state" workflows.

### Production: `S3Backend`

```python
from tulip.memory.backends import S3Backend

agent = Agent(
    model=...,
    tools=[...],
    checkpointer=S3Backend(
        bucket="my-app-checkpoints",
        endpoint_url="https://s3.amazonaws.com",  # or a MinIO / R2 endpoint
        prefix="prod/",
    ).as_checkpointer(),
)
```

S3-compatible object storage (S3 / MinIO / Cloudflare R2) with
bucket-level lifecycle rules ("delete threads older than 90 days"),
region replication, and IAM-controlled access. Workers across
processes / pods see the same threads.

### Postgres: `postgresql_checkpointer`

```python
from tulip.memory.backends import postgresql_checkpointer

agent = Agent(
    model=...,
    tools=[...],
    checkpointer=postgresql_checkpointer(
        dsn="postgresql://user:pass@host:5432/tulip",
        schema="tulip_threads",
    ),
)
```

Tables auto-created on first save. Index on `thread_id` plus a JSONB
column for ad-hoc metadata queries.

### MySQL: `mysql_checkpointer`

```python
from tulip.memory.backends import mysql_checkpointer

agent = Agent(
    model=...,
    tools=[...],
    checkpointer=mysql_checkpointer(
        dsn="mysql://user:pass@host:3306/tulip",
        table_name="tulip_threads",
    ),
)
```

Tables auto-created on first save. Uses the official
`mysql-connector-python` asyncio API, MySQL `JSON` columns, and
`JSON_CONTAINS` metadata queries.

### Redis: `redis_checkpointer`

```python
from tulip.memory.backends import redis_checkpointer

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

### SQL: `PostgreSQLBackend` / `MySQLBackend`

If your stack is already on PostgreSQL or MySQL, the SDK ships native
checkpointers so agent state can live alongside your app data. One row
per `thread_id` (upsert), `list_threads` / `vacuum` / `search` over a
JSON column.

```python
from tulip.memory.backends import PostgreSQLBackend

agent = Agent(
    model=...,
    tools=[...],
    checkpointer=PostgreSQLBackend(
        dsn="postgresql://tulip_app:pass@host:5432/tulip",
        table_name="tulip_checkpoints",
    ).as_checkpointer(),
)
```

`MySQLBackend` mirrors the same shape on the official MySQL
Connector/Python asyncio driver.

## Two checkpointer shapes — the gotcha to know

The SDK has **two** kinds of checkpointer implementations and you need
to wire them differently:

1. **Native checkpointers** implement `BaseCheckpointer` directly and
   accept `AgentState`:
   - `MemoryCheckpointer`, `FileCheckpointer`, `HTTPCheckpointer`.
   - Pass straight to `Agent(checkpointer=...)`.

2. **Storage backends** expose a simpler dict-shaped interface and
   need adapter wrapping:
   - `RedisBackend`, `PostgreSQLBackend`, `MySQLBackend`,
     `OpenSearchBackend`, `S3Backend`.
   - Use `.as_checkpointer()`.

```python
# WRONG — passing a storage backend directly will fail at save time
from tulip.memory.backends.redis import RedisBackend
agent = Agent(..., checkpointer=RedisBackend(url="..."))   # ✗

# RIGHT — use the factory
from tulip.memory.backends import redis_checkpointer
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
from tulip.memory.store import InMemoryStore   # tests / REPL

store = InMemoryStore()
store.put(("tulip_memory", "user"), "role", {"content": "Senior Python engineer"})
hit = store.get(("tulip_memory", "user"), "role")
```

The interface is `put / get / list / delete` keyed on a `(namespace,
key)` pair. The [`LLMMemoryManager`](memory-manager.md) builds on this
to give an agent a long-term memory layer; you can also use the store
directly for anything cross-thread that doesn't need LLM extraction
(API tokens, user preferences, rate-limit counters).

### The built-in store: `InMemoryStore`

The SDK ships an in-process `InMemoryStore` implementing the
`BaseStore` interface. Namespaces and keys live in a dict; it's the
default the [`LLMMemoryManager`](memory-manager.md) uses. For a durable
cross-thread store, subclass `BaseStore` over your backend of choice.

```python
from tulip.memory.store import InMemoryStore

store = InMemoryStore()
await store.put(("memory", "u42"), "fact-1", {"note": "user likes cats"})
hits = await store.search(("memory", "u42"), query=None, limit=5)
```

## Common gotchas

| Symptom | Likely cause |
|---|---|
| `AttributeError: 'RedisBackend' has no attribute 'save'` (with `state` arg) | Storage backend passed without the adapter. Use `.as_checkpointer()`. |
| Threads forgotten between deployments | `FileCheckpointer` directory inside an ephemeral container. Mount a volume, or move to `S3Backend`. |
| Two replicas show different conversation state for the same thread | The checkpointer isn't shared between replicas. `FileCheckpointer` is per-host; switch to a centralised backend (Redis, Postgres, MySQL, S3). |
| Slow first save | Some backends auto-create schema on first call. Pre-create in your deployment script if startup latency matters. |

## Source

- [`tulip.memory.backends`](https://github.com/tuliplabs-ai/sdk-python/tree/main/src/tulip/memory/backends) — every backend, plus `StorageBackendAdapter` and `.as_checkpointer()`.

## See also

- [State](state.md) — what `AgentState` actually contains.
- [Conversation management](conversation-management.md) — higher-level patterns built on checkpointers.
- [Idempotency](idempotency.md) — replay-safe side effects when a checkpoint resume re-issues a tool call.
- [How-to: custom checkpointer](../how-to/custom-checkpointer.md) — write your own backend.
