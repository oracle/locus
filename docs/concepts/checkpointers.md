# Checkpointers

A checkpointer is the contract for **persisting agent state** between
runs. Pass one to `Agent(checkpointer=...)` and the agent saves
`AgentState` at three points: when the run finishes or raises; before
it pauses on an interrupt; and, on a checkpointer that can delete a
single checkpoint, after each tool-calling iteration.
`checkpoint_every_n_iterations` sets that last interval, and the turn's
final save deletes the iteration saves it supersedes. Resume a thread by
re-running with the same `thread_id`: same code, same case context,
different process, different day.

This is the continuity story for production agents. A case outlives a
process: a refund case opens Friday, the service redeploys over the
weekend, and Monday's run picks up the same conversation — every
lookup, every approval, carried forward. Without a checkpointer the
agent forgets what it already did and re-litigates the case from
scratch. With one, the same `thread_id` round-trips through restarts,
across containers, and across regions, so the record of *what was done
and why* survives.

The same story, security-operations flavored: a host gets isolated at
02:00, the runner restarts, the on-call hands off, and the next shift
must pick up the *same* investigation — every enriched indicator, every
containment receipt, the case record carried forward.

{{ locus_model_tabs }}

```python
from locus.agent import Agent
from locus.memory.backends import S3Backend

from locus_security import security_toolset

agent = Agent(
    model="{{ locus_model }}",
    # query_siem, enrich_indicator, isolate_host, ...
    tools=security_toolset(allow_containment=True),
    checkpointer=S3Backend(
        bucket="soc-case-checkpoints",
        endpoint_url="https://<namespace>.compat.objectstorage.<region>.oraclecloud.com",
    ),
)

# 02:00 — triage fires, host gets isolated, the runner crashes mid-loop
agent.run_sync("Triage alert A-42 and contain the affected host.", thread_id="case-4821")

# 08:00 — next shift, different process, same thread_id: the saved case state is loaded back
agent.run_sync("What host did we isolate, and on what evidence?", thread_id="case-4821")
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
    [`AuditTrail`](agentic-ai-security.md) hash chain instead.

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
- [Idempotency](idempotency.md) — replay-safe side effects when a checkpoint resume re-issues a tool call.
- [How-to: custom checkpointer](../how-to/custom-checkpointer.md) — write your own backend.
