# Memory

Three orthogonal layers, each with its own contract:

1. **Conversation management** — what to do with the message history
   *within* a single run (keep, window, summarize) when it grows past
   the model's context window.
2. **Cross-thread store** — durable key-value scoped by namespace, used
   for long-term memory entries the agent reads at session start and
   writes at session end.
3. **Long-term memory manager** — the policy layer that decides *what*
   to extract from a run and *when* to retrieve, sitting on top of any
   `BaseStore` backend.

For checkpointing (state persistence between runs), see
[Checkpointers](checkpointers.md) — those backends live in
`locus.memory.backends`, with S3-compatible object storage as a
production target. Note those are *checkpointer* backends (per-thread
state); the cross-thread **store** layer below is a separate contract.

For the concepts, start with [Long-term memory](../concepts/memory-manager.md),
[Conversation management](../concepts/conversation-management.md) and
[the cross-thread store](../concepts/checkpointers.md#cross-thread-store).

## Conversation management

::: locus.memory.conversation.ConversationManager
::: locus.memory.conversation.NullManager
::: locus.memory.conversation.SlidingWindowManager
::: locus.memory.conversation.SummarizingManager

## Cross-thread store

The cross-thread KV the long-term memory manager writes to. Four
implementations ship:

| Store | Backing | Reach for it when |
|---|---|---|
| `InMemoryStore` | in-process dict | tests, and anything that can lose memory on exit |
| `HolographicStore` | SQLite + FTS5 + holographic reduced representations | you want durable memory with **no external infrastructure** |
| `PgMemory` | Postgres + pgvector, per-tenant row-level security | multi-tenant, where isolation has to hold at the database |
| `OracleStore` | Oracle Database 26ai — native `VECTOR` + JSON columns | memories next to the rest of your Oracle data, with semantic search in SQL |

The drivers in `locus.memory.backends` referenced by
[Checkpointers](checkpointers.md) implement `BaseCheckpointer` (per-thread
state), **not** `BaseStore`. The two layers are separate contracts and a
backend for one is not a backend for the other.

::: locus.memory.store.BaseStore
::: locus.memory.store.InMemoryStore
::: locus.memory.store.NamespacedStore
::: locus.memory.store.StoreContext
::: locus.memory.store.StoreItem
::: locus.memory.store.StoreCapabilities
::: locus.memory.store.StoreCapabilityError
::: locus.memory.store.SemanticSearchResult

### Durable store backends

`HolographicStore` and `PgMemory` shipped in 2.2.0; `OracleStore` in 3.0.0.

::: locus.memory.store_backends.holographic.HolographicStore
::: locus.memory.store_backends.postgresql.PgMemory
::: locus.memory.store_backends.oracle.OracleStore
::: locus.memory.store_backends.oracle.OracleStoreConfig

## Long-term memory manager

`LLMMemoryManager` is the default — with no `extract_fn` it uses a
built-in keyword heuristic to extract and categorize memories at
session end (pass an async `extract_fn` to use a model) and retrieves
the top-k relevant entries at session start. `NoopMemoryManager` is the
pass-through used in tests. `Mem0MemoryManager` (behind the `mem0`
extra, `pip install "locus-sdk[mem0]"`) hands the finished
conversation to the open-source mem0 library, which does its own
extraction and keeps the memories in mem0's own vector store rather
than a `BaseStore`. `OracleAgentMemoryManager` (behind the `agentmemory`
extra) delegates to Oracle's official
[`oracleagentmemory`](https://pypi.org/project/oracleagentmemory/) client —
context-card summaries and scoped retrieval by user, agent and thread on
Oracle Database 26ai.

::: locus.memory.manager.BaseMemoryManager
::: locus.memory.manager.LLMMemoryManager
::: locus.memory.manager.NoopMemoryManager
::: locus.memory.manager.Memory
::: locus.memory.manager.MemoryType
::: locus.memory.managers.oracle_agent_memory.OracleAgentMemoryManager
::: locus.memory.managers.mem0.Mem0MemoryManager
    options:
      docstring_options:
        warn_unknown_params: false
        warn_missing_types: false

## Delta checkpointing

Storage-efficient checkpointer that persists only the diff between
consecutive states (the module targets ~77% storage reduction on long
conversations; actual savings depend on workload). Layered on top of
any `DeltaStorage` backend.

::: locus.memory.delta.DeltaCheckpointer
::: locus.memory.delta.DeltaCheckpoint
::: locus.memory.delta.CheckpointMetadata
::: locus.memory.delta.DeltaStorage
::: locus.memory.delta.InMemoryDeltaStorage

## Registry

String-based checkpointer lookup — used when configuration passes a
provider name (e.g. `"s3"`, `"redis"`) instead of an instance.
Custom backends register themselves via `register_checkpointer`.

::: locus.memory.registry.get_checkpointer
::: locus.memory.registry.register_checkpointer
::: locus.memory.registry.list_checkpointers
