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
[Checkpointers](checkpointers.md) — those backends are also in
`tulip.memory.backends`, with **S3-compatible object storage** as a
production target.

## Conversation management

::: tulip.memory.conversation.ConversationManager
::: tulip.memory.conversation.NullManager
::: tulip.memory.conversation.SlidingWindowManager
::: tulip.memory.conversation.SummarizingManager

## Cross-thread store

The durable KV the long-term memory manager writes to. `InMemoryStore`
is for tests; production stores live in `tulip.memory.backends` (see
[Checkpointers](checkpointers.md) for S3-compatible object storage /
OpenSearch / Postgres / Redis backends, which all implement
`BaseStore` as well as `BaseCheckpointer`).

::: tulip.memory.store.BaseStore
::: tulip.memory.store.InMemoryStore
::: tulip.memory.store.NamespacedStore
::: tulip.memory.store.StoreContext
::: tulip.memory.store.StoreItem
::: tulip.memory.store.StoreCapabilities
::: tulip.memory.store.StoreCapabilityError
::: tulip.memory.store.SemanticSearchResult

## Long-term memory manager

`LLMMemoryManager` is the default — it uses an auxiliary model to
extract and categorize memories at session end and retrieves the
top-k relevant entries at session start. `NoopMemoryManager` is the
pass-through used in tests.

::: tulip.memory.manager.BaseMemoryManager
::: tulip.memory.manager.LLMMemoryManager
::: tulip.memory.manager.NoopMemoryManager
::: tulip.memory.manager.Memory
::: tulip.memory.manager.MemoryType

## Delta checkpointing

Storage-efficient checkpointer that persists only the diff between
consecutive states (~77% storage savings on long conversations).
Layered on top of any `DeltaStorage` backend.

::: tulip.memory.delta.DeltaCheckpointer
::: tulip.memory.delta.DeltaCheckpoint
::: tulip.memory.delta.CheckpointMetadata
::: tulip.memory.delta.DeltaStorage
::: tulip.memory.delta.InMemoryDeltaStorage

## Registry

String-based checkpointer lookup — used when configuration passes a
provider name (e.g. `"s3"`, `"redis"`) instead of an instance.
Custom backends register themselves via `register_checkpointer`.

::: tulip.memory.registry.get_checkpointer
::: tulip.memory.registry.register_checkpointer
::: tulip.memory.registry.list_checkpointers
